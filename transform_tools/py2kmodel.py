import argparse
import os, sys, shutil, tempfile
from pathlib import Path

# Respect an SDK path supplied by the user; do not assume a Linux-specific
# Conda installation on Windows or another Ubuntu machine.

#这个是01studio的k230用的.py转换成.kmodel的代码
# ================== 路径与参数配置 ==================
PT_PATH = "best.pt"
ONNX_PATH = "best.onnx"
KMODEL_PATH = "best.kmodel"
CALIB_DIR = ""    # 校准图片目录；请通过 --calib-dir 或此处填写
TARGET_SIZE = 320   #尺寸不能错
FILL_COLOR = (128, 128, 128)
CALIB_SAMPLES = 200 #修正用的图片数量

# ================== 与板端 AI2D 一致的 Letterbox ==================
def letterbox(img, target_size=640, color=FILL_COLOR):
    from PIL import Image
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_size, target_size), color)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas.paste(resized, (x_offset, y_offset))
    return canvas

# ================== 步骤1：导出 + 简化 ONNX ==================
def export_onnx(pt_path=PT_PATH, onnx_path=ONNX_PATH, target_size=TARGET_SIZE, rebuild=False):
    from ultralytics import YOLO
    import onnx
    import onnxsim
    print("=" * 50)
    print("[1/2] 导出并简化 ONNX ...")
    if target_size < 1:
        raise ValueError("输入边长必须大于 0")
    if os.path.exists(onnx_path) and not rebuild:
        print("ONNX 已存在，跳过导出。若需重新导出请删除旧文件。")
        return

    target = Path(onnx_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    source = Path(pt_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"未找到模型权重: {source}")
    # Ultralytics writes the export next to its input weight. Export from a
    # temporary copy so a failed rebuild cannot overwrite an existing ONNX.
    with tempfile.TemporaryDirectory(prefix=f".{target.stem}.", dir=target.parent) as workdir:
        staged_weights = Path(workdir) / source.name
        shutil.copy2(source, staged_weights)
        model = YOLO(str(staged_weights))
        exported = model.export(
            format="onnx",
            imgsz=target_size,
            opset=12,
            simplify=False,
            half=False,
            nms=False,
            dynamic=False
        )
        exported_path = Path(str(exported)).expanduser().resolve()
        if not exported_path.is_file() or not exported_path.is_relative_to(Path(workdir)):
            raise RuntimeError(f"导出结果不在临时目录内: {exported_path}")
        print(f"✅ ONNX 已导出: {exported_path}")

        onnx_model = onnx.load(str(exported_path))
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        input_shapes = {node.name: [1, 3, target_size, target_size]
                        for node in onnx_model.graph.input}
        onnx_model, check = onnxsim.simplify(onnx_model, input_shapes=input_shapes)
        if not check:
            raise RuntimeError("模型简化校验失败")
        staged_output = Path(workdir) / "simplified.onnx"
        onnx.save_model(onnx_model, str(staged_output))
        with staged_output.open("rb") as handle:
            os.fsync(handle.fileno())
        staged_output.replace(target)
    print(f"✅ ONNX 已简化: {onnx_path}")

# ================== 步骤2：量化生成 kmodel ==================
def quantize(onnx_path=ONNX_PATH, kmodel_path=KMODEL_PATH, calib_dir=CALIB_DIR, target_size=TARGET_SIZE, calib_samples=CALIB_SAMPLES, rebuild=False):
    import nncase
    import numpy as np
    from PIL import Image
    print("=" * 50)
    print("[2/2] INT8 量化（灰度 128 填充 + 200 张校准）...")
    if target_size < 1:
        raise ValueError("输入边长必须大于 0")
    if calib_samples < 1:
        raise ValueError("校准样本数必须大于 0")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"❌ 未找到 ONNX: {onnx_path}")
    if os.path.exists(kmodel_path) and not rebuild:
        print(f"kmodel 已存在，跳过量化：{kmodel_path}")
        return
    if not os.path.isdir(calib_dir):
        raise RuntimeError(f"❌ 校准文件夹不存在: {calib_dir}")

    target = Path(kmodel_path).expanduser().resolve()
    dump_dir = str(target.parent / f"{target.stem}_dump")
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    compile_options = nncase.CompileOptions()
    compile_options.target = "k230"
    compile_options.input_shape = [1, 3, target_size, target_size]
    compile_options.input_layout = "NCHW"
    compile_options.dump_dir = dump_dir
    compile_options.preprocess = True
    compile_options.input_type = "uint8"
    compile_options.input_range = [0, 1]
    compile_options.mean = [0, 0, 0]
    compile_options.std = [1, 1, 1]

    compiler = nncase.Compiler(compile_options)
    with open(onnx_path, "rb") as f:
        compiler.import_onnx(f.read(), nncase.ImportOptions())

    ptq = nncase.PTQTensorOptions()
    ptq.quant_type = "uint8"
    ptq.w_quant_type = "uint8"
    ptq.calibrate_method = "NoClip"

    print(f"从 '{calib_dir}' 加载校准图片（目标 {calib_samples} 张）...")
    images = []
    for f in sorted(os.listdir(calib_dir), key=str.casefold):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            try:
                img = Image.open(os.path.join(calib_dir, f)).convert("RGB")
                img = letterbox(img, target_size, FILL_COLOR)
                arr = np.array(img, dtype=np.uint8)
                arr = np.transpose(arr, (2, 0, 1))
                arr = arr[np.newaxis, ...]
                images.append([arr])
                if len(images) >= calib_samples:
                    break
            except Exception as e:
                print(f"跳过 {f}: {e}")

    print(f" → 成功加载 {len(images)} 张校准图片")
    if not images:
        raise RuntimeError("无有效图片！")

    ptq.samples_count = len(images)
    ptq.set_tensor_data(np.array(images))
    compiler.use_ptq(ptq)

    print("🚀 nncase 编译中...")
    compiler.compile()

    kmodel = compiler.gencode_tobytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
        f.write(kmodel)
        f.flush()
        os.fsync(f.fileno())
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"✅ kmodel 生成: {kmodel_path}")
    print(f"   文件大小: {len(kmodel)/1024/1024:.2f} MB")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch/Ultralytics 模型转换为 K230 .kmodel")
    parser.add_argument("--pt", default=PT_PATH, help="PyTorch 权重")
    parser.add_argument("--onnx", default=ONNX_PATH, help="ONNX 输出路径")
    parser.add_argument("--kmodel", default=KMODEL_PATH, help="kmodel 输出路径")
    parser.add_argument("--calib-dir", default=CALIB_DIR, help="校准图片目录")
    parser.add_argument("--size", type=int, default=TARGET_SIZE, help="输入边长")
    parser.add_argument("--samples", type=int, default=CALIB_SAMPLES, help="校准图片数量")
    parser.add_argument("--rebuild", action="store_true", help="覆盖已有 ONNX / kmodel")
    args = parser.parse_args()
    export_onnx(args.pt, args.onnx, args.size, args.rebuild)
    quantize(args.onnx, args.kmodel, args.calib_dir, args.size, args.samples, args.rebuild)

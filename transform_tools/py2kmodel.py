import os, sys, numpy as np

os.environ.setdefault("DOTNET_ROOT", os.path.expanduser("~/miniconda3/lib/dotnet"))

from PIL import Image
from ultralytics import YOLO
import onnx, onnxsim
import nncase
#这个是01studio的k230用的.py转换成.kmodel的代码
# ================== 路径与参数配置 ==================
PT_PATH = "best.pt"
ONNX_PATH = "best.onnx"
KMODEL_PATH = "best.kmodel"
CALIB_DIR = "/home/xyzkioo/datasets/merged_dataset/images/train"    #修正用的图片路径
TARGET_SIZE = 320   #尺寸不能错
FILL_COLOR = (128, 128, 128)
CALIB_SAMPLES = 200 #修正用的图片数量

# ================== 与板端 AI2D 一致的 Letterbox ==================
def letterbox(img, target_size=640, color=FILL_COLOR):
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
def export_onnx():
    print("=" * 50)
    print("[1/2] 导出并简化 ONNX ...")
    if os.path.exists(ONNX_PATH):
        print("ONNX 已存在，跳过导出。若需重新导出请删除旧文件。")
        return

    model = YOLO(PT_PATH)
    model.export(
        format="onnx",
        imgsz=TARGET_SIZE,
        opset=12,
        simplify=False,
        half=False,
        nms=False,
        dynamic=False
    )
    print(f"✅ ONNX 已导出: {ONNX_PATH}")

    onnx_model = onnx.load(ONNX_PATH)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    input_shapes = {node.name: [1, 3, TARGET_SIZE, TARGET_SIZE]
                    for node in onnx_model.graph.input}
    onnx_model, check = onnxsim.simplify(onnx_model, input_shapes=input_shapes)
    assert check, "模型简化校验失败"
    onnx.save_model(onnx_model, ONNX_PATH)
    print(f"✅ ONNX 已简化: {ONNX_PATH}")

# ================== 步骤2：量化生成 kmodel ==================
def quantize():
    print("=" * 50)
    print("[2/2] INT8 量化（灰度 128 填充 + 200 张校准）...")
    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(f"❌ 未找到 ONNX: {ONNX_PATH}")

    dump_dir = "dump"
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    compile_options = nncase.CompileOptions()
    compile_options.target = "k230"
    compile_options.input_shape = [1, 3, TARGET_SIZE, TARGET_SIZE]
    compile_options.input_layout = "NCHW"
    compile_options.dump_dir = dump_dir
    compile_options.preprocess = True
    compile_options.input_type = "uint8"
    compile_options.input_range = [0, 1]
    compile_options.mean = [0, 0, 0]
    compile_options.std = [1, 1, 1]

    compiler = nncase.Compiler(compile_options)
    with open(ONNX_PATH, "rb") as f:
        compiler.import_onnx(f.read(), nncase.ImportOptions())

    ptq = nncase.PTQTensorOptions()
    ptq.quant_type = "uint8"
    ptq.w_quant_type = "uint8"
    ptq.calibrate_method = "NoClip"

    if not os.path.exists(CALIB_DIR):
        raise RuntimeError(f"❌ 校准文件夹不存在: {CALIB_DIR}")

    print(f"从 '{CALIB_DIR}' 加载校准图片（目标 {CALIB_SAMPLES} 张）...")
    images = []
    for f in os.listdir(CALIB_DIR):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            try:
                img = Image.open(os.path.join(CALIB_DIR, f)).convert("RGB")
                img = letterbox(img, TARGET_SIZE, FILL_COLOR)
                arr = np.array(img, dtype=np.uint8)
                arr = np.transpose(arr, (2, 0, 1))
                arr = arr[np.newaxis, ...]
                images.append([arr])
                if len(images) >= CALIB_SAMPLES:
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
    with open(KMODEL_PATH, "wb") as f:
        f.write(kmodel)

    print(f"✅ kmodel 生成: {KMODEL_PATH}")
    print(f"   文件大小: {len(kmodel)/1024/1024:.2f} MB")
    print("=" * 50)

if __name__ == "__main__":
    export_onnx()
    quantize()

import argparse
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

try:
    from transform_tools.py2kmodel import configure_dotnet
except ModuleNotFoundError:
    # The script is also supported as `python transform_tools/PY2KM_validate.py`;
    # in that mode Python puts the script directory, rather than the repository
    # root, on sys.path.
    from py2kmodel import configure_dotnet

# ONNX vs kmodel 推理一致性校验脚本

# Respect an SDK path supplied by the user; do not assume a Linux-specific
# Conda installation on Windows or another Ubuntu machine.
_site_packages = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
os.environ["PATH"] = _site_packages + os.pathsep + os.environ.get("PATH", "")

TARGET_SIZE = 320                                                        # letterbox 目标尺寸（与 convert.py 保持一致）
FILL_COLOR = (128, 128, 128)                                             # letterbox 填充色 (R, G, B)
NORMALIZE = True                                                         # ONNX 输入是否 /255 归一化 (kmodel 始终 uint8)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def resolve_image_paths(image_path):
    """Return one image or all supported images in a directory."""

    target = Path(image_path).expanduser().resolve()
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(f"未找到图片或目录: {target}")
    paths = [
        child for child in sorted(target.iterdir(), key=lambda item: item.name.casefold())
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not paths:
        raise ValueError(f"目录中没有支持的图片（{', '.join(sorted(IMAGE_SUFFIXES))}）: {target}")
    return paths


@contextmanager
def _isolated_simulator_workspace():
    """Keep nncase's relative gmodel dump outside the repository or bundle."""

    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="vtools-kmodel-") as workspace:
        os.chdir(workspace)
        try:
            yield Path(workspace)
        finally:
            os.chdir(previous)


def letterbox(img, target_size, fill_color):
    from PIL import Image
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_size, target_size), fill_color)
    canvas.paste(resized, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return canvas


def onnx_infer(onnx_path, img, target_size, fill_color, normalize, session=None):
    import numpy as np
    import onnxruntime as ort
    print("[1/2] ONNX 推理 ...")
    if session is None:
        session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    img = letterbox(img, target_size, fill_color)
    input_type = str(session.get_inputs()[0].type).lower()
    declared_float = "float" in input_type or "double" in input_type
    if "double" in input_type:
        input_dtype = np.float64
    elif "float16" in input_type:
        input_dtype = np.float16
    elif "float" in input_type:
        input_dtype = np.float32
    elif "int64" in input_type:
        input_dtype = np.int64
    elif "int32" in input_type:
        input_dtype = np.int32
    else:
        input_dtype = np.uint8
    input_data = np.array(img, dtype=input_dtype)
    if normalize:
        if not declared_float:
            print(f"  警告：模型输入声明为 {input_type}，忽略 /255 归一化以保持输入 dtype")
        else:
            input_data = input_data.astype(np.float64 if input_dtype is np.float64 else np.float32) / 255.0
            if input_dtype is np.float16:
                input_data = input_data.astype(np.float16)
            elif input_dtype is np.float64:
                input_data = input_data.astype(np.float64)
    input_data = np.transpose(input_data, (2, 0, 1))[np.newaxis, ...]

    outputs = session.run(None, {input_name: input_data})
    print(f"  ONNX 输出数量: {len(outputs)}")
    for i, o in enumerate(outputs):
        value_range = f"[{o.min():.4f}, {o.max():.4f}]" if o.size else "empty"
        print(f"  output[{i}]: shape={o.shape}, range={value_range}")
    return outputs


def load_kmodel_simulator(kmodel_path):
    dotnet_root = configure_dotnet()
    if dotnet_root:
        print(f"使用 .NET 运行时: {dotnet_root}")
    import nncase

    simulator = nncase.Simulator()
    simulator.load_model(Path(kmodel_path).read_bytes())
    return simulator


def kmodel_infer(kmodel_path, img, target_size, fill_color, simulator=None):
    import numpy as np
    import nncase
    print("[2/2] kmodel 推理 ...")
    if simulator is None:
        simulator = load_kmodel_simulator(kmodel_path)

    img = letterbox(img, target_size, fill_color)
    input_tensor = np.array(img, dtype=np.uint8)
    input_tensor = np.transpose(input_tensor, (2, 0, 1))[np.newaxis, ...]

    simulator.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(input_tensor))
    simulator.run()

    num_outputs = simulator.outputs_size
    outputs = [simulator.get_output_tensor(i) for i in range(num_outputs)]

    print(f"  kmodel 输出数量: {num_outputs}")
    for i, o in enumerate(outputs):
        data = o.to_numpy()
        value_range = f"[{data.min():.4f}, {data.max():.4f}]" if data.size else "empty"
        print(f"  output[{i}]: shape={data.shape}, range={value_range}")
    return outputs


def compare(onnx_outputs, kmodel_outputs, cosine_threshold=0.99, mae_threshold=0.05):
    import numpy as np
    print("\n========== 对比 ==========")
    print(f"输出数量: ONNX={len(onnx_outputs)}, kmodel={len(kmodel_outputs)}")
    passed = bool(onnx_outputs) and len(onnx_outputs) == len(kmodel_outputs)
    if not onnx_outputs or not kmodel_outputs:
        print("  输出为空，无法证明推理一致")
    if not passed:
        print("  输出数量不一致")
    for i in range(min(len(onnx_outputs), len(kmodel_outputs))):
        ref = np.asarray(onnx_outputs[i], dtype=np.float64)
        q = np.asarray(kmodel_outputs[i].to_numpy(), dtype=np.float64)
        if ref.shape != q.shape or ref.size == 0 or not np.isfinite(ref).all() or not np.isfinite(q).all():
            print(f"  output[{i}]: shape/finite 校验失败（ONNX={ref.shape}, kmodel={q.shape}）")
            passed = False
            continue
        ref_flat, q_flat = ref.reshape(-1), q.reshape(-1)
        ref_norm, q_norm = np.linalg.norm(ref_flat), np.linalg.norm(q_flat)
        if ref_norm <= 1e-12 and q_norm <= 1e-12:
            cos_sim = 1.0
        elif ref_norm <= 1e-12 or q_norm <= 1e-12:
            cos_sim = 0.0
        else:
            cos_sim = float(np.dot(ref_flat, q_flat) / (ref_norm * q_norm))
        mae = np.mean(np.abs(ref - q))
        item_passed = bool(cos_sim >= cosine_threshold and mae <= mae_threshold)
        passed = passed and item_passed
        print(f"  output[{i}]: cosine_sim={cos_sim:.4f}, MAE={mae:.4f}, {'PASS' if item_passed else 'FAIL'}")
    print(f"阈值: cosine >= {cosine_threshold}, MAE <= {mae_threshold}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="ONNX vs kmodel 推理一致性验证")
    parser.add_argument("--onnx", required=True, help="ONNX 模型路径")
    parser.add_argument("--kmodel", required=True, help="kmodel 模型路径")
    parser.add_argument("--image", required=True, help="测试图片路径或图片目录（目录按文件名排序批量校验）")
    parser.add_argument("--size", type=int, default=TARGET_SIZE, help=f"letterbox 目标尺寸 (默认: {TARGET_SIZE})")
    parser.add_argument("--color", type=int, nargs=3, metavar=("R", "G", "B"), default=FILL_COLOR, help=f"letterbox 填充色 (默认: {FILL_COLOR})")
    parser.add_argument("--no-norm", dest="normalize", action="store_false", default=NORMALIZE, help="ONNX 输入不做 /255 归一化")
    parser.add_argument("--cosine-threshold", type=float, default=0.99)
    parser.add_argument("--mae-threshold", type=float, default=0.05)

    args = parser.parse_args()

    onnx_path = args.onnx
    kmodel_path = args.kmodel
    image_path = args.image
    size = args.size
    fill_color = tuple(args.color)
    normalize = args.normalize

    if size < 1:
        parser.error("--size 必须大于 0")
    if any(value < 0 or value > 255 for value in fill_color):
        parser.error("--color 的三个值必须在 0-255 之间")
    if not -1.0 <= args.cosine_threshold <= 1.0:
        parser.error("--cosine-threshold 必须在 -1 到 1 之间")
    if args.mae_threshold < 0:
        parser.error("--mae-threshold 不能小于 0")

    for name, path in (("ONNX", onnx_path), ("kmodel", kmodel_path)):
        if not os.path.exists(path):
            sys.exit(f"❌ 未找到{name}文件: {path}")
    try:
        image_paths = resolve_image_paths(image_path)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"❌ {exc}")
    onnx_path = str(Path(onnx_path).expanduser().resolve())
    kmodel_path = str(Path(kmodel_path).expanduser().resolve())

    print(f"配置: onnx={onnx_path}\n      kmodel={kmodel_path}\n      image={image_path}\n      size={size}, color={fill_color}, normalize={normalize}\n")

    from PIL import Image

    import onnxruntime as ort

    onnx_session = ort.InferenceSession(onnx_path)
    all_passed = True
    with _isolated_simulator_workspace():
        simulator = load_kmodel_simulator(kmodel_path)
        for index, image_file in enumerate(image_paths, start=1):
            print(f"\n===== 图片 {index}/{len(image_paths)}: {image_file.name} =====")
            try:
                with Image.open(image_file) as source:
                    img = source.convert("RGB")
            except Exception as exc:
                print(f"  图片读取失败: {exc}")
                all_passed = False
                continue
            onnx_outputs = onnx_infer(onnx_path, img, size, fill_color, normalize=normalize, session=onnx_session)
            kmodel_outputs = kmodel_infer(kmodel_path, img, size, fill_color, simulator=simulator)
            image_passed = compare(onnx_outputs, kmodel_outputs, args.cosine_threshold, args.mae_threshold)
            all_passed = all_passed and image_passed

    print(f"\n批量校验结果: {len(image_paths)} 张图片，整体 {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

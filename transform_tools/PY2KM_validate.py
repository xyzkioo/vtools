import os, sys, argparse
from pathlib import Path

#ONNX vs kmodel 推理一致性校验脚本
#在 PyCharm 中可直接右键运行；参数优先级: 命令行 > 环境变量 > 下方默认配置

# Respect an SDK path supplied by the user; do not assume a Linux-specific
# Conda installation on Windows or another Ubuntu machine.
_site_packages = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
os.environ["PATH"] = _site_packages + os.pathsep + os.environ.get("PATH", "")

# ================== 路径与参数配置 ==================
ONNX_PATH = "best.onnx"                                                  # ONNX 模型路径
KMODEL_PATH = "best.kmodel"                                              # kmodel 模型路径
IMAGE_PATH = ""                                                            # 测试图片路径
TARGET_SIZE = 320                                                        # letterbox 目标尺寸（与 convert.py 保持一致）
FILL_COLOR = (128, 128, 128)                                             # letterbox 填充色 (R, G, B)
NORMALIZE = True                                                         # ONNX 输入是否 /255 归一化 (kmodel 始终 uint8)

# 可选: 在 PyCharm Run Configuration 的环境变量中设置以覆盖默认值
# VALIDATE_ONNX / VALIDATE_KMODEL / VALIDATE_IMAGE / VALIDATE_SIZE / VALIDATE_COLOR / VALIDATE_NORMALIZE


def _get(key, default):
    return os.environ.get(key, default)


def letterbox(img, target_size, fill_color):
    from PIL import Image
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_size, target_size), fill_color)
    canvas.paste(resized, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return canvas


def onnx_infer(onnx_path, img, target_size, fill_color, normalize):
    import numpy as np
    import onnxruntime as ort
    print("[1/2] ONNX 推理 ...")
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


def kmodel_infer(kmodel_path, img, target_size, fill_color):
    import numpy as np
    import nncase
    print("[2/2] kmodel 推理 ...")
    simulator = nncase.Simulator()
    simulator.load_model(Path(kmodel_path).read_bytes())

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
    parser.add_argument("--onnx", help=f"ONNX 模型路径 (默认: {ONNX_PATH})")
    parser.add_argument("--kmodel", help=f"kmodel 模型路径 (默认: {KMODEL_PATH})")
    parser.add_argument("--image", help=f"测试图片路径 (默认: {IMAGE_PATH})")
    parser.add_argument("--size", type=int, help=f"letterbox 目标尺寸 (默认: {TARGET_SIZE})")
    parser.add_argument("--color", type=int, nargs=3, metavar=("R", "G", "B"), help=f"letterbox 填充色 (默认: {FILL_COLOR})")
    parser.add_argument("--no-norm", dest="normalize", action="store_false", default=None, help=f"ONNX 输入不做 /255 归一化 (默认: {'归一化' if NORMALIZE else '不归一化'})")
    parser.add_argument("--cosine-threshold", type=float, default=0.99)
    parser.add_argument("--mae-threshold", type=float, default=0.05)

    args = parser.parse_args()

    onnx_path = args.onnx or _get("VALIDATE_ONNX", ONNX_PATH)
    kmodel_path = args.kmodel or _get("VALIDATE_KMODEL", KMODEL_PATH)
    image_path = args.image or _get("VALIDATE_IMAGE", IMAGE_PATH)
    size = args.size if args.size is not None else int(_get("VALIDATE_SIZE", TARGET_SIZE))
    color_env = _get("VALIDATE_COLOR", None)
    fill_color = tuple(args.color) if args.color else (tuple(int(x) for x in color_env.split(",")) if color_env else FILL_COLOR)

    if args.normalize is None:
        norm_env = _get("VALIDATE_NORMALIZE", None)
        normalize = norm_env.lower() not in ("0", "false", "no") if norm_env else NORMALIZE
    else:
        normalize = args.normalize

    if size < 1:
        parser.error("--size 必须大于 0")
    if any(value < 0 or value > 255 for value in fill_color):
        parser.error("--color 的三个值必须在 0-255 之间")
    if not -1.0 <= args.cosine_threshold <= 1.0:
        parser.error("--cosine-threshold 必须在 -1 到 1 之间")
    if args.mae_threshold < 0:
        parser.error("--mae-threshold 不能小于 0")

    for name, path in (("ONNX", onnx_path), ("kmodel", kmodel_path), ("图片", image_path)):
        if not os.path.exists(path):
            sys.exit(f"❌ 未找到{name}文件: {path}")

    print(f"配置: onnx={onnx_path}\n      kmodel={kmodel_path}\n      image={image_path}\n      size={size}, color={fill_color}, normalize={normalize}\n")

    from PIL import Image

    img = Image.open(image_path).convert("RGB")

    onnx_outputs = onnx_infer(onnx_path, img, size, fill_color, normalize=normalize)
    kmodel_outputs = kmodel_infer(kmodel_path, img, size, fill_color)

    return 0 if compare(onnx_outputs, kmodel_outputs, args.cosine_threshold, args.mae_threshold) else 1


if __name__ == "__main__":
    raise SystemExit(main())

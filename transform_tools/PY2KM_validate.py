import os, sys, argparse, numpy as np

#ONNX vs kmodel 推理一致性校验脚本
#在 PyCharm 中可直接右键运行；参数优先级: 命令行 > 环境变量 > 下方默认配置

os.environ.setdefault("DOTNET_ROOT", os.path.expanduser("~/miniconda3/lib/dotnet"))
_site_packages = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
os.environ["PATH"] = _site_packages + os.pathsep + os.environ.get("PATH", "")

from PIL import Image
import onnxruntime as ort
import nncase


# ================== 路径与参数配置 ==================
ONNX_PATH = "best.onnx"                                                  # ONNX 模型路径
KMODEL_PATH = "best.kmodel"                                              # kmodel 模型路径
IMAGE_PATH = "IMG20260724155631/IMG20260724155007.jpg"                    # 测试图片路径
TARGET_SIZE = 320                                                        # letterbox 目标尺寸（与 convert.py 保持一致）
FILL_COLOR = (128, 128, 128)                                             # letterbox 填充色 (R, G, B)
NORMALIZE = True                                                         # ONNX 输入是否 /255 归一化 (kmodel 始终 uint8)

# 可选: 在 PyCharm Run Configuration 的环境变量中设置以覆盖默认值
# VALIDATE_ONNX / VALIDATE_KMODEL / VALIDATE_IMAGE / VALIDATE_SIZE / VALIDATE_COLOR / VALIDATE_NORMALIZE


def _get(key, default):
    return os.environ.get(key, default)


def letterbox(img, target_size, fill_color):
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_size, target_size), fill_color)
    canvas.paste(resized, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return canvas


def onnx_infer(onnx_path, img, target_size, fill_color, normalize):
    print("[1/2] ONNX 推理 ...")
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    img = letterbox(img, target_size, fill_color)
    input_data = np.array(img, dtype=np.uint8)
    if normalize:
        input_data = input_data.astype(np.float32) / 255.0
    input_data = np.transpose(input_data, (2, 0, 1))[np.newaxis, ...]

    outputs = session.run(None, {input_name: input_data})
    print(f"  ONNX 输出数量: {len(outputs)}")
    for i, o in enumerate(outputs):
        print(f"  output[{i}]: shape={o.shape}, range=[{o.min():.4f}, {o.max():.4f}]")
    return outputs


def kmodel_infer(kmodel_path, img, target_size, fill_color):
    print("[2/2] kmodel 推理 ...")
    simulator = nncase.Simulator()
    simulator.load_model(open(kmodel_path, "rb").read())

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
        print(f"  output[{i}]: shape={data.shape}, range=[{data.min():.4f}, {data.max():.4f}]")
    return outputs


def compare(onnx_outputs, kmodel_outputs):
    print("\n========== 对比 ==========")
    print(f"输出数量: ONNX={len(onnx_outputs)}, kmodel={len(kmodel_outputs)}")
    for i in range(min(len(onnx_outputs), len(kmodel_outputs))):
        ref = onnx_outputs[i]
        q = kmodel_outputs[i].to_numpy()
        cos_sim = np.dot(ref.flatten(), q.flatten()) / (np.linalg.norm(ref) * np.linalg.norm(q) + 1e-8)
        mae = np.mean(np.abs(ref - q))
        print(f"  output[{i}]: cosine_sim={cos_sim:.4f}, MAE={mae:.4f}")


def main():
    parser = argparse.ArgumentParser(description="ONNX vs kmodel 推理一致性验证")
    parser.add_argument("--onnx", help=f"ONNX 模型路径 (默认: {ONNX_PATH})")
    parser.add_argument("--kmodel", help=f"kmodel 模型路径 (默认: {KMODEL_PATH})")
    parser.add_argument("--image", help=f"测试图片路径 (默认: {IMAGE_PATH})")
    parser.add_argument("--size", type=int, help=f"letterbox 目标尺寸 (默认: {TARGET_SIZE})")
    parser.add_argument("--color", type=int, nargs=3, metavar=("R", "G", "B"), help=f"letterbox 填充色 (默认: {FILL_COLOR})")
    parser.add_argument("--no-norm", dest="normalize", action="store_false", default=None, help=f"ONNX 输入不做 /255 归一化 (默认: {'归一化' if NORMALIZE else '不归一化'})")

    args = parser.parse_args()

    onnx_path = args.onnx or _get("VALIDATE_ONNX", ONNX_PATH)
    kmodel_path = args.kmodel or _get("VALIDATE_KMODEL", KMODEL_PATH)
    image_path = args.image or _get("VALIDATE_IMAGE", IMAGE_PATH)
    size = args.size or int(_get("VALIDATE_SIZE", TARGET_SIZE))
    color_env = _get("VALIDATE_COLOR", None)
    fill_color = tuple(args.color) if args.color else (tuple(int(x) for x in color_env.split(",")) if color_env else FILL_COLOR)

    if args.normalize is None:
        norm_env = _get("VALIDATE_NORMALIZE", None)
        normalize = norm_env.lower() not in ("0", "false", "no") if norm_env else NORMALIZE
    else:
        normalize = args.normalize

    for name, path in (("ONNX", onnx_path), ("kmodel", kmodel_path), ("图片", image_path)):
        if not os.path.exists(path):
            sys.exit(f"❌ 未找到{name}文件: {path}")

    print(f"配置: onnx={onnx_path}\n      kmodel={kmodel_path}\n      image={image_path}\n      size={size}, color={fill_color}, normalize={normalize}\n")

    img = Image.open(image_path).convert("RGB")

    onnx_outputs = onnx_infer(onnx_path, img, size, fill_color, normalize=normalize)
    kmodel_outputs = kmodel_infer(kmodel_path, img, size, fill_color)

    compare(onnx_outputs, kmodel_outputs)


if __name__ == "__main__":
    main()

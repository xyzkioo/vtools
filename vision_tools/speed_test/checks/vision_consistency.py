#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复用现有 ONNX/engine，比对同一输入的 PyTorch/TensorRT 输出。

仅做正确性检查，不重复测速，不修改权重，也不自动重新导出 engine。
Torch/TensorRT/ONNX 在运行时加载：即使依赖缺失，也尽量保存 error 报告。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from core.vision_benchmark_config import apply_python_paths, get_model_entries, load_config
from checks.vision_consistency_metrics import (
    aggregate_status, align_outputs, combine_status, compare_array,
    compare_detections, validate_tolerances,
)
from core.vision_run_manager import prepare_run_directory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hook(adapter: Any, name: str) -> Any:
    function = getattr(getattr(adapter, "module", None), name, None)
    return function if callable(function) else None


def _sources(values: dict[str, Any], benchmark: dict[str, Any]) -> list[Path | None]:
    mode = values.get("input_mode", "image")
    if mode == "adapter":
        return [None]  # 一个 adapter 输入案例，不把相同随机张量伪装成多张图片。
    if mode != "image":
        raise ValueError("consistency.input_mode 只支持 image 或 adapter")
    source = values.get("source") or benchmark.get("image") or benchmark.get("source")
    if not source:
        raise ValueError("image 模式需要 consistency.source 或 benchmark.image；不会自动退回随机输入")
    path = Path(str(source))
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"检查图片/目录不存在：{path}")
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    limit = int(values.get("max_images", 10))
    if limit <= 0:
        raise ValueError("consistency.max_images 必须大于 0")
    paths = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)[:limit]
    if not paths:
        raise ValueError(f"目录中没有可用图片：{path}")
    return paths


def _inputs(adapter: Any, source: Path | None, config: Any, seed: int) -> Any:
    import torch
    from core.vision_benchmark_common import InputBundle, dtype_for_precision

    function = _hook(adapter, "make_validation_inputs")
    if function is not None:
        return InputBundle.from_value(function(source=source, config=config), config.batch_size)
    dtype = dtype_for_precision(config.precision, config.device)
    if source is None:
        # 限制随机种子修改范围，不改变后续测速的全局随机状态。
        with torch.random.fork_rng(devices=[config.device.index or 0]):
            torch.manual_seed(seed)
            return adapter.make_inputs(config.batch_size, config.image_size, config.device, dtype, for_export=True)
    if adapter.adapter_name != "ultralytics" or adapter.task != "detect":
        raise ValueError(
            "非内置 Ultralytics detect 图片预处理不能猜测；请在 adapter 中实现 "
            "make_validation_inputs(source, config)，或使用 input_mode: adapter 做示例输入检查"
        )
    import cv2
    import numpy as np
    from ultralytics.data.augment import LetterBox

    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"无法解码图片：{source}")
    # 固定 H/W 的 letterbox，与双方共用；不是直接拉伸原图。
    image = LetterBox(new_shape=config.image_size, auto=False)(image=image)
    image = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))
    tensor = torch.from_numpy(image).unsqueeze(0).repeat(config.batch_size, 1, 1, 1)
    tensor = tensor.to(config.device, dtype=dtype) / 255.0
    return InputBundle((tensor,), batch_size=config.batch_size, description="shared RGB letterbox /255 image")


def _clone_bundle(bundle: Any, device: Any, float_dtype: Any = None) -> Any:
    import torch
    from core.vision_benchmark_common import InputBundle

    def clone(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            dtype = float_dtype if value.is_floating_point() and float_dtype is not None else value.dtype
            return value.detach().to(device=device, dtype=dtype).clone().contiguous()
        if isinstance(value, tuple):
            return tuple(clone(item) for item in value)
        if isinstance(value, list):
            return [clone(item) for item in value]
        if isinstance(value, dict):
            return {key: clone(item) for key, item in value.items()}
        return value

    return InputBundle(clone(bundle.args), clone(bundle.kwargs), bundle.batch_size, bundle.description)


def _to_numpy(value: Any) -> Any:
    import torch
    if not isinstance(value, torch.Tensor):
        raise TypeError("validation_outputs 必须把每个输出名称映射到 torch.Tensor")
    value = value.detach().cpu()
    if value.dtype == torch.bfloat16:
        value = value.float()  # NumPy 不支持 bfloat16；转 float32 保留其数值。
    return value.numpy().copy()


def _reference_outputs(adapter: Any, model: Any, bundle: Any, names: list[str]) -> dict[str, Any]:
    from core.vision_benchmark_common import flatten_tensors
    function = _hook(adapter, "validation_forward")
    if function is not None:
        outputs = function(model=model, inputs=bundle)
    else:
        # 与默认 export_onnx 的 forward_model 边界完全相同；不能调用 predict，
        # 因为 predict 可能自动融合/改变模型，且包含独立的预处理和后处理。
        outputs = adapter.forward_model(model)(*bundle.args, **bundle.kwargs)
    normalizer = _hook(adapter, "validation_outputs")
    if normalizer is not None:
        named = normalizer(outputs=outputs, output_names=names)
        if not isinstance(named, dict) or set(named) != set(names):
            raise ValueError("validation_outputs 必须返回覆盖所有 ONNX 输出名称的字典")
        return {name: _to_numpy(named[name]) for name in names}
    tensors = flatten_tensors(outputs)
    if len(tensors) != len(names):
        raise ValueError(
            f"PyTorch 有 {len(tensors)} 个 Tensor 输出，ONNX 有 {len(names)} 个。"
            "自定义导出/后处理请实现 validation_forward 和 validation_outputs；不会截断或猜测"
        )
    # 默认 exporter 也是按 tensor/list/tuple/dict 的插入顺序扁平化。
    return {name: _to_numpy(tensor) for name, tensor in zip(names, tensors)}


def _check_case(
    adapter: Any, model: Any, runner: Any, source: Path | None, config: Any,
    values: dict[str, Any], reference_precision: str,
) -> dict[str, Any]:
    import torch
    from core.vision_benchmark_common import dtype_for_precision, flatten_tensors, synchronize

    seed = int(values.get("seed", 0))
    bundle = _clone_bundle(_inputs(adapter, source, config, seed), config.device)
    tensors = flatten_tensors(bundle.args) + flatten_tensors(bundle.kwargs)
    if len(tensors) != len(runner.input_names):
        raise ValueError("输入数量与 ONNX/engine 不匹配；多输入模型请检查 adapter 和 input_names")
    input_info = []
    for name, tensor in zip(runner.input_names, tensors):
        # 不允许 runner 的隐式 cast 掩盖两边输入 dtype 不一致。
        if tensor.dtype != runner.input_dtype(name):
            raise ValueError(f"输入 {name} dtype={tensor.dtype}，engine 需要 {runner.input_dtype(name)}；检查精度/旧 engine")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"输入 {name} 含 NaN/Inf")
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        input_info.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                           "sha256": hashlib.sha256(raw).hexdigest()})
    ref_dtype = dtype_for_precision(reference_precision, config.device)
    # 双方分别拿同一输入的副本，防止某一模型原地修改输入后污染另一方。
    ref = _reference_outputs(adapter, model, _clone_bundle(bundle, config.device, ref_dtype), runner.output_names)
    actual_tensors = runner.run(_clone_bundle(bundle, config.device))
    synchronize(config.device)
    if len(actual_tensors) != len(runner.output_names):
        raise ValueError("TensorRT 返回输出数量异常")
    actual = {name: _to_numpy(value) for name, value in zip(runner.output_names, actual_tensors)}
    atol, rtol = float(values.get("atol", 0.001)), float(values.get("rtol", 0.01))
    comparisons = [
        {"name": name, **compare_array(a, b, atol=atol, rtol=rtol)}
        for name, a, b in align_outputs(ref, actual)
    ]
    detection = None
    mode = values.get("detection_mode", "auto")
    if mode not in {"auto", "none", "xyxy6"}:
        raise ValueError("detection_mode 只支持 auto / none / xyxy6")
    target = adapter.forward_model(model)
    end2end = bool(getattr(target, "end2end", False))
    detect = mode == "xyxy6" or (
        mode == "auto" and adapter.adapter_name == "ultralytics" and adapter.task == "detect" and end2end
    )
    if detect:
        name = values.get("detection_output") or runner.output_names[0]
        if name not in ref:
            raise ValueError(f"detection_output 不在 ONNX 输出中：{name}")
        detection = {"name": name, **compare_detections(
            ref[name], actual[name], conf=float(values.get("detection_conf", config.conf)),
            iou_threshold=float(values.get("detection_iou", 0.95)),
            score_atol=float(values.get("score_atol", 0.01)),
        )}
    return {"source": str(source) if source else "adapter_example_input", "inputs": input_info,
            "input_description": bundle.description, "status": combine_status(comparisons, detection),
            "outputs": comparisons, "detections": detection,
            "note": "默认只比对模型调用输出，不含完整应用后处理；不是 mAP 验证"}


def _check_model(entry: dict[str, Any], config_file: dict[str, Any]) -> dict[str, Any]:
    import torch
    import onnx
    from core.vision_benchmark_common import BenchmarkConfig, create_adapter, normalize_precision, resolve_device
    from backends.pytorch_vision_tensorrt_benchmark_v2 import TensorRTRunner, parse_input_size, safe_name

    benchmark = config_file.get("benchmark", {})
    trt_values = config_file.get("tensorrt", {})
    values = config_file.get("consistency", {})
    device = resolve_device(benchmark.get("device", "auto"))
    if device.type != "cuda":
        raise RuntimeError("PyTorch/TensorRT 一致性检查需要 CUDA；CPU 不能运行 TensorRT engine")
    precision = normalize_precision(trt_values.get("precision", "fp16"))
    ref_precision = values.get("reference_precision", "same")
    ref_precision = precision if ref_precision == "same" else normalize_precision(ref_precision)
    if ref_precision not in {precision, "fp32"}:
        raise ValueError("reference_precision 仅支持 same 或 fp32")
    height, width = parse_input_size(benchmark.get("input_size", "832"))
    batch = int(benchmark.get("batch_size", 1))
    if batch <= 0:
        raise ValueError("batch_size 必须大于 0")
    base = f"{safe_name(str(entry['name']))}_{precision}_{height}x{width}_b{batch}"
    engine_path = Path(entry.get("engine") or Path(trt_values.get("engine_dir", "runs-profile/engines")) / f"{base}.engine")
    onnx_path = Path(entry.get("onnx") or Path(trt_values.get("onnx_dir", "runs-profile/onnx")) / f"{base}.onnx")
    weights = Path(entry["weights"])
    for path in (weights, engine_path, onnx_path):
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在：{path}。先运行 benchmark_tensorrt.py 成功构建；检查脚本不会自动构建")
    graph = onnx.load(str(onnx_path), load_external_data=False).graph
    initializers = {item.name for item in graph.initializer}
    input_names = [item.name for item in graph.input if item.name not in initializers]
    output_names = [item.name for item in graph.output]
    configured_names = values.get("input_names")
    if configured_names:
        if not isinstance(configured_names, list) or set(configured_names) != set(input_names):
            raise ValueError("consistency.input_names 必须列出所有 ONNX 输入，顺序对应 adapter 的 Tensor 扁平顺序")
        input_names = list(configured_names)
    cases = _sources(values, benchmark)
    conf = float(benchmark.get("conf", 0.25))
    adapter = create_adapter(entry["adapter"], entry.get("task", "detect"), conf,
                             float(benchmark.get("iou", 0.7)), int(benchmark.get("max_det", 300)))
    config = BenchmarkConfig(device, batch, height, width, precision, warmup=0, repeats=1,
                             fuse=bool(benchmark.get("fuse_model", False)), conf=conf)
    result: dict[str, Any] = {
        "name": entry["name"], "adapter": entry["adapter"], "precision": precision,
        "reference_precision": ref_precision, "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__, "onnx_version": onnx.__version__,
        "weights": str(weights), "weights_sha256": _sha256(weights),
        "engine": str(engine_path), "engine_sha256": _sha256(engine_path),
        "onnx": str(onnx_path), "onnx_sha256": _sha256(onnx_path),
        "input_names": input_names, "output_names": output_names, "cases": [],
        "provenance_note": "哈希标识本次检查的文件，不证明旧 engine 来自当前权重；换权重/源码后需重新构建",
    }
    model = runner = None
    # 禁用 PyTorch TF32 仅影响本次参考计算；结束时恢复，不影响后续测速。
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.cuda.device(device), torch.inference_mode():
            model = adapter.load_model(weights, device)
            model = adapter.prepare_model(model, device, ref_precision, config.fuse)
            target = adapter.forward_model(model)
            if isinstance(target, torch.nn.Module):
                target.eval()
                if ref_precision == "fp32":
                    target.float()
            prepare = _hook(adapter, "prepare_validation_model")
            if prepare is not None:
                prepared = prepare(model=model, config=replace(config, precision=ref_precision))
                if prepared is not None:
                    model = prepared
            runner = TensorRTRunner(engine_path, device, input_names=input_names, output_names=output_names)
            result["tensorrt_version"] = getattr(runner.trt, "__version__", "unknown")
            for source in cases:
                try:
                    case = _check_case(adapter, model, runner, source, config, values, ref_precision)
                except Exception as error:
                    case = {"source": str(source) if source else "adapter_example_input", "status": "error",
                            "error": f"{type(error).__name__}: {error}", "outputs": []}
                result["cases"].append(case)
                print(f"[一致性] {entry['name']} / {case['source']}：{case['status']}")
                if case.get("error"):
                    print(f"  {case['error']}")
                for item in case.get("outputs", []):
                    print(f"  {item['name']}：{item['status']}，max_abs={item.get('max_abs_error')}，close={item.get('close_fraction')}")
                detection = case.get("detections")
                if detection is not None:
                    print(f"  检测框匹配：{detection['status']}，原因={detection.get('reason', '')}")
                    for matched in detection.get("images", []):
                        print(f"    PyTorch={matched['pytorch_count']}，TensorRT={matched['tensorrt_count']}，匹配={matched['matched_count']}")
    finally:
        try:
            torch.cuda.synchronize(device)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32
            model = runner = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    result["status"] = aggregate_status([case["status"] for case in result["cases"]])
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def save_reports(report: dict[str, Any], csv_path: Path, json_path: Path) -> None:
    """独立正确性报告，不把误差指标混入速度 CSV。"""
    if csv_path.resolve() == json_path.resolve():
        raise ValueError("consistency.output 与 json_output 不能是同一文件")
    rows: list[dict[str, Any]] = []
    for model in report["models"]:
        cases = model.get("cases") or [{"source": "", "status": model["status"], "error": model.get("error", "")}]
        for case in cases:
            base = {"model": model["name"], "source": case["source"], "case_status": case["status"],
                    "precision": model.get("precision", ""), "error": case.get("error", "")}
            rows.append({**base, "record_type": "case", "status": case["status"]})
            for item in case.get("outputs", []):
                rows.append({**base, "record_type": "tensor", **item})
            detection = case.get("detections")
            if detection is not None:
                items = detection.get("images") or [{"status": detection["status"]}]
                for index, item in enumerate(items):
                    rows.append({**base, "record_type": "detections", "batch_index": index,
                                 "reason": detection.get("reason", ""), **item})
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["record_type", "status"]
    for path in (csv_path, json_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_json_safe(rows))
    json_path.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="同一输入的 PyTorch/TensorRT 输出一致性检查（不测速、不构建 engine）")
    parser.add_argument("--config", type=Path, default=None, help="默认读取同目录 benchmark_config.yaml")
    parser.add_argument("--run-dir", default=None, help="复用指定的 runN 目录；省略时自动创建 runs-profile/runN")
    parser.add_argument("--source", type=Path, default=None, help="临时覆盖检查图片/目录，命令行相对路径按当前工作目录解析")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = prepare_run_directory(config, args.run_dir)
    apply_python_paths(config)
    values = config.setdefault("consistency", {})
    if args.source is not None:
        values["source"] = str(args.source.resolve())
    validate_tolerances(float(values.get("atol", 0.001)), float(values.get("rtol", 0.01)))
    report: dict[str, Any] = {"created_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                              "config_path": config["_config_path"], "settings": values, "models": [],
                              "scope": "shared-input model-output comparison; not dataset accuracy/mAP"}
    if run_dir is not None:
        report["run_dir"] = str(run_dir)
    for entry in get_model_entries(config):
        try:
            result = _check_model(entry, config)
        except Exception as error:
            result = {"name": entry["name"], "status": "error", "error": f"{type(error).__name__}: {error}"}
            print(f"[一致性错误] {entry['name']}：{result['error']}")
        report["models"].append(result)
    report["status"] = aggregate_status([item["status"] for item in report["models"]])
    root = Path(config["project"]["root"])
    output = Path(values.get("output") or root / "runs-profile/consistency.csv")
    json_output = Path(values.get("json_output") or root / "runs-profile/consistency.json")
    save_reports(report, output, json_output)
    if run_dir is not None:
        print(f"本次运行目录：{run_dir}")
    print(f"一致性总状态：{report['status']}；CSV：{output}；JSON：{json_output}")
    print("passed 只表示这些输入在所设容差下通过；warning/空框/缺失依赖不算验收通过，也不代表 mAP 不变。")
    return {**report, "output": str(output), "json_output": str(json_output)}


if __name__ == "__main__":
    raise SystemExit(0 if main()["status"] == "passed" else 1)

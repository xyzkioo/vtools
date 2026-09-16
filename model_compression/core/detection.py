"""YOLO detection evaluation and checkpoint pruning (multiple boxes per image)."""
from __future__ import annotations

from pathlib import Path
import math
import statistics
import time
from typing import Any, Mapping

from .model_runtime import parameter_metrics, resolve_device
from .registry import artifact_info
from ..modules.pruning import prune_unstructured
from ..modules.dependency_pruning import prune_detector_dependency_aware
from ..modules.structured import build_structured_detector, finetune_structured_detector


def detection_data(config: Mapping[str, Any]) -> Path | None:
    dataset = config.get('dataset', {})
    explicit = dataset.get('data')
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'检测数据集 YAML 不存在：{path}')
        return path
    # Compatibility with the former train/val directory fields; never invent class names.
    candidates = set()
    for key in ('train', 'val'):
        value = dataset.get(key)
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        if path.suffix.lower() in {'.yaml', '.yml'} and path.is_file():
            candidates.add(path)
        else:
            for parent in (path, path.parent):
                for name in ('data.yaml', 'data.yml'):
                    candidate = parent / name
                    if candidate.is_file():
                        candidates.add(candidate)
    if len(candidates) > 1:
        raise ValueError('发现多份 data.yaml，请明确选择 dataset.data，避免使用错误的数据集。')
    return next(iter(candidates), None)


def is_detection(config: Mapping[str, Any]) -> bool:
    model = config.get('model') or {}
    task = str(model.get('task') or '').strip().lower()
    if task in {'detect', 'classify'}:
        return task == 'detect'
    if model.get('adapter') == 'ultralytics':
        return True
    dataset = config.get('dataset') or {}
    if dataset.get('data'):
        return True
    for key in ('train', 'val'):
        value = dataset.get(key)
        if value:
            path = Path(value)
            if (path / 'images').is_dir() and (path / 'labels').is_dir():
                return True
    return False


def load_detector(weights: str | Path):
    from ultralytics import YOLO
    path = Path(weights).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'找不到检测模型权重：{path}')
    detector = YOLO(str(path))
    if detector.task != 'detect':
        raise ValueError(f'需要目标检测模型，当前权重任务为 {detector.task}')
    return detector


def _clean_detector_checkpoint(detector) -> None:
    """Remove stale EMA/model snapshots before serializing a transformed model."""
    checkpoint = getattr(detector, "ckpt", None)
    if isinstance(checkpoint, Mapping):
        detector.ckpt = {key: value for key, value in checkpoint.items()
                         if key not in {"ema", "optimizer", "model"}}


def _parameter_signature(model) -> tuple[int, tuple[tuple[str, tuple[int, ...]], ...]]:
    params = list(model.named_parameters())
    return sum(int(param.numel()) for _, param in params), tuple(
        (str(name), tuple(int(dim) for dim in param.shape)) for name, param in params
    )


def _unique_target(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"无法为检测模型产物生成唯一文件名：{path}")


def evaluate_detector(detector, config: Mapping[str, Any], run_dir: Path, name: str) -> dict[str, Any]:
    data = detection_data(config)
    if data is None:
        raise ValueError('检测压缩评估需要 dataset.data（YOLO data.yaml，包含 names 和 val）。')
    evaluation = config.get('evaluation', {})
    device = config.get('model', {}).get('device', 'auto')
    device = None if device == 'auto' else str(device).replace('cuda:', '')
    # ``val`` may fuse the model in place. Callers that need to mutate the
    # checkpoint after validation reload it before doing so.
    metrics = detector.val(
        data=str(data), split=evaluation.get('split', 'val'),
        imgsz=config.get('dataset', {}).get('input_size', 640),
        batch=int(evaluation.get('batch_size', 1)), device=device,
        workers=0, project=str(run_dir), name=name, exist_ok=True,
        plots=True, save_json=False, half=False,
        conf=float(evaluation.get('conf', 0.001)), iou=float(evaluation.get('iou', 0.7)),
        fraction=float(evaluation.get('fraction', 1.0)),
    )
    box = getattr(metrics, 'box', None)
    if box is None:
        raise RuntimeError('Ultralytics 验证没有返回检测框指标。')

    def metric(name: str) -> float:
        value = getattr(box, name, None)
        if value is None:
            raise RuntimeError(f'Ultralytics 验证缺少检测指标：{name}')
        return float(value)

    return {'evaluation_map50': metric('map50'),
            'evaluation_map50_95': metric('map'),
            'evaluation_precision': metric('mp'),
            'evaluation_recall': metric('mr')}


def benchmark_detector(detector, config: Mapping[str, Any]) -> dict[str, Any]:
    """Measure model forward latency with fixed input and warmup settings."""

    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("检测测速需要 PyTorch") from exc
    compression = config.get('compression', {})
    section = compression.get('structured', {}) if isinstance(compression, Mapping) else {}
    section = section if isinstance(section, Mapping) else {}
    device_value = config.get('model', {}).get('device', 'auto')
    device = resolve_device(device_value)
    input_size = config.get('dataset', {}).get('input_size', 640)
    if isinstance(input_size, (list, tuple)):
        height, width = int(input_size[0]), int(input_size[1] if len(input_size) > 1 else input_size[0])
    else:
        height = width = int(input_size)
    warmup = int(section.get('benchmark_warmup', 5))
    repeats = int(section.get('benchmark_repeats', 20))
    if warmup < 0 or repeats < 1:
        raise ValueError('结构化测速参数必须满足 warmup >= 0 且 repeats >= 1')
    model = detector.model.to(device).eval()
    inputs = torch.zeros(1, 3, height, width, device=device)

    def sync() -> None:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        sync()
        durations: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(inputs)
            sync()
            durations.append((time.perf_counter() - start) * 1000.0)
    p95_index = max(0, min(len(durations) - 1, math.ceil(len(durations) * 0.95) - 1))
    return {
        'benchmark_device': str(device),
        'benchmark_input_size': [height, width],
        'benchmark_warmup': warmup,
        'benchmark_repeats': repeats,
        'benchmark_inference_ms_median': statistics.median(durations),
        'benchmark_inference_ms_p95': sorted(durations)[p95_index],
    }


def run_detection_operation(module_id, config, registry, run_dir, run_id):
    """Use detection metrics and Ultralytics checkpoints for supported operations."""
    if module_id in {'compression.quantize.dynamic_int8', 'distillation.classification'}:
        raise ValueError('该模块为分类流程；YOLO 检测当前支持基线评估、非结构化剪枝、结构化缩放和导出。')
    branch_name = config.get('branch', {}).get('name', 'main')
    explicit = config.get('model', {}).get('weights')
    head, resolved_weights = registry.resolve_input_version(branch_name, explicit)
    if not resolved_weights:
        raise ValueError('请先选择 YOLO 模型权重。')
    weights = Path(resolved_weights)
    input_weights = weights
    detector = load_detector(weights)
    metadata = parameter_metrics(detector.model)
    if module_id == 'baseline.evaluate':
        metadata.update(evaluate_detector(detector, config, run_dir, 'baseline_validation'))
    elif module_id == 'compression.prune.unstructured':
        # Always compare on the same validation split, even if baseline was not selected.
        before = evaluate_detector(detector, config, run_dir, 'before_unstructured_pruning')
        # Validation may fuse layers; always prune a fresh model loaded from
        # the original checkpoint and evaluate the serialized result later.
        detector = load_detector(weights)
        detector.model, pruning = prune_unstructured(detector.model.float(), config.get('compression', {}))
        metadata.update(pruning)
        target = _unique_target(run_dir / 'artifacts' / 'unstructured_pruning' / 'yolo-unstructured-l1.pt')
        target.parent.mkdir(parents=True, exist_ok=True)
        # Avoid exporting an old EMA that would hide the pruned weights on reload.
        detector.ckpt = {key: value for key, value in (detector.ckpt or {}).items()
                         if key not in {'ema', 'optimizer', 'model'}}
        detector.save(str(target))
        reloaded = load_detector(target)
        metadata.update(evaluate_detector(reloaded, config, run_dir, 'after_unstructured_pruning'))
        metadata['before_unstructured_pruning'] = before
        metadata['map50_95_delta'] = metadata['evaluation_map50_95'] - before['evaluation_map50_95']
        metadata['note'] = '非结构化剪枝减少非零权重，不保证 PT 文件更小或普通推理更快。'
        weights = target
    elif module_id == 'compression.prune.structured':
        before = evaluate_detector(detector, config, run_dir, 'before_structured_pruning')
        before_speed = benchmark_detector(detector, config)
        source_file_size = artifact_info(weights)['size_bytes']
        detector = load_detector(weights)
        data = detection_data(config)
        if data is None:
            raise ValueError('结构化检测压缩需要 dataset.data（YOLO data.yaml）。')
        structured_section = config.get('compression', {}).get('structured', {})
        structured_section = structured_section if isinstance(structured_section, Mapping) else {}
        structured_method = str(structured_section.get('method', 'scale')).strip().lower()
        structured_dir = run_dir / 'artifacts' / 'structured_pruning'
        structured_dir.mkdir(parents=True, exist_ok=True)
        if structured_method in {'torch_pruning', 'torch-pruning', 'dependency', 'dependency_aware'}:
            detector.model, structured = prune_detector_dependency_aware(
                detector.model, config, structured_dir
            )
        elif structured_method in {'scale', 'width_scaling', 'yolo_scale'}:
            detector, structured = build_structured_detector(detector, weights, config.get('compression', {}), structured_dir)
        else:
            raise ValueError(
                '未知结构化剪枝方法：'
                f'{structured_method}；可选 scale 或 torch_pruning。'
            )
        detector, finetune = finetune_structured_detector(detector, config, data, structured_dir)
        metadata.update(structured)
        metadata.update(finetune)
        target = _unique_target(structured_dir / 'yolo-structured.pt')
        target.parent.mkdir(parents=True, exist_ok=True)
        expected_signature = _parameter_signature(detector.model)
        _clean_detector_checkpoint(detector)
        detector.save(str(target))
        reloaded = load_detector(target)
        actual_signature = _parameter_signature(reloaded.model)
        if actual_signature != expected_signature:
            raise RuntimeError(
                '结构化模型保存后重新加载的参数形状与剪枝结果不一致，疑似旧 EMA 覆盖了新模型；'
                f'保存前参数量={expected_signature[0]}，加载后参数量={actual_signature[0]}'
            )
        metadata.update(evaluate_detector(reloaded, config, run_dir, 'after_structured_pruning'))
        after_speed = benchmark_detector(reloaded, config)
        metadata.update({f'after_{key}': value for key, value in after_speed.items()})
        metadata.update({f'before_{key}': value for key, value in before_speed.items()})
        metadata['source_model_file_size_bytes'] = source_file_size
        target_file_size = artifact_info(target)['size_bytes']
        metadata['file_size_reduction'] = (1.0 - target_file_size / source_file_size) if source_file_size else None
        before_ms = before_speed.get('benchmark_inference_ms_median')
        after_ms = after_speed.get('benchmark_inference_ms_median')
        metadata['benchmark_speedup_ratio'] = (before_ms / after_ms) if before_ms and after_ms else None
        metadata['before_structured_pruning'] = before
        metadata['map50_95_delta'] = metadata['evaluation_map50_95'] - before['evaluation_map50_95']
        structured_section = config.get('compression', {}).get('structured', {})
        structured_section = structured_section if isinstance(structured_section, Mapping) else {}
        max_drop = float(structured_section.get('max_map50_95_drop', 0.05))
        quality_gate_enabled = bool(structured_section.get('quality_gate', True))
        quality_gate_passed = metadata['map50_95_delta'] >= -max_drop
        metadata['quality_gate'] = {
            'enabled': quality_gate_enabled,
            'max_map50_95_drop': max_drop,
            'passed': quality_gate_passed,
        }
        metadata['structured_method'] = structured_method
        metadata['note'] = (
            '依赖感知剪枝物理删除通道并同步调整依赖层；'
            if structured_method in {'torch_pruning', 'torch-pruning', 'dependency', 'dependency_aware'}
            else '结构化缩放真实改变网络通道数；'
        ) + '速度必须按同一设备、输入尺寸和预热条件实测。'
        if quality_gate_enabled and not quality_gate_passed:
            raise ValueError(
                '结构化压缩未通过精度门禁：mAP50-95 下降 '
                f'{-metadata["map50_95_delta"]:.4f}，允许最大下降 {max_drop:.4f}。'
                '请增加微调轮数、换用更大的目标规模，或将 compression.structured.quality_gate 设为 false 仅保留实验产物。'
            )
        weights = target
    elif module_id == 'artifact.export':
        import shutil
        source = Path(weights).resolve()
        target = config.get('export', {}).get('path') or run_dir / 'artifacts' / 'multi_processing' / f'{source.stem}-export{source.suffix}'
        target = Path(target).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target != source:
            target = _unique_target(target)
            shutil.copy2(source, target)
        metadata.update(evaluate_detector(load_detector(target), config, run_dir, 'export_validation'))
        export_info = {'source': artifact_info(source), 'target': target}
        weights = target
    elif module_id != 'model.parameters':
        raise ValueError(f'未支持的检测压缩模块：{module_id}')
    metadata['task'] = 'detect'
    metadata['model_file_size_bytes'] = artifact_info(weights)['size_bytes']
    if not head:
        baseline_metadata = metadata
        if 'before_unstructured_pruning' in metadata:
            baseline_metadata = dict(metadata['before_unstructured_pruning'])
        elif 'before_structured_pruning' in metadata:
            baseline_metadata = dict(metadata['before_structured_pruning'])
        baseline_metadata['task'] = 'detect'
        original = registry.add_version(name=config.get('model', {}).get('name', 'yolo'),
            artifact=input_weights, parent_version_id=None, run_id=run_id, method='baseline',
            metadata=baseline_metadata)
        head = original['id']
        registry.advance_branch(branch_name, head)
    if module_id == 'compression.prune.unstructured':
        version = registry.add_version(name='yolo-pruned', artifact=weights, parent_version_id=head,
            run_id=run_id, method='unstructured_l1', metadata=metadata)
        head = version['id']
        registry.advance_branch(branch_name, head)
    elif module_id == 'compression.prune.structured':
        registry_method = 'structured_torch_pruning' if metadata.get('structured_method') in {
            'torch_pruning', 'torch-pruning', 'dependency', 'dependency_aware'
        } else 'structured_width_scaling'
        version = registry.add_version(name='yolo-structured', artifact=weights, parent_version_id=head,
            run_id=run_id, method=registry_method, metadata=metadata)
        head = version['id']
        registry.advance_branch(branch_name, head)
    elif module_id in {'baseline.evaluate', 'model.parameters'}:
        registry.data['versions'][head].setdefault('metadata', {}).update(metadata)
        registry.save()
    if 'export_info' in locals():
        metadata['verification'] = {'reload': True, 'validation': 'evaluated', 'task': 'detect'}
    result = {'module': module_id, 'status': 'succeeded', 'version_id': head,
              'artifact': artifact_info(weights), 'metadata': metadata}
    return result

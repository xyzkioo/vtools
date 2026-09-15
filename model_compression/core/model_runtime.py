"""PyTorch 模型加载、适配器和分类评测的最小运行时。"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional


def require_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("该操作需要 PyTorch，请先安装与设备匹配的 torch/torchvision") from exc
    return torch


def resolve_device(value: Any = "auto"):
    torch = require_torch()
    text = str(value or "auto").lower()
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text.isdigit():
        text = f"cuda:{text}"
    if text.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(text)


def _load_adapter(spec: Any):
    if spec in (None, "", "torch", "builtin"):
        return None
    text = str(spec)
    if text.endswith(".py") or "/" in text or "\\" in text:
        path = Path(text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到模型 adapter：{path}")
        name = f"vtools_model_adapter_{uuid.uuid4().hex}"
        module_spec = importlib.util.spec_from_file_location(name, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"无法加载模型 adapter：{path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        module_spec.loader.exec_module(module)
        return module
    return importlib.import_module(text)


def _call(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """只向 adapter 传递它声明支持的关键字，兼容旧模板。"""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args, **kwargs)
    if not any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in signature.parameters.values()):
        positional = [
            parameter for parameter in signature.parameters.values()
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        args = args[: len(positional)]
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return function(*args, **kwargs)
    accepted = {name for name in signature.parameters if name not in {"self"}}
    return function(*args, **{key: value for key, value in kwargs.items() if key in accepted})


def load_model(model_config: Mapping[str, Any], *, device: Any = "auto"):
    """通过内置 torch loader 或 adapter 构建模型。"""

    torch = require_torch()
    target_device = resolve_device(device)
    adapter = _load_adapter(model_config.get("adapter"))
    weights = model_config.get("weights")
    if adapter is not None:
        if hasattr(adapter, "load_model"):
            function = adapter.load_model
            first_name = next(iter(inspect.signature(function).parameters), "")
            first_value = weights if first_name in {"weights", "path", "checkpoint"} else model_config
            model = _call(function, first_value, device=target_device)
        elif hasattr(adapter, "build_model"):
            function = adapter.build_model
            first_name = next(iter(inspect.signature(function).parameters), "")
            first_value = weights if first_name in {"weights", "path", "checkpoint"} else model_config
            model = _call(function, first_value, device=target_device)
            if hasattr(adapter, "load_weights") and weights:
                _call(adapter.load_weights, model, weights, device=target_device)
        else:
            raise AttributeError("adapter 必须提供 load_model 或 build_model")
        if not hasattr(model, "to"):
            raise TypeError("adapter 返回的对象不是 PyTorch 模型")
        return model.to(target_device), adapter

    if not weights:
        factory = model_config.get("factory")
        if factory:
            model = load_factory(str(factory), model_config)
            if not hasattr(model, "to"):
                raise TypeError("model.factory 返回的对象不是 PyTorch 模型")
            return model.to(target_device), None
        raise ValueError("model.weights 不能为空；分支元数据操作可不加载模型")
    path = Path(str(weights)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到模型权重：{path}")
    try:
        payload = torch.load(path, map_location=target_device, weights_only=False)
    except TypeError:  # torch < 2.0
        payload = torch.load(path, map_location=target_device)
    force_factory = bool(model_config.get("force_factory")) and bool(model_config.get("factory"))
    if isinstance(payload, torch.nn.Module):
        if not force_factory:
            return payload.to(target_device), None
        model = load_factory(str(model_config["factory"]), model_config)
        model.load_state_dict(payload.state_dict(), strict=bool(model_config.get("strict", True)))
        return model.to(target_device), None
    if isinstance(payload, Mapping):
        for key in ("model", "ema", "module"):
            candidate = payload.get(key)
            if isinstance(candidate, torch.nn.Module) and not force_factory:
                return candidate.to(target_device), None
        state_dict = payload.get("state_dict", payload)
        if force_factory and isinstance(payload.get("model"), torch.nn.Module):
            state_dict = payload["model"].state_dict()
        if isinstance(state_dict, Mapping):
            factory = model_config.get("factory")
            if not factory:
                raise ValueError("权重是 state_dict，但 model.factory 未提供模型构造函数")
            model = load_factory(str(factory), model_config)
            model.load_state_dict(state_dict, strict=bool(model_config.get("strict", True)))
            return model.to(target_device), None
    raise TypeError(f"无法从 {path} 恢复 PyTorch 模型；支持 nn.Module 或 state_dict")


def load_factory(spec: str, config: Mapping[str, Any] | None = None) -> Any:
    if ":" in spec:
        module_name, function_name = spec.split(":", 1)
    else:
        module_name, function_name = spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return factory(config or {})
    return _call(factory, config or {}) if parameters else _call(factory)


def save_model(model: Any, path: str | Path, adapter: Any = None) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if adapter is not None and hasattr(adapter, "save_model"):
        _call(adapter.save_model, model, target)
    else:
        torch = require_torch()
        torch.save(model, target)
    if not target.is_file():
        raise RuntimeError(f"模型保存失败：{target}")
    return target


def extract_logits(output: Any):
    """从常见分类模型输出中取 logits。"""

    if isinstance(output, Mapping):
        for key in ("logits", "pred", "output"):
            if key in output:
                return extract_logits(output[key])
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("模型输出为空")
        return extract_logits(output[0])
    return output


def _default_loader(path: Any, *, batch_size: int, train: bool, config: Mapping[str, Any]):
    torch = require_torch()
    if not path:
        return None
    try:
        from torchvision import datasets, transforms  # type: ignore
    except ImportError as exc:
        raise RuntimeError("使用 ImageFolder 数据集需要 torchvision") from exc
    root = Path(str(path)).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"数据集目录不存在：{root}")
    if (root / "images").is_dir() and (root / "labels").is_dir():
        raise ValueError(
            "检测到 YOLO 数据集目录（images/labels）；模型压缩内置评测器当前只支持分类 ImageFolder，"
            "请提供 <数据集>/<类别名>/*.jpg 目录，或改用检测诊断/模型测速工具。"
        )
    size = config.get("input_size", [224, 224])
    if isinstance(size, int):
        size = [size, size]
    transform = transforms.Compose([
        transforms.Resize((int(size[0]), int(size[1]))),
        transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda image: image),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0)


def build_dataloaders(config: Mapping[str, Any], adapter: Any = None):
    dataset = config.get("dataset") if isinstance(config.get("dataset"), Mapping) else {}
    distill = config.get("distillation") if isinstance(config.get("distillation"), Mapping) else {}
    batch_size = int(distill.get("batch_size", config.get("batch_size", 1)))
    if adapter is not None and hasattr(adapter, "build_dataloader"):
        try:
            train = _call(adapter.build_dataloader, "train", config, batch_size=batch_size)
            val = _call(adapter.build_dataloader, "val", config, batch_size=batch_size)
            return train, val
        except NotImplementedError:
            return None, None
    train_path, val_path = dataset.get("train"), dataset.get("val")
    if not train_path and not val_path:
        return None, None
    common = {"input_size": config.get("input_size", [224, 224])}
    return (
        _default_loader(train_path, batch_size=batch_size, train=True, config=common),
        _default_loader(val_path, batch_size=batch_size, train=False, config=common),
    )


def evaluate_classification(model: Any, loader: Any, *, device: Any = "auto") -> dict[str, Any]:
    if loader is None:
        return {"status": "skipped", "reason": "未配置分类验证集", "samples": 0}
    torch = require_torch()
    target_device = resolve_device(device)
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, Mapping):
                inputs, labels = batch["image"], batch["label"]
            else:
                inputs, labels = batch[0], batch[1]
            inputs, labels = inputs.to(target_device), labels.to(target_device)
            logits = extract_logits(model(inputs))
            loss = criterion(logits, labels)
            loss_sum += float(loss.item()) * labels.shape[0]
            total += int(labels.shape[0])
            correct += int((logits.argmax(dim=1) == labels).sum().item())
    return {
        "status": "succeeded",
        "accuracy": correct / total if total else None,
        "loss": loss_sum / total if total else None,
        "samples": total,
    }


def parameter_metrics(model: Any) -> dict[str, int]:
    parameters = list(model.parameters())
    return {
        "parameter_count": sum(int(item.numel()) for item in parameters),
        "trainable_parameter_count": sum(int(item.numel()) for item in parameters if item.requires_grad),
        "zero_parameter_count": sum(int((item.detach() == 0).sum().item()) for item in parameters),
    }


def benchmark_model_call(model: Any, config: Mapping[str, Any], *, device: Any = "auto") -> dict[str, Any]:
    """用合成 NCHW 输入测量纯模型调用；输入不匹配时返回可读的跳过原因。"""

    torch = require_torch()
    values = config.get("benchmark") if isinstance(config.get("benchmark"), Mapping) else config
    shape = values.get("input_size", [224, 224])
    if isinstance(shape, int):
        shape = [shape, shape]
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        return {"status": "skipped", "reason": "benchmark.input_size 不是 [height, width]"}
    batch_size = int(values.get("batch_size", 1))
    channels = int(values.get("input_channels", 3))
    warmup = max(0, int(values.get("warmup", 2)))
    repeats = max(1, int(values.get("repeats", 10)))
    target_device = resolve_device(device)
    model.eval()
    try:
        sample = torch.randn(batch_size, channels, int(shape[0]), int(shape[1]), device=target_device)
        with torch.no_grad():
            for _ in range(warmup):
                model(sample)
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            durations: list[float] = []
            for _ in range(repeats):
                start = time.perf_counter()
                model(sample)
                if target_device.type == "cuda":
                    torch.cuda.synchronize(target_device)
                durations.append((time.perf_counter() - start) * 1000.0)
    except Exception as exc:  # noqa: BLE001 - 输入签名由 adapter 决定
        return {"status": "skipped", "reason": f"合成输入不适用：{type(exc).__name__}: {exc}"}
    ordered = sorted(durations)
    p50 = ordered[(len(ordered) - 1) * 50 // 100]
    p95 = ordered[(len(ordered) - 1) * 95 // 100]
    return {"status": "succeeded", "latency_mean_ms": sum(durations) / len(durations), "latency_p50_ms": p50, "latency_p95_ms": p95, "warmup": warmup, "repeats": repeats, "device": str(target_device)}


__all__ = [
    "build_dataloaders",
    "benchmark_model_call",
    "evaluate_classification",
    "extract_logits",
    "load_model",
    "parameter_metrics",
    "require_torch",
    "resolve_device",
    "save_model",
]

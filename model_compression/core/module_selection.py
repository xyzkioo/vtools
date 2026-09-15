"""模型压缩模块开关和命令行选择。"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


COMPRESSION_MODULES: dict[str, bool] = {
    "branch.create": False,
    "model.parameters": True,
    "baseline.evaluate": True,
    "compression.quantize.dynamic_int8": False,
    "compression.prune.unstructured": False,
    "compression.prune.structured": False,
    "distillation.classification": False,
    "comparison.report": False,
    "artifact.export": False,
}


def _split(values: Iterable[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or ():
        result.extend(item.strip() for item in str(value).split(",") if item.strip())
    return result


def resolve_compression_modules(
    config: Mapping[str, Any],
    *,
    only: Iterable[str] | None = None,
    enable: Iterable[str] | None = None,
    disable: Iterable[str] | None = None,
) -> dict[str, bool]:
    """按 YAML 和 CLI 解析模块。

    只要配置中存在 ``modules``，它就是显式 allow-list，省略的模块关闭，
    与 speed_test/model_diagnostics 的行为保持一致。
    """

    modules = dict(COMPRESSION_MODULES)
    configured = config.get("modules")
    if isinstance(configured, Mapping):
        unknown = set(map(str, configured)) - set(modules)
        if unknown:
            raise ValueError(f"未知模型压缩模块：{', '.join(sorted(unknown))}")
        modules = {key: False for key in modules}
        modules.update({str(key): bool(value) for key, value in configured.items()})
    elif configured is not None:
        raise ValueError("modules 必须是 module_id: true/false 的对象")

    only_values, enable_values, disable_values = (
        _split(only),
        _split(enable),
        _split(disable),
    )
    if only_values and (enable_values or disable_values):
        raise ValueError("--only 不能与 --enable/--disable 同时使用")
    requested = set(only_values + enable_values + disable_values)
    unknown = requested - set(modules)
    if unknown:
        raise ValueError(f"未知模型压缩模块：{', '.join(sorted(unknown))}")
    overlap = set(enable_values) & set(disable_values)
    if overlap:
        raise ValueError(f"模块同时启用和禁用：{', '.join(sorted(overlap))}")
    if only_values:
        selected = set(only_values)
        modules = {key: key in selected for key in modules}
    else:
        modules.update({key: True for key in enable_values})
        modules.update({key: False for key in disable_values})
    operation = str(config.get("operation", "all") or "all").strip().lower()
    if not (only_values or enable_values or disable_values) and operation in OPERATION_MODULES:
        modules = {key: key == OPERATION_MODULES[operation] for key in modules}
    return modules


__all__ = ["COMPRESSION_MODULES", "resolve_compression_modules"]


OPERATION_MODULES = {
    "branch": "branch.create",
    "baseline": "baseline.evaluate",
    "parameters": "model.parameters",
    "quantize": "compression.quantize.dynamic_int8",
    "quantization": "compression.quantize.dynamic_int8",
    "dynamic_int8": "compression.quantize.dynamic_int8",
    "prune": "compression.prune.unstructured",
    "pruning": "compression.prune.unstructured",
    "structured_prune": "compression.prune.structured",
    "structured": "compression.prune.structured",
    "distill": "distillation.classification",
    "distillation": "distillation.classification",
    "compare": "comparison.report",
    "export": "artifact.export",
}

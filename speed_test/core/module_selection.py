"""测速功能开关和命令行选择。"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


SPEED_MODULES = {
    "checkpoint.inspect": False,
    "checkpoint.load_check": False,
    "model.parameters": False,
    "model.flops": False,
    "speed.pytorch_call": True,
    "speed.pytorch_pipeline": False,
    "speed.tensorrt_call": True,
    "memory.pytorch_peak": False,
    "export.onnx": False,
    "build.tensorrt": False,
    "consistency.tensor": True,
    "consistency.detection": False,
}


def _split(values: Iterable[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or ():
        result.extend(item.strip() for item in str(value).split(",") if item.strip())
    return result


def resolve_speed_modules(config: Mapping[str, Any], *, only: Iterable[str] | None = None, enable: Iterable[str] | None = None, disable: Iterable[str] | None = None) -> dict[str, bool]:
    modules = dict(SPEED_MODULES)
    configured = config.get("modules")
    if isinstance(configured, Mapping):
        unknown = set(map(str, configured)) - set(modules)
        if unknown:
            raise ValueError(f"未知测速模块：{', '.join(sorted(unknown))}")
        # A new-style modules section is an explicit allow-list.  Legacy
        # configs without this section keep the compatibility defaults below.
        modules = {key: False for key in modules}
        modules.update({str(key): bool(value) for key, value in configured.items()})
    elif configured is not None:
        raise ValueError("modules 必须是 module_id: true/false 的对象")
    only_values, enable_values, disable_values = _split(only), _split(enable), _split(disable)
    if only_values and (enable_values or disable_values):
        raise ValueError("--only 不能与 --enable/--disable 同时使用")
    requested = set(only_values + enable_values + disable_values)
    unknown = requested - set(modules)
    if unknown:
        raise ValueError(f"未知测速模块：{', '.join(sorted(unknown))}")
    overlap = set(enable_values) & set(disable_values)
    if overlap:
        raise ValueError(f"模块同时启用和禁用：{', '.join(sorted(overlap))}")
    if only_values:
        selected = set(only_values)
        modules = {key: key in selected for key in modules}
    else:
        modules.update({key: True for key in enable_values})
        modules.update({key: False for key in disable_values})
    return modules

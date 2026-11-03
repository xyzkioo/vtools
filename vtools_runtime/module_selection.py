"""Shared selection rules for optional tool modules."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def _split(values: Iterable[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or ():
        result.extend(item.strip() for item in str(value).split(',') if item.strip())
    return result


def resolve_module_selection(
    defaults: Mapping[str, bool],
    config: Mapping[str, Any],
    *,
    only: Iterable[str] | None = None,
    enable: Iterable[str] | None = None,
    disable: Iterable[str] | None = None,
    label: str = '',
    unknown_hint: str = '',
) -> dict[str, bool]:
    """Apply a YAML allow-list and CLI overrides, preserving an empty --only."""
    modules = dict(defaults)
    configured = config.get('modules')
    if configured is not None:
        if not isinstance(configured, Mapping):
            raise ValueError('modules 必须是 module_id: true/false 的对象')
        unknown = sorted(set(map(str, configured)) - set(modules))
        if unknown:
            raise ValueError(f'未知{label}模块：{", ".join(unknown)}{unknown_hint}')
        modules = {key: False for key in modules}
        modules.update({str(key): bool(value) for key, value in configured.items()})
    only_values, enable_values, disable_values = _split(only), _split(enable), _split(disable)
    if only is not None and (enable_values or disable_values):
        raise ValueError('--only 不能与 --enable/--disable 同时使用')
    unknown = sorted((set(only_values) | set(enable_values) | set(disable_values)) - set(modules))
    if unknown:
        raise ValueError(f'未知{label}模块：{", ".join(unknown)}{unknown_hint}')
    overlap = set(enable_values) & set(disable_values)
    if overlap:
        raise ValueError(f'模块同时启用和禁用：{", ".join(sorted(overlap))}')
    if only is not None:
        selected = set(only_values)
        return {key: key in selected for key in modules}
    modules.update({key: True for key in enable_values})
    modules.update({key: False for key in disable_values})
    return modules

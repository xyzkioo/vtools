"""模型压缩模块开关和命令行选择。"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


COMPRESSION_MODULES: dict[str, bool] = {
    "model.parameters": True,
    "baseline.evaluate": True,
    "compression.quantize.dynamic_int8": False,
    "compression.prune.unstructured": False,
    "compression.prune.structured": False,
    "distillation.classification": False,
    "distillation.detection": False,
    "comparison.report": False,
    "artifact.export": False,
}


COMPRESSION_MODES: dict[str, dict[str, Any]] = {
    'baseline': {'label': '基线评估', 'tasks': ['detect', 'classify'],
                 'description': '统计公共模型参数并评估精度。', 'primary_module': None,
                 'modules': ['model.parameters', 'baseline.evaluate'], 'public_weights': True},
    'structured': {'label': '结构化剪枝', 'tasks': ['detect'],
                   'description': '对公共模型裁剪通道或缩小结构，并按配置微调、评估。',
                   'primary_module': 'compression.prune.structured',
                   'modules': ['model.parameters', 'baseline.evaluate', 'compression.prune.structured'], 'public_weights': True},
    'unstructured': {'label': '非结构化剪枝', 'tasks': ['detect', 'classify'],
                     'description': '对公共模型的部分权重置零，并评估剪枝结果。',
                     'primary_module': 'compression.prune.unstructured',
                     'modules': ['model.parameters', 'baseline.evaluate', 'compression.prune.unstructured'], 'public_weights': True},
    'detection_distillation': {'label': '检测知识蒸馏', 'tasks': ['detect'],
                               'description': '使用教师和学生权重；自动适配学生、评估学生基线并训练，无需公共模型权重。',
                               'primary_module': 'distillation.detection',
                               'modules': ['distillation.detection'], 'public_weights': False},
    'quantization': {'label': '动态 INT8 量化', 'tasks': ['classify'],
                     'description': '对公共分类模型执行动态 INT8 量化，并评估结果。',
                     'primary_module': 'compression.quantize.dynamic_int8',
                     'modules': ['model.parameters', 'baseline.evaluate', 'compression.quantize.dynamic_int8'], 'public_weights': True},
    'classification_distillation': {'label': '分类知识蒸馏', 'tasks': ['classify'],
                                    'description': '使用 YAML 中的 teacher/student 配置训练分类学生。',
                                    'primary_module': 'distillation.classification',
                                    'modules': ['model.parameters', 'baseline.evaluate', 'distillation.classification'], 'public_weights': True},
}


def compression_mode_modules(mode: str, task: str) -> list[str]:
    definition = COMPRESSION_MODES.get(mode)
    if definition is None:
        raise ValueError(f'未知处理模式：{mode}')
    if task not in definition['tasks']:
        raise ValueError(f'处理模式「{definition["label"]}」不支持 {task} 任务。')
    return list(definition['modules'])


def standalone_detection_distillation(modules: Mapping[str, bool]) -> bool:
    allowed = {'model.parameters', 'baseline.evaluate', 'distillation.detection', 'comparison.report', 'artifact.export'}
    return bool(modules.get('distillation.detection')) and all(not enabled or key in allowed for key, enabled in modules.items())


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
    if only is not None and (enable_values or disable_values):
        raise ValueError("--only 不能与 --enable/--disable 同时使用")
    requested = set(only_values + enable_values + disable_values)
    unknown = requested - set(modules)
    if unknown:
        raise ValueError(f"未知模型压缩模块：{', '.join(sorted(unknown))}")
    overlap = set(enable_values) & set(disable_values)
    if overlap:
        raise ValueError(f"模块同时启用和禁用：{', '.join(sorted(overlap))}")
    # Preserve an explicitly empty selection instead of restoring YAML defaults.
    if only is not None:
        selected = set(only_values)
        modules = {key: key in selected for key in modules}
    else:
        modules.update({key: True for key in enable_values})
        modules.update({key: False for key in disable_values})
    return modules


__all__ = ["COMPRESSION_MODULES", "COMPRESSION_MODES", "compression_mode_modules",
           "standalone_detection_distillation", "resolve_compression_modules"]

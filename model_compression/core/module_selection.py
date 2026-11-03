"""模型压缩模块开关和命令行选择。"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from vtools_runtime.module_selection import resolve_module_selection


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

    return resolve_module_selection(COMPRESSION_MODULES, config, only=only, enable=enable,
                                    disable=disable, label='模型压缩')


__all__ = ["COMPRESSION_MODULES", "COMPRESSION_MODES", "compression_mode_modules",
           "standalone_detection_distillation", "resolve_compression_modules"]

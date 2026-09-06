# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ultralytics 训练、验证、预测和导出流程的基础回调函数。"""

from collections import defaultdict
from copy import deepcopy

# 训练器回调 -----------------------------------------------------------------------------------------------------------


def on_pretrain_routine_start(trainer):
    """在预训练流程开始时调用，此时尚未加载数据和设置模型。"""


def on_pretrain_routine_end(trainer):
    """在预训练流程结束时调用，此时数据加载和模型设置已经完成。"""


def on_train_start(trainer):
    """训练开始时调用，此时第一个周期尚未开始。"""


def on_train_epoch_start(trainer):
    """每个训练周期开始时调用，此时尚未开始遍历批次。"""


def on_train_batch_start(trainer):
    """每个训练批次开始时调用，此时尚未执行前向传播。"""


def optimizer_step(trainer):
    """优化器执行更新步骤时调用。此回调保留给自定义集成，默认不会调用。"""


def on_before_zero_grad(trainer):
    """梯度清零前调用。此回调保留给自定义集成，默认不会调用。"""


def on_train_batch_end(trainer):
    """每个训练批次结束时调用，此时已经完成反向传播。优化器更新可能因梯度累积而延后。"""


def on_train_epoch_end(trainer):
    """每个训练周期结束时调用，此时已完成所有批次，但尚未开始验证。"""


def on_fit_epoch_end(trainer):
    """每个完整周期（训练 + 验证）结束时调用，此时已完成验证和所有必要的检查点保存。"""


def on_model_save(trainer):
    """模型检查点保存后调用。"""


def on_train_end(trainer):
    """训练结束时调用，此时已经完成最佳模型的最终评估。"""


def on_params_update(trainer):
    """模型参数更新后调用。此回调保留给自定义集成，默认不会调用。"""


def teardown(trainer):
    """训练流程清理阶段调用。"""


# 验证器回调 -----------------------------------------------------------------------------------------------------------


def on_val_start(validator):
    """验证开始时调用。"""


def on_val_batch_start(validator):
    """每个验证批次开始时调用。"""


def on_val_batch_end(validator):
    """每个验证批次结束时调用。"""


def on_val_end(validator):
    """验证结束时调用。"""


# 预测器回调 -----------------------------------------------------------------------------------------------------------


def on_predict_start(predictor):
    """预测开始时调用。"""


def on_predict_batch_start(predictor):
    """每个预测批次开始时调用。"""


def on_predict_batch_end(predictor):
    """每个预测批次结束时调用。"""


def on_predict_postprocess_end(predictor):
    """预测后处理结束后调用。"""


def on_predict_end(predictor):
    """预测结束时调用。"""


# 导出器回调 -----------------------------------------------------------------------------------------------------------


def on_export_start(exporter):
    """模型导出开始时调用。"""


def on_export_end(exporter):
    """模型导出结束时调用。"""


default_callbacks = {
    # 在训练器中运行
    "on_pretrain_routine_start": [on_pretrain_routine_start],
    "on_pretrain_routine_end": [on_pretrain_routine_end],
    "on_train_start": [on_train_start],
    "on_train_epoch_start": [on_train_epoch_start],
    "on_train_batch_start": [on_train_batch_start],
    "optimizer_step": [optimizer_step],
    "on_before_zero_grad": [on_before_zero_grad],
    "on_train_batch_end": [on_train_batch_end],
    "on_train_epoch_end": [on_train_epoch_end],
    "on_fit_epoch_end": [on_fit_epoch_end],  # fit = train + val
    "on_model_save": [on_model_save],
    "on_train_end": [on_train_end],
    "on_params_update": [on_params_update],
    "teardown": [teardown],
    # 在验证器中运行
    "on_val_start": [on_val_start],
    "on_val_batch_start": [on_val_batch_start],
    "on_val_batch_end": [on_val_batch_end],
    "on_val_end": [on_val_end],
    # 在预测器中运行
    "on_predict_start": [on_predict_start],
    "on_predict_batch_start": [on_predict_batch_start],
    "on_predict_postprocess_end": [on_predict_postprocess_end],
    "on_predict_batch_end": [on_predict_batch_end],
    "on_predict_end": [on_predict_end],
    # 在导出器中运行
    "on_export_start": [on_export_start],
    "on_export_end": [on_export_end],
}


def get_default_callbacks():
    """获取 Ultralytics 训练、验证、预测和导出流程的默认回调。

    返回：
        (dict): 各类训练事件的默认回调字典。每个键表示训练流程中的一个事件，对应值是该事件发生时执行的
            回调函数列表。

    示例：
        >>> callbacks = get_default_callbacks()
        >>> print(list(callbacks.keys()))  # 显示所有可用的回调事件
        ['on_pretrain_routine_start', 'on_pretrain_routine_end', ...]
    """
    return defaultdict(list, deepcopy(default_callbacks))


def add_integration_callbacks(instance):
    """将集成回调添加到实例的回调字典中。

    此函数为每个实例加载并添加分析回调。训练器实例还会接收 Platform，以及 ClearML、Comet、DVC、MLflow、
    Neptune、Ray Tune、TensorBoard 和 Weights & Biases 的实验日志回调。

    参数：
        instance (Trainer | Predictor | Validator | Exporter): 要添加回调的对象实例。
            type of instance determines which callbacks are loaded.

    示例：
        >>> from ultralytics.engine.trainer import BaseTrainer
        >>> trainer = BaseTrainer()
        >>> add_integration_callbacks(trainer)
    """
    from ultralytics.utils.events import callbacks as events_cb

    callbacks_list = [events_cb]

    # 加载训练回调
    if "Trainer" in instance.__class__.__name__:
        from .clearml import callbacks as clear_cb
        from .comet import callbacks as comet_cb
        from .dvc import callbacks as dvc_cb
        from .mlflow import callbacks as mlflow_cb
        from .neptune import callbacks as neptune_cb
        from .platform import callbacks as platform_cb
        from .raytune import callbacks as tune_cb
        from .tensorboard import callbacks as tb_cb
        from .wb import callbacks as wb_cb

        callbacks_list.extend([platform_cb, clear_cb, comet_cb, dvc_cb, mlflow_cb, neptune_cb, tune_cb, tb_cb, wb_cb])

    # 将回调添加到回调字典
    for callbacks in callbacks_list:
        for k, v in callbacks.items():
            if v not in instance.callbacks[k]:
                instance.callbacks[k].append(v)

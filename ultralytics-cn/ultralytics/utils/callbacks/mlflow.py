# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics YOLO 的 MLflow 日志记录。

此模块为 Ultralytics YOLO 启用 MLflow 日志记录，用于记录指标、参数和模型产物。
使用前应指定跟踪 URI，也可以通过环境变量自定义日志记录行为。

Commands:
    1. 设置项目名称：
        `export MLFLOW_EXPERIMENT_NAME=<your_experiment_name>` or use the project=<project> argument

    2. 设置运行名称：
        `export MLFLOW_RUN=<your_run_name>` or use the name=<name> argument

    3. 启动本地 MLflow 服务器：
        mlflow server --backend-store-uri runs/mlflow
       默认会在 http://127.0.0.1:5000 启动本地服务器。
       如需指定其他 URI，请设置 MLFLOW_TRACKING_URI 环境变量。

    4. 终止所有正在运行的 MLflow 服务器实例：
        ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9
"""

import os
from pathlib import Path

from ultralytics.utils import LOGGER, RUNS_DIR, SETTINGS, TESTS_RUNNING, colorstr, env_bool

PREFIX = colorstr("MLflow: ")

try:
    import mlflow

    assert hasattr(mlflow, "__version__")  # 确认导入的不是本地目录
except (ImportError, AssertionError):
    mlflow = None


def sanitize_dict(x: dict) -> dict:
    """清理字典键，移除括号并将值转换为浮点数。"""
    return {k.replace("(", "").replace(")", ""): float(v) for k, v in x.items()}


def on_pretrain_routine_end(trainer):
    """在预训练流程结束时将训练参数记录到 MLflow。

    此函数根据环境变量和训练器参数设置 MLflow 日志记录，包括跟踪 URI、实验名称和运行名称；
    如果尚未存在活动运行，则启动 MLflow 运行，最后记录训练器参数。

    参数：
        trainer (ultralytics.engine.trainer.BaseTrainer): 包含待记录参数和配置的训练对象。

    注意：
        MLFLOW_TRACKING_URI: MLflow 跟踪 URI。未设置时默认为 'runs/mlflow'。
        MLFLOW_EXPERIMENT_NAME: MLflow 实验名称。未设置时默认为 trainer.args.project。
        MLFLOW_RUN: MLflow 运行名称。未设置时默认为 trainer.args.name。
        MLFLOW_KEEP_RUN_ACTIVE: 训练结束后是否保持 MLflow 运行处于活动状态。真值
            "1"、"true"、"yes"、"on"、"y"、"t"（不区分大小写）表示 True；其他值均为 False。
    """
    # 在调用时（而不是导入时）解析启用状态，避免测试与训练的执行顺序永久禁用 MLflow：
    # `add_integration_callbacks` 会在首次训练时导入此模块，而首次训练可能未启用 mlflow。
    if not mlflow or SETTINGS["mlflow"] is not True:
        return
    if TESTS_RUNNING and "test_mlflow" not in os.environ.get("PYTEST_CURRENT_TEST", ""):
        return  # 不在无关的 pytest 测试期间记录日志

    uri = os.environ.get("MLFLOW_TRACKING_URI") or str(RUNS_DIR / "mlflow")
    LOGGER.debug(f"{PREFIX} tracking uri: {uri}")

    # 设置实验和运行名称
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME") or trainer.args.project or "/Shared/Ultralytics"
    run_name = os.environ.get("MLFLOW_RUN") or trainer.args.name

    trainer._mlflow_active = False
    trainer._mlflow_started_run = False
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        mlflow.autolog()
        active_run = mlflow.active_run()
        if active_run is None:
            active_run = mlflow.start_run(run_name=run_name)
            trainer._mlflow_started_run = True
        LOGGER.info(f"{PREFIX}logging run_id({active_run.info.run_id}) to {uri}")
        if Path(uri).is_dir():
            LOGGER.info(f"{PREFIX}view at http://127.0.0.1:5000 with 'mlflow server --backend-store-uri {uri}'")
        LOGGER.info(f"{PREFIX}disable with 'yolo settings mlflow=False'")
        mlflow.log_params(dict(trainer.args))
        trainer._mlflow_active = True
    except Exception as e:
        LOGGER.warning(f"{PREFIX}Failed to initialize: {e}")
        LOGGER.warning(f"{PREFIX}Not tracking this run")
        if trainer._mlflow_started_run:
            try:
                mlflow.end_run()
            except Exception:
                pass


def _log_metrics(trainer, metrics):
    """将指标记录到 MLflow；失败时禁用本次运行的跟踪，避免使训练崩溃。"""
    try:
        mlflow.log_metrics(metrics=metrics, step=trainer.epoch)
    except Exception as e:
        LOGGER.warning(f"{PREFIX}metric logging failed, disabling tracking for this run: {e}")
        trainer._mlflow_active = False


def on_train_epoch_end(trainer):
    """在每个训练周期结束时将训练指标记录到 MLflow。"""
    if mlflow and getattr(trainer, "_mlflow_active", False):
        _log_metrics(
            trainer,
            {
                **sanitize_dict(trainer.lr),
                **sanitize_dict(trainer.label_loss_items(trainer.tloss, prefix="train")),
            },
        )


def on_fit_epoch_end(trainer):
    """在每个拟合周期结束时将训练指标记录到 MLflow。"""
    if mlflow and getattr(trainer, "_mlflow_active", False):
        _log_metrics(trainer, sanitize_dict(trainer.metrics))


def on_train_end(trainer):
    """在训练结束时记录模型工件，并关闭此回调打开的任意运行。"""
    if not mlflow:
        return
    if getattr(trainer, "_mlflow_active", False):
        try:
            mlflow.log_artifact(str(trainer.best.parent))  # 记录包含 best.pt 和 last.pt 的 save_dir/权重目录
            for f in trainer.save_dir.glob("*"):  # log 所有 other 文件 in save_dir
                if f.suffix in {".png", ".jpg", ".csv", ".pt", ".yaml"}:
                    mlflow.log_artifact(str(f))
            LOGGER.info(
                f"{PREFIX}results logged to {mlflow.get_tracking_uri()}\n{PREFIX}disable with 'yolo settings mlflow=False'"
            )
        except Exception as e:
            LOGGER.warning(f"{PREFIX}failed to log artifacts: {e}")
    if getattr(trainer, "_mlflow_started_run", False):  # 仅关闭由本回调创建的运行
        if env_bool("MLFLOW_KEEP_RUN_ACTIVE"):
            LOGGER.info(f"{PREFIX}mlflow run still alive, remember to close it using mlflow.end_run()")
        else:
            try:
                mlflow.end_run()
                LOGGER.debug(f"{PREFIX}mlflow run ended")
            except Exception:
                pass


callbacks = (
    {
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if mlflow
    else {}
)

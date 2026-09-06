# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING, colorstr, torch_utils
from ultralytics.utils.torch_utils import smart_inference_mode

try:
    assert not TESTS_RUNNING  # 不记录 pytest 测试
    assert SETTINGS["tensorboard"] is True  # 确认已启用集成
    WRITER = None  # TensorBoard SummaryWriter 实例
    PREFIX = colorstr("TensorBoard: ")

    # 仅在启用 TensorBoard 时需要导入以下模块
    from copy import deepcopy

    import torch
    from torch.utils.tensorboard import SummaryWriter

except (ImportError, AssertionError, TypeError, AttributeError):
    # 在 Windows 中处理“Descriptors cannot not be created directly.”protobuf 错误导致的 TypeError。
    # 如果未安装 tensorflow，则会出现 AttributeError：模块“tensorflow”没有属性“io”。
    SummaryWriter = None


def _log_scalars(scalars: dict, step: int = 0) -> None:
    """将标量值记录到 TensorBoard。

    参数：
        scalars (dict): 要记录到 TensorBoard 的标量值字典。键为标量名称，值为对应的标量值。
        step (int): 与标量值一起记录的全局步数，用作 TensorBoard 图表的 x 轴。

    示例：
        记录训练指标
        >>> metrics = {"loss": 0.5, "accuracy": 0.95}
        >>> _log_scalars(metrics, step=100)
    """
    if WRITER:
        for k, v in scalars.items():
            WRITER.add_scalar(k, v, step)


@smart_inference_mode()
def _log_tensorboard_graph(trainer) -> None:
    """将模型计算图记录到 TensorBoard。

    此函数通过使用虚拟输入张量跟踪模型，在 TensorBoard 中可视化模型结构。
    它首先尝试适用于 YOLO 模型的简单方法；如果失败，则回退到适用于 RTDETR 等需要特殊处理模型的复杂方法。

    参数：
        trainer (ultralytics.engine.trainer.BaseTrainer): 包含待可视化模型的训练器对象，必须具有 model 属性以及包含 imgsz 的 args 属性。

    注意：
        此函数要求启用 TensorBoard 集成并初始化全局 WRITER。
        此函数会处理 PyTorch JIT 跟踪器可能产生的警告，并尝试兼容不同的模型架构。
    """
    # 输入图像
    imgsz = trainer.args.imgsz
    ch = trainer.data.get("channels", 3)
    imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    p = next(trainer.model.parameters())  # for device, type
    im = torch.zeros((1, ch, *imgsz), device=p.device, dtype=p.dtype)  # 输入 图像 (must be zeros, not empty)

    # 先尝试简单方法（YOLO）。
    try:
        trainer.model.eval()  # place in .eval() mode to avoid BatchNorm statistics changes
        WRITER.add_graph(torch.jit.trace(torch_utils.unwrap_model(trainer.model), im, strict=False), [])
        LOGGER.info(f"{PREFIX}model graph visualization added ✅")
        return
    except Exception as e1:
        # 回退到 TorchScript 导出步骤（RTDETR）
        try:
            model = deepcopy(torch_utils.unwrap_model(trainer.model))
            model.eval()
            model = model.fuse(verbose=False)
            for m in model.modules():
                if hasattr(m, "export"):  # Detect, RTDETRDecoder (Segment and Pose use Detect base 类别)
                    m.export = True
                    m.format = "torchscript"
            model(im)  # dry run
            WRITER.add_graph(torch.jit.trace(model, im, strict=False), [])
            LOGGER.info(f"{PREFIX}model graph visualization added ✅")
        except Exception as e2:
            LOGGER.warning(f"{PREFIX}TensorBoard graph visualization failure: {e1} -> {e2}")


def on_pretrain_routine_start(trainer) -> None:
    """使用 SummaryWriter 初始化 TensorBoard 日志记录。"""
    if SummaryWriter:
        try:
            global WRITER
            WRITER = SummaryWriter(str(trainer.save_dir))
            LOGGER.info(f"{PREFIX}Start with 'tensorboard --logdir {trainer.save_dir}', view at http://localhost:6006/")
        except Exception as e:
            LOGGER.warning(f"{PREFIX}TensorBoard not initialized correctly, not logging this run. {e}")


def on_train_start(trainer) -> None:
    """记录 TensorBoard 计算图。"""
    if WRITER:
        _log_tensorboard_graph(trainer)


def on_train_epoch_end(trainer) -> None:
    """在训练周期结束时记录标量统计信息。"""
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.epoch + 1)
    _log_scalars(trainer.lr, trainer.epoch + 1)


def on_fit_epoch_end(trainer) -> None:
    """在训练周期结束时记录周期指标。"""
    _log_scalars(trainer.metrics, trainer.epoch + 1)


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_start": on_train_start,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_epoch_end": on_train_epoch_end,
    }
    if SummaryWriter
    else {}
)

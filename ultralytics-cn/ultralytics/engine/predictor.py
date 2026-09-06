# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ yolo mode=predict model=yolo26n.pt source=0                               # webcam
                                                img.jpg                         # image
                                                vid.mp4                         # video
                                                screen                          # screenshot
                                                path/                           # directory
                                                list.txt                        # list of images
                                                list.streams                    # list of streams
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP, TCP stream

Usage - formats:
    $ yolo mode=predict model=yolo26n.pt                 # PyTorch
                              yolo26n.torchscript        # TorchScript
                              yolo26n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                              yolo26n_openvino_model     # OpenVINO
                              yolo26n.engine             # TensorRT
                              yolo26n.mlpackage          # CoreML (macOS-only)
                              yolo26n_saved_model        # TensorFlow SavedModel
                              yolo26n.pb                 # TensorFlow GraphDef
                              yolo26n_edgetpu.tflite     # TensorFlow Edge TPU
                              yolo26n_paddle_model       # PaddlePaddle
                              yolo26n.mnn                # MNN
                              yolo26n_ncnn_model         # NCNN
                              yolo26n_imx_model          # Sony IMX
                              yolo26n_rknn_model         # Rockchip RKNN
                              yolo26n_executorch_model   # PyTorch Executorch
                              yolo26n_axelera_model      # Axelera AI
                              yolo26n_deepx_model        # DEEPX
                              yolo26n_qnn.onnx           # Qualcomm QNN
                              yolo26n.tflite             # LiteRT
                              yolo26n_ascend_model       # Huawei Ascend
"""

from __future__ import annotations

import platform
import re
import threading
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from ultralytics.utils.checks import check_imgsz, check_imshow
from ultralytics.utils.plotting import class_activation_map
from ultralytics.utils.torch_utils import attempt_compile, select_device, smart_inference_mode

STREAM_WARNING = """
Inference results will accumulate in RAM unless `stream=True` is passed, which can cause out-of-memory errors for large
sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict for help.

Example:
    results = model(source=..., stream=True)  # Results 对象生成器
    for r in results:
        boxes = r.boxes  # 边界框输出的 Boxes 对象
        masks = r.masks  # 分割掩码输出的 Masks 对象
        probs = r.probs  # 分类输出的类别概率
"""


class BasePredictor:
    """用于创建预测器的基类。

    此类为预测功能提供基础，在各种输入来源上负责模型设置、推理和结果处理。

    属性：
        args (SimpleNamespace)：预测器配置。
        save_dir (Path)：保存结果的目录。
        done_warmup (bool)：预测器是否已完成预热。
        model (torch.nn.Module)：用于预测的模型。
        data (str | Path | None)：args.data 的副本；AutoBackend 在需要类别名称时回退到该数据集 YAML 文件。
        device (torch.device)：用于预测的设备。
        dataset (Dataset)：用于预测的数据集。
        vid_writer (dict[Path, cv2.VideoWriter])：用于保存视频的 ``{save_path: video_writer}`` 字典。
        plotted_img (np.ndarray)：最近一次绘制的图像。
        source_type (SimpleNamespace)：输入来源的类型。
        seen (int)：已处理的图像数量。
        speed (dict[str, float] | None)：运行完成后按图像统计的预处理、推理和后处理耗时（毫秒）。
        pixels (int | None)：运行完成后按图像统计的平均推理面积（像素数）。
        windows (list[str])：用于可视化的窗口名称列表。
        batch (tuple)：当前批次数据。
        results (list[Any])：当前批次结果。
        transforms (Callable)：用于分类的图像变换。
        callbacks (dict[str, list[Callable]])：不同事件对应的回调函数。
        txt_path (Path)：保存文本结果的路径。
        _lock (threading.Lock)：保证推理线程安全的锁。

    方法：
        preprocess：在推理前准备输入图像。
        inference：对给定图像执行推理。
        postprocess：将原始预测结果处理为结构化结果。
        predict_cli：通过命令行接口执行预测。
        setup_source：设置输入来源和推理模式。
        stream_inference：对输入来源执行流式推理。
        setup_model：初始化并配置模型。
        write_results：将推理结果写入文件。
        save_predicted_images：保存预测可视化结果。
        show：在窗口中显示结果。
        run_callbacks：执行事件对应的已注册回调。
        add_callback：注册新的回调函数。
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
    ):
        """初始化 BasePredictor 基类。

        参数：
            cfg (str | Path | dict | SimpleNamespace)：配置文件路径或配置字典。
            overrides (dict，可选)：要覆盖的配置项。
            _callbacks (dict，可选)：回调函数字典。
        """
        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        if self.args.conf is None:
            self.args.conf = 0.25  # 默认置信度为 0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # 设置完成后即可使用
        self.model = None
        self.data = self.args.data
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_writer = {}  # {保存路径: 视频写入器, ...} 字典
        self.plotted_img = None
        self.source_type = None
        self.seen = 0
        self.speed = None  # 按图像统计的速度，运行完成后设置
        self.pixels = None  # 按图像统计的平均推理面积，运行完成后设置
        self.windows = []
        self.screen = None  # 缓存的屏幕分辨率（宽度、高度），用于 show=True 时的缩放
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        self._lock = threading.Lock()  # 用于保证推理线程安全
        callbacks.add_integration_callbacks(self)

    def preprocess(self, im: torch.Tensor | list[np.ndarray]) -> torch.Tensor:
        """在推理前准备输入图像。

        参数：
            im (torch.Tensor | list[np.ndarray])：输入图像。张量形状为 ``(N, 3, H, W)``，列表形状为 ``[(H, W, 3) x N]``。

        返回：
            (torch.Tensor)：预处理后的图像张量，形状为 ``(N, 3, H, W)``。
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            if im.shape[-1] == 3:
                im = im[..., ::-1]  # 将 BGR 转换为 RGB
            im = im.transpose((0, 3, 1, 2))  # 将 BHWC 转换为 BCHW，即 (n, 3, h, w)
            im = np.ascontiguousarray(im)  # 转换为连续数组
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # 将 uint8 转换为 fp16 或 fp32
        if not_tensor:
            im /= 255  # 将 0～255 归一化到 0.0～1.0
        return im

    def inference(self, im: torch.Tensor, *args, **kwargs):
        """使用指定的模型和参数对给定图像执行推理。"""
        skip = self.source_type.tensor or self.args.augment or self.args.embed  # 激活图不支持这些模式
        if self.args.visualize and getattr(self.model, "base_model", True) and not skip:
            return class_activation_map(
                self.model,
                im,
                self.batch[0],
                self.save_dir,
                *args,
                conf=self.args.conf,
                classes=self.args.classes,
                **kwargs,
            )
        return self.model(im, *args, augment=self.args.augment, embed=self.args.embed, **kwargs)

    def pre_transform(self, im: list[np.ndarray]) -> list[np.ndarray]:
        """在推理前对输入图像执行预变换。

        参数：
            im (list[np.ndarray])：图像列表，形状为 ``[(H, W, 3) x N]``。

        返回：
            (list[np.ndarray])：变换后的图像列表。
        """
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(
            self.imgsz,
            auto=same_shapes
            and self.args.rect
            and (self.model.format == "pt" or (getattr(self.model, "dynamic", False) and self.model.format != "imx")),
            stride=self.model.stride,
        )
        return [letterbox(image=x) for x in im]

    def postprocess(self, preds, img, orig_imgs):
        """对一张图像的预测结果执行后处理并返回结果。"""
        return preds

    def __call__(self, source=None, model=None, stream: bool = False, *args, **kwargs):
        """对图像或数据流执行推理。

        参数：
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor，可选)：
                用于推理的数据来源。
            model (str | Path | torch.nn.Module，可选)：用于推理的模型。
            stream (bool)：是否以流式方式返回推理结果。为 True 时返回生成器。
            *args (Any)：传递给推理方法的其他位置参数。
            **kwargs (Any)：传递给推理方法的其他关键字参数。

        返回：
            (list[ultralytics.engine.results.Results] | generator)：Results 对象列表或 Results 对象生成器。
        """
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(self.stream_inference(source, model, *args, **kwargs))  # 将多个 Results 合并为列表

    def predict_cli(self, source=None, model=None):
        """通过命令行接口（CLI）执行预测。

        此函数用于通过 CLI 执行预测。它会设置输入来源和模型，然后以流式方式处理输入。通过消耗生成器而不
        保存结果，此方法可以确保输出不会在内存中累积。

        参数：
            source (str | Path | 列表[str] | 列表[Path] | 列表[np.ndarray] | np.ndarray | torch.Tensor, 可选):
                用于推理的数据来源。
            model (str | Path | torch.nn.Module，可选)：用于推理的模型。

        注意：
            不要修改此函数或移除生成器。生成器可以防止输出在内存中累积，这对于避免长时间运行预测时出现内存
            问题至关重要。
        """
        gen = self.stream_inference(source, model)
        for _ in gen:  # sourcery skip: remove-empty-nested-block, noqa
            pass

    def setup_source(self, source, stride: int | None = None):
        """设置输入来源和推理模式。

        参数：
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor)：用于推理的数据来源。
            stride (int，可选)：模型步长，用于检查图像尺寸。
        """
        self.imgsz = check_imgsz(self.args.imgsz, stride=stride or self.model.stride, min_dim=2)  # 检查图像尺寸
        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer,
            channels=getattr(self.model, "channels", 3),
        )
        self.source_type = self.dataset.source_type
        if (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000  # many 图像
            or any(getattr(self.dataset, "video_flag", [False]))
        ):  # long sequence
            import torchvision  # noqa (import here triggers torchvision NMS use in nms.py)

            if not getattr(self, "stream", True):  # videos
                LOGGER.warning(STREAM_WARNING)
        self.vid_writer = {}

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """对输入来源执行流式推理，并将结果保存到文件。

        参数：
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor，可选)：
                用于推理的数据来源。
            model (str | Path | torch.nn.Module，可选)：用于推理的模型。
            *args (Any)：传递给推理方法的其他位置参数。
            **kwargs (Any)：传递给推理方法的其他关键字参数。

        Yields:
            (ultralytics.engine.results.Results)：Results 对象。
        """
        if self.args.verbose:
            LOGGER.info("")

        # 设置模型
        if self.model is None:
            self.setup_model(model)
        if not getattr(self.model, "base_model", True) and (
            unsupported := [k for k in ("augment", "embed", "visualize") if getattr(self.args, k)]
        ):
            LOGGER.warning(f"{unsupported} not supported by this model (format='{self.model.format}'), ignoring.")
            self.args.augment, self.args.embed, self.args.visualize = False, None, False

        with self._lock:  # 保证推理线程安全
            # 每次调用预测时都重新设置输入来源
            self.setup_source(source if source is not None else self.args.source)

            # 检查保存目录或标签文件目录是否存在
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            self.seen, self.speed, self.pixels, self.windows, self.batch = 0, None, None, [], None
            px = 0  # 每张图像累加的推理像素数，使混合形状源计算平均值而不是只报告最后一张
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for batch in self.dataset:
                self.batch = batch
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch

                # 预处理
                with profilers[0]:
                    im = self.preprocess(im0s)

                if not self.done_warmup:
                    self.model.warmup(im=im)
                    self.done_warmup = True

                # 推理
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # 返回嵌入张量
                        continue

                # 后处理
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)
                self.run_callbacks("on_predict_postprocess_end")

                # 可视化、保存并写入结果
                n = len(im0s)
                try:
                    for i in range(n):
                        self.seen += 1
                        px += im.shape[2] * im.shape[3]
                        self.results[i].speed = {
                            "preprocess": profilers[0].dt * 1e3 / n,
                            "inference": profilers[1].dt * 1e3 / n,
                            "postprocess": profilers[2].dt * 1e3 / n,
                        }
                        if (
                            self.args.verbose
                            or self.args.save
                            or self.args.save_txt
                            or self.args.save_crop
                            or self.args.show
                        ):
                            s[i] += self.write_results(i, Path(paths[i]), im, s)
                except StopIteration:
                    break

                # 输出批次结果
                if self.args.verbose:
                    LOGGER.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results

            # 最终结果。必须在锁内读取：seen 会在每次运行时重置，在锁外读取可能被并发运行影响。
            # px 和 profilers 是当前运行的局部变量，不会被其他运行共享。
            if seen := self.seen:
                t = tuple(x.t / seen * 1e3 for x in profilers)  # 按图像统计的速度
                self.speed = dict(zip(("preprocess", "inference", "postprocess"), t))
                self.pixels = round(px / seen)  # 平均面积，与按图像统计的平均速度对应
                if self.args.verbose:
                    LOGGER.info(
                        f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                        f"{(min(self.args.batch, seen), getattr(self.model, 'channels', 3), *im.shape[2:])}" % t
                    )

        # 释放资源
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        if self.args.show:
            cv2.destroyAllWindows()  # 关闭所有打开的窗口

        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # 标签文件数量
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")

    def setup_model(self, model, verbose: bool = True):
        """使用给定参数初始化 YOLO 模型，并将其设置为评估模式。

        参数：
            model (str | Path | torch.nn.Module)：要加载或使用的模型。
            verbose (bool)：是否输出详细日志。
        """
        if hasattr(model, "end2end"):
            if self.args.end2end is not None:
                model.end2end = self.args.end2end
            if model.end2end:
                # 保持检测头 top-k >= 300，使 NMS 中的 `classes` 过滤在 `max_det` 截断前能看到所有候选框
                model.set_head_attr(max_det=max(self.args.max_det, 300), agnostic_nms=self.args.agnostic_nms)
        self.model = AutoBackend(
            model=model or self.args.model,
            device=select_device(self.args.device, verbose=verbose),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.quantize == 16,
            fuse=True,
            verbose=verbose,
        )

        self.device = self.model.device  # 更新设备
        self.args.quantize = 16 if self.model.fp16 else None  # 记录实际推理精度
        if hasattr(self.model, "imgsz") and not getattr(self.model, "dynamic", False):
            self.args.imgsz = self.model.imgsz  # 复用导出元数据中的图像尺寸
        self.model.eval()
        # channels_last（NHWC）仅适用于 CUDA 原生 PyTorch：在那里无损且适合 Tensor Core；MPS 不适用，CPU 无收益，
        # 并且只有原生 nn.Module 具有可转换的权重。
        channels_last = self.args.channels_last and self.device.type == "cuda" and self.model.format == "pt"
        if self.args.channels_last and not channels_last:
            LOGGER.warning(
                f"'channels_last=True' applies only to native PyTorch models on CUDA, ignoring for "
                f"format='{self.model.format}' on '{self.device.type}'."
            )
        if channels_last:
            self.model.to(memory_format=torch.channels_last)
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)

    def write_results(self, i: int, p: Path, im: torch.Tensor, s: list[str]) -> str:
        """将推理结果写入文件或目录。

        参数：
            i (int)：当前图像在批次中的索引。
            p (Path)：当前图像的路径。
            im (torch.Tensor)：预处理后的图像张量。
            s (list[str])：结果字符串列表。

        返回：
            (str)：包含结果信息的字符串。
        """
        string = ""  # 输出字符串
        if len(im.shape) == 3:
            im = im[None]  # 扩展批次维度
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None  # 无法确定帧号时为 None

        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        string += "{:g}x{:g} ".format(*im.shape[2:])
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()  # 供其他位置使用
        string += f"{result.verbose()}{result.speed['inference']:.1f}ms"

        # 将预测结果绘制到图像上
        if self.args.save or self.args.show:
            self.plotted_img = result.plot(
                line_width=self.args.line_width,
                boxes=self.args.show_boxes,
                conf=self.args.show_conf,
                labels=self.args.show_labels,
            )

        # 保存 结果
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p))
        if self.args.save:
            self.save_predicted_images(self.save_dir / p.name, frame)

        return string

    def save_predicted_images(self, save_path: Path, frame: int = 0):
        """将视频预测结果保存为 mp4/avi 文件，或将图像保存为 jpg 文件。

        参数：
            save_path (Path)：结果保存路径。
            frame (int)：视频模式下的帧编号。
        """
        im = self.plotted_img

        # 保存视频和流媒体
        if self.dataset.mode in {"stream", "video"}:
            fps = self.dataset.fps if self.dataset.mode == "video" else 30
            frames_path = self.save_dir / f"{save_path.stem}_frames"  # 将帧保存到单独的目录
            if save_path not in self.vid_writer:  # 新视频
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=fps,  # 必须使用整数，浮点数会导致 MP4 编码器报错
                    frameSize=(im.shape[1], im.shape[0]),  # (宽度, 高度)
                )

            # 保存视频
            self.vid_writer[save_path].write(im)
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}/{save_path.stem}_{frame}.jpg", im)

        # 保存图像
        else:
            cv2.imwrite(str(save_path.with_suffix(".jpg")), im)  # 保存为 JPG 以获得最佳兼容性

    def show(self, p: str = ""):
        """在窗口中显示图像。"""
        im = self.plotted_img
        if platform.system() in {"Linux", "Windows"} and p not in self.windows:  # macOS 会自动缩放
            self.windows.append(p)
            name = p.encode("unicode_escape").decode()  # 与修补后的 cv2.imshow 窗口名称保持一致
            cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # 允许调整窗口大小和缩放
            h, w = im.shape[:2]
            try:  # 如果图像大于屏幕分辨率，创建窗口时将尺寸调整到适合屏幕
                if self.screen is None:
                    root = __import__("tkinter").Tk()
                    root.withdraw()  # 隐藏空的 Tk 窗口
                    self.screen = 0.9 * root.winfo_screenwidth(), 0.9 * root.winfo_screenheight()  # 0.9 taskbar margin
                    root.destroy()
                r = min(self.screen[0] / w, self.screen[1] / h, 1.0)
                cv2.resizeWindow(name, max(1, int(w * r)), max(1, int(h * r)))  # (宽度, 高度)
            except Exception:
                cv2.resizeWindow(name, w, h)
        cv2.imshow(p, im)
        if cv2.waitKey(300 if self.dataset.mode == "image" else 1) & 0xFF == ord("q"):  # 图像等待 300 毫秒，否则等待 1 毫秒
            raise StopIteration

    def run_callbacks(self, event: str):
        """运行指定事件对应的所有已注册回调。"""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func: Callable):
        """为指定事件添加回调函数。"""
        self.callbacks[event].append(func)

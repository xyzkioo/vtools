"""UI metadata. Command construction stays in the Python service."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TOOLS: dict[str, dict] = {
    "diagnostics": {
        "title": "检测诊断", "group": "model", "description": "定位漏检、错分类、重复框和定位偏差。",
        "config": "model_diagnostics/config/config.yaml", "entry": "diagnostics",
        "fields": [
            {"key": "resource", "label": "模型权重或预测文件", "kind": "file", "hint": "支持 .pt、.pth、.onnx 或预测文件"},
            {"key": "data", "label": "数据集 YAML", "kind": "file",
             "hint": "此工具会加载模型进行检测诊断；只检查图片和标签请选左侧“数据集质量检查”。"},
            {"key": "device", "label": "运行设备", "kind": "select", "options": ["auto", "cpu", "cuda:0"], "default": "auto"},
        ],
        "modules": [("diagnostics.missed", "漏检与误检"), ("diagnostics.classification", "分类错误"), ("diagnostics.overlap", "重叠分析"), ("output.images", "差图输出")],
        "common": ["mode", "dataset.data", "dataset.split", "ultralytics_model.weights", "benchmark.device", "benchmark.input_size", "diagnostics.score_threshold", "diagnostics.match_iou"],
    },
    "visualization": {
        "title": "模型可视化", "group": "model", "description": "查看特征图、CAM 和检测阶段追踪。",
        "config": "model_visualization/config/config.yaml", "entry": "visualization",
        "fields": [
            {"key": "resource", "label": "模型权重", "kind": "file",
             "hint": "选择模型文件（通常为 .pt、.pth 或 .onnx），不是数据集目录。"},
            {"key": "source", "label": "输入图片或目录", "kind": "file_or_dir",
             "hint": "可选单张图片或图片目录（.jpg、.jpeg、.png、.bmp、.webp、.tif、.tiff）；只放图片即可，不需要 YOLO 标签或 data.yaml。"},
            {"key": "device", "label": "运行设备", "kind": "select", "options": ["auto", "cpu", "cuda:0"], "default": "auto"},
        ],
        "modules": [("visualization.features", "特征图"), ("visualization.cam", "CAM"), ("visualization.stage_trace", "阶段追踪")],
        "common": ["mode", "model.weights", "model.task", "input.source", "input.size", "input.max_images", "layers.preset", "cam.method"],
    },
    "benchmark": {
        "title": "性能测速", "group": "model", "description": "比较 PyTorch、TensorRT 和模型输出一致性。",
        "config": "speed_test/benchmark_config.yaml", "entry": "pytorch",
        "fields": [
            {"key": "backend", "label": "测速类型", "kind": "select", "options": ["pytorch", "tensorrt", "consistency", "all", "checkpoint"], "default": "pytorch"},
            {"key": "resource", "label": "模型权重", "kind": "file",
             "hint": "选择要测速的模型文件（通常为 .pt、.pth 或 .onnx）。"},
            {"key": "source", "label": "测速输入", "kind": "file_or_dir",
             "hint": "可选单张图片、视频或图片目录；目录只需放 .jpg/.jpeg/.png/.bmp/.webp 等图片，不是完整数据集，不需要标签或 data.yaml。"},
            {"key": "device", "label": "运行设备", "kind": "select", "options": ["auto", "cpu", "cuda:0"], "default": "auto"},
        ],
        "modules": [("speed.pytorch_call", "PyTorch 调用"), ("speed.tensorrt_call", "TensorRT"), ("memory.pytorch_peak", "显存统计"), ("consistency.detection", "一致性检查")],
        "common": ["models.0.weights", "models.0.adapter", "models.0.task", "benchmark.device", "benchmark.input_size", "benchmark.batch_size", "benchmark.warmup", "benchmark.repeats"],
    },
    "compression": {
        "title": "模型压缩", "group": "model", "description": "运行基线、量化、剪枝和蒸馏实验。",
        "config": "model_compression/config/config.yaml", "entry": "compression",
        "fields": [
            {"key": "task", "label": "模型任务", "kind": "select", "options": ["detect", "classify"], "default": "detect"},
            {"key": "resource", "label": "模型权重", "kind": "file",
             "hint": "选择模型文件（通常为 .pt、.pth 或 .onnx）。"},
            {"key": "data", "label": "检测数据集 YAML", "kind": "file", "when": {"task": "detect"},
             "hint": "选择完整的 data.yaml；它应指向 train/val 图片路径，并包含 names/classes。图片和对应标签目录由 YAML 定义。"},
            {"key": "train", "label": "训练集目录", "kind": "dir", "when": {"task": "classify"},
             "hint": "选择分类训练集根目录；下面按类别分子目录，例如 train/cat/*.jpg、train/dog/*.jpg。"},
            {"key": "val", "label": "验证集目录", "kind": "dir", "when": {"task": "classify"},
             "hint": "选择分类验证集根目录；下面按类别分子目录，例如 val/cat/*.jpg、val/dog/*.jpg。"},
            {"key": "device", "label": "运行设备", "kind": "select", "options": ["auto", "cpu", "cuda:0"], "default": "auto"},
        ],
        "modules": [("baseline.evaluate", "基线评估"), ("compression.quantize.dynamic_int8", "动态 INT8"), ("compression.prune.unstructured", "非结构化剪枝"), ("distillation.classification", "知识蒸馏"), ("compression.prune.structured", "结构化通道缩放")],
        "common": ["model.weights", "model.task", "dataset.data", "model.device", "compression.sparsity", "compression.structured.method", "compression.structured.target_scale", "compression.structured.finetune_epochs"],
    },
    "transform": {
        "title": "格式转换", "group": "utility", "description": "数据和模型格式转换。",
        "variants": {
            "ndjson": {"title": "NDJSON → YOLO", "script": "transform_tools/ndjson_to_yolo.py", "fields": [("input", "NDJSON 文件", "file", "--input", True), ("output", "输出目录", "dir", "--output", True)]},
            "kmodel": {"title": "PyTorch → K230", "script": "transform_tools/py2kmodel.py", "defaults": {"size": "320", "samples": "200"}, "fields": [("pt", "PyTorch 权重", "file", "--pt", True), ("output", "输出目录", "dir", "--output-dir", False), ("calib", "校准图片目录", "dir", "--calib-dir", True), ("size", "输入尺寸", "number", "--size", False), ("samples", "校准样本数", "number", "--samples", False), ("rebuild", "覆盖已有产物", "bool", "--rebuild", False)]},
            "validate": {"title": "ONNX / kmodel 校验", "script": "transform_tools/PY2KM_validate.py", "defaults": {"size": "320", "normalize": True}, "fields": [("onnx", "ONNX 模型", "file", "--onnx", True), ("kmodel", "kmodel 模型", "file", "--kmodel", True), ("image", "校验图片或目录", "file_or_dir", "--image", True), ("size", "输入尺寸", "number", "--size", True), ("normalize", "输入除以 255", "bool", "--normalize", False)]},
        },
    },
    "data": {
        "title": "数据工具", "group": "utility", "description": "视频抽帧、图片缩放和文件名转换。",
        "variants": {
            "video": {"title": "视频抽帧", "script": "others/video_frame_extractor/video_frame_extractor.py", "defaults": {"every_n": "8", "start": "0", "extension": ".jpg"}, "fields": [("input", "视频文件或目录", "file_or_dir", "--input", True), ("output", "输出目录", "dir", "--output-dir", False), ("every_n", "每隔 N 帧", "number", "--every-n", True), ("start", "起始帧", "number", "--start-frame", False), ("end", "结束帧", "number", "--end-frame", False), ("extension", "输出格式", "select:.jpg|.png", "--extension", False), ("recursive", "递归目录", "bool", "--recursive", False), ("overwrite", "覆盖已有帧", "bool", "--overwrite", False)]},
            "resize": {"title": "图片批量缩放", "script": "others/image_resize/image_resize.py", "defaults": {"width": "1024", "height": "1024", "mode": "fit", "format": "same"}, "fields": [("input", "图片文件或目录", "file_or_dir", "--input", True), ("output", "输出目录", "dir", "--output-dir", False), ("width", "目标宽度", "number", "--width", True), ("height", "目标高度", "number", "--height", True), ("mode", "缩放方式", "select:fit|stretch|crop", "--mode", False), ("format", "输出格式", "select:same|jpg|png|webp", "--output-format", False), ("recursive", "递归目录", "bool", "--recursive", False), ("overwrite", "覆盖已有图片", "bool", "--overwrite", False)]},
            "filename": {"title": "文件名转换", "script": "others/filename_transform/image_filename_converter.py", "defaults": {"dataset": "auto", "template": "{index:04d}_{stem}", "start": "0"}, "fields": [("dir", "数据集目录", "dir", "--dir", True), ("dataset", "数据集格式", "select:auto|images|yolo|coco", "--dataset-format", True), ("labels", "YOLO 标签目录", "dir", "--labels-dir", False), ("coco", "COCO JSON", "file", "--coco-json", False), ("template", "命名模板", "text", "--template", False), ("prefix", "前缀", "text", "--prefix", False), ("suffix", "后缀", "text", "--suffix", False), ("start", "编号起始值", "number", "--start", False), ("digits", "编号位数", "number", "--digits", False), ("plan", "计划 CSV", "file", "--plan", False), ("recursive", "递归子目录", "bool", "--recursive", False), ("apply", "实际执行改名", "bool", "--apply", False)]},
        },
    },
    "dataset_quality": {
        "title": "数据集质量检查", "group": "utility",
        "description": "训练前检查坏图、标签、尺寸、重复图片和数据划分泄漏。",
        "script": "model_diagnostics/check_dataset.py",
        "fields": [
            {"key": "data", "label": "数据集 YAML", "kind": "file", "required": True,
             "hint": "选择包含 train、val、test 和 names 的 Ultralytics data.yaml。"},
            {"key": "output_root", "label": "报告输出目录", "kind": "dir",
             "hint": "留空时写入 model_diagnostics/runs/dataset_quality。"},
            {"key": "sample_count", "label": "每个划分抽样图数量", "kind": "number", "default": 8,
             "hint": "设为 0 可关闭抽样图；问题明细仍会写入 CSV。"},
        ],
    },
    "environment": {
        "title": "环境检查", "group": "system", "description": "检查 Python、PyTorch、CUDA 和本地源码。",
        "script": "env_test/check_install.py",
        "fields": [
            {"key": "source", "label": "源码目录（可选）", "kind": "dir", "default": "", "hint": "留空时自动查找 vtools 同级的 Ultralytics Fork。"},
            {"key": "weights", "label": "模型权重", "kind": "file"},
            {"key": "yaml", "label": "模型结构 YAML（可选）", "kind": "file",
             "hint": "可选：已有 .pt 权重时留空；只有从模型结构创建网络时才选择包含 backbone 和 head 的 YAML。数据集 data.yaml 请不要填在这里。"},
            {"key": "device", "label": "运行设备", "kind": "select", "options": ["auto", "cpu", "cuda:0"], "default": "auto"},
            {"key": "size", "label": "合成输入尺寸", "kind": "number", "default": "64"},
            {"key": "export", "label": "检查 ONNX / TensorRT 导入", "kind": "bool"},
            {"key": "skip_model", "label": "跳过模型构建和前向传播", "kind": "bool"},
        ],
    },
}


def public_catalog() -> dict[str, dict]:
    """Return JSON-safe metadata without server-side script paths."""
    result: dict[str, dict] = {}
    for key, value in TOOLS.items():
        item = {name: content for name, content in value.items() if name not in {"script", "entry"}}
        if "variants" in item:
            item["variants"] = {
                name: {
                    "title": variant["title"],
                    "defaults": variant.get("defaults", {}),
                    "fields": [
                        {"key": field, "label": label, "kind": kind, "required": required}
                        for field, label, kind, _flag, required in variant["fields"]
                    ],
                }
                for name, variant in item["variants"].items()
            }
        result[key] = item
    return result

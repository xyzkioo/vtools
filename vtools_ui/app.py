"""Legacy PySide6 desktop shell for vtools.

The supported entry point is now ``vtools_ui.webapp.server``. This module is
kept as a compatibility surface for existing imports and regression tests; it
is not imported by ``python -m vtools_ui``.

The UI intentionally stays independent from the model/runtime dependencies. It
starts existing vtools entry points in a child process, so a TensorRT or CUDA
failure cannot take down the desktop shell.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import copy
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QSettings, QSize, Signal, QThread, QTimer, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .runner import ToolRunner, ToolSpec, append_history


ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = ROOT / ".vtools_ui" / "history.json"


def _path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


SPECS = [
    ToolSpec("diagnostics", "检测诊断", "定位漏检、错分类、重复框和定位偏差。", "diagnostics", _path("model_diagnostics", "config", "config.yaml")),
    ToolSpec("visualization", "模型可视化", "查看特征图、CAM 和检测阶段追踪。", "visualization", _path("model_visualization", "config", "config.yaml")),
    ToolSpec("benchmark", "性能测速", "比较 PyTorch / TensorRT 的速度与一致性。", "pytorch", _path("speed_test", "benchmark_config.yaml")),
    ToolSpec("compression", "模型压缩", "运行量化、剪枝和蒸馏实验。", "compression", _path("model_compression", "config", "config.yaml")),
    ToolSpec("transform", "格式转换", "进入 ONNX、K230 等模型转换流程。"),
    ToolSpec("data", "数据工具", "视频抽帧、文件名转换和数据集整理。"),
]


STYLE = """
QMainWindow, QWidget { background: #101318; color: #e7ebf2; font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; font-size: 14px; }
QFrame#sidebar { background: #171b22; border-right: 1px solid #2a303b; }
QFrame#topbar { background: #141820; border-bottom: 1px solid #2a303b; }
QFrame#statusbar { background: #171b22; border-top: 1px solid #2a303b; }
QLabel#brand { font-size: 19px; font-weight: 600; color: #f4f7fb; }
QLabel#brandSub { color: #8f9aaa; font-size: 11px; }
QLabel#pageTitle { font-size: 24px; font-weight: 600; color: #f4f7fb; }
QLabel#muted { color: #8f9aaa; }
QLabel#section { color: #d7dee9; font-size: 13px; font-weight: 600; padding-top: 10px; }
QListWidget { background: transparent; border: 0; outline: 0; padding: 8px 8px; }
QListWidget::item { padding: 10px 12px; border-radius: 6px; color: #aeb8c8; }
QListWidget::item:selected { background: #2c65d8; color: white; }
QListWidget::item:hover:!selected { background: #232a35; }
QLineEdit, QComboBox, QPlainTextEdit { background: #1b212b; border: 1px solid #313a47; border-radius: 5px; padding: 0 10px; min-height: 42px; color: #ecf1f8; selection-background-color: #2c65d8; font-size: 14px; }
QLineEdit:focus, QComboBox:focus { border-color: #4d85f5; }
QComboBox QAbstractItemView { background: #1b212b; color: #ecf1f8; selection-background-color: #2c65d8; min-height: 34px; }
QPushButton { background: #242b36; border: 1px solid #364150; border-radius: 5px; padding: 0 14px; min-height: 42px; color: #e7ebf2; font-size: 14px; }
QPushButton:hover { background: #2d3745; }
QPushButton:pressed { background: #1c2430; }
QPushButton#primary { background: #2f6fe4; border-color: #397cf5; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #3c7cf0; }
QPushButton#danger { color: #ffb4b4; border-color: #6d3b42; }
QCheckBox { spacing: 10px; min-height: 30px; font-size: 14px; }
QCheckBox::indicator { width: 20px; height: 20px; border: 1px solid #738198; border-radius: 4px; background: #10151c; }
QCheckBox::indicator:hover { border-color: #8fb4ff; background: #172235; }
QCheckBox::indicator:checked { border-color: #5d95ff; background: #2f6fe4; }
QScrollArea { border: 0; }
QFrame#panel { background: #171c24; border: 1px solid #29313c; border-radius: 7px; }
QFrame#metric { background: #1b222c; border: 1px solid #2d3745; border-radius: 6px; }
QLabel#metricValue { font-size: 22px; font-weight: 600; color: #f4f7fb; }
QLabel#metricName { color: #96a2b2; font-size: 12px; }
QPlainTextEdit#log { background: #0c0f13; border: 0; border-radius: 0; font-family: "Cascadia Mono", "DejaVu Sans Mono", monospace; font-size: 12px; color: #c9d4e4; }
QLabel#preview { background: #0c0f13; border: 1px solid #29313c; border-radius: 5px; }
QProgressBar { background: #252c37; border: 0; border-radius: 3px; text-align: center; color: #dce6f6; height: 6px; }
QProgressBar::chunk { background: #3e7af0; border-radius: 3px; }
QTableWidget { background: #171c24; border: 1px solid #29313c; gridline-color: #29313c; }
QHeaderView::section { background: #202630; color: #aeb8c8; border: 0; padding: 7px; }
QStatusBar { background: #171b22; color: #98a5b6; }
"""


class EmptyPage(QWidget):
    def __init__(self, title: str, description: str) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(34, 30, 34, 30)
        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")
        layout.addWidget(title_label)
        desc = QLabel(description)
        desc.setObjectName("muted")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        layout.addSpacing(24)
        panel = QFrame()
        panel.setObjectName("panel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(24, 22, 24, 22)
        hint = QLabel("这个入口已预留，接入对应脚本后会复用同一套任务配置和日志视图。")
        hint.setObjectName("muted")
        panel_layout.addWidget(hint)
        layout.addWidget(panel)
        layout.addStretch()


class StateCheckBox(QCheckBox):
    """High-contrast checkbox that remains readable on dark Qt styles."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setMinimumHeight(30)

    def paintEvent(self, _event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        size = 20
        top = max(0, (self.height() - size) // 2)
        box = self.rect()
        box.setX(2)
        box.setY(top)
        box.setWidth(size)
        box.setHeight(size)
        painter.setPen(QPen(QColor("#5d6c82"), 1))
        painter.setBrush(QColor("#10151c"))
        painter.drawRoundedRect(box, 4, 4)
        if self.isChecked():
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.setBrush(QColor("#2f6fe4"))
            painter.drawRoundedRect(box, 4, 4)
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, "✓")
        if self.text():
            painter.setPen(QColor("#e7ebf2"))
            text_rect = self.rect().adjusted(size + 10, 0, 0, 0)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text())


class _ResultScanWorker(QObject):
    finished = Signal(object, object)

    def __init__(self, root: Path, project_root: Path, skip_dirs: set[str], suffixes: set[str]) -> None:
        super().__init__()
        self.root = root
        self.project_root = project_root
        self.skip_dirs = skip_dirs
        self.suffixes = suffixes

    def run(self) -> None:
        project_scan = self.root == self.project_root
        # Results are self-contained in the run directories.  ``store`` was
        # used by the old compression registry and is no longer an output
        # location, so do not surface stale registry files in the UI.
        result_markers = {"runs", "runs-profile", "results", "outputs", "reports"}
        files: list[Path] = []
        try:
            for base, dirs, names in os.walk(self.root):
                dirs[:] = [name for name in dirs if name not in self.skip_dirs and not name.startswith(".")]
                for name in names:
                    path = Path(base) / name
                    if project_scan and not any(marker in path.parts for marker in result_markers):
                        continue
                    if path.suffix.lower() in self.suffixes:
                        files.append(path)
                    if len(files) >= 2000:
                        break
                if len(files) >= 2000:
                    break
            files.sort(key=lambda item: item.stat().st_mtime if item.exists() else 0, reverse=True)
            self.finished.emit(self.root, files)
        except OSError as exc:
            self.finished.emit(self.root, exc)


class _CondaScanWorker(QObject):
    finished = Signal(object)

    def run(self) -> None:
        self.finished.emit(MainWindow._conda_environments())


class SettingsPage(QWidget):
    changed = Signal()

    def __init__(self, settings: QSettings, history_path: Path) -> None:
        super().__init__()
        self.settings = settings
        self.history_path = history_path
        self.result_root = QLineEdit(str(settings.value("results_root") or ROOT))
        self.history_limit = QSpinBox()
        self.history_limit.setRange(1, 200)
        self.history_limit.setValue(int(settings.value("history_limit") or 40))
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(34, 28, 34, 20)
        title = QLabel("设置")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        desc = QLabel("设置结果查看目录和任务记录保留数量。修改会在下次扫描或任务结束后生效。")
        desc.setObjectName("muted")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        panel = QFrame()
        panel.setObjectName("panel")
        form = QFormLayout(panel)
        form.setContentsMargins(20, 18, 20, 18)
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(self.result_root, 1)
        choose = QPushButton("选择")
        choose.clicked.connect(self._choose_root)
        row_layout.addWidget(choose)
        form.addRow("默认结果目录", row)
        form.addRow("历史记录数量", self.history_limit)
        layout.addWidget(panel)
        actions = QHBoxLayout()
        clear = QPushButton("清空任务记录")
        clear.clicked.connect(self._clear_history)
        actions.addWidget(clear)
        actions.addStretch()
        save = QPushButton("保存设置")
        save.setObjectName("primary")
        save.clicked.connect(self._save)
        actions.addWidget(save)
        layout.addLayout(actions)
        layout.addStretch()

    def _choose_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择默认结果目录", self.result_root.text() or str(ROOT))
        if path:
            self.result_root.setText(path)

    def _save(self) -> None:
        path = Path(self.result_root.text()).expanduser().resolve() if self.result_root.text().strip() else ROOT
        self.settings.setValue("results_root", str(path))
        self.settings.setValue("history_limit", self.history_limit.value())
        self.changed.emit()
        QMessageBox.information(self, "设置已保存", "默认结果目录和历史记录数量已保存。")

    def _clear_history(self) -> None:
        answer = QMessageBox.question(self, "清空任务记录", "确认删除任务历史记录吗？")
        if answer == QMessageBox.StandardButton.Yes:
            try:
                self.history_path.unlink(missing_ok=True)
            except OSError as exc:
                QMessageBox.warning(self, "清理失败", str(exc))
                return
            self.changed.emit()


class ConfigEditorDialog(QDialog):
    """Graphical editor for common YAML fields with an advanced raw tab."""

    def __init__(self, path: Path | None, tool_key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.path = path
        self.tool_key = tool_key
        self.data: dict[str, object] = {}
        self.field_widgets: dict[str, tuple[QWidget, str]] = {}
        self.all_field_widgets: dict[str, tuple[QWidget, str]] = {}
        self._field_initial: dict[str, object] = {}
        self._all_initial: dict[str, object] = {}
        self.setWindowTitle("图形化配置（常用 / 全部 / YAML 高级）")
        self.resize(840, 680)
        raw_text = ""
        if path and path.exists():
            try:
                raw_text = path.read_text(encoding="utf-8")
            except OSError as exc:
                raw_text = f"# 无法读取配置：{exc}\n"
        self.parse_error: str | None = None
        try:
            self.data = self._parse_yaml(raw_text)
        except ValueError as exc:
            self.data = {}
            self.parse_error = str(exc)

        layout = QVBoxLayout(self)
        location = QLabel(str(path) if path else "尚未选择 YAML 文件；保存时会选择路径")
        location.setObjectName("muted")
        location.setWordWrap(True)
        layout.addWidget(location)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_graphical_tab(), "常用配置")
        self.tabs.addTab(self._build_all_tab(), "全部配置")
        self.editor = QPlainTextEdit(raw_text)
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.editor.setObjectName("log")
        self.tabs.addTab(self.editor, "YAML 高级")
        self.tabs.setCurrentIndex(1)
        self._active_config_tab = 1
        self.tabs.currentChanged.connect(self._config_tab_changed)
        layout.addWidget(self.tabs, 1)

        hint = QLabel("常用配置页适合日常修改；未知字段和自定义模块请在 YAML 高级页编辑。")
        hint.setObjectName("muted")
        layout.addWidget(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        save = buttons.addButton("保存", QDialogButtonBox.ButtonRole.AcceptRole)
        save.setObjectName("primary")
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self._save)
        layout.addWidget(buttons)

    def _parse_yaml(self, text: str) -> dict[str, object]:
        if not text.strip():
            return {}
        try:
            import yaml

            value = yaml.safe_load(text) or {}
            if not isinstance(value, dict):
                raise ValueError("YAML 顶层必须是对象")
            return value
        except Exception as exc:
            raise ValueError(f"YAML 解析失败：{exc}") from exc

    @staticmethod
    def _get(data: dict[str, object], path: str, default: object = None) -> object:
        current: object = data
        parts = path.split(".")
        for index, part in enumerate(parts):
            if isinstance(current, dict):
                remainder = ".".join(parts[index:])
                if remainder in current:
                    return current[remainder]
                current = current.get(part, default)
            elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
                current = current[int(part)]
            else:
                return default
        return current

    @staticmethod
    def _set(data: dict[str, object], path: str, value: object) -> None:
        parts = path.split(".")
        current: object = data
        for index, part in enumerate(parts[:-1]):
            next_part = parts[index + 1]
            if isinstance(current, dict):
                remainder = ".".join(parts[index:])
                if remainder in current:
                    current[remainder] = value
                    return
                if index > 0 and parts[index - 1] == "modules":
                    current[remainder] = value
                    return
                if part not in current or not isinstance(current[part], (dict, list)):
                    current[part] = [] if next_part.isdigit() else {}
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                position = int(part)
                while len(current) <= position:
                    current.append({})
                current = current[position]
        last = parts[-1]
        if isinstance(current, dict):
            current[last] = value
        elif isinstance(current, list) and last.isdigit():
            position = int(last)
            while len(current) <= position:
                current.append(None)
            current[position] = value

    def _display(self, value: object, kind: str, default: object) -> str:
        value = default if value is None else value
        if (kind == "size" or kind.startswith("list")) and isinstance(value, (list, tuple)):
            return ",".join(str(item) for item in value)
        if value is None:
            return ""
        return str(value)

    def _add_field(self, form: QFormLayout, path: str, label: str, kind: str = "text", default: object = "", options: list[str] | None = None) -> None:
        value = self._get(self.data, path, default)
        if kind == "bool":
            widget: QWidget = StateCheckBox()
            widget.setChecked(bool(value))  # type: ignore[attr-defined]
        elif kind == "combo":
            widget = QComboBox()
            widget.addItems(options or [])  # type: ignore[attr-defined]
            if self._display(value, kind, default) not in (options or []):
                widget.setEditable(True)  # type: ignore[attr-defined]
                widget.addItem(self._display(value, kind, default))  # type: ignore[attr-defined]
            widget.setCurrentText(self._display(value, kind, default))  # type: ignore[attr-defined]
        else:
            widget = QLineEdit(self._display(value, kind, default))
        self.field_widgets[path] = (widget, kind)
        self._field_initial[path] = self._field_value(widget, kind)
        form.addRow(label, widget)

    def _add_section(self, layout: QVBoxLayout, text: str) -> None:
        heading = QLabel(text)
        heading.setObjectName("section")
        layout.addWidget(heading)

    def _build_graphical_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(16, 16, 16, 16)
        form = QFormLayout()
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(10)
        layout.addLayout(form)

        if self.tool_key == "diagnostics":
            self._add_section(layout, "运行与数据")
            self._add_field(
                form,
                "mode",
                "运行模式",
                "combo",
                "ultralytics_model",
                ["predictions_file", "ultralytics_model", "custom_adapter"],
            )
            self._add_field(form, "dataset.data", "数据集 YAML", default="")
            self._add_field(form, "dataset.split", "数据集划分", "combo", "val", ["train", "val", "test"])
            self._add_field(form, "predictions_file.predictions", "已有预测文件", default="")
            self._add_field(form, "predictions_file.pred_format", "预测文件格式", "combo", "auto", ["auto", "coco", "ndjson", "yolo"])
            self._add_field(form, "ultralytics_model.weights", "Ultralytics 权重", default="")
            self._add_field(form, "ultralytics_model.task", "Ultralytics 任务", "combo", "detect", ["detect", "segment", "pose", "classify"])
            self._add_field(form, "custom_adapter.weights", "自定义模型权重", default="")
            self._add_field(form, "custom_adapter.adapter", "自定义 Adapter", default="")
            self._add_field(form, "custom_adapter.task", "自定义模型任务", "combo", "detect", ["detect", "segment", "pose", "classify"])
            self._add_section(layout, "推理与诊断")
            self._add_field(form, "benchmark.device", "运行设备", "combo", "auto", ["auto", "cpu", "cuda:0"])
            self._add_field(form, "benchmark.input_size", "输入尺寸", "size", [832, 832])
            self._add_field(form, "benchmark.conf", "候选置信度", "float", 0.001)
            self._add_field(form, "benchmark.iou", "NMS IoU", "float", 0.70)
            self._add_field(form, "diagnostics.score_threshold", "正式阈值", "float", 0.25)
            self._add_field(form, "diagnostics.match_iou", "匹配 IoU", "float", 0.5)
            for path, label, default in (("modules.diagnostics.missed", "漏检", True), ("modules.diagnostics.classification", "分类错误", True), ("modules.diagnostics.overlap", "重叠分析", True), ("modules.output.images", "输出差图", False)):
                self._add_field(form, path, label, "bool", default)
        elif self.tool_key == "visualization":
            self._add_section(layout, "模型与输入")
            self._add_field(form, "mode", "运行模式", "combo", "visualize", ["visualize", "list_layers"])
            self._add_field(form, "model.weights", "模型权重", default="")
            self._add_field(form, "model.task", "模型任务", "combo", "detect", ["detect", "segment", "pose", "classify"])
            self._add_field(form, "model.device", "运行设备", "combo", "auto", ["auto", "cpu", "cuda:0"])
            self._add_field(form, "input.source", "输入图片 / 目录", default="")
            self._add_field(form, "input.size", "输入尺寸", "size", [832, 832])
            self._add_field(form, "input.max_images", "最多图片数", "int", 10)
            self._add_section(layout, "可视化模块")
            for path, label, default in (("modules.visualization.features", "特征图", True), ("modules.visualization.cam", "CAM", False), ("modules.visualization.stage_trace", "阶段追踪", False)):
                self._add_field(form, path, label, "bool", default)
            self._add_field(form, "layers.preset", "层选择预设", "combo", "detect_inputs", ["detect_inputs", "custom"])
            self._add_field(form, "features.channels.count", "特征通道数", "int", 16)
            self._add_field(form, "cam.method", "CAM 方法", "combo", "layercam", ["gradcam", "layercam"])
            self._add_field(form, "detection.conf", "显示置信度", "float", 0.001)
        elif self.tool_key == "benchmark":
            self._add_section(layout, "模型与测速")
            self._add_field(form, "models.0.name", "模型名称", default="model")
            self._add_field(form, "models.0.weights", "模型权重", default="")
            self._add_field(form, "models.0.adapter", "模型适配器", "combo", "ultralytics", ["ultralytics", "checkpoint"])
            self._add_field(form, "models.0.task", "模型任务", "combo", "detect", ["detect", "segment", "pose", "classify"])
            self._add_field(form, "benchmark.device", "运行设备", "combo", "auto", ["auto", "cpu", "cuda:0"])
            self._add_field(form, "benchmark.input_size", "输入尺寸", "size", [832, 832])
            self._add_field(form, "benchmark.batch_size", "Batch", "int", 1)
            self._add_field(form, "benchmark.warmup", "预热次数", "int", 30)
            self._add_field(form, "benchmark.repeats", "重复次数", "int", 100)
            self._add_section(layout, "模块开关")
            for path, label, default in (("modules.speed.pytorch_call", "PyTorch 测速", True), ("modules.speed.tensorrt_call", "TensorRT 测速", True), ("modules.consistency.tensor", "张量一致性", True), ("modules.build.tensorrt", "构建 TensorRT", False)):
                self._add_field(form, path, label, "bool", default)
        elif self.tool_key == "compression":
            self._add_section(layout, "模型与数据")
            self._add_field(form, "model.weights", "模型权重", default="")
            self._add_field(form, "model.task", "模型任务", "combo", "detect", ["detect", "classify"])
            self._add_field(form, "dataset.data", "YOLO 数据集 YAML", default="")
            self._add_field(form, "model.device", "运行设备", "combo", "auto", ["auto", "cpu", "cuda:0"])
            self._add_field(form, "dataset.train", "训练集目录", default="")
            self._add_field(form, "dataset.val", "验证集目录", default="")
            self._add_field(form, "dataset.input_size", "输入尺寸", "size", [224, 224])
            self._add_section(layout, "压缩与蒸馏")
            self._add_field(form, "compression.sparsity", "剪枝稀疏度", "float", 0.30)
            self._add_field(form, "compression.structured.method", "结构化剪枝方法", "combo", "scale", ["scale", "torch_pruning"])
            self._add_field(form, "compression.structured.target_scale", "结构化目标规模", "combo", "n", ["n", "s", "m", "l", "x"])
            self._add_field(form, "compression.structured.initial_weights", "目标规模预训练权重", default="")
            self._add_field(form, "compression.structured.finetune_epochs", "结构化微调轮数", "int", 80)
            self._add_field(form, "compression.structured.train_input_size", "结构化训练尺寸", "int", 1024)
            self._add_field(form, "compression.structured.batch_size", "结构化 Batch", "int", 2)
            self._add_field(form, "compression.structured.benchmark_warmup", "测速预热次数", "int", 5)
            self._add_field(form, "compression.structured.benchmark_repeats", "测速重复次数", "int", 20)
            self._add_field(form, "compression.structured.optimizer", "结构化优化器", "combo", "AdamW", ["AdamW", "SGD", "auto"])
            self._add_field(form, "compression.structured.quality_gate", "精度门禁", "bool", True)
            self._add_field(form, "compression.structured.max_map50_95_drop", "允许 mAP50-95 下降", "float", 0.05)
            self._add_field(form, "compression.structured.channel_sparsity", "通道剪枝比例", "float", 0.10)
            self._add_field(form, "compression.structured.channel_round", "通道对齐数", "int", 8)
            self._add_field(form, "compression.structured.max_pruned_layers", "最多剪枝阶段数", "int", 6)
            self._add_field(form, "compression.structured.stage_selection", "剪枝阶段方向", "combo", "all", ["all", "deepest", "front_to_back"])
            self._add_field(form, "compression.structured.pruning_input_size", "剪枝示例输入尺寸", "int", 640)
            self._add_field(form, "distillation.teacher_weights", "教师模型", default="")
            self._add_field(form, "distillation.student_weights", "学生模型", default="")
            self._add_field(form, "distillation.temperature", "蒸馏温度", "float", 4.0)
            self._add_field(form, "distillation.alpha", "蒸馏权重", "float", 0.5)
            self._add_field(form, "distillation.epochs", "训练轮数", "int", 1)
            for path, label, default in (("modules.baseline.evaluate", "基线评估", True), ("modules.compression.quantize.dynamic_int8", "动态 INT8", False), ("modules.compression.prune.unstructured", "非结构化剪枝", False), ("modules.distillation.classification", "知识蒸馏", False), ("modules.compression.prune.structured", "结构化通道缩放", False)):
                self._add_field(form, path, label, "bool", default)
        else:
            layout.addWidget(QLabel("当前工具没有预置图形字段，请使用“YAML 高级”页。"))
        layout.addStretch()
        scroll.setWidget(body)
        return scroll

    def _flatten_fields(self, value: object, prefix: str = "") -> list[tuple[str, object, str]]:
        rows: list[tuple[str, object, str]] = []
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                rows.extend(self._flatten_fields(child, path))
            return rows
        if isinstance(value, list):
            if not value or all(not isinstance(item, (dict, list)) for item in value):
                if value and all(isinstance(item, bool) for item in value):
                    kind = "list-bool"
                elif value and all(isinstance(item, int) and not isinstance(item, bool) for item in value):
                    kind = "list-int"
                elif value and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
                    kind = "list-float"
                else:
                    kind = "list-text"
                return [(prefix, value, kind)]
            for index, child in enumerate(value):
                rows.extend(self._flatten_fields(child, f"{prefix}.{index}"))
            return rows
        if isinstance(value, bool):
            kind = "bool"
        elif value is None:
            # Keep YAML null values editable without silently turning an empty
            # field into the string "" when the graphical form is saved.
            kind = "null"
        elif isinstance(value, int):
            kind = "int"
        elif isinstance(value, float):
            kind = "float"
        else:
            kind = "text"
        return [(prefix, value, kind)]

    def _build_all_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(16, 16, 16, 16)
        form = QFormLayout()
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(8)
        rows = self._flatten_fields(self.data)
        if not rows:
            layout.addWidget(QLabel("当前配置为空，请先在 YAML 高级页填写内容。"))
        for path, value, kind in rows:
            if kind == "bool":
                widget: QWidget = StateCheckBox()
                widget.setChecked(bool(value))  # type: ignore[attr-defined]
            else:
                widget = QLineEdit(self._display(value, kind, ""))
            self.all_field_widgets[path] = (widget, kind)
            self._all_initial[path] = self._field_value(widget, kind)
            form.addRow(path, widget)
        layout.addLayout(form)
        layout.addStretch()
        scroll.setWidget(body)
        return scroll

    def _field_value(self, widget: QWidget, kind: str) -> object:
        if kind == "bool":
            return bool(widget.isChecked())  # type: ignore[attr-defined]
        value = widget.currentText() if isinstance(widget, QComboBox) else widget.text()  # type: ignore[attr-defined]
        if kind == "null":
            return None if not str(value).strip() else str(value)
        if kind == "int":
            try:
                return int(value)
            except ValueError:
                raise ValueError(f"整数格式错误：{value!r}")
        if kind == "float":
            try:
                return float(value)
            except ValueError:
                raise ValueError(f"小数格式错误：{value!r}")
        if kind == "size":
            parts = [part.strip() for part in str(value).replace("x", ",").split(",") if part.strip()]
            try:
                return [int(parts[0]), int(parts[1] if len(parts) > 1 else parts[0])]
            except (IndexError, ValueError):
                raise ValueError(f"尺寸格式错误：{value!r}，示例：832,832")
        if kind.startswith("list"):
            values = [item.strip() for item in str(value).split(",") if item.strip()]
            if kind == "list-int":
                try:
                    return [int(item) for item in values]
                except ValueError:
                    raise ValueError(f"整数列表格式错误：{value!r}")
            if kind == "list-float":
                try:
                    return [float(item) for item in values]
                except ValueError:
                    raise ValueError(f"小数列表格式错误：{value!r}")
            if kind == "list-bool":
                return [item.lower() in {"1", "true", "yes", "on"} for item in values]
            return values
        return str(value)

    def _commit_config_tab(self) -> None:
        if self._active_config_tab == 2:
            self.data = self._parse_yaml(self.editor.toPlainText())
            self.parse_error = None
            return
        if self.parse_error:
            raise ValueError(self.parse_error)
        widgets = self.field_widgets if self._active_config_tab == 0 else self.all_field_widgets
        initials = self._field_initial if self._active_config_tab == 0 else self._all_initial
        data = copy.deepcopy(self.data)
        for path, (widget, kind) in widgets.items():
            value = self._field_value(widget, kind)
            if value != initials.get(path):
                self._set(data, path, value)
        self.data = data

    def _config_tab_changed(self, index: int) -> None:
        import yaml

        try:
            if not (self.parse_error and index == 2):
                self._commit_config_tab()
        except ValueError as exc:
            self.tabs.blockSignals(True)
            self.tabs.setCurrentIndex(self._active_config_tab)
            self.tabs.blockSignals(False)
            QMessageBox.warning(self, "配置格式错误", str(exc))
            return
        self.tabs.blockSignals(True)
        try:
            if index == 2:
                if not self.parse_error:
                    self.editor.setPlainText(yaml.safe_dump(self.data, allow_unicode=True, sort_keys=False))
            else:
                old = self.tabs.widget(index)
                if index == 0:
                    self.field_widgets.clear()
                    self._field_initial.clear()
                    new = self._build_graphical_tab()
                else:
                    self.all_field_widgets.clear()
                    self._all_initial.clear()
                    new = self._build_all_tab()
                self.tabs.removeTab(index)
                self.tabs.insertTab(index, new, "常用配置" if index == 0 else "全部配置")
                self.tabs.setCurrentIndex(index)
                old.deleteLater()
            self._active_config_tab = index
        finally:
            self.tabs.blockSignals(False)

    def _save(self) -> None:
        path = self.path
        if path is None:
            selected, _ = QFileDialog.getSaveFileName(self, "保存 YAML 配置", str(ROOT), "YAML (*.yaml *.yml)")
            if not selected:
                return
            path = Path(selected).expanduser().resolve()
        try:
            import yaml

            self._commit_config_tab()
            text = yaml.safe_dump(self.data, allow_unicode=True, sort_keys=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(path)
            self.path = path
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", f"无法保存配置：\n{exc}")


class ToolPage(QWidget):
    run_requested = Signal(object, object, object)

    def __init__(self, spec: ToolSpec) -> None:
        super().__init__()
        self.spec = spec
        self.config_edit = QLineEdit(str(spec.default_config) if spec.default_config else "")
        self.config_edit.setPlaceholderText("选择已有 YAML 配置")
        self.resource_edit = QLineEdit()
        resource_hint = {
            "diagnostics": "可选：模型权重或已有预测文件",
            "visualization": "可选：.pt / .onnx 模型权重",
            "benchmark": "可选：测速模型权重",
            "compression": "可选：待压缩模型权重",
        }.get(spec.key, "可选：模型或输入路径")
        self.resource_edit.setPlaceholderText(resource_hint)
        self.dataset_edit = QLineEdit()
        self.dataset_edit.setPlaceholderText("可选：覆盖配置中的数据集 YAML")
        self.compression_task = QComboBox()
        self.compression_task.addItems(["YOLO 目标检测", "图像分类"])
        self.detection_data_edit = QLineEdit()
        self.detection_data_edit.setPlaceholderText("选择包含 names、train、val 的 data.yaml")
        self.val_edit = QLineEdit()
        self.val_edit.setPlaceholderText("可选：验证集目录")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["自动选择", "CPU", "CUDA:0"])
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["PyTorch 测速", "TensorRT 测速", "输出一致性检查", "一键测速", "Checkpoint 检查"])
        self._loading_module_selection = False
        self._module_selection_touched = False
        self._configured_modules: dict[str, bool] = {}
        # Keep form rows stable on both normal and high-DPI desktops.  A
        # QHBoxLayout wrapper can otherwise report a smaller height than its
        # line edit, which clips the text and lets the next row overlap it.
        for control in (
            self.config_edit,
            self.resource_edit,
            self.dataset_edit,
            self.detection_data_edit,
            self.val_edit,
            self.compression_task,
            self.device_combo,
            self.backend_combo,
        ):
            control.setMinimumHeight(56)
        self._build()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(34, 28, 34, 20)
        outer.setSpacing(14)

        title = QLabel(self.spec.title)
        title.setObjectName("pageTitle")
        outer.addWidget(title)
        description = QLabel(self.spec.description)
        description.setObjectName("muted")
        outer.addWidget(description)

        panel = QFrame()
        panel.setObjectName("panel")
        form = QFormLayout(panel)
        form.setContentsMargins(20, 18, 20, 18)
        form.setHorizontalSpacing(22)
        form.setVerticalSpacing(20)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        if self.spec.key == "compression":
            form.addRow("模型任务", self.compression_task)
        if self.spec.key == "benchmark":
            form.addRow("测速类型", self.backend_combo)

        config_row = QWidget()
        config_row.setMinimumHeight(56)
        config_layout = QHBoxLayout(config_row)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.addWidget(self.config_edit, 1)
        browse = QPushButton("选择")
        browse.setMinimumHeight(56)
        browse.clicked.connect(self._choose_config)
        config_layout.addWidget(browse)
        graphical = QPushButton("图形配置")
        graphical.setMinimumHeight(56)
        graphical.setToolTip("打开可编辑全部 YAML 字段的配置窗口")
        graphical.clicked.connect(self._open_config_editor)
        config_layout.addWidget(graphical)
        form.addRow("配置文件", config_row)

        resource_row = QWidget()
        resource_row.setMinimumHeight(56)
        resource_layout = QHBoxLayout(resource_row)
        resource_layout.setContentsMargins(0, 0, 0, 0)
        resource_layout.addWidget(self.resource_edit, 1)
        pick = QPushButton("选择")
        pick.setMinimumHeight(56)
        pick.clicked.connect(self._choose_resource)
        resource_layout.addWidget(pick)
        resource_label = "模型 / 预测文件"
        if self.spec.key in {"visualization", "benchmark", "compression"}:
            resource_label = "模型权重"
        form.addRow(resource_label, resource_row)
        if self.spec.key == "diagnostics":
            dataset_row = QWidget()
            dataset_row.setMinimumHeight(56)
            dataset_layout = QHBoxLayout(dataset_row)
            dataset_layout.setContentsMargins(0, 0, 0, 0)
            dataset_layout.addWidget(self.dataset_edit, 1)
            dataset_pick = QPushButton("选择")
            dataset_pick.setMinimumHeight(56)
            dataset_pick.clicked.connect(self._choose_dataset)
            dataset_layout.addWidget(dataset_pick)
            form.addRow("数据集 YAML", dataset_row)
        elif self.spec.key == "visualization":
            self.dataset_edit.setPlaceholderText("可选：图片、视频或输入目录")
            form.addRow("输入图片 / 目录", self._data_row(self.dataset_edit, "选择输入"))
        elif self.spec.key == "benchmark":
            self.dataset_edit.setPlaceholderText("可选：测速图片、视频或目录")
            form.addRow("测速输入", self._data_row(self.dataset_edit, "选择输入"))
        elif self.spec.key == "compression":
            data_row = QWidget()
            data_row.setMinimumHeight(56)
            data_layout = QHBoxLayout(data_row)
            data_layout.setContentsMargins(0, 0, 0, 0)
            data_layout.addWidget(self.detection_data_edit, 1)
            data_pick = QPushButton("选择 YAML")
            data_pick.setMinimumHeight(56)
            data_pick.clicked.connect(self._choose_detection_data)
            data_layout.addWidget(data_pick)
            form.addRow("检测数据集 YAML", data_row)
            self._compression_form = form
            self._compression_data_row = form.rowCount() - 1
            self.dataset_edit.setPlaceholderText("可选：训练集目录")
            train_row = self._data_row(self.dataset_edit, "选择目录", directory=True)
            val_row = self._data_row(self.val_edit, "选择目录", directory=True)
            form.addRow("训练集目录", train_row)
            self._compression_train_row = form.rowCount() - 1
            form.addRow("验证集目录", val_row)
            self._compression_val_row = form.rowCount() - 1
        form.addRow("运行设备", self.device_combo)
        outer.addWidget(panel)

        advanced_title = QLabel("常用模块")
        advanced_title.setObjectName("section")
        outer.addWidget(advanced_title)
        module_panel = QFrame()
        module_panel.setObjectName("panel")
        module_layout = QGridLayout(module_panel)
        module_layout.setContentsMargins(20, 14, 20, 14)
        labels = self._module_labels()
        self.module_buttons: list[QPushButton] = []
        for index, label in enumerate(labels):
            button = QPushButton(("✓  " if index == 0 else "○  ") + label)
            button.setCheckable(True)
            button.setChecked(index == 0)
            button.toggled.connect(lambda checked, b=button: b.setText(("✓  " if checked else "○  ") + b.text()[3:]))
            button.toggled.connect(self._module_toggled)
            self.module_buttons.append(button)
            module_layout.addWidget(button, index // 2, index % 2)
        outer.addWidget(module_panel)

        note_text = "YAML 仍是完整参数来源；模型、输入和数据集路径会作为本次运行的临时覆盖值传给对应工具。"
        if self.spec.key == "compression":
            note_text += " YOLO 检测支持 data.yaml、多框标签、剪枝及压缩前后 mAP 评估。非结构化剪枝不保证文件缩小或推理加速；结构化通道缩放会按 YAML 规模重建网络并可选微调。"
        note = QLabel(note_text)
        note.setObjectName("muted")
        note.setWordWrap(True)
        outer.addWidget(note)
        outer.addStretch()

        actions = QHBoxLayout()
        actions.addStretch()
        self.run_button = QPushButton("开始运行")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self._run)
        actions.addWidget(self.run_button)
        outer.addLayout(actions)
        if self.spec.key == "compression":
            self.compression_task.currentIndexChanged.connect(self._compression_task_changed)
            self._compression_task_changed()
        self._load_module_defaults()
        self.config_edit.textChanged.connect(lambda _text: self._load_module_defaults())

    def _module_toggled(self, _checked: bool) -> None:
        if not self._loading_module_selection:
            self._module_selection_touched = True

    def _load_module_defaults(self) -> None:
        """Reflect the selected YAML module states in the graphical controls."""
        import yaml

        states: dict[str, bool] = {}
        payload = {}
        path = Path(self.config_edit.text()).expanduser() if self.config_edit.text().strip() else None
        if path and path.exists():
            try:
                import yaml

                payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                configured = payload.get("modules") if isinstance(payload, dict) else None
                if isinstance(configured, dict):
                    states = {str(key): bool(value) for key, value in configured.items()}
            except (OSError, ValueError, TypeError, yaml.YAMLError):
                states = {}
        if isinstance(payload, dict):
            try:
                if self.spec.key == "diagnostics":
                    from model_diagnostics.diagnostics.modules import resolve_modules
                    states = resolve_modules(payload)
                elif self.spec.key == "compression":
                    from model_compression.core.module_selection import resolve_compression_modules
                    states = resolve_compression_modules(payload)
                elif self.spec.key == "benchmark":
                    from speed_test.core.module_selection import resolve_speed_modules
                    states = resolve_speed_modules(payload)
                elif self.spec.key == "visualization" and payload.get("modules") is None:
                    states = {"visualization." + key: bool((payload.get(key) or {}).get("enabled", key == "features"))
                              for key in ("features", "cam", "stage_trace")}
            except (ValueError, TypeError, AttributeError, ImportError):
                states = {}
        self._loading_module_selection = True
        try:
            if self.spec.key == "compression" and isinstance(payload, dict):
                from model_compression.core.detection import is_detection
                self.compression_task.blockSignals(True)
                self.compression_task.setCurrentIndex(0 if is_detection(payload) else 1)
                self.compression_task.blockSignals(False)
                self._compression_task_changed()
            for button, module_id in zip(self.module_buttons, self._module_ids()):
                button.setChecked(states.get(module_id, False))
        finally:
            self._loading_module_selection = False
        self._configured_modules = copy.deepcopy(states)
        self._module_selection_touched = False

    def _choose_detection_data(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 YOLO 数据集", str(ROOT), "YAML (*.yaml *.yml)")
        if path:
            self.detection_data_edit.setText(path)

    def _compression_task_changed(self) -> None:
        detection = self.compression_task.currentIndex() == 0
        self.detection_data_edit.setEnabled(detection)
        self.dataset_edit.setEnabled(not detection)
        self.val_edit.setEnabled(not detection)
        self._compression_form.setRowVisible(self._compression_data_row, detection)
        self._compression_form.setRowVisible(self._compression_train_row, not detection)
        self._compression_form.setRowVisible(self._compression_val_row, not detection)
        for index, button in enumerate(self.module_buttons):
            if index in (1, 3):
                # Dynamic INT8 and distillation still use the classification
                # pipeline; do not leave a disabled module selected.
                button.setEnabled(not detection)
                if detection:
                    button.setChecked(False)
                button.setToolTip("仅支持分类模型" if detection else "")
            elif index == 4:
                # Structured width scaling is the detection-only path.
                button.setEnabled(detection)
                if not detection:
                    button.setChecked(False)
                button.setToolTip("仅支持 YOLO 检测模型" if not detection else "")
            else:
                button.setEnabled(True)
                button.setToolTip("")

    def _module_labels(self) -> list[str]:
        if self.spec.key == "diagnostics":
            return ["漏检与误检", "分类错误", "重叠分析", "差图输出"]
        if self.spec.key == "visualization":
            return ["特征图", "CAM", "阶段追踪"]
        if self.spec.key == "benchmark":
            return ["PyTorch 调用", "TensorRT", "显存统计", "一致性检查"]
        if self.spec.key == "compression":
            return ["基线评估", "动态 INT8", "非结构化剪枝", "知识蒸馏", "结构化通道缩放"]
        return ["默认流程"]

    def _module_ids(self) -> list[str]:
        if self.spec.key == "diagnostics":
            return ["diagnostics.missed", "diagnostics.classification", "diagnostics.overlap", "output.images"]
        if self.spec.key == "visualization":
            return ["visualization.features", "visualization.cam", "visualization.stage_trace"]
        if self.spec.key == "benchmark":
            return ["speed.pytorch_call", "speed.tensorrt_call", "memory.pytorch_peak", "consistency.detection"]
        if self.spec.key == "compression":
            return [
                "baseline.evaluate",
                "compression.quantize.dynamic_int8",
                "compression.prune.unstructured",
                "distillation.classification",
                "compression.prune.structured",
            ]
        return []

    def _choose_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 YAML 配置", str(ROOT), "YAML (*.yaml *.yml)")
        if path:
            self.config_edit.setText(path)
            self._load_module_defaults()

    def _open_config_editor(self) -> None:
        raw = self.config_edit.text().strip()
        path = Path(raw).expanduser().resolve() if raw else None
        dialog = ConfigEditorDialog(path, self.spec.key, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.path:
            self.config_edit.setText(str(dialog.path))
            self._load_module_defaults()

    def _choose_resource(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择模型或数据", str(ROOT), "所有文件 (*)")
        if path:
            self.resource_edit.setText(path)

    def _choose_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择数据集 YAML", str(ROOT), "YAML (*.yaml *.yml)")
        if path:
            self.dataset_edit.setText(path)

    def _data_row(self, edit: QLineEdit, title: str, directory: bool = False) -> QWidget:
        row = QWidget()
        row.setMinimumHeight(56)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        if directory:
            button = QPushButton(title)
            button.setMinimumHeight(56)
            button.clicked.connect(lambda: self._choose_directory(edit, title))
            layout.addWidget(button)
        else:
            file_button = QPushButton("文件")
            file_button.setMinimumHeight(56)
            file_button.clicked.connect(lambda: self._choose_input(edit, title))
            layout.addWidget(file_button)
            dir_button = QPushButton("目录")
            dir_button.setMinimumHeight(56)
            dir_button.clicked.connect(lambda: self._choose_directory(edit, title))
            layout.addWidget(dir_button)
        return row

    def _choose_input(self, edit: QLineEdit, title: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, str(ROOT), "所有文件 (*)")
        if path:
            edit.setText(path)

    def _choose_directory(self, edit: QLineEdit, title: str) -> None:
        path = QFileDialog.getExistingDirectory(self, title, str(ROOT))
        if path:
            edit.setText(path)

    def _save_preset(self) -> None:
        # This button used to claim that a preset was saved while doing
        # nothing. Hide it until the preset format can represent every tool
        # specific option without silently losing YAML fields.
        return

    def _run(self) -> None:
        config = Path(self.config_edit.text()).expanduser() if self.config_edit.text() else None
        selected_spec = self.spec
        extra_args: list[str] = []
        module_ids = self._module_ids()
        module_states = dict(self._configured_modules)
        if self._module_selection_touched or not module_states:
            module_states.update({module_id: button.isChecked() for button, module_id in zip(self.module_buttons, module_ids)})
        selected_modules = [module_id for module_id, enabled in module_states.items() if enabled]
        if self.spec.key != "benchmark" and not selected_modules:
            QMessageBox.warning(self, "未选择模块", "请至少选择一个要运行的模块。")
            return
        if "output.images" in selected_modules and "output.bad_cases" not in selected_modules:
            selected_modules.append("output.bad_cases")
        resource = self.resource_edit.text().strip()
        device = {"自动选择": "auto", "CPU": "cpu", "CUDA:0": "cuda:0"}.get(self.device_combo.currentText(), "auto")
        if device != "auto":
            extra_args.extend(["--device", device])
        if self.spec.key == "diagnostics" and resource:
            # A model checkpoint and an already generated prediction file are
            # different inputs. Keep the CLI flag aligned with the selected
            # file so a .pt path is never treated as JSON predictions.
            flag = "--weights" if Path(resource).suffix.lower() in {".pt", ".pth", ".onnx"} else "--predictions"
            extra_args.extend([flag, resource])
        if self.spec.key == "diagnostics" and self.dataset_edit.text().strip():
            extra_args.extend(["--data", self.dataset_edit.text().strip()])
        elif self.spec.key == "visualization" and resource:
            flag = "--weights" if Path(resource).suffix.lower() in {".pt", ".pth", ".onnx"} else "--source"
            extra_args.extend([flag, resource])
        if self.spec.key == "visualization" and self.dataset_edit.text().strip():
            extra_args.extend(["--source", self.dataset_edit.text().strip()])
        if self.spec.key == "benchmark":
            if resource and self.backend_combo.currentText() in {"PyTorch 测速", "TensorRT 测速", "输出一致性检查"}:
                extra_args.extend(["--weights", resource])
            if self.dataset_edit.text().strip():
                extra_args.extend(["--source", self.dataset_edit.text().strip()])
            if self.backend_combo.currentText() == "一键测速" and resource:
                extra_args.extend(["--model", resource])
        if self.spec.key == "compression":
            detection = self.compression_task.currentIndex() == 0
            extra_args.extend(["--task", "detect" if detection else "classify"])
            if detection and self.detection_data_edit.text().strip():
                extra_args.extend(["--data", self.detection_data_edit.text().strip()])
            if resource:
                extra_args.extend(["--weights", resource])
            if not detection and self.dataset_edit.text().strip():
                extra_args.extend(["--train", self.dataset_edit.text().strip()])
            if not detection and self.val_edit.text().strip():
                extra_args.extend(["--val", self.val_edit.text().strip()])
        if self.spec.key != "benchmark" and self._module_selection_touched:
            for module_id in selected_modules:
                extra_args.extend(["--only", module_id])
        if self.spec.key == "benchmark":
            backend = self.backend_combo.currentText()
            if backend == "PyTorch 测速":
                selected_modules = [item for item in selected_modules if item in {
                    "checkpoint.inspect", "checkpoint.load_check", "model.parameters", "model.flops",
                    "speed.pytorch_call", "speed.pytorch_pipeline", "memory.pytorch_peak",
                }]
            elif backend == "TensorRT 测速":
                selected_modules = [item for item in selected_modules if item in {
                    "checkpoint.inspect", "checkpoint.load_check", "model.parameters", "model.flops",
                    "speed.tensorrt_call", "export.onnx", "build.tensorrt",
                }]
            elif backend == "输出一致性检查":
                selected_modules = [item for item in selected_modules if item.startswith("consistency.")]
            elif backend == "Checkpoint 检查":
                selected_modules = []
            if self._module_selection_touched and backend != "Checkpoint 检查" and not selected_modules:
                QMessageBox.warning(self, "未选择模块", "请为当前测速类型选择至少一个模块。")
                return
            if self._module_selection_touched:
                for module_id in selected_modules:
                    extra_args.extend(["--only", module_id])
            if backend == "TensorRT 测速":
                selected_spec = ToolSpec("benchmark-tensorrt", "TensorRT 测速", self.spec.description, "tensorrt", self.spec.default_config)
            elif backend == "输出一致性检查":
                selected_spec = ToolSpec("benchmark-consistency", "输出一致性检查", self.spec.description, "consistency", self.spec.default_config)
            elif backend == "一键测速":
                selected_spec = ToolSpec("benchmark-all", "一键测速", self.spec.description, script=ROOT / "speed_test" / "run_all.py")
                if config:
                    extra_args.extend(["--config", str(config)])
            elif backend == "Checkpoint 检查":
                selected_spec = ToolSpec("checkpoint", "Checkpoint 检查", self.spec.description, script=ROOT / "speed_test" / "inspect_checkpoint.py")
                if config:
                    extra_args.extend(["--config", str(config)])
                if resource:
                    extra_args.extend(["--model", resource])
        self.run_requested.emit(selected_spec, config, extra_args)

    def set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)


class UtilityPage(QWidget):
    """Form-driven access to the small standalone scripts in the repository."""

    run_requested = Signal(object, object, object)

    def __init__(self, category: str) -> None:
        super().__init__()
        self.category = category
        self.fields: dict[str, dict[str, QWidget]] = {}
        self.specs: dict[str, ToolSpec] = {}
        self._build()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(34, 28, 34, 20)
        title_text = "格式转换" if self.category == "transform" else "数据工具"
        title = QLabel(title_text)
        title.setObjectName("pageTitle")
        outer.addWidget(title)
        description = QLabel("选择一个脚本，填写必要路径后直接运行；所有输出仍由原工具写入。")
        description.setObjectName("muted")
        outer.addWidget(description)
        outer.addSpacing(14)

        self.selector = QComboBox()
        options = self._options()
        self.selector.addItems([label for _key, label in options])
        self.selector.currentIndexChanged.connect(self._switch_form)
        outer.addWidget(self.selector)
        self.stack = QStackedWidget()
        for key, _label in options:
            panel = QFrame()
            panel.setObjectName("panel")
            form = QFormLayout(panel)
            form.setContentsMargins(20, 18, 20, 18)
            form.setHorizontalSpacing(20)
            form.setVerticalSpacing(12)
            self.fields[key] = {}
            self._build_form(key, form)
            self.stack.addWidget(panel)
        outer.addWidget(self.stack)
        note = QLabel("路径可以包含中文和空格；任务会在下方日志区显示完整命令和输出。")
        note.setObjectName("muted")
        note.setWordWrap(True)
        outer.addWidget(note)
        outer.addStretch()
        actions = QHBoxLayout()
        actions.addStretch()
        self.run_button = QPushButton("开始运行")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self._run)
        actions.addWidget(self.run_button)
        outer.addLayout(actions)

    def _options(self) -> list[tuple[str, str]]:
        if self.category == "transform":
            return [("ndjson", "NDJSON → YOLO 数据集"), ("kmodel", "PyTorch → ONNX → K230 .kmodel"), ("validate", "ONNX / .kmodel 一致性校验")]
        return [("video", "视频抽帧"), ("resize", "图片批量缩放"), ("filename", "图片 / YOLO / COCO 文件名转换")]

    def _line(self, form: QFormLayout, key: str, label: str, value: str = "", path_kind: str | None = None) -> QLineEdit:
        edit = QLineEdit(value)
        self.fields[self._current_key][key] = edit
        if path_kind:
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(edit, 1)
            if path_kind == "file_or_dir":
                file_button = QPushButton("文件")
                file_button.clicked.connect(lambda: self._choose_path(edit, "file"))
                layout.addWidget(file_button)
                dir_button = QPushButton("目录")
                dir_button.clicked.connect(lambda: self._choose_path(edit, "dir"))
                layout.addWidget(dir_button)
            else:
                button = QPushButton("选择")
                button.clicked.connect(lambda: self._choose_path(edit, path_kind))
                layout.addWidget(button)
            form.addRow(label, row)
        else:
            form.addRow(label, edit)
        return edit

    def _combo(self, form: QFormLayout, key: str, label: str, values: list[str], current: str = "") -> QComboBox:
        combo = QComboBox()
        combo.addItems(values)
        if current and current not in values:
            combo.setEditable(True)
            combo.addItem(current)
        if current:
            combo.setCurrentText(current)
        self.fields[self._current_key][key] = combo
        form.addRow(label, combo)
        return combo

    def _check(self, form: QFormLayout, key: str, label: str, checked: bool = False) -> QCheckBox:
        check = StateCheckBox(label)
        check.setChecked(checked)
        self.fields[self._current_key][key] = check
        form.addRow("", check)
        return check

    def _build_form(self, key: str, form: QFormLayout) -> None:
        self._current_key = key
        if self.category == "transform" and key == "ndjson":
            self._line(form, "input", "NDJSON 文件", path_kind="file")
            self._line(form, "output", "输出目录", path_kind="dir")
            self.specs[key] = ToolSpec(key, "NDJSON → YOLO", "转换 NDJSON 数据集", script=ROOT / "transform_tools" / "ndjson_to_yolo.py")
        elif self.category == "transform" and key == "kmodel":
            self._line(form, "pt", "PyTorch 权重", path_kind="file")
            self._line(form, "onnx", "ONNX 输出", path_kind="save_file")
            self._line(form, "kmodel", "kmodel 输出", path_kind="save_file")
            self._line(form, "calib", "校准图片目录", path_kind="dir")
            self._line(form, "size", "输入尺寸", "320")
            self._line(form, "samples", "校准样本数", "200")
            self._check(form, "rebuild", "覆盖已有 ONNX / kmodel", False)
            self.specs[key] = ToolSpec(key, "PyTorch → K230 .kmodel", "转换 K230 模型", script=ROOT / "transform_tools" / "py2kmodel.py")
        elif self.category == "transform" and key == "validate":
            self._line(form, "onnx", "ONNX 模型", path_kind="file")
            self._line(form, "kmodel", "kmodel 模型", path_kind="file")
            self._line(form, "image", "测试图片", path_kind="file")
            self._line(form, "size", "输入尺寸", "320")
            self._check(form, "normalize", "ONNX 输入除以 255", True)
            self.specs[key] = ToolSpec(key, "ONNX / .kmodel 校验", "验证转换前后输出", script=ROOT / "transform_tools" / "PY2KM_validate.py")
        elif self.category == "data" and key == "video":
            self._line(form, "input", "视频文件或目录", path_kind="file_or_dir")
            self._line(form, "output", "输出目录", path_kind="dir")
            self._line(form, "every_n", "每隔 N 帧", "8")
            self._line(form, "start", "起始帧", "0")
            self._line(form, "end", "结束帧（可空）")
            self._combo(form, "extension", "输出格式", [".jpg", ".png"], ".jpg")
            self._check(form, "recursive", "递归处理目录", False)
            self._check(form, "overwrite", "覆盖已有帧", False)
            self.specs[key] = ToolSpec(key, "视频抽帧", "将视频保存为图片帧", script=ROOT / "others" / "video_frame_extractor" / "video_frame_extractor.py")
        elif self.category == "data" and key == "resize":
            self._line(form, "input", "图片文件或目录", path_kind="file_or_dir")
            self._line(form, "output", "输出目录", path_kind="dir")
            self._line(form, "width", "目标宽度", "1024")
            self._line(form, "height", "目标高度", "1024")
            self._combo(form, "mode", "缩放方式", ["fit", "stretch", "crop"], "fit")
            self._combo(form, "format", "输出格式", ["same", "jpg", "png", "webp"], "same")
            self._check(form, "recursive", "递归处理目录", False)
            self._check(form, "overwrite", "覆盖已有图片", False)
            self.specs[key] = ToolSpec(key, "图片批量缩放", "批量调整图片分辨率", script=ROOT / "others" / "image_resize" / "image_resize.py")
        elif self.category == "data" and key == "filename":
            self._line(form, "dir", "图片 / 数据集目录", path_kind="dir")
            self._combo(form, "dataset", "数据集格式", ["auto", "images", "yolo", "coco"], "auto")
            self._line(form, "labels", "YOLO 标签目录（可空）", path_kind="dir")
            self._line(form, "coco", "COCO JSON（可空）", path_kind="file")
            self._line(form, "template", "命名模板", "{index:04d}_{stem}")
            self._line(form, "prefix", "前缀")
            self._line(form, "suffix", "后缀")
            self._line(form, "start", "编号起始值", "0")
            self._line(form, "digits", "编号位数（留空时使用模板）")
            self._line(form, "plan", "计划 CSV（可空）", path_kind="file")
            self._check(form, "recursive", "递归处理子目录", False)
            self._check(form, "apply", "实际执行改名（默认仅预览）", False)
            self._check(form, "yes", "跳过执行确认", True)
            self.specs[key] = ToolSpec(key, "文件名转换", "批量修改图片、YOLO 或 COCO 文件名", script=ROOT / "others" / "filename_transform" / "image_filename_converter.py")

    def _switch_form(self, index: int) -> None:
        self.stack.setCurrentIndex(index)

    def _choose_path(self, edit: QLineEdit, kind: str) -> None:
        if kind == "dir":
            path = QFileDialog.getExistingDirectory(self, "选择目录", str(ROOT))
        elif kind == "save_file":
            path, _ = QFileDialog.getSaveFileName(self, "选择输出文件", str(ROOT), "所有文件 (*)")
        elif kind == "file_or_dir":
            path, _ = QFileDialog.getOpenFileName(self, "选择文件", str(ROOT), "所有文件 (*)")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择文件", str(ROOT), "所有文件 (*)")
        if path:
            edit.setText(path)

    def _value(self, key: str, field: str) -> str:
        widget = self.fields[key][field]
        if isinstance(widget, QLineEdit):
            return widget.text().strip()
        if isinstance(widget, QComboBox):
            return widget.currentText().strip()
        return ""

    def _checked(self, key: str, field: str) -> bool:
        return bool(getattr(self.fields[key][field], "isChecked")())

    def _arg(self, args: list[str], flag: str, value: str, *, required: bool = False) -> None:
        if required and not value:
            raise ValueError(f"请填写 {flag} 对应的必填项")
        if value:
            args.extend([flag, value])

    def _build_args(self, key: str) -> list[str]:
        args: list[str] = []
        if key == "ndjson":
            self._arg(args, "--input", self._value(key, "input"), required=True)
            self._arg(args, "--output", self._value(key, "output"), required=True)
        elif key == "kmodel":
            for field, flag in (("pt", "--pt"), ("onnx", "--onnx"), ("kmodel", "--kmodel"), ("calib", "--calib-dir"), ("size", "--size"), ("samples", "--samples")):
                self._arg(args, flag, self._value(key, field), required=field in {"pt", "calib"})
            if self._checked(key, "rebuild"):
                args.append("--rebuild")
        elif key == "validate":
            for field, flag in (("onnx", "--onnx"), ("kmodel", "--kmodel"), ("image", "--image"), ("size", "--size")):
                self._arg(args, flag, self._value(key, field), required=True)
            if not self._checked(key, "normalize"):
                args.append("--no-norm")
        elif key == "video":
            for field, flag in (("input", "--input"), ("output", "--output-dir"), ("every_n", "--every-n"), ("start", "--start-frame"), ("end", "--end-frame"), ("extension", "--extension")):
                self._arg(args, flag, self._value(key, field), required=field in {"input", "every_n"})
            if self._checked(key, "recursive"): args.append("--recursive")
            if self._checked(key, "overwrite"): args.append("--overwrite")
        elif key == "resize":
            for field, flag in (("input", "--input"), ("output", "--output-dir"), ("width", "--width"), ("height", "--height"), ("mode", "--mode"), ("format", "--output-format")):
                self._arg(args, flag, self._value(key, field), required=field in {"input", "width", "height"})
            if self._checked(key, "recursive"): args.append("--recursive")
            if self._checked(key, "overwrite"): args.append("--overwrite")
        elif key == "filename":
            for field, flag in (("dir", "--dir"), ("dataset", "--dataset-format"), ("labels", "--labels-dir"), ("coco", "--coco-json"), ("template", "--template"), ("prefix", "--prefix"), ("suffix", "--suffix"), ("start", "--start"), ("digits", "--digits"), ("plan", "--plan")):
                self._arg(args, flag, self._value(key, field), required=field in {"dir", "dataset"})
            args.append("--recursive" if self._checked(key, "recursive") else "--no-recursive")
            if self._checked(key, "apply"): args.append("--apply")
            else: args.append("--dry-run")
            if self._checked(key, "yes"): args.append("--yes")
        return args

    def _run(self) -> None:
        key = self._options()[self.selector.currentIndex()][0]
        try:
            args = self._build_args(key)
        except ValueError as exc:
            QMessageBox.warning(self, "参数不完整", str(exc))
            return
        if key == "filename" and self._checked(key, "apply") and not self._checked(key, "yes"):
            answer = QMessageBox.question(
                self,
                "确认批量改名",
                "已选择直接修改文件名。确认继续吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            # The child process must not wait for stdin after the GUI has
            # already confirmed the destructive operation.
            args.append("--yes")
        self.run_requested.emit(self.specs[key], None, args)

    def set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)


class EnvironmentPage(QWidget):
    run_requested = Signal(object, object, object)

    def __init__(self) -> None:
        super().__init__()
        self.source = QLineEdit()
        self.source.setPlaceholderText("留空自动查找 vtools 同级的 Ultralytics Fork")
        self.weights = QLineEdit()
        self.yaml = QLineEdit()
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "cuda:0"])
        self.size = QLineEdit("64")
        self.export_check = StateCheckBox("检查 ONNX / TensorRT 导入")
        self.skip_model = StateCheckBox("跳过模型构建和前向传播")
        self._build()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(34, 28, 34, 20)
        title = QLabel("环境检查")
        title.setObjectName("pageTitle")
        outer.addWidget(title)
        desc = QLabel("检查当前 Conda 环境、PyTorch、CUDA 以及本地 Ultralytics 源码是否可用。")
        desc.setObjectName("muted")
        outer.addWidget(desc)
        outer.addSpacing(14)
        panel = QFrame()
        panel.setObjectName("panel")
        form = QFormLayout(panel)
        form.setContentsMargins(20, 18, 20, 18)
        form.addRow("源码目录", self._path_row(self.source, "dir"))
        form.addRow("权重（可选）", self._path_row(self.weights, "file"))
        form.addRow("模型 YAML（可选）", self._path_row(self.yaml, "file"))
        form.addRow("设备", self.device)
        form.addRow("合成输入尺寸", self.size)
        form.addRow("", self.export_check)
        form.addRow("", self.skip_model)
        outer.addWidget(panel)
        outer.addStretch()
        actions = QHBoxLayout()
        actions.addStretch()
        run = QPushButton("开始检查")
        run.setObjectName("primary")
        run.clicked.connect(self._run)
        self.run_button = run
        actions.addWidget(run)
        outer.addLayout(actions)

    def _path_row(self, edit: QLineEdit, kind: str) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        button = QPushButton("选择")
        button.clicked.connect(lambda: self._choose(edit, kind))
        layout.addWidget(button)
        return row

    def _choose(self, edit: QLineEdit, kind: str) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择目录", str(ROOT)) if kind == "dir" else QFileDialog.getOpenFileName(self, "选择文件", str(ROOT), "所有文件 (*)")[0]
        if path:
            edit.setText(path)

    def _run(self) -> None:
        args = ["--device", self.device.currentText(), "--input-size", self.size.text().strip()]
        if self.source.text().strip():
            args[0:0] = ["--source", self.source.text().strip()]
        if self.weights.text().strip(): args.extend(["--weights", self.weights.text().strip()])
        if self.yaml.text().strip(): args.extend(["--yaml", self.yaml.text().strip()])
        if self.export_check.isChecked(): args.append("--check-export")
        if self.skip_model.isChecked(): args.append("--skip-model")
        spec = ToolSpec("environment", "环境检查", "检查 vtools 运行环境", script=ROOT / "env_test" / "check_install.py")
        self.run_requested.emit(spec, None, args)

    def set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)


class ResultsPage(QWidget):
    """Browse result files produced by vtools runs."""

    _IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff", ".svg"}
    _TEXT_SUFFIXES = {".json", ".csv", ".html", ".md", ".txt", ".yaml", ".yml"}
    _MODEL_SUFFIXES = {".pt", ".pth", ".onnx", ".engine", ".kmodel"}
    _PREVIEW_BYTES = 256 * 1024
    _SKIP_DIRS = {".git", ".vtools_ui", "__pycache__"}

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self._image_path: Path | None = None
        self._scan_thread: QThread | None = None
        self._scan_worker: _ResultScanWorker | None = None
        self._build()
        QTimer.singleShot(0, self._scan)

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(34, 28, 34, 20)
        outer.setSpacing(12)

        title = QLabel("结果查看")
        title.setObjectName("pageTitle")
        outer.addWidget(title)
        desc = QLabel("浏览运行目录中的概要、详细数据、图片和模型产物。")
        desc.setObjectName("muted")
        desc.setWordWrap(True)
        outer.addWidget(desc)

        controls = QHBoxLayout()
        self.root_edit = QLineEdit(str(self.root))
        self.root_edit.setPlaceholderText("选择项目结果目录")
        controls.addWidget(self.root_edit, 1)
        choose = QPushButton("选择目录")
        choose.clicked.connect(self._choose_root)
        controls.addWidget(choose)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self._scan)
        controls.addWidget(refresh)
        open_root = QPushButton("打开目录")
        open_root.clicked.connect(self._open_root)
        controls.addWidget(open_root)
        outer.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.file_list = QListWidget()
        # Keep the file browser wide enough to read run-relative paths.  Without
        # an explicit initial size QSplitter may collapse the first pane to its
        # size hint.
        self.file_list.setMinimumWidth(340)
        self.file_list.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.file_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.file_list.setWordWrap(False)
        self.file_list.itemSelectionChanged.connect(self._preview_selected)
        splitter.addWidget(self.file_list)

        preview_panel = QFrame()
        preview_panel.setObjectName("panel")
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        self.preview_title = QLabel("选择一个结果文件")
        self.preview_title.setObjectName("muted")
        self.preview_title.setWordWrap(True)
        preview_layout.addWidget(self.preview_title)
        self.preview_image = QLabel()
        self.preview_image.setObjectName("preview")
        self.preview_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image.setMinimumSize(280, 220)
        self.preview_image.hide()
        preview_layout.addWidget(self.preview_image, 1)
        self.preview_text = QPlainTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.hide()
        preview_layout.addWidget(self.preview_text, 1)
        self.preview_status = QLabel("选择文件查看内容；模型文件可从外部程序打开。")
        self.preview_status.setObjectName("muted")
        self.preview_status.setWordWrap(True)
        self.preview_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.preview_status, 1)
        self.open_file_button = QPushButton("打开所选文件")
        self.open_file_button.setEnabled(False)
        self.open_file_button.clicked.connect(self._open_selected)
        preview_layout.addWidget(self.open_file_button)
        splitter.addWidget(preview_panel)
        splitter.setSizes([380, 820])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, 1)

    def _choose_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择结果目录", self.root_edit.text() or str(ROOT))
        if path:
            self.root_edit.setText(path)
            self._scan()

    def _open_root(self) -> None:
        path = Path(self.root_edit.text()).expanduser().resolve()
        if path.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _scan(self) -> None:
        root = Path(self.root_edit.text()).expanduser().resolve()
        self.file_list.clear()
        self._image_path = None
        self.preview_title.setText("选择一个结果文件")
        self.preview_image.hide()
        self.preview_text.hide()
        self.open_file_button.setEnabled(False)
        self.preview_status.show()
        self.preview_status.setText("正在扫描结果文件…")
        if not root.is_dir():
            self.preview_status.setText(f"目录不存在：{root}")
            return
        if self._scan_thread is not None and self._scan_thread.isRunning():
            return
        self._scan_thread = QThread(self)
        suffixes = self._IMAGE_SUFFIXES | self._TEXT_SUFFIXES | self._MODEL_SUFFIXES
        self._scan_worker = _ResultScanWorker(root, ROOT.resolve(), self._SKIP_DIRS, suffixes)
        self._scan_worker.moveToThread(self._scan_thread)
        self._scan_thread.started.connect(self._scan_worker.run)
        self._scan_worker.finished.connect(self._scan_finished)
        self._scan_worker.finished.connect(self._scan_thread.quit)
        self._scan_worker.finished.connect(self._scan_worker.deleteLater)
        self._scan_thread.finished.connect(self._scan_thread.deleteLater)
        self._scan_thread.finished.connect(self._scan_thread_done)
        self._scan_thread.start()

    def _scan_thread_done(self) -> None:
        self._scan_thread = None
        self._scan_worker = None

    def _scan_finished(self, root: Path, result: object) -> None:
        current_root = Path(self.root_edit.text()).expanduser().resolve()
        if root != current_root:
            return
        if isinstance(result, Exception):
            self.preview_status.setText(f"扫描失败：{result}")
            return
        files = result if isinstance(result, list) else []
        for path in files:
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            relative = path.relative_to(root)
            item = QListWidgetItem(f"{relative}  ({self._format_size(size)})")
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(relative))
            self.file_list.addItem(item)
        self.preview_status.setText(f"找到 {len(files)} 个结果文件。")

    def _preview_selected(self) -> None:
        item = self.file_list.currentItem()
        if not item:
            self._image_path = None
            self.preview_image.hide()
            self.preview_text.hide()
            self.open_file_button.setEnabled(False)
            self.preview_status.show()
            self.preview_status.setText("选择一个文件预览")
            return
        path = Path(str(item.data(Qt.ItemDataRole.UserRole)))
        self.preview_title.setText(str(path))
        self.open_file_button.setEnabled(path.is_file())
        self.preview_image.hide()
        self.preview_text.hide()
        suffix = path.suffix.lower()
        if suffix in self._IMAGE_SUFFIXES:
            self._image_path = path
            self.preview_status.hide()
            self.preview_image.show()
            self._show_image()
            return
        self._image_path = None
        if suffix in self._TEXT_SUFFIXES:
            try:
                with path.open("rb") as source:
                    content = source.read(self._PREVIEW_BYTES + 1)
            except OSError as exc:
                self.preview_status.show()
                self.preview_status.setText(f"无法读取文件：{exc}")
                return
            truncated = len(content) > self._PREVIEW_BYTES
            self.preview_text.setPlainText(content[:self._PREVIEW_BYTES].decode("utf-8", errors="replace"))
            self.preview_text.show()
            self.preview_status.setVisible(truncated)
            if truncated:
                self.preview_status.setText("仅预览前 256 KB；打开文件可查看完整内容。")
            return
        self.preview_status.show()
        self.preview_status.setText("模型产物无法在此预览，可打开文件。")

    def _open_selected(self) -> None:
        item = self.file_list.currentItem()
        if item:
            path = Path(str(item.data(Qt.ItemDataRole.UserRole)))
            if path.is_file():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _show_image(self) -> None:
        if not self._image_path:
            return
        pixmap = QPixmap(str(self._image_path))
        if pixmap.isNull():
            self.preview_status.show()
            self.preview_image.hide()
            self.preview_status.setText("无法读取图片。")
            return
        target = self.preview_image.size()
        self.preview_image.setPixmap(pixmap.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    @staticmethod
    def _format_size(size: int) -> str:
        if size < 1024:
            return f"{size} B"
        if size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size / (1024 * 1024):.1f} MB"

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        if self._image_path:
            self._show_image()


class HomePage(QWidget):
    page_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(34, 28, 34, 20)
        title = QLabel("工作台")
        title.setObjectName("pageTitle")
        outer.addWidget(title)
        desc = QLabel("把常用的模型诊断、可视化和性能检查集中到一个入口。")
        desc.setObjectName("muted")
        outer.addWidget(desc)
        outer.addSpacing(20)

        metrics = QGridLayout()
        for col, (value, label) in enumerate([("6", "工具入口"), ("0", "运行中的任务"), ("—", "最近一次结果")]):
            frame = QFrame()
            frame.setObjectName("metric")
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(16, 14, 16, 14)
            value_label = QLabel(value)
            value_label.setObjectName("metricValue")
            frame_layout.addWidget(value_label)
            label_widget = QLabel(label)
            label_widget.setObjectName("metricName")
            frame_layout.addWidget(label_widget)
            metrics.addWidget(frame, 0, col)
        outer.addLayout(metrics)
        outer.addSpacing(22)

        section = QLabel("快速开始")
        section.setObjectName("section")
        outer.addWidget(section)
        grid = QGridLayout()
        quick = [
            ("diagnostics", "检测诊断", "从预测结果或权重开始排查问题"),
            ("visualization", "模型可视化", "查看特征图和检测阶段"),
            ("benchmark", "性能测速", "快速比较 PyTorch 运行速度"),
            ("compression", "模型压缩", "管理压缩实验和模型版本"),
        ]
        for index, (key, title_text, desc_text) in enumerate(quick):
            card = QPushButton()
            card.setMinimumHeight(86)
            card.setStyleSheet("text-align: left; padding: 14px;")
            card.setText(f"{title_text}\n{desc_text}")
            card.clicked.connect(lambda _checked=False, k=key: self.page_requested.emit(k))
            grid.addWidget(card, index // 2, index % 2)
        outer.addLayout(grid)
        outer.addStretch()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"vtools 工作台 · UI {__version__}")
        self.setMinimumSize(QSize(1080, 680))
        self.resize(1280, 780)
        self.settings = QSettings("vtools", "desktop")
        self.runner = ToolRunner(ROOT, self)
        self.runner.output.connect(self._append_log)
        self.runner.state_changed.connect(self._set_run_state)
        self.runner.finished.connect(self._run_finished)
        self.spec_by_key = {spec.key: spec for spec in SPECS}
        self.pages: dict[str, QWidget] = {}
        self.tool_pages: dict[str, QWidget] = {}
        self._active_config: Path | None = None
        self._build()
        self._restore_geometry()

    def _build(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        topbar = QFrame()
        topbar.setObjectName("topbar")
        top_layout = QHBoxLayout(topbar)
        top_layout.setContentsMargins(22, 12, 22, 12)
        brand = QLabel("vtools")
        brand.setObjectName("brand")
        top_layout.addWidget(brand)
        sub = QLabel(f"视觉模型工具工作台 · UI {__version__}")
        sub.setObjectName("brandSub")
        top_layout.addWidget(sub)
        top_layout.addSpacing(28)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索工具或任务…")
        self.search.setMaximumWidth(330)
        top_layout.addWidget(self.search)
        top_layout.addStretch()
        current_python = Path(sys.executable).resolve()
        self.env_combo = QComboBox()
        self.env_combo.addItem(self._environment_label(current_python), str(current_python))
        self.env_combo.addItem("选择 Python 解释器…")
        self.env_combo.setToolTip(f"任务解释器：{current_python}")
        self.env_combo.setMinimumWidth(220)
        self.env_combo.currentIndexChanged.connect(self._environment_changed)
        top_layout.addWidget(self.env_combo)
        QTimer.singleShot(0, self._scan_environments)
        root_layout.addWidget(topbar)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(210)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(10, 18, 10, 12)
        side_layout.setSpacing(3)
        self.nav = QListWidget()
        self.nav.setIconSize(QSize(16, 16))
        nav_items = [("home", "工作台"), *[(spec.key, spec.title) for spec in SPECS], ("results", "结果查看"), ("history", "任务记录"), ("environment", "环境检查"), ("settings", "设置")]
        for key, text in nav_items:
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.nav.addItem(item)
        self.nav.currentItemChanged.connect(self._navigate)
        side_layout.addWidget(self.nav)
        side_layout.addStretch()
        body.addWidget(sidebar)

        self.stack = QStackedWidget()
        self.pages["home"] = HomePage()
        self.pages["home"].page_requested.connect(self._select_key)  # type: ignore[attr-defined]
        self.stack.addWidget(self.pages["home"])
        for spec in SPECS:
            if spec.tool:
                page = ToolPage(spec)
                page.run_requested.connect(self._start_run)
                self.tool_pages[spec.key] = page
                # Tool pages contain a relatively tall form.  Keep the page
                # at its natural minimum height so a short or high-DPI window
                # scrolls instead of compressing rows and clipping controls.
                scroll = QScrollArea()
                scroll.setWidgetResizable(True)
                scroll.setWidget(page)
                self.pages[spec.key] = scroll
            else:
                page = UtilityPage(spec.key) if spec.key in {"transform", "data"} else EmptyPage(spec.title, spec.description)
                if isinstance(page, UtilityPage):
                    page.run_requested.connect(self._start_run)
                    self.tool_pages[spec.key] = page
                self.pages[spec.key] = page
            self.stack.addWidget(self.pages[spec.key])
        self.pages["history"] = self._make_history_page()
        self.stack.addWidget(self.pages["history"])
        results_root = Path(str(self.settings.value("results_root") or ROOT)).expanduser().resolve()
        self.pages["results"] = ResultsPage(results_root)
        self.stack.addWidget(self.pages["results"])
        environment_page = EnvironmentPage()
        environment_page.run_requested.connect(self._start_run)
        self.pages["environment"] = environment_page
        self.tool_pages["environment"] = environment_page
        self.stack.addWidget(self.pages["environment"])
        self.pages["settings"] = SettingsPage(self.settings, HISTORY_PATH)
        self.pages["settings"].changed.connect(self._settings_changed)  # type: ignore[attr-defined]
        self.stack.addWidget(self.pages["settings"])
        body.addWidget(self.stack, 1)
        root_layout.addLayout(body, 1)

        console = QFrame()
        console.setObjectName("statusbar")
        console_layout = QVBoxLayout(console)
        console_layout.setContentsMargins(16, 8, 16, 8)
        console_layout.setSpacing(6)
        status_line = QHBoxLayout()
        self.state_label = QLabel("就绪")
        self.state_label.setObjectName("muted")
        status_line.addWidget(self.state_label)
        status_line.addStretch()
        console_layout.addLayout(status_line)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        console_layout.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setObjectName("log")
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("运行日志会显示在这里…")
        self.log.setMaximumHeight(130)
        console_layout.addWidget(self.log)
        root_layout.addWidget(console)
        self.setCentralWidget(root)

        self.nav.setCurrentRow(0)
        self.search.textChanged.connect(self._filter_navigation)

    def _make_history_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(34, 28, 34, 20)
        title = QLabel("任务记录")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        desc = QLabel("每次运行的时间、入口和配置都会保留在项目的 .vtools_ui 目录。")
        desc.setObjectName("muted")
        layout.addWidget(desc)
        self.history_table = QTableWidget(0, 4)
        self.history_table.setHorizontalHeaderLabels(["时间", "任务", "状态", "配置"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addSpacing(20)
        layout.addWidget(self.history_table)
        self._reload_history()
        return page

    def _settings_changed(self) -> None:
        results_page = self.pages.get("results")
        if isinstance(results_page, ResultsPage):
            results_page.root_edit.setText(str(self.settings.value("results_root") or ROOT))
            results_page._scan()
        self._reload_history()

    def _reload_history(self) -> None:
        if not hasattr(self, "history_table"):
            return
        self.history_table.setRowCount(0)
        if not HISTORY_PATH.exists():
            return
        try:
            import json

            records = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            records = []
        for record in records:
            row = self.history_table.rowCount()
            self.history_table.insertRow(row)
            for col, key in enumerate(("time", "task", "status", "config")):
                self.history_table.setItem(row, col, QTableWidgetItem(str(record.get(key, ""))))

    def _navigate(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None = None) -> None:
        if not current:
            return
        key = current.data(Qt.ItemDataRole.UserRole)
        self.stack.setCurrentWidget(self.pages[key])

    def _select_key(self, key: str) -> None:
        for row in range(self.nav.count()):
            item = self.nav.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == key:
                self.nav.setCurrentItem(item)
                return

    def _filter_navigation(self, text: str) -> None:
        needle = text.strip().lower()
        for row in range(self.nav.count()):
            item = self.nav.item(row)
            key = str(item.data(Qt.ItemDataRole.UserRole))
            item.setHidden(bool(needle) and needle not in item.text().lower() and needle not in key)

    def _start_run(self, spec: ToolSpec, config: Path | None, extra_args: list[str] | None = None) -> None:
        if self.runner.is_running:
            QMessageBox.warning(self, "任务进行中", "请先等待当前任务完成；关闭窗口会结束当前任务。")
            return
        if config and not config.exists():
            QMessageBox.warning(self, "配置不存在", f"找不到配置文件：\n{config}")
            return
        try:
            self._active_config = config
            self.runner.start(spec, config, extra_args)
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            QMessageBox.critical(self, "无法启动", str(exc))

    def _append_log(self, text: str) -> None:
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(text)
        self.log.ensureCursorVisible()

    def _set_run_state(self, state: str) -> None:
        self.state_label.setText(state)
        running = self.runner.is_running
        self.progress.setVisible(running)
        for page in self.tool_pages.values():
            page.set_running(running)

    def _latest_failure_detail(self) -> str:
        for line in reversed(self.log.toPlainText().splitlines()):
            text = line.strip()
            if "失败：" in text or text.startswith("错误："):
                return text[:600]
        return ""

    def _run_finished(self, exit_code: int, task_name: str, was_stopped: bool = False) -> None:
        history_limit = int(self.settings.value("history_limit") or 40)
        append_history(HISTORY_PATH, task_name, exit_code, self._active_config, "已停止" if was_stopped else None, history_limit)
        run_dir = self.runner.last_run_dir
        self._active_config = None
        self._reload_history()
        status = "已停止" if was_stopped else ("成功" if exit_code == 0 else f"失败（退出码 {exit_code}）")
        self._set_run_state(status)
        if run_dir:
            self._append_log(f"\n结果目录：{run_dir}\n")
            self.state_label.setText(f"{status} · 结果目录：{run_dir}")
        message = status
        if run_dir:
            message += f"\n结果目录：\n{run_dir}"
        else:
            message += "\n未检测到结果目录，请查看运行日志。"
        if exit_code != 0 and not was_stopped:
            detail = self._latest_failure_detail()
            if detail:
                message += f"\n\n{detail}"
        if not was_stopped:
            box = QMessageBox.information if exit_code == 0 else QMessageBox.warning
            box(self, "任务完成" if exit_code == 0 else "任务失败", message)

    def _environment_changed(self, index: int) -> None:
        raw_path = self.env_combo.itemData(index)
        if raw_path:
            executable = Path(str(raw_path)).expanduser().resolve()
            self.runner.set_python_executable(executable)
            self.settings.setValue("python_executable", str(executable))
            self.env_combo.setToolTip(f"任务解释器：{executable}")
            return
        if self.env_combo.itemText(index) == "选择 Python 解释器…":
            path, _ = QFileDialog.getOpenFileName(self, "选择 Python 解释器", str(ROOT), "Python (*)")
            if not path:
                self.env_combo.blockSignals(True)
                self.env_combo.setCurrentIndex(0)
                self.env_combo.blockSignals(False)
                return
            executable = Path(path).expanduser().resolve()
            self.env_combo.insertItem(index, f"自定义：{executable.parent.parent.name}", str(executable))
            self.env_combo.setItemData(index, f"任务解释器：{executable}", Qt.ItemDataRole.ToolTipRole)
            self.env_combo.setCurrentIndex(index)

    def _scan_environments(self) -> None:
        """Discover Conda and common system interpreters without changing the current one."""
        current = Path(sys.executable).resolve()
        if getattr(self, "_env_scan_thread", None) is not None and self._env_scan_thread.isRunning():
            return
        self._env_scan_current = current
        self._env_scan_thread = QThread(self)
        self._env_scan_worker = _CondaScanWorker()
        self._env_scan_worker.moveToThread(self._env_scan_thread)
        self._env_scan_thread.started.connect(self._env_scan_worker.run)
        self._env_scan_worker.finished.connect(lambda conda: self._populate_environments(current, conda))
        self._env_scan_worker.finished.connect(self._env_scan_thread.quit)
        self._env_scan_worker.finished.connect(self._env_scan_worker.deleteLater)
        self._env_scan_thread.finished.connect(self._env_scan_thread.deleteLater)
        self._env_scan_thread.finished.connect(self._env_scan_thread_done)
        self._env_scan_thread.start()

    def _env_scan_thread_done(self) -> None:
        self._env_scan_thread = None
        self._env_scan_worker = None

    def _populate_environments(self, current: Path, conda_values: object) -> None:
        found: list[tuple[str, Path]] = [(self._environment_label(current), current)]
        seen = {str(current)}

        def add(label: str, executable: Path) -> None:
            executable = executable.expanduser().resolve()
            if not executable.is_file() or str(executable) in seen:
                return
            seen.add(str(executable))
            found.append((label, executable))

        values = conda_values if isinstance(conda_values, list) else []
        for env_path, label in values:
            add(label, self._python_in_prefix(env_path))

        for command, label in (("python3", "系统 Python 3"), ("python", "PATH 中的 Python")):
            executable = shutil.which(command)
            if executable:
                add(label, Path(executable))
        for candidate in (Path("/usr/bin/python3"), Path("/usr/local/bin/python3")):
            add("系统 Python 3", candidate)

        selected = str(self.settings.value("python_executable") or self.env_combo.currentData() or current)
        self.env_combo.blockSignals(True)
        self.env_combo.clear()
        selected_index = 0
        for index, (label, executable) in enumerate(found):
            self.env_combo.addItem(label, str(executable))
            self.env_combo.setItemData(index, f"任务解释器：{executable}", Qt.ItemDataRole.ToolTipRole)
            if str(executable) == selected:
                selected_index = index
        self.env_combo.addItem("选择 Python 解释器…")
        self.env_combo.setCurrentIndex(selected_index)
        self.env_combo.blockSignals(False)
        self.env_combo.setToolTip(f"任务解释器：{found[selected_index][1]}")
        self.runner.set_python_executable(found[selected_index][1])

    @staticmethod
    def _python_in_prefix(prefix: Path) -> Path:
        if os.name == "nt":
            return prefix / "python.exe"
        return prefix / "bin" / "python"

    @staticmethod
    def _conda_environments() -> list[tuple[Path, str]]:
        executables: list[str] = []
        env_exe = os.environ.get("CONDA_EXE")
        if env_exe:
            executables.append(env_exe)
        found = shutil.which("conda")
        if found:
            executables.append(found)
        prefix = Path(sys.prefix).resolve()
        roots = [prefix.parent.parent, prefix.parent]
        for root in roots:
            executables.extend([str(root / "bin" / "conda"), str(root / "Scripts" / "conda.exe")])

        result: list[tuple[Path, str]] = []
        seen: set[str] = set()
        for executable in executables:
            if executable in seen or not Path(executable).is_file():
                continue
            seen.add(executable)
            try:
                completed = subprocess.run(
                    [executable, "env", "list", "--json"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=4,
                )
                payload = json.loads(completed.stdout or "{}")
            except (OSError, subprocess.SubprocessError, ValueError):
                continue
            paths = [Path(str(item)) for item in payload.get("envs", []) if item]
            root_prefix = payload.get("root_prefix")
            if root_prefix:
                paths.insert(0, Path(str(root_prefix)))
            details = payload.get("envs_details", {})
            for path in paths:
                detail = details.get(str(path), {}) if isinstance(details, dict) else {}
                name = str(detail.get("name") or path.name)
                if detail.get("base") or (not root_prefix and path.parent.name != "envs"):
                    name = "base"
                result.append((path, f"Conda：{name}"))
            if result:
                break
        return result

    @staticmethod
    def _environment_label(executable: Path) -> str:
        """Show a short environment name while keeping the full path in the tooltip."""

        environment = executable.parent.parent.name if executable.parent.name in {"bin", "Scripts"} else executable.parent.name
        return f"当前环境：{environment}"

    def _restore_geometry(self) -> None:
        geometry = self.settings.value("geometry")
        if geometry and self.restoreGeometry(geometry):
            screen = QApplication.primaryScreen()
            available = screen.availableGeometry() if screen else None
            intersection = available.intersected(self.frameGeometry()) if available else None
            if available and (intersection is None or intersection.width() < 120 or intersection.height() < 80):
                # A monitor layout can change between runs. Keep an old saved
                # window from opening completely outside the current desktop.
                self.move(available.left() + 24, available.top() + 24)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.runner.is_running:
            self.runner.stop()
        for thread in (
            getattr(self, "_env_scan_thread", None),
            getattr(self.pages.get("results"), "_scan_thread", None),
        ):
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait(5000)
        self.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)


def run() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("vtools")
    app.setOrganizationName("vtools")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)
    window = MainWindow()
    window.show()
    window.raise_()
    window.activateWindow()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())

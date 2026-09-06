#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 vtools 中的精简版 Ultralytics 是否安装并能运行。

这个脚本专门解决一个容易误判的问题：电脑上同时存在 pip 版
``ultralytics`` 和仓库里的本地源码时，``import ultralytics`` 到底加载了哪一份。
脚本会在隔离的 Python 子进程中检查真实导入路径，避免当前工作目录或临时的
``sys.path`` 修改把未安装的源码误判为已安装。

常用命令：

    python install_check/check_install.py --source ./ultralytics-cn
    python install_check/check_install.py --source ./ultralytics-cn --weights ./best.pt

退出码为 0 表示必需检查通过；非 0 表示至少有一项必需检查失败。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import textwrap
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_MODULE = "ultralytics"
DEFAULT_DISTRIBUTION = "ultralytics"
KNOWN_VERSION = "8.4.128"
DEFAULT_INPUT_SIZE = 64
PROBE_MARKER = "__VTOOLS_INSTALL_CHECK__"


@dataclass
class CheckEvent:
    status: str
    name: str
    message: str
    detail: str = ""


@dataclass
class Reporter:
    events: List[CheckEvent] = field(default_factory=list)

    def _add(self, status: str, name: str, message: str, detail: str = "") -> None:
        self.events.append(CheckEvent(status, name, message, detail))
        tag = {"PASS": "通过", "FAIL": "失败", "WARN": "警告", "SKIP": "跳过"}.get(status, status)
        print(f"[{tag}] {name}: {message}")
        if detail:
            for line in str(detail).splitlines():
                print(f"       {line}")

    def passed(self, name: str, message: str, detail: str = "") -> None:
        self._add("PASS", name, message, detail)

    def failed(self, name: str, message: str, detail: str = "") -> None:
        self._add("FAIL", name, message, detail)

    def warned(self, name: str, message: str, detail: str = "") -> None:
        self._add("WARN", name, message, detail)

    def skipped(self, name: str, message: str, detail: str = "") -> None:
        self._add("SKIP", name, message, detail)

    @property
    def failures(self) -> int:
        return sum(event.status == "FAIL" for event in self.events)

    @property
    def warnings(self) -> int:
        return sum(event.status == "WARN" for event in self.events)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检验 vtools 的本地 Ultralytics 源码是否真正安装并能够构建/运行模型。"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="源码项目根目录，例如 ./ultralytics-cn；也可填写包含 ultralytics/ 的目录。",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="可选 .pt 权重。提供后会加载权重，并用一张合成图片执行一次预测。",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=None,
        help="可选模型 YAML。默认自动寻找源码/已安装包中的 cfg/models/26/yolo26.yaml。",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="运行设备：auto、cpu、cuda、cuda:0 等。默认 auto，有 CUDA 时优先使用 cuda:0。",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help="合成输入的边长，自动调整为不小于 32 且可被 32 整除；默认 64。",
    )
    parser.add_argument(
        "--module",
        default=DEFAULT_MODULE,
        help="Python 导入名，默认 ultralytics。只有同步修改了源码导入名时才需要改它。",
    )
    parser.add_argument(
        "--distribution",
        default=DEFAULT_DISTRIBUTION,
        help="安装元数据中的 distribution 名，默认 ultralytics；仅改了项目名时可调整。",
    )
    parser.add_argument(
        "--expected-version",
        default=KNOWN_VERSION,
        help=f"期望版本号。默认 {KNOWN_VERSION}；只检查来源时可写 none。",
    )
    parser.add_argument(
        "--check-export",
        action="store_true",
        help="额外检查 ONNX 和 TensorRT Python API；未安装时只给出警告。",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="跳过 YAML 构建和前向传播检查。",
    )
    parser.add_argument(
        "--no-local-check",
        action="store_true",
        help="不要求导入路径来自本地源码，只检查包本身能否导入。",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="把版本不一致、pip check 失败和可选依赖缺失也作为失败处理。",
    )
    return parser.parse_args()


def resolve_path(path: Path, base: Path) -> Path:
    """按调用目录解析路径，避免 PyCharm 的工作目录不同导致误判。"""
    if path.is_absolute():
        return path.expanduser().resolve()
    return (base / path).expanduser().resolve()


def normalise_source_root(path: Path) -> Path:
    """把项目根目录、源码根目录或 ultralytics 包目录统一为项目根目录。"""
    path = path.expanduser().resolve()
    if path.name == "ultralytics" and (path / "__init__.py").is_file():
        return path.parent
    if (path / "ultralytics" / "__init__.py").is_file():
        return path
    return path


def infer_source_root(script_dir: Path) -> Optional[Path]:
    """在 vtools 克隆目录中自动寻找 ultralytics-cn。"""
    repo_root = script_dir.parent
    candidates = (
        repo_root / "ultralytics-cn",
        repo_root / "ultralytics",
        repo_root.parent / "ultralytics-cn",
        repo_root.parent / "ultralytics",
    )
    for candidate in candidates:
        root = normalise_source_root(candidate)
        if (root / "ultralytics" / "__init__.py").is_file():
            return root
    return None


def source_package_file(source_root: Optional[Path], module_name: str) -> Optional[Path]:
    if source_root is None:
        return None
    package_parts = module_name.split(".")
    path = source_root.joinpath(*package_parts, "__init__.py")
    return path if path.is_file() else None


def isolated_environment() -> Dict[str, str]:
    """删除 PYTHONPATH，防止外部路径把未安装源码伪装成安装成功。"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return env


def run_isolated_probe(code: str, timeout: int = 180) -> Tuple[int, str, str]:
    """在临时工作目录运行当前 Python，返回 returncode/stdout/stderr。"""
    with tempfile.TemporaryDirectory(prefix="vtools-install-check-") as temp_dir:
        try:
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=temp_dir,
                env=isolated_environment(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return 124, exc.stdout or "", f"子进程超过 {timeout} 秒未结束"
        except OSError as exc:
            return 125, "", f"无法启动 Python 子进程：{exc}"
    return completed.returncode, completed.stdout, completed.stderr


def extract_probe_json(stdout: str) -> Optional[Dict[str, Any]]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(PROBE_MARKER):
            try:
                return json.loads(line[len(PROBE_MARKER) :])
            except json.JSONDecodeError:
                return None
    return None


def compact_error(stderr: str, stdout: str = "", limit: int = 3000) -> str:
    text = (stderr.strip() or stdout.strip()).strip()
    if len(text) > limit:
        text = text[-limit:]
        text = "（错误信息过长，已显示末尾）\n" + text
    return text or "未返回错误信息"


def format_versions(info: Dict[str, Any]) -> str:
    versions = info.get("versions") or {}
    return ", ".join(f"{key}={value}" for key, value in versions.items() if value) or "未读取到版本"


def run_import_probe(module_name: str, distribution_name: str) -> Tuple[Optional[Dict[str, Any]], str]:
    code = textwrap.dedent(
        f"""
        import importlib
        import importlib.metadata as metadata
        import json
        import platform
        import sys
        import traceback

        result = {{
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "module": {module_name!r},
            "distribution": {distribution_name!r},
            "versions": {{}},
            "imports": {{}},
        }}

        modules = {{
            "torch": "torch",
            "torchvision": "torchvision",
            "numpy": "numpy",
            "pyyaml": "yaml",
            "opencv": "cv2",
            "pillow": "PIL",
        }}
        for label, import_name in modules.items():
            try:
                loaded = importlib.import_module(import_name)
                result["imports"][label] = {{
                    "ok": True,
                    "version": getattr(loaded, "__version__", ""),
                    "file": getattr(loaded, "__file__", ""),
                }}
                if getattr(loaded, "__version__", ""):
                    result["versions"][label] = loaded.__version__
            except Exception as exc:
                result["imports"][label] = {{
                    "ok": False,
                    "error": f"{{type(exc).__name__}}: {{exc}}",
                }}

        try:
            package = importlib.import_module({module_name!r})
            result["module_file"] = getattr(package, "__file__", "")
            result["module_version"] = getattr(package, "__version__", "")
            result["has_yolo"] = hasattr(package, "YOLO")
            try:
                distribution = metadata.distribution({distribution_name!r})
                result["distribution_version"] = distribution.version
                result["distribution_location"] = str(distribution.locate_file(""))
            except Exception as exc:
                result["distribution_error"] = f"{{type(exc).__name__}}: {{exc}}"
            try:
                import torch
                result["cuda_available"] = bool(torch.cuda.is_available())
                result["torch_cuda"] = torch.version.cuda
                if result["cuda_available"]:
                    result["gpu"] = torch.cuda.get_device_name(0)
            except Exception as exc:
                result["cuda_error"] = f"{{type(exc).__name__}}: {{exc}}"
        except Exception:
            result["import_error"] = traceback.format_exc()

        print({PROBE_MARKER!r} + json.dumps(result, ensure_ascii=False))
        """
    )
    returncode, stdout, stderr = run_isolated_probe(code)
    result = extract_probe_json(stdout)
    if result is None:
        return None, compact_error(stderr, stdout)
    if returncode != 0 and not result.get("module_file"):
        return result, compact_error(stderr, stdout)
    return result, compact_error(stderr) if stderr.strip() else ""


def run_export_probe() -> Tuple[Dict[str, Any], str]:
    code = textwrap.dedent(
        f"""
        import importlib
        import json
        import traceback

        result = {{}}
        for label, name in (("onnx", "onnx"), ("tensorrt", "tensorrt")):
            try:
                module = importlib.import_module(name)
                result[label] = {{"ok": True, "version": getattr(module, "__version__", "")}}
            except Exception as exc:
                result[label] = {{"ok": False, "error": f"{{type(exc).__name__}}: {{exc}}"}}
        print({PROBE_MARKER!r} + json.dumps(result, ensure_ascii=False))
        """
    )
    _, stdout, stderr = run_isolated_probe(code, timeout=120)
    return extract_probe_json(stdout) or {}, compact_error(stderr, stdout)


def describe_output(value: Any, depth: int = 0) -> Any:
    """把 Tensor/tuple/dict 输出压缩成可读的 JSON。"""
    if depth > 2:
        return {"type": type(value).__name__}
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        shape = getattr(value, "shape", None)
        try:
            shape = list(shape)
        except Exception:
            shape = str(shape)
        return {"type": type(value).__name__, "shape": shape, "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): describe_output(item, depth + 1) for key, item in list(value.items())[:8]}
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "items": [describe_output(item, depth + 1) for item in list(value)[:4]],
        }
    return {"type": type(value).__name__}


def run_model_probe(
    module_name: str,
    yaml_path: Optional[Path],
    weights_path: Optional[Path],
    device: str,
    input_size: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    yaml_literal = repr(str(yaml_path)) if yaml_path else "None"
    weights_literal = repr(str(weights_path)) if weights_path else "None"
    code = textwrap.dedent(
        f"""
        import importlib
        import json
        import traceback
        from pathlib import Path

        import numpy as np
        import torch

        result = {{"yaml": None, "weights": None}}
        try:
            package = importlib.import_module({module_name!r})
            YOLO = getattr(package, "YOLO")
            yaml_path = {yaml_literal}
            weights_path = {weights_literal}
            device = {device!r}
            input_size = {int(input_size)}

            if yaml_path:
                model = YOLO(yaml_path)
                network = getattr(model, "model", None)
                if network is None:
                    raise RuntimeError("YOLO 对象没有 model 属性")
                network = network.to(device).eval()
                tensor = torch.zeros((1, 3, input_size, input_size), device=device)
                with torch.inference_mode():
                    output = network(tensor)
                result["yaml"] = {{"ok": True, "output": {describe_output.__name__}(output)}}

            if weights_path:
                loaded = YOLO(weights_path)
                image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
                predictions = loaded.predict(
                    source=image,
                    imgsz=input_size,
                    device=device,
                    conf=0.001,
                    iou=0.7,
                    max_det=10,
                    save=False,
                    verbose=False,
                )
                result["weights"] = {{"ok": True, "result_count": len(predictions)}}
        except Exception:
            result["error"] = traceback.format_exc()

        print({PROBE_MARKER!r} + json.dumps(result, ensure_ascii=False, default=str))
        """
    )
    # The child needs this helper definition; insert its source instead of relying on
    # the parent process, which is a separate interpreter.
    helper_source = textwrap.dedent(
        """
        def describe_output(value, depth=0):
            if depth > 2:
                return {"type": type(value).__name__}
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                try:
                    shape = list(value.shape)
                except Exception:
                    shape = str(value.shape)
                return {"type": type(value).__name__, "shape": shape, "dtype": str(value.dtype)}
            if isinstance(value, dict):
                return {str(key): describe_output(item, depth + 1) for key, item in list(value.items())[:8]}
            if isinstance(value, (list, tuple)):
                return {
                    "type": type(value).__name__,
                    "length": len(value),
                    "items": [describe_output(item, depth + 1) for item in list(value)[:4]],
                }
            return {"type": type(value).__name__}
        """
    )
    code = code.replace("result[\"yaml\"] = {\"ok\": True, \"output\": describe_output(output)}", "result[\"yaml\"] = {\"ok\": True, \"output\": describe_output(output)}")
    code = helper_source + "\n" + code
    returncode, stdout, stderr = run_isolated_probe(code, timeout=300)
    result = extract_probe_json(stdout)
    if result is None:
        return None, compact_error(stderr, stdout)
    if returncode != 0 and result.get("error"):
        return result, compact_error(result.get("error", ""), stderr)
    return result, compact_error(stderr) if stderr.strip() else ""


def choose_device(requested: str, cuda_available: bool, reporter: Reporter) -> Optional[str]:
    value = requested.strip().lower()
    if value == "auto":
        chosen = "cuda:0" if cuda_available else "cpu"
        reporter.passed("运行设备", f"自动选择 {chosen}")
        if not cuda_available:
            reporter.warned("CUDA", "未检测到可用 CUDA，将使用 CPU；这不影响基础安装检查。")
        return chosen
    if value in {"cuda", "gpu"}:
        value = "cuda:0"
    if value.startswith("cuda") and not cuda_available:
        reporter.failed("运行设备", f"请求使用 {requested}，但 torch.cuda.is_available() 为 False。")
        return None
    chosen = requested
    reporter.passed("运行设备", f"使用 {chosen}")
    return chosen


def run_pip_check(reporter: Reporter, strict: bool) -> None:
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except Exception as exc:
        reporter.warned("pip check", f"无法执行 pip check：{type(exc).__name__}: {exc}")
        return
    if completed.returncode == 0:
        reporter.passed("pip check", "当前解释器的已安装依赖没有发现冲突。")
        return
    detail = compact_error(completed.stdout, completed.stderr, limit=2000)
    if strict:
        reporter.failed("pip check", "发现依赖冲突（--strict 下视为失败）。", detail)
    else:
        reporter.warned("pip check", "发现依赖冲突；核心模型检查仍会继续。", detail)


def main() -> int:
    args = parse_args()
    reporter = Reporter()
    script_dir = Path(__file__).resolve().parent
    call_dir = Path.cwd().resolve()

    if sys.version_info >= (3, 8):
        reporter.passed("Python", f"{platform.python_version()}，解释器：{sys.executable}")
    else:
        reporter.failed("Python", f"需要 Python >= 3.8，当前为 {platform.python_version()}。")

    input_size = max(32, int(args.input_size))
    if input_size % 32:
        adjusted = ((input_size + 31) // 32) * 32
        reporter.warned("输入尺寸", f"{input_size} 不是 32 的倍数，自动调整为 {adjusted}。")
        input_size = adjusted
    else:
        reporter.passed("输入尺寸", f"合成输入 {input_size}×{input_size}。")

    source_root: Optional[Path]
    source_was_explicit = args.source is not None
    if args.source is not None:
        source_root = normalise_source_root(resolve_path(args.source, call_dir))
    else:
        source_root = infer_source_root(script_dir)

    expected_file = source_package_file(source_root, args.module)
    if source_root is None:
        reporter.warned(
            "本地源码目录",
            "没有自动找到 ultralytics-cn；将只检查当前 Python 中安装的包。",
            "若要强制确认本地源码，请加 --source /path/to/ultralytics-cn。",
        )
    elif expected_file is None:
        reporter.failed(
            "本地源码目录",
            f"目录中找不到 {args.module}/__init__.py：{source_root}",
            "请把 --source 指向包含 pyproject.toml 和包目录的项目根目录。",
        )
    else:
        pyproject = source_root / "pyproject.toml"
        detail = f"项目根目录：{source_root}\n包入口：{expected_file}"
        if pyproject.is_file():
            detail += f"\n安装配置：{pyproject}"
        else:
            detail += "\n警告：该目录没有 pyproject.toml。"
        reporter.passed("本地源码目录", "已找到可检查的源码。", detail)

    info, probe_error = run_import_probe(args.module, args.distribution)
    if info is None:
        reporter.failed("导入包", "隔离子进程无法读取包信息。", probe_error)
        return finish(reporter, args)

    if info.get("module_file"):
        module_file = Path(str(info["module_file"])).expanduser().resolve()
        reporter.passed(
            "导入包",
            f"{args.module} 可导入，版本 {info.get('module_version') or '未知'}。",
            f"实际文件：{module_file}\n当前解释器：{info.get('executable', sys.executable)}",
        )
    else:
        reporter.failed("导入包", f"无法导入 {args.module}。", info.get("import_error", probe_error))
        for label, data in (info.get("imports") or {}).items():
            if data.get("ok"):
                version = f"，版本 {data.get('version')}" if data.get("version") else ""
                reporter.passed(f"依赖/{label}", f"可导入{version}。")
            else:
                reporter.failed(f"依赖/{label}", "无法导入。", data.get("error", ""))
        return finish(reporter, args)

    if info.get("has_yolo"):
        reporter.passed("YOLO 接口", f"检测到 {args.module}.YOLO。")
    else:
        reporter.failed("YOLO 接口", f"{args.module} 已导入，但没有公共 YOLO 类。")

    if info.get("distribution_version"):
        reporter.passed(
            "安装元数据",
            f"找到 distribution {args.distribution}=={info['distribution_version']}。",
            f"安装位置：{info.get('distribution_location', '未知')}",
        )
    else:
        metadata_message = f"没有找到 distribution 元数据：{args.distribution}。"
        metadata_detail = info.get(
            "distribution_error",
            "如果你只把源码放在目录中而没有执行 pip install -e，可按向导安装。",
        )
        if args.strict:
            reporter.failed("安装元数据", metadata_message, metadata_detail)
        else:
            reporter.warned("安装元数据", metadata_message, metadata_detail)

    expected_version = str(args.expected_version).strip()
    actual_version = str(info.get("module_version") or info.get("distribution_version") or "")
    if expected_version.lower() in {"", "none", "skip"}:
        reporter.skipped("版本", "按参数跳过版本号比较。", f"实际版本：{actual_version or '未知'}")
    elif actual_version == expected_version:
        reporter.passed("版本", f"版本符合预期：{actual_version}。")
    else:
        message = f"期望 {expected_version}，实际 {actual_version or '未知'}。"
        if args.strict:
            reporter.failed("版本", message)
        else:
            reporter.warned("版本", message, "如果源码来自更新提交，可用 --expected-version none 跳过比较。")

    if source_root is not None and expected_file is not None and not args.no_local_check:
        actual_file = Path(str(info.get("module_file"))).expanduser().resolve()
        if actual_file == expected_file.resolve():
            reporter.passed("源码来源", "Python 实际加载的是指定的本地源码。", str(actual_file))
        else:
            reporter.failed(
                "源码来源",
                "实际加载的不是指定本地源码，可能仍然导入了 pip 版。",
                f"期望：{expected_file.resolve()}\n实际：{actual_file}",
            )
    elif args.no_local_check:
        reporter.skipped("源码来源", "已通过 --no-local-check 跳过本地路径比较。")

    for label, data in (info.get("imports") or {}).items():
        if data.get("ok"):
            version = f"，版本 {data.get('version')}" if data.get("version") else ""
            reporter.passed(f"依赖/{label}", f"可导入{version}。")
        else:
            detail = data.get("error", "")
            # 这些是普通 PyTorch 运行和本仓库声明的核心依赖，缺失视为失败。
            reporter.failed(f"依赖/{label}", "无法导入。", detail)

    run_pip_check(reporter, args.strict)

    device = choose_device(args.device, bool(info.get("cuda_available")), reporter)
    if info.get("cuda_available"):
        reporter.passed(
            "CUDA",
            f"可用，GPU：{info.get('gpu', '未知')}。",
            f"PyTorch CUDA runtime：{info.get('torch_cuda', '未知')}",
        )
    elif args.device.lower().startswith("cuda"):
        device = None

    if args.check_export:
        export_info, export_error = run_export_probe()
        if not export_info:
            reporter.warned("可选/ONNX/TensorRT", "无法读取可选依赖检查结果。", export_error)
        else:
            for label, data in export_info.items():
                if data.get("ok"):
                    reporter.passed(f"可选/{label}", f"可导入，版本 {data.get('version') or '未知'}。")
                else:
                    message = "未安装或导入失败。"
                    detail = data.get("error", export_error)
                    if args.strict:
                        reporter.failed(f"可选/{label}", message, detail)
                    else:
                        reporter.warned(f"可选/{label}", message, detail)
    else:
        reporter.skipped("ONNX/TensorRT", "未指定 --check-export。")

    yaml_path = resolve_path(args.yaml, call_dir) if args.yaml else None
    if yaml_path is not None and not yaml_path.is_file():
        reporter.failed("模型 YAML", f"文件不存在：{yaml_path}")
        yaml_path = None
    if yaml_path is None and source_root is not None:
        candidate = source_root / "ultralytics" / "cfg" / "models" / "26" / "yolo26.yaml"
        if candidate.is_file():
            yaml_path = candidate
    if yaml_path is None and info.get("module_file"):
        installed_candidate = (
            Path(str(info["module_file"])).resolve().parent
            / "cfg"
            / "models"
            / "26"
            / "yolo26.yaml"
        )
        if installed_candidate.is_file():
            yaml_path = installed_candidate

    weights_path = resolve_path(args.weights, call_dir) if args.weights else None
    if weights_path is not None and not weights_path.is_file():
        reporter.failed("权重文件", f"文件不存在：{weights_path}")
        weights_path = None

    if args.skip_model:
        reporter.skipped("模型构建/前向", "已通过 --skip-model 跳过。")
    elif device is None:
        reporter.skipped("模型构建/前向", "设备不可用，未执行。")
    elif yaml_path is None and weights_path is None:
        reporter.warned(
            "模型构建/前向",
            "没有找到 YAML 或权重，未执行实际模型检查。",
            "在 vtools 根目录运行，或使用 --source/--yaml/--weights 指定文件。",
        )
    else:
        model_info, model_error = run_model_probe(
            args.module,
            yaml_path,
            weights_path,
            device,
            input_size,
        )
        if model_info is None:
            reporter.failed("模型构建/前向", "模型子进程没有返回结果。", model_error)
        elif model_info.get("error"):
            reporter.failed("模型构建/前向", "模型构建或推理失败。", model_info["error"])
        else:
            if yaml_path:
                yaml_result = model_info.get("yaml") or {}
                if yaml_result.get("ok"):
                    reporter.passed(
                        "YAML 构建/前向",
                        "已构建模型并完成一次 Tensor 前向传播。",
                        f"文件：{yaml_path}\n输出摘要：{yaml_result.get('output')}",
                    )
                else:
                    reporter.failed("YAML 构建/前向", "YAML 模型检查没有通过。")
            if weights_path:
                weight_result = model_info.get("weights") or {}
                if weight_result.get("ok"):
                    reporter.passed(
                        "权重加载/预测",
                        "已加载权重并完成一次合成图片预测。",
                        f"文件：{weights_path}\n结果数量：{weight_result.get('result_count')}",
                    )
                else:
                    reporter.failed("权重加载/预测", "权重检查没有通过。")

    return finish(reporter, args)


def finish(reporter: Reporter, args: argparse.Namespace) -> int:
    print("\n" + "=" * 64)
    if reporter.failures:
        print(f"检验结束：失败 {reporter.failures} 项，警告 {reporter.warnings} 项。")
        print("请先处理标记为【失败】的项目，再重新运行本脚本。")
        return 1
    if reporter.warnings:
        print(f"检验结束：核心检查通过，警告 {reporter.warnings} 项。")
    else:
        print("检验结束：全部必需检查通过。")
    print(f"Python：{sys.executable}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断检验。", file=sys.stderr)
        raise SystemExit(130)
    except Exception:
        print("检验脚本自身发生未处理异常：", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(2)

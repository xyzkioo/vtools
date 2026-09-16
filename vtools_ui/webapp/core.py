"""Qt-independent configuration, command and result services."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .catalog import ROOT, TOOLS

STATE_DIR = ROOT / ".vtools_ui"
SETTINGS_PATH = STATE_DIR / "settings.json"
HISTORY_PATH = STATE_DIR / "history.json"
TEXT_SUFFIXES = {".json", ".csv", ".html", ".md", ".txt", ".yaml", ".yml"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff", ".svg"}
MODEL_SUFFIXES = {".pt", ".pth", ".onnx", ".engine", ".kmodel"}
SKIP_DIRS = {".git", ".vtools_ui", "__pycache__", "node_modules"}
PREVIEW_BYTES = 256 * 1024


def project_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_settings() -> dict[str, Any]:
    defaults = {"results_root": str(ROOT), "history_limit": 40, "python_executable": sys.executable}
    if SETTINGS_PATH.exists():
        try:
            loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                defaults.update({key: loaded[key] for key in defaults if key in loaded})
        except (OSError, ValueError):
            pass
    else:
        # One-time read of settings saved by the existing Qt workbench.
        try:
            from PySide6.QtCore import QSettings

            previous = QSettings("vtools", "desktop")
            for key in defaults:
                value = previous.value(key)
                if value is not None:
                    defaults[key] = value
        except ImportError:
            pass
    try:
        defaults["history_limit"] = max(1, min(200, int(defaults["history_limit"])))
    except (TypeError, ValueError):
        defaults["history_limit"] = 40
    return defaults


def save_settings(changes: dict[str, Any]) -> dict[str, Any]:
    settings = read_settings()
    if "results_root" in changes:
        settings["results_root"] = str(project_path(str(changes["results_root"])))
    if "python_executable" in changes:
        candidate = project_path(str(changes["python_executable"]))
        if not candidate.is_file():
            raise ValueError(f"Python 解释器不存在：{candidate}")
        settings["python_executable"] = str(candidate)
    if "history_limit" in changes:
        settings["history_limit"] = max(1, min(200, int(changes["history_limit"])))
    _atomic_text(SETTINGS_PATH, json.dumps(settings, ensure_ascii=False, indent=2))
    return settings


def history() -> list[dict[str, Any]]:
    try:
        payload = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    except (OSError, ValueError):
        return []


def add_history(record: dict[str, Any]) -> None:
    records = [record, *history()][: read_settings()["history_limit"]]
    _atomic_text(HISTORY_PATH, json.dumps(records, ensure_ascii=False, indent=2))


def clear_history() -> None:
    HISTORY_PATH.unlink(missing_ok=True)


def parse_yaml(text: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML 解析失败：{exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("YAML 顶层必须是对象")
    return value


def _segments(path: str) -> list[str | int]:
    parts = path.split(".")
    if parts[0] == "modules" and len(parts) > 2:
        parts = ["modules", ".".join(parts[1:])]
    return [int(part) if part.isdigit() else part for part in parts]


def config_get(data: Any, path: str) -> Any:
    current = data
    for part in _segments(path):
        try:
            current = current[part]
        except (KeyError, IndexError, TypeError):
            return None
    return current


def config_set(data: dict[str, Any], path: str, value: Any) -> None:
    parts = _segments(path)
    current: Any = data
    for index, part in enumerate(parts[:-1]):
        next_part = parts[index + 1]
        if isinstance(part, int):
            if not isinstance(current, list):
                raise ValueError(f"无效的列表字段：{path}")
            while len(current) <= part:
                current.append(None)
            if not isinstance(current[part], (dict, list)):
                current[part] = [] if isinstance(next_part, int) else {}
            current = current[part]
        else:
            if not isinstance(current, dict):
                raise ValueError(f"无效的配置字段：{path}")
            if not isinstance(current.get(part), (dict, list)):
                current[part] = [] if isinstance(next_part, int) else {}
            current = current[part]
    last = parts[-1]
    if isinstance(last, int):
        if not isinstance(current, list):
            raise ValueError(f"无效的列表字段：{path}")
        while len(current) <= last:
            current.append(None)
        current[last] = value
    elif isinstance(current, dict):
        current[last] = value
    else:
        raise ValueError(f"无效的配置字段：{path}")


def flatten_config(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.extend(flatten_config(child, path))
    elif isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            result.append({"path": prefix, "value": value, "kind": "list"})
        else:
            for index, child in enumerate(value):
                result.extend(flatten_config(child, f"{prefix}.{index}"))
    else:
        kind = "bool" if isinstance(value, bool) else "number" if isinstance(value, (int, float)) else "text"
        result.append({"path": prefix, "value": value, "kind": kind})
    return result


def load_config(path: str) -> dict[str, Any]:
    target = project_path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"无法读取配置：{exc}") from exc
    data = parse_yaml(text)
    return {"path": str(target), "text": text, "data": data, "fields": flatten_config(data)}


def patch_config(text: str, path: str, value: Any) -> dict[str, Any]:
    data = parse_yaml(text)
    config_set(data, path, value)
    updated = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    return {"text": updated, "data": data, "fields": flatten_config(data)}


def save_config(path: str, text: str) -> dict[str, Any]:
    data = parse_yaml(text)
    target = project_path(path)
    if target.exists() and not target.is_file():
        raise ValueError("配置路径不是文件")
    if target.exists():
        backup = target.with_suffix(target.suffix + ".bak")
        backup_temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=target.parent, prefix=f".{backup.name}.", delete=False) as handle:
                backup_temporary = Path(handle.name)
                with target.open("rb") as original:
                    shutil.copyfileobj(original, handle)
                handle.flush()
                os.fsync(handle.fileno())
            backup_temporary.replace(backup)
        finally:
            if backup_temporary is not None:
                backup_temporary.unlink(missing_ok=True)
    _atomic_text(target, text)
    return {"path": str(target), "data": data}


def _field_path(value: Any) -> str:
    return str(project_path(str(value))) if value else ""


def _option(raw: Any, options: list[str], name: str) -> str:
    value = str(raw or options[0])
    if value not in options:
        raise ValueError(f"{name} 无效：{value}")
    return value


def build_command(request: dict[str, Any]) -> tuple[str, list[str], str | None]:
    tool_id = str(request.get("tool_id", ""))
    if tool_id not in TOOLS:
        raise ValueError("未知工具")
    tool = TOOLS[tool_id]
    values = request.get("values") or {}
    if not isinstance(values, dict):
        raise ValueError("参数必须是对象")
    settings = read_settings()
    executable = project_path(str(request.get("python_executable") or settings["python_executable"]))
    if not executable.is_file():
        raise ValueError(f"Python 解释器不存在：{executable}")
    config: str | None = None
    args: list[str] = []
    if "variants" in tool:
        variant_id = str(request.get("variant", ""))
        variant = tool["variants"].get(variant_id)
        if variant is None:
            raise ValueError("请选择有效的工具类型")
        args = [str(ROOT / variant["script"])]
        for key, label, kind, flag, required in variant["fields"]:
            value = values.get(key)
            if kind == "bool":
                if key == "normalize":
                    if value is False:
                        args.append("--no-norm")
                elif key == "apply":
                    args.append("--apply" if value else "--dry-run")
                    if value:
                        args.append("--yes")
                elif value:
                    args.append(flag)
                continue
            if required and (value is None or str(value).strip() == ""):
                raise ValueError(f"请填写{label}")
            if value is None or str(value).strip() == "":
                continue
            if kind.startswith("select:"):
                value = _option(value, kind[7:].split("|"), label)
            elif kind in {"file", "dir", "file_or_dir", "save_file"}:
                value = _field_path(value)
            args += [flag, str(value)]
        if variant_id == "filename":
            args.append("--recursive" if values.get("recursive") else "--no-recursive")
        title = variant["title"]
    elif tool_id == "environment":
        args = [str(ROOT / tool["script"]), "--source", _field_path(values.get("source") or "ultralytics-cn"), "--device", _option(values.get("device"), ["auto", "cpu", "cuda:0"], "设备"), "--input-size", str(values.get("size") or 64)]
        for key, flag in (("weights", "--weights"), ("yaml", "--yaml")):
            if values.get(key):
                args += [flag, _field_path(values[key])]
        if values.get("export"):
            args.append("--check-export")
        if values.get("skip_model"):
            args.append("--skip-model")
        title = tool["title"]
    else:
        config = _field_path(request.get("config_path") or tool["config"])
        if not Path(config).is_file():
            raise ValueError(f"配置文件不存在：{config}")
        args = [str(ROOT / "run_tools.py")]
        backend = ""
        if tool_id == "benchmark":
            backend = _option(values.get("backend"), ["pytorch", "tensorrt", "consistency", "all", "checkpoint"], "测速类型")
        if backend == "all":
            args = [str(ROOT / "speed_test/run_all.py"), "--config", config]
        elif backend == "checkpoint":
            args = [str(ROOT / "speed_test/inspect_checkpoint.py"), "--config", config]
        else:
            args += ["--tool", backend or tool["entry"], "--config", config]
        title = tool["title"] if not backend else {"pytorch": "PyTorch 测速", "tensorrt": "TensorRT 测速", "consistency": "输出一致性检查", "all": "一键测速", "checkpoint": "Checkpoint 检查"}[backend]
        resource = values.get("resource")
        if resource:
            resource = _field_path(resource)
            if tool_id == "diagnostics":
                args += ["--weights" if Path(resource).suffix.lower() in {".pt", ".pth", ".onnx"} else "--predictions", resource]
            elif tool_id == "visualization":
                args += ["--weights", resource]
            elif tool_id == "compression":
                args += ["--weights", resource]
            elif backend in {"pytorch", "tensorrt", "consistency"}:
                args += ["--weights", resource]
            else:
                args += ["--model", resource]
        for key, flag in (("data", "--data"), ("source", "--source"), ("train", "--train"), ("val", "--val")):
            if values.get(key) and ((key == "data" and tool_id in {"diagnostics", "compression"}) or (key == "source" and (tool_id == "visualization" or tool_id == "benchmark" and backend != "checkpoint")) or (key in {"train", "val"} and tool_id == "compression")):
                args += [flag, _field_path(values[key])]
        if tool_id == "compression":
            args += ["--task", _option(values.get("task"), ["detect", "classify"], "模型任务")]
        if values.get("device") and values["device"] != "auto" and backend != "checkpoint":
            args += ["--device", _option(values["device"], ["auto", "cpu", "cuda:0"], "设备")]
        edits = request.get("module_edits") or {}
        if not isinstance(edits, dict):
            raise ValueError("模块修改必须是对象")
        available = {item[0] for item in tool.get("modules", [])}
        data = load_config(config)["data"]
        configured = data.get("modules")
        if isinstance(configured, dict):
            available.update(str(key) for key in configured)
        for module_id, enabled in edits.items():
            if module_id not in available or not isinstance(enabled, bool):
                raise ValueError(f"无效模块：{module_id}")
            if backend == "checkpoint":
                raise ValueError("该测速类型请在 YAML 中设置模块")
            args += ["--enable" if enabled else "--disable", module_id]
    if not Path(args[0]).is_file():
        raise ValueError(f"入口文件不存在：{args[0]}")
    return str(executable), args, config


def list_results(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise ValueError(f"结果目录不存在：{root}")
    files: list[dict[str, Any]] = []
    suffixes = TEXT_SUFFIXES | IMAGE_SUFFIXES | MODEL_SUFFIXES
    project_scan = root.resolve() == ROOT.resolve()
    result_markers = {"runs", "runs-profile", "results", "outputs", "reports"}
    for parent, dirs, names in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name not in SKIP_DIRS and not name.startswith("."))
        for name in sorted(names):
            path = Path(parent) / name
            if project_scan and not any(part in result_markers for part in path.relative_to(root).parts[:-1]):
                continue
            if path.suffix.lower() not in suffixes or path.is_symlink():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            kind = "image" if path.suffix.lower() in IMAGE_SUFFIXES else "text" if path.suffix.lower() in TEXT_SUFFIXES else "model"
            files.append({"path": str(path), "relative": str(path.relative_to(root)), "size": stat.st_size, "kind": kind, "mtime": stat.st_mtime})
            if len(files) >= 5000:
                break
        if len(files) >= 5000:
            break
    files.sort(key=lambda item: item["mtime"], reverse=True)
    for item in files:
        item.pop("mtime")
    return files


def result_file(root: Path, raw: str) -> Path:
    path = project_path(raw)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("只能查看所选结果目录内的文件") from exc
    if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES | IMAGE_SUFFIXES | MODEL_SUFFIXES:
        raise ValueError("不支持的结果文件")
    return path


def preview_text(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        content = handle.read(PREVIEW_BYTES + 1)
    return {"text": content[:PREVIEW_BYTES].decode("utf-8", errors="replace"), "truncated": len(content) > PREVIEW_BYTES}


def environments() -> list[dict[str, str]]:
    candidates = {str(Path(sys.executable).resolve())}
    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            candidates.add(str(Path(found).resolve()))
    conda_exe = shutil.which("conda")
    if conda_exe:
        try:
            import subprocess

            completed = subprocess.run([conda_exe, "env", "list", "--json"], capture_output=True, text=True, timeout=8, check=False)
            for raw in json.loads(completed.stdout).get("envs", []):
                env = Path(raw)
                candidate = env / ("python.exe" if os.name == "nt" else "bin/python")
                if candidate.is_file():
                    candidates.add(str(candidate.resolve()))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    return [{"label": Path(path).parent.parent.name or path, "path": path} for path in sorted(candidates)]

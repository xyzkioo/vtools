"""Qt-independent configuration, command and result services."""

from __future__ import annotations

import json
import math
import os
import secrets
import shutil
import string
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .catalog import ROOT, TOOLS
from .config_fields import SIZE_FIELDS, SQUARE_SIZE_FIELDS, STRING_LIST_FIELDS, field_hint

STATE_DIR = (
    Path(os.environ.get("VTOOLS_USER_DATA", "")).expanduser()
    if getattr(sys, "frozen", False) and os.environ.get("VTOOLS_USER_DATA")
    else Path.home() / ".vtools_ui"
    if getattr(sys, "frozen", False)
    else ROOT / ".vtools_ui"
)
SETTINGS_PATH = STATE_DIR / "settings.json"
HISTORY_PATH = STATE_DIR / "history.json"
TEXT_SUFFIXES = {".json", ".csv", ".html", ".md", ".txt", ".yaml", ".yml"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff", ".svg"}
MODEL_SUFFIXES = {".pt", ".pth", ".onnx", ".engine", ".kmodel"}
SKIP_DIRS = {".git", ".vtools_ui", "__pycache__", "node_modules"}
PREVIEW_BYTES = 256 * 1024


def _visible_run_name(root: Path) -> str:
    """Return a readable sequential run name with a short random suffix."""

    largest = 0
    if root.is_dir():
        for child in root.iterdir():
            if not child.is_dir() or not child.name.startswith("run"):
                continue
            number, separator, _suffix = child.name[3:].partition("-")
            if separator and number.isdigit():
                largest = max(largest, int(number))
    suffix = "".join(secrets.choice(string.ascii_lowercase) for _ in range(5))
    return f"run{largest + 1}-{suffix}"


def _uv_project_python_environments() -> list[tuple[str, Path]]:
    """Find uv project interpreters in the user's conventional environment roots."""

    roots = [Path.home() / "python-envs"]
    configured_roots = os.environ.get("VTOOLS_UV_PROJECTS_ROOT", "")
    roots.extend(Path(raw).expanduser() for raw in configured_roots.split(os.pathsep) if raw)
    found: dict[Path, str] = {}
    interpreter_name = "Scripts/python.exe" if os.name == "nt" else "bin/python"

    active_environment = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if active_environment:
        prefix = Path(active_environment).expanduser()
        if not prefix.is_absolute():
            prefix = Path.cwd() / prefix
        executable = prefix / interpreter_name
        if executable.is_file():
            found[Path(os.path.abspath(executable))] = f"{prefix.parent.name} (uv)"

    for root in dict.fromkeys(roots):
        projects = [root]
        try:
            projects.extend(child for child in root.iterdir() if child.is_dir())
        except OSError:
            continue
        for project in projects:
            if not (project / "pyproject.toml").is_file():
                continue
            executable = project / ".venv" / interpreter_name
            if executable.is_file():
                found[Path(os.path.abspath(executable))] = f"{project.name} (uv)"

    return [(label, path) for path, label in sorted(found.items(), key=lambda item: str(item[0]))]


def _default_python_executable() -> str:
    """Use a real Python environment for jobs launched by a desktop bundle."""

    if getattr(sys, "frozen", False):
        uv_environments = _uv_project_python_environments()
        for label, executable in uv_environments:
            if label.removesuffix(" (uv)").casefold() == "vtools":
                return str(executable)
        if os.environ.get("VTOOLS_UV_PROJECTS_ROOT") and uv_environments:
            return str(uv_environments[0][1])
        for name in ("python", "python3"):
            candidate = shutil.which(name)
            if candidate:
                return os.path.abspath(candidate)
    return sys.executable


def _default_results_root() -> str:
    """Keep packaged run output outside the read-only application directory."""

    return str(STATE_DIR / "runs") if getattr(sys, "frozen", False) else str(ROOT)


def _results_root_path(raw: Any) -> Path:
    """Resolve the configured results directory without returning to the bundle."""

    candidate = Path(str(raw or "")).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    if getattr(sys, "frozen", False) and (candidate == ROOT or ROOT in candidate.parents):
        return (STATE_DIR / "runs").resolve()
    return candidate


def project_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def python_executable_path(raw: str) -> Path:
    """Normalize an interpreter path without resolving venv symlinks."""

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return Path(os.path.abspath(path))


def _editable_config_path(target: Path) -> Path:
    """Copy packaged config templates to a writable per-user location."""

    if not getattr(sys, "frozen", False):
        return target
    try:
        relative = target.relative_to(ROOT)
    except ValueError:
        return target
    user_target = STATE_DIR / "configs" / relative
    if not user_target.exists():
        if not target.is_file():
            return target
        text = target.read_text(encoding="utf-8")
        data = parse_yaml(text)
        project = data.get("project")
        if isinstance(project, dict) and project.get("root") is not None:
            root = Path(str(project["root"])).expanduser()
            if not root.is_absolute():
                # Moving a template must not change where its relative inputs
                # are resolved. Existing user copies remain user-owned.
                project["root"] = str((target.parent / root).resolve())
                text = (
                    json.dumps(data, ensure_ascii=False, indent=2) if target.suffix.lower() == ".json"
                    else yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
                )
        _atomic_text(user_target, text)
    return user_target


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
    defaults = {
        "results_root": _default_results_root(),
        "history_limit": 40,
        "python_executable": _default_python_executable(),
    }
    if SETTINGS_PATH.exists():
        try:
            loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                defaults.update({key: loaded[key] for key in defaults if key in loaded})
        except (OSError, ValueError):
            pass
    defaults["results_root"] = str(_results_root_path(defaults["results_root"]))
    try:
        defaults["history_limit"] = max(1, min(200, int(defaults["history_limit"])))
    except (TypeError, ValueError):
        defaults["history_limit"] = 40
    return defaults


def save_settings(changes: dict[str, Any]) -> dict[str, Any]:
    settings = read_settings()
    if "results_root" in changes:
        settings["results_root"] = str(_results_root_path(changes["results_root"]))
        if getattr(sys, "frozen", False):
            Path(settings["results_root"]).mkdir(parents=True, exist_ok=True)
    if "python_executable" in changes:
        candidate = python_executable_path(str(changes["python_executable"]))
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
        value = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML 解析失败：{exc}") from exc
    if value is None:
        value = {}
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
            result.append(_config_field(prefix, value, "list"))
        else:
            for index, child in enumerate(value):
                result.extend(flatten_config(child, f"{prefix}.{index}"))
    else:
        kind = "bool" if isinstance(value, bool) else "number" if isinstance(value, (int, float)) else "text"
        result.append(_config_field(prefix, value, kind))
    return result


def _config_field(path: str, value: Any, kind: str) -> dict[str, Any]:
    if path in SIZE_FIELDS:
        kind = "size"
    elif path in SQUARE_SIZE_FIELDS:
        kind = "number"
    return {"path": path, "value": value, "kind": kind, "hint": field_hint(path, kind)}


def _form_value(path: str, raw: Any, original: Any) -> Any:
    """Parse graphical input before patching, preserving scalar/list types."""

    if path in SIZE_FIELDS or path in SQUARE_SIZE_FIELDS:
        values = _form_list(str(raw))
        if len(values) == 1 and not str(raw).strip().startswith("["):
            values *= 2
        if len(values) != 2 or any(type(item) is not int or item <= 0 for item in values):
            raise ValueError(f"{path} 需要正整数尺寸，填写 高,宽（如 640,640）或单个尺寸（如 640）")
        if path in SQUARE_SIZE_FIELDS:
            if values[0] != values[1]:
                raise ValueError(f"{path} 使用方形输入，高、宽需相同，如 640 或 640,640")
            return values[0]
        return values
    if isinstance(original, bool):
        if not isinstance(raw, bool):
            raise ValueError(f"{path} 需要布尔值")
        return raw
    if isinstance(original, (int, float)):
        return _form_number(path, raw, integer=isinstance(original, int))
    if isinstance(original, list):
        # String lists (paths, layer names, precisions) must remain strings,
        # including entries such as "001", "true" and "null".
        strings = path in STRING_LIST_FIELDS or bool(original) and all(isinstance(item, str) for item in original)
        values = _form_list(str(raw), strings=strings)
        if original and all(type(item) in (int, float) for item in original):
            if any(type(item) not in (int, float) or not math.isfinite(item) for item in values):
                raise ValueError(f"{path} 需要数字列表，如 0.1,0.5")
        return values
    return raw


def _form_list(raw: str, *, strings: bool = False) -> list[Any]:
    text = raw.strip()
    if not text:
        return []
    if not text.startswith("["):
        text = f"[{text}]"
    try:
        values = yaml.load(text, Loader=yaml.BaseLoader if strings else yaml.SafeLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"列表格式无效：{exc}") from exc
    if not isinstance(values, list) or any(isinstance(item, (dict, list)) for item in values):
        raise ValueError("请填写由数字或文本组成的一维列表")
    return values


def _form_number(path: str, raw: Any, *, integer: bool) -> int | float | None:
    if not isinstance(raw, str):
        raise ValueError(f"{path} 需要有效数字")
    if not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{path} 需要有效数字") from exc
    if not math.isfinite(value) or (integer and not value.is_integer()):
        raise ValueError(f"{path} 需要{'整数' if integer else '有限数字'}")
    return int(value) if integer else value


def load_config(path: str) -> dict[str, Any]:
    target = _editable_config_path(project_path(path))
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"无法读取配置：{exc}") from exc
    data = parse_yaml(text)
    return {"path": str(target), "text": text, "data": data, "fields": flatten_config(data)}


def patch_config(text: str, path: str, value: Any, *, from_form: bool = False) -> dict[str, Any]:
    data = parse_yaml(text)
    if from_form:
        value = _form_value(path, value, config_get(data, path))
    config_set(data, path, value)
    updated = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    return {"text": updated, "data": data, "fields": flatten_config(data)}


def save_config(path: str, text: str) -> dict[str, Any]:
    data = parse_yaml(text)
    target = _editable_config_path(project_path(path))
    if target.exists() and not target.is_file():
        raise ValueError("配置路径不是文件")
    # The graphical patch API emits YAML; JSON configs must still be readable
    # by backends that select their parser from the filename suffix.
    if target.suffix.lower() == ".json":
        text = json.dumps(data, ensure_ascii=False, indent=2)
    _atomic_text(target, text)
    return {"path": str(target), "data": data}


def _field_path(value: Any) -> str:
    return str(project_path(str(value))) if value else ""


def _option(raw: Any, options: list[str], name: str) -> str:
    value = str(raw or options[0])
    if value not in options:
        raise ValueError(f"{name} 无效：{value}")
    return value


def _uses_bundled_tool_runtime(
    tool_id: str, request: dict[str, Any], config_data: dict[str, Any],
) -> bool:
    """Select only paths that run without importing a model framework."""

    if not getattr(sys, "frozen", False):
        return False
    if tool_id == "diagnostics":
        # Existing predictions use the same matching/report engine as model
        # inference, but never create an adapter. An explicit model input takes
        # precedence over the YAML mode and still requires external packages.
        values = request.get("values") or {}
        dataset = config_data.get("dataset")
        dataset = dataset if isinstance(dataset, dict) else {}
        ground_truth = values.get("data") or dataset.get("data") or dataset.get("yaml") or dataset.get("gt")
        # NDJSON ground truth invokes the upstream dataset converter, so that
        # path still needs Ultralytics even when predictions are already saved.
        if str(dataset.get("gt_format", "")).lower() == "ndjson" or Path(str(ground_truth or "")).suffix.lower() == ".ndjson":
            return False
        resource = str(values.get("resource") or "").strip()
        if resource:
            return Path(resource).suffix.lower() not in MODEL_SUFFIXES
        return str(config_data.get("mode", "")).strip().lower() == "predictions_file"
    # Keep this list explicit: NDJSON conversion currently calls Ultralytics;
    # checkpoint inspection needs torch even though it does not run inference.
    return tool_id == "dataset_quality" or (
        tool_id == "data" and request.get("variant") in {"filename", "resize", "video"}
    )


def build_command(request: dict[str, Any]) -> tuple[str, list[str], str | None]:
    tool_id = str(request.get("tool_id", ""))
    if tool_id not in TOOLS:
        raise ValueError("未知工具")
    tool = TOOLS[tool_id]
    values = request.get("values") or {}
    if not isinstance(values, dict):
        raise ValueError("参数必须是对象")
    settings = read_settings()
    config: str | None = None
    config_data: dict[str, Any] = {}
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
                if getattr(sys, "frozen", False) and variant_id == "kmodel" and key in {"onnx", "kmodel"}:
                    output = Path(value)
                    if output == ROOT or ROOT in output.parents:
                        raise ValueError(f"{label}不能放在安装目录，请选择用户可写目录")
            args += [flag, str(value)]
        if variant_id == "filename":
            args.append("--recursive" if values.get("recursive") else "--no-recursive")
        if getattr(sys, "frozen", False) and variant_id == "kmodel" and "--output-dir" not in args:
            # py2kmodel has repository-relative defaults for optional ONNX and
            # kmodel outputs. A packaged app must never target its read-only
            # _internal directory when those fields are left blank.
            output_parent = _results_root_path(settings["results_root"]) / "transform" / "kmodel"
            output_root = output_parent / _visible_run_name(output_parent)
            output_root.mkdir(parents=True, exist_ok=True)
            args += ["--output-dir", str(output_root)]
        title = variant["title"]
    elif tool_id == "environment":
        args = [str(ROOT / tool["script"])]
        if values.get("source") and str(values["source"]).strip():
            args += ["--source", _field_path(values["source"])]
        args += ["--device", _option(values.get("device"), ["auto", "cpu", "cuda:0"], "设备"), "--input-size", str(values.get("size") or 64)]
        for key, flag in (("weights", "--weights"), ("yaml", "--yaml")):
            if values.get(key):
                args += [flag, _field_path(values[key])]
        if values.get("export"):
            args.append("--check-export")
        if values.get("skip_model"):
            args.append("--skip-model")
        title = tool["title"]
    elif tool.get("script"):
        args = [str(ROOT / tool["script"])]
        for field in tool.get("fields", []):
            key = str(field.get("key", ""))
            label = str(field.get("label", key))
            kind = str(field.get("kind", "text"))
            value = values.get(key)
            required = bool(field.get("required", False))
            if required and (value is None or str(value).strip() == ""):
                raise ValueError(f"请填写{label}")
            if value is None or str(value).strip() == "":
                continue
            flag = str(field.get("flag", "--" + key.replace("_", "-")))
            if kind == "number":
                try:
                    number = int(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{label}必须是整数") from exc
                if number < 0:
                    raise ValueError(f"{label}不能为负数")
                args += [flag, str(number)]
            elif kind in {"file", "dir", "file_or_dir", "save_file"}:
                resolved = _field_path(value)
                if getattr(sys, "frozen", False) and key in {"output", "output_root", "onnx", "kmodel", "plan"}:
                    output = Path(resolved)
                    if output == ROOT or ROOT in output.parents:
                        raise ValueError(f"{label}不能放在安装目录，请选择用户可写目录")
                args += [flag, resolved]
            else:
                args += [flag, str(value)]
        if getattr(sys, "frozen", False) and tool_id == "dataset_quality" and "--output-root" not in args:
            output_root = _results_root_path(settings["results_root"]) / "dataset_quality"
            output_root.mkdir(parents=True, exist_ok=True)
            args += ["--output-root", str(output_root)]
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
        compression_mode = values.get("processing_mode") if tool_id == "compression" else None
        mode_modules = None
        compression_task = None
        if tool_id == "compression":
            compression_task = _option(values.get("task"), ["detect", "classify"], "模型任务")
        if compression_mode is not None:
            from model_compression.core.module_selection import COMPRESSION_MODES, compression_mode_modules
            mode_modules = compression_mode_modules(str(compression_mode), compression_task)
        resource = values.get("resource")
        if compression_mode is not None and not COMPRESSION_MODES[str(compression_mode)]["public_weights"]:
            resource = None
        if resource:
            resource = _field_path(resource)
            if tool_id == "diagnostics":
                args += ["--weights" if Path(resource).suffix.lower() in MODEL_SUFFIXES else "--predictions", resource]
            elif tool_id == "visualization":
                args += ["--weights", resource]
            elif tool_id == "compression":
                args += ["--weights", resource]
            elif backend in {"pytorch", "tensorrt", "consistency"}:
                args += ["--weights", resource]
            else:
                args += ["--model", resource]
        for key, flag in (("data", "--data"), ("source", "--source"), ("train", "--train"), ("val", "--val")):
            if compression_mode is not None and ((compression_task == "detect" and key in {"train", "val"})
                                                  or (compression_task == "classify" and key == "data")):
                continue
            if values.get(key) and ((key == "data" and tool_id in {"diagnostics", "compression"}) or (key == "source" and (tool_id == "visualization" or tool_id == "benchmark" and backend != "checkpoint")) or (key in {"train", "val"} and tool_id == "compression")):
                args += [flag, _field_path(values[key])]
        if tool_id == "compression":
            args += ["--task", compression_task]
            for key, flag in (("teacher_weights", "--teacher-weights"),
                              ("student_weights", "--student-weights")):
                if compression_mode is not None and compression_mode != "detection_distillation":
                    continue
                if values.get(key):
                    args += [flag, _field_path(values[key])]
        if values.get("device") and values["device"] != "auto" and backend != "checkpoint":
            args += ["--device", _option(values["device"], ["auto", "cpu", "cuda:0"], "设备")]
        edits = request.get("module_edits") or {}
        if not isinstance(edits, dict):
            raise ValueError("模块修改必须是对象")
        if mode_modules is not None:
            if edits:
                raise ValueError("选择处理模式时不能同时指定模块开关。")
            for module_id in mode_modules:
                args += ["--only", module_id]
        available = {item[0] for item in tool.get("modules", [])}
        loaded = load_config(config)
        config = loaded["path"]
        data = loaded["data"]
        config_data = data
        if "--config" in args:
            args[args.index("--config") + 1] = config
        configured = data.get("modules")
        if isinstance(configured, dict):
            available.update(str(key) for key in configured)
        for module_id, enabled in edits.items():
            if module_id not in available or not isinstance(enabled, bool):
                raise ValueError(f"无效模块：{module_id}")
            if backend == "checkpoint":
                raise ValueError("该测速类型请在 YAML 中设置模块")
            args += ["--enable" if enabled else "--disable", module_id]
        if getattr(sys, "frozen", False) and backend != "checkpoint":
            results_root = _results_root_path(settings["results_root"])
            tool_root = results_root / tool_id
            tool_root.mkdir(parents=True, exist_ok=True)
            args += ["--run-dir", str(tool_root / _visible_run_name(tool_root))]
    # Decide after loading the effective config, so predictions_file configured
    # in YAML also works without an external interpreter or a UI file override.
    bundled_runtime = _uses_bundled_tool_runtime(tool_id, request, config_data)
    executable = (
        Path(sys.executable) if bundled_runtime
        else python_executable_path(str(request.get("python_executable") or settings["python_executable"]))
    )
    if not executable.is_file():
        raise ValueError(f"Python 解释器不存在：{executable}")
    bundled_executable = getattr(sys, "frozen", False) and executable.resolve() == Path(sys.executable).resolve()
    if bundled_executable and not bundled_runtime:
        raise ValueError("此功能需要外部 Python 模型环境，请在环境选择器或设置中选择已安装相应依赖的解释器")
    script = Path(args[0])
    if not script.is_file():
        raise ValueError(f"入口文件不存在：{args[0]}")
    if bundled_executable:
        args = ["--run-script", *args]
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
    candidates: set[str] = set()
    labels: dict[str, str] = {}

    def add_python(raw: str | Path, label: str | None = None) -> None:
        path = Path(raw).expanduser()
        if path.is_file():
            executable = os.path.abspath(path)
            # A frozen workbench can dispatch bundled scripts, but it is not a
            # user Python environment and excludes heavy model dependencies.
            if not (
                getattr(sys, "frozen", False)
                and Path(executable).resolve() == Path(sys.executable).resolve()
            ):
                candidates.add(executable)
                if label:
                    labels[executable] = label

    if not getattr(sys, "frozen", False):
        add_python(sys.executable)
    saved = read_settings().get("python_executable")
    if saved:
        add_python(str(saved))
    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            add_python(found)

    conda_executables: list[Path] = []
    for raw in (
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
        Path.home() / "miniconda3/bin/conda",
        Path.home() / "anaconda3/bin/conda",
        Path.home() / "miniforge3/bin/conda",
        Path.home() / "mambaforge/bin/conda",
    ):
        if raw:
            path = Path(raw).expanduser()
            if path.is_file() and path.resolve() not in conda_executables:
                conda_executables.append(path.resolve())

    prefixes: set[Path] = set()
    environments_file = Path.home() / ".conda/environments.txt"
    try:
        prefixes.update(Path(line.strip()).expanduser() for line in environments_file.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        pass
    for conda_exe in conda_executables:
        try:
            completed = subprocess.run([str(conda_exe), "env", "list", "--json"], capture_output=True, text=True, timeout=8, check=False)
            for raw in json.loads(completed.stdout).get("envs", []):
                prefixes.add(Path(raw).expanduser())
        except (OSError, ValueError, subprocess.TimeoutExpired):
            continue
        if prefixes:
            break
    for prefix in prefixes:
        add_python(prefix / ("python.exe" if os.name == "nt" else "bin/python"))

    for label, executable in _uv_project_python_environments():
        add_python(executable, label)

    return [
        {
            "label": labels.get(path, Path(path).parent.parent.name or path),
            "path": path,
        }
        for path in sorted(candidates)
    ]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small, human-readable TensorRT build metadata.

The project intentionally uses the existing ``rebuild_engine`` switch as the
cache policy.  This sidecar records what was used to build/reuse an engine; it
does not hash every source file and it never decides on its own to rebuild.
"""

from __future__ import annotations

import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def metadata_path(engine_path: Path) -> Path:
    return engine_path.with_name(engine_path.name + ".build.json")


def _version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception:
        return "unavailable"
    return str(getattr(module, "__version__", "unknown"))


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(torch.cuda.current_device()))
    except Exception:
        pass
    return "unavailable"


def collect_build_metadata(**values: Any) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "host": platform.node(),
        "python": platform.python_version(),
        "environment": {
            "torch": _version("torch"),
            "tensorrt": _version("tensorrt"),
            "cuda": _cuda_version(),
            "gpu": _gpu_name(),
        },
    }
    payload.update(values)
    return payload


def _cuda_version() -> str:
    try:
        import torch

        return str(getattr(torch.version, "cuda", None) or "unavailable")
    except Exception:
        return "unavailable"


def write_build_metadata(engine_path: Path, payload: Mapping[str, Any]) -> Path:
    path = metadata_path(engine_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def read_build_metadata(engine_path: Path) -> dict[str, Any] | None:
    path = metadata_path(engine_path)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def compare_build_request(metadata: Mapping[str, Any] | None, request: Mapping[str, Any]) -> list[str]:
    """Return informational differences; caller still controls rebuild policy."""
    if not metadata:
        return ["缺少 .build.json 元数据"]
    differences = []
    for key in ("precision", "batch", "input_size", "fuse", "opset", "workspace_mb", "builder", "weights", "onnx"):
        if key in request and key in metadata and str(request[key]) != str(metadata[key]):
            differences.append(f"{key}: 当前={request[key]!s}，记录={metadata[key]!s}")
    return differences


__all__ = ["collect_build_metadata", "compare_build_request", "metadata_path", "read_build_metadata", "write_build_metadata"]

"""测速功能开关和命令行选择。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from vtools_runtime.module_selection import resolve_module_selection


SPEED_MODULES = {
    "checkpoint.inspect": False,
    "checkpoint.load_check": False,
    "model.parameters": False,
    "model.flops": False,
    "speed.pytorch_call": True,
    "speed.pytorch_pipeline": False,
    "speed.tensorrt_call": True,
    "memory.pytorch_peak": False,
    "export.onnx": False,
    "build.tensorrt": False,
    "consistency.tensor": True,
    "consistency.detection": False,
}


def resolve_speed_modules(config: Mapping[str, Any], *, only: Iterable[str] | None = None, enable: Iterable[str] | None = None, disable: Iterable[str] | None = None) -> dict[str, bool]:
    return resolve_module_selection(SPEED_MODULES, config, only=only, enable=enable,
                                    disable=disable, label='测速')

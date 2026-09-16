"""模型压缩运行目录管理。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional


def prepare_run_directory(config: Mapping[str, Any], requested: str | Path | None = None) -> Optional[Path]:
    run = config.get("run") if isinstance(config.get("run"), Mapping) else {}
    if not bool(run.get("enabled", True)):
        return None
    root = Path(str(run.get("root", "model_compression/runs"))).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if requested:
        target = Path(str(requested)).expanduser()
        if not target.is_absolute():
            target = root / target
        target = target.resolve()
        target.mkdir(parents=True, exist_ok=True)
    elif str(run.get("name", "auto")).lower() == "auto":
        index = 1
        while True:
            target = root / f"run{index}"
            try:
                target.mkdir()
                break
            except FileExistsError:
                index += 1
    else:
        target = root / str(run["name"])
        target.mkdir(parents=True, exist_ok=False)
    for category in ("structured_pruning", "unstructured_pruning", "distillation", "quantization", "multi_processing"):
        (target / "artifacts" / category).mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return target


__all__ = ["prepare_run_directory", "write_json"]

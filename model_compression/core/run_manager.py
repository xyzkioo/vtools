"""模型压缩运行目录管理。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
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
    info = {
        "run_dir": str(target),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config.get("_config_path", "")),
    }
    info_path = target / "run_info.json"
    if not info_path.exists():
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return target


__all__ = ["prepare_run_directory", "write_json"]

"""Discover an optional external Ultralytics checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping


CHECKOUT_NAMES = ("ultralytics-cn", "ultralytics-ezcn", "ultralytics")


def _as_path(value: Any, base: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _is_checkout(path: Path) -> bool:
    return (path / "ultralytics" / "__init__.py").is_file()


def find_repo(
    settings: Mapping[str, Any] | None,
    project_root: Path,
    input_paths: tuple[Path, ...] = (),
) -> Path | None:
    """Find a checkout near the project or the selected model and dataset."""
    project = settings.get("project", {}) if isinstance(settings, Mapping) else {}
    configured = project.get("ultralytics_repo") if isinstance(project, Mapping) else None
    candidates: list[Path] = []
    for value in (configured, os.environ.get("VTOOLS_ULTRALYTICS_REPO")):
        path = _as_path(value, project_root)
        if path is not None:
            candidates.append(path)
    for parent in (project_root, project_root.parent):
        candidates.extend(parent / name for name in CHECKOUT_NAMES)
    # A packaged workbench lives under /opt/vtools while the user's data and
    # Ultralytics checkout commonly share a separate workspace directory.
    for input_path in input_paths:
        path = input_path.expanduser().resolve()
        for parent in path.parents:
            candidates.extend(parent / name for name in CHECKOUT_NAMES)
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if _is_checkout(candidate):
            return candidate
    return None


def add_repo_to_path(
    settings: Mapping[str, Any] | None,
    project_root: Path,
    input_paths: tuple[Path, ...] = (),
) -> Path | None:
    repo = find_repo(settings, project_root, input_paths)
    if repo is not None and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def import_error_message(repo: Path | None) -> str:
    if repo is None:
        return (
            "未找到 Ultralytics。请将你的 Fork 放在 vtools 同级的 ultralytics-cn/ 或 ultralytics-ezcn/ 目录，"
            "或设置 VTOOLS_ULTRALYTICS_REPO，或填写 project.ultralytics_repo。"
        )
    return f"无法从 {repo} 导入 Ultralytics，请检查该 Fork 的依赖和源码完整性"

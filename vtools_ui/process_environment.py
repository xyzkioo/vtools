"""Build a predictable environment for tool subprocesses."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping


def _prepend_path(environment: dict[str, str], name: str, value: Path) -> None:
    """Put *value* first in a path-list variable without duplicating it."""

    target = value.expanduser().resolve()
    parts = [item for item in environment.get(name, "").split(os.pathsep) if item]
    kept: list[str] = []
    for item in parts:
        try:
            if Path(item).expanduser().resolve() == target:
                continue
        except OSError:
            pass
        kept.append(item)
    environment[name] = os.pathsep.join([str(target), *kept])


def build_tool_environment(
    executable: str | Path,
    runtime_root: str | Path,
    inherited: Mapping[str, str] | None = None,
    *,
    frozen: bool | None = None,
) -> dict[str, str]:
    """Return the environment used by a selected external Python.

    A frozen workbench launches source files from its ``_internal`` directory
    with a user-selected Python.  Merely replacing the executable does not make
    that directory importable and leaves PyInstaller's library search path in
    the child.  Construct the bridge explicitly for every tool process.
    """

    environment = dict(os.environ if inherited is None else inherited)
    environment.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
    executable_path = Path(executable).expanduser().resolve()
    root = Path(runtime_root).expanduser().resolve()

    _prepend_path(environment, "PATH", executable_path.parent)
    _prepend_path(environment, "PYTHONPATH", root)

    is_frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    if is_frozen and sys.platform.startswith("linux"):
        # PyInstaller saves the pre-launch value here.  Its own bundled
        # libraries are correct for the UI executable but can break an external
        # Conda Python and its compiled extensions.
        original = environment.get("LD_LIBRARY_PATH_ORIG")
        if original:
            environment["LD_LIBRARY_PATH"] = original
        else:
            environment.pop("LD_LIBRARY_PATH", None)

    prefix = executable_path.parent.parent
    conda_library = prefix / "lib"
    if sys.platform.startswith("linux") and (conda_library / "libstdc++.so.6").is_file():
        _prepend_path(environment, "LD_LIBRARY_PATH", conda_library)
    if (prefix / "conda-meta").is_dir():
        environment["CONDA_PREFIX"] = str(prefix)
        environment["CONDA_DEFAULT_ENV"] = prefix.name

    return environment


__all__ = ["build_tool_environment"]

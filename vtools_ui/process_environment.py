"""Build a predictable environment for tool subprocesses."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping


def _prepend_path(environment: dict[str, str], name: str, value: Path) -> None:
    """Put *value* first in a path-list variable without duplicating it."""

    target = Path(os.path.abspath(value.expanduser()))
    parts = [item for item in environment.get(name, "").split(os.pathsep) if item]
    kept: list[str] = []
    for item in parts:
        try:
            if Path(os.path.abspath(Path(item).expanduser())) == target:
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
    """Return the environment used by a bundled tool or external Python.

    A frozen workbench launches source files from its ``_internal`` directory
    with either its own lightweight runtime or a user-selected model runtime.
    Both need the tool sources on PYTHONPATH. Only external interpreters must
    have PyInstaller's private library path removed.
    """

    environment = dict(os.environ if inherited is None else inherited)
    environment.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
    # Preserve .venv/bin/python symlinks: resolving them selects uv's base
    # interpreter and makes Python lose the virtual environment's site-packages.
    executable_path = Path(os.path.abspath(Path(executable).expanduser()))
    root = Path(runtime_root).expanduser().resolve()

    _prepend_path(environment, "PATH", executable_path.parent)
    _prepend_path(environment, "PYTHONPATH", root)

    is_frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    is_bundled_executable = executable_path.resolve() == Path(sys.executable).resolve()
    # Bundled data tools re-enter the PyInstaller executable and need its
    # packaged Qt/OpenCV libraries; restoring LD_LIBRARY_PATH_ORIG breaks them.
    if is_frozen and sys.platform.startswith("linux") and not is_bundled_executable:
        # PyInstaller saves the pre-launch value here.  Its own bundled
        # libraries are correct for the UI executable but can break an external
        # Conda Python and its compiled extensions.
        original = environment.get("LD_LIBRARY_PATH_ORIG")
        if original:
            environment["LD_LIBRARY_PATH"] = original
        else:
            environment.pop("LD_LIBRARY_PATH", None)

    if is_frozen and is_bundled_executable:
        return environment

    prefix = executable_path.parent.parent
    conda_library = prefix / "lib"
    if sys.platform.startswith("linux") and (conda_library / "libstdc++.so.6").is_file():
        _prepend_path(environment, "LD_LIBRARY_PATH", conda_library)
    if (prefix / "conda-meta").is_dir():
        environment["CONDA_PREFIX"] = str(prefix)
        environment["CONDA_DEFAULT_ENV"] = prefix.name
    else:
        # A uv job launched from a Conda shell must not let vision_runtime
        # discover and inject the previous environment's compiled libraries.
        environment.pop("CONDA_PREFIX", None)
        environment.pop("CONDA_DEFAULT_ENV", None)
    if (prefix / "pyvenv.cfg").is_file():
        environment["VIRTUAL_ENV"] = str(prefix)
    else:
        environment.pop("VIRTUAL_ENV", None)

    # nncase loads hostfxr through the .NET runtime bundled with the Conda
    # installation.  A frozen PyInstaller process may otherwise leave the
    # child with only the workbench's private libraries on its search path.
    dotnet_root = prefix.parent.parent / "lib" / "dotnet"
    if (dotnet_root / "host" / "fxr").is_dir() and not environment.get("DOTNET_ROOT"):
        environment["DOTNET_ROOT"] = str(dotnet_root)
        environment["DOTNET_ROOT_X64"] = str(dotnet_root)

    return environment


__all__ = ["build_tool_environment"]

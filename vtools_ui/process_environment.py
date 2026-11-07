"""Build a predictable environment for tool subprocesses."""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
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

    A frozen workbench launches pure source files from ``_internal/tool-sources``
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


    is_frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    is_bundled_executable = executable_path.resolve() == Path(sys.executable).resolve()
    if is_frozen and sys.platform == "win32":
        is_bundled_executable |= executable_path.resolve() == Path(sys.executable).with_name("vtools-worker.exe").resolve()
    if is_frozen:
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if is_frozen and not is_bundled_executable:
        bundle = Path(getattr(sys, "_MEIPASS", root.parent if root.name == "tool-sources" else root)).resolve()
        def private_path(raw: str) -> bool:
            try:
                return Path(raw).expanduser().resolve().is_relative_to(bundle)
            except (OSError, ValueError):
                return False
        for name in ("PYTHONPATH", "PATH"):
            parts = [part for part in environment.get(name, "").split(os.pathsep) if part]
            environment[name] = os.pathsep.join(part for part in parts if not private_path(part))
        for name in ("PYTHONHOME", "PYTHONEXECUTABLE", "__PYVENV_LAUNCHER__"):
            environment.pop(name, None)
        for name in list(environment):
            if name.startswith("_PYI_") or name == "PYINSTALLER_RESET_ENVIRONMENT":
                environment.pop(name, None)
        for name in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QML2_IMPORT_PATH", "QTWEBENGINEPROCESS_PATH"):
            if any(private_path(part) for part in environment.get(name, "").split(os.pathsep) if part):
                environment.pop(name, None)
    _prepend_path(environment, "PYTHONPATH", root)
    _prepend_path(environment, "PATH", executable_path.parent)
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

    prefix = executable_path.parent
    if sys.platform != "win32" or prefix.name.casefold() == "scripts" or (prefix.parent / "pyvenv.cfg").is_file():
        prefix = prefix.parent
    if sys.platform == "win32":
        for folder in (prefix, prefix / "Library" / "bin"):
            if folder.is_dir():
                _prepend_path(environment, "PATH", folder)
        _prepend_path(environment, "PATH", executable_path.parent)
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


_DLL_SEARCH_LOCK = threading.RLock()


@contextmanager
def external_dll_search_path(executable: str | Path):
    """Remove the inherited frozen DLL directory only while spawning a child."""
    target = Path(executable).resolve()
    bundled = {Path(sys.executable).resolve(), Path(sys.executable).with_name("vtools-worker.exe").resolve()}
    if sys.platform != "win32" or not getattr(sys, "frozen", False) or target in bundled:
        yield
        return
    import ctypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetDllDirectoryW.argtypes = [ctypes.c_uint, ctypes.c_wchar_p]
    kernel.GetDllDirectoryW.restype = ctypes.c_uint
    kernel.SetDllDirectoryW.argtypes = [ctypes.c_wchar_p]
    kernel.SetDllDirectoryW.restype = ctypes.c_int
    with _DLL_SEARCH_LOCK:
        size = kernel.GetDllDirectoryW(0, None)
        buffer = ctypes.create_unicode_buffer(size + 1)
        kernel.GetDllDirectoryW(len(buffer), buffer)
        original = buffer.value or None
        if not kernel.SetDllDirectoryW(None):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield
        finally:
            if not kernel.SetDllDirectoryW(original):
                raise ctypes.WinError(ctypes.get_last_error())


__all__ = ["build_tool_environment", "external_dll_search_path"]

from pathlib import Path
import subprocess

from PyInstaller.building.build_main import Analysis, COLLECT, EXE, PYZ


ROOT = Path(SPECPATH).parent
SOURCE_DIRS = (
    "config", "model_diagnostics", "model_visualization", "speed_test",
    "model_compression", "transform_tools", "others", "env_test", "ultralytics-cn",
)
SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".cfg", ".txt", ".toml", ".typed"}
EXCLUDED_DIRS = {
    ".git", ".idea", ".vtools_ui", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "node_modules", "runs", "runs-profile",
    "outputs", "tests", "weights", "build", "dist",
}


def source_datas():
    """Bundle versioned tool sources, never local datasets or private files."""

    datas = [(str(ROOT / "run_tools.py"), ".")]
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", "--", *SOURCE_DIRS],
        check=True, capture_output=True,
    ).stdout.decode("utf-8").split("\0")
    tracked_files = {ROOT / relative for relative in tracked if relative}
    for directory in SOURCE_DIRS:
        for path in (ROOT / directory).rglob("*"):
            relative = path.relative_to(ROOT)
            if path not in tracked_files or not path.is_file() or any(part in EXCLUDED_DIRS for part in relative.parts[1:]):
                continue
            if path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            datas.append((str(path), relative.parent.as_posix()))
    frontend = ROOT / "vtools_ui" / "webapp" / "frontend" / "dist"
    for path in frontend.rglob("*"):
        if path.is_file():
            datas.append((str(path), path.relative_to(ROOT).parent.as_posix()))
    return datas


# The desktop bundle owns the web workbench only. Model/data jobs continue to
# run through the Python interpreter selected in Settings, so the bundle does
# not copy the several-gigabyte torch/TensorRT/Ultralytics stack into every
# distribution. pywebview is imported lazily by the desktop host and must be
# listed explicitly for PyInstaller's module graph.
hiddenimports = ["webview"]
analysis = Analysis(
    [str(ROOT / "vtools_ui" / "__main__.py")],
    pathex=[str(ROOT), str(ROOT / "ultralytics-cn")],
    binaries=[],
    datas=source_datas(),
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest", "_pytest", "expecttest", "tests",
        "torch", "torchvision", "ultralytics", "cv2", "PIL",
        "matplotlib", "onnxruntime", "tensorrt", "cupy",
    ],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    [],
    name="vtools",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    exclude_binaries=True,
)
app = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    name="vtools",
)

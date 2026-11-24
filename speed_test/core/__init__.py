"""测速工具的核心配置、计时、checkpoint 和运行目录组件。"""

import sys
from pathlib import Path

if __name__ == "core":
    # Direct speed_test scripts and existing adapters still use the short package name.
    repository_root = str(Path(__file__).resolve().parents[2])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

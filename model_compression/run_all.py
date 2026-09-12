#!/usr/bin/env python3
"""模型压缩工具的一键兼容入口。"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model_compression.run_model_compression import main
else:
    from .run_model_compression import main


if __name__ == "__main__":
    raise SystemExit(main())

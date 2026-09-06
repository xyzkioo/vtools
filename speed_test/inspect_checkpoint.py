#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立运行 checkpoint 架构检查；详细实现见 core/vision_checkpoint_inspector.py。"""

if __name__ == "__main__":
    from core.vision_runtime import ensure_conda_library_path

    ensure_conda_library_path()
    from core.vision_checkpoint_inspector import main

    main()

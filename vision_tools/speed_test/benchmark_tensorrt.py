#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyCharm 直接运行入口：读取 benchmark_config.yaml 执行 TensorRT 测速。"""

if __name__ == "__main__":
    from core.vision_runtime import ensure_conda_library_path

    ensure_conda_library_path()
    from backends.pytorch_vision_tensorrt_benchmark_v2 import main

    main()

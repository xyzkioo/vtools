#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyCharm 直接运行入口：读取 benchmark_config.yaml 执行 TensorRT 测速。"""

from backends.pytorch_vision_tensorrt_benchmark_v2 import main


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TensorRT 测速入口：读取 benchmark_config.yaml 执行测速。"""

if __name__ == "__main__":
    from core.vision_runtime import ensure_conda_library_path

    ensure_conda_library_path()
    from backends.pytorch_vision_tensorrt_benchmark_v2 import main

    rows = main()
    raise SystemExit(1 if any(row.get("status") in {"failed", "error"} for row in rows) else 0)

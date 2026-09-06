#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyCharm 直接运行：读取同目录 YAML，检查 PyTorch/TensorRT 输出一致性。"""

if __name__ == "__main__":
    # 必须在 checks.vision_consistency 延迟导入 onnx 前准备动态库路径。
    from core.vision_runtime import ensure_conda_library_path

    ensure_conda_library_path()
    from checks.vision_consistency import main

    report = main()
    # 失败、环境错误、排序差异或证据不足均不作为自动化验收通过。
    raise SystemExit(0 if report["status"] == "passed" else 1)

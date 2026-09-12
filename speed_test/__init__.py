"""测速工具包。

保留 ``core``/``backends`` 的直接脚本兼容导入，同时支持
``python -m speed_test.run_all`` 这类包方式。
"""

from pathlib import Path
import sys

_PACKAGE_DIR = Path(__file__).resolve().parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))


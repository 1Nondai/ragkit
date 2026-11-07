"""让 `pytest` / 直接运行单测时都能 import 到 src/ragkit。"""

import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

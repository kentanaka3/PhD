"""
Pytest configuration and central test environment fixture initialization.
"""

import ctypes
import os
from pathlib import Path
import sys

# Ensure OGS/src is accessible on sys.path
TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
  sys.path.insert(0, str(SRC_DIR))

# Preload Conda environment libstdc++.so.6 if present to avoid GLIBCXX version mismatch
_conda_prefix = os.path.dirname(os.path.dirname(sys.executable))
_libstdcxx = os.path.join(_conda_prefix, "lib", "libstdc++.so.6")
if os.path.exists(_libstdcxx):
  try:
    ctypes.CDLL(_libstdcxx, mode=ctypes.RTLD_GLOBAL)
  except Exception:
    pass

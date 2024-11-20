#!/bin/python
import os
from pathlib import Path
PRJ_PATH = Path(os.path.dirname(__file__)).parent
SRC_PATH = os.path.join(PRJ_PATH, "src")
import sys
# Add to path
if SRC_PATH not in sys.path: sys.path.append(SRC_PATH)
import unittest

from constants import *
import catalogger as ctg

DATA_PATH = Path(PRJ_PATH, "data", "test")
TEST_PATH = Path(DATA_PATH, "waveforms")

if __name__ == "__main__": unittest.main()
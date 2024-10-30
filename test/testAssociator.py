#!/bin/python
import os
from pathlib import Path
import unittest.mock
PRJ_PATH = Path(os.path.dirname(__file__)).parent
SRC_PATH = os.path.join(PRJ_PATH, "src")
import sys
# Add to path
if SRC_PATH not in sys.path: sys.path.append(SRC_PATH)
import unittest
from Analyzer import *
import Picker as Pkr
from constants import *

BASE_PATH = Path(PRJ_PATH, "data")
DATA_PATH = Path(BASE_PATH, "test")
TEST_PATH = Path(DATA_PATH, "waveforms")



if __name__ == "__main__": unittest.main()
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
from parser import *

BASE_PATH = Path(PRJ_PATH, "data")
DATA_PATH = Path(BASE_PATH, "test")
TEST_PATH = Path(DATA_PATH, "waveforms")
MNL_DATA_PATH = Path(DATA_PATH, "manual")

FILENAME = "RSFVG-2023"

class TestParser(unittest.TestCase):
  def test_event_parser_dat(self):
    filename = Path(MNL_DATA_PATH, FILENAME + DAT_EXT)
    data = event_parser_dat(filename)

  def test_event_parser_pun(self):
    filename = Path(MNL_DATA_PATH, FILENAME + PUN_EXT)
    data = event_parser_pun(filename)

  def test_event_parser_qml(self):
    filename = Path(MNL_DATA_PATH, FILENAME + QML_EXT)
    data = event_parser_qml(filename)
    print(data)

if __name__ == "__main__": unittest.main()
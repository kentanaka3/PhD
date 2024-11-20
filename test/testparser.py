#!/bin/python
import os
from pathlib import Path
PRJ_PATH = Path(os.path.dirname(__file__)).parent
INC_PATH = os.path.join(PRJ_PATH, "inc")
import sys
# Add to path
if INC_PATH not in sys.path: sys.path.append(INC_PATH)
import unittest

from parser import *

DATA_PATH = Path(PRJ_PATH, "data", "test")
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

if __name__ == "__main__": unittest.main()
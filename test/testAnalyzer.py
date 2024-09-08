#!/bin/python
import os
from pathlib import Path
import unittest.mock
PRJ_PATH = Path(os.path.dirname(__file__)).parent
SRC_PATH = os.path.join(PRJ_PATH, "src")
import sys
# Add to path
if SRC_PATH not in sys.path: sys.path.append(SRC_PATH)
import json
import shutil
import unittest
from Analyzer import *
import AdriaArray as AA
from constants import *
from datetime import timedelta as td

BASE_PATH = Path(PRJ_PATH, "data")
DATA_PATH = Path(BASE_PATH, "test")
TEST_PATH = Path(DATA_PATH, "waveforms")
MNL_DATA_PATH = Path(DATA_PATH, "manual")

EXPECTED_STR = "expected"

def timedeltafmt(string):
  numbers = [float(n) for n in string.split(":")]
  return td(hours=numbers[0], minutes=numbers[1], seconds=numbers[2])

class TestPickParser(unittest.TestCase):
  def test_parse_pick(self):
    events = event_parser(Path(MNL_DATA_PATH, "manual.dat"))
    # with open(Path(MNL_DATA_PATH, EXPECTED_STR + JSON_EXT), 'w') as fp:
    #   json.dump(events, fp, default=str)
    with open(Path(MNL_DATA_PATH, EXPECTED_STR + PERIOD_STR + JSON_EXT), 'r') as fr:
      expected = json.load(fr)
    for key, event in events.items():
      for s, station in enumerate(event):
        expected[str(key)][s][BEG_DATE_STR] = \
          UTCDateTime(expected[str(key)][s][BEG_DATE_STR])
        if expected[str(key)][s][P_TIME_STR] is not None:
          expected[str(key)][s][P_TIME_STR] = \
            timedeltafmt(expected[str(key)][s][P_TIME_STR])
        if expected[str(key)][s][S_TIME_STR] is not None:
          expected[str(key)][s][S_TIME_STR] = \
            timedeltafmt(expected[str(key)][s][S_TIME_STR])
    for key, event in events.items():
      for s, station in enumerate(event):
        for k, v in station.items():
          self.assertEqual(v, expected[str(key)][s][k])

class TestEventCounter(unittest.TestCase):
  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-D", "230601", "230605", "-v",
                        "-d", TEST_PATH.__str__(), "-M", PHASENET_STR])
  def test_event_counter(self):
    args = AA.parse_arguments()
    DATA = load_data(args)
    plot_data(DATA, args)

class TestEventMerger(unittest.TestCase):
  def test_event_merger(self):
    TRUE = [UTCDateTime(2023, 6, 1, 0, 0), UTCDateTime(2023, 6, 1, 0, 2),
            UTCDateTime(2023, 6, 1, 0, 4), UTCDateTime(2023, 6, 1, 0, 6)]
    PRED = [UTCDateTime(2023, 6, 1, 0, 0), UTCDateTime(2023, 6, 1, 0, 1),
            UTCDateTime(2023, 6, 1, 0, 2), UTCDateTime(2023, 6, 1, 0, 3),
            UTCDateTime(2023, 6, 1, 0, 5), UTCDateTime(2023, 6, 1, 0, 7)]
    EXPECTED = [(UTCDateTime(2023, 6, 1, 0, 0),  1.0),  # TP
                (UTCDateTime(2023, 6, 1, 0, 1), -1.0),  # FP
                (UTCDateTime(2023, 6, 1, 0, 2),  1.0),  # TP
                (UTCDateTime(2023, 6, 1, 0, 3), -1.0),  # FP
                (UTCDateTime(2023, 6, 1, 0, 4),  0.0),  # TN
                (UTCDateTime(2023, 6, 1, 0, 5), -1.0),  # FP
                (UTCDateTime(2023, 6, 1, 0, 6),  0.0),  # TN
                (UTCDateTime(2023, 6, 1, 0, 7), -1.0)]  # FP
    self.assertEqual(event_merger(TRUE, PRED, PICK_OFFSET), EXPECTED)

if __name__ == "__main__":
  unittest.main()
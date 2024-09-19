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
  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-D", "230601", "230604", "-v",
                        "-d", TEST_PATH.__str__()])
  def test_parse_pick(self):
    args = AA.parse_arguments()
    events = event_parser(Path(MNL_DATA_PATH, "manual.dat"), args)
    # with open(Path(MNL_DATA_PATH, EXPECTED_STR + JSON_EXT), 'w') as fp:
    #   json.dump(events, fp, default=str)
    with open(Path(MNL_DATA_PATH, EXPECTED_STR + JSON_EXT), 'r') as fr:
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
    #                   YEAR, M, D, H, M, S, MS
    TRUE = [UTCDateTime(2023, 6, 1, 0, 0, 0, 0),
            UTCDateTime(2023, 6, 1, 0, 2, 0, 0),
            UTCDateTime(2023, 6, 1, 0, 4, 0, 0),
            UTCDateTime(2023, 6, 1, 0, 6, 0, 0)]
    PRED = [UTCDateTime(2023, 6, 1, 0, 0, 0, 400),
            UTCDateTime(2023, 6, 1, 0, 1, 0, 0),
            UTCDateTime(2023, 6, 1, 0, 2, 0, 0),
            UTCDateTime(2023, 6, 1, 0, 3, 0, 0),
            UTCDateTime(2023, 6, 1, 0, 5, 0, 0),
            UTCDateTime(2023, 6, 1, 0, 7, 0, 0)]
    EXPECTED = [
      (UTCDateTime(2023, 6, 1, 0, 0, 0, 0),  1.0,
       UTCDateTime(2023, 6, 1, 0, 0, 0, 400), None),  # TP
      (UTCDateTime(2023, 6, 1, 0, 1, 0, 0), -1.0,
        None, None),                                  # FP
      (UTCDateTime(2023, 6, 1, 0, 2, 0, 0),  1.0,
       UTCDateTime(2023, 6, 1, 0, 2, 0, 0), None),    # TP
      (UTCDateTime(2023, 6, 1, 0, 3, 0, 0), -1.0,
       None, None),                                   # FP
      (UTCDateTime(2023, 6, 1, 0, 4, 0, 0),  0.0,
       None, None),                                   # FN
      (UTCDateTime(2023, 6, 1, 0, 5, 0, 0), -1.0,
       None, None),                                   # FP
      (UTCDateTime(2023, 6, 1, 0, 6, 0, 0),  0.0,
       None, None),                                   # FN
      (UTCDateTime(2023, 6, 1, 0, 7, 0, 0), -1.0,
       None, None)]                                   # FP
    self.assertEqual(event_merger(TRUE, PRED, PICK_OFFSET), EXPECTED)

  def test_event_merger_empty_anchor(self):
    TRUE = []
    #                   YEAR, M, D, H, M, S, MS
    PRED = [UTCDateTime(2023, 6, 1, 0, 0, 0, 0),
            UTCDateTime(2023, 6, 1, 0, 1, 0, 0),
            UTCDateTime(2023, 6, 1, 0, 2, 0, 0),
            UTCDateTime(2023, 6, 1, 0, 3, 0, 0),
            UTCDateTime(2023, 6, 1, 0, 5, 0, 0),
            UTCDateTime(2023, 6, 1, 0, 7, 0, 0)]
    EXPECTED = [(UTCDateTime(2023, 6, 1, 0, 0, 0, 0), -1.0, None, None),
                (UTCDateTime(2023, 6, 1, 0, 1, 0, 0), -1.0, None, None),
                (UTCDateTime(2023, 6, 1, 0, 2, 0, 0), -1.0, None, None),
                (UTCDateTime(2023, 6, 1, 0, 3, 0, 0), -1.0, None, None),
                (UTCDateTime(2023, 6, 1, 0, 5, 0, 0), -1.0, None, None),
                (UTCDateTime(2023, 6, 1, 0, 7, 0, 0), -1.0, None, None)]
    self.assertEqual(event_merger(TRUE, PRED, PICK_OFFSET), EXPECTED)

  def test_event_merger_empty_pred(self):
    #                   YEAR, M, D, H, M, S, MS
    TRUE = [UTCDateTime(2023, 6, 1, 0, 0, 0, 0),
            UTCDateTime(2023, 6, 1, 0, 2, 0, 0),
            UTCDateTime(2023, 6, 1, 0, 4, 0, 0),
            UTCDateTime(2023, 6, 1, 0, 6, 0, 0)]
    PRED = []
    EXPECTED = [(UTCDateTime(2023, 6, 1, 0, 0, 0, 0), 0.0, None, None),
                (UTCDateTime(2023, 6, 1, 0, 2, 0, 0), 0.0, None, None),
                (UTCDateTime(2023, 6, 1, 0, 4, 0, 0), 0.0, None, None),
                (UTCDateTime(2023, 6, 1, 0, 6, 0, 0), 0.0, None, None)]
    self.assertEqual(event_merger(TRUE, PRED, PICK_OFFSET), EXPECTED)

class TestConfMtx(unittest.TestCase):
  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-D", "230601", "230604", "-v",
                        "-d", TEST_PATH.__str__()])
  def test_conf_mtx(self):
    args = AA.parse_arguments()
    TRUE = event_parser(Path(DATA_PATH, "manual", "manual.dat"), args)
    PRED = load_data(args)
    EXPECTED = pd.DataFrame(
      [[EQTRANSFORMER_STR, INSTANCE_STR, 13, 134, 0, 345453,
        0.999612, 0.088435, 1.0, 0.1625],
       [EQTRANSFORMER_STR, ORIGINAL_STR, 12, 32, 1, 345555,
        0.999905, 0.272727, 0.923077, 0.421053],
       [EQTRANSFORMER_STR, SCEDC_STR, 13, 2424, 0, 343163,
        0.992986, 0.005334, 1.0, 0.010612],
       [EQTRANSFORMER_STR, STEAD_STR, 10, 59, 3, 345528,
        0.999821, 0.144928, 0.769231, 0.243902],
       [PHASENET_STR, INSTANCE_STR, 13, 114, 0, 345473,
        0.999670, 0.102362, 1.0, 0.185714],
       [PHASENET_STR, ORIGINAL_STR, 13, 512, 0, 345075,
        0.998519, 0.024762, 1.0, 0.048327],
       [PHASENET_STR, SCEDC_STR, 11, 1065, 2, 344522,
        0.996913, 0.010223, 0.846154, 0.020202],
       [PHASENET_STR, STEAD_STR, 9, 48, 4, 345539,
        0.999850, 0.157895, 0.692308, 0.257143]],
      columns=[MODEL_STR, WEIGHT_STR, TP_STR, FP_STR, FN_STR, TN_STR,
               ACCURACY_STR, PRECISION_STR, RECALL_STR, F1_STR])
    # self.assertAlmostEqual(conf_mtx(TRUE, PRED, args), EXPECTED)

if __name__ == "__main__":
  unittest.main()
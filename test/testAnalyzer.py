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
import Picker as Pkr
from constants import *
from datetime import timedelta as td

BASE_PATH = Path(PRJ_PATH, "data")
DATA_PATH = Path(BASE_PATH, "test")
TEST_PATH = Path(DATA_PATH, "waveforms")
MNL_DATA_PATH = Path(DATA_PATH, "manual")

EXPECTED_STR = "expected"

TRUE_HEADER = [STATION_STR, TIMESTAMP_STR, PHASE_STR, WEIGHT_STR]
PRED_HEADER = [MODEL_STR, WEIGHT_STR, TIMESTAMP_STR, STATION_STR, PHASE_STR,
               PROBABILITY_STR]

class TestPickParser(unittest.TestCase):
  @unittest.mock.patch("sys.argv",
                       ["Analyzer.py", "-D", "230601", "230604", "-v", "-d",
                        TEST_PATH.__str__()])
  def test_parse_pick(self):
    args = Pkr.parse_arguments()
    PRED = Pkr.load_data(args)
    stations = args.station if (args.station is not None and
                                args.station != ALL_WILDCHAR_STR) else \
               PRED[STATION_STR].unique()
    TRUE = event_parser(Path(DATA_PATH, "manual.dat"), stations, args)

class TestEventCounter(unittest.TestCase):
  @unittest.mock.patch("sys.argv",
                       ["Analyzer.py", "-D", "230601", "230605", "-v", "-d",
                        TEST_PATH.__str__(), "-M", PHASENET_STR])
  def test_event_counter(self):
    args = Pkr.parse_arguments()
    PRED = Pkr.load_data(args)
    stations = args.station if (args.station is not None and
                                args.station != ALL_WILDCHAR_STR) else \
               PRED[STATION_STR].unique()
    TRUE = event_parser(Path(DATA_PATH, "manual.dat"), stations, args)
    plot_data(TRUE, PRED, args)

class TestConfMtx(unittest.TestCase):
  @unittest.mock.patch("sys.argv",
                       ["Analyzer.py", "-D", "230601", "230601", "-v", "-d",
                        TEST_PATH.__str__()])
  def test_perfect(self):
    """
    OFFSET = 0.5 : |--"--|
           |------------------------------- 66 -------------------------------|
    TRUE : |----------------P---------------------------------S---------------|
                         |--"--|                           |--"--|
    PRED : |----------------P---------------------------------S---------------|
    OUTPUT: P [1, 0,  0]
            S [0, 1,  0]
            N [0, 0, 64]
               P  S   N
    """
    args = Pkr.parse_arguments()
    STATION = "EG"
    #        STATION              YEAR, M, D, H, M, S, mS  PHASE WEIGHT
    TRUE = [[STATION, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), PWAVE, 3],
            [STATION, UTCDateTime(2023, 6, 1, 0, 4, 5, 6), SWAVE, 0]]
    TRUE = pd.DataFrame(TRUE, columns=TRUE_HEADER)
    PRED = [[PHASENET_STR, ORIGINAL_STR, UTCDateTime(2023, 6, 1, 0, 1, 2, 3),
             STATION, PWAVE, 0.43185002],
            [PHASENET_STR, ORIGINAL_STR, UTCDateTime(2023, 6, 1, 0, 4, 5, 6),
             STATION, SWAVE, 0.3372562]]
    PRED = pd.DataFrame(PRED, columns=PRED_HEADER)
    THRESHOLD = 0.3
    CFN_MTX, TP, FN, FP = recall(TRUE, PRED, PHASENET_STR, ORIGINAL_STR,
                                 THRESHOLD, args)
    EXPECTED = [[1, 0, 0],
                [0, 1, 0],
                [0, 0, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = [[PHASENET_STR, ORIGINAL_STR, STATION, PWAVE, THRESHOLD,
                 (UTCDateTime(2023, 6, 1, 0, 1, 2, 3),
                  UTCDateTime(2023, 6, 1, 0, 1, 2, 3)), 3],
                [PHASENET_STR, ORIGINAL_STR, STATION, SWAVE, THRESHOLD,
                 (UTCDateTime(2023, 6, 1, 0, 4, 5, 6),
                  UTCDateTime(2023, 6, 1, 0, 4, 5, 6)), 0]]
    self.assertListEqual(EXPECTED, TP)
    EXPECTED = []
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = []
    self.assertListEqual(EXPECTED, FP)

  @unittest.mock.patch("sys.argv",
                       ["Analyzer.py", "-D", "230601", "230601", "-v", "-d",
                        TEST_PATH.__str__()])
  def test_flipped(self):
    """
    OFFSET = 0.5 : |--"--|
           |------------------------------- 66 -------------------------------|
    TRUE : |----------------P---------------------------------S---------------|
                         |--"--|                           |--"--|
    PRED : |----------------S---------------------------------P---------------|
    OUTPUT: P [0, 1,  0]
            S [1, 0,  0]
            N [0, 0, 64]
               P  S   N
    """
    args = Pkr.parse_arguments()
    STATION = "EG"
    #        STATION              YEAR, M, D, H, M, S, mS  PHASE WEIGHT
    TRUE = [[STATION, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), PWAVE, 3],
            [STATION, UTCDateTime(2023, 6, 1, 0, 4, 5, 6), SWAVE, 0]]
    TRUE = pd.DataFrame(TRUE, columns=TRUE_HEADER)
    PRED = [[PHASENET_STR, ORIGINAL_STR, UTCDateTime(2023, 6, 1, 0, 1, 2, 3),
             STATION, SWAVE, 0.43185002],
            [PHASENET_STR, ORIGINAL_STR, UTCDateTime(2023, 6, 1, 0, 4, 5, 6),
             STATION, PWAVE, 0.3372562]]
    PRED = pd.DataFrame(PRED, columns=PRED_HEADER)
    THRESHOLD = 0.3
    CFN_MTX, TP, FN, FP = recall(TRUE, PRED, PHASENET_STR, ORIGINAL_STR,
                                 THRESHOLD, args)
    EXPECTED = [[0, 1, 0],
                [1, 0, 0],
                [0, 0, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = []
    self.assertListEqual(EXPECTED, TP)
    EXPECTED = []
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = []
    self.assertListEqual(EXPECTED, FP)

  @unittest.mock.patch("sys.argv",
                       ["Analyzer.py", "-D", "230601", "230601", "-v", "-d",
                        TEST_PATH.__str__()])
  def test_pred_over_true(self):
    """
    OFFSET = 0.5 : |--"--|
           |------------------------------- 66 -------------------------------|
    TRUE : |--------------:-P---:-----------------------------S---------------|
                         |:-"--|:                          |--"--|
    PRED : |--------------:-PP--:-----------------------------S---------------|
                          |--"--|
    OUTPUT: P [1, 0,  0]
            S [0, 1,  0]
            N [1, 0, 63]
               P  S   N : PRED
    """
    args = Pkr.parse_arguments()
    STATION = "EG"
    #        STATION              YEAR, M, D, H, M, S, mS  PHASE WEIGHT
    TRUE = [[STATION, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), PWAVE, 3],
            [STATION, UTCDateTime(2023, 6, 1, 0, 4, 5, 6), SWAVE, 0]]
    TRUE = pd.DataFrame(TRUE, columns=TRUE_HEADER)
    PRED = [
      [PHASENET_STR, ORIGINAL_STR, UTCDateTime(2023, 6, 1, 0, 1, 2, 3),
       STATION, PWAVE, 0.43185002],
      [PHASENET_STR, ORIGINAL_STR, UTCDateTime(2023, 6, 1, 0, 1, 2, 4),
       STATION, PWAVE, 0.43185002],
      [PHASENET_STR, ORIGINAL_STR, UTCDateTime(2023, 6, 1, 0, 4, 5, 6),
       STATION, SWAVE, 0.3372562]]
    PRED = pd.DataFrame(PRED, columns=PRED_HEADER)
    THRESHOLD = 0.3
    CFN_MTX, TP, FN, FP = recall(TRUE, PRED, PHASENET_STR, ORIGINAL_STR,
                                 THRESHOLD, args)
    EXPECTED = [[1, 0, 0],
                [0, 1, 0],
                [1, 0, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = [[PHASENET_STR, ORIGINAL_STR, STATION, PWAVE, THRESHOLD,
                 (UTCDateTime(2023, 6, 1, 0, 1, 2, 3),
                  UTCDateTime(2023, 6, 1, 0, 1, 2, 3)), 3],
                 [PHASENET_STR, ORIGINAL_STR, STATION, SWAVE, THRESHOLD,
                  (UTCDateTime(2023, 6, 1, 0, 4, 5, 6),
                   UTCDateTime(2023, 6, 1, 0, 4, 5, 6)), 0]]
    self.assertListEqual(EXPECTED, TP)
    EXPECTED = []
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = [[PHASENET_STR, ORIGINAL_STR, STATION, PWAVE, THRESHOLD,
                 UTCDateTime(2023, 6, 1, 0, 1, 2, 4), 0.43185002]]
    self.assertListEqual(EXPECTED, FP)

  @unittest.mock.patch("sys.argv",
                       ["Analyzer.py", "-D", "230601", "230601", "-v", "-d",
                        TEST_PATH.__str__()])
  def test_true_over_pred(self):
    """
    OFFSET = 0.5 : |--"--|
           |------------------------------- 66 -------------------------------|
                         |--"--|
    TRUE : |--------------:-P---P---------------------------------------------|
                          :  |--"--|
    PRED : |--------------:--P--:-----------------------------S---------------|
                          |--"--|
    OUTPUT: P [1, 0,  1]
            S [0, 0,  0]
            N [0, 1, 63]
               P  S   N : PRED
    """
    args = Pkr.parse_arguments()
    STATION = "EG"
    #        STATION              YEAR, M, D, H, M, S, mS  PHASE WEIGHT
    TRUE = [[STATION, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), PWAVE, 3],
            [STATION, UTCDateTime(2023, 6, 1, 0, 1, 2, 4), PWAVE, 3]]
    TRUE = pd.DataFrame(TRUE, columns=TRUE_HEADER)
    PRED = [
      [PHASENET_STR, ORIGINAL_STR, UTCDateTime(2023, 6, 1, 0, 1, 2, 3),
       STATION, PWAVE, 0.43185002],
      [PHASENET_STR, ORIGINAL_STR, UTCDateTime(2023, 6, 1, 0, 4, 5, 6),
       STATION, SWAVE, 0.3372562]]
    PRED = pd.DataFrame(PRED, columns=PRED_HEADER)
    THRESHOLD = 0.3
    CFN_MTX, TP, FN, FP = recall(TRUE, PRED, PHASENET_STR, ORIGINAL_STR,
                                 THRESHOLD, args)
    EXPECTED = [[1, 0, 1],
                [0, 0, 0],
                [0, 1, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = [[PHASENET_STR, ORIGINAL_STR, STATION, PWAVE, THRESHOLD,
                 (UTCDateTime(2023, 6, 1, 0, 1, 2, 3),
                  UTCDateTime(2023, 6, 1, 0, 1, 2, 3)), 3]]
    self.assertListEqual(EXPECTED, TP)
    EXPECTED = [[PHASENET_STR, ORIGINAL_STR, STATION, PWAVE, THRESHOLD,
                 UTCDateTime(2023, 6, 1, 0, 1, 2, 4), 3]]
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = [[PHASENET_STR, ORIGINAL_STR, STATION, SWAVE, THRESHOLD,
                 UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0.3372562]]
    self.assertListEqual(EXPECTED, FP)

  @unittest.mock.patch("sys.argv",
                       ["Analyzer.py", "-D", "230601", "230601", "-v", "-d",
                        TEST_PATH.__str__()])
  def test_no_pred(self):
    """
    OFFSET = 0.5 : |--"--|
           |------------------------------- 66 -------------------------------|
    TRUE : |----------------P---------------------------------S---------------|
                         |--"--|
    PRED : |------------------------------------------------------------------|
    OUTPUT: P [0, 0,  1]
            S [0, 0,  1]
            N [0, 0, 64]
               P  S   N : PRED
    """
    args = Pkr.parse_arguments()
    STATION = "EG"
    #        STATION              YEAR, M, D, H, M, S, mS  PHASE WEIGHT
    TRUE = [[STATION, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), PWAVE, 3],
            [STATION, UTCDateTime(2023, 6, 1, 0, 4, 5, 6), SWAVE, 0]]
    TRUE = pd.DataFrame(TRUE, columns=TRUE_HEADER)
    PRED = []
    PRED = pd.DataFrame(PRED, columns=PRED_HEADER)
    THRESHOLD = 0.3
    CFN_MTX, TP, FN, FP = recall(TRUE, PRED, PHASENET_STR, ORIGINAL_STR,
                                 THRESHOLD, args)
    EXPECTED = [[0, 0, 1],
                [0, 0, 1],
                [0, 0, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = []
    self.assertListEqual(EXPECTED, TP)
    EXPECTED = [[PHASENET_STR, ORIGINAL_STR, STATION, PWAVE, THRESHOLD,
                 UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 3],
                [PHASENET_STR, ORIGINAL_STR, STATION, SWAVE, THRESHOLD,
                 UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0]]
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = []
    self.assertListEqual(EXPECTED, FP)

  @unittest.mock.patch("sys.argv",
                       ["Analyzer.py", "-D", "230601", "230601", "-v", "-d",
                        TEST_PATH.__str__()])
  def test_no_true(self):
    """
    OFFSET = 0.5 : |--"--|
           |------------------------------- 66 -------------------------------|
    TRUE : |------------------------------------------------------------------|
    PRED : |----------------P---------------------------------S---------------|
                         |--"--|
    OUTPUT: P [0, 0,  0]
            S [0, 0,  0]
            N [1, 1, 64]
               P  S   N : PRED
    """
    args = Pkr.parse_arguments()
    STATION = "EG"
    #        STATION              YEAR, M, D, H, M, S, mS  PHASE WEIGHT
    TRUE = []
    TRUE = pd.DataFrame(TRUE, columns=TRUE_HEADER)
    PRED = [
      [PHASENET_STR, ORIGINAL_STR, UTCDateTime(2023, 6, 1, 0, 1, 2, 3),
       STATION, PWAVE, 0.43185002],
      [PHASENET_STR, ORIGINAL_STR, UTCDateTime(2023, 6, 1, 0, 4, 5, 6),
       STATION, SWAVE, 0.3372562]]
    PRED = pd.DataFrame(PRED, columns=PRED_HEADER)
    THRESHOLD = 0.3
    CFN_MTX, TP, FN, FP = recall(TRUE, PRED, PHASENET_STR, ORIGINAL_STR,
                                 THRESHOLD, args)
    EXPECTED = [[0, 0, 0],
                [0, 0, 0],
                [1, 1, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = []
    self.assertListEqual(EXPECTED, TP)
    EXPECTED = []
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = [[PHASENET_STR, ORIGINAL_STR, STATION, PWAVE, THRESHOLD,
                 UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 0.43185002],
                [PHASENET_STR, ORIGINAL_STR, STATION, SWAVE, THRESHOLD,
                 UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0.3372562]]
    self.assertListEqual(EXPECTED, FP)

  @unittest.mock.patch("sys.argv",
                       ["Analyzer.py", "-D", "230601", "230601", "-v", "-d",
                        TEST_PATH.__str__()])
  def test_no_true_no_pred(self):
    """
    OFFSET = 0.5 : |--"--|
           |------------------------------- 66 -------------------------------|
    TRUE : |------------------------------------------------------------------|
    PRED : |------------------------------------------------------------------|
    OUTPUT: P [0, 0,  0]
            S [0, 0,  0]
            N [0, 0, 66]
               P  S   N : PRED
    """
    args = Pkr.parse_arguments()
    STATION = "EG"
    #        STATION              YEAR, M, D, H, M, S, mS  PHASE WEIGHT
    TRUE = []
    TRUE = pd.DataFrame(TRUE, columns=TRUE_HEADER)
    PRED = []
    PRED = pd.DataFrame(PRED, columns=PRED_HEADER)
    THRESHOLD = 0.3
    CFN_MTX, TP, FN, FP = recall(TRUE, PRED, PHASENET_STR, ORIGINAL_STR,
                                 THRESHOLD, args)
    EXPECTED = [[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = []
    self.assertListEqual(EXPECTED, TP)
    EXPECTED = []
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = []

  @unittest.mock.patch("sys.argv",
                       ["Analyzer.py", "-D", "230601", "230601", "-v", "-d",
                        TEST_PATH.__str__()])
  def test_true_pred(self):
    """
    OFFSET = 0.5 : |--"--|
           |------------------------------- 66 -------------------------------|
                         |--"--|
    TRUE : |---------------:P--:S:---:----------------------------------------|
                           : |-:":-| :
    PRED : |---------------:--P:-:S--:------------------------S---------------|
                           |--"--|
    OUTPUT: P [1, 0,  0]
            S [0, 1,  0]
            N [0, 1, 63]
               P  S   N : PRED
    """
    args = Pkr.parse_arguments()
    STATION = "EG"
    #        STATION              YEAR, M, D, H, M, S, mS  PHASE WEIGHT
    TRUE = [[STATION, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), PWAVE, 3],
            [STATION, UTCDateTime(2023, 6, 1, 0, 1, 2, 4), SWAVE, 0]]
    TRUE = pd.DataFrame(TRUE, columns=TRUE_HEADER)
    PRED = [
      [PHASENET_STR, ORIGINAL_STR, UTCDateTime(2023, 6, 1, 0, 1, 2, 3),
       STATION, PWAVE, 0.43185002],
      [PHASENET_STR, ORIGINAL_STR, UTCDateTime(2023, 6, 1, 0, 1, 2, 4),
       STATION, SWAVE, 0.3372562],
      [PHASENET_STR, ORIGINAL_STR, UTCDateTime(2023, 6, 1, 0, 4, 5, 6),
       STATION, SWAVE, 0.3372562]]
    PRED = pd.DataFrame(PRED, columns=PRED_HEADER)
    THRESHOLD = 0.3
    CFN_MTX, TP, FN, FP = recall(TRUE, PRED, PHASENET_STR, ORIGINAL_STR,
                                 THRESHOLD, args)
    EXPECTED = [[1, 0, 0],
                [0, 1, 0],
                [0, 1, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = [[PHASENET_STR, ORIGINAL_STR, STATION, PWAVE, THRESHOLD,
                 (UTCDateTime(2023, 6, 1, 0, 1, 2, 3),
                  UTCDateTime(2023, 6, 1, 0, 1, 2, 3)), 3],
                [PHASENET_STR, ORIGINAL_STR, STATION, SWAVE, THRESHOLD,
                 (UTCDateTime(2023, 6, 1, 0, 1, 2, 4),
                  UTCDateTime(2023, 6, 1, 0, 1, 2, 4)), 0]]
    self.assertListEqual(EXPECTED, TP)
    EXPECTED = []
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = [[PHASENET_STR, ORIGINAL_STR, STATION, SWAVE, THRESHOLD,
                 UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0.3372562]]
    self.assertListEqual(EXPECTED, FP)

if __name__ == "__main__":
  unittest.main()
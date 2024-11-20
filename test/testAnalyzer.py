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
MNL_DATA_PATH = Path(DATA_PATH, "manual")

EXPECTED_STR = "expected"

TRUE_HEADER = [STATION_STR, TIMESTAMP_STR, PHASE_STR, WEIGHT_STR]
PRED_HEADER = [MODEL_STR, WEIGHT_STR, TIMESTAMP_STR, STATION_STR, PHASE_STR,
               PROBABILITY_STR]

STATION = "EG"
THRESHOLD = 0.3

class TestPickParser(unittest.TestCase):
  @unittest.mock.patch("sys.argv",
                       ["Analyzer.py", "-d", TEST_PATH.__str__(),
                        "-D", "230601", "230604", "-v", "--file",
                        Path(MNL_DATA_PATH, "RSFVG-2023.dat").__str__()])
  def test_parse_pick(self):
    args = Pkr.parse_arguments()
    PRED = load_data(args)
    stations = args.station if (args.station is not None and
                                args.station != ALL_WILDCHAR_STR) else \
               PRED[STATION_STR].unique()
    TRUE = event_parser(args.file, stations, args)
    EXPECTED = [[2, "CAE", PWAVE, "2023-06-01T09:27:58.710000Z",0],
                [2, "CAE", SWAVE, "2023-06-01T09:27:59.360000Z",2],
                [3, "CAE", PWAVE, "2023-06-01T09:53:32.530000Z",2],
                [3, "CAE", SWAVE, "2023-06-01T09:53:36.360000Z",3],
                [4, "VARA", PWAVE, "2023-06-01T10:02:12.860000Z",3],
                [7, "CAE", PWAVE, "2023-06-01T21:41:16.740000Z",0],
                [7, "CAE", SWAVE, "2023-06-01T21:41:19.090000Z",0],
                [9, "VARA", PWAVE, "2023-06-02T02:18:27.430000Z",0],
                [9, "VARA", SWAVE, "2023-06-02T02:18:29.510000Z",1],
                [10, "TRI", PWAVE, "2023-06-02T08:50:44.010000Z",0],
                [10, "BAD", PWAVE, "2023-06-02T08:50:46.420000Z",2],
                [12, "BAD", PWAVE, "2023-06-03T00:31:40.050000Z",2],
                [12, "BAD", SWAVE, "2023-06-03T00:31:43.560000Z",1],
                [13, "VARA", PWAVE, "2023-06-03T05:18:36.010000Z",2],
                [13, "VARA", SWAVE, "2023-06-03T05:18:39.980000Z",2],
                [15, "VARA", PWAVE, "2023-06-03T12:33:23.350000Z",0],
                [15, "VARA", SWAVE, "2023-06-03T12:33:25.980000Z",0],
                [16, "BAD", PWAVE, "2023-06-03T16:54:16.760000Z",2],
                [16, "BAD", SWAVE, "2023-06-03T16:54:19.950000Z",2],
                [19, "BAD", PWAVE, "2023-06-04T00:03:05.450000Z",0],
                [19, "BAD", SWAVE, "2023-06-04T00:04:23.400000Z",2],
                [20, "TRI", PWAVE, "2023-06-04T00:25:09.150000Z",3],
                [20, "TRI", SWAVE, "2023-06-04T00:25:14.520000Z",3],
                [20, "BAD", PWAVE, "2023-06-04T00:25:10.760000Z",1],
                [20, "BAD", SWAVE, "2023-06-04T00:25:16.420000Z",2],
                [23, "CAE", PWAVE, "2023-06-04T17:57:11.390000Z",3],
                [23, "CAE", SWAVE, "2023-06-04T17:57:17.250000Z",2]]
    self.assertListEqual(EXPECTED, TRUE.values.tolist())

class TestEventCounter(unittest.TestCase):
  @unittest.mock.patch("sys.argv",
                       ["Analyzer.py", "-D", "230601", "230604", "-v", "-d",
                        TEST_PATH.__str__(), "-M", PHASENET_STR, "--file",
                        Path(MNL_DATA_PATH, "RSFVG-2023.dat").__str__()])
  def test_event_counter(self):
    args = Pkr.parse_arguments()
    PRED = load_data(args)
    stations = args.station if (args.station is not None and
                                args.station != ALL_WILDCHAR_STR) else \
               PRED[STATION_STR].unique()
    TRUE = event_parser(args.file, stations, args)
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
    OUTPUT: P [1, 0, 0]
            S [0, 1, 0]
            N [0, 0, 0]
               P  S  N : PRED
    """
    args = Pkr.parse_arguments()
    #        STATION              YEAR, M, D, H, M, S, mS  PHASE WEIGHT
    TRUE = [[STATION, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), PWAVE, 3],
            [STATION, UTCDateTime(2023, 6, 1, 0, 4, 5, 6), SWAVE, 0]]
    TRUE = pd.DataFrame(TRUE, columns=TRUE_HEADER)
    PRED = [[PHASENET_STR, ORIGINAL_STR, UTCDateTime(2023, 6, 1, 0, 1, 2, 3),
             STATION, PWAVE, 0.43185002],
            [PHASENET_STR, ORIGINAL_STR, UTCDateTime(2023, 6, 1, 0, 4, 5, 6),
             STATION, SWAVE, 0.3372562]]
    PRED = pd.DataFrame(PRED, columns=PRED_HEADER)
    CFN_MTX, TP, FN, FP = conf_mtx(TRUE, PRED, PHASENET_STR, ORIGINAL_STR,
                                   THRESHOLD, args)
    EXPECTED = [[1, 0, 0],
                [0, 1, 0],
                [0, 0, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = set()
    EXPECTED.add((PHASENET_STR, ORIGINAL_STR, STATION, PWAVE,
                  (str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3)),
                   str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3))), (3, 0.43185002)))
    EXPECTED.add((PHASENET_STR, ORIGINAL_STR, STATION, SWAVE,
                  (str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6)),
                   str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6))), (0, 0.3372562)))
    self.assertSetEqual(EXPECTED, TP)
    EXPECTED = []
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = set()
    self.assertSetEqual(EXPECTED, FP)

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
    OUTPUT: P [0, 1, 0]
            S [1, 0, 0]
            N [0, 0, 0]
               P  S  N : PRED
    """
    args = Pkr.parse_arguments()
    #        STATION              YEAR, M, D, H, M, S, mS  PHASE WEIGHT
    TRUE = [[STATION, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), PWAVE, 3],
            [STATION, UTCDateTime(2023, 6, 1, 0, 4, 5, 6), SWAVE, 0]]
    TRUE = pd.DataFrame(TRUE, columns=TRUE_HEADER)
    PRED = [[PHASENET_STR, ORIGINAL_STR, UTCDateTime(2023, 6, 1, 0, 1, 2, 3),
             STATION, SWAVE, 0.43185002],
            [PHASENET_STR, ORIGINAL_STR, UTCDateTime(2023, 6, 1, 0, 4, 5, 6),
             STATION, PWAVE, 0.3372562]]
    PRED = pd.DataFrame(PRED, columns=PRED_HEADER)
    CFN_MTX, TP, FN, FP = conf_mtx(TRUE, PRED, PHASENET_STR, ORIGINAL_STR,
                                   THRESHOLD, args)
    EXPECTED = [[0, 1, 0],
                [1, 0, 0],
                [0, 0, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = set()
    self.assertSetEqual(EXPECTED, TP)
    EXPECTED = []
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = set()
    self.assertSetEqual(EXPECTED, FP)

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
    OUTPUT: P [1, 0, 0]
            S [0, 1, 0]
            N [1, 0, 0]
               P  S  N : PRED
    """
    args = Pkr.parse_arguments()
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
    CFN_MTX, TP, FN, FP = conf_mtx(TRUE, PRED, PHASENET_STR, ORIGINAL_STR,
                                   THRESHOLD, args)
    EXPECTED = [[1, 0, 0],
                [0, 1, 0],
                [1, 0, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = set()
    EXPECTED.add((PHASENET_STR, ORIGINAL_STR, STATION, PWAVE,
                  (str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3)),
                   str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3))), (3, 0.43185002)))
    EXPECTED.add((PHASENET_STR, ORIGINAL_STR, STATION, SWAVE,
                  (str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6)),
                   str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6))), (0, 0.3372562)))
    self.assertSetEqual(EXPECTED, TP)
    EXPECTED = []
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = set()
    EXPECTED.add((PHASENET_STR, ORIGINAL_STR, STATION, PWAVE,
                  str(UTCDateTime(2023, 6, 1, 0, 1, 2, 4)), 0.43185002))
    self.assertSetEqual(EXPECTED, FP)

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
                          |--"--|                          |--"--|
    OUTPUT: P [1, 0, 1]
            S [0, 0, 0]
            N [0, 1, 0]
               P  S  N : PRED
    """
    args = Pkr.parse_arguments()
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
    CFN_MTX, TP, FN, FP = conf_mtx(TRUE, PRED, PHASENET_STR, ORIGINAL_STR,
                                   THRESHOLD, args)
    EXPECTED = [[1, 0, 1],
                [0, 0, 0],
                [0, 1, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = set()
    EXPECTED.add((PHASENET_STR, ORIGINAL_STR, STATION, PWAVE,
                  (str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3)),
                   str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3))), (3, 0.43185002)))
    self.assertSetEqual(EXPECTED, TP)
    EXPECTED = [[PHASENET_STR, ORIGINAL_STR, STATION, PWAVE, THRESHOLD,
                 UTCDateTime(2023, 6, 1, 0, 1, 2, 4), 3]]
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = set()
    EXPECTED.add((PHASENET_STR, ORIGINAL_STR, STATION, SWAVE,
                  str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6)), 0.3372562))
    self.assertSetEqual(EXPECTED, FP)

  @unittest.mock.patch("sys.argv",
                       ["Analyzer.py", "-D", "230601", "230601", "-v", "-d",
                        TEST_PATH.__str__()])
  def test_no_pred(self):
    """
    OFFSET = 0.5 : |--"--|
           |------------------------------- 66 -------------------------------|
    TRUE : |----------------P---------------------------------S---------------|
                         |--"--|                           |--"--|
    PRED : |------------------------------------------------------------------|
    OUTPUT: P [0, 0, 1]
            S [0, 0, 1]
            N [0, 0, 0]
               P  S  N : PRED
    """
    args = Pkr.parse_arguments()
    #        STATION              YEAR, M, D, H, M, S, mS  PHASE WEIGHT
    TRUE = [[STATION, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), PWAVE, 3],
            [STATION, UTCDateTime(2023, 6, 1, 0, 4, 5, 6), SWAVE, 0]]
    TRUE = pd.DataFrame(TRUE, columns=TRUE_HEADER)
    PRED = []
    PRED = pd.DataFrame(PRED, columns=PRED_HEADER)
    CFN_MTX, TP, FN, FP = conf_mtx(TRUE, PRED, PHASENET_STR, ORIGINAL_STR,
                                   THRESHOLD, args)
    EXPECTED = [[0, 0, 1],
                [0, 0, 1],
                [0, 0, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = set()
    self.assertSetEqual(EXPECTED, TP)
    EXPECTED = [[PHASENET_STR, ORIGINAL_STR, STATION, PWAVE, THRESHOLD,
                 UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 3],
                [PHASENET_STR, ORIGINAL_STR, STATION, SWAVE, THRESHOLD,
                 UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0]]
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = set()
    self.assertSetEqual(EXPECTED, FP)

  @unittest.mock.patch("sys.argv",
                       ["Analyzer.py", "-D", "230601", "230601", "-v", "-d",
                        TEST_PATH.__str__()])
  def test_no_true(self):
    """
    OFFSET = 0.5 : |--"--|
           |------------------------------- 66 -------------------------------|
    TRUE : |------------------------------------------------------------------|
    PRED : |----------------P---------------------------------S---------------|
                         |--"--|                           |--"--|
    OUTPUT: P [0, 0, 0]
            S [0, 0, 0]
            N [1, 1, 0]
               P  S  N : PRED
    """
    args = Pkr.parse_arguments()
    #        STATION              YEAR, M, D, H, M, S, mS  PHASE WEIGHT
    TRUE = []
    TRUE = pd.DataFrame(TRUE, columns=TRUE_HEADER)
    PRED = [
      [PHASENET_STR, ORIGINAL_STR, UTCDateTime(2023, 6, 1, 0, 1, 2, 3),
       STATION, PWAVE, 0.43185002],
      [PHASENET_STR, ORIGINAL_STR, UTCDateTime(2023, 6, 1, 0, 4, 5, 6),
       STATION, SWAVE, 0.3372562]]
    PRED = pd.DataFrame(PRED, columns=PRED_HEADER)
    CFN_MTX, TP, FN, FP = conf_mtx(TRUE, PRED, PHASENET_STR, ORIGINAL_STR,
                                   THRESHOLD, args)
    EXPECTED = [[0, 0, 0],
                [0, 0, 0],
                [1, 1, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = set()
    self.assertSetEqual(EXPECTED, TP)
    EXPECTED = []
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = set()
    EXPECTED.add((PHASENET_STR, ORIGINAL_STR, STATION, PWAVE,
                  str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3)), 0.43185002))
    EXPECTED.add((PHASENET_STR, ORIGINAL_STR, STATION, SWAVE,
                  str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6)), 0.3372562))
    self.assertSetEqual(EXPECTED, FP)

  @unittest.mock.patch("sys.argv",
                       ["Analyzer.py", "-D", "230601", "230601", "-v", "-d",
                        TEST_PATH.__str__()])
  def test_no_true_no_pred(self):
    """
    OFFSET = 0.5 : |--"--|
           |------------------------------- 66 -------------------------------|
    TRUE : |------------------------------------------------------------------|
    PRED : |------------------------------------------------------------------|
    OUTPUT: P [0, 0, 0]
            S [0, 0, 0]
            N [0, 0, 0]
               P  S  N : PRED
    """
    args = Pkr.parse_arguments()
    #        STATION              YEAR, M, D, H, M, S, mS  PHASE WEIGHT
    TRUE = []
    TRUE = pd.DataFrame(TRUE, columns=TRUE_HEADER)
    PRED = []
    PRED = pd.DataFrame(PRED, columns=PRED_HEADER)
    CFN_MTX, TP, FN, FP = conf_mtx(TRUE, PRED, PHASENET_STR, ORIGINAL_STR,
                                   THRESHOLD, args)
    EXPECTED = [[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = set()
    self.assertSetEqual(EXPECTED, TP)
    EXPECTED = []
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = set()
    self.assertSetEqual(EXPECTED, FP)

  @unittest.mock.patch("sys.argv",
                       ["Analyzer.py", "-D", "230601", "230601", "-v", "-d",
                        TEST_PATH.__str__()])
  def test_true_pred(self):
    """
    OFFSET = 0.5 : |--"--|
           |------------------------------- 66 -------------------------------|
                         |--"--|
    TRUE : |----------------P--:S-:--:----------------------------------------|
                            :|-:"-:| :
    PRED : |----------------:--P--:S-:------------------------S---------------|
                            |--"--|                        |--"--|
    OUTPUT: P [1, 0, 0]
            S [0, 1, 0]
            N [0, 1, 0]
               P  S  N : PRED
    """
    args = Pkr.parse_arguments()
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
    CFN_MTX, TP, FN, FP = conf_mtx(TRUE, PRED, PHASENET_STR, ORIGINAL_STR,
                                   THRESHOLD, args)
    EXPECTED = [[1, 0, 0],
                [0, 1, 0],
                [0, 1, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = set()
    EXPECTED.add((PHASENET_STR, ORIGINAL_STR, STATION, PWAVE,
                  (str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3)),
                   str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3))), (3, 0.43185002)))
    EXPECTED.add((PHASENET_STR, ORIGINAL_STR, STATION, SWAVE,
                  (str(UTCDateTime(2023, 6, 1, 0, 1, 2, 4)),
                   str(UTCDateTime(2023, 6, 1, 0, 1, 2, 4))), (0, 0.3372562)))
    self.assertSetEqual(EXPECTED, TP)
    EXPECTED = []
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = set()
    EXPECTED.add((PHASENET_STR, ORIGINAL_STR, STATION, SWAVE,
                  str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6)), 0.3372562))
    self.assertSetEqual(EXPECTED, FP)

  @unittest.mock.patch("sys.argv",
                       ["Analyzer.py", "-D", "230601", "230601", "-v", "-d",
                        TEST_PATH.__str__()])
  def test_true_pred_complex(self):
    """
    OFFSET = 0.5 : |--"--|
           |------------------------------- 66 -------------------------------|
                         |--"--|
    TRUE : |----------------P--:S-:--:----------------------------------------|
                            :|-:"-:| :
    PRED : |----------------:--P--:S-:------------------------S---------------|
                            |--"--|                        |--"--|
    OUTPUT: P [1, 0, 0]
            S [0, 1, 0]
            N [0, 1, 0]
               P  S  N : PRED
    """
    args = Pkr.parse_arguments()
    #        STATION              YEAR, M, D, H, M, S, mS  PHASE WEIGHT

if __name__ == "__main__": unittest.main()
#!/bin/python
import os
from pathlib import Path
PRJ_PATH = Path(os.path.dirname(__file__)).parent
SRC_PATH = os.path.join(PRJ_PATH, "src")
import sys
# Add to path
if SRC_PATH not in sys.path: sys.path.append(SRC_PATH)
import unittest

from analyzer import *
import initializer as ini

DATA_PATH = Path(PRJ_PATH, "data", "test")
TEST_PATH = Path(DATA_PATH, "waveforms")
MNL_DATA_PATH = Path(DATA_PATH, "manual")

EXPECTED_STR = "expected"

STATION = "EG"
THRESHOLD = 0.3

class TestPickParser(unittest.TestCase):
  @unittest.mock.patch("sys.argv",
                       ["Analyzer.py", "-d", TEST_PATH.__str__(),
                        "-D", "230601", "230604", "-v", "--file",
                        Path(MNL_DATA_PATH, "RSFVG-2023.dat").__str__()])
  def test_parse_pick(self):
    args = ini.parse_arguments()
    PRED = ini.classified_loader(args)
    stations = args.station if (args.station is not None and
                                args.station != ALL_WILDCHAR_STR) else \
               PRED[STATION_STR].unique()
    SOURCE, DETECT = event_parser(args.file, args)
    DETECT = DETECT.values.tolist()
    EXPECTED = [
      [None, UTCDateTime(2023, 6, 1, 9, 27, 58, 710000), 0, 'P', 'CAE'],
      [None, UTCDateTime(2023, 6, 1, 9, 27, 59, 360000), 2, 'S', 'CAE'],
      [None, UTCDateTime(2023, 6, 1, 9, 53, 32, 530000), 2, 'P', 'CAE'],
      [None, UTCDateTime(2023, 6, 1, 9, 53, 36, 360000), 3, 'S', 'CAE'],
      [None, UTCDateTime(2023, 6, 1, 10, 2, 12, 860000), 3, 'P', 'VARA'],
      [None, UTCDateTime(2023, 6, 1, 21, 41, 16, 740000), 0, 'P', 'CAE'],
      [None, UTCDateTime(2023, 6, 1, 21, 41, 19, 90000), 0, 'S', 'CAE'],
      [None, UTCDateTime(2023, 6, 2, 8, 50, 44, 10000), 0, 'P', 'TRI'],
      [None, UTCDateTime(2023, 6, 2, 8, 50, 46, 420000), 2, 'P', 'BAD'],
      [None, UTCDateTime(2023, 6, 3, 0, 31, 40, 50000), 2, 'P', 'BAD'],
      [None, UTCDateTime(2023, 6, 3, 0, 31, 43, 560000), 1, 'S', 'BAD'],
      [None, UTCDateTime(2023, 6, 3, 5, 18, 36, 10000), 2, 'P', 'VARA'],
      [None, UTCDateTime(2023, 6, 3, 5, 18, 39, 980000), 2, 'S', 'VARA'],
      [None, UTCDateTime(2023, 6, 3, 12, 33, 23, 350000), 0, 'P', 'VARA'],
      [None, UTCDateTime(2023, 6, 3, 12, 33, 25, 980000), 0, 'S', 'VARA'],
      [None, UTCDateTime(2023, 6, 3, 16, 54, 16, 760000), 2, 'P', 'BAD'],
      [None, UTCDateTime(2023, 6, 3, 16, 54, 19, 950000), 2, 'S', 'BAD'],
      [None, UTCDateTime(2023, 6, 4, 0, 3, 5, 450000), 0, 'P', 'BAD'],
      [None, UTCDateTime(2023, 6, 4, 0, 4, 23, 400000), 2, 'S', 'BAD'],
      [None, UTCDateTime(2023, 6, 4, 0, 25, 9, 150000), 3, 'P', 'TRI'],
      [None, UTCDateTime(2023, 6, 4, 0, 25, 14, 520000), 3, 'S', 'TRI'],
      [None, UTCDateTime(2023, 6, 4, 0, 25, 10, 760000), 1, 'P', 'BAD'],
      [None, UTCDateTime(2023, 6, 4, 0, 25, 16, 420000), 2, 'S', 'BAD'],
      [None, UTCDateTime(2023, 6, 4, 17, 57, 11, 390000), 3, 'P', 'CAE'],
      [None, UTCDateTime(2023, 6, 4, 17, 57, 17, 250000), 2, 'S', 'CAE']
    ]
    self.assertListEqual(EXPECTED, DETECT)

class TestEventCounter(unittest.TestCase):
  @unittest.mock.patch("sys.argv",
                       ["Analyzer.py", "-D", "230601", "230604", "-v", "-d",
                        TEST_PATH.__str__(), "-M", PHASENET_STR, "--file",
                        Path(MNL_DATA_PATH, "RSFVG-2023.dat").__str__()])
  def test_event_counter(self):
    args = ini.parse_arguments()
    PRED = ini.classified_loader(args)
    stations = args.station if (args.station is not None and
                                args.station != ALL_WILDCHAR_STR) else \
               PRED[STATION_STR].unique()
    SOURCE, DETECT = event_parser(args.file, args)
    plot_data(DETECT, PRED, args)

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
    args = ini.parse_arguments()
    #        STATION              YEAR, M, D, H, M, S, mS  PHASE WEIGHT
    TRUE = [[None, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 3, PWAVE, STATION],
            [None, UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0, SWAVE, STATION]]
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    PRED = [
      [PHASENET_STR, ORIGINAL_STR, None, UTCDateTime(2023, 6, 1, 0, 1, 2, 3),
       0.43185002, PWAVE, STATION],
      [PHASENET_STR, ORIGINAL_STR, None, UTCDateTime(2023, 6, 1, 0, 4, 5, 6),
       0.3372562, SWAVE, STATION]
    ]
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
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
    args = ini.parse_arguments()
    #          ID            YEAR, M, D, H, M, S, mS WEIGHT PHASE STATION
    TRUE = [[None, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 3, PWAVE, STATION],
            [None, UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0, SWAVE, STATION]]
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    PRED = [
      [PHASENET_STR, ORIGINAL_STR, None, UTCDateTime(2023, 6, 1, 0, 1, 2, 3),
       0.43185002, SWAVE, STATION],
      [PHASENET_STR, ORIGINAL_STR, None, UTCDateTime(2023, 6, 1, 0, 4, 5, 6),
       0.3372562, PWAVE, STATION]
    ]
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
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
    args = ini.parse_arguments()
    #        STATION              YEAR, M, D, H, M, S, mS  PHASE WEIGHT
    TRUE = [[None, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 3, PWAVE, STATION],
            [None, UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0, SWAVE, STATION]]
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    PRED = [
      [PHASENET_STR, ORIGINAL_STR, None, UTCDateTime(2023, 6, 1, 0, 1, 2, 3),
       0.43185002, PWAVE, STATION],
      [PHASENET_STR, ORIGINAL_STR, None, UTCDateTime(2023, 6, 1, 0, 1, 2, 4),
       0.43185002, PWAVE, STATION],
      [PHASENET_STR, ORIGINAL_STR, None, UTCDateTime(2023, 6, 1, 0, 4, 5, 6),
       0.3372562, SWAVE, STATION]
    ]
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
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
    args = ini.parse_arguments()
    #        STATION              YEAR, M, D, H, M, S, mS  PHASE WEIGHT
    TRUE = [[None, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 3, PWAVE, STATION],
            [None, UTCDateTime(2023, 6, 1, 0, 1, 2, 4), 3, PWAVE, STATION]]
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    PRED = [
      [PHASENET_STR, ORIGINAL_STR, None, UTCDateTime(2023, 6, 1, 0, 1, 2, 3),
       0.43185002, PWAVE, STATION],
      [PHASENET_STR, ORIGINAL_STR, None, UTCDateTime(2023, 6, 1, 0, 4, 5, 6),
       0.3372562, SWAVE, STATION]
    ]
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
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
    args = ini.parse_arguments()
    #        STATION              YEAR, M, D, H, M, S, mS  PHASE WEIGHT
    TRUE = [[None, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 3, PWAVE, STATION],
            [None, UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0, SWAVE, STATION]]
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    PRED = []
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
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
    args = ini.parse_arguments()
    #        STATION              YEAR, M, D, H, M, S, mS  PHASE WEIGHT
    TRUE = []
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    PRED = [
      [PHASENET_STR, ORIGINAL_STR, None, UTCDateTime(2023, 6, 1, 0, 1, 2, 3),
       0.43185002, PWAVE, STATION],
      [PHASENET_STR, ORIGINAL_STR, None, UTCDateTime(2023, 6, 1, 0, 4, 5, 6),
       0.3372562, SWAVE, STATION]
    ]
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
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
    args = ini.parse_arguments()
    #        STATION              YEAR, M, D, H, M, S, mS  PHASE WEIGHT
    TRUE = []
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    PRED = []
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
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
    args = ini.parse_arguments()
    #        STATION              YEAR, M, D, H, M, S, mS  PHASE WEIGHT
    TRUE = [[None, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 3, PWAVE, STATION],
            [None, UTCDateTime(2023, 6, 1, 0, 1, 2, 4), 0, SWAVE, STATION]]
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    PRED = [
      [PHASENET_STR, ORIGINAL_STR, None, UTCDateTime(2023, 6, 1, 0, 1, 2, 3),
       0.43185002, PWAVE, STATION],
      [PHASENET_STR, ORIGINAL_STR, None, UTCDateTime(2023, 6, 1, 0, 1, 2, 4),
       0.3372562, SWAVE, STATION],
      [PHASENET_STR, ORIGINAL_STR, None, UTCDateTime(2023, 6, 1, 0, 4, 5, 6),
       0.3372562, SWAVE, STATION]
    ]
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
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
    args = ini.parse_arguments()
    #        STATION              YEAR, M, D, H, M, S, mS  PHASE WEIGHT

if __name__ == "__main__": unittest.main()
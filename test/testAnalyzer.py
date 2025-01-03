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

ID = 1
NETWORK = "MY"
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
      [898, UTCDateTime(2023, 6, 1, 9, 27, 58, 710000), 0, PWAVE, None,
       'CAE'],
      [898, UTCDateTime(2023, 6, 1, 9, 27, 59, 360000), 2, SWAVE, None,
       'CAE'],
      [899, UTCDateTime(2023, 6, 1, 9, 53, 32, 530000), 2, PWAVE, None,
       'CAE'],
      [899, UTCDateTime(2023, 6, 1, 9, 53, 36, 360000), 3, SWAVE, None,
       'CAE'],
      [900, UTCDateTime(2023, 6, 1, 10, 2, 12, 860000), 3, PWAVE, None,
       'VARA'],
      [903, UTCDateTime(2023, 6, 1, 21, 41, 16, 740000), 0, PWAVE, None,
       'CAE'],
      [903, UTCDateTime(2023, 6, 1, 21, 41, 19, 90000), 0, SWAVE, None,
       'CAE'],
      [906, UTCDateTime(2023, 6, 2, 8, 50, 44, 10000), 0, PWAVE, None,
       'TRI'],
      [906, UTCDateTime(2023, 6, 2, 8, 50, 46, 420000), 2, PWAVE, None,
       'BAD'],
      [908, UTCDateTime(2023, 6, 3, 0, 31, 40, 50000), 2, PWAVE, None,
       'BAD'],
      [908, UTCDateTime(2023, 6, 3, 0, 31, 43, 560000), 1, SWAVE, None,
       'BAD'],
      [909, UTCDateTime(2023, 6, 3, 5, 18, 36, 10000), 2, PWAVE, None,
       'VARA'],
      [909, UTCDateTime(2023, 6, 3, 5, 18, 39, 980000), 2, SWAVE, None,
       'VARA'],
      [911, UTCDateTime(2023, 6, 3, 12, 33, 23, 350000), 0, PWAVE, None,
       'VARA'],
      [911, UTCDateTime(2023, 6, 3, 12, 33, 25, 980000), 0, SWAVE, None,
       'VARA'],
      [912, UTCDateTime(2023, 6, 3, 16, 54, 16, 760000), 2, PWAVE, None,
       'BAD'],
      [912, UTCDateTime(2023, 6, 3, 16, 54, 19, 950000), 2, SWAVE, None,
       'BAD'],
      [915, UTCDateTime(2023, 6, 4, 0, 3, 5, 450000), 0, PWAVE, None,
       'BAD'],
      [915, UTCDateTime(2023, 6, 4, 0, 3, 8, 340000), 2, SWAVE, None,
       'BAD'],
      [916, UTCDateTime(2023, 6, 4, 0, 25, 9, 150000), 3, PWAVE, None,
       'TRI'],
      [916, UTCDateTime(2023, 6, 4, 0, 25, 14, 520000), 3, SWAVE, None,
       'TRI'],
      [916, UTCDateTime(2023, 6, 4, 0, 25, 10, 760000), 1, PWAVE, None,
       'BAD'],
      [916, UTCDateTime(2023, 6, 4, 0, 25, 16, 420000), 2, SWAVE, None,
       'BAD'],
      [919, UTCDateTime(2023, 6, 4, 17, 57, 11, 390000), 3, PWAVE, None,
       'CAE'],
      [919, UTCDateTime(2023, 6, 4, 17, 57, 17, 250000), 2, SWAVE, None,
       'CAE']
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
    TRUE = [
    #  ID             YEAR, M, D, H, M, S, mS prob PHASE NETWORK STATION
      [ID, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 3, PWAVE, None, STATION],
      [ID, UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0, SWAVE, None, STATION]
    ]
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    PRED = [
      [PHASENET_STR, ORIGINAL_STR, None, None,
       UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 0.43185002, PWAVE, NETWORK,
       STATION],
      [PHASENET_STR, ORIGINAL_STR, None, None,
       UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0.3372562, SWAVE, NETWORK,
       STATION]
    ]
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
    bpg = myBPGraph(TRUE, PRED, dist_default)
    EXPECTED = [[0.9943185002, 0.0        ],
                [0.0,          0.993372562]]
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    bpg.makeMatch()
    EXPECTED = [[0.9943185002, 0.0        ],
                [0.0,          0.993372562]]
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    CFN_MTX, TP, FN, FP = bpg.confMtx()
    EXPECTED = [[1, 0, 0],
                [0, 1, 0],
                [0, 0, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = set()
    EXPECTED.add((ID, (str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3)),
                       str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3))),
                  (3, 0.43185002), PWAVE, NETWORK, STATION))
    EXPECTED.add((ID, (str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6)),
                       str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6))),
                  (0, 0.3372562), SWAVE, NETWORK, STATION))
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
    TRUE = [
    #  ID             YEAR, M, D, H, M, S, mS Prob PHASE NETWORK STATION
      [ID, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 3, PWAVE, None, STATION],
      [ID, UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0, SWAVE, None, STATION]
    ]
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    PRED = [
      [PHASENET_STR, ORIGINAL_STR, None, None,
       UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 0.43185002, SWAVE, NETWORK,
       STATION],
      [PHASENET_STR, ORIGINAL_STR, None, None,
       UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0.3372562, PWAVE, NETWORK,
       STATION]
    ]
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
    bpg = myBPGraph(TRUE, PRED, dist_default)
    EXPECTED = [[0.10331850020000001, 0.0                ],
                [0.0,                 0.10237256200000001]]
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    EXPECTED = [(0, 0, 0.10331850020000001),
                (1, 1, 0.10237256200000001)]
    self.assertListEqual(EXPECTED, bpg.maxWmatch())
    bpg.makeMatch()
    EXPECTED = [[0.10331850020000001, 0.0                ],
                [0.0,                 0.10237256200000001]]
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    CFN_MTX, TP, FN, FP = bpg.confMtx()
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
    TRUE = [
    #  ID             YEAR, M, D, H, M, S, mS Prob PHASE NETWORK STATION
      [ID, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 3, PWAVE, None, STATION],
      [ID, UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0, SWAVE, None, STATION]
    ]
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    PRED = [
      [PHASENET_STR, ORIGINAL_STR, None, None,
       UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 0.43185002, PWAVE, NETWORK,
       STATION],
      [PHASENET_STR, ORIGINAL_STR, None, None,
       UTCDateTime(2023, 6, 1, 0, 1, 2, 4), 0.43185002, PWAVE, NETWORK,
       STATION],
      [PHASENET_STR, ORIGINAL_STR, None, None,
       UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0.3372562, SWAVE, NETWORK,
       STATION]
    ]
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
    bpg = myBPGraph(TRUE, PRED, dist_default)
    EXPECTED = [[0.9943185002, 0.9943183022,         0.0],
                [         0.0,          0.0, 0.993372562]]
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    EXPECTED = [(0, 0, 0.9943185002),
                (1, 2, 0.993372562)]
    self.assertListEqual(EXPECTED, bpg.maxWmatch())
    bpg.makeMatch()
    EXPECTED = [[0.9943185002, 0.0,         0.0],
                [0.0,          0.0, 0.993372562]]
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    CFN_MTX, TP, FN, FP = bpg.confMtx()
    EXPECTED = [[1, 0, 0],
                [0, 1, 0],
                [1, 0, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = set()
    EXPECTED.add((ID, (str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3)),
                       str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3))),
                  (3, 0.43185002), PWAVE, NETWORK, STATION))
    EXPECTED.add((ID, (str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6)),
                       str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6))),
                  (0, 0.3372562), SWAVE, NETWORK, STATION))
    self.assertSetEqual(EXPECTED, TP)
    EXPECTED = []
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = set()
    EXPECTED.add((None, str(UTCDateTime(2023, 6, 1, 0, 1, 2, 4)), 0.43185002,
                  PWAVE, NETWORK, STATION))
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
    TRUE = [
    #  ID             YEAR, M, D, H, M, S, mS Prob PHASE NETWORK STATION
      [ID, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 3, PWAVE, None, STATION],
      [ID, UTCDateTime(2023, 6, 1, 0, 1, 2, 4), 3, PWAVE, None, STATION]
    ]
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    PRED = [
      [PHASENET_STR, ORIGINAL_STR, None, None,
       UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 0.43185002, PWAVE, NETWORK,
       STATION],
      [PHASENET_STR, ORIGINAL_STR, None, None,
       UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0.3372562, SWAVE, NETWORK,
       STATION]
    ]
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
    bpg = myBPGraph(TRUE, PRED, dist_default)
    EXPECTED = [[0.9943185002, 0.0],
                [0.9943183022, 0.0]]
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    EXPECTED = [(0, 0, 0.9943185002)]
    self.assertListEqual(EXPECTED, bpg.maxWmatch())
    bpg.makeMatch()
    EXPECTED = [[0.9943185002, 0.0],
                [0.0,          0.0]]
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    CFN_MTX, TP, FN, FP = bpg.confMtx()
    EXPECTED = [[1, 0, 1],
                [0, 0, 0],
                [0, 1, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = set()
    EXPECTED.add((ID, (str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3)),
                       str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3))),
                  (3, 0.43185002), PWAVE, NETWORK, STATION))
    self.assertSetEqual(EXPECTED, TP)
    EXPECTED = [[ID, UTCDateTime(2023, 6, 1, 0, 1, 2, 4), 3, PWAVE, None,
                 STATION]]
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = set()
    EXPECTED.add((None, str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6)), 0.3372562,
                  SWAVE, NETWORK, STATION))
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
    TRUE = [
    #  ID             YEAR, M, D, H, M, S, mS Prob PHASE NETWORK STATION
      [ID, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 3, PWAVE, None, STATION],
      [ID, UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0, SWAVE, None, STATION]
    ]
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    PRED = []
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
    bpg = myBPGraph(TRUE, PRED, dist_default)
    EXPECTED = [[], []]
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    EXPECTED = []
    self.assertListEqual(EXPECTED, bpg.maxWmatch())
    bpg.makeMatch()
    EXPECTED = [[], []]
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    CFN_MTX, TP, FN, FP = bpg.confMtx()
    EXPECTED = [[0, 0, 1],
                [0, 0, 1],
                [0, 0, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = set()
    self.assertSetEqual(EXPECTED, TP)
    EXPECTED = [
      [ID, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 3, PWAVE, None, STATION],
      [ID, UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0, SWAVE, None, STATION]
    ]
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
    #    ID             YEAR, M, D, H, M, S, mS Prob PHASE NETWORK STATION
    TRUE = []
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    PRED = [
      [PHASENET_STR, ORIGINAL_STR, None, None,
       UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 0.43185002, PWAVE, NETWORK,
       STATION],
      [PHASENET_STR, ORIGINAL_STR, None, None,
       UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0.3372562, SWAVE, NETWORK,
       STATION]
    ]
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
    bpg = myBPGraph(TRUE, PRED, dist_default)
    EXPECTED = []
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    EXPECTED = []
    self.assertListEqual(EXPECTED, bpg.maxWmatch())
    bpg.makeMatch()
    EXPECTED = []
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    CFN_MTX, TP, FN, FP = bpg.confMtx()
    EXPECTED = [[0, 0, 0],
                [0, 0, 0],
                [1, 1, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = set()
    self.assertSetEqual(EXPECTED, TP)
    EXPECTED = []
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = set()
    EXPECTED.add((None, str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3)), 0.43185002,
                  PWAVE, NETWORK, STATION))
    EXPECTED.add((None, str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6)), 0.3372562,
                  SWAVE, NETWORK, STATION))
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
    #    ID             YEAR, M, D, H, M, S, mS Prob PHASE NETWORK STATION
    TRUE = []
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    PRED = []
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
    bpg = myBPGraph(TRUE, PRED, dist_default)
    EXPECTED = []
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    EXPECTED = []
    self.assertListEqual(EXPECTED, bpg.maxWmatch())
    bpg.makeMatch()
    EXPECTED = []
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    CFN_MTX, TP, FN, FP = bpg.confMtx()
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
    TRUE = [
    #  ID             YEAR, M, D, H, M, S, mS Prob PHASE NETWORK STATION
      [ID, UTCDateTime(2023, 6, 1, 0, 1, 2, 0), 3, PWAVE, None, STATION],
      [ID, UTCDateTime(2023, 6, 1, 0, 1, 2, 500001), 0, SWAVE, None, STATION]]
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    PRED = [
      [PHASENET_STR, ORIGINAL_STR, None, None,
       UTCDateTime(2023, 6, 1, 0, 1, 2, 500000), 0.43185002, PWAVE, NETWORK,
       STATION],
      [PHASENET_STR, ORIGINAL_STR, None, None,
       UTCDateTime(2023, 6, 1, 0, 1, 3, 1), 0.3372562, SWAVE, NETWORK,
       STATION],
      [PHASENET_STR, ORIGINAL_STR, None, None,
       UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0.3372562, SWAVE, NETWORK,
       STATION]
    ]
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
    bpg = myBPGraph(TRUE, PRED, dist_default)
    EXPECTED = [[0.8953185002, 0.0,         0.0],
                [0.1033183022, 0.894372562, 0.0]]
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    EXPECTED = [(0, 0, 0.8953185002),
                (1, 1, 0.894372562)]
    self.assertListEqual(EXPECTED, bpg.maxWmatch())
    bpg.makeMatch()
    EXPECTED = [[0.8953185002, 0.0,         0.0],
                [0.0,          0.894372562, 0.0]]
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    CFN_MTX, TP, FN, FP = bpg.confMtx()
    EXPECTED = [[1, 0, 0],
                [0, 1, 0],
                [0, 1, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = set()
    EXPECTED.add((ID, (str(UTCDateTime(2023, 6, 1, 0, 1, 2, 0)),
                       str(UTCDateTime(2023, 6, 1, 0, 1, 2, 500000))),
                  (3, 0.43185002), PWAVE, NETWORK, STATION))
    EXPECTED.add((ID, (str(UTCDateTime(2023, 6, 1, 0, 1, 2, 500001)),
                       str(UTCDateTime(2023, 6, 1, 0, 1, 3, 1))),
                  (0, 0.3372562), SWAVE, NETWORK, STATION))
    self.assertSetEqual(EXPECTED, TP)
    EXPECTED = []
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = set()
    EXPECTED.add((None, str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6)), 0.3372562,
                  SWAVE, NETWORK, STATION))
    self.assertSetEqual(EXPECTED, FP)

  @unittest.mock.patch("sys.argv",
                       ["Analyzer.py", "-D", "230601", "230601", "-v", "-d",
                        TEST_PATH.__str__()])
  def test_true_pred_complex(self):
    """
    OFFSET = 0.5 : |--"--|
           |------------------------------- 66 -------------------------------|
                         |--"--|
    TRUE : |----------------P--S----------------------------------------------|
    PRED : |----------------S--P-:----------------------------S---------------|
                            |--"--|                        |--"--|
    OUTPUT: P [1, 0, 0]
            S [0, 1, 0]
            N [0, 1, 0]
               P  S  N : PRED
    """
    args = ini.parse_arguments()
    TRUE = [
    #  ID             YEAR, M, D, H, M, S, mS Prob PHASE NETWORK STATION
      [ID, UTCDateTime(2023, 6, 1, 0, 1, 2, 0), 3, PWAVE, None, STATION],
      [ID, UTCDateTime(2023, 6, 1, 0, 1, 2, 500000), 0, SWAVE, None, STATION]]
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    PRED = [
      [PHASENET_STR, ORIGINAL_STR, None, None,
       UTCDateTime(2023, 6, 1, 0, 1, 2, 0), 0.43185002, SWAVE, NETWORK,
       STATION],
      [PHASENET_STR, ORIGINAL_STR, None, None,
       UTCDateTime(2023, 6, 1, 0, 1, 2, 500000), 0.3372562, PWAVE, NETWORK,
       STATION],
      [PHASENET_STR, ORIGINAL_STR, None, None,
       UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0.3372562, SWAVE, NETWORK,
       STATION]
    ]
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
    bpg = myBPGraph(TRUE, PRED, dist_default)
    EXPECTED = [[0.10331850020000001, 0.894372562,         0.0],
                [0.8953185002,        0.10237256200000001, 0.0]]
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    EXPECTED = [(0, 1, 0.894372562),
                (1, 0, 0.8953185002)]
    self.assertListEqual(EXPECTED, bpg.maxWmatch())
    bpg.makeMatch()
    EXPECTED = [[0.0,          0.894372562, 0.0],
                [0.8953185002, 0.0,         0.0]]
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    CFN_MTX, TP, FN, FP = bpg.confMtx()
    EXPECTED = [[1, 0, 0],
                [0, 1, 0],
                [0, 1, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = set()
    EXPECTED.add((ID, (str(UTCDateTime(2023, 6, 1, 0, 1, 2, 0)),
                       str(UTCDateTime(2023, 6, 1, 0, 1, 2, 500000))),
                  (3, 0.3372562), PWAVE, NETWORK, STATION))
    EXPECTED.add((ID, (str(UTCDateTime(2023, 6, 1, 0, 1, 2, 500000)),
                       str(UTCDateTime(2023, 6, 1, 0, 1, 2, 0))),
                  (0, 0.43185002), SWAVE, NETWORK, STATION))
    self.assertSetEqual(EXPECTED, TP)
    EXPECTED = []
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = set()
    EXPECTED.add((None, str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6)), 0.3372562,
                  SWAVE, NETWORK, STATION))
    self.assertSetEqual(EXPECTED, FP)

if __name__ == "__main__": unittest.main()
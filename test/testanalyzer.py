#!/bin/python
import unittest
import initializer as ini
from analyzer import *
import sys
import os
from pathlib import Path
PRJ_PATH = Path(os.path.dirname(__file__)).parent
SRC_PATH = os.path.join(PRJ_PATH, "src")
# Add to path
if SRC_PATH not in sys.path:
  sys.path.append(SRC_PATH)


DATA_PATH = Path(PRJ_PATH, "data", "test")
TEST_PATH = Path(DATA_PATH, "waveforms")
MNL_DATA_PATH = Path(DATA_PATH, "manual")

FILENAME = Path(MNL_DATA_PATH, "RSFVG-2023" + DAT_EXT)

EXPECTED_STR = "expected"

ID = 1
NETWORK = "MY"
STATION = "EG"
THRESHOLD = 0.3
START = "230601"


class TestPickParser(unittest.TestCase):
  @unittest.mock.patch("sys.argv",
                       ["Analyzer.py", "-D", "230601", "230604", "-v", "-d",
                        str(TEST_PATH), "--file", MNL_DATA_PATH.__str__()])
  def test_parse_pick(self):
    args = ini.parse_arguments()
    SOURCE, DETECT = ini.true_loader(args)
    SOURCE = SOURCE.values.tolist()
    DETECT = DETECT.values.tolist()
    EXPECTED = [
        [2023000898, UTCDateTime(
            2023, 6, 1, 9, 27, 58, 710000),  0, PWAVE, None, 'CAE'],
        [2023000898, UTCDateTime(2023, 6, 1, 9, 27, 59,
                                 360000),  2, SWAVE, None, 'CAE'],
        [2023000899, UTCDateTime(2023, 6, 1, 9, 53, 32,
                                 530000),  2, PWAVE, None, 'CAE'],
        [2023000899, UTCDateTime(2023, 6, 1, 9, 53, 36,
                                 360000),  3, SWAVE, None, 'CAE'],
        [2023000903, UTCDateTime(2023, 6, 1, 21, 41, 16,
                                 740000), 0, PWAVE, None, 'CAE'],
        [2023000903, UTCDateTime(2023, 6, 1, 21, 41, 19,
                                 90000),  0, SWAVE, None, 'CAE'],
        [2023000908, UTCDateTime(2023, 6, 3, 0, 31, 40,
                                 50000),   2, PWAVE, None, 'BAD'],
        [2023000908, UTCDateTime(2023, 6, 3, 0, 31, 43,
                                 560000),  1, SWAVE, None, 'BAD'],
        [2023000909, UTCDateTime(
            2023, 6, 3, 5, 18, 36, 10000),   2, PWAVE, None, 'VARA'],
        [2023000909, UTCDateTime(
            2023, 6, 3, 5, 18, 39, 980000),  2, SWAVE, None, 'VARA'],
        [2023000911, UTCDateTime(
            2023, 6, 3, 12, 33, 23, 350000), 0, PWAVE, None, 'VARA'],
        [2023000911, UTCDateTime(
            2023, 6, 3, 12, 33, 25, 980000), 0, SWAVE, None, 'VARA'],
        [2023000912, UTCDateTime(2023, 6, 3, 16, 54, 16,
                                 760000), 2, PWAVE, None, 'BAD'],
        [2023000912, UTCDateTime(2023, 6, 3, 16, 54, 19,
                                 950000), 2, SWAVE, None, 'BAD'],
        [2023000915, UTCDateTime(2023, 6, 4, 0, 3, 5, 450000),
         0, PWAVE, None, 'BAD'],
        [2023000915, UTCDateTime(2023, 6, 4, 0, 3, 8, 340000),
         2, SWAVE, None, 'BAD'],
        [2023000916, UTCDateTime(
            2023, 6, 4, 0, 25, 9, 150000),   3, PWAVE, None, 'TRI'],
        [2023000916, UTCDateTime(2023, 6, 4, 0, 25, 14,
                                 520000),  3, SWAVE, None, 'TRI'],
        [2023000916, UTCDateTime(2023, 6, 4, 0, 25, 10,
                                 760000),  1, PWAVE, None, 'BAD'],
        [2023000916, UTCDateTime(2023, 6, 4, 0, 25, 16,
                                 420000),  2, SWAVE, None, 'BAD'],
        [2023000919, UTCDateTime(2023, 6, 4, 17, 57, 11,
                                 390000), 3, PWAVE, None, 'CAE'],
        [2023000919, UTCDateTime(2023, 6, 4, 17, 57, 17,
                                 250000), 2, SWAVE, None, 'CAE']
    ]
    self.assertListEqual(EXPECTED, DETECT)
    EXPECTED = [
        [2023000898, UTCDateTime(2023, 6, 1, 9, 27, 56, 980000), 45.942,
         12.377166666666668, 2.09, 0.98, 6, 347, 8.8, 0.28, 3.0, 4.7, 'D1',
         None],
        [2023000899, UTCDateTime(2023, 6, 1, 9, 53, 27, 930000), 46.176,
         12.502333333333333, 19.15, 1.21, 10, 125, 9.1, 0.12, 0.9, 0.9, 'B1',
         None],
        [2023000903, UTCDateTime(2023, 6, 1, 21, 41, 13, 560000),
         46.096833333333336, 12.3035, 12.3, 0.99, 12, 131, 7.6, 0.15, 0.6, 1.0,
         'B1', None],
        [2023000908, UTCDateTime(2023, 6, 3, 0, 31, 35, 710000),
         46.38433333333333, 13.031, 10.83, 0.44, 12, 137, 4.0, 0.03, 0.1, 0.2,
         'B1', None],
        [2023000909, UTCDateTime(2023, 6, 3, 5, 18, 30, 810000), 45.6645,
         10.626333333333333, 10.53, 1.6, 19, 147, 4.0, 0.1, 0.3, 0.5, 'B1',
         None],
        [2023000911, UTCDateTime(2023, 6, 3, 12, 33, 19, 650000),
         45.750166666666665, 11.105333333333334, 9.43, 1.2, 13, 127, 14.9,
         0.11, 0.4, 1.0, 'B1', None],
        [2023000912, UTCDateTime(2023, 6, 3, 16, 54, 12, 690000), 46.3875,
         13.114833333333333, 13.12, 0.51, 14, 99, 7.6, 0.1, 0.4, 0.6, 'B1',
         None],
        [2023000915, UTCDateTime(2023, 6, 4, 0, 3, 1, 710000), 46.1995,
         13.4425, 15.65, 1.03, 20, 96, 7.2, 0.09, 0.3, 0.4, 'B1', None],
        [2023000916, UTCDateTime(2023, 6, 4, 0, 25, 3, 360000),
         45.88916666666667, 13.4015, 13.86, 2.03, 34, 165, 5.8, 0.14, 0.4, 0.6,
         'B1', None],
        [2023000919, UTCDateTime(2023, 6, 4, 17, 57, 3, 710000), 46.031,
         11.869333333333334, 4.95, 1.72, 22, 64, 15.7, 0.1, 0.2, 1.1, 'B1',
         None]
    ]
    self.assertListEqual(EXPECTED, SOURCE)


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
    #        ID              YEAR, M, D, H, M, S mS prob PHASE NETWORK STATION
    TRUE = [[ID, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 3, PWAVE, None, STATION],
            [ID, UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0, SWAVE, None, STATION]]
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    #                    YEAR, M, D, H, M, S, mS prob PHASE NETWORK STATION
    PRED = [[PHASENET_STR, ORIGINAL_STR, 0.4, None,
             UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 0.43185002, PWAVE, NETWORK,
             STATION],
            [PHASENET_STR, ORIGINAL_STR, 0.3, None,
             UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0.3372562, SWAVE, NETWORK,
             STATION]]
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
    bpg = myBPGraph(TRUE, PRED)
    EXPECTED = [[0.9943185002, 0.0],
                [0.0,          0.993372562]]
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    bpg.makeMatch()
    EXPECTED = [[0.9943185002, 0.0],
                [0.0,          0.993372562]]
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    CFN_MTX, TP, FN, FP = bpg.confMtx()
    EXPECTED = [[1, 0, 0],
                [0, 1, 0],
                [0, 0, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = set()
    EXPECTED.add((0.4, (ID, NONE_STR),
                  (str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3)),
                   str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3))), (3, 0.43185002),
                  PWAVE, NETWORK, STATION))
    EXPECTED.add((0.3, (ID, NONE_STR),
                  (str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6)),
                   str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6))), (0, 0.3372562),
                  SWAVE, NETWORK, STATION))
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
    #        ID              YEAR, M, D, H, M, S mS Prob PHASE NETWORK STATION
    TRUE = [[ID, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 3, PWAVE, None, STATION],
            [ID, UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0, SWAVE, None, STATION]]
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    #                   YEAR, M, D, H, M, S mS Prob PHASE NETWORK STATION
    PRED = [[PHASENET_STR, ORIGINAL_STR, 0.4, None,
             UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 0.43185002, SWAVE, NETWORK,
             STATION],
            [PHASENET_STR, ORIGINAL_STR, 0.3, None,
            UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0.3372562, PWAVE, NETWORK,
            STATION]]
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
    bpg = myBPGraph(TRUE, PRED)
    EXPECTED = [[0.10331850020000001, 0.0],
                [0.0,                 0.10237256200000001]]
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    EXPECTED = [(0, 0, 0.10331850020000001),
                (1, 1, 0.10237256200000001)]
    self.assertListEqual(EXPECTED, bpg.maxWmatch())
    bpg.makeMatch()
    EXPECTED = [[0.10331850020000001, 0.0],
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
    #        ID              YEAR, M, D, H, M, S mS Prob PHASE NETWORK STATION
    TRUE = [[ID, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 3, PWAVE, None, STATION],
            [ID, UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0, SWAVE, None, STATION]]
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    #                   YEAR, M, D, H, M, S mS Prob PHASE NETWORK STATION
    PRED = [[PHASENET_STR, ORIGINAL_STR, 0.4, None,
             UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 0.43185002, PWAVE, NETWORK,
             STATION],
            [PHASENET_STR, ORIGINAL_STR, 0.4, None,
             UTCDateTime(2023, 6, 1, 0, 1, 2, 4), 0.43185002, PWAVE, NETWORK,
             STATION],
            [PHASENET_STR, ORIGINAL_STR, 0.3, None,
             UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0.3372562, SWAVE, NETWORK,
             STATION]]
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
    bpg = myBPGraph(TRUE, PRED)
    EXPECTED = [[0.9943185002, 0.9943183022,         0.0],
                [0.0,          0.0, 0.993372562]]
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
    EXPECTED.add((0.4, (ID, NONE_STR),
                  (str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3)),
                   str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3))), (3, 0.43185002),
                  PWAVE, NETWORK, STATION))
    EXPECTED.add((0.3, (ID, NONE_STR),
                  (str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6)),
                   str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6))), (0, 0.3372562),
                  SWAVE, NETWORK, STATION))
    self.assertSetEqual(EXPECTED, TP)
    EXPECTED = []
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = set()
    EXPECTED.add((None, str(UTCDateTime(2023, 6, 1, 0, 1, 2, 4)),
                  0.43185002, PWAVE, NETWORK, STATION))
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
    #        ID              YEAR, M, D, H, M, S mS Prob PHASE NETWORK STATION
    TRUE = [[ID, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 3, PWAVE, None, STATION],
            [ID, UTCDateTime(2023, 6, 1, 0, 1, 2, 4), 3, PWAVE, None, STATION]]
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    #                    YEAR, M, D, H, M, S mS Prob PHASE NETWORK STATION
    PRED = [[PHASENET_STR, ORIGINAL_STR, 0.4, None,
             UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 0.43185002, PWAVE, NETWORK,
             STATION],
            [PHASENET_STR, ORIGINAL_STR, 0.3, None,
             UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0.3372562, SWAVE, NETWORK,
             STATION]]
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
    bpg = myBPGraph(TRUE, PRED)
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
    EXPECTED.add((0.4, (ID, NONE_STR),
                  (str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3)),
                   str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3))), (3, 0.43185002),
                  PWAVE, NETWORK, STATION))
    self.assertSetEqual(EXPECTED, TP)
    EXPECTED = [[ID, UTCDateTime(2023, 6, 1, 0, 1, 2, 4), 3, PWAVE, None,
                 STATION]]
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = set()
    EXPECTED.add((None, str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6)),
                  0.3372562, SWAVE, NETWORK, STATION))
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
    #        ID              YEAR, M, D, H, M, S mS Prob PHASE NETWORK STATION
    TRUE = [[ID, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 3, PWAVE, None, STATION],
            [ID, UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0, SWAVE, None, STATION]]
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    PRED = []
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
    bpg = myBPGraph(TRUE, PRED)
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
    EXPECTED = [[ID, UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 3, PWAVE, None,
                 STATION],
                [ID, UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0, SWAVE, None,
                 STATION]]
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
    #                   YEAR, M, D, H, M, S mS Prob PHASE NETWORK STATION
    PRED = [[PHASENET_STR, ORIGINAL_STR, 0.4, None,
             UTCDateTime(2023, 6, 1, 0, 1, 2, 3), 0.43185002, PWAVE, NETWORK,
             STATION],
            [PHASENET_STR, ORIGINAL_STR, 0.3, None,
             UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0.3372562, SWAVE, NETWORK,
             STATION]]
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
    bpg = myBPGraph(TRUE, PRED)
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
    EXPECTED.add((None, str(UTCDateTime(2023, 6, 1, 0, 1, 2, 3)),
                  0.43185002, PWAVE, NETWORK, STATION))
    EXPECTED.add((None, str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6)),
                  0.3372562, SWAVE, NETWORK, STATION))
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
    bpg = myBPGraph(TRUE, PRED)
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
                         |--"--|--"--|
    TRUE : |----------------P--:--S--:----------------------------------------|
                            |  :  |  :
    PRED : |----------------|--P--|S-:-----------------------S---------------|
                            |--"--|                       |--"--|
    OUTPUT: P [1, 0, 0]
            S [0, 1, 0]
            N [0, 1, 0]
               P  S  N : PRED
    """
    args = ini.parse_arguments()
    #        ID              YEAR, M, D, H, M, S, mS Prob PHASE NETWORK STATION
    TRUE = [[ID, UTCDateTime(2023, 6, 1, 0, 1, 2, 0), 3, PWAVE, None, STATION],
            [ID, UTCDateTime(2023, 6, 1, 0, 1, 3, 0), 0, SWAVE, None, STATION]]
    TRUE = pd.DataFrame(TRUE, columns=HEADER_MANL)
    #                   YEAR, M, D, H, M, S mS Prob PHASE NETWORK STATION
    PRED = [[PHASENET_STR, ORIGINAL_STR, 0.4, None,
             UTCDateTime(2023, 6, 1, 0, 1, 2, 500000), 0.43185002, PWAVE,
             NETWORK, STATION],
            [PHASENET_STR, ORIGINAL_STR, 0.3, None,
             UTCDateTime(2023, 6, 1, 0, 1, 3, 1), 0.3372562, SWAVE, NETWORK,
             STATION],
            [PHASENET_STR, ORIGINAL_STR, 0.3, None,
             UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0.3372562, SWAVE, NETWORK,
             STATION]]
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
    bpg = myBPGraph(TRUE, PRED)
    EXPECTED = [[0.8953185002, 0.0,         0.0],
                [0.0043185002, 0.993372364, 0.0]]
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    EXPECTED = [(0, 0, 0.8953185002),
                (1, 1, 0.993372364)]
    self.assertListEqual(EXPECTED, bpg.maxWmatch())
    bpg.makeMatch()
    EXPECTED = [[0.8953185002, 0.0,         0.0],
                [0.0,          0.993372364, 0.0]]
    self.assertListEqual(EXPECTED, bpg.adjMtx().tolist())
    CFN_MTX, TP, FN, FP = bpg.confMtx()
    EXPECTED = [[1, 0, 0],
                [0, 1, 0],
                [0, 1, 0]]
    self.assertListEqual(EXPECTED, CFN_MTX.values.tolist())
    EXPECTED = set()
    EXPECTED.add((0.4, (ID, NONE_STR),
                  (str(UTCDateTime(2023, 6, 1, 0, 1, 2, 0)),
                   str(UTCDateTime(2023, 6, 1, 0, 1, 2, 500000))),
                  (3, 0.43185002), PWAVE, NETWORK, STATION))
    EXPECTED.add((0.3, (ID, NONE_STR),
                  (str(UTCDateTime(2023, 6, 1, 0, 1, 3, 0)),
                   str(UTCDateTime(2023, 6, 1, 0, 1, 3, 1))), (0, 0.3372562),
                  SWAVE, NETWORK, STATION))
    self.assertSetEqual(EXPECTED, TP)
    EXPECTED = []
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = set()
    EXPECTED.add((None, str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6)),
                  0.3372562, SWAVE, NETWORK, STATION))
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
    PRED : |----------------S--P------------------------------S---------------|
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
    #                    YEAR, M, D, H, M, S, mS Prob PHASE NETWORK STATION
    PRED = [[PHASENET_STR, ORIGINAL_STR, 0.4, None,
             UTCDateTime(2023, 6, 1, 0, 1, 2, 0), 0.43185002, SWAVE, NETWORK,
             STATION],
            [PHASENET_STR, ORIGINAL_STR, 0.3, None,
             UTCDateTime(2023, 6, 1, 0, 1, 2, 500000), 0.3372562, PWAVE,
             NETWORK, STATION],
            [PHASENET_STR, ORIGINAL_STR, 0.3, None,
             UTCDateTime(2023, 6, 1, 0, 4, 5, 6), 0.3372562, SWAVE, NETWORK,
            STATION]]
    PRED = pd.DataFrame(PRED, columns=HEADER_PRED)
    bpg = myBPGraph(TRUE, PRED)
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
    EXPECTED.add((0.3, (ID, NONE_STR),
                  (str(UTCDateTime(2023, 6, 1, 0, 1, 2, 0)),
                   str(UTCDateTime(2023, 6, 1, 0, 1, 2, 500000))),
                  (3, 0.3372562), PWAVE, NETWORK, STATION))
    EXPECTED.add((0.4, (ID, NONE_STR),
                  (str(UTCDateTime(2023, 6, 1, 0, 1, 2, 500000)),
                   str(UTCDateTime(2023, 6, 1, 0, 1, 2, 0))), (0, 0.43185002),
                  SWAVE, NETWORK, STATION))
    self.assertSetEqual(EXPECTED, TP)
    EXPECTED = []
    self.assertListEqual(EXPECTED, FN)
    EXPECTED = set()
    EXPECTED.add((None, str(UTCDateTime(2023, 6, 1, 0, 4, 5, 6)),
                  0.3372562, SWAVE, NETWORK, STATION))
    self.assertSetEqual(EXPECTED, FP)


if __name__ == "__main__":
  unittest.main()

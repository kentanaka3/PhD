#!/bin/python
import unittest
import sys
import os
from pathlib import Path
PRJ_PATH = Path(os.path.dirname(__file__)).parent
INC_PATH = os.path.join(PRJ_PATH, "inc")
# Add to path
if INC_PATH not in sys.path:
  sys.path.append(INC_PATH)
  import initializer as ini
  from parser import *


DATA_PATH = Path(PRJ_PATH, "data", "test")
TEST_PATH = Path(DATA_PATH, "waveforms")
MNL_DATA_PATH = Path(DATA_PATH, "manual")

FILENAME = Path(MNL_DATA_PATH, "RSFVG-2023")


class TestParser(unittest.TestCase):
  @unittest.mock.patch("sys.argv", ["parser.py", "-v", "-d", str(TEST_PATH),
                                    "--file", FILENAME.__str__() + DAT_EXT,
                                    "-S", "CAE", "BAD", "-D", "230601",
                                    "230604"])
  def test_event_parser_dat(self):
    args = ini.parse_arguments()
    _, stations = ini.station_loader(args)
    _, data = event_parser_dat(args.file[0], *args.dates, stations=stations)
    data = data.values.tolist()
    EXPECTED = [
        [2023000898, UTCDateTime(2023, 6, 1, 9, 27, 58, 710000), 0, PWAVE,
         None, 'CAE'],
        [2023000898, UTCDateTime(2023, 6, 1, 9, 27, 59, 360000), 2, SWAVE,
         None, 'CAE'],
        [2023000899, UTCDateTime(2023, 6, 1, 9, 53, 32, 530000), 2, PWAVE,
         None, 'CAE'],
        [2023000899, UTCDateTime(2023, 6, 1, 9, 53, 36, 360000), 3, SWAVE,
         None, 'CAE'],
        [2023000903, UTCDateTime(2023, 6, 1, 21, 41, 16, 740000), 0, PWAVE,
         None, 'CAE'],
        [2023000903, UTCDateTime(2023, 6, 1, 21, 41, 19, 90000), 0, SWAVE,
         None, 'CAE'],
        [2023000908, UTCDateTime(2023, 6, 3, 0, 31, 40, 50000), 2, PWAVE,
         None, 'BAD'],
        [2023000908, UTCDateTime(2023, 6, 3, 0, 31, 43, 560000), 1, SWAVE,
         None, 'BAD'],
        [2023000912, UTCDateTime(2023, 6, 3, 16, 54, 16, 760000), 2, PWAVE,
         None, 'BAD'],
        [2023000912, UTCDateTime(2023, 6, 3, 16, 54, 19, 950000), 2, SWAVE,
         None, 'BAD'],
        [2023000915, UTCDateTime(2023, 6, 4, 0, 3, 5, 450000), 0, PWAVE,
         None, 'BAD'],
        [2023000915, UTCDateTime(2023, 6, 4, 0, 3, 8, 340000), 2, SWAVE,
         None, 'BAD'],
        [2023000916, UTCDateTime(2023, 6, 4, 0, 25, 10, 760000), 1, PWAVE,
         None, 'BAD'],
        [2023000916, UTCDateTime(2023, 6, 4, 0, 25, 16, 420000), 2, SWAVE,
         None, 'BAD'],
        [2023000919, UTCDateTime(2023, 6, 4, 17, 57, 11, 390000), 3, PWAVE,
         None, 'CAE'],
        [2023000919, UTCDateTime(2023, 6, 4, 17, 57, 17, 250000), 2, SWAVE,
         None, 'CAE']
    ]
    self.assertListEqual(EXPECTED, data)

  @unittest.mock.patch("sys.argv", ["parser.py", "-v", "-d", str(TEST_PATH),
                                    "--file", FILENAME.__str__() + PUN_EXT])
  def test_event_parser_pun(self):
    args = ini.parse_arguments()
    data, _ = event_parser_pun(args.file[0], *args.dates)
    data = data.values.tolist()
    EXPECTED = [
        [None, UTCDateTime(2023, 6, 1, 9, 27, 56, 980000), 45.942,
         12.377166666666668, 2.09, 0.98, 6, 347, 8.8, 0.28, 3.0, 4.7, 'D1',
         None],
        [None, UTCDateTime(2023, 6, 1, 9, 53, 27, 930000), 46.176,
         12.502333333333333, 19.15, 1.21, 10, 125, 9.1, 0.12, 0.9, 0.9, 'B1',
         None],
        [None, UTCDateTime(2023, 6, 1, 19, 8, 21, 300000), 46.4745,
         13.493333333333334, 6.99, 0.55, 10, 137, 2.7, 0.08, 0.4, 0.6, 'B1',
         None],
        [None, UTCDateTime(2023, 6, 1, 21, 41, 13, 560000), 46.096833333333336,
         12.3035, 12.3, 0.99, 12, 131, 7.6, 0.15, 0.6, 1.0, 'B1', None],
        [None, UTCDateTime(2023, 6, 1, 23, 54, 18, 10000), 46.2965,
         14.465833333333332, 6.14, 1.22, 8, 301, 1.8, 0.05, 0.6, 0.3, 'C1',
         None],
        [None, UTCDateTime(2023, 6, 2, 2, 18, 24, 150000), 45.90083333333333,
         10.894166666666667, 14.72, 1.62, 22, 74, 7.6, 0.26, 0.6, 1.1, 'B1',
         None],
        [None, UTCDateTime(2023, 6, 3, 0, 31, 35, 710000), 46.38433333333333,
         13.031, 10.83, 0.44, 12, 137, 4.0, 0.03, 0.1, 0.2, 'B1', None],
        [None, UTCDateTime(2023, 6, 3, 5, 18, 30, 810000), 45.6645,
         10.626333333333333, 10.53, 1.6, 19, 147, 4.0, 0.1, 0.3, 0.5, 'B1',
         None],
        [None, UTCDateTime(2023, 6, 3, 8, 29, 27, 140000), 46.233333333333334,
         13.815833333333334, 5.81, 0.71, 12, 187, 6.1, 0.2, 1.0, 2.1, 'C1',
         None],
        [None, UTCDateTime(2023, 6, 3, 12, 33, 19, 650000), 45.750166666666665,
         11.105333333333334, 9.43, 1.2, 13, 127, 14.9, 0.11, 0.4, 1.0, 'B1',
         None],
        [None, UTCDateTime(2023, 6, 3, 16, 54, 12, 690000), 46.3875,
         13.114833333333333, 13.12, 0.51, 14, 99, 7.6, 0.1, 0.4, 0.6, 'B1',
         None],
        [None, UTCDateTime(2023, 6, 3, 21, 13, 45, 630000), 46.07233333333333,
         13.655, 13.59, 0.86, 11, 140, 9.6, 0.07, 0.3, 0.5, 'B1', None],
        [None, UTCDateTime(2023, 6, 3, 21, 45, 26, 960000), 46.503166666666665,
         12.414, 10.73, 1.28, 12, 202, 18.3, 0.11, 0.6, 1.4, 'C1', None],
        [None, UTCDateTime(2023, 6, 4, 0, 3, 1, 710000), 46.1995, 13.4425,
         15.65, 1.03, 20, 96, 7.2, 0.09, 0.3, 0.4, 'B1', None],
        [None, UTCDateTime(2023, 6, 4, 0, 25, 3, 360000), 45.88916666666667,
         13.4015, 13.86, 2.03, 34, 165, 5.8, 0.14, 0.4, 0.6, 'B1', None],
        [None, UTCDateTime(2023, 6, 4, 3, 9, 42, 90000), 46.352,
         12.777333333333333, 6.78, 0.69, 16, 109, 8.4, 0.11, 0.3, 0.8, 'B1',
         None],
        [None, UTCDateTime(2023, 6, 4, 17, 57, 3, 710000), 46.031,
         11.869333333333334, 4.95, 1.72, 22, 64, 15.7, 0.1, 0.2, 1.1, 'B1',
         None]]
    self.assertListEqual(EXPECTED, data)

  @unittest.mock.patch("sys.argv", ["parser.py", "-v", "-d", str(TEST_PATH),
                                    "--file", FILENAME.__str__() + HPL_EXT,
                                    "-S", "CAE", "BAD", "-D", "230601",
                                    "230604"])
  def test_event_parser_hpl(self):
    args = ini.parse_arguments()
    _, stations = ini.station_loader(args)
    source, detect = event_parser_hpl(args.file[0], *args.dates,
                                      stations=stations)
    source = source.values.tolist()
    EXPECTED = [
        [2023000896, UTCDateTime(2023, 6, 1, 7, 39), *([NONE_STR] * 3), 1.69,
         NONE_STR, *([None] * 6), 'explosion ?'],
        [2023000897, UTCDateTime(2023, 6, 1, 8, 51), *([NONE_STR] * 3), 1.02,
         NONE_STR, *([None] * 6), 'explosion ?'],
        [2023000898, UTCDateTime(2023, 6, 1, 9, 27, 56, 980000), 45.942,
         12.377166666666668, 2.09, 0.98, 6.0, *([None] * 6), None],
        [2023000899, UTCDateTime(2023, 6, 1, 9, 53, 27, 930000), 46.176,
         12.502333333333333, 19.15, 1.21, 10.0, *([None] * 6), None],
        [2023000900, UTCDateTime(2023, 6, 1, 10, 2), *([NONE_STR] * 3), 1.86,
         NONE_STR, *([None] * 6), 'explosion ?'],
        [2023000901, UTCDateTime(2023, 6, 1, 14, 59), *([NONE_STR] * 3), 1.54,
         NONE_STR, *([None] * 6), 'explosion ?'],
        [2023000902, UTCDateTime(2023, 6, 1, 19, 8, 21, 300000), 46.4745,
         13.493333333333334, 6.99, 0.55, 10.0, *([None] * 6), None],
        [2023000903, UTCDateTime(2023, 6, 1, 21, 41, 13, 560000),
         46.096833333333336, 12.3035, 12.3, 0.99, 12.0, *([None] * 6), None],
        [2023000904, UTCDateTime(2023, 6, 1, 23, 54, 18, 10000), 46.2965,
         14.465833333333332, 6.14, 1.22, 8.0, *([None] * 6), None],
        [2023000905, UTCDateTime(2023, 6, 2, 2, 18, 24, 150000),
         45.90083333333333, 10.894166666666667, 14.72, 1.62, 22.0, *
         ([None] * 6),
            None],
        [2023000906, UTCDateTime(2023, 6, 2, 8, 50), *([NONE_STR] * 3), 1.46,
         NONE_STR, *([None] * 6), 'explosion ?'],
        [2023000907, UTCDateTime(2023, 6, 2, 9, 6), *([NONE_STR] * 3), 1.65,
         NONE_STR, *([None] * 6), 'explosion ?'],
        [2023000908, UTCDateTime(2023, 6, 3, 0, 31, 35, 710000),
         46.38433333333333, 13.031, 10.83, 0.44, 12.0, *([None] * 6), None],
        [2023000909, UTCDateTime(2023, 6, 3, 5, 18, 30, 810000),
         45.6645, 10.626333333333333, 10.53, 1.6, 19.0, *([None] * 6), None],
        [2023000910, UTCDateTime(2023, 6, 3, 8, 29, 27, 140000),
         46.233333333333334, 13.815833333333334, 5.81, 0.71, 12.0,
         *([None] * 6), None],
        [2023000911, UTCDateTime(2023, 6, 3, 12, 33, 19, 650000),
         45.750166666666665, 11.105333333333334, 9.43, 1.2, 13.0,
         *([None] * 6), None],
        [2023000912, UTCDateTime(2023, 6, 3, 16, 54, 12, 690000), 46.3875,
         13.114833333333333, 13.12, 0.51, 14.0, *([None] * 6), None],
        [2023000913, UTCDateTime(2023, 6, 3, 21, 13, 45, 630000),
         46.07233333333333, 13.655, 13.59, 0.86, 11.0, *([None] * 6), None],
        [2023000914, UTCDateTime(2023, 6, 3, 21, 45, 26, 960000),
         46.503166666666665, 12.414, 10.73, 1.28, 12.0, *([None] * 6),
         None],
        [2023000915, UTCDateTime(2023, 6, 4, 0, 3, 1, 710000), 46.1995,
         13.4425, 15.65, 1.03, 20.0, *([None] * 6), None],
        [2023000916, UTCDateTime(2023, 6, 4, 0, 25, 3, 360000),
         45.88916666666667, 13.4015, 13.86, 2.03, 34.0, *([None] * 6),
         None],
        [2023000917, UTCDateTime(2023, 6, 4, 2, 28), *([NONE_STR] * 3), 0.59,
         NONE_STR, *([None] * 6), None],
        [2023000918, UTCDateTime(2023, 6, 4, 3, 9, 42, 90000), 46.352,
         12.777333333333333, 6.78, 0.69, 16.0, *([None] * 6), None],
        [2023000919, UTCDateTime(2023, 6, 4, 17, 57, 3, 710000), 46.031,
         11.869333333333334, 4.95, 1.72, 22.0, *([None] * 6), None]]
    self.assertListEqual(EXPECTED, source)
    detect = detect.values.tolist()
    EXPECTED = [
        [2023000898.0, UTCDateTime(2023, 6, 1, 9, 27, 58, 710000), 0, PWAVE,
         None, 'CAE'],
        [2023000898.0, UTCDateTime(2023, 6, 1, 9, 27, 59, 360000), 2, SWAVE,
            None, 'CAE'],
        [2023000899.0, UTCDateTime(2023, 6, 1, 9, 53, 32, 530000), 2, PWAVE,
            None, 'CAE'],
        [2023000899.0, UTCDateTime(2023, 6, 1, 9, 53, 36, 360000), 3, SWAVE,
            None, 'CAE'],
        [2023000903.0, UTCDateTime(2023, 6, 1, 21, 41, 16, 740000), 0, PWAVE,
            None, 'CAE'],
        [2023000903.0, UTCDateTime(2023, 6, 1, 21, 41, 19, 90000), 0, SWAVE,
            None, 'CAE'],
        [2023000908.0, UTCDateTime(2023, 6, 3, 0, 31, 40, 50000), 2, PWAVE,
            None, 'BAD'],
        [2023000908.0, UTCDateTime(2023, 6, 3, 0, 31, 43, 560000), 1, SWAVE,
            None, 'BAD'],
        [2023000912.0, UTCDateTime(2023, 6, 3, 16, 54, 16, 760000), 2, PWAVE,
            None, 'BAD'],
        [2023000912.0, UTCDateTime(2023, 6, 3, 16, 54, 19, 950000), 2, SWAVE,
            None, 'BAD'],
        [2023000915.0, UTCDateTime(2023, 6, 4, 0, 3, 5, 450000), 0, PWAVE,
            None, 'BAD'],
        [2023000915.0, UTCDateTime(2023, 6, 4, 0, 3, 8, 340000), 2, SWAVE,
            None, 'BAD'],
        [2023000916.0, UTCDateTime(2023, 6, 4, 0, 25, 10, 760000), 1, PWAVE,
            None, 'BAD'],
        [2023000916.0, UTCDateTime(2023, 6, 4, 0, 25, 16, 420000), 2, SWAVE,
            None, 'BAD'],
        [2023000919.0, UTCDateTime(2023, 6, 4, 17, 57, 11, 390000), 3, PWAVE,
            None, 'CAE'],
        [2023000919.0, UTCDateTime(2023, 6, 4, 17, 57, 17, 250000), 2, SWAVE,
            None, 'CAE']
    ]
    self.assertListEqual(EXPECTED, detect)

  def test_event_parser_hpc(self):
    pass

  def test_event_parser_qml(self):
    pass

  @unittest.mock.patch("sys.argv", ["parser.py", "-v", "-d", str(TEST_PATH),
                                    "--file", MNL_DATA_PATH.__str__(),
                                    "-S", "CAE", "BAD", "-D", "230601",
                                    "230604"])
  def test_event_parser_folder(self):
    args = ini.parse_arguments()
    _, STATIONS = ini.station_loader(args)
    source, detect = event_parser(args.file[0], *args.dates,
                                  stations=STATIONS)
    source = source.values.tolist()
    EXPECTED = [
        [2023000898.0, UTCDateTime(2023, 6, 1, 9, 27, 56, 980000), 45.942,
         12.377166666666668, 2.09, 0.98, 6, 347, 8.8, 0.28, 3.0, 4.7, 'D1',
         None],
        [2023000899.0, UTCDateTime(2023, 6, 1, 9, 53, 27, 930000), 46.176,
         12.502333333333333, 19.15, 1.21, 10, 125, 9.1, 0.12, 0.9, 0.9, 'B1',
         None],
        [2023000903.0, UTCDateTime(2023, 6, 1, 21, 41, 13, 560000),
         46.096833333333336, 12.3035, 12.3, 0.99, 12, 131, 7.6, 0.15, 0.6, 1.0,
         'B1', None],
        [2023000908.0, UTCDateTime(2023, 6, 3, 0, 31, 35, 710000),
         46.38433333333333, 13.031, 10.83, 0.44, 12, 137, 4.0, 0.03, 0.1, 0.2,
         'B1', None],
        [2023000912.0, UTCDateTime(2023, 6, 3, 16, 54, 12, 690000), 46.3875,
         13.114833333333333, 13.12, 0.51, 14, 99, 7.6, 0.1, 0.4, 0.6, 'B1',
         None],
        [2023000915.0, UTCDateTime(2023, 6, 4, 0, 3, 1, 710000), 46.1995,
         13.4425, 15.65, 1.03, 20, 96, 7.2, 0.09, 0.3, 0.4, 'B1', None],
        [2023000916.0, UTCDateTime(2023, 6, 4, 0, 25, 3, 360000),
         45.88916666666667, 13.4015, 13.86, 2.03, 34, 165, 5.8, 0.14, 0.4, 0.6,
         'B1', None],
        [2023000919.0, UTCDateTime(2023, 6, 4, 17, 57, 3, 710000), 46.031,
         11.869333333333334, 4.95, 1.72, 22, 64, 15.7, 0.1, 0.2, 1.1, 'B1',
         None]]
    self.assertListEqual(EXPECTED, source)
    detect = detect.values.tolist()
    EXPECTED = [
        [2023000898.0, UTCDateTime(2023, 6, 1, 9, 27, 58, 710000), 0, PWAVE,
         None, 'CAE'],
        [2023000898.0, UTCDateTime(2023, 6, 1, 9, 27, 59, 360000), 2, SWAVE,
            None, 'CAE'],
        [2023000899.0, UTCDateTime(2023, 6, 1, 9, 53, 32, 530000), 2, PWAVE,
            None, 'CAE'],
        [2023000899.0, UTCDateTime(2023, 6, 1, 9, 53, 36, 360000), 3, SWAVE,
            None, 'CAE'],
        [2023000903.0, UTCDateTime(2023, 6, 1, 21, 41, 16, 740000), 0, PWAVE,
            None, 'CAE'],
        [2023000903.0, UTCDateTime(2023, 6, 1, 21, 41, 19, 90000), 0, SWAVE,
            None, 'CAE'],
        [2023000908.0, UTCDateTime(2023, 6, 3, 0, 31, 40, 50000), 2, PWAVE,
            None, 'BAD'],
        [2023000908.0, UTCDateTime(2023, 6, 3, 0, 31, 43, 560000), 1, SWAVE,
            None, 'BAD'],
        [2023000912.0, UTCDateTime(2023, 6, 3, 16, 54, 16, 760000), 2, PWAVE,
            None, 'BAD'],
        [2023000912.0, UTCDateTime(2023, 6, 3, 16, 54, 19, 950000), 2, SWAVE,
            None, 'BAD'],
        [2023000915.0, UTCDateTime(2023, 6, 4, 0, 3, 5, 450000), 0, PWAVE,
            None, 'BAD'],
        [2023000915.0, UTCDateTime(2023, 6, 4, 0, 3, 8, 340000), 2, SWAVE,
            None, 'BAD'],
        [2023000916.0, UTCDateTime(2023, 6, 4, 0, 25, 10, 760000), 1, PWAVE,
            None, 'BAD'],
        [2023000916.0, UTCDateTime(2023, 6, 4, 0, 25, 16, 420000), 2, SWAVE,
            None, 'BAD'],
        [2023000919.0, UTCDateTime(2023, 6, 4, 17, 57, 11, 390000), 3, PWAVE,
            None, 'CAE'],
        [2023000919.0, UTCDateTime(2023, 6, 4, 17, 57, 17, 250000), 2, SWAVE,
            None, 'CAE']
    ]
    self.assertListEqual(EXPECTED, detect)


if __name__ == "__main__":
  unittest.main()

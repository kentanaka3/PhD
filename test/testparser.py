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
import initializer as ini

DATA_PATH = Path(PRJ_PATH, "data", "test")
TEST_PATH = Path(DATA_PATH, "waveforms")
MNL_DATA_PATH = Path(DATA_PATH, "manual")

FILENAME = Path(MNL_DATA_PATH, "RSFVG-2023")

class TestParser(unittest.TestCase):
  @unittest.mock.patch("sys.argv", ["parser.py", "-v", "-d", str(TEST_PATH),
                                    "--file", FILENAME.__str__() + DAT_EXT,
                                    "-S", "CAE", "BAD"])
  def test_event_parser_dat(self):
    args = ini.parse_arguments()
    data = event_parser_dat(args.file, *args.dates, True,
                            args.station).values.tolist()
    EXPECTED = [
      [2, 'CAE', PWAVE, UTCDateTime(2023, 6, 1, 9, 27, 58, 710000), 0],
      [2, 'CAE', SWAVE, UTCDateTime(2023, 6, 1, 9, 27, 59, 360000), 2],
      [3, 'CAE', PWAVE, UTCDateTime(2023, 6, 1, 9, 53, 32, 530000), 2],
      [3, 'CAE', SWAVE, UTCDateTime(2023, 6, 1, 9, 53, 36, 360000), 3],
      [7, 'CAE', PWAVE, UTCDateTime(2023, 6, 1, 21, 41, 16, 740000), 0],
      [7, 'CAE', SWAVE, UTCDateTime(2023, 6, 1, 21, 41, 19, 90000), 0],
      [10, 'BAD', PWAVE, UTCDateTime(2023, 6, 2, 8, 50, 46, 420000), 2],
      [12, 'BAD', PWAVE, UTCDateTime(2023, 6, 3, 0, 31, 40, 50000), 2],
      [12, 'BAD', SWAVE, UTCDateTime(2023, 6, 3, 0, 31, 43, 560000), 1],
      [16, 'BAD', PWAVE, UTCDateTime(2023, 6, 3, 16, 54, 16, 760000), 2],
      [16, 'BAD', SWAVE, UTCDateTime(2023, 6, 3, 16, 54, 19, 950000), 2],
      [19, 'BAD', PWAVE, UTCDateTime(2023, 6, 4, 0, 3, 5, 450000), 0],
      [19, 'BAD', SWAVE, UTCDateTime(2023, 6, 4, 0, 4, 23, 400000), 2],
      [20, 'BAD', PWAVE, UTCDateTime(2023, 6, 4, 0, 25, 10, 760000), 1],
      [20, 'BAD', SWAVE, UTCDateTime(2023, 6, 4, 0, 25, 16, 420000), 2],
      [23, 'CAE', PWAVE, UTCDateTime(2023, 6, 4, 17, 57, 11, 390000), 3],
      [23, 'CAE', SWAVE, UTCDateTime(2023, 6, 4, 17, 57, 17, 250000), 2]
    ]
    self.assertListEqual(EXPECTED, data)

  @unittest.mock.patch("sys.argv", ["parser.py", "-v", "-d", str(TEST_PATH),
                                    "--file", FILENAME.__str__() + PUN_EXT])
  def test_event_parser_pun(self):
    args = ini.parse_arguments()
    data = event_parser_pun(args.file, *args.dates, True).values.tolist()
    EXPECTED = [
      [UTCDateTime(2023, 6, 1, 9, 27, 56, 980000), '45-56.52', '12-22.63',
       2.09, 0.98, 6, 347, 8.8, 0.28, 3.0, 4.7, 'D1'],
      [UTCDateTime(2023, 6, 1, 9, 53, 27, 930000), '46-10.56', '12-30.14',
       19.15, 1.21, 10, 125, 9.1, 0.12, 0.9, 0.9, 'B1'],
      [UTCDateTime(2023, 6, 1, 19, 8, 21, 300000), '46-28.47', '13-29.60',
       6.99, 0.55, 10, 137, 2.7, 0.08, 0.4, 0.6, 'B1'],
      [UTCDateTime(2023, 6, 1, 21, 41, 13, 560000), '46-05.81', '12-18.21',
       12.3, 0.99, 12, 131, 7.6, 0.15, 0.6, 1.0, 'B1'],
      [UTCDateTime(2023, 6, 1, 23, 54, 18, 10000), '46-17.79', '14-27.95',
       6.14, 1.22, 8, 301, 1.8, 0.05, 0.6, 0.3, 'C1'],
      [UTCDateTime(2023, 6, 2, 2, 18, 24, 150000), '45-54.05', '10-53.65',
       14.72, 1.62, 22, 74, 7.6, 0.26, 0.6, 1.1, 'B1'],
      [UTCDateTime(2023, 6, 3, 0, 31, 35, 710000), '46-23.06', '13-01.86',
       10.83, 0.44, 12, 137, 4.0, 0.03, 0.1, 0.2, 'B1'],
      [UTCDateTime(2023, 6, 3, 5, 18, 30, 810000), '45-39.87', '10-37.58',
       10.53, 1.6, 19, 147, 4.0, 0.1, 0.3, 0.5, 'B1'],
      [UTCDateTime(2023, 6, 3, 8, 29, 27, 140000), '46-14.00', '13-48.95',
       5.81, 0.71, 12, 187, 6.1, 0.2, 1.0, 2.1, 'C1'],
      [UTCDateTime(2023, 6, 3, 12, 33, 19, 650000), '45-45.01', '11-06.32',
       9.43, 1.2, 13, 127, 14.9, 0.11, 0.4, 1.0, 'B1'],
      [UTCDateTime(2023, 6, 3, 16, 54, 12, 690000), '46-23.25', '13-06.89',
       13.12, 0.51, 14, 99, 7.6, 0.1, 0.4, 0.6, 'B1'],
      [UTCDateTime(2023, 6, 3, 21, 13, 45, 630000), '46-04.34', '13-39.30',
       13.59, 0.86, 11, 140, 9.6, 0.07, 0.3, 0.5, 'B1'],
      [UTCDateTime(2023, 6, 3, 21, 45, 26, 960000), '46-30.19', '12-24.84',
       10.73, 1.28, 12, 202, 18.3, 0.11, 0.6, 1.4, 'C1'],
      [UTCDateTime(2023, 6, 4, 0, 3, 1, 710000), '46-11.97', '13-26.55',
       15.65, 1.03, 20, 96, 7.2, 0.09, 0.3, 0.4, 'B1'],
      [UTCDateTime(2023, 6, 4, 0, 25, 3, 360000), '45-53.35', '13-24.09',
       13.86, 2.03, 34, 165, 5.8, 0.14, 0.4, 0.6, 'B1'],
      [UTCDateTime(2023, 6, 4, 3, 9, 42, 90000), '46-21.12', '12-46.64',
       6.78, 0.69, 16, 109, 8.4, 0.11, 0.3, 0.8, 'B1'],
      [UTCDateTime(2023, 6, 4, 17, 57, 3, 710000), '46-01.86', '11-52.16',
       4.95, 1.72, 22, 64, 15.7, 0.1, 0.2, 1.1, 'B1']
    ]
    self.assertListEqual(EXPECTED, data)

  @unittest.mock.patch("sys.argv", ["parser.py", "-v", "-d", str(TEST_PATH),
                                    "--file", FILENAME.__str__() + HPL_EXT,
                                    "-S", "CAE", "BAD"])
  def test_event_parser_hpl(self):
    args = ini.parse_arguments()
    data = event_parser_hpl(args.file, *args.dates, True,
                            args.station).values.tolist()
    print(data)

if __name__ == "__main__": unittest.main()
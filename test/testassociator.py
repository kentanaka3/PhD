#!/bin/python
import unittest
import initializer as ini
import associator as asc
from obspy.core.utcdatetime import UTCDateTime
from mpi4py import MPI
import copy
import sys
import os
from pathlib import Path
PRJ_PATH = Path(os.path.dirname(__file__)).parent
SRC_PATH = os.path.join(PRJ_PATH, "src")
INC_PATH = os.path.join(PRJ_PATH, "inc")
# Add to path
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)


DATA_PATH = Path(PRJ_PATH, "data", "test")
TEST_PATH = Path(DATA_PATH, "waveforms")
"""
class TestAssociator(unittest.TestCase):
  @unittest.mock.patch("sys.argv", ["associator.py", "-D", "230601", "230604"])
  def test_dates2associate_1(self):
    args = ini.parse_arguments()
    if asc.MPI_RANK == 0:
      size = copy.deepcopy(asc.MPI_SIZE)
      asc.MPI_SIZE = 1
      dates = asc.dates2associate(args)
      self.assertEqual(dates, (UTCDateTime(2023, 6, 1, 0, 0, 0),
                               UTCDateTime(2023, 6, 4, 0, 0, 0)))
      asc.MPI_SIZE = size

  @unittest.mock.patch("sys.argv", ["associator.py", "-D", "230601", "230604"])
  def test_dates2associate_2_divisible(self):
    args = ini.parse_arguments()
    size = copy.deepcopy(asc.MPI_SIZE)
    asc.MPI_SIZE = 2
    dates = asc.dates2associate(args)
    if asc.MPI_RANK == 0:
      self.assertEqual(dates, (UTCDateTime(2023, 6, 1, 0, 0, 0),
                               UTCDateTime(2023, 6, 2, 0, 0, 0)))
    elif asc.MPI_RANK == 1:
      self.assertEqual(dates, (UTCDateTime(2023, 6, 3, 0, 0, 0),
                               UTCDateTime(2023, 6, 4, 0, 0, 0)))
    asc.MPI_SIZE = size

  @unittest.mock.patch("sys.argv", ["associator.py", "-D", "230601", "230605"])
  def test_dates2associate_2_undivisible(self):
    args = ini.parse_arguments()
    size = copy.deepcopy(asc.MPI_SIZE)
    asc.MPI_SIZE = 2
    dates = asc.dates2associate(args)
    if asc.MPI_RANK == 0:
      self.assertEqual(dates, (UTCDateTime(2023, 6, 1, 0, 0, 0),
                               UTCDateTime(2023, 6, 3, 0, 0, 0)))
    elif asc.MPI_RANK == 1:
      self.assertEqual(dates, (UTCDateTime(2023, 6, 4, 0, 0, 0),
                               UTCDateTime(2023, 6, 5, 0, 0, 0)))
    asc.MPI_SIZE = size

  @unittest.mock.patch("sys.argv", ["associator.py", "-D", "230601", "230605"])
  def test_dates2associate_2_undivisible(self):
    args = ini.parse_arguments()
    size = copy.deepcopy(asc.MPI_SIZE)
    asc.MPI_SIZE = 4
    dates = asc.dates2associate(args)
    if asc.MPI_RANK == 0:
      self.assertEqual(dates, (UTCDateTime(2023, 6, 1, 0, 0, 0),
                               UTCDateTime(2023, 6, 2, 0, 0, 0)))
    elif asc.MPI_RANK == 1:
      self.assertEqual(dates, (UTCDateTime(2023, 6, 3, 0, 0, 0),
                               UTCDateTime(2023, 6, 3, 0, 0, 0)))
    elif asc.MPI_RANK == 2:
      self.assertEqual(dates, (UTCDateTime(2023, 6, 4, 0, 0, 0),
                               UTCDateTime(2023, 6, 4, 0, 0, 0)))
    elif asc.MPI_RANK == 3:
      self.assertEqual(dates, (UTCDateTime(2023, 6, 5, 0, 0, 0),
                               UTCDateTime(2023, 6, 5, 0, 0, 0)))

    asc.MPI_SIZE = size

if __name__ == "__main__":
  asc.MPI_COMM = MPI.COMM_WORLD
  asc.MPI_RANK = asc.MPI_COMM.Get_rank()
  asc.MPI_SIZE = asc.MPI_COMM.Get_size()
  unittest.main()
"""


if __name__ == "__main__":
    unittest.main()

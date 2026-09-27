"""
===============================================================================
OGS DAT Parser Test Suite - Unit Tests for Bulletin Phase Pick Extraction
===============================================================================

OVERVIEW:
Unit test suite for ``ogsdat.py``. Validates command-line argument parsing and
extraction of seismic phase picks from legacy OGS ``.dat`` bulletin files.

TEST CASES & INVARIANTS:
  1. test_args: Validates CLI argument parsing, input path resolution, and
     output configuration.
  2. test_read: Tests fixed-width parsing of station codes, phase types (P/S),
     first-motion polarities, pick timestamps, and pick weightings into
     standardized DataFrames.

USAGE:
python -m unittest OGS/test/testogsdat.py

DEPENDENCIES:
- unittest / unittest.mock: test runner and filesystem isolation
  - pandas: parsed DataFrame structure verification
  - ogsdat: parser module under test

AUTHORS:
  - 健
  - Istituto Nazionale di Oceanografia e di Geofisica Sperimentale (OGS)
    Centro di Ricerche Sismologiche (CRS)
  - Università degli Studi di Trieste (UniTS)
    Dipartimento di Matematica, Informatica e Geoscienze (MIGe)
    Applied Data Science and Artificial Intelligence (ADSAI)
  - Terabit Network for Research and Academic Big Data in Italy (TeRABIT)
    Consorzio Interuniversitario del Nord-Est per il Calcolo Automatico (CINECA)

===============================================================================
"""

import ogsconstants as OGS_C
from ogsutils import parse_dat_args
from ogsdat import DataFileDAT
import pandas as pd
import os
from pathlib import Path
import unittest.mock
from datetime import datetime

THIS_DIR = os.path.dirname(__file__)


DATA_DIR = Path(os.path.abspath(THIS_DIR + "/../data"))
DATA_FILE = "onlyEQ-2024.dat"


class TestOGSDAT(unittest.TestCase):
  @unittest.mock.patch("sys.argv", [
      "ogsdat.py", "-D", "20240320", "20240620",
      "-f", str(DATA_DIR / "manual" / DATA_FILE),
      "-v"
  ])
  def test_args(self):
    args = parse_dat_args()
    self.assertEqual(args.file, [Path(DATA_DIR / "manual" / DATA_FILE)])
    self.assertEqual(args.dates[0],
                     datetime.strptime("20240320", OGS_C.YYYYMMDD_FMT))
    self.assertEqual(args.dates[1],
                     datetime.strptime("20240620", OGS_C.YYYYMMDD_FMT))
    self.assertTrue(args.verbose)

  @unittest.mock.patch("sys.argv", [
      "ogsdat.py", "-f", str(DATA_DIR / "manual" / DATA_FILE),
      "-q"
  ])
  def test_quiet_arg(self):
    args = parse_dat_args()
    self.assertTrue(args.quiet)
    self.assertFalse(args.verbose)

  @unittest.mock.patch("sys.argv", [
      "ogsdat.py", "-J", "2024172", "2024080",
      "-f", str(DATA_DIR / "manual" / DATA_FILE),
  ])
  def test_julian_dates_arg(self):
    args = parse_dat_args()
    self.assertEqual(args.dates[0],
                     datetime.strptime("20240320", OGS_C.YYYYMMDD_FMT))
    self.assertEqual(args.dates[1],
                     datetime.strptime("20240620", OGS_C.YYYYMMDD_FMT))

  def test_read(self):
    print()
    input_file = DATA_DIR / "manual" / DATA_FILE
    if not input_file.is_file():
      self.skipTest(f"Sample data file {input_file} not found")
    datafile = DataFileDAT(
        input_file,
        start=datetime.strptime("20240101", OGS_C.YYYYMMDD_FMT),
        end=datetime.strptime("20241231", OGS_C.YYYYMMDD_FMT),
        verbose=True
    )
    datafile.read()
    filepath = Path(THIS_DIR) / "OGSCatalog" / (DATA_FILE + ".parquet")
    # datafile.PICKS.to_parquet(filepath)
    expected = pd.read_parquet(filepath)
    pd.testing.assert_frame_equal(
        datafile.PICKS.reset_index(drop=True),
        expected.reset_index(drop=True)
    )


if __name__ == "__main__":
  unittest.main()

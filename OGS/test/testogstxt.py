"""
===============================================================================
OGS TXT Parser Test Suite - Unit Tests for Event Summary Bulletin Parsing
===============================================================================

OVERVIEW:
Unit test suite for ``ogstxt.py``. Validates CLI argument handling and parsing
of published OGS summary bulletin text files (``.txt`` format).

TEST CASES & INVARIANTS:
  1. test_args: Validates CLI parser configuration and date filtering
     parameters.
  2. test_read: Validates regex extraction of earthquake origin times,
     hypocenter locations, depths, durations, and local magnitudes ($M_L$).

USAGE:
python -m unittest OGS/test/testogstxt.py

DEPENDENCIES:
- unittest / unittest.mock: test runner and filesystem mocking
  - pandas: parsed DataFrame structure verification
  - ogstxt: .txt summary bulletin parser under test

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
from ogstxt import DataFileTXT
from ogsutils import parse_txt_args
import os
import unittest.mock
from datetime import datetime
from pathlib import Path
import pandas as pd

THIS_DIR = os.path.dirname(__file__)

DATA_DIR = Path(os.path.abspath(THIS_DIR + "/../data"))
DATA_FILE = "onlyEQ-2024.txt"
TEST_FILE = Path(__file__).resolve()


class TestOGSTXT(unittest.TestCase):
  @unittest.mock.patch("sys.argv", [
      "ogstxt.py", "-D", "20240320", "20240620",
      "-f", str(TEST_FILE),
      "-v"
  ])
  def test_args(self):
    args = parse_txt_args()
    self.assertEqual(args.file, [Path(DATA_DIR / "manual" / DATA_FILE)])
    self.assertEqual(args.dates[0],
                     datetime.strptime("20240320", OGS_C.YYYYMMDD_FMT))
    self.assertEqual(args.dates[1],
                     datetime.strptime("20240620", OGS_C.YYYYMMDD_FMT))
    self.assertTrue(args.verbose)

  @unittest.mock.patch("sys.argv", [
      "ogstxt.py", "-f", str(DATA_DIR / "manual" / DATA_FILE),
      "-q"
  ])
  def test_quiet_arg(self):
    args = parse_txt_args()
    self.assertTrue(args.quiet)
    self.assertFalse(args.verbose)

  def test_read(self):
    print()
    input_file = DATA_DIR / "manual" / DATA_FILE
    if not input_file.is_file():
      self.skipTest(f"Sample data file {input_file} not found")
    datafile = DataFileTXT(
        input_file,
        start=datetime.strptime("20240101", OGS_C.YYYYMMDD_FMT),
        end=datetime.strptime("20241231", OGS_C.YYYYMMDD_FMT),
        verbose=True
    )
    datafile.read()
    filepath = Path(THIS_DIR) / "OGSCatalog" / (DATA_FILE + ".parquet")
    # datafile.EVENTS.to_parquet(filepath)
    expected = pd.read_parquet(filepath)
    pd.testing.assert_frame_equal(
        datafile.EVENTS.reset_index(drop=True),
        expected.reset_index(drop=True)
    )


if __name__ == "__main__":
  unittest.main()

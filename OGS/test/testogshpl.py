"""
=============================================================================
OGS HPL Parser Test Suite - Unit Tests for Hypo71 Phase & Event Extraction
=============================================================================

OVERVIEW:
Unit test suite for ``ogshpl.py``. Validates command-line argument handling and
fixed-format parsing of legacy Hypo71 ``.hpl`` bulletin files.

TEST CASES & INVARIANTS:
  1. test_args: Validates CLI parser flags, date bounds, and path resolution.
  2. test_read: Validates parsing of paired Hypo71 event hypocenters (origin
     time, latitude, longitude, depth, magnitude) and associated station phase
     arrivals (P/S travel times, residuals, weights).

USAGE:
python -m unittest OGS/test/testogshpl.py

DEPENDENCIES:
- unittest / unittest.mock: test runner and environment mocking
  - pandas: tabular data structure verification
  - ogshpl: Hypo71 .hpl parser under test

AUTHORS:
  - 健
  - Istituto Nazionale di Oceanografia e di Geofisica Sperimentale (OGS)
    Centro di Ricerche Sismologiche (CRS)
  - Università degli Studi di Trieste (UniTS)
    Dipartimento di Matematica, Informatica e Geoscienze (MIGe)
    Applied Data Science and Artificial Intelligence (ADSAI)
  - Terabit Network for Research and Academic Big Data in Italy (TeRABIT)
    Consorzio Interuniversitario del Nord-Est per il Calcolo Automatico (CINECA)
=============================================================================
"""

from ogshpl import DataFileHPL, parse_arguments
import ogsconstants as OGS_C
from datetime import datetime
import os
import sys
import pandas as pd
import unittest.mock
from pathlib import Path
THIS_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(THIS_DIR + "/../src"))


DATA_DIR = Path(os.path.abspath(THIS_DIR + "/../data"))
DATA_FILES = [
    "onlyEQ-2024.hpl",
    "onlyEQ_NLL1D-2024.hpl"
]


class TestOGSHPL(unittest.TestCase):
  @unittest.mock.patch("sys.argv", [
      "ogshpl.py", "-D", "20240320", "20240620",
      "-f", str(DATA_DIR / "manual" / "onlyEQ-2024.hpl"),
      "-v"
  ])
  def test_args(self):
    args = parse_arguments()
    self.assertEqual(args.file,
                     [Path(DATA_DIR / "manual" / "onlyEQ-2024.hpl")])
    self.assertEqual(args.dates[0],
                     datetime.strptime("20240320", OGS_C.YYYYMMDD_FMT))
    self.assertEqual(args.dates[1],
                     datetime.strptime("20240620", OGS_C.YYYYMMDD_FMT))
    self.assertTrue(args.verbose)

  def test_read(self):
    print()
    for file in DATA_FILES:
      input_file = DATA_DIR / "manual" / file
      if not input_file.is_file():
        self.skipTest(f"Sample data file {input_file} not found")
      datafile = DataFileHPL(
          input_file,
          start=datetime.strptime("20240101", OGS_C.YYYYMMDD_FMT),
          end=datetime.strptime("20241231", OGS_C.YYYYMMDD_FMT),
          verbose=True
      )
      datafile.read()
      filepath = Path(THIS_DIR) / "OGSCatalog" / (file + ".parquet")
      # datafile.EVENTS.to_parquet(filepath)
      expected = pd.read_parquet(filepath)
      pd.testing.assert_frame_equal(
          datafile.EVENTS.reset_index(drop=True),
          expected.reset_index(drop=True)
      )


if __name__ == "__main__":
  unittest.main()

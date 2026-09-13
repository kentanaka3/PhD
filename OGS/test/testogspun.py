"""
=============================================================================
OGS PUN Parser Test Suite - Unit Tests for Hypo71 Summary Extraction
=============================================================================

OVERVIEW:
Unit test suite for ``ogspun.py``. Validates command-line arguments and
extraction of earthquake hypocenters and statistical quality parameters from
Hypo71 punch-card ``.pun`` bulletin files.

TEST CASES & INVARIANTS:
  1. test_args: Validates argument parsing and output directory resolution.
  2. test_read: Validates fixed-column extraction of origin times, coordinates,
     depths, magnitudes, number of stations, azimuthal gap, RMS travel-time
     residuals, and horizontal/vertical error bounds (ERH, ERZ).

USAGE:
python -m unittest OGS/test/testogspun.py

DEPENDENCIES:
- unittest / unittest.mock: test runner and mock isolation
  - pandas: extracted event DataFrame validation
  - ogspun: Hypo71 .pun parser under test

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

from ogspun import DataFilePUN, parse_arguments
import ogsconstants as OGS_C
from datetime import datetime
import os
import sys
import pandas as pd
import unittest
import unittest.mock
from pathlib import Path
THIS_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(THIS_DIR + "/../src"))


DATA_DIR = Path(os.path.abspath(THIS_DIR + "/../data"))
DATA_FILE = "onlyEQ-2024.pun"


class TestOGSPUN(unittest.TestCase):
  @unittest.mock.patch("sys.argv", [
      "ogspun.py", "-D", "20240320", "20240620",
      "-f", str(DATA_DIR / "manual" / "onlyEQ-2024.pun"),
      "-v"
  ])
  def test_args(self):
    args = parse_arguments()
    self.assertEqual(
        args.file, [Path(DATA_DIR / "manual" / "onlyEQ-2024.pun")]
    )
    self.assertEqual(args.dates[0],
                     datetime.strptime("20240320", OGS_C.YYYYMMDD_FMT))
    self.assertEqual(args.dates[1],
                     datetime.strptime("20240620", OGS_C.YYYYMMDD_FMT))
    self.assertTrue(args.verbose)

  def test_read(self):
    print()
    input_file = DATA_DIR / "manual" / DATA_FILE
    if not input_file.is_file():
      self.skipTest(f"Sample data file {input_file} not found")
    datafile = DataFilePUN(
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

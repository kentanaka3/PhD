"""
=============================================================================
OGS Parser Test Suite - Unit Tests for Catalog Parsing & Registration
=============================================================================

OVERVIEW:
Unit test suite for ``ogsparser.py``. Validates CLI argument handling, catalog
format auto-detection, and ``DataCatalog`` file registration across disparate
bulletin formats (.dat, .hpl, .pun, .txt).

TEST CASES & INVARIANTS:
  1. test_parse_arguments_file_mode: Validates CLI execution when processing
     individual catalog files.
  2. test_datacatalog_file_registration: Verifies that ``DataCatalog`` correctly
     identifies format types and registers input files into parser pipelines.

USAGE:
python -m unittest OGS/test/testogsparser.py

DEPENDENCIES:
- unittest / unittest.mock: test runner and mock frameworks
  - ogsparser: catalog aggregator and parser dispatcher under test
  - ogsconstants: shared formats and file patterns

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

from ogsparser import DataCatalog, parse_arguments
import ogsconstants as OGS_C
import os
import sys
import unittest
import unittest.mock
from pathlib import Path
from datetime import datetime

import pandas as pd

THIS_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(THIS_DIR + "/../src"))


DATA_DIR = Path(os.path.abspath(THIS_DIR + "/../data"))


class TestOGSParser(unittest.TestCase):
  @unittest.mock.patch("os.path.isfile", return_value=True)
  @unittest.mock.patch("sys.argv", [
      "ogsparser.py",
      "-D", "20240320", "20240620",
      "-f", str(DATA_DIR / "manual" / "onlyEQ-2024.hpl"),
      "-v",
      "--merge"
  ])
  def test_parse_arguments_file_mode(self, mock_isfile):
    args = parse_arguments()
    self.assertEqual(
        args.file, [Path(DATA_DIR / "manual" / "onlyEQ-2024.hpl")]
    )
    self.assertEqual(
        args.dates[0], datetime.strptime("20240320", OGS_C.YYYYMMDD_FMT)
    )
    self.assertEqual(
        args.dates[1], datetime.strptime("20240620", OGS_C.YYYYMMDD_FMT)
    )
    self.assertTrue(args.verbose)
    self.assertTrue(args.merge)

  def test_datacatalog_file_registration(self):
    args = unittest.mock.MagicMock()
    args.directory = None
    args.file = [
        Path(DATA_DIR / "manual" / "onlyEQ-2024.hpl"),
        Path(DATA_DIR / "manual" / "onlyEQ-2024.dat"),
    ]
    args.dates = [
        datetime.strptime("20240101", OGS_C.YYYYMMDD_FMT),
        datetime.strptime("20241231", OGS_C.YYYYMMDD_FMT),
    ]
    args.verbose = False
    args.output = str(DATA_DIR / "test_output")

    catalog = DataCatalog(args)
    self.assertIn(OGS_C.HPL_EXT, catalog.DATAFILE_TYPES)
    self.assertIn(OGS_C.DAT_EXT, catalog.DATAFILE_TYPES)
    self.assertIn(OGS_C.TXT_EXT, catalog.DATAFILE_TYPES)
    self.assertIn(OGS_C.PUN_EXT, catalog.DATAFILE_TYPES)


if __name__ == "__main__":
  unittest.main()

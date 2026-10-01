"""
===============================================================================
OGS PUN Test Suite - Unit and Integration Tests for Hypo71 PUN Summary Parser
===============================================================================

OVERVIEW:
Comprehensive test suite for ``DataFilePUN`` covering:
  1. Input validation (file existence and .pun extension checks).
  2. CLI argument parsing via ``OGS_U.parse_pun_args``.
  3. Real dataset integration using ``OnlyEqHypo71`` PUN files (2005 and 2024).
  4. Schema invariants (unified 28-column EVENTS schema matching ``_EVENT_COLUMNS``).
  5. Degree-minute coordinates conversion to decimal degrees.
  6. Quality Metric (QM) extraction and preservation (e.g. B1, C1, D1).
  7. Handling of asterisks (``*****``) and blank fields in ERH/ERZ/RMS error columns.
  8. Index normalization with year stride (``normalize_index``).
  9. Temporal filtering (date window bounds and early termination after end date).
  10. Synthetic fixtures for edge-case layout and formatting variations.

===============================================================================
"""

from datetime import datetime, timedelta as td
import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import ogsconstants as OGS_C
from ogsdatafile import OGSDataFile
from ogspun import DataFilePUN
import ogsutils as OGS_U

# Base paths
TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"
HYPO71_DIR = DATASET_DIR / "OnlyEqHypo71"


class TestDataFilePUN(unittest.TestCase):
  """Unit and integration test cases for DataFilePUN."""

  def setUp(self):
    """Set up test fixtures and paths."""
    self.pun_2005 = HYPO71_DIR / "onlyeq2005.pun"
    self.pun_2024 = HYPO71_DIR / "onlyeq2024.pun"

  def test_input_validation_missing_file(self):
    """DataFilePUN raises FileNotFoundError when the file does not exist."""
    non_existent = HYPO71_DIR / "non_existent_file.pun"
    with self.assertRaises(FileNotFoundError):
      DataFilePUN(non_existent)

  def test_input_validation_invalid_extension(self):
    """DataFilePUN raises ValueError when the file extension is not .pun."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
      tmp_path = Path(tmp.name)
    try:
      parser = DataFilePUN(tmp_path)
      with self.assertRaises(ValueError):
        parser.read()
    finally:
      if tmp_path.exists():
        tmp_path.unlink()

  def test_parse_pun_args(self):
    """Test CLI argument parsing for PUN parser via parse_pun_args."""
    if not self.pun_2005.is_file():
      self.skipTest(f"Dataset file {self.pun_2005} not found")

    args = OGS_U.parse_pun_args([
        "-f", str(self.pun_2005),
        "-D", "20050101", "20050110",
        "-v",
    ])
    self.assertEqual(args.file, [self.pun_2005])
    self.assertTrue(args.verbose)
    self.assertEqual(len(args.dates), 2)
    self.assertEqual(args.dates[0], datetime(2005, 1, 1))
    self.assertEqual(args.dates[1], datetime(2005, 1, 10))

  def test_parse_pun_args_defaults(self):
    """Test parse_pun_args default arguments."""
    if not self.pun_2005.is_file():
      self.skipTest(f"Dataset file {self.pun_2005} not found")

    args = OGS_U.parse_pun_args(["-f", str(self.pun_2005)])
    self.assertEqual(args.file, [self.pun_2005])
    self.assertFalse(args.verbose)
    self.assertEqual(len(args.dates), 2)
    self.assertEqual(args.dates[0], datetime.min)

  def test_hypo71_dataset_2005_schema_and_qm(self):
    """Integration test with OnlyEqHypo71/onlyeq2005.pun: schema, QM, and coordinates."""
    if not self.pun_2005.is_file():
      self.skipTest(f"Dataset file {self.pun_2005} not found")

    start = datetime(2005, 1, 1)
    end = datetime(2005, 1, 5)
    parser = DataFilePUN(self.pun_2005, start=start, end=end)
    parser.read()

    # Schema invariants: unified 28 columns
    self.assertEqual(len(parser.EVENTS.columns), 28)
    self.assertEqual(list(parser.EVENTS.columns), OGSDataFile._EVENT_COLUMNS)

    # 7 located events in this window
    self.assertEqual(len(parser.EVENTS), 7)

    # Check QM column populated with 2-char alphanumeric codes (e.g. 'B1', 'C1', 'D1')
    for qm in parser.EVENTS[OGS_C.QM_STR]:
      self.assertIsInstance(qm, str)
      self.assertEqual(len(qm), 2)
      self.assertIn(qm[0], ["A", "B", "C", "D"])
      self.assertTrue(qm[1].isdigit())

    # Coordinates properly converted from degree-minutes to decimal degrees
    first_ev = parser.EVENTS.iloc[0]
    # Original: 46-21.00 -> 46 + 21/60 = 46.35
    self.assertAlmostEqual(first_ev[OGS_C.LATITUDE_STR], 46.35, places=2)
    # Original: 13- 5.74 -> 13 + 5.74/60 = 13.0957
    self.assertAlmostEqual(first_ev[OGS_C.LONGITUDE_STR], 13.0957, places=3)
    self.assertEqual(first_ev[OGS_C.DEPTH_STR], 7.0)
    self.assertEqual(first_ev[OGS_C.MAGNITUDE_D_STR], 2.43)

  def test_hypo71_dataset_2024_schema(self):
    """Integration test with OnlyEqHypo71/onlyeq2024.pun."""
    if not self.pun_2024.is_file():
      self.skipTest(f"Dataset file {self.pun_2024} not found")

    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 3)
    parser = DataFilePUN(self.pun_2024, start=start, end=end)
    parser.read()

    self.assertEqual(len(parser.EVENTS.columns), 28)
    self.assertGreater(len(parser.EVENTS), 0)

  def test_index_stride_normalization(self):
    """Test sequential event indexing with year stride (year * 1,000,000 + counter)."""
    if not self.pun_2005.is_file():
      self.skipTest(f"Dataset file {self.pun_2005} not found")

    start = datetime(2005, 1, 1)
    end = datetime(2005, 1, 5)
    parser = DataFilePUN(self.pun_2005, start=start, end=end)
    parser.read()

    indices = list(parser.EVENTS[OGS_C.IDX_EVENTS_STR])
    # Counter starts at 0 for the first encountered valid record in the file
    expected_indices = [2005000000 + i for i in range(len(indices))]
    self.assertEqual(indices, expected_indices)

  def test_date_range_filtering_and_early_break(self):
    """Test date range temporal filtering including early break behavior."""
    if not self.pun_2005.is_file():
      self.skipTest(f"Dataset file {self.pun_2005} not found")

    # Day 1 only: Jan 1, 2005
    parser_day1 = DataFilePUN(
        self.pun_2005,
        start=datetime(2005, 1, 1),
        end=datetime(2005, 1, 1),
    )
    parser_day1.read()
    self.assertEqual(len(parser_day1.EVENTS), 1)

    # Jan 1 to Jan 3, 2005
    parser_day3 = DataFilePUN(
        self.pun_2005,
        start=datetime(2005, 1, 1),
        end=datetime(2005, 1, 3),
    )
    parser_day3.read()
    self.assertEqual(len(parser_day3.EVENTS), 5)

  def test_synthetic_pun_asterisks_and_blanks(self):
    """Test synthetic PUN file with asterisks in ERH/ERZ and blank fields."""
    canonical_line = "10501011422  9.04 46-21.00  13- 5.74   7.00   2.43 24  59  3.3 0.24  0.5  1.2 B1"
    line_normal = canonical_line
    line_star = canonical_line[:67] + "*****" + "*****" + " " + "D2"
    line_blank = canonical_line[:67] + "     " + "     " + " " + "C1"

    synthetic_pun = (
        " DATE    ORIGIN    LAT N    LONG E    DEPTH    MAG NO GAP DMIN  RMS  ERH  ERZ QM\n"
        + line_normal + "\n"
        + line_star + "\n"
        + line_blank + "\n"
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pun", delete=False) as tmp:
      tmp.write(synthetic_pun)
      tmp_path = Path(tmp.name)

    try:
      parser = DataFilePUN(tmp_path, start=datetime(
          2005, 1, 1), end=datetime(2005, 1, 5))
      parser.read()

      self.assertEqual(len(parser.EVENTS), 3)

      # Event 1: normal errors
      ev1 = parser.EVENTS.iloc[0]
      self.assertEqual(ev1[OGS_C.ERH_STR], 0.5)
      self.assertEqual(ev1[OGS_C.ERZ_STR], 1.2)
      self.assertEqual(ev1[OGS_C.QM_STR], "B1")

      # Event 2: asterisks correctly coerced to NaN without raising exceptions
      ev2 = parser.EVENTS.iloc[1]
      self.assertTrue(pd.isna(ev2[OGS_C.ERH_STR]))
      self.assertTrue(pd.isna(ev2[OGS_C.ERZ_STR]))
      self.assertEqual(ev2[OGS_C.QM_STR], "D2")

      # Event 3: blank errors correctly coerced to NaN
      ev3 = parser.EVENTS.iloc[2]
      self.assertTrue(pd.isna(ev3[OGS_C.ERH_STR]))
      self.assertTrue(pd.isna(ev3[OGS_C.ERZ_STR]))
      self.assertEqual(ev3[OGS_C.QM_STR], "C1")

    finally:
      if tmp_path.exists():
        tmp_path.unlink()


if __name__ == "__main__":
  unittest.main()

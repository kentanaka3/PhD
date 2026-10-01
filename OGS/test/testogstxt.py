"""
===============================================================================
OGS TXT Test Suite - Unit and Integration Tests for Catalog Event Summary TXT
===============================================================================

OVERVIEW:
Comprehensive test suite for ``DataFileTXT`` covering:
  1. Input validation (file existence and .txt extension checks).
  2. CLI argument parsing via ``OGS_U.parse_txt_args``.
  3. Real dataset integration using ``OnlyEqHypo71`` and ``OnlyEqNLL1D`` TXT files.
  4. Schema invariants (unified 28-column EVENTS schema matching ``_EVENT_COLUMNS``).
  5. ISO datetime parsing (ISO-8601 millisecond timestamp resolution).
  6. Date range temporal filtering (start and end date filtering).
  7. Unlocated event skipping (placeholder dashes ``-------`` and "Not localized").
  8. Event-type classification filtering (skipping chemical explosions, suspected slides).
  9. Synthetic fixtures validating header skipping and filtering edge cases.

===============================================================================
"""

from datetime import datetime
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import ogsconstants as OGS_C
from ogsdatafile import OGSDataFile
from ogstxt import DataFileTXT, DEFAULT_FILTERED_EVENT_TYPES
import ogsutils as OGS_U

# Base paths
TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"
HYPO71_DIR = DATASET_DIR / "OnlyEqHypo71"
NLL1D_DIR = DATASET_DIR / "OnlyEqNLL1D"


class TestDataFileTXT(unittest.TestCase):
  """Unit and integration test cases for DataFileTXT."""

  def setUp(self):
    """Set up test fixtures and paths."""
    self.txt_hypo_2005 = HYPO71_DIR / "onlyeq2005.txt"
    self.txt_hypo_2024 = HYPO71_DIR / "onlyeq2024.txt"
    self.txt_nll_2005 = NLL1D_DIR / "onlyeq2005.nll1D.txt"
    self.txt_nll_2024 = NLL1D_DIR / "onlyeq2024.nll1D.txt"

  def test_input_validation_missing_file(self):
    """DataFileTXT raises FileNotFoundError when the file does not exist."""
    non_existent = HYPO71_DIR / "non_existent_file.txt"
    with self.assertRaises(FileNotFoundError):
      DataFileTXT(non_existent)

  def test_input_validation_invalid_extension(self):
    """DataFileTXT raises ValueError when the file extension is not .txt."""
    with tempfile.NamedTemporaryFile(suffix=".hpl", delete=False) as tmp:
      tmp_path = Path(tmp.name)
    try:
      parser = DataFileTXT(tmp_path)
      with self.assertRaises(ValueError):
        parser.read()
    finally:
      if tmp_path.exists():
        tmp_path.unlink()

  def test_parse_txt_args(self):
    """Test CLI argument parsing for TXT parser via parse_txt_args."""
    if not self.txt_hypo_2005.is_file():
      self.skipTest(f"Dataset file {self.txt_hypo_2005} not found")

    args = OGS_U.parse_txt_args([
        "-f", str(self.txt_hypo_2005),
        "-D", "20050101", "20050110",
        "-v",
    ])
    self.assertEqual(args.file, [self.txt_hypo_2005])
    self.assertTrue(args.verbose)
    self.assertEqual(len(args.dates), 2)
    self.assertEqual(args.dates[0], datetime(2005, 1, 1))
    self.assertEqual(args.dates[1], datetime(2005, 1, 10))

  def test_parse_txt_args_defaults(self):
    """Test parse_txt_args default dates and options."""
    if not self.txt_hypo_2005.is_file():
      self.skipTest(f"Dataset file {self.txt_hypo_2005} not found")

    args = OGS_U.parse_txt_args(["-f", str(self.txt_hypo_2005)])
    self.assertEqual(args.file, [self.txt_hypo_2005])
    self.assertFalse(args.verbose)
    self.assertEqual(len(args.dates), 2)
    self.assertEqual(args.dates[0], datetime.min)

  def test_iso_datetime_parsing(self):
    """Test that ISO datetime strings are correctly parsed into datetime objects."""
    parser = DataFileTXT(self.txt_hypo_2005)
    dt_str = "2005-01-01T14:22:09.040"
    parsed_dt = parser._parse_event_datetime(dt_str)
    expected_dt = datetime(2005, 1, 1, 14, 22, 9, 40000)
    self.assertEqual(parsed_dt, expected_dt)

  def test_hypo71_dataset_2005_schema_and_filtering(self):
    """Integration test with OnlyEqHypo71/onlyeq2005.txt: schema and unlocated skipping."""
    if not self.txt_hypo_2005.is_file():
      self.skipTest(f"Dataset file {self.txt_hypo_2005} not found")

    start = datetime(2005, 1, 1)
    end = datetime(2005, 1, 5)
    parser = DataFileTXT(self.txt_hypo_2005, start=start, end=end)
    parser.read()

    # Schema invariants: unified 28 columns
    self.assertEqual(len(parser.EVENTS.columns), 28)
    self.assertEqual(list(parser.EVENTS.columns), OGSDataFile._EVENT_COLUMNS)

    # 7 located events in this window (unlocated events properly skipped)
    self.assertEqual(len(parser.EVENTS), 7)

    # Ensure unlocated placeholders did not enter latitude/longitude
    self.assertTrue((parser.EVENTS[OGS_C.LATITUDE_STR] > 0).all())
    self.assertTrue((parser.EVENTS[OGS_C.LONGITUDE_STR] > 0).all())

    # Ensure event types are valid
    for etype in parser.EVENTS[OGS_C.EVENT_TYPE_STR]:
      self.assertNotIn(etype, DEFAULT_FILTERED_EVENT_TYPES)

  def test_nll1d_dataset_2005_schema_and_counts(self):
    """Integration test with OnlyEqNLL1D/onlyeq2005.nll1D.txt."""
    if not self.txt_nll_2005.is_file():
      self.skipTest(f"Dataset file {self.txt_nll_2005} not found")

    start = datetime(2005, 1, 1)
    end = datetime(2005, 1, 5)
    parser = DataFileTXT(self.txt_nll_2005, start=start, end=end)
    parser.read()

    self.assertEqual(len(parser.EVENTS.columns), 28)
    self.assertEqual(len(parser.EVENTS), 7)
    self.assertEqual(list(parser.EVENTS.columns), OGSDataFile._EVENT_COLUMNS)

  def test_dataset_2024_schema_hypo71_and_nll1d(self):
    """Integration test with 2024 Hypo71 and NLL1D TXT files."""
    if not self.txt_hypo_2024.is_file() or not self.txt_nll_2024.is_file():
      self.skipTest("2024 TXT dataset files not found")

    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 3)

    parser_hypo = DataFileTXT(self.txt_hypo_2024, start=start, end=end)
    parser_hypo.read()
    self.assertEqual(len(parser_hypo.EVENTS.columns), 28)

    parser_nll = DataFileTXT(self.txt_nll_2024, start=start, end=end)
    parser_nll.read()
    self.assertEqual(len(parser_nll.EVENTS.columns), 28)

    self.assertEqual(len(parser_hypo.EVENTS), len(parser_nll.EVENTS))

  def test_date_range_filtering(self):
    """Test start and end date filtering on TXT catalog records."""
    if not self.txt_hypo_2005.is_file():
      self.skipTest(f"Dataset file {self.txt_hypo_2005} not found")

    # Jan 1 only
    parser_day1 = DataFileTXT(
        self.txt_hypo_2005,
        start=datetime(2005, 1, 1),
        end=datetime(2005, 1, 1),
    )
    parser_day1.read()
    self.assertEqual(len(parser_day1.EVENTS), 1)

    # Jan 1 to Jan 3
    parser_day3 = DataFileTXT(
        self.txt_hypo_2005,
        start=datetime(2005, 1, 1),
        end=datetime(2005, 1, 3),
    )
    parser_day3.read()
    self.assertEqual(len(parser_day3.EVENTS), 5)

  def test_synthetic_txt_unlocated_and_event_type_filtering(self):
    """Test synthetic TXT file filtering chemical explosions, suspected slides, and unlocated rows."""
    synthetic_txt = (
        "index event-id     origin_time(UTC)      t_err   lat     lon   h_err depth v_err gap  ml   md  place event_type\n"
        # 1. Unlocated event with dashes and 'Not localized' in place
        "00001 2005_00001 2005-01-01T03:22:00.000 ----- ------- ------- ----- ----- ----- --- ---- ---- Not localized with first station DRE [earthquake]\n"
        # 2. Valid local earthquake
        "00002 2005_00002 2005-01-01T14:22:09.040  0.24 46.3500 13.0957   0.5   7.0   0.5  59 ----  2.4 MOGGIO UDINESE (FRIULI) [earthquake]\n"
        # 3. Filtered chemical explosion
        "00003 2005_00003 2005-01-01T15:30:00.000  0.15 46.2000 13.1000   0.4   1.0   0.4  80 ----  1.5 CAVE DEL PREDIL [chemical explosion]\n"
        # 4. Filtered suspected slide
        "00004 2005_00004 2005-01-01T16:00:00.000  0.18 46.4000 12.8000   0.6   0.5   0.6  90 ----  1.2 VAJONT VALLEY [suspected slide]\n"
        # 5. Filtered suspected explosion
        "00005 2005_00005 2005-01-01T17:00:00.000  0.20 45.9000 13.5000   0.5   2.0   0.5  75 ----  1.8 MONFALCONE QUARRY [suspected explosion]\n"
        # 6. Another valid earthquake
        "00006 2005_00006 2005-01-02T11:56:15.320  0.30 46.3482 13.0905   0.6   7.0   0.6  56 ----  2.7 TOLMEZZO (FRIULI) [earthquake]\n"
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
      tmp.write(synthetic_txt)
      tmp_path = Path(tmp.name)

    try:
      parser = DataFileTXT(tmp_path, start=datetime(
          2005, 1, 1), end=datetime(2005, 1, 5))
      parser.read()

      # Out of 6 rows:
      # - Row 1 is unlocated (skipped)
      # - Row 3 is chemical explosion (skipped)
      # - Row 4 is suspected slide (skipped)
      # - Row 5 is suspected explosion (skipped)
      # Only rows 2 and 6 must remain (2 events)
      self.assertEqual(len(parser.EVENTS), 2)

      # Check event indexes (normalized with year stride)
      indices = list(parser.EVENTS[OGS_C.IDX_EVENTS_STR])
      self.assertEqual(indices, [2005000002, 2005000006])

      # Check coordinates and magnitudes
      ev1 = parser.EVENTS.iloc[0]
      self.assertEqual(ev1[OGS_C.LOC_NAME_STR], "MOGGIO UDINESE (FRIULI)")
      self.assertEqual(ev1[OGS_C.EVENT_TYPE_STR], "[earthquake]")
      self.assertAlmostEqual(ev1[OGS_C.LATITUDE_STR], 46.35)
      self.assertAlmostEqual(ev1[OGS_C.LONGITUDE_STR], 13.0957)
      self.assertEqual(ev1[OGS_C.DEPTH_STR], 7.0)
      self.assertEqual(ev1[OGS_C.MAGNITUDE_D_STR], 2.4)
      self.assertTrue(pd.isna(ev1[OGS_C.MAGNITUDE_L_STR]))

    finally:
      if tmp_path.exists():
        tmp_path.unlink()


if __name__ == "__main__":
  unittest.main()

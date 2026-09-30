"""
===============================================================================
OGS DAT Test Suite - Unit and Integration Tests for Legacy Phase Pick DAT Parser
===============================================================================

OVERVIEW:
Comprehensive test suite for ``DataFileDAT`` covering:
  1. Input validation (file existence and .dat extension checks).
  2. CLI argument parsing via ``OGS_U.parse_dat_args``.
  3. Real dataset integration with ``OnlyEqHypo71`` (2005, 2024) and ``RSFVG`` (1977, 2004).
  4. Schema invariants (standardized 11-column PICKS schema matching ``_PICK_COLUMNS``).
  5. Pick time offset calculation (SSCC centisecond offset resolution).
  6. Multi-phase pick extraction (P+S paired picks, P-only picks, and S-picks).
  7. Century and format marker compatibility (modern '1' vs legacy 1970s space ' ').
  8. Weight parsing and fallback handling (0-4 integer quality weights).
  9. Temporal filtering across calendar windows.
  10. Synthetic fixtures for minute rollover, station formatting, and event type filtering.

===============================================================================
"""

from datetime import datetime, timedelta as td
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import ogsconstants as OGS_C
from ogsdatafile import OGSDataFile
from ogsdat import DataFileDAT
import ogsutils as OGS_U

# Base paths
TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"
HYPO71_DIR = DATASET_DIR / "OnlyEqHypo71"
RSFVG_DIR = DATASET_DIR / "RSFVG"


class TestDataFileDAT(unittest.TestCase):
  """Unit and integration test cases for DataFileDAT."""

  def setUp(self):
    """Set up test fixtures and paths."""
    self.dat_hypo_2005 = HYPO71_DIR / "onlyeq2005.dat"
    self.dat_hypo_2024 = HYPO71_DIR / "onlyeq2024.dat"
    self.dat_rsfvg_1977 = RSFVG_DIR / "RSFVG-1977.dat"
    self.dat_rsfvg_2004 = RSFVG_DIR / "RSFVG-2004.dat"

  def test_input_validation_missing_file(self):
    """DataFileDAT raises FileNotFoundError when the file does not exist."""
    non_existent = HYPO71_DIR / "non_existent_file.dat"
    with self.assertRaises(FileNotFoundError):
      DataFileDAT(non_existent)

  def test_input_validation_invalid_extension(self):
    """DataFileDAT raises ValueError when the file extension is not .dat."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
      tmp_path = Path(tmp.name)
    try:
      parser = DataFileDAT(tmp_path)
      with self.assertRaises(ValueError):
        parser.read()
    finally:
      if tmp_path.exists():
        tmp_path.unlink()

  def test_parse_dat_args(self):
    """Test CLI argument parsing for DAT parser via parse_dat_args."""
    if not self.dat_hypo_2005.is_file():
      self.skipTest(f"Dataset file {self.dat_hypo_2005} not found")

    args = OGS_U.parse_dat_args([
        "-f", str(self.dat_hypo_2005),
        "-D", "20050101", "20050110",
        "-v",
    ])
    self.assertEqual(args.file, [self.dat_hypo_2005])
    self.assertTrue(args.verbose)
    self.assertEqual(len(args.dates), 2)
    self.assertEqual(args.dates[0], datetime(2005, 1, 1))
    self.assertEqual(args.dates[1], datetime(2005, 1, 10))

  def test_parse_dat_args_defaults(self):
    """Test parse_dat_args default arguments."""
    if not self.dat_hypo_2005.is_file():
      self.skipTest(f"Dataset file {self.dat_hypo_2005} not found")

    args = OGS_U.parse_dat_args(["-f", str(self.dat_hypo_2005)])
    self.assertEqual(args.file, [self.dat_hypo_2005])
    self.assertFalse(args.verbose)
    self.assertEqual(len(args.dates), 2)
    self.assertEqual(args.dates[0], datetime.min)

  def test_pick_time_offset_calculation(self):
    """Test SSCC centisecond string conversion into absolute timestamp."""
    base_time = datetime(2005, 1, 1, 14, 22)
    # "1018" -> 10.18 seconds offset
    t1 = DataFileDAT._parse_pick_time(base_time, "1018")
    self.assertEqual(t1, base_time + td(seconds=10.18))

    # " 471" -> 4.71 seconds offset (spaces replaced by 0)
    t2 = DataFileDAT._parse_pick_time(base_time, " 471")
    self.assertEqual(t2, base_time + td(seconds=4.71))

    # "  50" -> 0.50 seconds offset
    t3 = DataFileDAT._parse_pick_time(base_time, "  50")
    self.assertEqual(t3, base_time + td(seconds=0.50))

  def test_hypo71_dataset_2005_schema_and_picks(self):
    """Integration test with OnlyEqHypo71/onlyeq2005.dat: schema invariants and P/S phases."""
    if not self.dat_hypo_2005.is_file():
      self.skipTest(f"Dataset file {self.dat_hypo_2005} not found")

    start = datetime(2005, 1, 1)
    end = datetime(2005, 1, 5)
    parser = DataFileDAT(self.dat_hypo_2005, start=start, end=end)
    parser.read()

    # Schema invariants: 11 columns
    self.assertEqual(len(parser.PICKS.columns), 11)
    self.assertEqual(list(parser.PICKS.columns), OGSDataFile._PICK_COLUMNS)

    # Both P and S arrivals must be present
    phases = set(parser.PICKS[OGS_C.PHASE_STR].unique())
    self.assertEqual(phases, {OGS_C.PWAVE, OGS_C.SWAVE})

    # Station codes must follow '.STATION.' format
    for station in parser.PICKS[OGS_C.STATION_STR]:
      self.assertTrue(station.startswith("."))
      self.assertTrue(station.endswith("."))

    # Groups column must contain valid ISO date strings
    for grp in parser.PICKS[OGS_C.GROUPS_STR]:
      datetime.strptime(grp, OGS_C.DATE_FMT)

  def test_rsfvg_1977_dataset_legacy_marker_and_picks(self):
    """Integration test with RSFVG/RSFVG-1977.dat: legacy space marker and picks extraction."""
    if not self.dat_rsfvg_1977.is_file():
      self.skipTest(f"Dataset file {self.dat_rsfvg_1977} not found")

    start = datetime(1977, 5, 1)
    end = datetime(1977, 5, 10)
    parser = DataFileDAT(self.dat_rsfvg_1977, start=start, end=end)
    parser.read()

    # Schema invariants
    self.assertEqual(len(parser.PICKS.columns), 11)
    self.assertEqual(list(parser.PICKS.columns), OGSDataFile._PICK_COLUMNS)
    self.assertGreater(len(parser.PICKS), 0)

    # Legacy 1977 RSFVG data contains both P and S picks
    counts = parser.PICKS[OGS_C.PHASE_STR].value_counts()
    self.assertIn(OGS_C.PWAVE, counts)
    self.assertIn(OGS_C.SWAVE, counts)

  def test_rsfvg_2004_dataset(self):
    """Integration test with RSFVG/RSFVG-2004.dat."""
    if not self.dat_rsfvg_2004.is_file():
      self.skipTest(f"Dataset file {self.dat_rsfvg_2004} not found")

    start = datetime(2004, 1, 1)
    end = datetime(2004, 1, 3)
    parser = DataFileDAT(self.dat_rsfvg_2004, start=start, end=end)
    parser.read()

    self.assertEqual(len(parser.PICKS.columns), 11)
    self.assertGreater(len(parser.PICKS), 0)

  def test_date_range_filtering(self):
    """Test start and end date filtering on DAT pick records."""
    if not self.dat_hypo_2005.is_file():
      self.skipTest(f"Dataset file {self.dat_hypo_2005} not found")

    parser_day1 = DataFileDAT(
        self.dat_hypo_2005,
        start=datetime(2005, 1, 1),
        end=datetime(2005, 1, 1),
    )
    parser_day1.read()
    picks_day1_len = len(parser_day1.PICKS)
    self.assertGreater(picks_day1_len, 0)

    parser_day3 = DataFileDAT(
        self.dat_hypo_2005,
        start=datetime(2005, 1, 1),
        end=datetime(2005, 1, 3),
    )
    parser_day3.read()
    self.assertGreater(len(parser_day3.PICKS), picks_day1_len)

  def test_synthetic_dat_modern_and_legacy_markers(self):
    """Test synthetic DAT records testing '1' vs space century markers and P+S vs P-only picks."""
    synthetic_dat = (
        # Line 1: Modern 2005 style with '1' marker and both P and S picks
        "BOO iPC010501011422 1018        1095iS 2                      FLD       480   3 gg                                      \n"
        # Line 2: Modern 2005 style with '1' marker and P-only pick (empty S block)
        "ROBSiP 210501011422 1459                                      FLD             3 g                                       \n"
        # Line 3: Event separator marker line (must be skipped)
        "                 1                                              D                                                       \n"
        # Line 4: Legacy 1977 style with space ' ' marker, empty weights (defaults to 0), and P+S
        "BUA eP   7705061140 5400        5650eS                        FL         70   1 gg                                      \n"
        # Line 5: Filtered event type (quarry explosion 'Q' with local localization should be skipped by default)
        "BAD eP 010501011422 1210        1478iS 2                      FQ        586   4 gg                                      \n"
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as tmp:
      tmp.write(synthetic_dat)
      tmp_path = Path(tmp.name)

    try:
      # Use broad date range to capture both 1977 and 2005 records
      parser = DataFileDAT(tmp_path, start=datetime(
          1970, 1, 1), end=datetime(2010, 1, 1))
      parser.read()

      # Line 1 produces 2 picks (BOO P + S)
      # Line 2 produces 1 pick (ROBS P only)
      # Line 3 is event summary line (skipped)
      # Line 4 produces 2 picks (BUA P + S with legacy marker)
      # Line 5 is type Q (skipped)
      # Total expected picks: 2 + 1 + 2 = 5 picks
      self.assertEqual(len(parser.PICKS), 5)

      # Check phases
      phases = list(parser.PICKS[OGS_C.PHASE_STR])
      self.assertEqual(
          phases, [OGS_C.PWAVE, OGS_C.SWAVE, OGS_C.PWAVE, OGS_C.PWAVE, OGS_C.SWAVE])

      # Check station names normalized with surrounding dots
      stations = list(parser.PICKS[OGS_C.STATION_STR])
      self.assertEqual(stations, [".BOO.", ".BOO.",
                       ".ROBS.", ".BUA.", ".BUA."])

      # Check weights: line 4 had spaces -> parsed with default weight 0
      weights = list(parser.PICKS[OGS_C.WEIGHT_STR])
      self.assertEqual(weights, [0, 2, 2, 0, 0])

      # Check event indexes normalized with year stride
      idxs = list(parser.PICKS[OGS_C.IDX_PICKS_STR])
      # 2005 event 3 -> 2005000003
      self.assertEqual(idxs[:3], [2005000003, 2005000003, 2005000003])
      # 1977 event 1 -> 1977000001
      self.assertEqual(idxs[3:], [1977000001, 1977000001])

    finally:
      if tmp_path.exists():
        tmp_path.unlink()

  def test_minute_rollover_datetime_parsing(self):
    """Test DAT date parsing when minute field is >= 60 (rollover handling)."""
    # Test valid date without rollover
    dt1 = DataFileDAT._parse_event_datetime("0501011422")
    self.assertEqual(dt1, datetime(2005, 1, 1, 14, 22))

    # Test rollover when minute is 60: "0501011460" -> 14:00 + 1 hour = 15:00
    dt2 = DataFileDAT._parse_event_datetime("0501011460")
    self.assertEqual(dt2, datetime(2005, 1, 1, 15, 0))


if __name__ == "__main__":
  unittest.main()

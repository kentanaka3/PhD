"""
=============================================================================
OGS DataFile Test Suite - Unit Tests for Base Class Optimizations
=============================================================================

OVERVIEW:
Unit test suite for ``ogsdatafile.py`` optimizations:
  1. Coordinate vectorization (_vectorized_to_decimal)
  2. Coordinate normalization & regex sweep elimination (normalize_coordinates)
  3. Vectorized GROUPS_STR derivation in _build_picks_dataframe & deferred pick row
  4. Vectorized ID stride normalization & magnitude coercion in _build_events_dataframe
  5. Fast-path datetime parsing in HPL, PUN, and DAT parsers
=============================================================================
"""

from pathlib import Path
import unittest
from datetime import datetime, timedelta as td
import numpy as np
import pandas as pd
import ogsconstants as OGS_C
from ogsdatafile import OGSDataFile
from ogshpl import DataFileHPL
from ogspun import DataFilePUN
from ogsdat import DataFileDAT


class TestOGSDataFile(unittest.TestCase):
  """Test suite for OGSDataFile optimizations."""

  def test_vectorized_to_decimal_various_formats(self):
    """Test _vectorized_to_decimal with degrees-minutes, decimals, negatives, and invalid values."""
    s = pd.Series([
        "46-07.38",     # 46 + 7.38/60 = 46.123
        "-46-07.38",    # -46.123
        "13.5",         # 13.5
        "-13.5",        # -13.5
        14.25,          # numeric
        "",             # empty
        "None",         # string None
        np.nan,         # nan
        "nan",          # string nan
        "invalid",      # invalid format
    ])
    result = OGSDataFile._vectorized_to_decimal(s)
    self.assertAlmostEqual(result.iloc[0], 46.123, places=3)
    self.assertAlmostEqual(result.iloc[1], -46.123, places=3)
    self.assertAlmostEqual(result.iloc[2], 13.5, places=1)
    self.assertAlmostEqual(result.iloc[3], -13.5, places=1)
    self.assertAlmostEqual(result.iloc[4], 14.25, places=2)
    self.assertTrue(pd.isna(result.iloc[5]))
    self.assertTrue(pd.isna(result.iloc[6]))
    self.assertTrue(pd.isna(result.iloc[7]))
    self.assertTrue(pd.isna(result.iloc[8]))
    self.assertTrue(pd.isna(result.iloc[9]))

  def test_normalize_coordinates(self):
    """Test normalize_coordinates vectorization, bounds checking, and error metric coercion."""
    df = pd.DataFrame({
        OGS_C.LATITUDE_STR: ["46-00.00", "95.0", "-95.0", "None"],
        OGS_C.LONGITUDE_STR: ["13-00.00", "190.0", "-190.0", ""],
        OGS_C.DEPTH_STR: ["10.5", "---", "***", "  "],
        OGS_C.ERH_STR: ["1.2", "---", "***", " "],
        OGS_C.ERZ_STR: ["2.3", " - ", "*", ""],
        OGS_C.RMS_STR: ["0.45", "None", "nan", "0.5"],
    })
    norm = OGSDataFile.normalize_coordinates(df.copy())
    # Latitude
    self.assertAlmostEqual(norm[OGS_C.LATITUDE_STR].iloc[0], 46.0)
    # 95 out of bounds
    self.assertTrue(pd.isna(norm[OGS_C.LATITUDE_STR].iloc[1]))
    # -95 out of bounds
    self.assertTrue(pd.isna(norm[OGS_C.LATITUDE_STR].iloc[2]))
    self.assertTrue(pd.isna(norm[OGS_C.LATITUDE_STR].iloc[3]))

    # Longitude
    self.assertAlmostEqual(norm[OGS_C.LONGITUDE_STR].iloc[0], 13.0)
    # 190 out of bounds
    self.assertTrue(pd.isna(norm[OGS_C.LONGITUDE_STR].iloc[1]))
    # -190 out of bounds
    self.assertTrue(pd.isna(norm[OGS_C.LONGITUDE_STR].iloc[2]))

    # Depth & Error metrics coerced
    self.assertEqual(norm[OGS_C.DEPTH_STR].iloc[0], 10.5)
    self.assertTrue(pd.isna(norm[OGS_C.DEPTH_STR].iloc[1]))
    self.assertTrue(pd.isna(norm[OGS_C.DEPTH_STR].iloc[2]))
    self.assertTrue(pd.isna(norm[OGS_C.DEPTH_STR].iloc[3]))

    self.assertEqual(norm[OGS_C.ERH_STR].iloc[0], 1.2)
    self.assertTrue(pd.isna(norm[OGS_C.ERH_STR].iloc[1]))
    self.assertEqual(norm[OGS_C.ERZ_STR].iloc[0], 2.3)
    self.assertTrue(pd.isna(norm[OGS_C.ERZ_STR].iloc[1]))
    self.assertEqual(norm[OGS_C.RMS_STR].iloc[0], 0.45)
    self.assertEqual(norm[OGS_C.RMS_STR].iloc[3], 0.5)

  def test_build_picks_dataframe_groups_str(self):
    """Test that _build_picks_dataframe derives vectorized GROUPS_STR from TIME_STR."""
    test_file = Path(__file__).resolve()
    parser = DataFileHPL(test_file)
    picks_data = [
        OGSDataFile._build_pick_row(
            1, datetime(2024, 3, 20, 12, 0, 0), "BOO", OGS_C.PWAVE, 0
        ),
        OGSDataFile._build_pick_row(
            2, datetime(2024, 3, 21, 14, 30, 0), "TRI", OGS_C.SWAVE, 1
        ),
    ]
    df = parser._build_picks_dataframe(picks_data)
    self.assertEqual(df[OGS_C.GROUPS_STR].iloc[0], "2024-03-20")
    self.assertEqual(df[OGS_C.GROUPS_STR].iloc[1], "2024-03-21")
    self.assertEqual(df[OGS_C.STATION_STR].iloc[0], ".BOO.")
    self.assertEqual(df[OGS_C.STATION_STR].iloc[1], ".TRI.")

  def test_build_events_dataframe_stride_and_magnitudes(self):
    """Test _build_events_dataframe vectorized ID stride check and magnitude coercion."""
    test_file = Path(__file__).resolve()
    parser = DataFilePUN(test_file)
    events_df = pd.DataFrame([
        # Already normalized ID (2024 * 1_000_000 + 42)
        {
            OGS_C.IDX_EVENTS_STR: 2024000042,
            OGS_C.TIME_STR: datetime(2024, 5, 10, 8, 30, 0),
            OGS_C.LATITUDE_STR: 46.1,
            OGS_C.LONGITUDE_STR: 13.2,
            OGS_C.DEPTH_STR: 10.0,
            OGS_C.MAGNITUDE_D_STR: 2.1,
        },
        # Un-normalized raw ID (43)
        {
            OGS_C.IDX_EVENTS_STR: 43,
            OGS_C.TIME_STR: datetime(2024, 5, 11, 9, 0, 0),
            OGS_C.LATITUDE_STR: 46.2,
            OGS_C.LONGITUDE_STR: 13.3,
            OGS_C.DEPTH_STR: 12.0,
            OGS_C.MAGNITUDE_D_STR: "---",
        },
    ])
    df = parser._build_events_dataframe(events_df, columns=list(events_df.columns))
    self.assertEqual(df[OGS_C.IDX_EVENTS_STR].iloc[0], 2024000042)
    self.assertEqual(df[OGS_C.IDX_EVENTS_STR].iloc[1], 2024000043)
    self.assertEqual(df[OGS_C.GROUPS_STR].iloc[0], "2024-05-10")
    self.assertEqual(df[OGS_C.GROUPS_STR].iloc[1], "2024-05-11")
    self.assertEqual(df[OGS_C.MAGNITUDE_D_STR].iloc[0], 2.1)
    self.assertTrue(pd.isna(df[OGS_C.MAGNITUDE_D_STR].iloc[1]))

  def test_fast_path_datetime_equivalence(self):
    """Test fast-path datetime construction across HPL, PUN, and DAT."""
    test_file = Path(__file__).resolve()
    # HPL: date string is 'YYMMDD HHMM'
    hpl_parser = DataFileHPL(test_file)
    res_hpl = {
        OGS_C.DATE_STR: "240320 1234",
        OGS_C.SECONDS_STR: "56.78",
    }
    dt_hpl = hpl_parser._parse_event_datetime(res_hpl)
    expected_hpl = datetime(2024, 3, 20, 12, 34) + td(seconds=56.78)
    self.assertEqual(dt_hpl, expected_hpl)

    # PUN: date string is 'YYMMDDHHMM'
    pun_dt = DataFilePUN._parse_origin_time("2403201234", td(seconds=56.78))
    self.assertEqual(pun_dt, expected_hpl)

    # 1900s year century check (e.g. 98 -> 1998)
    pun_90s = DataFilePUN._parse_origin_time("9806151015", td(seconds=12.34))
    self.assertEqual(pun_90s, datetime(
        1998, 6, 15, 10, 15) + td(seconds=12.34))

    # DAT: date string is 'YYMMDDHHMM', rollover minute >= 60
    dat_dt = DataFileDAT._parse_event_datetime("2403201234")
    self.assertEqual(dat_dt, datetime(2024, 3, 20, 12, 34))

    dat_rollover = DataFileDAT._parse_event_datetime("2403201260")
    self.assertEqual(dat_rollover, datetime(2024, 3, 20, 12) + td(hours=1))

  def test_parse_event_datetime_all_formats(self):
    """Test all input forms of centralized _parse_event_datetime."""
    # 1. Dict with OGS_C.TIME_STR (ISO string and datetime)
    dt_target = datetime(2024, 3, 20, 12, 34, 56)
    res_iso = {OGS_C.TIME_STR: "2024-03-20T12:34:56"}
    self.assertEqual(OGSDataFile._parse_event_datetime(res_iso), dt_target)

    res_dt = {OGS_C.TIME_STR: dt_target}
    self.assertEqual(OGSDataFile._parse_event_datetime(res_dt), dt_target)

    # String ISO format
    self.assertEqual(
        OGSDataFile._parse_event_datetime("2024-03-20T12:34:56"), dt_target
    )

    # 2. Dict with OGS_C.DATE_STR + OGS_C.SECONDS_STR (HPL format 'YYMMDD HHMM')
    expected_hpl = datetime(2024, 3, 20, 12, 34) + td(seconds=56.78)
    res_hpl = {
        OGS_C.DATE_STR: "240320 1234",
        OGS_C.SECONDS_STR: "56.78",
    }
    self.assertEqual(OGSDataFile._parse_event_datetime(res_hpl), expected_hpl)

    # 3. Dict with OGS_C.DATE_STR + OGS_C.SECONDS_STR (PUN format 'YYMMDDHHMM')
    res_pun = {
        OGS_C.DATE_STR: "2403201234",
        OGS_C.SECONDS_STR: "56.78",
    }
    self.assertEqual(OGSDataFile._parse_event_datetime(res_pun), expected_hpl)

    # String PUN format with td seconds, float seconds, str seconds
    self.assertEqual(
        OGSDataFile._parse_event_datetime("2403201234", td(seconds=56.78)),
        expected_hpl,
    )
    self.assertEqual(
        OGSDataFile._parse_event_datetime("2403201234", 56.78),
        expected_hpl,
    )
    self.assertEqual(
        OGSDataFile._parse_event_datetime("2403201234", "56.78"),
        expected_hpl,
    )

    # 4. String with rollover minute (DAT format 'YYMMDDHHMM')
    dat_rollover = OGSDataFile._parse_event_datetime("2403201260")
    self.assertEqual(dat_rollover, datetime(2024, 3, 20, 12) + td(hours=1))

    # 5. Backwards compatibility of DataFilePUN._parse_origin_time
    test_file = Path(__file__).resolve()
    pun_parser = DataFilePUN(test_file)
    self.assertEqual(
        DataFilePUN._parse_origin_time("2403201234", td(seconds=56.78)),
        expected_hpl,
    )
    self.assertEqual(
        pun_parser._parse_origin_time("2403201234", td(seconds=56.78)),
        expected_hpl,
    )


if __name__ == "__main__":
  unittest.main()

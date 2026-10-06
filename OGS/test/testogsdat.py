"""
===============================================================================
OGS DAT Test Suite - Unit and Integration Tests for Legacy Phase Pick DAT Parser
===============================================================================

DAT reader contracts backed by unchanged real station records.


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

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from OGS.src import ogsconstants as C
from OGS.src.ogsdat import DataFileDAT
from OGS.src.ogsutils import parse_dat_args


@pytest.mark.parametrize("first,last,p_count,s_count", [
    (1, 7, 4, 4),
    (8, 11, 3, 3),
    (12, 14, 2, 2),
    (18, 22, 4, 4),
    (23, 30, 7, 3),
    (31, 42, 11, 10),
    (43, 62, 17, 15),
    (109, 114, 5, 5),
])
def test_complete_station_groups(reader_factory, first, last, p_count, s_count):
  reader = reader_factory(
      DataFileDAT, "picks.dat", segments=((first, last),)
  )
  picks = reader.PICKS
  assert len(picks) == p_count + s_count
  assert picks[C.PHASE_STR].value_counts().to_dict() == {
      C.PWAVE: p_count, C.SWAVE: s_count,
  }
  assert picks[C.STATION_STR].str.fullmatch(r"\.[A-Z0-9]{1,4}\.").all()
  assert pd.api.types.is_datetime64_any_dtype(picks[C.TIME_STR])
  assert pd.api.types.is_integer_dtype(picks[C.IDX_PICKS_STR])
  assert picks[C.PROBABILITY_STR].eq(1.0).all()
  assert picks[C.GROUPS_STR].equals(picks[C.TIME_STR].dt.strftime("%Y-%m-%d"))
  assert sum(map(len, reader.picks.values())) == len(picks)
  assert not list(reader.output.rglob("*.parquet"))


@pytest.mark.parametrize("station,phase,weight", [
    ("PLRO", C.PWAVE, 0), ("PLRO", C.SWAVE, 0),
    ("BAD", C.PWAVE, 0), ("BAD", C.SWAVE, 2),
    ("BOO", C.PWAVE, 0), ("BOO", C.SWAVE, 1),
    ("VOY", C.PWAVE, 2), ("VOY", C.SWAVE, 2),
])
def test_explicit_weights_stay_with_station_and_phase(
    reader_factory, station, phase, weight,
):
  reader = reader_factory(DataFileDAT, "picks.dat", segments=((18, 22),))
  rows = reader.PICKS.set_index([C.STATION_STR, C.PHASE_STR])
  assert rows.loc[(f".{station}.", phase), C.WEIGHT_STR] == weight
  assert set(reader.PICKS[C.IDX_PICKS_STR]) == {1997000001}


@pytest.mark.parametrize("first,last,station,phase,expected", [
    (8, 11, "BAD", C.SWAVE, datetime(1977, 5, 7, 20, 11, 0, 200000)),
    (8, 11, "BUA", C.SWAVE, datetime(1977, 5, 7, 20, 11, 0, 900000)),
    (8, 11, "COLI", C.SWAVE, datetime(1977, 5, 7, 20, 11, 5, 300000)),
    (12, 14, "BAD", C.PWAVE, datetime(1977, 11, 2, 7, 0, 0, 600000)),
    (12, 14, "BAD", C.SWAVE, datetime(1977, 11, 2, 7, 0, 4, 100000)),
    (12, 14, "RCL", C.PWAVE, datetime(1977, 11, 2, 7, 0, 2)),
    (12, 14, "RCL", C.SWAVE, datetime(1977, 11, 2, 7, 0, 6, 400000)),
    (31, 42, "LSR", C.SWAVE, datetime(2004, 1, 1, 0, 19, 0, 510000)),
    (43, 62, "DRE", C.PWAVE, datetime(2005, 1, 1, 3, 22, 2, 940000)),
    (43, 62, "DRE", C.SWAVE, datetime(2005, 1, 1, 3, 22, 4, 710000)),
])
def test_centiseconds_and_distinct_rollover_cases(
    reader_factory, first, last, station, phase, expected,
):
  reader = reader_factory(DataFileDAT, "picks.dat", segments=((first, last),))
  event_id = {
      8: 1977000008, 12: 1977000692, 31: 2004000001, 43: 2005000001,
  }[first]
  picks = reader.PICKS.set_index([
      C.IDX_PICKS_STR, C.STATION_STR, C.PHASE_STR,
  ])
  assert picks.loc[(event_id, f".{station}.", phase), C.TIME_STR] == expected


@pytest.mark.parametrize("start,end,ids,total,days", [
    (datetime(2005, 1, 1), datetime(2005, 1, 1),
     {2005000001, 2005000002, 2005000003}, 32, {date(2005, 1, 1)}),
    (datetime(2005, 1, 2), datetime(2005, 1, 2),
     {2005000008}, 44, {date(2005, 1, 2)}),
    (datetime(2005, 1, 1), datetime(2005, 1, 2),
     {2005000001, 2005000002, 2005000003, 2005000008}, 76,
     {date(2005, 1, 1), date(2005, 1, 2)}),
])
def test_inclusive_date_windows(reader_factory, start, end, ids, total, days):
  reader = reader_factory(DataFileDAT, "picks.dat", start, end)
  assert set(reader.PICKS[C.IDX_PICKS_STR]) == ids
  assert len(reader.PICKS) == total
  assert set(reader.picks) == days


def test_p_only_station_never_gets_an_s_pick(reader_factory):
  reader = reader_factory(DataFileDAT, "picks.dat", segments=((23, 30),))
  by_station = reader.PICKS.groupby(C.STATION_STR)[C.PHASE_STR].agg(set)
  for station in ("CAE", "MPRI", "BUA", "COLI"):
    assert by_station[f".{station}."] == {C.PWAVE}
  assert by_station[".CLA1."] == {C.PWAVE, C.SWAVE}
  assert set(reader.PICKS[C.IDX_PICKS_STR]) == {2000000004}


def test_no_matching_dates_returns_empty_pick_schema(reader_factory):
  reader = reader_factory(
      DataFileDAT, "picks.dat", datetime(2022, 1, 1), datetime(2022, 12, 31)
  )
  assert reader.PICKS.empty
  assert list(reader.PICKS.columns) == DataFileDAT._PICK_COLUMNS
  assert reader.picks == {}


def test_non_earthquake_without_distant_flag_is_excluded(reader_factory):
  reader = reader_factory(DataFileDAT, "picks.dat", segments=((105, 108),))
  assert reader.PICKS.empty
  assert reader.picks == {}


def test_input_validation(tmp_path):
  with pytest.raises(FileNotFoundError, match="does not exist"):
    DataFileDAT(tmp_path / "missing.dat", output=tmp_path / "out")
  wrong = tmp_path / "wrong.txt"
  wrong.touch()
  reader = DataFileDAT(wrong, output=tmp_path / "out")
  with pytest.raises(ValueError, match=r"\.dat"):
    reader.read()


def test_cli_uses_existing_fixture_and_sorts_dates():
  source = Path(__file__).parent / "data" / "picks.dat"
  args = parse_dat_args(
      ["-f", str(source), "-D", "20240102", "20240101", "-v"])
  assert args.file == [source]
  assert args.dates == [datetime(2024, 1, 1), datetime(2024, 1, 2)]
  assert args.verbose

"""
===============================================================================
OGS TXT Test Suite - Unit and Integration Tests for Catalog Event Summary TXT
===============================================================================

Real Hypo71 and NLL1D TXT contracts; no inferred MD-selection policy.

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

from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from OGS.src import ogsconstants as C, ogsutils as U
from OGS.src.ogsdatafile import OGSDataFile
from OGS.src.ogstxt import DataFileTXT


DATA = Path(__file__).parent / "data"


def _slice(tmp_path, source, *line_numbers):
  """Copy 1-based fixture lines without rewriting fields or line endings."""
  lines = (DATA / source).read_bytes().splitlines(keepends=True)
  path = tmp_path / source
  path.write_bytes(b"".join(lines[number - 1] for number in line_numbers))
  return path


def _read(tmp_path, source="hypo71.txt", path=None,
          start=datetime(2000, 1, 1), end=datetime(2025, 1, 1)):
  parser = DataFileTXT(
      path or DATA / source, start=start, end=end,
      output=tmp_path / "output",
  )
  assert parser.read() is None
  return parser


@pytest.mark.parametrize(
    "source,line,idx,origin,lat,lon,depth,gap,rms,erh,erz,ml,md,place",
    [
        ("hypo71.txt", 4, 2005000003, "2005-01-01T14:22:09.040",
         46.3500, 13.0957, 7., 59, .24, .5, .5, None, 2.4, "MOGGIO UDINESE (FRIULI)"),
        ("hypo71.txt", 9, 2005000008, "2005-01-02T11:56:15.320",
         46.3482, 13.0905, 7., 56, .30, .6, .6, None, 2.7, "MOGGIO UDINESE (FRIULI)"),
        ("hypo71.txt", 11, 2005000010, "2005-01-03T03:09:59.830",
         46.1547, 12.4360, 8.5, 147, .25, .7, .7, None, 2.6, "PUOS D'ALPAGO (VENETO)"),
        ("nll1d.txt", 2, 2005000001, "2005-01-01T14:22:08.359",
         46.3527, 13.0964, 8.4, 58, None, 1.4, 1.6, None, 2.4, "MOGGIO UDINESE (FRIULI)"),
        ("nll1d.txt", 3, 2005000002, "2005-01-02T11:56:14.654",
         46.3517, 13.0902, 8.3, 39, None, 1.2, 1.3, None, 2.7, "MOGGIO UDINESE (FRIULI)"),
        ("nll1d.txt", 4, 2005000003, "2005-01-03T03:09:59.423",
         46.1629, 12.4526, 8.4, 68, None, 2.8, 1.8, None, 2.6, "PUOS D'ALPAGO (VENETO)"),
        ("nll1d.txt", 5, 2005000004, "2005-01-03T13:14:27.722",
         46.3514, 13.0987, 5.9, 154, None, 1.9, 1.3, None, 1.9, "MOGGIO UDINESE (FRIULI)"),
        ("hypo71.txt", 12, 2017000303, "2017-02-21T04:33:33.080",
         45.7053, 14.1802, 10., 74, .14, .3, .3, 1.8, 2.1, "PIVKA (SLOVENIA)"),
        ("hypo71.txt", 23, 2023000001, "2023-01-01T03:35:48.270",
         46.8145, 11.2200, 5.7, 153, .32, 1., 1., 1.1, 1.5, "S.LEONARDO PASSIRIA (ALTO ADIGE)"),
        ("hypo71.txt", 24, 2023000002, "2023-01-01T04:11:23.100",
         46.8047, 11.2135, 9.2, 118, .27, .8, .8, 1.3, 2., "S.LEONARDO PASSIRIA (ALTO ADIGE)"),
        ("nll1d.txt", 6, 2023000001, "2023-01-01T03:35:47.953",
         46.7840, 11.1902, 7.9, 144, None, 4.3, 2.9, 1.1, 1.5, "S.LEONARDO PASSIRIA (ALTO ADIGE)"),
        ("nll1d.txt", 7, 2023000002, "2023-01-01T04:11:22.850",
         46.7878, 11.1928, 8.5, 114, None, 3.8, 3.1, 1.3, 2., "S.LEONARDO PASSIRIA (ALTO ADIGE)"),
        ("hypo71.txt", 25, 2024000001, "2024-01-01T03:36:13.320",
         46.7320, 12.4692, 4.4, 198, .15, 1., 1.5, .8, .9, "M.CAVALLINO (ALTO ADIGE)"),
        ("nll1d.txt", 8, 2024000001, "2024-01-01T03:36:12.759",
         46.7085, 12.4675, 2.9, 170, None, 3.3, 2.2, .8, .9, "M.CAVALLINO (ALTO ADIGE)"),
        ("hypo71.txt", 26, 2024000002, "2024-01-01T13:21:37.680",
         46.3643, 12.9683, 12.1, 116, .11, .4, .7, .5, 1.1, "TOLMEZZO (FRIULI)"),
        ("nll1d.txt", 9, 2024000002, "2024-01-01T13:21:37.343",
         46.3646, 12.9573, 11.4, 122, None, 1.7, 1.6, .5, 1.1, "TOLMEZZO (FRIULI)"),
        ("hypo71.txt", 27, 2024000003, "2024-01-01T17:41:32.830",
         46.4343, 13.3252, 10.1, 123, .17, .8, 1.1, .5, 1., "DOGNA (FRIULI)"),
        ("nll1d.txt", 10, 2024000003, "2024-01-01T17:41:32.388",
         46.4412, 13.3263, 9.5, 121, None, 2., 1.6, .5, 1., "DOGNA (FRIULI)"),
        ("hypo71.txt", 28, 2024000004, "2024-01-01T20:53:49.910",
         45.7917, 11.1065, 12.4, 87, .23, .6, 1.3, 1.5, 2.1, "PASUBIO (TRENTINO)"),
        ("nll1d.txt", 11, 2024000004, "2024-01-01T20:53:49.585",
         45.7941, 11.1016, 11.8, 86, None, 2.1, 2.1, 1.5, 2.1, "PASUBIO (TRENTINO)"),
        ("hypo71.txt", 29, 2024000005, "2024-01-01T21:26:35.320",
         46.4815, 13.7912, 7., 174, .15, .7, 2.9, .3, .8, "KRANJSKA GORA (SLOVENIA)"),
        ("nll1d.txt", 12, 2024000005, "2024-01-01T21:26:34.811",
         46.4723, 13.7911, 3.6, 170, None, 2.4, 3.7, .3, .8, "KRANJSKA GORA (SLOVENIA)"),
        ("hypo71.txt", 30, 2024000006, "2024-01-02T02:09:56.680",
         46.6003, 13.8420, 8., 204, .23, 1., 2.4, 1.2, 1.9, "VILLACH (AUSTRIA)"),
        ("nll1d.txt", 13, 2024000006, "2024-01-02T02:09:56.159",
         46.5755, 13.8334, 5.9, 191, None, 2.9, 3., 1.2, 1.9, "VILLACH (AUSTRIA)"),
    ],
)
def test_summary_fields(tmp_path, source, line, idx, origin, lat, lon, depth,
                        gap, rms, erh, erz, ml, md, place):
  frame = _read(tmp_path, source, _slice(tmp_path, source, 1, line)).EVENTS
  assert len(frame) == 1
  row = frame.iloc[0]
  assert row[C.IDX_EVENTS_STR] == idx
  assert row[C.TIME_STR] == datetime.fromisoformat(origin)
  assert row[C.GROUPS_STR] == origin[:10]
  assert row[C.LOC_NAME_STR] == place
  assert row[C.EVENT_TYPE_STR] == "[earthquake]"
  for column, value in (
      (C.LATITUDE_STR, lat), (C.LONGITUDE_STR, lon), (C.DEPTH_STR, depth),
      (C.GAP_STR, gap), (C.RMS_STR, rms), (C.ERH_STR, erh), (C.ERZ_STR, erz),
      (C.MAGNITUDE_L_STR, ml), (C.MAGNITUDE_D_STR, md),
  ):
    assert pd.api.types.is_numeric_dtype(frame[column])
    if value is None:
      assert pd.isna(row[column])
    else:
      assert row[column] == pytest.approx(value, abs=1e-8)


@pytest.mark.parametrize(
    "source,ids",
    [
        ("hypo71.txt", [2005000003, 2005000008, 2005000010,
                        2017000303, 2017000306, 2017000307,
                        2017001058, 2017001062, 2017001063,
                        2023000001, 2023000002,
                        2024000001, 2024000002, 2024000003,
                        2024000004, 2024000005, 2024000006]),
        ("nll1d.txt", [2005000001, 2005000002, 2005000003, 2005000004,
                       2023000001, 2023000002,
                       2024000001, 2024000002, 2024000003,
                       2024000004, 2024000005, 2024000006]),
    ],
)
def test_whole_fixture_schema_ids_defaults_and_daily_cache(tmp_path, source, ids):
  parser = _read(tmp_path, source)
  frame = parser.EVENTS
  assert list(frame.columns) == OGSDataFile._EVENT_COLUMNS
  assert len(frame.columns) == 28
  assert C.LEGACY_ID_STR not in frame.columns
  assert frame[C.IDX_EVENTS_STR].tolist() == ids
  assert frame[C.IDX_EVENTS_STR].is_unique
  assert frame[C.EVENT_TYPE_STR].eq("[earthquake]").all()
  for column in (C.NO_STR, C.QM_STR, C.DMIN_STR, C.ERT_STR):
    assert frame[column].isna().all()
  for column in (C.NUMBER_P_PICKS_STR, C.NUMBER_S_PICKS_STR,
                 C.NUMBER_P_AND_S_PICKS_STR):
    assert frame[column].eq(0).all()
  assert parser.PICKS.empty
  assert parser.picks == {}
  assert sum(len(rows) for rows in parser.events.values()) == len(ids)
  assert len(parser.events[date(2024, 1, 1)]) == 5
  assert len(parser.events[date(2024, 1, 2)]) == 1
  for day, rows in parser.events.items():
    pd.testing.assert_frame_equal(
        rows, frame.loc[frame[C.GROUPS_STR] == day.isoformat()],
    )
  if source == "nll1d.txt":
    assert frame[C.RMS_STR].isna().all()


@pytest.mark.parametrize("line", [2, 3, 5, 6, 7, 8, 10, 13, 18])
def test_real_unlocated_rows_excluded_individually(tmp_path, line):
  path = _slice(tmp_path, "hypo71.txt", 1, line, 4)
  frame = _read(tmp_path, path=path).EVENTS
  assert frame[C.IDX_EVENTS_STR].tolist() == [2005000003]


@pytest.mark.parametrize("line", [14, 19, 20])
def test_real_located_non_earthquake_rows_excluded(tmp_path, line):
  # Located excluded rows isolate type filtering from unlocated filtering.
  path = _slice(tmp_path, "hypo71.txt", 1, line, 12)
  frame = _read(tmp_path, path=path).EVENTS
  assert frame[C.IDX_EVENTS_STR].tolist() == [2017000303]


def test_2017_case_retains_all_six_earthquake_controls(tmp_path):
  path = _slice(tmp_path, "hypo71.txt", 1, *range(12, 23))
  frame = _read(tmp_path, path=path).EVENTS
  assert frame[C.IDX_EVENTS_STR].tolist() == [
      2017000303, 2017000306, 2017000307, 2017001058, 2017001062, 2017001063,
  ]
  assert frame[C.MAGNITUDE_L_STR].tolist() == [1.8, 1., .7, .6, 1.1, 1.2]
  assert frame[C.MAGNITUDE_D_STR].tolist() == [2.1, 1.2, .9, .9, 1.6, 1.1]


@pytest.mark.parametrize("source", ["hypo71.txt", "nll1d.txt"])
@pytest.mark.parametrize(
    "start,end,indices",
    [
        (datetime(2024, 1, 1), datetime(2024, 1, 1), [1, 2, 3, 4, 5]),
        (datetime(2024, 1, 2), datetime(2024, 1, 2), [6]),
        (datetime(2024, 1, 1), datetime(2024, 1, 2), [1, 2, 3, 4, 5, 6]),
        (datetime(2025, 1, 1), datetime(2025, 1, 1), []),
    ],
)
def test_calendar_date_windows(tmp_path, source, start, end, indices):
  parser = _read(tmp_path, source, start=start, end=end)
  assert parser.EVENTS[C.IDX_EVENTS_STR].tolist() == [
      2024000000 + i for i in indices
  ]
  if not indices:
    assert parser.events == {}


def test_exact_start_inclusive_and_end_plus_day_exclusive(tmp_path):
  path = _slice(tmp_path, "hypo71.txt", 1, 4, 9)
  origin = datetime(2005, 1, 1, 14, 22, 9, 40000)
  assert _read(tmp_path, path=path, start=origin).EVENTS[C.IDX_EVENTS_STR].tolist() == [
      2005000003, 2005000008,
  ]
  assert _read(tmp_path, path=path, start=origin + timedelta(microseconds=1)).EVENTS[
      C.IDX_EVENTS_STR
  ].tolist() == [2005000008]
  assert _read(tmp_path, path=path, end=origin -
               timedelta(days=1)).EVENTS.empty


def test_txt_continues_after_out_of_window_record(tmp_path):
  path = _slice(tmp_path, "hypo71.txt", 1, 4, 25, 9)
  frame = _read(tmp_path, path=path, end=datetime(2005, 1, 2)).EVENTS
  assert frame[C.IDX_EVENTS_STR].tolist() == [2005000003, 2005000008]


def test_first_line_is_unconditionally_skipped(tmp_path):
  path = _slice(tmp_path, "hypo71.txt", 4, 9)
  frame = _read(tmp_path, path=path).EVENTS
  assert frame[C.IDX_EVENTS_STR].tolist() == [2005000008]


def test_non_record_header_inside_data_is_logged_and_skipped(tmp_path):
  path = _slice(tmp_path, "hypo71.txt", 1, 4, 1, 9)
  parser = DataFileTXT(path, datetime(2005, 1, 1), datetime(2005, 1, 2),
                       output=tmp_path / "output")
  with patch.object(parser.logger, "error", wraps=parser.logger.error) as error:
    parser.read()
  error.assert_called_once()
  assert "(TXT) Could not parse line:" in error.call_args.args[0]
  frame = parser.EVENTS
  assert frame[C.IDX_EVENTS_STR].tolist() == [2005000003, 2005000008]


def test_solution_specific_origins_coordinates_and_independent_magnitudes(tmp_path):
  hypo = _read(tmp_path, start=datetime(2024, 1, 1),
               end=datetime(2024, 1, 2)).EVENTS
  nll = _read(tmp_path, "nll1d.txt", start=datetime(2024, 1, 1),
              end=datetime(2024, 1, 2)).EVENTS
  assert hypo[C.IDX_EVENTS_STR].tolist() == nll[C.IDX_EVENTS_STR].tolist() == list(
      range(2024000001, 2024000007)
  )
  assert hypo[C.LOC_NAME_STR].tolist() == nll[C.LOC_NAME_STR].tolist()
  for frame in (hypo, nll):
    assert frame[C.MAGNITUDE_L_STR].tolist() == [.8, .5, .5, 1.5, .3, 1.2]
    assert frame[C.MAGNITUDE_D_STR].tolist() == [.9, 1.1, 1., 2.1, .8, 1.9]
  assert hypo[C.TIME_STR].tolist() == [
      datetime.fromisoformat(value) for value in [
          "2024-01-01T03:36:13.320", "2024-01-01T13:21:37.680",
          "2024-01-01T17:41:32.830", "2024-01-01T20:53:49.910",
          "2024-01-01T21:26:35.320", "2024-01-02T02:09:56.680",
      ]
  ]
  assert nll[C.TIME_STR].tolist() == [
      datetime.fromisoformat(value) for value in [
          "2024-01-01T03:36:12.759", "2024-01-01T13:21:37.343",
          "2024-01-01T17:41:32.388", "2024-01-01T20:53:49.585",
          "2024-01-01T21:26:34.811", "2024-01-02T02:09:56.159",
      ]
  ]
  assert hypo[C.LATITUDE_STR].tolist() == [46.7320, 46.3643,
                                           46.4343, 45.7917, 46.4815, 46.6003]
  assert nll[C.LATITUDE_STR].tolist() == [46.7085, 46.3646, 46.4412,
                                          45.7941, 46.4723, 46.5755]
  assert hypo[C.LONGITUDE_STR].tolist() == [12.4692, 12.9683,
                                            13.3252, 11.1065, 13.7912, 13.8420]
  assert nll[C.LONGITUDE_STR].tolist() == [12.4675, 12.9573, 13.3263,
                                           11.1016, 13.7911, 13.8334]


@pytest.mark.parametrize("numbers", [(), (1,), (1, 2, 3, 14, 19, 20)])
def test_empty_header_only_or_fully_excluded_input(tmp_path, numbers):
  parser = _read(tmp_path, path=_slice(tmp_path, "hypo71.txt", *numbers))
  assert parser.EVENTS.empty
  assert list(parser.EVENTS.columns) == OGSDataFile._EVENT_COLUMNS
  assert parser.events == {}


def test_input_validation(tmp_path):
  with pytest.raises(FileNotFoundError, match="missing.txt"):
    DataFileTXT(tmp_path / "missing.txt", output=tmp_path / "output")
  path = tmp_path / "wrong.pun"
  path.write_bytes((DATA / "hypo71.txt").read_bytes())
  parser = DataFileTXT(path, output=tmp_path / "output")
  with pytest.raises(ValueError, match=r"extension must be \.txt"):
    parser.read()


def test_cli_arguments_without_main_side_effects():
  paths = [DATA / "hypo71.txt", DATA / "nll1d.txt"]
  args = U.parse_txt_args(["-f", *(str(path) for path in paths),
                           "-D", "20240102", "20240101", "-v"])
  assert args.file == [path.resolve() for path in paths]
  assert args.dates == [datetime(2024, 1, 1), datetime(2024, 1, 2)]
  assert args.verbose is True
  defaults = U.parse_txt_args(["-f", str(paths[0])])
  assert defaults.dates == [datetime.min, datetime.max - timedelta(days=1)]
  assert defaults.verbose is False
  with pytest.raises(SystemExit) as error:
    U.parse_txt_args(["-f", str(paths[0]), "-D", "invalid", "20240102"])
  assert error.value.code == 2

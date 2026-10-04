"""Source-derived PUN contracts using byte-preserved real summary records."""

from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from OGS.src import ogsconstants as C, ogsutils as U
from OGS.src.ogsdatafile import OGSDataFile
from OGS.src.ogspun import DataFilePUN
from OGS.src.ogstxt import DataFileTXT


DATA = Path(__file__).parent / "data"


def _slice(tmp_path, *line_numbers, name="case.pun"):
  """Copy selected 1-based lines verbatim, including their original endings."""
  lines = (DATA / "events.pun").read_bytes().splitlines(keepends=True)
  path = tmp_path / name
  path.write_bytes(b"".join(lines[number - 1] for number in line_numbers))
  return path


def _read(tmp_path, path=None, start=datetime(2000, 1, 1),
          end=datetime(2025, 1, 1)):
  parser = DataFilePUN(
      path or DATA / "events.pun", start=start, end=end,
      output=tmp_path / "output",
  )
  assert parser.read() is None
  return parser


# Expected fields are transcribed from the input, not produced by parser helpers.
# Coordinates are independently converted and rounded to the shared four decimals.
@pytest.mark.parametrize(
    "line,origin,lat,lon,depth,md,no,gap,dmin,rms,erh,erz,qm",
    [
        (2, "2005-01-01T14:22:09.040", 46.3500, 13.0957, 7.00, 2.43, 24, 59, 3.3, .24, .5, 1.2, "B1"),
        (3, "2005-01-02T11:56:15.320", 46.3482, 13.0905, 7.00, 2.70, 27, 56, 3.2, .30, .6, 1.2, "B1"),
        (4, "2005-01-03T03:09:59.830", 46.1547, 12.4360, 8.52, 2.64, 22, 147, 13.8, .25, .7, 1.8, "C1"),
        (5, "2005-01-03T13:14:28.370", 46.3443, 13.1000, 5.72, 1.85, 10, 152, 2.7, .21, 1., 1.7, "C1"),
        (6, "2005-01-03T14:57:28.290", 46.4895, 13.4357, 8.35, 1.73, 15, 175, 7.2, .12, .5, .9, "B1"),
        (7, "2005-01-04T03:59:30.160", 46.2175, 12.5032, 6.01, 2.25, 8, 142, 11.4, .13, .6, 2., "C1"),
        (8, "2005-01-04T09:42:30.860", 45.7512, 10.6732, 3.40, 2.82, 16, 232, 19.4, .31, 1.5, 3.7, "D1"),
        (9, "2023-04-19T22:12:50.250", 46.4268, 11.8472, 10.04, 1.43, 16, 105, 22.2, .15, .5, 1.8, "B1"),
        (10, "2023-04-20T10:16:21.480", 45.6388, 14.2962, 13.40, 1.32, 11, 110, 13.9, .14, .6, 1.3, "B1"),
        (11, "2023-04-20T16:32:35.430", 46.4138, 13.7940, .23, 1.73, 9, 196, 17.9, 2.24, 10.1, None, "D1"),
        (12, "2023-04-20T17:48:39.680", 45.9948, 13.7668, 10.93, 1.70, 33, 85, 10.2, .30, .6, 1.2, "B1"),
        (13, "2023-04-21T01:00:41.460", 45.7290, 11.0165, 11.40, 1.49, 19, 128, 14.3, .19, .5, 1.2, "B1"),
        (14, "2024-01-01T03:36:13.320", 46.7320, 12.4692, 4.39, .93, 10, 198, 3.7, .15, 1., 1.5, "C1"),
        (15, "2024-01-01T13:21:37.680", 46.3643, 12.9683, 12.12, 1.06, 14, 116, 6.1, .11, .4, .7, "B1"),
        (16, "2024-01-01T17:41:32.830", 46.4343, 13.3252, 10.14, 1.04, 10, 123, 3.9, .17, .8, 1.1, "B1"),
        (17, "2024-01-01T20:53:49.910", 45.7917, 11.1065, 12.37, 2.10, 19, 87, 11.7, .23, .6, 1.3, "B1"),
        (18, "2024-01-01T21:26:35.320", 46.4815, 13.7912, 6.99, .83, 10, 174, 17.9, .15, .7, 2.9, "C1"),
        (19, "2024-01-02T02:09:56.680", 46.6003, 13.8420, 7.97, 1.93, 18, 204, 15.7, .23, 1., 2.4, "C1"),
        (20, "2024-06-25T01:07:32.550", 45.5357, 14.4407, 10.63, 1.20, 5, 262, 5., .07, 2.4, 1.5, "C1"),
        (21, "2024-06-25T21:35:22.070", 45.6568, 12.0373, 7.47, .79, 4, 235, 5.8, .00, None, None, "C1"),
        (22, "2024-06-26T01:01:30.160", 45.8860, 11.1190, 10.84, 1.63, 15, 129, 5.5, .10, .4, .5, "B1"),
    ],
)
def test_fixed_width_summary(tmp_path, line, origin, lat, lon, depth, md,
                             no, gap, dmin, rms, erh, erz, qm):
  parser = _read(tmp_path, _slice(tmp_path, 1, line))
  assert len(parser.EVENTS) == 1
  row = parser.EVENTS.iloc[0]
  expected_time = datetime.fromisoformat(origin)
  assert row[C.TIME_STR] == expected_time
  assert row[C.IDX_EVENTS_STR] == expected_time.year * 1_000_000
  assert row[C.GROUPS_STR] == origin[:10]
  for column, value in (
      (C.LATITUDE_STR, lat), (C.LONGITUDE_STR, lon), (C.DEPTH_STR, depth),
      (C.MAGNITUDE_D_STR, md), (C.NO_STR, no), (C.GAP_STR, gap),
      (C.DMIN_STR, dmin), (C.RMS_STR, rms), (C.ERH_STR, erh), (C.ERZ_STR, erz),
  ):
    assert pd.api.types.is_numeric_dtype(parser.EVENTS[column])
    if value is None:
      assert pd.isna(row[column])
    else:
      assert row[column] == pytest.approx(value, abs=1e-8)
  assert row[C.QM_STR] == qm
  assert pd.isna(row[C.MAGNITUDE_L_STR])


def test_whole_fixture_schema_counters_and_daily_cache(tmp_path):
  parser = _read(tmp_path)
  frame = parser.EVENTS
  assert list(frame.columns) == OGSDataFile._EVENT_COLUMNS
  assert len(frame.columns) == 28
  assert frame[C.IDX_EVENTS_STR].tolist() == (
      [2005000000 + i for i in range(7)]
      + [2023000000 + i for i in range(7, 12)]
      + [2024000000 + i for i in range(12, 21)]
  )
  for column in (C.NUMBER_P_PICKS_STR, C.NUMBER_S_PICKS_STR,
                 C.NUMBER_P_AND_S_PICKS_STR):
    assert frame[column].eq(0).all()
  assert parser.PICKS.empty
  assert parser.picks == {}
  assert {day: len(rows) for day, rows in parser.events.items()} == {
      date(2005, 1, 1): 1, date(2005, 1, 2): 1,
      date(2005, 1, 3): 3, date(2005, 1, 4): 2,
      date(2023, 4, 19): 1, date(2023, 4, 20): 3,
      date(2023, 4, 21): 1, date(2024, 1, 1): 5,
      date(2024, 1, 2): 1, date(2024, 6, 25): 2,
      date(2024, 6, 26): 1,
  }
  for day, rows in parser.events.items():
    pd.testing.assert_frame_equal(
        rows, frame.loc[frame[C.GROUPS_STR] == day.isoformat()],
    )


@pytest.mark.parametrize(
    "numbers,erh,erz,qm",
    [
        ((9, 10, 11, 12, 13), [.5, .6, 10.1, .6, .5],
         [1.8, 1.3, None, 1.2, 1.2], ["B1", "B1", "D1", "B1", "B1"]),
        ((20, 21, 22), [2.4, None, .4],
         [1.5, None, .5], ["C1", "C1", "B1"]),
    ],
)
def test_real_uncertainty_placeholders_preserve_neighbors(tmp_path, numbers,
                                                          erh, erz, qm):
  frame = _read(tmp_path, _slice(tmp_path, 1, *numbers)).EVENTS
  assert len(frame) == len(numbers)
  assert frame[C.QM_STR].tolist() == qm
  for column, expected in ((C.ERH_STR, erh), (C.ERZ_STR, erz)):
    for actual, value in zip(frame[column], expected):
      if value is None:
        assert pd.isna(actual)
      else:
        assert actual == pytest.approx(value)
  if numbers[0] == 20:
    assert frame.iloc[1][C.RMS_STR] == 0.
    assert frame.iloc[1][C.NO_STR] == 4
    assert frame.iloc[1][C.MAGNITUDE_D_STR] == .79


@pytest.mark.parametrize(
    "start,end,origins",
    [
        (datetime(2005, 1, 1), datetime(2005, 1, 1),
         ["2005-01-01T14:22:09.040"]),
        (datetime(2005, 1, 3), datetime(2005, 1, 3),
         ["2005-01-03T03:09:59.830", "2005-01-03T13:14:28.370",
          "2005-01-03T14:57:28.290"]),
        (datetime(2024, 1, 2), datetime(2024, 1, 2),
         ["2024-01-02T02:09:56.680"]),
        (datetime(2025, 1, 1), datetime(2025, 1, 1), []),
    ],
)
def test_inclusive_calendar_windows_and_retained_counter(tmp_path, start, end,
                                                         origins):
  frame = _read(tmp_path, start=start, end=end).EVENTS
  assert frame[C.TIME_STR].tolist() == [
      datetime.fromisoformat(value) for value in origins
  ]
  assert frame[C.IDX_EVENTS_STR].tolist() == [
      start.year * 1_000_000 + i for i in range(len(origins))
  ]


def test_exact_start_inclusive_and_end_plus_day_exclusive(tmp_path):
  path = _slice(tmp_path, 1, 2, 3)
  origin = datetime(2005, 1, 1, 14, 22, 9, 40000)
  assert len(_read(tmp_path, path, start=origin).EVENTS) == 2
  assert len(_read(tmp_path, path, start=origin + timedelta(microseconds=1)).EVENTS) == 1
  assert _read(tmp_path, path, end=origin - timedelta(days=1)).EVENTS.empty


def test_pun_stops_at_first_record_after_end(tmp_path):
  # Reorder intact rows to expose PUN's documented chronological-input assumption.
  path = _slice(tmp_path, 1, 2, 14, 3)
  frame = _read(tmp_path, path, end=datetime(2005, 1, 2)).EVENTS
  assert frame[C.TIME_STR].tolist() == [datetime(2005, 1, 1, 14, 22, 9, 40000)]


def test_first_line_is_unconditionally_skipped(tmp_path):
  path = _slice(tmp_path, 2, 3)
  frame = _read(tmp_path, path).EVENTS
  assert frame[C.TIME_STR].tolist() == [datetime(2005, 1, 2, 11, 56, 15, 320000)]
  assert frame[C.IDX_EVENTS_STR].tolist() == [2005000000]


def test_non_record_header_inside_data_is_logged_and_skipped(tmp_path):
  path = _slice(tmp_path, 1, 2, 1, 3)
  parser = DataFilePUN(path, datetime(2005, 1, 1), datetime(2005, 1, 2),
                       output=tmp_path / "output")
  with patch.object(parser.logger, "error", wraps=parser.logger.error) as error:
    parser.read()
  error.assert_called_once()
  assert "(PUN) Could not parse line:" in error.call_args.args[0]
  frame = parser.EVENTS
  assert frame[C.TIME_STR].tolist() == [
      datetime(2005, 1, 1, 14, 22, 9, 40000),
      datetime(2005, 1, 2, 11, 56, 15, 320000),
  ]
  assert frame[C.IDX_EVENTS_STR].tolist() == [2005000000, 2005000001]


def test_same_hypo71_origins_but_independent_magnitude_precision(tmp_path):
  pun = _read(tmp_path, _slice(tmp_path, 1, *range(14, 20))).EVENTS
  txt = DataFileTXT(DATA / "hypo71.txt", datetime(2024, 1, 1),
                    datetime(2024, 1, 2), output=tmp_path / "txt")
  txt.read()
  assert pun[C.TIME_STR].tolist() == txt.EVENTS[C.TIME_STR].tolist()
  assert pun[C.MAGNITUDE_D_STR].tolist() == [.93, 1.06, 1.04, 2.10, .83, 1.93]
  assert txt.EVENTS[C.MAGNITUDE_D_STR].tolist() == [.9, 1.1, 1., 2.1, .8, 1.9]
  assert pun[C.IDX_EVENTS_STR].tolist() == list(range(2024000000, 2024000006))
  assert txt.EVENTS[C.IDX_EVENTS_STR].tolist() == list(range(2024000001, 2024000007))


@pytest.mark.parametrize("content", [b"", (DATA / "events.pun").read_bytes().splitlines(keepends=True)[0]])
def test_empty_or_header_only_input(tmp_path, content):
  path = tmp_path / "empty.pun"
  path.write_bytes(content)
  parser = _read(tmp_path, path)
  assert parser.EVENTS.empty
  assert list(parser.EVENTS.columns) == OGSDataFile._EVENT_COLUMNS
  assert parser.events == {}


def test_input_validation(tmp_path):
  with pytest.raises(FileNotFoundError, match="missing.pun"):
    DataFilePUN(tmp_path / "missing.pun", output=tmp_path / "output")
  path = _slice(tmp_path, 1, 2, name="wrong.txt")
  parser = DataFilePUN(path, output=tmp_path / "output")
  with pytest.raises(ValueError, match=r"extension must be \.pun"):
    parser.read()


@pytest.mark.parametrize("column", [C.DEPTH_STR, C.MAGNITUDE_D_STR])
@pytest.mark.parametrize("value,expected", [
    (" 0.00", 0.0), ("     ", None), ("-----", None),
])
def test_builder_converts_pun_depth_and_md(tmp_path, column, value, expected):
  parser = DataFilePUN(
      DATA / "events.pun", datetime(2005, 1, 1), datetime(2005, 1, 1),
      output=tmp_path / "output",
  )
  lines = (DATA / "events.pun").read_text().splitlines()
  line = lines[1]
  match = parser.EVENT_EXTRACTOR.match(line)
  assert match is not None
  start, end = match.span(column)
  assert len(value) == end - start
  line = line[:start] + value + line[end:]
  assert parser.EVENT_EXTRACTOR.match(line) is not None
  source = tmp_path / "converted.pun"
  source.write_text(lines[0] + "\n" + line + "\n")
  parser = _read(tmp_path, source)
  assert len(parser.EVENTS) == 1
  assert pd.api.types.is_numeric_dtype(parser.EVENTS[column])
  actual = parser.EVENTS[column].iloc[0]
  if expected is None:
    assert pd.isna(actual)
  else:
    assert actual == expected


def test_cli_arguments_without_main_side_effects():
  path = DATA / "events.pun"
  args = U.parse_pun_args(["-f", str(path), "-D", "20240102", "20240101", "-v"])
  assert args.file == [path.resolve()]
  assert args.dates == [datetime(2024, 1, 1), datetime(2024, 1, 2)]
  assert args.verbose is True
  defaults = U.parse_pun_args(["-f", str(path)])
  assert defaults.dates == [datetime.min, datetime.max - timedelta(days=1)]
  assert defaults.verbose is False
  with pytest.raises(SystemExit) as error:
    U.parse_pun_args(["-f", str(path), "-D", "invalid", "20240102"])
  assert error.value.code == 2

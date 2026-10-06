"""HPL block, field, and arrival contracts from separate location solutions."""

from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from OGS.src import ogsconstants as OGS_C
from OGS.src.ogshpl import DataFileHPL
from OGS.src.ogsutils import parse_hpl_args


FIRST_EVENT_ARRIVALS = [
    ("BOO", 10.18, 10.95, 0, 2),
    ("MPRI", 11.78, 14.06, 0, 2),
    ("BUA", 11.83, 14.23, 0, 2),
    ("BAD", 12.10, 14.78, 0, 2),
    ("PLRO", 13.00, 16.94, 0, 2),
    ("ZOU", 13.45, 17.23, 0, 2),
    ("COLI", 14.76, 19.78, 0, 2),
    ("ROBS", 14.59, None, 2, None),
    ("LSR", 15.33, 20.16, 0, 0),
    ("CSM", 15.54, 20.98, 0, 0),
    ("MLN", 16.50, 22.80, 0, 2),
    ("DRE", 16.76, 23.41, 0, 0),
    ("VOJS", 20.45, 31.09, 4, 2),
    ("JAVS", 23.54, None, 4, None),
    ("OBKA", 27.10, 42.30, 4, 4),
]


@pytest.mark.parametrize(
    "filename,segments,event_id,origin,lat,lon,depth,rms,erh,erz", [(
        "hypo71.hpl",
        ((1, 22),),
        2005000003,
        datetime(2005, 1, 1, 14, 22, 9, 40000),
        46.35,
        13.0957,
        7.0,
        .24,
        .5,
        1.2
    ), (
        "nll1d.hpl",
        ((1, 18),),
        2005000001,
        datetime(2005, 1, 1, 14, 22, 8, 350000),
        46.3527,
        13.0963,
        8.38,
        .27,
        1.4,
        1.6
    ),])
def test_first_located_event_and_every_arrival(
    reader_factory, filename, segments, event_id, origin, lat, lon, depth,
    rms, erh, erz,
):
  reader = reader_factory(DataFileHPL, filename, segments=segments)
  assert len(reader.EVENTS) == 1
  assert list(reader.EVENTS.columns) == DataFileHPL._EVENT_COLUMNS
  event = reader.EVENTS.iloc[0]
  assert event[OGS_C.IDX_EVENTS_STR] == event_id
  assert event[OGS_C.TIME_STR] == origin
  for column, expected in [
      (OGS_C.LATITUDE_STR, lat),
      (OGS_C.LONGITUDE_STR, lon),
      (OGS_C.DEPTH_STR, depth),
      (OGS_C.RMS_STR, rms),
      (OGS_C.ERH_STR, erh),
      (OGS_C.ERZ_STR, erz),
      (OGS_C.GAP_STR, 59),
  ]:
    assert event[column] == pytest.approx(expected)
  assert event[OGS_C.NUMBER_P_PICKS_STR] == 15
  assert event[OGS_C.NUMBER_S_PICKS_STR] == 13
  assert event[OGS_C.NUMBER_P_AND_S_PICKS_STR] == 13
  assert pd.isna(event[OGS_C.MAGNITUDE_L_STR])

  expected_rows = []
  base = datetime(2005, 1, 1, 14, 22)
  for station, p_seconds, s_seconds, p_weight, s_weight in FIRST_EVENT_ARRIVALS:
    expected_rows.append((
        f".{station}.",
        OGS_C.PWAVE,
        base + timedelta(seconds=p_seconds),
        p_weight
    ))
    if s_seconds is not None:
      expected_rows.append((
          f".{station}.",
          OGS_C.SWAVE,
          base + timedelta(seconds=s_seconds),
          s_weight
      ))
  columns = [
      OGS_C.STATION_STR, OGS_C.PHASE_STR, OGS_C.TIME_STR, OGS_C.WEIGHT_STR
  ]
  expected = pd.DataFrame(expected_rows, columns=columns).sort_values(
      columns[:2]
  ).reset_index(drop=True)
  actual = reader.PICKS[columns].sort_values(
      columns[:2]
  ).reset_index(drop=True)
  pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
  assert set(reader.PICKS[OGS_C.IDX_PICKS_STR]) == {event_id}
  assert set(reader.events) == {date(2005, 1, 1)}
  assert sum(map(len, reader.picks.values())) == 28
  assert not list(reader.output.rglob("*.parquet"))


@pytest.mark.parametrize("start,end,ids,counts", [
    (datetime(2005, 1, 1), datetime(2005, 1, 1), {2005000003}, (15, 13)),
    (datetime(2005, 1, 2), datetime(2005, 1, 2), {2005000008}, (25, 19)),
    (datetime(2005, 1, 1), datetime(2005, 1, 2),
     {2005000003, 2005000008}, (40, 32)),
])
def test_complete_blocks_survive_inclusive_date_windows(
    reader_factory, start, end, ids, counts,
):
  reader = reader_factory(DataFileHPL, "hypo71.hpl", start, end)
  assert set(reader.EVENTS[OGS_C.IDX_EVENTS_STR]) == ids
  assert set(reader.PICKS[OGS_C.IDX_PICKS_STR]) == ids
  assert reader.PICKS[OGS_C.PHASE_STR].value_counts().to_dict() == {
      OGS_C.PWAVE: counts[0], OGS_C.SWAVE: counts[1],
  }
  assert set(reader.events) == set(reader.picks)
  assert sum(map(len, reader.events.values())) == len(ids)


@pytest.mark.parametrize("filename,first,last,event_id,p_count,s_count", [
    ("hypo71.hpl", 51, 56, 2011000861, 3, 2),
    ("hypo71.hpl", 57, 68, 2023000001, 9, 8),
    ("hypo71.hpl", 69, 76, 2024000001, 5, 5),
    ("nll1d.hpl", 19, 23, 2005000020, 2, 2),
    ("nll1d.hpl", 24, 35, 2023000001, 9, 8),
    ("nll1d.hpl", 36, 43, 2024000001, 5, 5),
])
def test_complete_year_specific_blocks(
    reader_factory, filename, first, last, event_id, p_count, s_count,
):
  reader = reader_factory(DataFileHPL, filename, segments=((first, last),))
  assert reader.EVENTS[OGS_C.IDX_EVENTS_STR].tolist() == [event_id]
  assert set(reader.PICKS[OGS_C.IDX_PICKS_STR]) == {event_id}
  event = reader.EVENTS.iloc[0]
  assert event[OGS_C.NUMBER_P_PICKS_STR] == p_count
  assert event[OGS_C.NUMBER_S_PICKS_STR] == s_count
  assert event[OGS_C.NUMBER_P_AND_S_PICKS_STR] == s_count
  assert len(reader.PICKS) == p_count + s_count


@pytest.mark.parametrize("header_id,pick_id,event_id", [
    ("     3", "     3", 2005000003),
    ("  1 2 ", "  1 2 ", 2005000102),
    ("      ", "     3", None),
])
def test_header_ids_are_normalized_before_building(
    reader_factory, monkeypatch, header_id, pick_id, event_id,
):
  reader = reader_factory(DataFileHPL, "hypo71.hpl", segments=())
  lines = (
      Path(__file__).parent / "data" / "hypo71.hpl"
  ).read_text().splitlines()
  records = []
  for line, extractor, value in (
      (lines[6], reader.EVENT_EXTRACTOR, header_id),
      (lines[7], reader.RECORD_EXTRACTOR, pick_id),
  ):
    match = extractor.match(line)
    assert match is not None
    start, end = match.span(OGS_C.IDX_EVENTS_STR)
    assert end - start == len(value)
    records.append(line[:start] + value + line[end:])
  reader.input.write_text("\n".join(records) + "\n")
  build_events = Mock(wraps=reader._build_events_dataframe)
  monkeypatch.setattr(reader, "_build_events_dataframe", build_events)

  reader.read()

  assert build_events.call_args.args[0][0][OGS_C.IDX_EVENTS_STR] == event_id
  assert reader.EVENTS[OGS_C.IDX_EVENTS_STR].tolist() == [
      0 if event_id is None else event_id,
  ]
  expected_pick_id = 2005000003 if event_id is None else event_id
  assert set(reader.PICKS[OGS_C.IDX_PICKS_STR]) == {expected_pick_id}
  event = reader.EVENTS.iloc[0]
  assert event[OGS_C.NUMBER_P_PICKS_STR] == (0 if event_id is None else 1)
  assert event[OGS_C.NUMBER_S_PICKS_STR] == (0 if event_id is None else 1)
  assert event[OGS_C.NUMBER_P_AND_S_PICKS_STR] == (
      0 if event_id is None else 1
  )


@pytest.mark.parametrize("filename,line,primary,tail,stations", [
    ("hypo71.hpl", 7, 2.43, 2.4, 11),
    ("hypo71.hpl", 53, -.06, -.1, 2),
    ("hypo71.hpl", 59, 1.54, 1.5, 7),
    ("hypo71.hpl", 71, .93, .9, 3),
    ("nll1d.hpl", 3, 2.43, 2.4, 2),
    ("nll1d.hpl", 21, 0.0, 0.0, 0),
])
def test_raw_md_fields_are_extracted_independently(
    reader_factory, filename, line, primary, tail, stations,
):
  reader = reader_factory(DataFileHPL, filename, segments=())
  raw = (
      Path(__file__).parent / "data" / filename
  ).read_text().splitlines()[line - 1]
  match = reader.EVENT_EXTRACTOR.match(raw)
  assert match is not None
  fields = match.groupdict()
  assert float(fields[OGS_C.MAGNITUDE_D_STR]) == primary
  assert float(fields[OGS_C.HYPO71_MAG_STR]) == tail
  assert int(fields[OGS_C.MD_STATIONS_STR]) == stations


def test_station_clock_bases_and_seconds_rollover(reader_factory):
  reader = reader_factory(DataFileHPL, "hypo71.hpl", segments=((57, 68),))
  picks = reader.PICKS.set_index([OGS_C.STATION_STR, OGS_C.PHASE_STR])
  assert picks.loc[(".KOSI.", OGS_C.SWAVE), OGS_C.TIME_STR] == datetime(
      2023, 1, 1, 3, 36, 0, 680000
  )
  assert picks.loc[(".ABTA.", OGS_C.PWAVE), OGS_C.TIME_STR] == datetime(
      2023, 1, 1, 3, 36, 5, 190000
  )
  assert picks.loc[(".CSM.", OGS_C.PWAVE), OGS_C.TIME_STR] == datetime(
      2023, 1, 1, 3, 36, 8, 40000
  )
  assert (".CSM.", OGS_C.SWAVE) not in picks.index


def test_empty_window_and_empty_file_have_full_schemas(reader_factory):
  for segments in (None, ()):
    reader = reader_factory(
        DataFileHPL, "hypo71.hpl", datetime(2022, 1, 1),
        datetime(2022, 12, 31), segments=segments,
    )
    assert reader.EVENTS.empty and reader.PICKS.empty
    assert list(reader.EVENTS.columns) == DataFileHPL._EVENT_COLUMNS
    assert list(reader.PICKS.columns) == DataFileHPL._PICK_COLUMNS
    assert reader.events == reader.picks == {}


def test_blank_and_unmatched_lines_outside_blocks_are_ignored(reader_factory):
  reader = reader_factory(DataFileHPL, "hypo71.hpl", segments=())
  reader.input.write_text("\n   \nnot an event or metadata\n")
  reader.read()
  assert reader.EVENTS.empty and reader.PICKS.empty
  assert list(reader.EVENTS.columns) == DataFileHPL._EVENT_COLUMNS
  assert list(reader.PICKS.columns) == DataFileHPL._PICK_COLUMNS
  assert reader.events == reader.picks == {}


def test_note_metadata_updates_only_the_latest_named_record(reader_factory):
  reader = reader_factory(DataFileHPL, "hypo71.hpl", segments=())
  records = [
      {OGS_C.IDX_EVENTS_STR: 1, OGS_C.NOTES_STR: None},
      {OGS_C.IDX_EVENTS_STR: 2, OGS_C.NOTES_STR: None},
  ]
  assert reader._apply_metadata_line("* reviewed  ", records)
  assert records == [
      {OGS_C.IDX_EVENTS_STR: 1, OGS_C.NOTES_STR: None},
      {OGS_C.IDX_EVENTS_STR: 2, OGS_C.NOTES_STR: "reviewed"},
  ]
  assert reader._apply_metadata_line("* no retained event", [])
  assert not reader._apply_metadata_line("not metadata", records)


def test_input_validation(tmp_path):
  with pytest.raises(FileNotFoundError, match="does not exist"):
    DataFileHPL(tmp_path / "missing.hpl", output=tmp_path / "out")
  wrong = tmp_path / "wrong.dat"
  wrong.touch()
  reader = DataFileHPL(wrong, output=tmp_path / "out")
  with pytest.raises(ValueError, match=r"\.hpl"):
    reader.read()
  source = tmp_path / "removed.hpl"
  source.touch()
  reader = DataFileHPL(source, output=tmp_path / "out")
  source.unlink()
  with pytest.raises(FileNotFoundError, match="does not exist"):
    reader.read()


@pytest.mark.parametrize("column", [OGS_C.DEPTH_STR, OGS_C.RMS_STR])
@pytest.mark.parametrize("value,expected", [
    (" 0.00", 0.0), ("     ", None), (".....", None),
])
def test_builder_converts_hpl_depth_and_rms(
    reader_factory, column, value, expected,
):
  reader = reader_factory(DataFileHPL, "hypo71.hpl", segments=())
  line = (
      Path(__file__).parent / "data" / "hypo71.hpl"
  ).read_text().splitlines()[6]
  match = reader.EVENT_EXTRACTOR.match(line)
  assert match is not None
  start, end = match.span(column)
  assert len(value) == end - start
  line = line[:start] + value + line[end:]
  assert reader.EVENT_EXTRACTOR.match(line) is not None
  reader.input.write_text(line + "\n")
  reader.read()
  assert len(reader.EVENTS) == 1
  assert pd.api.types.is_numeric_dtype(reader.EVENTS[column])
  actual = reader.EVENTS[column].iloc[0]
  if expected is None:
    assert pd.isna(actual)
  else:
    assert actual == expected


def test_cli_accepts_real_fixture():
  source = Path(__file__).parent / "data" / "hypo71.hpl"
  args = parse_hpl_args(["-f", str(source), "-D", "20240101", "20240102"])
  assert args.file == [source]
  assert args.dates == [datetime(2024, 1, 1), datetime(2024, 1, 2)]

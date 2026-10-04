"""
=============================================================================
OGS Parser Test Suite - Unit Tests for Catalog Parsing & Registration
=============================================================================

Real-reader registration and cross-format catalog merge contracts.

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
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from OGS.src import ogsconstants as C
from OGS.src.ogsdat import DataFileDAT
from OGS.src.ogsdatafile import OGSDataFile
from OGS.src.ogshpl import DataFileHPL
from OGS.src.ogsparser import DataCatalog
from OGS.src.ogspun import DataFilePUN
from OGS.src.ogstxt import DataFileTXT
from OGS.src.ogsutils import parse_catalog_args


DATA = Path(__file__).parent / "data"


@pytest.fixture
def catalog(tmp_path):
  return DataCatalog(Namespace(
      output=tmp_path / "merged", dates=(datetime(1977, 1, 1), datetime(2024, 12, 31)),
      verbose=False, directory=None, file=[], ext=[".dat", ".hpl", ".pun", ".txt"],
  ))


@pytest.mark.parametrize("directory_mode", [False, True])
def test_registration_parses_real_files_without_writing_catalog_outputs(
    catalog, tmp_path, monkeypatch, directory_mode,
):
  filenames = ["picks.dat", "hypo71.hpl", "events.pun", "hypo71.txt"]
  expected_types = {DataFileDAT, DataFileHPL, DataFilePUN, DataFileTXT}
  log = Mock()
  monkeypatch.setattr(OGSDataFile, "log", log)
  if directory_mode:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for filename in filenames:
      (inputs / filename).write_bytes((DATA / filename).read_bytes())
    (inputs / "ignored.csv").write_text("not a catalog")
    catalog.args.directory = inputs
  else:
    catalog.args.file = [
        DATA / name for name in filenames] + [tmp_path / "ignored.csv"]
  catalog.args.dates = (datetime(2024, 1, 1), datetime(2024, 1, 2))
  catalog.read()
  assert {type(reader) for reader in catalog.files} == expected_types
  assert len(catalog.files) == log.call_count == 4
  for reader in catalog.files:
    assert reader.output == catalog.output
    assert not reader.PICKS.empty or not reader.EVENTS.empty
  assert not list(catalog.output.rglob("*.parquet"))


def test_reader_errors_propagate_without_successful_logging(catalog, tmp_path, monkeypatch):
  log = Mock()
  monkeypatch.setattr(OGSDataFile, "log", log)
  catalog.args.file = [tmp_path / "missing.dat"]
  with pytest.raises(FileNotFoundError):
    catalog.read()
  log.assert_not_called()


def test_multi_year_hpl_txt_merge_preserves_complementary_events(
    catalog, reader_factory,
):
  start, end = datetime(2023, 1, 1), datetime(2024, 12, 31)
  hpl = reader_factory(DataFileHPL, "hypo71.hpl", start, end)
  txt = reader_factory(DataFileTXT, "hypo71.txt", start, end)
  catalog.files = [txt, hpl]
  picks = catalog.merge_picks()
  events = catalog.merge_events().set_index(C.IDX_EVENTS_STR)
  assert set(events.index) == {2023000001,
                               2023000002, *range(2024000001, 2024000007)}
  assert len(events) == 8
  assert len(picks) == 27
  for event_id, p_count, s_count, ml, location in [
      (2023000001, 9, 8, 1.1, "S.LEONARDO PASSIRIA (ALTO ADIGE)"),
      (2024000001, 5, 5, .8, "M.CAVALLINO (ALTO ADIGE)"),
  ]:
    assert events.loc[event_id, C.NUMBER_P_PICKS_STR] == p_count
    assert events.loc[event_id, C.NUMBER_S_PICKS_STR] == s_count
    assert events.loc[event_id, C.NUMBER_P_AND_S_PICKS_STR] == s_count
    assert events.loc[event_id, C.TIME_STR] == hpl.EVENTS.set_index(
        C.IDX_EVENTS_STR).loc[event_id, C.TIME_STR]
    assert events.loc[event_id, C.MAGNITUDE_L_STR] == ml
    assert events.loc[event_id, C.LOC_NAME_STR] == location
    assert events.loc[event_id, C.EVENT_TYPE_STR] == "[earthquake]"
  assert events.loc[2023000002, C.MAGNITUDE_L_STR] == 1.3
  txt_only = events.loc[[2023000002, *range(2024000002, 2024000007)]]
  assert txt_only[[C.NUMBER_P_PICKS_STR,
                   C.NUMBER_S_PICKS_STR]].eq(0).all().all()
  assert set(events[C.GROUPS_STR]) == {
      "2023-01-01", "2024-01-01", "2024-01-02"}
  assert "__event_year" not in events


def test_real_dat_hpl_pick_identity_and_deduplication(catalog, reader_factory):
  start, end = datetime(2005, 1, 1), datetime(2005, 1, 2)
  dat = reader_factory(DataFileDAT, "picks.dat", start, end)
  hpl = reader_factory(DataFileHPL, "hypo71.hpl", start, end)
  keys = [C.IDX_PICKS_STR, C.STATION_STR, C.PHASE_STR]
  located = dat.PICKS[dat.PICKS[C.IDX_PICKS_STR].isin(
      {2005000003, 2005000008})]
  columns = keys + [C.TIME_STR, C.WEIGHT_STR]
  pd.testing.assert_frame_equal(
      located[columns].sort_values(keys).reset_index(drop=True),
      hpl.PICKS[columns].sort_values(keys).reset_index(drop=True),
      check_dtype=False,
  )
  catalog.files = [dat, hpl]
  picks = catalog.merge_picks()
  assert len(picks) == 76
  assert not picks.duplicated(keys).any()
  events = catalog.merge_events().set_index(C.IDX_EVENTS_STR)
  assert set(events.index) == {2005000003, 2005000008}
  assert events.loc[2005000003, C.NUMBER_P_PICKS_STR] == 15
  assert events.loc[2005000003, C.NUMBER_S_PICKS_STR] == 13
  assert events.loc[2005000008, C.NUMBER_P_PICKS_STR] == 25
  assert events.loc[2005000008, C.NUMBER_S_PICKS_STR] == 19


def test_pun_merges_on_origin_and_location_not_its_counter(catalog, reader_factory):
  start = end = datetime(2005, 1, 1)
  hpl = reader_factory(DataFileHPL, "hypo71.hpl", start, end)
  pun = reader_factory(DataFilePUN, "events.pun", start, end)
  catalog.files = [pun, hpl]
  catalog.merge_picks()
  events = catalog.merge_events()
  assert len(events) == 1
  assert events[C.IDX_EVENTS_STR].tolist() == [2005000003]
  assert events.iloc[0][C.TIME_STR] == datetime(2005, 1, 1, 14, 22, 9, 40000)
  assert events.iloc[0][C.NUMBER_P_PICKS_STR] == 15


def test_conflicting_hpl_solutions_are_not_silently_combined(catalog, reader_factory):
  catalog.files = [
      reader_factory(DataFileHPL, "hypo71.hpl", segments=((69, 76),)),
      reader_factory(DataFileHPL, "nll1d.hpl", segments=((36, 43),)),
  ]
  with pytest.raises(ValueError, match="Duplicate year/event ID in HPL"):
    catalog.merge_events()


def test_duplicate_identity_within_one_input_is_rejected(catalog, reader_factory):
  reader = reader_factory(DataFileHPL, "hypo71.hpl", segments=((69, 76),))
  reader.EVENTS = pd.concat([reader.EVENTS, reader.EVENTS], ignore_index=True)
  catalog.files = [reader]
  with pytest.raises(ValueError, match="Duplicate year/event ID"):
    catalog.merge_events()


def test_missing_event_year_is_explicitly_rejected(catalog, reader_factory):
  reader = reader_factory(DataFileHPL, "hypo71.hpl", segments=((69, 76),))
  reader.EVENTS[C.TIME_STR] = pd.NaT
  reader.EVENTS[C.GROUPS_STR] = None
  catalog.files = [reader]
  with pytest.raises(ValueError, match="Cannot determine event year"):
    catalog.merge_events()


def test_empty_merge_is_safe(catalog):
  assert catalog.merge_picks().empty
  assert catalog.merge_events().empty


def test_cli_selects_real_files_and_sorts_dates(tmp_path):
  args = parse_catalog_args([
      "-f", str(DATA / "hypo71.hpl"), str(DATA / "picks.dat"),
      "-D", "20240102", "20240101", "--merge", "-v", "-o", str(tmp_path),
  ])
  assert args.file == [DATA / "hypo71.hpl", DATA / "picks.dat"]
  assert args.dates == [datetime(2024, 1, 1), datetime(2024, 1, 2)]
  assert args.merge and args.verbose
  assert args.output == tmp_path

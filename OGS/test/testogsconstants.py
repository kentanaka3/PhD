"""
===============================================================================
OGS Constants Test Suite - Unit Tests for Inventory & Waveform Lookups
===============================================================================

Configuration contracts and local inventory/archive lookup tests.

OVERVIEW:
Unit test suite for ``ogsconstants.py`` and associated data lookups. Verifies
station inventory parsing and miniSEED waveform file discovery.

TEST CASES & INVARIANTS:
  1. test_inventory: Validates station inventory DataFrame schema, coordinates,
     and network/station code extraction.
  2. test_waveforms: Validates waveform path indexing, date parsing, and
     channel discovery across archive directories.

USAGE:
From the repository root:
python -m unittest OGS.test.testogsconstants

DEPENDENCIES:
- unittest: standard library testing framework
  - pandas: DataFrame verification
  - ogsconstants / ogsutils: constants and inventory utilities under test

AUTHORS:
  - 健
  - Istituto Nazionale di Oceanografia e di Geofisica Sperimentale (OGS)
    Centro di Ricerche Sismologiche (CRS)
  - Università degli Studi di Trieste (UniTS)
    Dipartimento di Matematica, Informatica e Geoscienze (MIGe)
    Applied Data Science and Artificial Intelligence (ADSAI)
  - Terabit Network for Research and Academic Big Data in Italy (TeRABIT)
    Consorzio Interuniversitario del Nord-Est per il Calcolo Automatico (CINECA)

===============================================================================
"""

import logging
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock
import pandas as pd
import pytest
from matplotlib.path import Path as Polygon
from obspy import Inventory
from OGS.src import ogsconstants as OGS_C, ogsplotter, ogsutils
from obspy.core.inventory import Network, Station


@pytest.mark.parametrize("name,value,text", [
    ("DATE_FMT", datetime(2024, 2, 29), "2024-02-29"),
    ("YYMMDD_FMT", datetime(2024, 2, 29), "240229"),
    ("YYYYMMDD_FMT", datetime(2024, 2, 29), "20240229"),
    ("TIME_FMT", datetime(1900, 1, 1, 3, 4, 5), "030405"),
    ("DATETIME_FMT", datetime(2024, 2, 29, 3, 4, 5), "240229030405"),
])
def test_date_formats_have_exact_parseable_representations(name, value, text):
  format_string = getattr(OGS_C, name)
  assert value.strftime(format_string) == text
  assert datetime.strptime(text, format_string) == value


@pytest.mark.parametrize("env,expected", [
    ({}, 1),
    ({"SLURM_CPUS_PER_TASK": "8"}, 8),
    ({"CORES": "3", "SLURM_CPUS_PER_TASK": "8"}, 3),
])
def test_cpu_configuration_precedence(monkeypatch, env, expected):
  for key in ("CORES", "SLURM_CPUS_PER_TASK"):
    monkeypatch.delenv(key, raising=False)
  for key, value in env.items():
    monkeypatch.setenv(key, value)
  spec = importlib.util.spec_from_file_location(
      "isolated_constants", OGS_C.__file__)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  assert module.DEFAULT_CORES_COUNT == expected
  assert (module.MPI_RANK, module.MPI_SIZE, module.MPI_COMM) == (0, 1, None)
  assert (module.GPU_RANK, module.GPU_SIZE) == (-1, 0)


def test_matching_and_training_intervals_are_configuration_not_measurements():
  assert OGS_C.ONE_DAY == timedelta(days=1)
  assert OGS_C.PICK_TIME_OFFSET == timedelta(seconds=.5)
  assert OGS_C.EVENT_TIME_OFFSET == timedelta(seconds=2)
  assert OGS_C.PICK_TRAIN_OFFSET == timedelta(seconds=60)
  assert OGS_C.EVENT_DIST_OFFSET == 8
  assert OGS_C.H71_OFFSET == {0: .01, 1: .04, 2: .2, 3: 1, 4: 5}
  assert OGS_C.MAX_PICKS_YEAR == 1_000_000
  assert OGS_C.PWAVE == "P" and OGS_C.SWAVE == "S"


@pytest.mark.parametrize("name,suffix", [
    ("DAT_EXT", ".dat"), ("HPL_EXT", ".hpl"), ("PUN_EXT", ".pun"),
    ("TXT_EXT", ".txt"), ("MSEED_EXT", ".mseed"), ("XML_EXT", ".xml"),
])
def test_file_suffix_contracts(name, suffix):
  assert getattr(OGS_C, name) == suffix
  assert Path(f"example{suffix}").suffix == getattr(OGS_C, name)


def test_event_classification_and_zone_codes_are_explicit():
  assert OGS_C.OGS_EVENT_TYPES == {
      "B": "bomb", "E": "explosion", "F": "landslide",
      "L": "local_eq", "R": "regional", "U": "UNKNOWN",
  }
  assert OGS_C.OGS_GEO_ZONES["F"] == "Friuli"
  assert OGS_C.OGS_GEO_ZONES["A"] == "Alto Adige"
  assert all(len(code) == 1 for code in OGS_C.OGS_GEO_ZONES)
  assert OGS_C.SEED_ID_FMT.format(
      NETWORK="OX", STATION="TEST", CHANNEL="HHZ") == "OX.TEST..HHZ"


def test_region_uses_longitude_latitude_order_and_contains_friuli():
  west, east, south, north = OGS_C.OGS_STUDY_REGION
  assert west < east and south < north
  assert all(west <= lon <= east and south <= lat <=
             north for lon, lat in OGS_C.OGS_POLY_REGION)
  polygon = Polygon(OGS_C.OGS_POLY_REGION)
  assert polygon.contains_point((13.1, 46.2))
  assert not polygon.contains_point((0, 0))


@pytest.fixture
def station_directory(tmp_path):
  directory = tmp_path / "stations"
  directory.mkdir()
  stations = [
      Station("TEST", latitude=46.2, longitude=13.1, elevation=120),
      Station("OTHER", latitude=46.3, longitude=13.2, elevation=240),
  ]
  Inventory([Network("OX", stations=stations)], source="unit test").write(
      str(directory / "local.xml"), format="STATIONXML"
  )
  return directory


def test_inventory_reads_local_stationxml_and_writes_only_requested_output(
    tmp_path, station_directory,
):
  frame = ogsutils.inventory(station_directory, output=tmp_path)
  columns = [OGS_C.IDX_EVENTS_STR, OGS_C.LONGITUDE_STR, OGS_C.LATITUDE_STR,
             OGS_C.DEPTH_STR, OGS_C.NETWORK_STR, OGS_C.STATION_STR]
  expected = pd.DataFrame([
      ["OX.OTHER.", 13.2, 46.3, 240.0, "OX", "OTHER"],
      ["OX.TEST.", 13.1, 46.2, 120.0, "OX", "TEST"],
  ], columns=columns)
  pd.testing.assert_frame_equal(frame[columns], expected)
  for column in (OGS_C.NETCOLOR_STR, OGS_C.STACOLOR_STR):
    assert all(len(color) == 4 and all(0 <= component <= 1 for component in color)
               for color in frame[column])
  assert frame[OGS_C.NETCOLOR_STR].iloc[0] == frame[OGS_C.NETCOLOR_STR].iloc[1]
  assert (tmp_path / "OGSInventory.csv").is_file()


def test_invalid_inventory_is_reported_not_compared_to_old_parquet(
    tmp_path, station_directory, caplog, monkeypatch,
):
  monkeypatch.setattr(
      ogsutils, "setup_logger", Mock(
          return_value=logging.getLogger("test.inventory")),
  )
  (station_directory / "broken.xml").write_text("not StationXML")
  assert len(ogsutils.inventory(station_directory)) == 2
  assert "Unable to read" in caplog.text
  empty = tmp_path / "empty"
  empty.mkdir()
  with pytest.raises(FileNotFoundError, match="No valid StationXML"):
    ogsutils.inventory(empty)
  with pytest.raises(FileNotFoundError, match="Station directory not found"):
    ogsutils.inventory(tmp_path / "missing")


@pytest.mark.parametrize("threads", [1, 2])
def test_waveform_discovery_filters_days_and_matches_network_station(
    tmp_path, station_directory, monkeypatch, threads,
):
  root = tmp_path / "waveforms"
  for day in ("2024/01/01", "2024/01/02", "2024/01/03"):
    directory = root / day
    directory.mkdir(parents=True)
    compact = day.replace("/", "")
    (directory /
     f"OX.TEST.00.HHZ__{compact}T000000Z__{compact}T235959Z.mseed").touch()
  (root / "2024/01/02" / "unrelated.txt").touch()
  map_plot = Mock()
  availability_plot = Mock()
  monkeypatch.setattr(ogsplotter, "map_plotter", map_plot)
  monkeypatch.setattr(ogsplotter, "stack_plotter", availability_plot)
  frame, inventory = ogsutils.waveforms(
      root, station_directory, datetime(2024, 1, 1), datetime(2024, 1, 2),
      output=tmp_path, threads=threads,
  )
  assert len(frame) == 2
  assert frame[OGS_C.NETWORK_STR].tolist() == ["OX", "OX"]
  assert frame[OGS_C.STATION_STR].tolist() == ["TEST", "TEST"]
  assert frame[OGS_C.LOC_NAME_STR].tolist() == ["00", "00"]
  assert frame[OGS_C.CHANNEL_STR].tolist() == ["HHZ", "HHZ"]
  assert [str(value)[:10] for value in frame[OGS_C.DATE_STR]] == [
      "2024-01-01", "2024-01-02"]
  assert all(isinstance(path, Path) and path.is_file()
             for path in frame[OGS_C.FILENAME_STR])
  assert inventory[OGS_C.IDX_EVENTS_STR].tolist() == ["OX.TEST."]
  assert (tmp_path / "OGSWaveforms.csv").is_file()
  map_plot.return_value.savefig.assert_called_once_with(
      tmp_path / "OGSStations.png")
  availability_plot.assert_called_once()

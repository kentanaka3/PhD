"""
=============================================================================
OGS Data Test Suite - Unit Tests for Squirrel Waveform Cache & Indexing
=============================================================================

OVERVIEW:
Unit test suite for ``ogsdata.py``. Validates the day-sharded Pyrocko Squirrel
data access layer, path filtering, database cache invalidation, and multi-core
parallel shard indexing.

TEST CASES & INVARIANTS:
  1. Day Filtering: Verifies that ``_select_squirrel_add_paths`` prunes paths
     outside configured target dates while preserving station XML metadata.
  2. Non-OGS Path Preservation: Ensures external data directories and arbitrary
     waveform sources are retained without date pruning.
  3. Cache Invalidation: Validates cache-key generation and ensures stale cache
     entries are invalidated when directory layouts change.
  4. DataSource Registration: Confirms Squirrel data sources correctly ingest
     selected day shards into the active runtime.

USAGE:
python -m unittest OGS/test/testogsdata.py

DEPENDENCIES:
- unittest / tempfile: isolated filesystem testing
  - ml_catalog.data: base Squirrel data source primitives
  - ogsdata: waveform cache and path selector under test

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

from ogsdata import _SQUIRREL_OGS_DAY_DIR_CACHE, _select_squirrel_add_paths
from ml_catalog import data
import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from obspy import UTCDateTime

THIS_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(THIS_DIR + "/../src"))


class TestOGSData(unittest.TestCase):
  def setUp(self):
    self.tempdir = tempfile.TemporaryDirectory()
    self.tmp_path = Path(self.tempdir.name)
    _SQUIRREL_OGS_DAY_DIR_CACHE.clear()

  def tearDown(self):
    _SQUIRREL_OGS_DAY_DIR_CACHE.clear()
    self.tempdir.cleanup()

  def _make_day(self, root, day):
    year, month, day_of_month = day.split("-")
    day_path = root / year / month / day_of_month
    day_path.mkdir(parents=True)
    return day_path

  def test_select_squirrel_add_paths_filters_days(self):
    root = self.tmp_path / "waveforms"
    before = self._make_day(root, "2020-01-01")
    start = self._make_day(root, "2020-01-02")
    middle = self._make_day(root, "2020-01-03")
    end = self._make_day(root, "2020-01-04")
    after = self._make_day(root, "2020-01-05")

    selected = _select_squirrel_add_paths(
        [root], UTCDateTime("2020-01-02"), UTCDateTime("2020-01-04")
    )

    self.assertEqual(
        selected, [start, middle, end])
    self.assertNotIn(before, selected)
    self.assertNotIn(after, selected)

  def test_select_squirrel_add_paths_preserves_metadata(self):
    root = self.tmp_path / "waveforms"
    station_dir = root / "station"
    station_dir.mkdir(parents=True)
    selected_day = self._make_day(root, "2020-01-02")
    inventory = self.tmp_path / "stations.xml"
    inventory.touch()

    selected = _select_squirrel_add_paths(
        [inventory, root], UTCDateTime("2020-01-02"), UTCDateTime("2020-01-02")
    )

    self.assertEqual(selected, [inventory, station_dir, selected_day])

  def test_select_squirrel_add_paths_preserves_non_ogs_paths(self):
    root = self.tmp_path / "waveforms"
    (root / "network" / "station").mkdir(parents=True)

    selected = _select_squirrel_add_paths(
        [root], UTCDateTime("2020-01-02"), UTCDateTime("2020-01-02")
    )

    self.assertEqual(selected, [root])

  def test_select_squirrel_add_paths_cache_key_and_invalidation(self):
    root = self.tmp_path / "waveforms"
    self._make_day(root, "2020-01-01")
    start = UTCDateTime("2020-01-02")
    end = UTCDateTime("2020-01-02")

    self.assertEqual(_select_squirrel_add_paths([root], start, end), [root])
    self.assertIn((root.resolve(strict=False), start.date, end.date),
                  _SQUIRREL_OGS_DAY_DIR_CACHE)

    day = self._make_day(root, "2020-01-02")
    self.assertEqual(_select_squirrel_add_paths([root], start, end), [day])

  def test_squirrel_data_source_adds_selected_paths(self):
    root = self.tmp_path / "waveforms"
    station_dir = root / "station"
    station_dir.mkdir(parents=True)
    start_day = self._make_day(root, "2020-01-02")
    end_day = self._make_day(root, "2020-01-03")
    self._make_day(root, "2020-01-04")
    env = self.tmp_path / "squirrel_env"
    (env / ".squirrel").mkdir(parents=True)

    class FakeSquirrel:
      add_calls = []

      def __init__(self, **kwargs):
        self.kwargs = kwargs

      def add(self, paths, check=False):
        self.add_calls.append((list(paths), check))

      def get_stations(self, codes):
        return []

    with unittest.mock.patch.object(data, "Squirrel", FakeSquirrel):
      data.SquirrelDataSource(
          env=str(env), paths=[root],
          starttime=UTCDateTime("2020-01-02"),
          endtime=UTCDateTime("2020-01-03"), check=True,
      )

    self.assertEqual(
        FakeSquirrel.add_calls, [([station_dir, start_day, end_day], True)]
    )


if __name__ == "__main__":
  unittest.main()

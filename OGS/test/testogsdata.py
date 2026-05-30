from pathlib import Path

import pytest
from obspy import UTCDateTime

from ml_catalog import data
from ml_catalog.data import _SQUIRREL_OGS_DAY_DIR_CACHE, _select_squirrel_add_paths


@pytest.fixture(autouse=True)
def clear_squirrel_ogs_day_dir_cache():
	_SQUIRREL_OGS_DAY_DIR_CACHE.clear()
	yield
	_SQUIRREL_OGS_DAY_DIR_CACHE.clear()


def _make_day(root: Path, day: str) -> Path:
	year, month, day_of_month = day.split("-")
	day_path = root / year / month / day_of_month
	day_path.mkdir(parents=True)
	return day_path


def test_select_squirrel_add_paths_filters_ogs_days_inclusively(tmp_path):
	root = tmp_path / "waveforms"
	before = _make_day(root, "2020-01-01")
	start = _make_day(root, "2020-01-02")
	middle = _make_day(root, "2020-01-03")
	end = _make_day(root, "2020-01-04")
	after = _make_day(root, "2020-01-05")
	(root / "2020" / "13" / "01").mkdir(parents=True)
	(root / "2020" / "02" / "30").mkdir(parents=True)
	(root / "2020" / "01" / "not-a-day").mkdir()

	selected = _select_squirrel_add_paths(
		[root], UTCDateTime("2020-01-02"), UTCDateTime("2020-01-04")
	)

	assert selected == [start, middle, end]
	assert before not in selected
	assert after not in selected


@pytest.mark.parametrize("root_name", ["station_waveforms", "inventory_waveforms"])
def test_select_squirrel_add_paths_prefers_ogs_days_for_metadata_named_roots(
	tmp_path, root_name
):
	root = tmp_path / root_name
	selected_day = _make_day(root, "2020-01-02")
	_make_day(root, "2020-01-03")

	selected = _select_squirrel_add_paths(
		[root], UTCDateTime("2020-01-02"), UTCDateTime("2020-01-02")
	)

	assert selected == [selected_day]
	assert root not in selected


def test_select_squirrel_add_paths_preserves_station_metadata(tmp_path):
	root = tmp_path / "waveforms"
	station_dir = root / "station"
	station_dir.mkdir(parents=True)
	selected_day = _make_day(root, "2020-01-02")
	explicit_inventory = tmp_path / "stations.xml"
	explicit_inventory.touch()

	selected = _select_squirrel_add_paths(
		[explicit_inventory, root],
		UTCDateTime("2020-01-02"),
		UTCDateTime("2020-01-02"),
	)

	assert selected == [explicit_inventory, station_dir, selected_day]


def test_select_squirrel_add_paths_passes_non_ogs_layout_through(tmp_path):
	non_ogs_root = tmp_path / "waveforms"
	(non_ogs_root / "network" / "station").mkdir(parents=True)

	selected = _select_squirrel_add_paths(
		[non_ogs_root],
		UTCDateTime("2020-01-02"),
		UTCDateTime("2020-01-02"),
	)

	assert selected == [non_ogs_root]


def test_select_squirrel_add_paths_keeps_ogs_empty_window_narrow(tmp_path):
	root = tmp_path / "waveforms"
	station_dir = root / "station"
	station_dir.mkdir(parents=True)
	_make_day(root, "2020-01-01")

	selected = _select_squirrel_add_paths(
		[root], UTCDateTime("2020-01-02"), UTCDateTime("2020-01-03")
	)

	assert selected == [station_dir]


def test_select_squirrel_add_paths_deduplicates_preserving_order(tmp_path):
	root = tmp_path / "waveforms"
	first_day = _make_day(root, "2020-01-02")
	second_day = _make_day(root, "2020-01-03")
	non_ogs_root = tmp_path / "other"
	non_ogs_root.mkdir()

	selected = _select_squirrel_add_paths(
		[str(root), root, non_ogs_root, str(non_ogs_root)],
		UTCDateTime("2020-01-02"),
		UTCDateTime("2020-01-03"),
	)

	assert selected == [first_day, second_day, non_ogs_root]


def test_select_squirrel_add_paths_preserves_original_paths_without_valid_window(
	tmp_path,
):
	root = tmp_path / "waveforms"
	_make_day(root, "2020-01-02")
	station_dir = root / "station"
	station_dir.mkdir()

	selected = _select_squirrel_add_paths([str(root), root, station_dir], None, None)

	assert selected == [str(root), station_dir]


def test_select_squirrel_add_paths_caches_ogs_day_index(tmp_path):
	root = tmp_path / "waveforms"
	day = _make_day(root, "2020-01-02")

	selected = _select_squirrel_add_paths(
		[root], UTCDateTime("2020-01-02"), UTCDateTime("2020-01-02")
	)

	assert selected == [day]
	assert root.resolve(strict=False) in _SQUIRREL_OGS_DAY_DIR_CACHE


def test_squirrel_data_source_adds_selected_ogs_paths(monkeypatch, tmp_path):
	root = tmp_path / "waveforms"
	station_dir = root / "station"
	station_dir.mkdir(parents=True)
	start_day = _make_day(root, "2020-01-02")
	end_day = _make_day(root, "2020-01-03")
	_make_day(root, "2020-01-04")

	env = tmp_path / "squirrel_env"
	(env / ".squirrel").mkdir(parents=True)

	class FakeSquirrel:
		add_calls = []

		def __init__(self, **kwargs):
			self.kwargs = kwargs

		def add(self, paths, check=False):
			self.add_calls.append((list(paths), check))

		def get_stations(self, codes):
			return []

	monkeypatch.setattr(data, "Squirrel", FakeSquirrel)

	data.SquirrelDataSource(
		env=str(env),
		paths=[root],
		starttime=UTCDateTime("2020-01-02"),
		endtime=UTCDateTime("2020-01-03"),
		check=True,
	)

	assert FakeSquirrel.add_calls == [([station_dir, start_day, end_day], True)]

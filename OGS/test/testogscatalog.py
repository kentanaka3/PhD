"""
=============================================================================
OGS Catalog Test Suite - Unit Tests for Catalog Indexing & BGMA Review
=============================================================================

OVERVIEW:
Offline catalog contracts using generated daily files and hand-scored graphs.

Catalog loading preserves source schemas; coordinate normalization belongs to
OGSDataFile, not OGSCatalog. Matching scores and horizontal-only eligibility
are implementation contracts, not evidence of scientific event identity.
Polygon boundary policy, duplicate files within one category, cache refresh,
and depth units need separate policy decisions; these tests do not invent them.

Run with pytest, bytecode/cache disabled, and a repository-local --basetemp.
No fixtures from real catalogs, plotting, downloads, or inference are used.

TEST CASES & INVARIANTS:
  1. TestOGSCatalogEventPrefilter:
     - Geographic candidate masking and projection onto boundary polygons.
     - Prefilter event tracking for out-of-region rows outside BGMA review.
     - Feasible event position pruning based on velocity and travel time.
     - Mixed timezone UTC normalization for event origin timestamps.
     - Integrity of partitioned review frames (both, base-only, target-only).
  2. TestOGSBPGraphPicks:
     - Station candidate window pruning and temporal tolerance bounds.
  3. TestOGSDistanceMetrics:
     - Pick phase match scoring, probability ratios, and zero-division guards.
  4. TestOGSCatalogNormalizeCoordinates:
     - 4-decimal rounding of decimal floats.
     - Degree-minute conversion to decimal degrees.
     - Physical bounds masking (-90 <= lat <= 90, -180 <= lon <= 180).
     - Coercion of dash sequences and 'None' placeholders to NaN.

USAGE:
python -m unittest OGS/test/testogscatalog.py

DEPENDENCIES:
- unittest / unittest.mock: test runner and catalog mocking
  - numpy / pandas: array computations and tabular verification
  - matplotlib.path: polygon membership testing
  - ogscatalog / ogsconstants / ogsutils: core catalog implementation
  - ogsdatafile: base parser class and normalization routines

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

import unittest
import unittest.mock
from OGS.src.ogsutils import (
    OGSBPGraph, OGSBPGraphEvents, OGSBPGraphPicks,
    diff_space, dist_prob, dist_pick, dist_event, dist_time,
)
from OGS.src.ogsdatafile import OGSDataFile
from OGS.src.ogscatalog import OGSCatalog, _EVENTS_MH_COLUMNS, _EVENTS_PHASES
from OGS.src import ogscatalog as catalog_module
from OGS.src import ogsconstants as OGS_C
import numpy as np
import pandas as pd
from matplotlib.path import Path as mplPath
from obspy import UTCDateTime
import pytest
from unittest.mock import Mock
from typing import Any, Sequence
from pathlib import Path
from datetime import date, datetime, timedelta


class TestOGSCatalogEventPrefilter(unittest.TestCase):
  def setUp(self) -> None:
    self.catalog = OGSCatalog.__new__(OGSCatalog)
    self.catalog.logger = unittest.mock.MagicMock()

  def _event_frame(self) -> pd.DataFrame:
    return pd.DataFrame({
        OGS_C.IDX_EVENTS_STR: [1, 2, 3],
        OGS_C.TIME_STR: [
            "2024-01-01T00:00:00",
            "2024-01-01T00:01:00",
            "2024-01-01T00:02:00",
        ],
        OGS_C.LATITUDE_STR: [0.5, 3.0, 1.5],
        OGS_C.LONGITUDE_STR: [0.5, 3.0, 1.5],
        OGS_C.DEPTH_STR: [1.0, 2.0, 3.0],
        OGS_C.ERH_STR: [0.1, 0.2, 0.3],
        OGS_C.ERZ_STR: [0.1, 0.2, 0.3],
        OGS_C.GAP_STR: [10, 20, 30],
        OGS_C.MAGNITUDE_L_STR: [1.1, 2.2, 3.3],
        OGS_C.GROUPS_STR: ["2024-01-01", "2024-01-01", "2024-01-01"],
    })

  def test_event_candidate_mask_projects_onto_polygon(self):
    events = self._event_frame()
    polygon = mplPath([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])

    mask = self.catalog._event_candidate_mask(
        events, np.asarray(polygon.vertices))

    np.testing.assert_array_equal(mask, np.array([True, False, True]))

  def test_prefilter_events_tracks_filtered_rows_outside_bpgma(self):
    events = self._event_frame()
    polygon = mplPath([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
    skimmed_frames: list[pd.DataFrame] = []

    candidates = self.catalog._prefilter_events(
        events,
        np.asarray(polygon.vertices),
        skimmed_frames,
        pd.Timestamp("2024-01-01"),
        OGS_C.TARGET_STR,
    )

    expected_candidates = events.iloc[[0, 2]].reset_index(drop=True)
    pd.testing.assert_frame_equal(candidates, expected_candidates)
    self.assertEqual(len(skimmed_frames), 1)
    pd.testing.assert_frame_equal(
        skimmed_frames[0],
        events.iloc[[1]].reindex(
            columns=_EVENTS_MH_COLUMNS
        ).reset_index(
            drop=True
        ),
    )

  def test_event_feasible_positions_prune_impossible_rows(self):
    base = pd.DataFrame({
        OGS_C.TIME_STR: [
            "2024-01-01T00:00:00",
            "2024-01-01T00:10:00",
        ],
        OGS_C.LATITUDE_STR: [46.0, 46.0],
        OGS_C.LONGITUDE_STR: [13.0, 13.0],
        OGS_C.DEPTH_STR: [1000.0, 1000.0],
    })
    target = pd.DataFrame({
        OGS_C.TIME_STR: [
            "2024-01-01T00:00:01",
            "2024-01-01T01:00:00",
        ],
        OGS_C.LATITUDE_STR: [46.0, 47.0],
        OGS_C.LONGITUDE_STR: [13.0, 14.0],
        OGS_C.DEPTH_STR: [1000.0, 1000.0],
    })

    base_pos, target_pos = self.catalog._event_feasible_positions(base, target)

    np.testing.assert_array_equal(base_pos, np.array([0]))
    np.testing.assert_array_equal(target_pos, np.array([0]))

  def test_mh_diff_normalizes_mixed_timezone_event_times(self):
    self.catalog.EventsMH = pd.DataFrame({
        f"{OGS_C.TIME_STR}_base": [
            pd.Timestamp("2024-01-01T00:00:00"),
            pd.Timestamp("2024-01-01T00:00:10Z"),
        ],
        f"{OGS_C.TIME_STR}_target": [
            pd.Timestamp("2024-01-01T00:00:01Z"),
            pd.Timestamp("2024-01-01T00:00:11"),
        ],
    })

    diff = self.catalog._mh_diff(OGS_C.TIME_STR)

    np.testing.assert_allclose(diff.to_numpy(), np.array([1.0, 1.0]))

  def test_bpgma_events_review_includes_prefiltered_partitions(self):
    target = OGSCatalog.__new__(OGSCatalog)
    target.logger = unittest.mock.MagicMock()
    target.EVENTS = pd.DataFrame({OGS_C.TIME_STR: [1, 2, 3, 4]})
    target.events = {}
    target.events_ = {}
    target._write_csv = unittest.mock.Mock()

    self.catalog.EVENTS = pd.DataFrame({OGS_C.TIME_STR: [1, 2, 3]})
    self.catalog.events = {}
    self.catalog.events_ = {}
    self.catalog.EventsMH = pd.DataFrame(index=range(1))
    self.catalog.EventsMS = pd.DataFrame(index=range(1))
    self.catalog.EventsSM = pd.DataFrame(index=range(1))
    self.catalog.EventsPS = pd.DataFrame(index=range(2))
    self.catalog.EventsSP = pd.DataFrame(index=range(1))
    self.catalog._write_csv = unittest.mock.Mock()
    self.catalog._log_review_checks = unittest.mock.Mock()

    events_cfn_mtx = self.catalog._empty_cfn_mtx(_EVENTS_PHASES)
    self.catalog._add(events_cfn_mtx, OGS_C.EVENT_STR, OGS_C.EVENT_STR, 1)
    self.catalog._add(events_cfn_mtx, OGS_C.EVENT_STR, OGS_C.NONE_STR, 1)
    self.catalog._add(events_cfn_mtx, OGS_C.NONE_STR, OGS_C.EVENT_STR, 2)

    recall, fdr = self.catalog._BPGMA_events_review(target, events_cfn_mtx)

    checks = self.catalog._log_review_checks.call_args.args[0]
    self.assertEqual(checks[OGS_C.BASE_STR]["check_sum"], 3)
    self.assertEqual(checks[OGS_C.TARGET_STR]["check_sum"], 4)
    self.assertEqual(checks[OGS_C.BASE_STR]["BPGMA"]["FILTERED"], 1)
    self.assertEqual(checks[OGS_C.TARGET_STR]["BPGMA"]["FILTERED"], 1)
    self.assertEqual(recall, 0.5)
    self.assertAlmostEqual(fdr, 2 / 3)
    self.assertEqual(
        [call.args[2] for call in self.catalog._write_csv.call_args_list],
        ["EventsMH", "EventsMS", "EventsSM", "EventsPS", "EventsSP"],
    )


class TestOGSBPGraphPicks(unittest.TestCase):
  def test_make_match_limits_candidates_by_station_and_time_window(self):
    base = pd.DataFrame({
        OGS_C.TIME_STR: [
            "2024-01-01T00:00:00",
            "2024-01-01T00:00:10",
        ],
        OGS_C.STATION_STR: ["AAA", "AAA"],
        OGS_C.PHASE_STR: [OGS_C.PWAVE, OGS_C.SWAVE],
    })
    target = pd.DataFrame({
        OGS_C.TIME_STR: [
            "2024-01-01T00:00:00.2",
            "2024-01-01T00:00:10.3",
            "2024-01-01T00:00:10.8",
            "2024-01-01T00:00:00.1",
        ],
        OGS_C.STATION_STR: ["AAA", "AAA", "AAA", "BBB"],
        OGS_C.PHASE_STR: [
            OGS_C.PWAVE,
            OGS_C.SWAVE,
            OGS_C.SWAVE,
            OGS_C.PWAVE,
        ],
        OGS_C.PROBABILITY_STR: [0.9, 0.8, 0.7, 0.9],
    })

    matcher = OGSBPGraphPicks(base, target, verbose=False)

    self.assertEqual(
        {tuple(edge) for edge in matcher.G.edges()}, {(0, 2), (1, 3)}
    )
    pairs = matcher.matched_pairs_array()
    pairs = pairs[np.argsort(pairs[:, 0])]
    np.testing.assert_array_equal(
        pairs,
        np.array([[0, 2], [1, 3]], dtype=np.int64),
    )


class TestOGSDistanceMetrics(unittest.TestCase):
  def test_dist_prob_target_over_base(self):
    base = pd.Series({OGS_C.PROBABILITY_STR: 1.0})
    target_high = pd.Series({OGS_C.PROBABILITY_STR: 0.9})
    target_low = pd.Series({OGS_C.PROBABILITY_STR: 0.3})

    score_high = dist_prob(base, target_high)
    score_low = dist_prob(base, target_low)

    self.assertAlmostEqual(score_high, 0.9)
    self.assertAlmostEqual(score_low, 0.3)
    self.assertGreater(score_high, score_low)

  def test_dist_prob_zero_division_guard(self):
    base_zero = pd.Series({OGS_C.PROBABILITY_STR: 0.0})
    target = pd.Series({OGS_C.PROBABILITY_STR: 0.5})

    score = dist_prob(base_zero, target)
    self.assertGreaterEqual(score, 0.0)
    self.assertLessEqual(score, 1.0)

  def test_dist_pick_higher_confidence_higher_score(self):
    t0 = UTCDateTime("2024-01-01T00:00:00")
    base = pd.Series({
        OGS_C.TIME_STR: t0,
        OGS_C.PHASE_STR: OGS_C.PWAVE,
        OGS_C.PROBABILITY_STR: 1.0,
    })
    target_high = pd.Series({
        OGS_C.TIME_STR: t0,
        OGS_C.PHASE_STR: OGS_C.PWAVE,
        OGS_C.PROBABILITY_STR: 0.95,
    })
    target_low = pd.Series({
        OGS_C.TIME_STR: t0,
        OGS_C.PHASE_STR: OGS_C.PWAVE,
        OGS_C.PROBABILITY_STR: 0.20,
    })

    score_high = dist_pick(base, target_high)
    score_low = dist_pick(base, target_low)

    self.assertGreater(score_high, score_low)


class TestOGSCatalogNormalizeCoordinates(unittest.TestCase):
  def test_normalize_coordinates(self):
    df = pd.DataFrame({
        OGS_C.LATITUDE_STR: [46.123456, "46-12.300", 95.0, -95.0, 90.0, -90.0],
        OGS_C.LONGITUDE_STR: [13.987654, "13-30.000", 185.0, -185.0, 180.0, -180.0],
        OGS_C.DEPTH_STR: ["12.5", "-----", "None", "", "  ", "10"],
        OGS_C.ERH_STR: ["0.5", "---", "None", "***", " ", "1.1"],
        OGS_C.ERZ_STR: ["1.2", "-----", "None", "   ", None, "2.2"],
        OGS_C.ERT_STR: ["0.1", "---", "None", "***", " ", "0.3"],
        OGS_C.RMS_STR: ["0.25", "---", "None", "***", " ", "0.4"],
        OGS_C.GAP_STR: ["120", "---", "None", "***", " ", "150"],
        OGS_C.DMIN_STR: ["5.4", "---", "None", "***", " ", "6.0"],
    })

    normalized = OGSDataFile.normalize_coordinates(df)

    # 1. 4-decimal rounding of decimal floats
    self.assertEqual(normalized.loc[0, OGS_C.LATITUDE_STR], 46.1235)
    self.assertEqual(normalized.loc[0, OGS_C.LONGITUDE_STR], 13.9877)

    # 2. Conversion of degree-minute string ("46-12.300" -> 46.205)
    assert normalized.loc[1, OGS_C.LATITUDE_STR] == pytest.approx(
        46.2050, abs=0.00005)
    assert normalized.loc[1, OGS_C.LONGITUDE_STR] == pytest.approx(
        13.5000, abs=0.00005)

    # 3. Physical bounds masking (lat > 90 -> NaN, lon < -180 -> NaN)
    # lat = 95.0 > 90
    self.assertTrue(pd.isna(normalized.loc[2, OGS_C.LATITUDE_STR]))
    # lon = 185.0 > 180
    self.assertTrue(pd.isna(normalized.loc[2, OGS_C.LONGITUDE_STR]))
    # lat = -95.0 < -90
    self.assertTrue(pd.isna(normalized.loc[3, OGS_C.LATITUDE_STR]))
    # lon = -185.0 < -180
    self.assertTrue(pd.isna(normalized.loc[3, OGS_C.LONGITUDE_STR]))

    # Valid boundary limits should be preserved
    self.assertEqual(normalized.loc[4, OGS_C.LATITUDE_STR], 90.0)
    self.assertEqual(normalized.loc[4, OGS_C.LONGITUDE_STR], 180.0)
    self.assertEqual(normalized.loc[5, OGS_C.LATITUDE_STR], -90.0)
    self.assertEqual(normalized.loc[5, OGS_C.LONGITUDE_STR], -180.0)

    # 4. Coercion of dash sequences and string "None" to NaN in depth and error columns
    # Valid values coerced to numeric
    self.assertEqual(normalized.loc[0, OGS_C.DEPTH_STR], 12.5)
    self.assertEqual(normalized.loc[0, OGS_C.ERH_STR], 0.5)
    self.assertEqual(normalized.loc[0, OGS_C.ERZ_STR], 1.2)
    self.assertEqual(normalized.loc[0, OGS_C.ERT_STR], 0.1)
    self.assertEqual(normalized.loc[0, OGS_C.RMS_STR], 0.25)
    self.assertEqual(normalized.loc[0, OGS_C.GAP_STR], 120.0)
    self.assertEqual(normalized.loc[0, OGS_C.DMIN_STR], 5.4)

    # Dash sequences and string "None" converted to NaN
    for col in [
        OGS_C.DEPTH_STR, OGS_C.ERH_STR, OGS_C.ERZ_STR,
        OGS_C.ERT_STR, OGS_C.RMS_STR, OGS_C.GAP_STR, OGS_C.DMIN_STR,
    ]:
      self.assertTrue(
          pd.isna(normalized.loc[1, col]), f"Failed NaN for dashes in {col}")
      self.assertTrue(
          pd.isna(normalized.loc[2, col]), f"Failed NaN for 'None' in {col}")

    # Empty DataFrame edge case
    empty_df = pd.DataFrame()
    self.assertTrue(OGSDataFile.normalize_coordinates(empty_df).empty)


DAY = date(2024, 1, 1)
ORIGIN = UTCDateTime("2024-01-01T00:00:00Z")
RECTANGLE = mplPath([(10, 40), (14, 40), (14, 42), (10, 42)])


@pytest.fixture
def make_catalog(tmp_path):
  """Real constructor and readers; only logger output is isolated."""
  sequence = 0

  def create(root=None, **options):
    nonlocal sequence
    sequence += 1
    if root is None:
      root = tmp_path / f"input-{sequence}"
      root.mkdir()
    kwargs: dict[str, Any] = dict(
        start=datetime(2024, 1, 1), end=datetime(2024, 1, 3),
        polygon=None, output=tmp_path / f"output-{sequence}",
    )
    kwargs.update(options)
    result = OGSCatalog(root, **kwargs)
    result.logger = Mock()
    return result

  return create


def write_day(root, category, day, frame):
  directory = root / category
  directory.mkdir(parents=True, exist_ok=True)
  path = directory / f"{day}.csv"
  frame.to_csv(path, index=False)
  return path


def event_frame(
    seconds, longitude: float | Sequence[float] = 12.0,
    latitude: float | Sequence[float] = 41.0,
):
  return pd.DataFrame({
      OGS_C.IDX_EVENTS_STR: np.arange(1, len(seconds) + 1),
      OGS_C.TIME_STR: [str(ORIGIN + offset) for offset in seconds],
      OGS_C.LONGITUDE_STR: longitude,
      OGS_C.LATITUDE_STR: latitude,
      OGS_C.DEPTH_STR: 1000.0,
      OGS_C.GROUPS_STR: DAY.isoformat(),
  })


def pick_frame(seconds, stations=None, phases=None, probabilities=None):
  frame = pd.DataFrame({
      OGS_C.TIME_STR: [str(ORIGIN + offset) for offset in seconds],
      OGS_C.STATION_STR: stations if stations is not None else "AAA",
      OGS_C.PHASE_STR: phases if phases is not None else OGS_C.PWAVE,
  })
  if probabilities is not None:
    frame[OGS_C.PROBABILITY_STR] = probabilities
  return frame


def oriented_pairs(matcher):
  return {tuple(pair) for pair in matcher.matched_pairs_array()}


def test_constructor_missing_input_has_no_output_side_effect(tmp_path):
  output = tmp_path / "must-not-exist"
  with pytest.raises(FileNotFoundError, match="does not exist"):
    OGSCatalog(tmp_path / "absent", output=output)
  assert not output.exists()


def test_constructor_empty_schemas_and_output_directories(make_catalog):
  catalog = make_catalog()
  assert catalog.name == catalog.output.name
  assert catalog.output.is_dir()
  assert (catalog.output / "img").is_dir()
  assert catalog.events_ == catalog.picks_ == {}
  assert catalog.events == catalog.picks == {}
  assert catalog.waveforms is None and catalog.stations is None
  assert catalog.EVENTS.empty and catalog.PICKS.empty
  assert list(catalog.PICKS) == OGSCatalog._PICK_COLUMNS
  assert list(catalog.EVENTS) == OGSCatalog._EVENT_COLUMNS


def test_indexing_is_lazy_date_inclusive_and_category_scoped(
    tmp_path, make_catalog, monkeypatch,
):
  root = tmp_path / "run"
  frame = event_frame([0])
  expected = {}
  for day in ("2023-12-31", "2024-01-01", "2024-01-03", "2024-01-04"):
    path = write_day(root, "events", day, frame)
    if day in ("2024-01-01", "2024-01-03"):
      expected[date.fromisoformat(day)] = path
  for stem in (".2024-01-02", "notes", "2024-99-99"):
    write_day(root, "events", stem, frame)
  write_day(root, "unrelated", "2024-01-02", frame)
  (root / "events" / "2024-01-02.csv").mkdir()
  reader = Mock(side_effect=AssertionError("Indexing must not read data"))
  monkeypatch.setattr(catalog_module.pd, "read_csv", reader)
  catalog = make_catalog(
      root, start=datetime(2024, 1, 1, 23, 59),
      end=datetime(2024, 1, 3, 0, 1), name="Reference",
  )
  assert catalog.events_ == expected
  assert catalog.picks_ == {}
  assert catalog.name == "Reference"
  assert catalog.events == {}
  reader.assert_not_called()


def test_picks_override_assignments_independent_of_creation_order(
    tmp_path, make_catalog,
):
  root = tmp_path / "run"
  pick_path = write_day(root, "picks", DAY, pd.DataFrame({"value": [7]}))
  write_day(root, "assignments", DAY, pd.DataFrame({"value": [3]}))
  assignments_only = write_day(
      root, "assignments", "2024-01-02", pd.DataFrame({"value": [5]}),
  )
  catalog = make_catalog(root)
  assert catalog.picks_ == {
      DAY: pick_path, date(2024, 1, 2): assignments_only,
  }
  assert catalog.load("picks")[DAY]["value"].tolist() == [7]


def test_preload_reuses_discovery_but_can_expand_date_window(
    tmp_path, make_catalog,
):
  root = tmp_path / "run"
  first = write_day(root, "events", DAY, event_frame([0]))
  second = write_day(root, "events", "2024-01-02", event_frame([10]))
  catalog = make_catalog(root, end=datetime(2024, 1, 1))
  assert catalog.events_ == {DAY: first}
  write_day(root, "events", "2024-01-03", event_frame([20]))
  catalog.end = datetime(2024, 1, 3)
  catalog.preload()
  # The API indexes discovered paths, not a filesystem-refresh operation.
  assert catalog.events_ == {DAY: first, date(2024, 1, 2): second}


def test_default_inverted_date_window_does_not_index(tmp_path):
  root = tmp_path / "run"
  write_day(root, "events", DAY, event_frame([0]))
  catalog = OGSCatalog(root, polygon=None, output=tmp_path / "output")
  assert catalog.events_ == {}


def test_csv_loading_preserves_source_schema_and_raw_values(
    tmp_path, make_catalog,
):
  root = tmp_path / "run"
  original = pd.DataFrame({
      OGS_C.TIME_STR: ["2024-01-01T00:00:00Z", "invalid"],
      OGS_C.LATITUDE_STR: ["46-12.300", "bad"],
      "source_only": [17, 23],
  })
  path = write_day(root, "events", DAY, original)
  catalog = make_catalog(root)
  pd.testing.assert_frame_equal(catalog.load_(path), original)
  pd.testing.assert_frame_equal(catalog.get("EVENTS"), original)
  assert catalog.PICKS.empty


@pytest.mark.parametrize("suffix", [".parquet", ".dat", ".CSV"])
def test_non_csv_suffix_dispatches_to_parquet(make_catalog, monkeypatch, suffix):
  catalog = make_catalog()
  path = catalog.input / ("2024-01-01" + suffix)
  expected = pd.DataFrame({"source_only": [11, 19]})
  reader = Mock(return_value=expected)
  monkeypatch.setattr(catalog_module.pd, "read_parquet", reader)
  assert catalog.load_(path) is expected
  reader.assert_called_once_with(path)


def test_real_parquet_daily_loading_preserves_numeric_and_datetime_dtypes(
    tmp_path, make_catalog,
):
  root = tmp_path / "run"
  directory = root / "events"
  directory.mkdir(parents=True)
  path = directory / "2024-01-01.parquet"
  original = pd.DataFrame({
      OGS_C.TIME_STR: pd.to_datetime(["2024-01-01T00:00:00Z"]),
      OGS_C.LATITUDE_STR: [41.25],
      OGS_C.LONGITUDE_STR: [12.5],
      "source_only": pd.Series([17], dtype="int32"),
  })
  original.to_parquet(path, index=False)
  catalog = make_catalog(root)
  assert catalog.events_ == {DAY: path}
  pd.testing.assert_frame_equal(catalog.get("EVENTS"), original)


@pytest.mark.parametrize("failure", ["missing", "empty", "corrupt_parquet"])
def test_read_failure_is_logged_and_returns_empty_frame(make_catalog, failure):
  catalog = make_catalog()
  path = catalog.input / "2024-01-01.csv"
  if failure == "empty":
    path.touch()
  elif failure == "corrupt_parquet":
    path = path.with_suffix(".parquet")
    path.write_bytes(b"not a parquet file")
  result = catalog.load_(path)
  assert result.empty and result.shape == (0, 0)
  catalog.logger.exception.assert_called_once()
  assert str(path) in catalog.logger.exception.call_args.args[0]


@pytest.mark.parametrize("key", ["events", "picks"])
def test_daily_and_aggregate_caches_avoid_rereads(
    tmp_path, make_catalog, monkeypatch, key,
):
  root = tmp_path / "run"
  original = pd.DataFrame({"source_only": [31, 41]})
  path = write_day(root, key, DAY, original)
  catalog = make_catalog(root)
  reader = Mock(wraps=catalog.load_)
  monkeypatch.setattr(catalog, "load_", reader)
  cache = catalog.load(key)
  assert cache is getattr(catalog, key)
  assert catalog.load(key) is cache
  assert catalog._load_day(key, DAY) is cache[DAY]
  first = catalog.get(key.upper())
  pd.testing.assert_frame_equal(first, original)
  path.write_text("source_only\n99\n")
  assert catalog.get(key.upper()) is first
  pd.testing.assert_frame_equal(cache[DAY], original)
  reader.assert_called_once_with(path)


@pytest.mark.parametrize("key", ["events", "picks"])
def test_aggregate_uses_union_schema_resets_index_without_order_assumption(
    tmp_path, make_catalog, key,
):
  root = tmp_path / "run"
  write_day(root, key, DAY, pd.DataFrame({"id": [4], "left": ["a"]}))
  write_day(root, key, "2024-01-02", pd.DataFrame({"id": [9], "right": ["b"]}))
  result = make_catalog(root).get(key.upper()).sort_values("id")
  expected = pd.DataFrame({
      "id": [4, 9], "left": ["a", np.nan], "right": [np.nan, "b"],
  })
  pd.testing.assert_frame_equal(
      result.reset_index(drop=True).reindex(
          columns=expected.columns), expected,
  )
  assert sorted(result.index) == [0, 1]


@pytest.mark.parametrize("key", ["events", "picks"])
def test_empty_days_are_cached_even_when_aggregate_is_empty(
    tmp_path, make_catalog, monkeypatch, key,
):
  root = tmp_path / "run"
  path = write_day(root, key, DAY, pd.DataFrame(columns=["source_only"]))
  catalog = make_catalog(root)
  reader = Mock(wraps=catalog.load_)
  monkeypatch.setattr(catalog, "load_", reader)
  for _ in range(2):
    assert catalog.get(key.upper()).empty
  assert list(catalog.get(key.upper())) == ["source_only"]
  assert DAY in getattr(catalog, key)
  reader.assert_called_once_with(path)


@pytest.mark.parametrize("key", ["EVENTS", "PICKS"])
def test_empty_catalog_get_keeps_initial_schema(make_catalog, key):
  catalog = make_catalog()
  original = getattr(catalog, key)
  assert catalog.get(key) is original
  assert catalog.get(key) is original
  assert original.empty


@pytest.mark.parametrize("operation,key", [
    ("load", "EVENTS"), ("load", "unknown"),
    ("get", "events"), ("get", "unknown"),
    ("postload", "PICKS"), ("postload", "unknown"),
])
def test_public_invalid_keys_fail_without_loading(make_catalog, operation, key):
  catalog = make_catalog()
  with pytest.raises(ValueError, match="Unknown key"):
    getattr(catalog, operation)(key)
  assert catalog.events == catalog.picks == {}


def test_daily_missing_date_and_invalid_key_are_distinct_errors(make_catalog):
  catalog = make_catalog()
  with pytest.raises(KeyError):
    catalog._load_day("events", DAY)
  with pytest.raises(ValueError, match="Unknown key"):
    catalog._load_day("EVENTS", DAY)


@pytest.mark.parametrize("key", ["events", "picks"])
def test_postload_groups_and_replaces_only_present_days(make_catalog, key):
  catalog = make_catalog()
  older = pd.DataFrame({"id": [-1]})
  stale = pd.DataFrame({"id": [-2]})
  cache = getattr(catalog, key)
  cache.update({date(2023, 12, 31): older, DAY: stale})
  frame = pd.DataFrame({
      OGS_C.GROUPS_STR: ["2024-01-02", "2024-01-01", "2024-01-02"],
      "id": [20, 10, 21],
  }, index=[7, 8, 9])
  setattr(catalog, key.upper(), frame)
  assert catalog.postload(key, update=False) is cache
  assert cache[DAY] is stale
  assert catalog.postload(key) is cache
  assert cache[date(2023, 12, 31)] is older
  pd.testing.assert_frame_equal(cache[DAY], frame.iloc[[1]])
  pd.testing.assert_frame_equal(cache[date(2024, 1, 2)], frame.iloc[[0, 2]])
  pd.testing.assert_frame_equal(getattr(catalog, key.upper()), frame)


def test_postload_requires_group_schema_and_parseable_dates(make_catalog):
  catalog = make_catalog()
  catalog.EVENTS = pd.DataFrame({"id": [1]})
  with pytest.raises(KeyError, match=OGS_C.GROUPS_STR):
    catalog.postload("events")
  catalog.EVENTS = pd.DataFrame({OGS_C.GROUPS_STR: ["not-a-date"]})
  with pytest.raises((ValueError, TypeError)):
    catalog.postload("events")
  assert catalog.events == {}


def test_loaded_event_polygon_is_longitude_latitude_and_picks_are_unfiltered(
    tmp_path, make_catalog,
):
  root = tmp_path / "run"
  frame = event_frame([0, 1, 2], longitude=[12, 15, 41], latitude=[41, 41, 12])
  write_day(root, "events", DAY, frame)
  write_day(root, "picks", DAY, frame)
  catalog = make_catalog(root, polygon=RECTANGLE)
  pd.testing.assert_frame_equal(catalog.load("events")[DAY], frame.iloc[[0]])
  pd.testing.assert_frame_equal(catalog.get("EVENTS"), frame.iloc[[0]])
  pd.testing.assert_frame_equal(catalog.get("PICKS"), frame)


def test_all_events_filtered_returns_cached_source_schema(
    tmp_path, make_catalog,
):
  root = tmp_path / "run"
  frame = event_frame([0], longitude=20)
  write_day(root, "events", DAY, frame)
  catalog = make_catalog(root, polygon=RECTANGLE)
  pd.testing.assert_frame_equal(catalog.get("EVENTS"), frame.iloc[0:0])
  assert catalog.events[DAY].empty
  catalog.logger.warning.assert_called_once()


def test_filter_requires_coordinate_columns_and_numpy_polygon_vertices(
    tmp_path, make_catalog,
):
  root = tmp_path / "run"
  write_day(root, "events", DAY, pd.DataFrame({"id": [1]}))
  catalog = make_catalog(root, polygon=RECTANGLE)
  with pytest.raises(KeyError):
    catalog.load("events")
  assert DAY not in catalog.events
  catalog.polygon = object()
  with pytest.raises(TypeError, match="NumPy vertices"):
    catalog._polygon_vertices()


@pytest.mark.parametrize("mode", ["none", "empty", "all_outside"])
def test_prefilter_partition_conservation_and_input_preservation(make_catalog, mode):
  catalog = make_catalog()
  original = event_frame([0, 1], longitude=[12, 20])
  original["source_only"] = ["keep", "drop"]
  original.index = [8, 13]
  if mode == "empty":
    original = original.iloc[0:0]
  elif mode == "all_outside":
    original[OGS_C.LONGITUDE_STR] = 20
  snapshot = original.copy(deep=True)
  excluded = []
  vertices = None if mode == "none" else RECTANGLE.vertices
  candidates = catalog._prefilter_events(
      original, vertices, excluded, DAY, OGS_C.BASE_STR,
  )
  pd.testing.assert_frame_equal(original, snapshot)
  assert len(candidates) + sum(len(part) for part in excluded) == len(original)
  if mode == "all_outside":
    assert candidates.empty and len(excluded) == 1
    pd.testing.assert_frame_equal(
        excluded[0], original.reindex(
            columns=_EVENTS_MH_COLUMNS).reset_index(drop=True),
    )
  else:
    pd.testing.assert_frame_equal(candidates, original.reset_index(drop=True))
    assert excluded == []


def test_event_feasibility_inclusive_time_horizontal_distance_and_positions(make_catalog):
  base = event_frame([0, 10])
  target = event_frame([2, -2, 2.00001, 10, 10],
                       longitude=[12, 12, 12, 20, 12])
  base.index = [31, 44]
  target.index = [11, 13, 15, 17, 19]
  target[OGS_C.DEPTH_STR] = 999000.0
  base_positions, target_positions = make_catalog(
  )._event_feasible_positions(base, target)
  np.testing.assert_array_equal(base_positions, [0, 1])
  np.testing.assert_array_equal(target_positions, [0, 1, 4])


@pytest.mark.parametrize("distance,eligible", [(7.9999, True), (8.0, True), (8.0001, False)])
def test_event_feasibility_and_graph_share_inclusive_distance_cutoff(
    make_catalog, monkeypatch, distance, eligible,
):
  # Isolate threshold accounting; actual geodetic units are checked separately.
  from OGS.src import ogsutils
  monkeypatch.setattr(ogsutils, "diff_space", Mock(return_value=distance))
  base, target = event_frame([0]), event_frame([0])
  positions = make_catalog()._event_feasible_positions(base, target)
  for selected in positions:
    assert selected.tolist() == ([0] if eligible else [])
  matcher = OGSBPGraphEvents(base, target, verbose=False)
  assert oriented_pairs(matcher) == ({(0, 1)} if eligible else set())


@pytest.mark.parametrize("empty_side", ["base", "target", "both"])
def test_empty_event_feasibility_does_not_require_schema(make_catalog, empty_side):
  base = pd.DataFrame() if empty_side in ("base", "both") else event_frame([0])
  target = pd.DataFrame() if empty_side in (
      "target", "both") else event_frame([0])
  positions = make_catalog()._event_feasible_positions(base, target)
  for result in positions:
    assert result.tolist() == [] and result.dtype.kind == "i"


@pytest.mark.parametrize("key", ["events", "picks"])
def test_shared_extra_dates_partition_each_day_once(make_catalog, key):
  base, target = make_catalog(), make_catalog()
  setattr(base, key + "_", {date(2024, 1, 2): Path("a"), DAY: Path("b")})
  setattr(target, key + "_", {DAY: Path("c"), date(2024, 1, 3): Path("d")})
  assert list(base._iter_shared_and_extra_dates(target, key)) == [
      (date(2024, 1, 2), "base_only"), (DAY, "both"),
      (date(2024, 1, 3), "target_only"),
  ]
  with pytest.raises(ValueError, match="Unknown key"):
    list(base._iter_shared_and_extra_dates(target, "EVENTS"))


def test_review_csv_roundtrip_has_no_index_and_uses_catalog_names(make_catalog):
  base, target = make_catalog(), make_catalog()
  frame = pd.DataFrame(
      {"id": [7, 9], "review": ["missed", "proposed"]}, index=[20, 30])
  base._write_csv(frame, target, "review")
  path = base.output / f"{base.input.name}_{target.input.name}_review.csv"
  pd.testing.assert_frame_equal(
      pd.read_csv(path), frame.reset_index(drop=True))
  assert list(target.output.glob("*.csv")) == []


@pytest.mark.parametrize("wide", [True, False])
def test_matched_residuals_are_signed_utc_seconds_and_invalid_times_are_nat(
    make_catalog, wide,
):
  catalog = make_catalog()
  base = [
      pd.Timestamp("2024-01-01T00:00:00"),
      pd.Timestamp("2024-01-01T01:00:00+01:00"), pd.NaT,
  ]
  target = [
      pd.Timestamp("2024-01-01T00:00:01Z"),
      pd.Timestamp("2023-12-31T23:59:58Z"), pd.NaT,
  ]
  if wide:
    catalog.EventsMH = pd.DataFrame({
        f"{OGS_C.TIME_STR}_base": base, f"{OGS_C.TIME_STR}_target": target,
        f"{OGS_C.DEPTH_STR}_base": [100, 300, 400],
        f"{OGS_C.DEPTH_STR}_target": [120, 250, 400],
    })
  else:
    catalog.EventsMH = pd.DataFrame({
        OGS_C.TIME_STR: list(zip(base, target)),
        OGS_C.DEPTH_STR: [(100, 120), (300, 250), (400, 400)],
    })
  np.testing.assert_allclose(catalog._mh_diff(OGS_C.TIME_STR), [1, -2, np.nan])
  np.testing.assert_array_equal(
      catalog._mh_diff(OGS_C.DEPTH_STR), [20, -50, 0])


def test_pick_inventory_cleaning_normalizes_fdsn_and_partitions_rows(make_catalog):
  catalog = make_catalog()
  picks = pick_frame([0, 1, 2], stations=[
                     "IV.AAA..HHZ", "BBB", "XX.CCC.00.HHN"])
  picks.index = [5, 7, 9]
  columns = [OGS_C.TIME_STR, OGS_C.STATION_STR, OGS_C.PHASE_STR]
  excluded = []
  result = catalog._clean_picks(
      picks, np.array(["AAA", "BBB"]), columns, excluded, DAY, OGS_C.BASE_STR,
  )
  assert result[OGS_C.STATION_STR].tolist() == ["AAA", "BBB"]
  assert result[OGS_C.NETWORK_STR].iloc[0] == "IV"
  assert pd.isna(result[OGS_C.NETWORK_STR].iloc[1])
  assert result.index.tolist() == [0, 1]
  assert excluded == [
      [str(ORIGIN + 2), "CCC", OGS_C.PWAVE],
  ]
  assert picks[OGS_C.STATION_STR].tolist() == ["AAA", "BBB", "CCC"]
  assert len(result) + len(excluded) == len(picks)
  catalog.logger.warning.assert_called_once()


@pytest.mark.parametrize("missing", [None, np.nan])
def test_missing_station_is_filtered_without_crashing_review_logging(make_catalog, missing):
  """Missing identifiers are outside inventory, not a logging exception."""
  catalog = make_catalog()
  picks = pick_frame([0, 1], stations=["IV.AAA..HHZ", missing])
  columns = [OGS_C.TIME_STR, OGS_C.STATION_STR, OGS_C.PHASE_STR]
  excluded = []
  result = catalog._clean_picks(
      picks, np.array(["AAA"]), columns, excluded, DAY, OGS_C.BASE_STR,
  )
  assert result[OGS_C.STATION_STR].tolist() == ["AAA"]
  assert len(excluded) == 1
  assert excluded[0][0] == str(ORIGIN + 1)
  assert pd.isna(excluded[0][1])
  assert excluded[0][2] == OGS_C.PWAVE
  catalog.logger.warning.assert_called_once()
  warning_args = catalog.logger.warning.call_args.args
  assert warning_args[3] == "nan"
  assert "STATION (nan) not in INVENTORY" in warning_args[0] % warning_args[1:]


@pytest.mark.parametrize("mode", ["empty", "missing_station", "empty_inventory"])
def test_pick_cleaning_empty_cases(make_catalog, mode):
  catalog = make_catalog()
  picks = pick_frame([0])
  if mode == "empty":
    picks = picks.iloc[0:0]
  elif mode == "missing_station":
    picks = picks.drop(columns=OGS_C.STATION_STR)
  excluded = []
  result = catalog._clean_picks(
      picks, np.array([], dtype=str), list(
          picks), excluded, DAY, OGS_C.TARGET_STR,
  )
  assert result.empty
  assert len(excluded) == (1 if mode == "empty_inventory" else 0)


def test_shared_day_pick_review_counts_match_swap_miss_proposal_and_filtered(
    make_catalog,
):
  base, target = make_catalog(), make_catalog()
  base.picks[DAY] = pick_frame(
      [0, 10, 20, 30], stations=["AAA", "BBB", "CCC", "OUT"],
      phases=[OGS_C.PWAVE, OGS_C.PWAVE, OGS_C.SWAVE, OGS_C.PWAVE],
      probabilities=[1.0] * 4,
  ).assign(**{OGS_C.IDX_PICKS_STR: [101, 102, 103, 104]})
  target.picks[DAY] = pick_frame(
      [0.1, 10.1, 40, 50], stations=["AAA", "BBB", "DDD", "OUT"],
      phases=[OGS_C.PWAVE, OGS_C.SWAVE, OGS_C.PWAVE, OGS_C.SWAVE],
      probabilities=[0.8, 0.7, 0.6, 0.5],
  ).assign(**{OGS_C.IDX_PICKS_STR: [201, 202, 203, 204]})
  for bucket in ("PicksMH", "PicksSW", "PicksMS", "PicksPS", "PicksSM", "PicksSP"):
    setattr(base, bucket, [])
  columns = [
      OGS_C.IDX_PICKS_STR, OGS_C.TIME_STR, OGS_C.PHASE_STR,
      OGS_C.STATION_STR, OGS_C.PROBABILITY_STR,
  ]
  phases = (OGS_C.PWAVE, OGS_C.SWAVE, OGS_C.NONE_STR)
  matrix = base._empty_cfn_mtx(phases)
  base._BPGMA_picks_both(target, DAY, np.array(
      ["AAA", "BBB", "CCC", "DDD"]), columns, matrix)
  np.testing.assert_array_equal(matrix.to_numpy(), [
      [1, 1, 0], [0, 0, 1], [1, 0, 0],
  ])
  assert base.PicksMH == [[
      (101, 201), (str(ORIGIN), str(ORIGIN + 0.1)), OGS_C.PWAVE, "AAA", (1.0, 0.8),
  ]]
  assert base.PicksSW == [[
      (102, 202), (str(ORIGIN + 10), str(ORIGIN + 10.1)),
      (OGS_C.PWAVE, OGS_C.SWAVE), "BBB", (1.0, 0.7),
  ]]
  assert [row[0] for row in base.PicksMS] == [103]
  assert [row[0] for row in base.PicksPS] == [203]
  assert [row[0] for row in base.PicksSM] == [104]
  assert [row[0] for row in base.PicksSP] == [204]
  assert len(base.PicksMH) + len(base.PicksSW) + \
      len(base.PicksMS) + len(base.PicksSM) == 4
  assert len(base.PicksMH) + len(base.PicksSW) + \
      len(base.PicksPS) + len(base.PicksSP) == 4


def test_shared_day_event_review_maps_pruned_positions_back_to_original_ids(make_catalog):
  base, target = make_catalog(
      polygon=RECTANGLE), make_catalog(polygon=RECTANGLE)
  # Cache represents already-loaded data; opposite catalog domain still filters.
  base.events[DAY] = event_frame([100, 0, 200], longitude=[12, 12, 20])
  target.events[DAY] = event_frame([300, 0.5, 400], longitude=[12, 12, 20])
  base.events[DAY][OGS_C.IDX_EVENTS_STR] = [101, 102, 103]
  target.events[DAY][OGS_C.IDX_EVENTS_STR] = [201, 202, 203]
  matrix = base._empty_cfn_mtx(_EVENTS_PHASES)
  matched, missed, filtered_base, proposed, filtered_target = [], [], [], [], []
  base._BPGMA_events_both(
      target, DAY, matrix, matched, missed, filtered_base, proposed, filtered_target,
  )
  np.testing.assert_array_equal(matrix.to_numpy(), [[1, 1], [1, 0]])
  assert len(matched) == 1
  assert matched[0][f"{OGS_C.IDX_EVENTS_STR}_base"].tolist() == [102]
  assert matched[0][f"{OGS_C.IDX_EVENTS_STR}_target"].tolist() == [202]
  assert matched[0][f"{OGS_C.TIME_STR}_base"].tolist() == [str(ORIGIN)]
  assert matched[0][f"{OGS_C.TIME_STR}_target"].tolist() == [str(ORIGIN + 0.5)]
  for frames, expected_id in (
      (missed, 101), (filtered_base, 103), (proposed, 201), (filtered_target, 203),
  ):
    result = pd.concat(frames)
    assert result[OGS_C.IDX_EVENTS_STR].tolist() == [expected_id]
    assert list(result) == _EVENTS_MH_COLUMNS


@pytest.mark.parametrize("base,target,expected", [
    (1.0, 0.3, 0.3), (0.4, 0.2, 0.5), (0.2, 0.9, 1.0),
    (0.0, 0.5, 1.0), (0.0, 0.0, 0.0), (1.0, -0.5, 0.0),
    (None, 0.25, 0.25), (1.0, None, 1.0), (None, None, 1.0),
])
def test_probability_score_exact_ratio_clipping_and_null_defaults(base, target, expected):
  assert dist_prob(
      pd.Series({OGS_C.PROBABILITY_STR: base}),
      pd.Series({OGS_C.PROBABILITY_STR: target}),
  ) == pytest.approx(expected)


def test_probability_absent_columns_default_to_one():
  assert dist_prob(pd.Series(dtype=float), pd.Series(dtype=float)) == 1.0


@pytest.mark.parametrize("seconds,phase,probability,expected", [
    (0, OGS_C.PWAVE, 1.0, 1.0),
    (0.25, OGS_C.PWAVE, 0.5, 0.51),
    (-0.25, OGS_C.SWAVE, 0.5, 0.49),
    (0.5, OGS_C.PWAVE, 1.0, 0.03),
])
def test_pick_score_hand_computed_weighted_components(seconds, phase, probability, expected):
  base = pd.Series({
      OGS_C.TIME_STR: ORIGIN, OGS_C.PHASE_STR: OGS_C.PWAVE,
      OGS_C.PROBABILITY_STR: 1.0,
  })
  target = pd.Series({
      OGS_C.TIME_STR: ORIGIN + seconds, OGS_C.PHASE_STR: phase,
      OGS_C.PROBABILITY_STR: probability,
  })
  assert dist_pick(base, target) == pytest.approx(expected)


@pytest.mark.parametrize("seconds,expected", [(0, 1), (0.25, 0.5), (-0.5, 0), (1, -1)])
def test_time_similarity_is_seconds_symmetric_and_not_clipped(seconds, expected):
  base = pd.Series({OGS_C.TIME_STR: ORIGIN})
  target = pd.Series({OGS_C.TIME_STR: ORIGIN + seconds})
  assert dist_time(base, target) == pytest.approx(expected)
  assert dist_time(target, base) == pytest.approx(expected)


def test_spatial_units_horizontal_vs_vertical_and_event_score():
  base = event_frame([0]).iloc[0].copy()
  target = event_frame([1]).iloc[0].copy()
  base[OGS_C.TIME_STR], target[OGS_C.TIME_STR] = ORIGIN, ORIGIN + 1
  target[OGS_C.DEPTH_STR] = 4000.0
  assert diff_space(base, target) == 0.0
  assert diff_space(base, target, ndim=3) == 3.0
  assert diff_space(target, base, ndim=3) == 3.0
  assert dist_event(base, target) == pytest.approx(0.505)
  assert dist_event(base, target, time_offset_sec=timedelta(
      seconds=4)) == pytest.approx(0.7525)
  # WGS84 equatorial arc: pi * 6378137 / 180 metres per degree.
  equator_a = pd.Series({OGS_C.LATITUDE_STR: 0, OGS_C.LONGITUDE_STR: 0})
  equator_b = pd.Series({OGS_C.LATITUDE_STR: 0, OGS_C.LONGITUDE_STR: 1})
  assert diff_space(equator_a, equator_b) == pytest.approx(
      111.3195, abs=0.00005)


def test_pick_graph_unsorted_candidates_inclusive_window_and_phase_swaps():
  base = pick_frame([0], stations=["AAA"])
  target = pick_frame(
      [0.50001, 0.5, -0.5, 0.1, 0.0],
      stations=["AAA", "AAA", "AAA", "BBB", "AAA"],
      phases=[OGS_C.PWAVE] * 4 + [OGS_C.SWAVE],
  )
  base.index, target.index = [100], [8, 9, 10, 11, 12]
  matcher = OGSBPGraphPicks(base, target, verbose=False)
  assert {tuple(sorted(edge))
          for edge in matcher.G.edges()} == {(0, 2), (0, 3), (0, 5)}
  assert matcher.G[0][2]["weight"] == pytest.approx(0.03)
  assert matcher.G[0][3]["weight"] == pytest.approx(0.03)
  assert matcher.G[0][5]["weight"] == pytest.approx(0.98)
  assert oriented_pairs(matcher) == {(0, 5)}
  assert base[OGS_C.PROBABILITY_STR].tolist() == [1.0]
  assert isinstance(base[OGS_C.TIME_STR].iloc[0], UTCDateTime)


def test_pick_graph_global_assignment_beats_greedy_nearest_first():
  base = pick_frame([0, 0.3])
  target = pick_frame([0.2, -0.4])
  matcher = OGSBPGraphPicks(base, target, verbose=False)
  # Edges: .612, .224, .806. Greedy first .612 loses the .806 option;
  # the two-edge optimum is .224 + .806 = 1.030.
  assert matcher.G[0][2]["weight"] == pytest.approx(0.612)
  assert matcher.G[0][3]["weight"] == pytest.approx(0.224)
  assert matcher.G[1][2]["weight"] == pytest.approx(0.806)
  assert oriented_pairs(matcher) == {(0, 3), (1, 2)}


def test_pick_graph_maximum_weight_is_not_maximum_cardinality():
  matcher = OGSBPGraphPicks(pick_frame(
      [0, 0.49]), pick_frame([0, -0.49]), verbose=False)
  assert len(matcher.G.edges) == 3
  assert matcher.G[0][2]["weight"] == pytest.approx(1.0)
  assert matcher.G[0][3]["weight"] == pytest.approx(0.0494)
  assert matcher.G[1][2]["weight"] == pytest.approx(0.0494)
  assert oriented_pairs(matcher) == {(0, 2)}


def test_event_graph_inclusive_time_and_horizontal_only_eligibility():
  base = event_frame([0, 10])
  target = event_frame([2, -2, 2.00001, 10], longitude=[12, 12, 12, 20])
  target[OGS_C.DEPTH_STR] = 900000
  matcher = OGSBPGraphEvents(base, target, verbose=False)
  assert {tuple(sorted(edge))
          for edge in matcher.G.edges()} == {(0, 2), (0, 3)}
  assert matcher.G[0][2]["weight"] == pytest.approx(0.01)
  assert matcher.G[0][3]["weight"] == pytest.approx(0.01)
  pairs = oriented_pairs(matcher)
  assert len(pairs) == 1
  # Equal optima have no promised tie ordering.
  assert pairs <= {(0, 2), (0, 3)}


def test_event_graph_global_assignment_has_hand_computed_optimum():
  matcher = OGSBPGraphEvents(event_frame(
      [0, 1.2]), event_frame([0.8, -1.6]), verbose=False)
  assert matcher.G[0][2]["weight"] == pytest.approx(0.604)
  assert matcher.G[0][3]["weight"] == pytest.approx(0.208)
  assert matcher.G[1][2]["weight"] == pytest.approx(0.802)
  assert oriented_pairs(matcher) == {(0, 3), (1, 2)}


@pytest.mark.parametrize("matcher_type", [OGSBPGraphPicks, OGSBPGraphEvents])
def test_graph_invalid_time_is_not_silently_dropped(matcher_type):
  factory = pick_frame if matcher_type is OGSBPGraphPicks else event_frame
  base, target = factory([0]), factory([0])
  target[OGS_C.TIME_STR] = "not-a-time"
  with pytest.raises((ValueError, TypeError)):
    matcher_type(base, target, verbose=False)


@pytest.mark.parametrize("matcher_type,column", [
    (OGSBPGraphPicks, OGS_C.STATION_STR), (OGSBPGraphPicks, OGS_C.PHASE_STR),
    (OGSBPGraphEvents, OGS_C.LATITUDE_STR),
])
def test_graph_required_schema_failure_is_explicit(matcher_type, column):
  factory = pick_frame if matcher_type is OGSBPGraphPicks else event_frame
  base, target = factory([0]), factory([0]).drop(columns=column)
  with pytest.raises(KeyError, match=column):
    matcher_type(base, target, verbose=False)


@pytest.mark.parametrize("matcher_type", [OGSBPGraphPicks, OGSBPGraphEvents])
@pytest.mark.parametrize("side", ["base", "target", "both"])
def test_graph_empty_sides_have_oriented_empty_pairs(matcher_type, side):
  factory = pick_frame if matcher_type is OGSBPGraphPicks else event_frame
  base = factory([] if side in ("base", "both") else [0])
  target = factory([] if side in ("target", "both") else [0])
  matcher = matcher_type(base, target, verbose=False)
  assert matcher.E == set() and len(matcher.G.edges) == 0
  pairs = matcher.matched_pairs_array()
  assert pairs.shape == (0, 2) and pairs.dtype == np.int64


def test_pair_orientation_does_not_mutate_matching_and_rejects_invalid_edges():
  class FixedMatchingGraph(OGSBPGraph):
    def makeMatch(self):
      self.E = {(2, 0), (1, 3)}

  matcher = FixedMatchingGraph(
      pd.DataFrame({"id": [1, 2]}), pd.DataFrame({"id": [3, 4]}),
      verbose=False,
  )
  assert oriented_pairs(matcher) == {(0, 2), (1, 3)}
  assert matcher.E == {(2, 0), (1, 3)}
  matcher.E = {(0, 1)}
  with pytest.raises(ValueError, match="Unexpected matching edge"):
    matcher.matched_pairs_array()


@pytest.mark.parametrize("side", ["base", "target", "both", "neither"])
def test_graph_abstract_contract_rejects_incomplete_classes_before_initialization(
    side, monkeypatch,
):
  class IncompleteGraph(OGSBPGraph):
    pass

  initialize = Mock()
  monkeypatch.setattr(OGSBPGraph, "__init__", initialize)
  base = pd.DataFrame() if side in (
      "base", "both") else pd.DataFrame({"id": [1]})
  target = pd.DataFrame() if side in (
      "target", "both") else pd.DataFrame({"id": [2]})
  for matcher_type in (OGSBPGraph, IncompleteGraph):
    with pytest.raises(TypeError, match="abstract.*makeMatch"):
      matcher_type(base, target)
  initialize.assert_not_called()


def test_concrete_graph_hook_runs_only_for_two_nonempty_inputs():
  class RecordingGraph(OGSBPGraph):
    def __init__(self, base, target):
      self.match_calls = 0
      super().__init__(base, target, verbose=False)

    def makeMatch(self):
      self.match_calls += 1

  for base_size, target_size in ((0, 0), (1, 0), (0, 1), (1, 1)):
    matcher = RecordingGraph(
        pd.DataFrame({"id": range(base_size)}),
        pd.DataFrame({"id": range(target_size)}),
    )
    assert matcher.match_calls == int(base_size > 0 and target_size > 0)
    assert matcher.E == set()
    pairs = matcher.matched_pairs_array()
    assert pairs.shape == (0, 2) and pairs.dtype == np.int64


def test_owner_coordinate_normalization_signed_minutes_custom_columns_and_rounding():
  frame = pd.DataFrame({
      "lat": ["-46-30.000", " 12.34567 ", "bad", 90.00001],
      "lon": ["-13-15.000", 180, "-180", -180.00001],
      "z": ["-2.5", "0", "None", "-----"],
      "error": ["0.125", "***", " ", None],
      "untouched": ["raw"] * 4,
  }, index=[3, 5, 7, 9])
  result = OGSDataFile.normalize_coordinates(
      frame, lat_col="lat", lon_col="lon", depth_col="z",
      error_cols=("error",), round_decimals=2,
  )
  assert result is frame
  np.testing.assert_allclose(result["lat"], [-46.5, 12.35, np.nan, np.nan])
  np.testing.assert_allclose(result["lon"], [-13.25, 180, -180, np.nan])
  np.testing.assert_allclose(result["z"], [-2.5, 0, np.nan, np.nan])
  np.testing.assert_allclose(result["error"], [0.125, np.nan, np.nan, np.nan])
  assert result.index.tolist() == [3, 5, 7, 9]
  assert result["untouched"].tolist() == ["raw"] * 4


def test_owner_coordinate_normalization_missing_columns_is_noop():
  frame = pd.DataFrame({"source_only": ["untouched"]}, index=[13])
  snapshot = frame.copy(deep=True)
  assert OGSDataFile.normalize_coordinates(frame) is frame
  pd.testing.assert_frame_equal(frame, snapshot)

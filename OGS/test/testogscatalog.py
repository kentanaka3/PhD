from ogsutils import OGSBPGraphPicks, dist_prob, dist_pick, dist_event
from ogscatalog import OGSCatalog, _EVENTS_MH_COLUMNS, _EVENTS_PHASES
import ogsconstants as OGS_C
import os
import sys
import unittest
import unittest.mock

import numpy as np
import pandas as pd
from matplotlib.path import Path as mplPath

THIS_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(THIS_DIR + "/../src"))


class TestOGSCatalogEventPrefilter(unittest.TestCase):
  def setUp(self) -> None:
    self.catalog = OGSCatalog.__new__(OGSCatalog)
    self.catalog.logger = unittest.mock.MagicMock()

  def _event_frame(self) -> pd.DataFrame:
    return pd.DataFrame({
        OGS_C.INDEX_STR: [1, 2, 3],
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

    mask = self.catalog._event_candidate_mask(events, polygon.vertices)

    np.testing.assert_array_equal(mask, np.array([True, False, True]))

  def test_prefilter_events_tracks_filtered_rows_outside_bgma(self):
    events = self._event_frame()
    polygon = mplPath([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
    skimmed_frames: list[pd.DataFrame] = []

    candidates = self.catalog._prefilter_events(
        events,
        polygon.vertices,
        skimmed_frames,
        pd.Timestamp("2024-01-01"),
        "TARGET",
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

  def test_bgma_events_review_includes_prefiltered_partitions(self):
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

    self.catalog._bgma_events_review(target, events_cfn_mtx)

    checks = self.catalog._log_review_checks.call_args.args[0]
    self.assertEqual(checks[" BASE "]["check_sum"], 3)
    self.assertEqual(checks["TARGET"]["check_sum"], 4)
    self.assertEqual(checks[" BASE "]["bgma"]["FILTERED"], 1)
    self.assertEqual(checks["TARGET"]["bgma"]["FILTERED"], 1)


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
    from obspy import UTCDateTime
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


if __name__ == "__main__":
  unittest.main()

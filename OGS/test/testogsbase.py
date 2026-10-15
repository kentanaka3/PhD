"""
=============================================================================
OGS Base Test Suite - Unit Tests for State Registry, Caching, and Merger
=============================================================================

OVERVIEW:
Unit tests verifying foundational execution primitives in ogsbase.py:
  1. OGSCacheHelper group-level and global path resolution.
  2. OGSStatus and OGSGroupStatus parameter lifecycle, caching, and serialization.
  3. OGSMerge polymorphic catalog unification:
     - Standalone picks with index normalization.
     - Standalone events with idx normalization and time sorting.
     - Pick-to-event association mapping event_idx across groups.

AUTHORS:
  - 健
  - Istituto Nazionale di Oceanografia e di Geofisica Sperimentale (OGS)
=============================================================================
"""

import tempfile
import unittest
from pathlib import Path
import dask
import pandas as pd
import numpy as np

from OGS.src import ogsconstants as OGS_C
from OGS.src.ogsbase import (
    OGSCacheHelper,
    OGSStatus,
    OGSGroupStatus,
    OGSModule,
    OGSMerge,
    _merge_catalog_task,
)


class TestOGSCacheHelper(unittest.TestCase):
  """Tests for OGSCacheHelper directory layout and path resolution."""

  def test_group_level_path(self) -> None:
    helper = OGSCacheHelper(output_path="/tmp/catalog", group="2024-01-01")
    resolved = helper.resolve_path(param="picks", caller="SeisBenchPicker")
    expected = Path("/tmp/catalog/SeisBenchPicker/picks/2024-01-01.parquet")
    self.assertEqual(resolved, expected)

  def test_global_path_drop_group(self) -> None:
    helper = OGSCacheHelper(output_path="/tmp/catalog", group="2024-01-01")
    resolved = helper.resolve_path(param="picks", drop_group=True)
    expected = Path("/tmp/catalog/picks.parquet")
    self.assertEqual(resolved, expected)

  def test_global_path_no_group(self) -> None:
    helper = OGSCacheHelper(output_path="/tmp/catalog")
    resolved = helper.resolve_path(param="events")
    expected = Path("/tmp/catalog/events.parquet")
    self.assertEqual(resolved, expected)

  def test_get_cache_path_context_manager(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      helper = OGSCacheHelper(output_path=tmpdir, group="day1")
      with helper.get_cache_path(
          param="picks", caller="picker", mode="w"
      ) as cache_file:
        self.assertTrue(cache_file.parent.is_dir())
        cache_file.write_text("dummy")
      self.assertTrue(cache_file.is_file())


class TestOGSStatusAndGroupStatus(unittest.TestCase):
  """Tests for OGSStatus and OGSGroupStatus parameter lifecycle and caching."""

  def test_register_parameter(self) -> None:
    status = OGSStatus("/tmp/test")
    status.register_parameter("param1")
    self.assertIn("param1", status._parameters)

    # Duplicate without duplicate=True raises KeyError
    with self.assertRaises(KeyError):
      status.register_parameter("param1", duplicate=False)

    # Duplicate with duplicate=True succeeds
    status.register_parameter("param1", duplicate=True)

  def test_set_and_get_in_memory_param(self) -> None:
    status = OGSStatus("/tmp/test")
    status.set_param(12345, "scalar_param")
    self.assertEqual(status.get_param("scalar_param"), 12345)

  def test_set_and_get_dataframe_cached(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      group_status = OGSGroupStatus(group="2024-01-01", output_path=tmpdir)
      df = pd.DataFrame({
          OGS_C.TIME_STR: ["2024-01-01T00:00:00"],
          OGS_C.STATION_STR: ["STA1"],
      })
      group_status.set_param(df, "picks", caller="Picker")

      # Verify cache file exists
      self.assertTrue(group_status.param_is_cached("picks", caller="Picker"))

      # Retrieve parameter
      loaded_df = group_status.get_param("picks")
      self.assertEqual(len(loaded_df), 1)
      self.assertEqual(loaded_df[OGS_C.STATION_STR].iloc[0], "STA1")

  def test_status_update(self) -> None:
    status1 = OGSStatus("/tmp/test1")
    status2 = OGSStatus("/tmp/test2")
    status2.set_param("val2", "param2")
    status1.update(status2)
    self.assertEqual(status1.get_param("param2"), "val2")


class TestOGSMerge(unittest.TestCase):
  """Tests for OGSMerge catalog concatenation, reindexing, and association."""

  def test_standalone_picks_normalization(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      status = OGSStatus(tmpdir)
      gs1 = OGSGroupStatus(group="2024-01-01", output_path=tmpdir)
      gs2 = OGSGroupStatus(group="2024-01-02", output_path=tmpdir)

      picks1 = pd.DataFrame({
          OGS_C.IDX_PICKS_STR: [0, 1],
          OGS_C.TIME_STR: ["2024-01-01T01:00:00", "2024-01-01T02:00:00"],
          OGS_C.STATION_STR: ["STA1", "STA2"],
          OGS_C.PHASE_STR: ["P", "S"],
          OGS_C.GROUPS_STR: ["2024-01-01", "2024-01-01"],
      })
      picks2 = pd.DataFrame({
          OGS_C.IDX_PICKS_STR: [0, 1, 2],
          OGS_C.TIME_STR: [
              "2024-01-02T01:00:00",
              "2024-01-02T02:00:00",
              "2024-01-02T03:00:00",
          ],
          OGS_C.STATION_STR: ["STA3", "STA4", "STA5"],
          OGS_C.PHASE_STR: ["P", "P", "S"],
          OGS_C.GROUPS_STR: ["2024-01-02", "2024-01-02", "2024-01-02"],
      })

      gs1.set_param(picks1, "picks", caller="Picker")
      gs2.set_param(picks2, "picks", caller="Picker")

      merger = OGSMerge()
      merger.run(status, [gs1, gs2])

      merged_picks_delayed = status.get_param("picks")
      merged_picks = dask.compute(merged_picks_delayed)[0]

      self.assertEqual(len(merged_picks), 5)
      # Normalized continuous index [0, 1, 2, 3, 4]
      self.assertEqual(list(merged_picks[OGS_C.IDX_PICKS_STR]), [0, 1, 2, 3, 4])
      # Preserved original index in idx_in_group
      self.assertEqual(list(merged_picks["idx_in_group"]), [0, 1, 0, 1, 2])

  def test_standalone_events_normalization(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      status = OGSStatus(tmpdir)
      gs1 = OGSGroupStatus(group="2024-01-01", output_path=tmpdir)
      gs2 = OGSGroupStatus(group="2024-01-02", output_path=tmpdir)

      # Day 1 has an event at 12:00:00; Day 2 has an event at 08:00:00
      events1 = pd.DataFrame({
          OGS_C.IDX_EVENTS_STR: [0],
          OGS_C.TIME_STR: ["2024-01-01T12:00:00"],
          OGS_C.LATITUDE_STR: [46.0],
          OGS_C.LONGITUDE_STR: [13.0],
          OGS_C.DEPTH_STR: [10.0],
          OGS_C.GROUPS_STR: ["2024-01-01"],
      })
      events2 = pd.DataFrame({
          OGS_C.IDX_EVENTS_STR: [0],
          OGS_C.TIME_STR: ["2024-01-02T08:00:00"],
          OGS_C.LATITUDE_STR: [46.1],
          OGS_C.LONGITUDE_STR: [13.1],
          OGS_C.DEPTH_STR: [12.0],
          OGS_C.GROUPS_STR: ["2024-01-02"],
      })

      gs1.set_param(events1, "events", caller="Locator")
      gs2.set_param(events2, "events", caller="Locator")

      merger = OGSMerge()
      merger.run(status, [gs1, gs2])

      merged_events_delayed = status.get_param("events")
      merged_events = dask.compute(merged_events_delayed)[0]

      self.assertEqual(len(merged_events), 2)
      self.assertEqual(list(merged_events[OGS_C.IDX_EVENTS_STR]), [0, 1])
      # Events sorted chronologically
      self.assertEqual(
          merged_events[OGS_C.TIME_STR].iloc[0], "2024-01-01T12:00:00"
      )
      self.assertEqual(
          merged_events[OGS_C.TIME_STR].iloc[1], "2024-01-02T08:00:00"
      )

  def test_combined_pick_to_event_association(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      status = OGSStatus(tmpdir)
      gs1 = OGSGroupStatus(group="2024-01-01", output_path=tmpdir)
      gs2 = OGSGroupStatus(group="2024-01-02", output_path=tmpdir)

      # Group 1: 1 event (idx=0), 2 picks associated with event_idx=0
      events1 = pd.DataFrame({
          OGS_C.IDX_EVENTS_STR: [0],
          OGS_C.TIME_STR: ["2024-01-01T12:00:00"],
          OGS_C.LATITUDE_STR: [46.0],
          OGS_C.LONGITUDE_STR: [13.0],
          OGS_C.DEPTH_STR: [10.0],
          OGS_C.GROUPS_STR: ["2024-01-01"],
      })
      picks1 = pd.DataFrame({
          OGS_C.IDX_PICKS_STR: [0, 1],
          OGS_C.TIME_STR: ["2024-01-01T12:00:01", "2024-01-01T12:00:02"],
          OGS_C.STATION_STR: ["STA1", "STA2"],
          OGS_C.PHASE_STR: ["P", "S"],
          "event_idx": [0, 0],
          OGS_C.GROUPS_STR: ["2024-01-01", "2024-01-01"],
      })

      # Group 2: 1 event (idx=0), 2 picks associated with event_idx=0
      events2 = pd.DataFrame({
          OGS_C.IDX_EVENTS_STR: [0],
          OGS_C.TIME_STR: ["2024-01-02T14:00:00"],
          OGS_C.LATITUDE_STR: [46.2],
          OGS_C.LONGITUDE_STR: [13.2],
          OGS_C.DEPTH_STR: [15.0],
          OGS_C.GROUPS_STR: ["2024-01-02"],
      })
      picks2 = pd.DataFrame({
          OGS_C.IDX_PICKS_STR: [0, 1],
          OGS_C.TIME_STR: ["2024-01-02T14:00:01", "2024-01-02T14:00:02"],
          OGS_C.STATION_STR: ["STA3", "STA4"],
          OGS_C.PHASE_STR: ["P", "S"],
          "event_idx": [0, 0],
          OGS_C.GROUPS_STR: ["2024-01-02", "2024-01-02"],
      })

      gs1.set_param(events1, "events")
      gs1.set_param(picks1, "picks")
      gs2.set_param(events2, "events")
      gs2.set_param(picks2, "picks")

      merger = OGSMerge()
      merger.run(status, [gs1, gs2])

      ev_delayed, pk_delayed = status.get_param("events"), status.get_param("picks")
      merged_events, merged_picks = dask.compute(ev_delayed, pk_delayed)

      self.assertEqual(len(merged_events), 2)
      self.assertEqual(len(merged_picks), 4)

      # Check event indices are unified 0 and 1
      self.assertEqual(list(merged_events[OGS_C.IDX_EVENTS_STR]), [0, 1])

      # Group 1 picks must now point to event 0
      pk_g1 = merged_picks[merged_picks[OGS_C.GROUPS_STR] == "2024-01-01"]
      self.assertTrue((pk_g1["event_idx"] == 0).all())

      # Group 2 picks must now point to event 1
      pk_g2 = merged_picks[merged_picks[OGS_C.GROUPS_STR] == "2024-01-02"]
      self.assertTrue((pk_g2["event_idx"] == 1).all())


if __name__ == "__main__":
  unittest.main()


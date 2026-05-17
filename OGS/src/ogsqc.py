import dask
import pandas as pd

from typing import Optional
from ml_catalog.base import Status
from ml_catalog.modules import PickStatQC
from matplotlib.path import Path as mplPath

OGS_STUDY_REGION = [
  (9.5, 47.5),
  (15.0, 47.5),
  (15.0, 44.3),
  (9.5, 44.3),
  (9.5, 47.5)
]

class OGSPickStatQC(PickStatQC):
  def __init__(
    self,
    p_picks: Optional[int] = None,
    s_picks: Optional[int] = None,
    total_picks: Optional[int] = None,
    p_and_s_picks: Optional[int] = None,
    region: Optional[mplPath] = mplPath(OGS_STUDY_REGION, closed=True),
  ):
    super().__init__(
      p_picks=p_picks,
      s_picks=s_picks,
      total_picks=total_picks,
      p_and_s_picks=p_and_s_picks,
    )
    self.region = region

  def _filter_events(
      self, events: pd.DataFrame, assignments: pd.DataFrame
  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    events, assignments = super()._filter_events(events, assignments)
    # Apply region filter
    events = events[events[
      ["longitude", "latitude"]].apply(
        lambda x: self.region.contains_point(
          (x["longitude"], x["latitude"])), axis=1)
    ]
    assignments = assignments[
      assignments["event_idx"].isin(events.index)
    ].copy()
    return events, assignments

class EventStatQC(OGSPickStatQC):
  """
  A quality control module based on event statistics.
  For each of the parameters evaluated (see below), only events with at least
  that many picks will be retained. In addition to performing quality control,
  this module writes statistics on the picks per event to the event dataframe.
  Therefore, it's often convenient to include it even without using it to
  filter events. This module is intended to be used after the associator, so
  that the statistics are computed on the associated picks. If used before the
  associator, it will compute statistics on the unassociated picks, which may
  not be as meaningful.

  :param p_picks: Minimum number of P picks per event
  :param s_picks: Minimum number of S picks per event
  :param total_picks: Minimum total number of picks per event
  :param p_and_s_picks: Minimum number of P and S picks per event
  :param region: Geographical region to consider for events
  :param base: Path to the base directory to compare outputs with
  """

  def __init__(
      self,
      p_picks: Optional[int] = None,
      s_picks: Optional[int] = None,
      total_picks: Optional[int] = None,
      p_and_s_picks: Optional[int] = None,
      region: Optional[mplPath] = mplPath(OGS_STUDY_REGION, closed=True),
      base: Optional[str] = None,
  ):
    super().__init__(
      p_picks=p_picks,
      s_picks=s_picks,
      total_picks=total_picks,
      p_and_s_picks=p_and_s_picks,
    )
    self.region = region
    self.base = base

  def run(self, status: Status) -> None:
    if status.param_is_cached("events", self.name) and status.param_is_cached(
      "assignments", self.name
    ):
      status.set_cached_param(pd.DataFrame(), "events", self.name)
      status.set_cached_param(pd.DataFrame(), "assignments", self.name)
    else:
      events = status.get_param("events")
      assignments = status.get_param("assignments")
      events_assignments = self._event_stats_qc(events, assignments)
      status.set_cached_param(events_assignments[0], "events", self.name)
      status.set_cached_param(events_assignments[1], "assignments", self.name)

  @dask.delayed
  def _event_stats_qc(
      self, events: pd.DataFrame, assignments: pd.DataFrame
  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    events, assignments = super()._filter_events(events, assignments)
    if len(events) == 0 or len(assignments) == 0:
      return events, assignments
    events = self._get_pick_stats(events, assignments)
    return self._filter_events(events, assignments)

  def _filter_events(
      self, events: pd.DataFrame, assignments: pd.DataFrame
  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Apply region filter
    events = events[events[
      ["longitude", "latitude"]].apply(
        lambda x: self.region.contains_point(
          (x["longitude"], x["latitude"])), axis=1)
    ]
    assignments = assignments[
      assignments["event_idx"].isin(events.index)
    ].copy()
    return events, assignments
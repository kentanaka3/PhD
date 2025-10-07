import pandas as pd

from typing import Optional
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
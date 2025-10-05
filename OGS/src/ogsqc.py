import pandas as pd

from typing import Optional
from ml_catolog.modules import PickStatQC
from matplotlib.path import Path as mplPath

import ogsconstants as OGS_C

OGS_STUDY_REGION = [
  (9.5, 47.5),  (44.3, 47.5),
  (9.5, 15.0),  (44.3, 15.0),
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
    self.p_picks = p_picks
    self.s_picks = s_picks
    self.total_picks = total_picks
    self.p_and_s_picks = (
      p_and_s_picks  # Number of stations that have both P and S picks
    )
    self.region = region

  def _filter_events(
      self, events: pd.DataFrame, assignments: pd.DataFrame
  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    events, assignments = super()._filter_events(events, assignments)
    # Apply region filter
    events = events[events[
      [OGS_C.LONGITUDE_STR, OGS_C.LATITUDE_STR]].apply(
        lambda x: self.region.contains_point(
          (x[OGS_C.LONGITUDE_STR], x[OGS_C.LATITUDE_STR])), axis=1)
    ]
    assignments = assignments[assignments["event_idx"].isin(events.index)].copy()
    return events, assignments
"""
===============================================================================
OGS Quality-Control Modules - Region-Aware Pick & Event Statistics
===============================================================================

OVERVIEW:
Extends the ``ml_catalog`` ``PickStatQC`` module with OGS-specific filtering:

1. Geographic region filtering using the OGS study polygon (longitude/latitude
   bounding box from 9.5 to 15.0 degrees longitude and 44.3 to 47.5 latitude).
2. Event-level statistics computation for downstream catalog QC.
3. Storage of an optional base catalog path; no comparison is implemented here.

The module implements two classes:

- ``OGSPickStatQC``: subclass of ``PickStatQC`` adding a polygon-based
  region filter on top of the standard pick-count thresholds.
- ``OGSEventStatQC``: further extension that computes per-event pick statistics
  via :meth:`_get_pick_stats` after the initial threshold/region filter, then
  reapplies the region filter. Useful when
  attached after the associator.

MODULE CONSTANTS:
    OGS_STUDY_REGION : list[tuple[float, float]]
        Closed polygon defining the OGS study region in (lon, lat) degrees.

USAGE:
    from OGS.src.ogsqc import OGSPickStatQC, OGSEventStatQC

    qc = OGSPickStatQC(p_picks=3, s_picks=2, total_picks=6)
    builder.add_module(qc)

DEPENDENCIES:
    - dask: lazy ``@dask.delayed`` execution of per-event QC
    - pandas: tabular pick / event manipulation
    - matplotlib.path: polygon point-in-region testing
    - ml_catalog.modules.PickStatQC: base QC class

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
    events = events[events[["longitude", "latitude"]].apply(
        lambda x: self.region.contains_point(
            (x["longitude"], x["latitude"])
        ), axis=1
    )]
    assignments = assignments[
        assignments["event_idx"].isin(events.index)
    ].copy()
    return events, assignments


class OGSEventStatQC(OGSPickStatQC):
  """
  A quality control module based on event statistics.
  For each of the parameters evaluated (see below), only events with at least
  that many picks will be retained. In addition to performing quality control,
  this module writes statistics on the picks per event to the event dataframe.
  Therefore, it's often convenient to include it even without using it to
  filter by pick counts. The geographic filter still applies. This module
  requires associated events and assignments with event_idx links; it is
  intended for use after the associator.

  :param p_picks: Minimum number of P picks per event
  :param s_picks: Minimum number of S picks per event
  :param total_picks: Minimum total number of picks per event
  :param p_and_s_picks: Paired P/S threshold forwarded to PickStatQC
  :param region: Polygon used for contains_point testing; None is not handled
  :param base: Optional base directory stored but not used for comparison
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
    events = events[events[["longitude", "latitude"]].apply(
        lambda x: self.region.contains_point(
            (x["longitude"], x["latitude"])
        ), axis=1
    )]
    assignments = assignments[
        assignments["event_idx"].isin(events.index)
    ].copy()
    return events, assignments

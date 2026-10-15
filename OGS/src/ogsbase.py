"""
===============================================================================
OGS Base Extensions - Dask Serialization-Safe Catalog Merge
===============================================================================

OVERVIEW:
Provides :class:`OGSMergeEvents`, a subclass of ``ml_catalog.base.MergeEvents``
that resolves Dask distributed serialization failures.

By defining the delayed task function ``_merge_events_task`` as a module-level
pure function rather than an instance method bound to ``self``, Dask task-graph
serialization bypasses class/module identity mismatches across worker processes.

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
import dask
import pandas as pd
from ml_catalog.base import MergeEvents, Status

logger = logging.getLogger("ml_catalog")


@dask.delayed
def _merge_catalog_task(
    event_dfs: list[pd.DataFrame], pick_dfs: list[pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """
  Concatenate event and assignment DataFrames across group statuses.

  Executed lazily by Dask without capturing class instance state.
  """
  logger.debug("Merging catalogs")
  events, assignments = pd.concat(event_dfs), pd.concat(assignment_dfs)

  if len(events) > 0:
    events.sort_values("time", inplace=True)
    events.rename(columns={"idx": "idx_in_group"}, inplace=True)
    assignments.rename(
        columns={"event_idx": "event_idx_in_group"}, inplace=True
    )
    events.reset_index(drop=True, inplace=True)
    assignments.reset_index(drop=True, inplace=True)
    events["idx"] = events.index

  return events, assignments


class OGSMergeEvents(MergeEvents):
  """
  OGS specialization of MergeEvents that avoids capturing ``self`` in Dask graphs.
  """

  def run(self, status: OGSStatus, group_statuses: list[OGSGroupStatus]) -> None:
    if not group_statuses:
      return

    has_picks = any(
        "picks" in s._parameters and s._parameters["picks"][0] is not None
        for s in group_statuses
    )
    has_events = any(
        "events" in s._parameters and s._parameters["events"][0] is not None
        for s in group_statuses
    )

    if not has_picks and not has_events:
      logger.debug(
          "No picks or events present in group statuses; skipping merge.")
      return

    if status.param_is_cached("events", self.name) and status.param_is_cached(
        "assignments", self.name
    ):
      status.set_cached_param(pd.DataFrame(), "events", self.name)
      status.set_cached_param(pd.DataFrame(), "assignments", self.name)
    else:
      event_dfs = [s.get_param("events") for s in group_statuses]
      assignment_dfs = [s.get_param("assignments") for s in group_statuses]
      output = _merge_events_task(event_dfs, assignment_dfs)
      events, assignments = output[0], output[1]
      status.set_param(events, "events", self.name)
      status.set_param(assignments, "assignments", self.name)


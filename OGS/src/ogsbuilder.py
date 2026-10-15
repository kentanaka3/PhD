"""
===============================================================================
OGS Catalog Builder - Distributed DAG Pipeline Orchestrator
===============================================================================

OVERVIEW:
Provides the core catalog builder engine (OGSCatalogBuilder) decoupling the OGS
seismic processing pipeline from external frameworks:
  1. STATE & DAG ORCHESTRATION: Populates global state (OGSStatus), partitions
     execution into per-group DAGs (OGSGroupStatus), executes group-level modules
     (pickers, associators, locators), merges catalog artifacts via OGSMerge,
     and applies joint post-processing modules.
  2. DASK COMPUTE EXECUTION: Optimizes and materializes lazy Dask graphs across
     local multithreading or distributed clusters.
  3. TABULAR ARTIFACT EXPORT: Serializes unified seismic picks, event catalogs,
     and quality metrics to Parquet and CSV files following OGS conventions.

POLYMORPHIC UNIFICATION:
Anchored in Tier-0 modules:
  - Unified column schemas via `OGS_C` constants.
  - Logging via `OGS_U.setup_logger`.
  - State management and merging via `OGSStatus`, `OGSGroupStatus`, and `OGSMerge`.

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

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union
import dask
import pandas as pd

from . import ogsconstants as OGS_C, ogsutils as OGS_U
from .ogsbase import OGSCacheHelper, OGSGroupStatus, OGSModule, OGSMerge, OGSStatus


class OGSCatalogBuilder:
  """
  Universal pipeline builder and DAG orchestrator for OGS seismic processing.
  """

  def __init__(
      self,
      output_path: str | Path,
      data: Any = None,
      group_modules: Any = None,
      merge_module: Optional[OGSModule] = None,
      joint_modules: Any = None,
      outputs: Optional[Sequence[str]] = None,
      formats: Optional[Sequence[str]] = None,
      **kwargs: Any,
  ):
    """
    Initialize OGSCatalogBuilder.

    Args:
      output_path: Filesystem path where catalog outputs and caches will be written.
      data: Data source object providing waveform groups and station metadata.
      group_modules: Module or collection of modules executed per group (day).
      merge_module: Merger module combining group artifacts (defaults to OGSMerge).
      joint_modules: Post-processing modules executed across the unified catalog.
      outputs: Names of artifacts to export (e.g., ['picks', 'events', 'assignments']).
      formats: Target file formats ('parquet', 'csv'). Defaults to both.
    """
    self.output_path = Path(output_path).resolve()
    self.data = data
    self.logger = OGS_U.setup_logger(self.__class__.__name__)

    # Normalize group modules into a flat list
    if group_modules is None:
      self.group_modules = []
    elif isinstance(group_modules, dict):
      self.group_modules = list(group_modules.values())
    elif isinstance(group_modules, (list, tuple)):
      self.group_modules = list(group_modules)
    else:
      self.group_modules = [group_modules]

    # Normalize merger module
    if merge_module is not None:
      self.merge_module = merge_module
    else:
      self.merge_module = OGSMerge()

    # Normalize joint modules into a flat list
    if joint_modules is None:
      self.joint_modules = []
    elif isinstance(joint_modules, dict):
      self.joint_modules = list(joint_modules.values())
    elif isinstance(joint_modules, (list, tuple)):
      self.joint_modules = list(joint_modules)
    else:
      self.joint_modules = [joint_modules]

    self.outputs = list(outputs) if outputs is not None else [
        "picks", "events"]
    self.formats = [f.lower().lstrip(".")
                    for f in (formats or ["parquet", "csv"])]
    self.kwargs = kwargs

    self.status = OGSStatus(self.output_path)

  def setup(self) -> None:
    """Prepare environment, load inventories, and call module setups."""
    self.output_path.mkdir(parents=True, exist_ok=True)

    # Populate station inventory if available from data source
    if self.data is not None:
      if hasattr(self.data, "populate_status") and callable(self.data.populate_status):
        try:
          self.data.populate_status(self.status)
        except Exception as e:
          self.logger.warning(
              f"Could not populate status from data source: {e}"
          )

      if hasattr(self.data, "get_inventory") and callable(self.data.get_inventory):
        try:
          inv = self.data.get_inventory()
          if inv is not None:
            self.status.set_param(inv, "inventory")
        except Exception as e:
          self.logger.warning(
              f"Could not load inventory from data source: {e}")

      if hasattr(self.data, "get_stations") and callable(self.data.get_stations):
        try:
          stas = self.data.get_stations()
          if stas is not None:
            self.status.set_param(stas, "stations")
        except Exception as e:
          self.logger.warning(f"Could not load stations from data source: {e}")

    # Invoke setup on each module if implemented
    for mod in self.group_modules:
      if hasattr(mod, "setup") and callable(mod.setup):
        mod.setup(self.status)

    if hasattr(self.merge_module, "setup") and callable(self.merge_module.setup):
      self.merge_module.setup(self.status)

    for mod in self.joint_modules:
      if hasattr(mod, "setup") and callable(mod.setup):
        mod.setup(self.status)

  def build_dag(self) -> tuple[OGSStatus, list[OGSGroupStatus]]:
    """
    Construct the full Dask computation graph across groups and pipeline stages.

    Returns:
      Tuple of (global_status, list_of_group_statuses)
    """
    groups: list[Any] = []
    if self.data is not None and hasattr(self.data, "groups"):
      groups_attr = self.data.groups
      groups = groups_attr() if callable(groups_attr) else list(groups_attr)

    self.logger.info(f"Building DAG across {len(groups)} group(s)...")
    group_statuses: list[OGSGroupStatus] = []

    for group in groups:
      group_str = str(group)
      gs = OGSGroupStatus(group=group_str, output_path=self.output_path)
      gs.set_param(group_str, OGS_C.GROUPS_STR)

      # Copy shared global state (e.g. inventory / stations) to group status
      gs.update(self.status)

      # Connect lazy waveform data function / stream if available
      if hasattr(self.data, "get_group") and callable(self.data.get_group):
        data_func = functools.partial(self.data.get_group, group)
        gs.set_param(data_func, "data_func")
        gs.set_param(data_func, "stream")

      # Execute per-group processing modules
      for mod in self.group_modules:
        if hasattr(mod, "run") and callable(mod.run):
          mod.run(gs)

      group_statuses.append(gs)

    # Execute catalog merger across group statuses
    if self.merge_module is not None and hasattr(self.merge_module, "run"):
      self.merge_module.run(self.status, group_statuses)

    # Execute joint post-processing modules on global status
    for mod in self.joint_modules:
      if hasattr(mod, "run") and callable(mod.run):
        mod.run(self.status)

    return self.status, group_statuses

  def run(self) -> OGSStatus:
    """
    Execute pipeline setup, construct compute graph, evaluate Dask tasks,
    and persist requested tabular outputs to disk.

    Returns:
      The populated global OGSStatus.
    """
    self.setup()
    status, _ = self.build_dag()

    self.logger.info("Computing task graph and writing artifacts...")

    target_keys: dict[str, str] = {}
    lazy_map: dict[str, Any] = {}

    for key in self.outputs:
      target_key = key
      # Polymorphic alias: 'assignments' maps to 'picks' if assignments not explicit
      if target_key == "assignments" and target_key not in status._parameters:
        if "picks" in status._parameters:
          target_key = "picks"

      if target_key not in status._parameters:
        self.logger.debug(
            f"Output '{key}' not found in status registry; skipping."
        )
        continue

      target_keys[key] = target_key
      payload, _, _ = status._parameters[target_key]
      if dask.is_dask_collection(payload) and target_key not in lazy_map:
        lazy_map[target_key] = payload

    if lazy_map:
      self.logger.info(
          f"Jointly evaluating {len(lazy_map)} lazy Dask collection(s)..."
      )
      computed_results = dask.compute(*lazy_map.values())
      for k, res in zip(lazy_map.keys(), computed_results):
        status._parameters[k] = (res, lambda x: x, True)

    written_files: set[Path] = set()

    for key in self.outputs:
      if key not in target_keys:
        continue
      target_key = target_keys[key]
      val = status.get_param(target_key)

      if isinstance(val, pd.DataFrame):
        for fmt in self.formats:
          out_file = self.output_path / f"{key}.{fmt}"
          if out_file not in written_files:
            out_file.parent.mkdir(parents=True, exist_ok=True)
            if fmt == "parquet":
              val.to_parquet(out_file, index=False)
            elif fmt == "csv":
              val.to_csv(out_file, index=False)
            written_files.add(out_file)
            self.logger.info(f"Persisted {len(val)} rows to {out_file}")

          # If exporting picks and assignments is in outputs, also export assignments
          if key == "picks" and "assignments" in self.outputs:
            assign_file = self.output_path / f"assignments.{fmt}"
            if assign_file not in written_files:
              assign_file.parent.mkdir(parents=True, exist_ok=True)
              if fmt == "parquet":
                val.to_parquet(assign_file, index=False)
              elif fmt == "csv":
                val.to_csv(assign_file, index=False)
              written_files.add(assign_file)
              self.logger.info(f"Persisted {len(val)} rows to {assign_file}")
      else:
        self.logger.info(
            f"Parameter '{key}' is {type(val)}; skipping tabular serialization."
        )

    self.logger.info(
        f"Pipeline run completed successfully for {self.output_path}"
    )
    return self.status

"""
=============================================================================
OGS Catalog Builder MPI - Distributed Dask-MPI Catalog Construction Engine
=============================================================================

OVERVIEW:
HPC-optimized catalog builder implementing distributed execution via Dask-MPI.
Extends the base ``ml_catalog.CatalogBuilder`` framework with cluster-aware
resource pinning, memory management, and task graph optimization.

KEY FEATURES:
  1. CUDA GPU PINNING: Detects local Slurm / OpenMPI rank and pins each worker
     process to a distinct physical GPU before Dask intercepts worker loops.
  2. DASK-MPI CLUSTER INITIALIZATION: Manages distributed scheduler-worker
     topologies under Slurm allocations with customizable memory limits.
  3. COMPUTE GRAPH EXECUTION: Orchestrates group modules, merge modules, and
     joint post-processing modules with automatic task-graph optimization.
  4. FORMAT SERIALIZATION: Streams processed seismic event catalogs and pick
     assignments to CSV and Parquet files with distributed performance reporting.

USAGE:
# Dispatched within a Slurm allocation via LAUNCHME.sh or srun:
  srun -n <ntasks> python -m mpi4py ogsbuilderMPI.py ...

DEPENDENCIES:
- dask / dask.distributed: distributed DAG task scheduling
  - dask_mpi: MPI-based cluster bootstrapping
  - torch: CUDA device detection and worker GPU pinning
  - ml_catalog: base CatalogBuilder, status registry, and pipeline modules
  - pandas: tabular data export

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

import os
import functools

import dask
import dask.distributed
import dask.optimization
import pandas as pd

from ml_catalog.base import AbstractModule, GroupStatus, MergeModule, Status
from ml_catalog.util import logger

from dask_mpi import initialize
from ml_catalog import CatalogBuilder


class OGSCatalogBuilderMPI(CatalogBuilder):
  """
  OGS-specific implementation of the ML CatalogBuilder.

  This class extends the base CatalogBuilder to include OGS-specific
  configurations and methods for building seismic event catalogs.
  """

  def run(self) -> None:
    """
    Run the builder as configured.
    This function will build and execute the compute graph.
    """
    slurm_mem = os.getenv("SLURM_MEM_PER_CPU")
    if slurm_mem is not None and slurm_mem.strip().isdigit():
      mem_limit = f"{int(slurm_mem)}MB"
    else:
      mem_limit = "8000MB"
    # Pin GPU device based on local MPI rank if CUDA is available BEFORE dask_mpi intercepts workers
    local_rank_str = (
        os.getenv("SLURM_LOCALID")
        or os.getenv("OMPI_COMM_WORLD_LOCAL_RANK")
        or os.getenv("MPI_LOCALRANKID")
        or os.getenv("LOCAL_RANK")
    )
    if local_rank_str is not None:
      try:
        local_rank = int(local_rank_str)
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
          num_gpus = torch.cuda.device_count()
          device_id = local_rank % num_gpus
          torch.cuda.set_device(device_id)
          logger.info(
              f"Local rank {local_rank} pinned to CUDA device {device_id} "
              f"of {num_gpus} available GPUs"
          )
      except Exception as e:
        logger.warning(
            f"Failed to pin CUDA device for local rank {local_rank_str}: {e}"
        )

    initialize(memory_limit=mem_limit)

    client = dask.distributed.Client()
    ntasks = os.getenv("SLURM_NTASKS")
    if ntasks is not None and ntasks.strip().isdigit():
      n_workers = max(int(ntasks) - 2, 1)
      client.wait_for_workers(n_workers=n_workers)
    print(client.scheduler_info()["workers"])

    status = Status(self.output_path)

    logger.debug("Populating status")
    self.data.populate_status(status)

    for module in (
        list(self.group_modules.values())
        + [self.merge_module]
        + list(self.joint_modules.values())
    ):
      logger.debug(f"Setting up {module.name}")
      for param in module.output_keys():
        status.register_parameter(param, duplicate=True)
        module.setup(status)

    self._write_citations()
    self._write_versions()

    group_statuses = []
    groups = self.data.groups()

    regrouped = self._regroup(groups)

    self._regrouped_to_df(regrouped).to_csv(
        self.output_path / "groups.csv", index=False
    )

    for group, subgroups in regrouped.items():
      logger.debug(f"Setting up group {group}")
      group_status = GroupStatus(group, self.output_path)
      group_status.update(status, deepcopy=False)

      group_status.register_parameter("data_func")
      group_status.set_param(
          functools.partial(
              self._get_multigroup,
              self.data.get_group,
              subgroups
          ),
          "data_func",
          None,
      )

      for module in self.group_modules.values():
        module.run(group_status)

      group_statuses.append(group_status)

    self.merge_module.run(status, group_statuses)

    for module in self.joint_modules.values():
      module.run(status)

    # Trigger computations and write outputs
    logger.debug("Starting computation")
    outputs = [status.get_param(out) for out in self.outputs]
    if self.adaptive_maximum is not None:
      self.cluster.adapt(minimum=1, maximum=self.adaptive_maximum)

    with dask.config.set(delayed_optimize=self._optimize_dask_graph):
      with dask.distributed.performance_report(
          self.output_path / "dask-report.html"
      ):
        outputs = client.compute(outputs, sync=True)

    logger.debug("Writing outputs")
    for param, output in zip(self.outputs, outputs):
      for output_format in self.formats:
        if output_format == "csv":
          output.to_csv(self.output_path / (param + ".csv"), index=False)
        elif output_format == "parquet":
          output.to_parquet(
              self.output_path / (param + ".parquet"), index=False
          )
        else:
          raise NotImplementedError(f"Unknown format '{output_format}'")

    client.shutdown()

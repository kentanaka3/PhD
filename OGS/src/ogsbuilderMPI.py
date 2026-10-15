"""
===============================================================================
OGS Catalog Builder MPI - Distributed Dask-MPI Catalog Construction Engine
===============================================================================

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
===============================================================================
"""

import os

import dask.distributed

from dask_mpi import initialize
from ml_catalog import CatalogBuilder
from ml_catalog.util import logger


class OGSCatalogBuilderMPI(OGSCatalogBuilder):
  """
  Distributed CatalogBuilder subclass that bootstraps Dask-MPI, assigns CUDA
  devices by local rank, executes configured modules across cluster workers,
  and exports unified catalog artifacts.
  """

  def run(self) -> OGSStatus:
    """
    Bootstrap Dask-MPI, pin GPU resources, execute compute graph, and persist
    outputs.
    """
    slurm_mem = os.getenv("SLURM_MEM_PER_CPU")
    if slurm_mem is not None and slurm_mem.strip().isdigit():
      mem_limit = f"{int(slurm_mem)}MB"
    else:
      mem_limit = "8000MB"

    # Pin GPU device based on local MPI rank via environment before dask_mpi intercepts workers
    local_rank_str = (
        os.getenv("SLURM_LOCALID")
        or os.getenv("OMPI_COMM_WORLD_LOCAL_RANK")
        or os.getenv("MPI_LOCALRANKID")
        or os.getenv("LOCAL_RANK")
    )
    if local_rank_str is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
      try:
        local_rank = int(local_rank_str)
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
          num_gpus = torch.cuda.device_count()
          device_id = local_rank % num_gpus
          os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
          logger.info(
              f"Local rank {local_rank} assigned CUDA device {device_id} "
              f"of {num_gpus} available GPUs"
          )
      except Exception as e:
        logger.warning(
            f"Failed to assign CUDA device for local rank {local_rank_str}: {e}"
        )

    initialize(memory_limit=mem_limit)

    # In Dask-MPI, the scheduler is already running in rank 0; passing self.cluster=None
    # connects Client() to the active Dask-MPI scheduler address directly.
    self.cluster = None

    client = dask.distributed.Client()
    ntasks = os.getenv("SLURM_NTASKS")
    if ntasks is not None and ntasks.strip().isdigit() and int(ntasks) > 2:
      n_workers = int(ntasks) - 2
      client.wait_for_workers(n_workers=n_workers)

    workers = client.scheduler_info().get("workers", {})
    logger.info(f"Dask-MPI cluster active with {len(workers)} worker(s)")

    try:
      return super().run()
    finally:
      logger.info("Shutting down Dask-MPI client...")
      client.shutdown()

"""
===============================================================================
OGS NonLinLoc Module - Table Staging and Local Process Parallelism
===============================================================================

OVERVIEW:
Provides :class:`OGSNonLinLoc`, an OGS-specific subclass of
``ml_catalog.modules.NonLinLoc`` with configurable table staging and local
process parallelism for cluster workflows:

1. Environment-Selected Table Staging:
   - Staging travel-time and angle buffers/headers to a directory selected
     by NLLOC_SHM_DIR or TMPDIR, using POSIX file locking (`fcntl.flock`).
     RAM-backed storage and node/job isolation depend on those environment
     paths; the implementation does not enforce /dev/shm.
   - Directing temporary chunk execution files (`loc_{chunk}`, `obs_{chunk}`, `run_{chunk}`)
     under TMPDIR when set, otherwise the system temporary directory.
   - Fixing travel-time table filename normalization to correctly recognize `.angle.buf`.

2. Local Process Parallelism:
   - Dynamic CPU worker resolution (`_resolve_n_jobs`) supporting explicit overrides,
     SLURM environment variables ($SLURM_CPUS_PER_TASK, $LOKY_MAX_CPU_COUNT),
     and special-case count reductions (224 -> 112, 64 -> 32) for non-explicit
     worker counts. It does not discover physical cores generally.
   - Dynamic chunksize computation (`_compute_dynamic_chunksize`) targeting ~4 chunks
     per worker for load balancing, clamped between 10 and 100.
   - Parallelized chunk relocation via `joblib.Parallel(backend="loky")`.
   - Parallelized travel-time table generation (`Grid2Time`) across stations and
     phases (P/S), with velocity model grids pre-copied to the output directory.

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

from ml_catalog.util import logger
from ml_catalog.modules import NonLinLoc
from ml_catalog.base import CacheHelper
import copy
import fcntl
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import dask
import pandas as pd
from joblib import Parallel, delayed

from . import ogsconstants as OGS_C

# Ensure NonLinLoc binary directory is always on PATH
_NLL_DEFAULT_BIN = Path(os.environ.get(
    "NLL_PATH", "/leonardo_work/IscrC_AISeism/NonLinLoc")) / "bin"
if _NLL_DEFAULT_BIN.is_dir() and str(_NLL_DEFAULT_BIN) not in os.environ.get("PATH", "").split(os.pathsep):
  os.environ["PATH"] = f"{_NLL_DEFAULT_BIN}{os.pathsep}{os.environ.get('PATH', '')}"


def _run_single_grid2time(
    bin_path: str,
    cfg: dict,
    in_path: Path,
    verbose: bool = False,
) -> None:
  """
  Execute a single Grid2Time job for one station and phase.
  """
  in_path.parent.mkdir(parents=True, exist_ok=True)
  with open(in_path, "w") as f:
    for key, val in cfg.items():
      if not isinstance(val[0], list):
        val = [val]
      for row in val:
        f.write(" ".join([key] + [str(x) for x in row]))
        f.write("\n")

  kwargs = {} if verbose else {
      "stdout": subprocess.DEVNULL,
      "stderr": subprocess.DEVNULL,
  }
  subprocess.run(
      [bin_path, str(in_path)],
      check=True,
      **kwargs,
  )


class OGSNonLinLoc(NonLinLoc):
  """
  NonLinLoc wrapper with environment-selected table staging, dynamic chunking,
  and local joblib process parallelism. Distribution of groups is delegated
  to the surrounding Dask pipeline.

  :param pick_std: Standard deviation for the picks in seconds
  :param mode_3d: If true, NonLinLoc will be executed with a 3D velocity model.
  :param velocity_models_3d: Dictionary mapping phases (``P``, ``S``) to the 3D velocity model
  :param tt_tables: Path to precomputed travel-time tables.
  :param location_output: The location output to use from NonLinLoc ('mle' or 'exp').
  :param nlloc_prefix: Prefix to the NLLoc binaries.
  :param verbose: If true, logging will be enabled
  :param chunksize: Number of event IDs per relocation chunk (optional override).
  :param parallel: If true, location will be performed in parallel using multiple CPU cores.
  :param n_jobs: Number of parallel workers to dispatch (optional user override).
  """

  def __init__(
      self,
      pick_std: dict[str, float] | float = 0.05,
      mode_3d: bool = False,
      velocity_models_3d: Optional[dict[str, str]] = None,
      tt_tables: Optional[str | Path] = None,
      location_output: str = "mle",
      nlloc_prefix: str = "",
      verbose: bool = False,
      chunksize: Optional[int] = None,
      parallel: bool = True,
      n_jobs: Optional[int] = None,
      **nlloc_cfg,
  ):
    actual_chunksize = 100 if chunksize is None else chunksize
    super().__init__(
        pick_std=pick_std,
        mode_3d=mode_3d,
        velocity_models_3d=velocity_models_3d,
        tt_tables=tt_tables,
        location_output=location_output,
        nlloc_prefix=nlloc_prefix,
        verbose=verbose,
        chunksize=actual_chunksize,
        parallel=parallel,
        **nlloc_cfg,
    )
    self.n_jobs = n_jobs
    self._user_chunksize = chunksize is not None

  def _resolve_n_jobs(self) -> int:
    """
    Resolve effective number of parallel workers for NLLoc.

    Resolution hierarchy:
    1. Explicit self.n_jobs (> 0)
    2. Environment variables: $SLURM_CPUS_PER_TASK, $LOKY_MAX_CPU_COUNT
    3. CPU affinity via os.sched_getaffinity(0)
    4. Fallback to os.cpu_count() or 1
    Counts 224 and 64 are reduced to 112 and 32 in steps 2-4, not step 1.
    """
    if self.n_jobs is not None and self.n_jobs > 0:
      return self.n_jobs

    for env_var in ("SLURM_CPUS_PER_TASK", "LOKY_MAX_CPU_COUNT"):
      val = os.environ.get(env_var)
      if val:
        try:
          n = int(val)
          if n > 0:
            if n == 224:
              return 112
            elif n == 64:
              return 32
            return n
        except ValueError:
          pass

    try:
      affinity = len(os.sched_getaffinity(0))
      if affinity == 224:
        return 112
      elif affinity == 64:
        return 32
      elif affinity > 0:
        return affinity
    except (AttributeError, OSError):
      pass

    cpus = os.cpu_count() or 1
    if cpus == 224:
      return 112
    elif cpus == 64:
      return 32
    return cpus

  def _compute_dynamic_chunksize(self, n_events: int, n_workers: int) -> int:
    """
    Compute dynamic chunksize for load balancing.

    Respect an explicitly passed chunksize (including 100), or a current value
    other than 100. Otherwise, compute:
    target_chunks = n_workers * 4 (~4 chunks per core for load balancing)
    raw_chunksize = ceil(n_events / target_chunks)
    Clamp between 10 (overhead amortization) and 100 (straggler prevention).
    """
    if getattr(self, "_user_chunksize", False) or self.chunksize != 100:
      return self.chunksize

    if n_events <= 0:
      return 10

    target_chunks = max(1, n_workers * 4)
    raw_chunksize = math.ceil(n_events / target_chunks)
    return max(10, min(100, raw_chunksize))

  @property
  def tmpdir(self) -> Path:
    """
    Lazily create/reuse the chunk temporary directory under TMPDIR if set,
    otherwise under the system temporary directory. Storage is RAM-backed
    only when the selected directory is on tmpfs.
    """
    if self._tmpdir is None or not Path(self._tmpdir).is_dir():
      tmp_base = os.environ.get("TMPDIR")
      if tmp_base:
        Path(tmp_base).mkdir(parents=True, exist_ok=True)
        self._tmpdir = Path(tempfile.mkdtemp(dir=tmp_base))
      else:
        self._tmpdir = Path(tempfile.mkdtemp())
    return self._tmpdir

  def _cleanup_full(self) -> None:
    """
    Remove the chunk temporary directory if present and reset its cached path.
    """
    if self._tmpdir is not None and Path(self._tmpdir).is_dir():
      shutil.rmtree(self._tmpdir)
    self._tmpdir = None

  def _get_shm_time_path(self, remote_time_path: Path) -> Path:
    """
    Stage top-level time/angle buffers and headers to an environment-selected path.

    Uses NLLOC_SHM_DIR or derives a time directory from TMPDIR; without either,
    returns the original path. A lock serializes copying and a done marker
    enables reuse. The marker does not identify the source or verify freshness;
    before it exists, files are copied only if absent or their sizes differ.
    Returns the staged or original path, without copying mapping.csv.
    """
    remote_time_path = Path(remote_time_path)
    if not remote_time_path.exists():
      logger.warning(
          f"Remote travel-time path {remote_time_path} does not exist; skipping SHM staging."
      )
      return remote_time_path

    # Determine SHM target directory from $NLLOC_SHM_DIR or $TMPDIR
    shm_dir = os.environ.get("NLLOC_SHM_DIR")
    if shm_dir:
      base_dir = Path(shm_dir)
      shm_time_path = base_dir / "time" if base_dir.name != "time" else base_dir
    elif "TMPDIR" in os.environ:
      tmp_path = Path(os.environ["TMPDIR"])
      if (tmp_path.parent / "time").is_dir() or tmp_path.name == "tmp":
        shm_time_path = tmp_path.parent / "time"
      elif tmp_path.name == "time":
        shm_time_path = tmp_path
      else:
        shm_time_path = tmp_path / "time"
    else:
      logger.debug(
          "Neither NLLOC_SHM_DIR nor TMPDIR is set; using remote travel-time path."
      )
      return remote_time_path

    try:
      if remote_time_path.resolve() == shm_time_path.resolve():
        return remote_time_path
    except Exception:
      pass

    shm_time_path.mkdir(parents=True, exist_ok=True)
    done_file = shm_time_path / ".time_stage.done"
    lock_file = shm_time_path / ".time_stage.lock"

    # Fast path: already staged by an earlier worker or Dask group
    if done_file.is_file():
      return shm_time_path

    # Synchronize staging across concurrent workers using POSIX file locking
    with open(lock_file, "w") as lock_f:
      fcntl.flock(lock_f, fcntl.LOCK_EX)
      try:
        # Double-check after acquiring the lock
        if done_file.is_file():
          return shm_time_path

        logger.info(
            f"Staging travel-time tables from {remote_time_path} to {shm_time_path}"
        )
        copied_count = 0
        for pattern in ("*.time.buf", "*.time.hdr", "*.angle.buf", "*.angle.hdr"):
          for src in remote_time_path.glob(pattern):
            dst = shm_time_path / src.name
            if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
              shutil.copyfile(src, dst)
            copied_count += 1

        logger.info(
            f"Successfully staged {copied_count} travel-time tables into {shm_time_path}"
        )
        done_file.touch()
      finally:
        fcntl.flock(lock_f, fcntl.LOCK_UN)

    return shm_time_path

  @dask.delayed
  def _run_nlloc_internal(
      self,
      events: pd.DataFrame,
      assignments: pd.DataFrame,
      setup_token: bool,
      path_helper: CacheHelper,
      group: str,
  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run NonLinLoc relocation for the assigned group events.
    Retrieves the travel-time path and optionally stages tables via
    `_get_shm_time_path`. Dispatches chunks via local joblib processes when
    parallel is true, otherwise sequentially. Merges relocation results with
    inputs and removes the temporary directory after successful completion.
    """
    if len(events) == 0 or len(assignments) == 0:
      return pd.DataFrame(), pd.DataFrame()

    with path_helper.get_cache_path(
        "time", self.name, drop_group=True, mode="r", localize=False
    ) as remote_time_path:
      time_path = self._get_shm_time_path(remote_time_path)

      indices = assignments["event_idx"].unique()
      n_workers = self._resolve_n_jobs()
      effective_chunksize = self._compute_dynamic_chunksize(
          len(indices), n_workers
      )

      p = 0
      results = []

      while p < len(indices):
        sub_assignments = assignments[
            assignments["event_idx"].isin(
                indices[p: p + effective_chunksize]
            )
        ]
        if self.parallel:
          results.append(
              delayed(self._locate_chunk)(
                  sub_assignments, time_path, group, p, self.tmpdir
              )
          )
        else:
          results.append(
              self._locate_chunk(
                  sub_assignments, time_path, group, p, self.tmpdir
              )
          )
        p += effective_chunksize

      if self.parallel:
        results = Parallel(n_jobs=n_workers, backend="loky")(results)

    new_events = pd.concat([x for x, _ in results])
    new_assignments = pd.concat([y for _, y in results])

    new_events["group"] = group
    new_assignments["group"] = group

    events, assignments = self._merge_results(
        events, new_events, assignments, new_assignments
    )

    self._cleanup_full()

    return events, assignments

  def _calculate_traveltimes(self, path_helper: CacheHelper) -> None:
    """
    Calculate travel-time tables with parallelized Grid2Time across stations
    and phases (P and S) using joblib Parallel.
    Pre-copies model grids to the time-output directory before dispatch.
    """
    with path_helper.get_cache_path(
        "model/layer", self.name, mode="w"
    ) as model_output, path_helper.get_cache_path(
        "time/layer", self.name, mode="w"
    ) as time_output:
      self.nlloc_cfg["VGOUT"] = [str(model_output)]

      if self.mode_3d:
        bin_name = "Vel2Grid3D"
        for phase in "PS":
          self.nlloc_cfg["VGTYPE"] = [phase]
          self.nlloc_cfg["VGINP"] = [self.velocity_models_3d[phase]]

          with path_helper.get_cache_path(
              "run", self.name, suffix="tt_grid.in", mode="w"
          ) as path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._write_infile(self.nlloc_cfg, path=path)
            subprocess.run(
                [self.nlloc_prefix + bin_name, str(path)],
                check=True,
                **self._subprocess_kwargs,
            )
      else:
        bin_name = "Vel2Grid"
        with path_helper.get_cache_path(
            "run", self.name, suffix="tt_grid.in", mode="w"
        ) as path:
          path.parent.mkdir(parents=True, exist_ok=True)
          self._write_infile(self.nlloc_cfg, path=path)
          subprocess.run(
              [self.nlloc_prefix + bin_name, str(path)],
              check=True,
              **self._subprocess_kwargs,
          )

      # Copy layer.*.mod.buf and .mod.hdr from model_output to time_output beforehand
      # before dispatching the parallel Grid2Time jobs.
      model_dir = model_output.parent
      time_dir = time_output.parent
      time_dir.mkdir(parents=True, exist_ok=True)
      for mod_pattern in ("*.mod.buf", "*.mod.hdr"):
        for mod_file in model_dir.glob(mod_pattern):
          target_file = time_dir / mod_file.name
          if not target_file.is_file() or target_file.stat().st_size != mod_file.stat().st_size:
            shutil.copyfile(mod_file, target_file)

      # Build tasks list [(st_entry, phase, st_code), ...]
      station_entries = self.nlloc_cfg.get("GTSRCE", [])
      tasks = []
      for st_entry in station_entries:
        st_code = st_entry[0]
        for phase in (OGS_C.PWAVE, OGS_C.SWAVE):
          tasks.append((st_entry, phase, st_code))

      if not tasks:
        return

      n_workers = self._resolve_n_jobs()
      grid2time_bin = self.nlloc_prefix + "Grid2Time"

      with path_helper.get_cache_path("run", self.name, mode="w") as run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)

        job_list = []
        for st_entry, phase, st_code in tasks:
          task_cfg = copy.deepcopy(self.nlloc_cfg)
          task_cfg["GTFILES"] = [[str(model_output), str(time_output), phase]]
          task_cfg["GTSRCE"] = [st_entry]
          in_path = run_dir / f"tt_grid_{phase}_{st_code}.in"
          job_list.append(
              delayed(_run_single_grid2time)(
                  grid2time_bin, task_cfg, in_path, self.verbose
              )
          )

        Parallel(n_jobs=min(n_workers, len(tasks)), backend="loky")(job_list)

      self.nlloc_cfg["GTFILES"] = [[str(model_output), str(time_output), OGS_C.SWAVE]]

  def _load_tt_tables(self, path_helper: CacheHelper) -> None:
    """
    Load precalculated travel-time tables
    """
    if self.tt_tables is None:
      return

    tt_path = Path(self.tt_tables)
    mapping = pd.read_csv(tt_path / "mapping.csv", keep_default_na=False)

    with path_helper.get_cache_path("time", self.name, mode="w") as time_output:
      for _, row in mapping.iterrows():
        station = row["id"]
        if station not in self._station_mapper:
          logger.debug(
              f"Station {station} in TT mapping but not in station list"
          )
          continue

        short = self._station_mapper.translate_station(station)

        for phase in "ps":
          table = self._normalize_table_name(row[f"{phase}_table"])
          if table.strip() == "":
            continue

          copy_count = 0
          for suffix in [".time.hdr", ".time.buf", ".angle.hdr", ".angle.buf"]:
            org_table = tt_path / (table + suffix)
            if not org_table.is_file():
              continue
            target_table = (
                time_output / f"layer.{phase.upper()}.{short}{suffix}"
            )

            shutil.copyfile(org_table, target_table)

            copy_count += 1

          if copy_count == 0:
            logger.warning(f"TT table {table} not found")

  @staticmethod
  def _normalize_table_name(x: str) -> str:
    """
    Normalize table name by stripping known suffixes including .angle.buf.
    """
    for suffix in [".time.hdr", ".time.buf", ".angle.hdr", ".angle.buf"]:
      if x.endswith(suffix):
        x = x[: -len(suffix)]
    return x

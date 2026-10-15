"""
===============================================================================
OGS Run - Universal Command-Line Pipeline Entrypoint
===============================================================================

OVERVIEW:
Universal CLI entrypoint replacing Hydra and ml_catalog_run:
  1. CLI ARGUMENT PARSING: Parses `--config-dir`, `--config-name`, and arbitrary
     dot/slash parameter overrides, stripping leading `+` for Makefile compatibility
     (e.g., `+index_only=true`, `+libpath=.`).
  2. SQUIRREL INDEXING BARRIER: When `index_only=true` is set, runs fast Squirrel
     archive synchronization and exits cleanly without launching pipeline workers.
  3. DYNAMIC COMPONENT INSTANTIATION: Resolves and builds data sources, group
     modules, mergers, and joint modules via `ogsconfig.instantiate`.
  4. CLUSTER & HARDWARE DISPATCH: Inspects the runtime environment for Slurm or
     MPI process contexts and dispatches either `OGSCatalogBuilderMPI` or
     `OGSCatalogBuilder`.

USAGE:
  python -m OGS.src.ogsrun --config-dir conf --config-name config \\
      data=ogsDB group_modules=Picker output_path=catalogs/2024

  python -m OGS.src.ogsrun --config-dir conf +index_only=true data.check=true

AUTHORS:
  - 健
  - Istituto Nazionale di Oceanografia e di Geofisica Sperimentale (OGS)
    Centro di Ricerche Sismologiche (CRS)
  - Università degli Studi di Trieste (UniTS)
    Dipartimento di Matematica, Informatica e Geoscienze (MIGe)
    Applied Data Science and Artificial Intelligence (ADSAI)
===============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from . import ogsconstants as OGS_C, ogsutils as OGS_U
from .ogsbase import OGSStatus
from .ogsbuilder import OGSCatalogBuilder
from .ogsbuilderMPI import OGSCatalogBuilderMPI
from .ogsconfig import instantiate, load_config

logger = OGS_U.setup_logger("OGSRun")


def is_mpi_environment() -> bool:
  """
  Detect if executing inside an MPI or Slurm distributed allocation.
  """
  slurm_tasks = os.getenv("SLURM_NTASKS")
  if slurm_tasks is not None and slurm_tasks.strip().isdigit() and int(slurm_tasks) > 1:
    return True

  mpi_indicators = (
      "OMPI_COMM_WORLD_RANK",
      "PMIX_RANK",
      "MPI_LOCALRANKID",
      "PMI_RANK",
  )
  return any(var in os.environ for var in mpi_indicators)


def build_parser() -> argparse.ArgumentParser:
  """Create the command-line argument parser."""
  parser = argparse.ArgumentParser(
      description="OGS Seismic Pipeline Universal Runner",
      allow_abbrev=False,
  )
  parser.add_argument(
      "--config-dir",
      type=str,
      default=None,
      help="Directory containing YAML configuration files (defaults to OGS/conf).",
  )
  parser.add_argument(
      "--config-name",
      type=str,
      default="config",
      help="Root configuration filename without .yaml extension (defaults to config).",
  )
  parser.add_argument(
      "--mpi",
      action="store_true",
      default=False,
      help="Force distributed execution using OGSCatalogBuilderMPI.",
  )
  return parser


def parse_run_args(
    argv: Optional[Sequence[str]] = None,
) -> tuple[argparse.Namespace, list[str]]:
  """
  Parse known CLI flags and collect remaining parameters as config overrides.
  """
  parser = build_parser()
  args, remaining = parser.parse_known_args(argv)

  overrides: list[str] = []
  for item in remaining:
    # Handle --config-dir=val or --config-name=val passed in remaining
    if item.startswith("--config-dir="):
      args.config_dir = item.split("=", 1)[1]
    elif item.startswith("--config-name="):
      args.config_name = item.split("=", 1)[1]
    else:
      overrides.append(item)

  return args, overrides


def run_pipeline(
    config_dir: str | Path,
    config_name: str = "config",
    overrides: Optional[list[str]] = None,
    force_mpi: bool = False,
) -> Optional[OGSStatus]:
  """
  Load configuration, instantiate pipeline components, and execute the builder.
  """
  overrides = overrides or []

  # Process +libpath=. or libpath=. overrides before imports
  for arg in overrides:
    clean_arg = arg.lstrip("+")
    if clean_arg.startswith("libpath="):
      lp = clean_arg.split("=", 1)[1].strip()
      resolved_lp = str(Path(lp).resolve())
      if resolved_lp not in sys.path:
        sys.path.insert(0, resolved_lp)
        logger.info(f"Added library path to sys.path: {resolved_lp}")

  # Resolve configuration directory
  if config_dir is None:
    default_conf = Path(__file__).resolve().parent.parent / "conf"
    if default_conf.is_dir():
      resolved_dir = default_conf
    else:
      resolved_dir = Path.cwd() / "conf"
  else:
    resolved_dir = Path(config_dir).resolve()

  logger.info(
      f"Loading configuration from '{resolved_dir}' (root: '{config_name}')"
  )
  cfg = load_config(
      config_dir=resolved_dir, config_name=config_name, overrides=overrides
  )

  # Check for index_only barrier
  index_only = cfg.get("index_only", False)
  if isinstance(index_only, str):
    index_only = index_only.lower() in ("true", "1", "yes", "on")

  if index_only:
    logger.info("Operating in index_only mode: triggering Squirrel database update...")
    if "data" in cfg and cfg.data is not None:
      if isinstance(cfg.data, dict):
        cfg.data["check"] = True
      data_obj = instantiate(cfg.data)
      if hasattr(data_obj, "index_parallel") and callable(data_obj.index_parallel):
        workers = getattr(data_obj, "index_workers", None)
        data_obj.index_parallel(workers)
    logger.info("Squirrel index update completed successfully. Exiting.")
    return None

  # Instantiate pipeline components
  data = instantiate(cfg.data) if "data" in cfg and cfg.data is not None else None
  group_modules = (
      instantiate(cfg.group_modules)
      if "group_modules" in cfg and cfg.group_modules is not None
      else []
  )
  merge_module = (
      instantiate(cfg.merge_module)
      if "merge_module" in cfg and cfg.merge_module is not None
      else None
  )
  joint_modules = (
      instantiate(cfg.joint_modules)
      if "joint_modules" in cfg and cfg.joint_modules is not None
      else []
  )

  output_path = cfg.get("output_path", "catalog/OGS")

  # Dispatch builder based on MPI/Slurm environment
  is_mpi = force_mpi or is_mpi_environment()
  builder_cls = OGSCatalogBuilderMPI if is_mpi else OGSCatalogBuilder
  logger.info(f"Dispatching catalog builder: {builder_cls.__name__} (MPI={is_mpi})")

  builder_config = cfg.get("builder", {})
  if isinstance(builder_config, dict) and "_target_" in builder_config:
    builder = instantiate(
        builder_config,
        output_path=output_path,
        data=data,
        group_modules=group_modules,
        merge_module=merge_module,
        joint_modules=joint_modules,
    )
  else:
    builder_params = dict(builder_config) if isinstance(builder_config, dict) else {}
    builder = builder_cls(
        output_path=output_path,
        data=data,
        group_modules=group_modules,
        merge_module=merge_module,
        joint_modules=joint_modules,
        **builder_params,
    )

  return builder.run()


def main(argv: Optional[Sequence[str]] = None) -> int:
  """Main entrypoint function."""
  if argv is None:
    argv = sys.argv[1:]
  args, overrides = parse_run_args(argv)
  try:
    run_pipeline(
        config_dir=args.config_dir,
        config_name=args.config_name,
        overrides=overrides,
        force_mpi=args.mpi,
    )
    return 0
  except Exception as e:
    logger.error(f"Execution failed: {e}", exc_info=True)
    return 1


if __name__ == "__main__":
  sys.exit(main())


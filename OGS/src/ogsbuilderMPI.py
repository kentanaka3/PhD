import os
import functools
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Optional

import dask
import dask.distributed
import dask.optimization
import pandas as pd
import yaml
from dask.highlevelgraph import HighLevelGraph

from ml_catalog.base import AbstractModule, GroupStatus, MergeModule, Status
from ml_catalog.data import AbstractDataSource
from ml_catalog.types import DataInterface, pathlike
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
    initialize(memory_limit=f"{os.getenv('SLURM_MEM_PER_CPU'):d}MB")
    client = dask.distributed.Client()
    client.wait_for_workers(n_workers=(int(os.getenv("SLURM_NTASKS")) - 2)) # type: ignore
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
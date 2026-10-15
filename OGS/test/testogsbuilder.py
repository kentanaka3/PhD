"""
=============================================================================
OGS Builder Test Suite - Unit Tests for Pipeline DAG & Artifact Export
=============================================================================

OVERVIEW:
Unit tests verifying the end-to-end execution of OGSCatalogBuilder:
  1. DAG construction across groups using mock data and mock modules.
  2. Integration with OGSMerge for pick index unification across groups.
  3. Artifact serialization to <output_path>/picks.parquet, picks.csv, etc.
  4. Mocked cluster execution and rank pinning in OGSCatalogBuilderMPI.

AUTHORS:
  - 健
  - Istituto Nazionale di Oceanografia e di Geofisica Sperimentale (OGS)
=============================================================================
"""

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import dask
import pandas as pd

import sys
if "dask_mpi" not in sys.modules:
  sys.modules["dask_mpi"] = MagicMock()

from OGS.src import ogsconstants as OGS_C
from OGS.src.ogsbase import OGSGroupStatus, OGSModule, OGSMerge, OGSStatus
from OGS.src.ogsbuilder import OGSCatalogBuilder
from OGS.src.ogsbuilderMPI import OGSCatalogBuilderMPI


class MockDataSource:
  """Mock data source providing deterministic daily groups."""

  def __init__(self, groups: list[str]):
    self._groups = groups

  def groups(self) -> list[str]:
    return self._groups

  def get_group(self, group: str) -> dict[str, Any]:
    return {"group": group, "traces": []}


class MockPickerModule(OGSModule):
  """Mock picker module generating deterministic picks per group."""

  def __init__(self, **kwargs: Any):
    super().__init__(**kwargs)
    self.name = "MockPicker"

  def run(self, status: OGSStatus) -> None:
    group = status.get_param(OGS_C.GROUPS_STR)
    picks = pd.DataFrame({
        OGS_C.IDX_PICKS_STR: [0, 1],
        OGS_C.TIME_STR: [f"{group}T01:00:00", f"{group}T02:00:00"],
        OGS_C.STATION_STR: ["STA1", "STA2"],
        OGS_C.PHASE_STR: ["P", "S"],
        OGS_C.GROUPS_STR: [group, group],
    })
    status.set_param(picks, "picks", caller=self.name)


class TestOGSCatalogBuilder(unittest.TestCase):
  """End-to-end tests for OGSCatalogBuilder."""

  def test_end_to_end_dag_and_artifact_export(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      output_path = Path(tmpdir) / "catalog"
      data = MockDataSource(["2024-01-01", "2024-01-02"])
      picker = MockPickerModule()

      builder = OGSCatalogBuilder(
          output_path=output_path,
          data=data,
          group_modules=[picker],
          outputs=["picks", "assignments"],
          formats=["parquet", "csv"],
      )

      final_status = builder.run()
      self.assertIsInstance(final_status, OGSStatus)

      # Verify exported files exist
      picks_parquet = output_path / "picks.parquet"
      picks_csv = output_path / "picks.csv"
      self.assertTrue(picks_parquet.is_file())
      self.assertTrue(picks_csv.is_file())

      # Read back parquet and verify unification
      df_picks = pd.read_parquet(picks_parquet)
      # 2 groups * 2 picks per group = 4 total picks
      self.assertEqual(len(df_picks), 4)
      # Continuous unified indexing
      self.assertEqual(list(df_picks[OGS_C.IDX_PICKS_STR]), [0, 1, 2, 3])
      # Preserved group indices in idx_in_group
      self.assertEqual(list(df_picks["idx_in_group"]), [0, 1, 0, 1])

      # Read back CSV and verify matching record count
      df_csv = pd.read_csv(picks_csv)
      self.assertEqual(len(df_csv), 4)

  def test_dict_and_empty_modules_normalization(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      output_path = Path(tmpdir) / "catalog"
      data = MockDataSource(["2024-01-01"])
      picker = MockPickerModule()

      builder = OGSCatalogBuilder(
          output_path=output_path,
          data=data,
          group_modules={"picker": picker},
          joint_modules=None,
          outputs=["picks"],
          formats=["parquet"],
      )
      self.assertEqual(len(builder.group_modules), 1)
      self.assertEqual(len(builder.joint_modules), 0)

      builder.run()
      self.assertTrue((output_path / "picks.parquet").is_file())


class TestOGSCatalogBuilderMPI(unittest.TestCase):
  """Tests for OGSCatalogBuilderMPI GPU pinning and Dask-MPI bootstrapping."""

  @patch("dask.distributed.Client")
  @patch("dask_mpi.initialize")
  def test_mpi_bootstrap_and_gpu_pinning(
      self, mock_dask_mpi_init: MagicMock, mock_client_cls: MagicMock
  ) -> None:
    mock_client = MagicMock()
    mock_client.scheduler_info.return_value = {"workers": {"w1": {}, "w2": {}}}
    mock_client_cls.return_value = mock_client

    with tempfile.TemporaryDirectory() as tmpdir:
      output_path = Path(tmpdir) / "catalog"
      data = MockDataSource(["2024-01-01"])
      picker = MockPickerModule()

      builder = OGSCatalogBuilderMPI(
          output_path=output_path,
          data=data,
          group_modules=[picker],
          outputs=["picks"],
          formats=["parquet"],
      )

      # Simulate Leonardo Slurm environment with 4 tasks and local rank 6
      # (rank 6 % 4 = GPU 2)
      env_updates = {
          "SLURM_LOCALID": "6",
          "SLURM_NTASKS": "4",
          "SLURM_MEM_PER_CPU": "4000",
      }
      with patch.dict(os.environ, env_updates, clear=False):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        builder.run()

        # Check GPU pinning
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "2")

        # Check memory limit passed to dask_mpi.initialize
        mock_dask_mpi_init.assert_called_once_with(memory_limit="4000MB")

        # Check wait_for_workers called with n_workers = 4 - 2 = 2 and timeout=120
        mock_client.wait_for_workers.assert_called_once_with(
            n_workers=2, timeout=120
        )

        # Check client was cleanly shut down
        mock_client.shutdown.assert_called_once()

      # Artifact export should succeed
      self.assertTrue((output_path / "picks.parquet").is_file())


if __name__ == "__main__":
  unittest.main()


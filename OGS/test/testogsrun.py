"""
=============================================================================
OGS Run Test Suite - Unit Tests for CLI Parsing & Pipeline Dispatch
=============================================================================

OVERVIEW:
Unit tests verifying the universal CLI entrypoint (ogsrun.py):
  1. CLI argument and override parsing, including stripping '+' prefixes.
  2. MPI and Slurm execution environment detection.
  3. Fast index_only execution barrier bypassing pipeline execution.
  4. Dynamic dispatch to OGSCatalogBuilder or OGSCatalogBuilderMPI.
  5. End-to-end runner execution via main(argv).

AUTHORS:
  - 健
  - Istituto Nazionale di Oceanografia e di Geofisica Sperimentale (OGS)
=============================================================================
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml
from OGS.src import ogsconstants as OGS_C
from OGS.src.ogsbase import OGSStatus
from OGS.src.ogsbuilder import OGSCatalogBuilder
from OGS.src.ogsbuilderMPI import OGSCatalogBuilderMPI
from OGS.src.ogsrun import (
    build_parser,
    is_mpi_environment,
    main,
    parse_run_args,
    run_pipeline,
)


class TestCLIArgParsing(unittest.TestCase):
  """Tests for parse_run_args and option extraction."""

  def test_parse_flags_and_overrides(self) -> None:
    argv = [
        "--config-dir",
        "/tmp/custom_conf",
        "--config-name",
        "my_config",
        "+libpath=.",
        "+index_only=true",
        "output_path=catalogs/test",
        "group_modules=Picker",
    ]
    args, overrides = parse_run_args(argv)
    self.assertEqual(args.config_dir, "/tmp/custom_conf")
    self.assertEqual(args.config_name, "my_config")
    self.assertFalse(args.mpi)
    self.assertEqual(
        overrides,
        [
            "+libpath=.",
            "+index_only=true",
            "output_path=catalogs/test",
            "group_modules=Picker",
        ],
    )

  def test_parse_config_dir_equals_syntax(self) -> None:
    argv = [
        "--config-dir=/tmp/conf_eq",
        "--config-name=custom",
        "--mpi",
        "data.check=true",
    ]
    args, overrides = parse_run_args(argv)
    self.assertEqual(args.config_dir, "/tmp/conf_eq")
    self.assertEqual(args.config_name, "custom")
    self.assertTrue(args.mpi)
    self.assertEqual(overrides, ["data.check=true"])


class TestMPIEnvironmentDetection(unittest.TestCase):
  """Tests for is_mpi_environment under various cluster environments."""

  def test_serial_environment(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      self.assertFalse(is_mpi_environment())

  def test_slurm_multiple_tasks(self) -> None:
    with patch.dict(os.environ, {"SLURM_NTASKS": "4"}, clear=True):
      self.assertTrue(is_mpi_environment())

  def test_slurm_single_task(self) -> None:
    with patch.dict(os.environ, {"SLURM_NTASKS": "1"}, clear=True):
      self.assertFalse(is_mpi_environment())

  def test_openmpi_environment(self) -> None:
    with patch.dict(os.environ, {"OMPI_COMM_WORLD_RANK": "0"}, clear=True):
      self.assertTrue(is_mpi_environment())

  def test_pmix_environment(self) -> None:
    with patch.dict(os.environ, {"PMIX_RANK": "1"}, clear=True):
      self.assertTrue(is_mpi_environment())


class TestPipelineDispatch(unittest.TestCase):
  """Tests for run_pipeline execution, index_only barrier, and builder dispatch."""

  def setUp(self) -> None:
    self.tmpdir = tempfile.TemporaryDirectory()
    self.conf_dir = Path(self.tmpdir.name)
    # Write minimal config.yaml
    cfg = {
        "output_path": str(self.conf_dir / "out"),
        "builder": {
            "outputs": ["picks"],
            "formats": ["parquet"],
        },
    }
    with open(self.conf_dir / "config.yaml", "w") as f:
      yaml.dump(cfg, f)

  def tearDown(self) -> None:
    self.tmpdir.cleanup()

  @patch.object(OGSCatalogBuilder, "run")
  def test_serial_dispatch(self, mock_builder_run: MagicMock) -> None:
    mock_status = MagicMock(spec=OGGSStatus if "OGGSStatus" in globals() else OGSStatus)
    mock_builder_run.return_value = mock_status

    with patch("OGS.src.ogsrun.is_mpi_environment", return_value=False):
      status = run_pipeline(
          config_dir=self.conf_dir,
          config_name="config",
          overrides=["+extra_test=1"],
      )
      self.assertEqual(status, mock_status)
      mock_builder_run.assert_called_once()

  @patch.object(OGSCatalogBuilderMPI, "run")
  def test_mpi_dispatch(self, mock_mpi_builder_run: MagicMock) -> None:
    mock_status = MagicMock(spec=OGSStatus)
    mock_mpi_builder_run.return_value = mock_status

    with patch("OGS.src.ogsrun.is_mpi_environment", return_value=True):
      status = run_pipeline(
          config_dir=self.conf_dir,
          config_name="config",
          overrides=[],
      )
      self.assertEqual(status, mock_status)
      mock_mpi_builder_run.assert_called_once()

  def test_index_only_barrier(self) -> None:
    # When index_only=true, run_pipeline returns None without running builders
    with patch("OGS.src.ogsrun.OGSCatalogBuilder.run") as mock_run:
      res = run_pipeline(
          config_dir=self.conf_dir,
          config_name="config",
          overrides=["+index_only=true"],
      )
      self.assertIsNone(res)
      mock_run.assert_not_called()

  @patch.object(OGSCatalogBuilder, "run")
  def test_main_cli_entrypoint(self, mock_builder_run: MagicMock) -> None:
    mock_builder_run.return_value = MagicMock(spec=OGSStatus)
    argv = [
        f"--config-dir={self.conf_dir}",
        "--config-name=config",
        "+libpath=.",
    ]
    exit_code = main(argv)
    self.assertEqual(exit_code, 0)
    mock_builder_run.assert_called_once()


if __name__ == "__main__":
  unittest.main()


#!/usr/bin/env python3
"""
=============================================================================
Leonardo Workspace Smoke Test - Makefile Init Verification Suite
=============================================================================

OVERVIEW:
Smoke test suite executed by the Makefile 'init' target on the Leonardo HPC
cluster. Verifies that the initialized workspace in WORK_PATH contains the
required execution launchers, configuration files, and symbolic links (config,
data, src) back to the shared project tree.

Ensures that the compute environment is verified before submitting resource-
intensive multi-node SLURM jobs.

USAGE:
# Standalone test execution:
  python dummy.py

  # Invoked via Makefile:
  make init -C OGS/utils/Leonardo

  # Dispatched as a Slurm batch job:
  sbatch LAUNCHME.sh OGS/utils/Leonardo/test/dummy.py

DEPENDENCIES:
- unittest: standard library test framework
  - pathlib / os: workspace filesystem inspection

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

from __future__ import annotations

import os
from pathlib import Path
import unittest


class InitializedWorkspaceTests(unittest.TestCase):
  """Validate files and links prepared by the Makefile init target."""

  @classmethod
  def setUpClass(cls) -> None:
    # LAUNCHME.sh executes the payload after changing to WORK_PATH.
    cls.workspace = Path.cwd()

  def test_running_from_workspace(self) -> None:
    """Ensure the payload is running in the copied workspace."""
    self.assertTrue(
        (self.workspace / "Makefile").is_file(),
        f"Makefile is missing from workspace {self.workspace}",
    )

  def test_required_initialization_files_exist(self) -> None:
    """Check every regular file copied by the init target.

    Keeping this list explicit makes a missing launcher, configuration
    file, or helper script immediately visible when the smoke job runs.
    The test directory is checked separately because it is a directory,
    not a regular file.
    """
    required_files = (
        "ACTIVATEME.sh",
        "LAUNCHME.sh",
        "LEONARDO.yml",
        "Makefile",
        "download.sh",
        "dummy.sh",
        "ktanakah.sh",
    )
    for relative_path in required_files:
      with self.subTest(relative_path=relative_path):
        self.assertTrue(
            (self.workspace / relative_path).is_file(),
            f"required initialization file is missing: {relative_path}",
        )

    # The test directory and its payload must also survive the copy.
    self.assertTrue((self.workspace / "test").is_dir())
    self.assertTrue((self.workspace / "test/dummy.py").is_file())

  def test_shared_directories_are_symbolic_links(self) -> None:
    """Verify config, data, and src are links to the shared OGS tree."""
    expected_target_names = {"config": "conf", "data": "data", "src": "src"}
    for link_name, target_name in expected_target_names.items():
      with self.subTest(link_name=link_name):
        link = self.workspace / link_name
        self.assertTrue(link.is_symlink(),
                        f"{link_name} is not a symbolic link")
        self.assertTrue(
            link.exists(), f"{link_name} is a broken symbolic link")
        self.assertEqual(
            link.resolve().name,
            target_name,
            f"{link_name} does not resolve to a {target_name} directory",
        )
        self.assertTrue(link.resolve().is_dir())

  def test_external_repositories_are_git_checkouts(self) -> None:
    """Verify that init prepared both repositories beside WORK_PATH.

    The Makefile exposes these locations as NLL_PATH and HYPO71_PATH and
    passes them into this test through the environment. Checking the
    .git directory confirms that init did not merely create an empty
    placeholder directory or leave an unrelated directory in its place.
    """
    expected_paths = {
        "NonLinLoc": os.environ.get(
            "NLL_PATH", str(self.workspace.parent / "NonLinLoc")
        ),
        "bollettino_ogs_hypo71": os.environ.get(
            "HYPO71_PATH", str(self.workspace.parent / "bollettino_ogs_hypo71")
        ),
    }
    for repository_name, repository_path in expected_paths.items():
      with self.subTest(repository_name=repository_name):
        checkout = Path(repository_path)
        self.assertTrue(
            (checkout / ".git").exists(),
            f"{repository_name} is not a Git checkout: {checkout}",
        )

  def test_python_environment_is_usable(self) -> None:
    """Confirm that the interpreter can read the initialized payload."""
    self.assertTrue(os.access(self.workspace / "test/dummy.py", os.R_OK))


if __name__ == "__main__":
  # unittest returns a non-zero exit status on failure, it propagates failures
  # through LAUNCHME.sh and causes the SLURM smoke test to fail visibly.
  unittest.main(verbosity=2)

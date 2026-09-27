"""
=============================================================================
Unit & Integration Tests for OGSNonLinLoc & NonLinLoc Binaries
=============================================================================
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from OGS.src.ogsnonlinloc import OGSNonLinLoc
from ml_catalog.modules import NonLinLoc


NLL_BIN_DIR = Path("/leonardo_work/IscrC_AISeism/NonLinLoc/bin")


def test_binaries_installed():
    """Verify that all core NonLinLoc binaries exist and are executable."""
    required_binaries = ["NLLoc", "Vel2Grid", "Grid2Time", "LocSum"]
    for binary_name in required_binaries:
        bin_path = NLL_BIN_DIR / binary_name
        assert bin_path.is_file(), f"Binary {binary_name} not found in {NLL_BIN_DIR}"
        assert os.access(bin_path, os.X_OK), f"Binary {binary_name} is not executable"


def test_nlloc_invocation():
    """Verify that NLLoc executes cleanly and prints expected usage."""
    nlloc_bin = NLL_BIN_DIR / "NLLoc"
    res = subprocess.run([str(nlloc_bin)], capture_output=True, text=True)
    # NLLoc without arguments returns code 254 and prints usage
    assert "Usage: NLLoc <control file>" in res.stdout or "Usage: NLLoc <control file>" in res.stderr


def test_ogsnonlinloc_subclass():
    """Verify OGSNonLinLoc inheritance and initialization."""
    assert issubclass(OGSNonLinLoc, NonLinLoc)
    locator = OGSNonLinLoc(n_jobs=8, chunksize=25)
    assert locator.n_jobs == 8
    assert locator.chunksize == 25


def test_core_resolution_and_smt_clamping():
    """Verify core resolution handles SMT clamping and environment variables."""
    locator = OGSNonLinLoc(n_jobs=16)
    assert locator._resolve_n_jobs() == 16

    locator_auto = OGSNonLinLoc(n_jobs=None)

    with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "112"}, clear=False):
        assert locator_auto._resolve_n_jobs() == 112

    # Override LOKY_MAX_CPU_COUNT while popping SLURM_CPUS_PER_TASK
    clean_env = {k: v for k, v in os.environ.items() if k not in ("SLURM_CPUS_PER_TASK", "LOKY_MAX_CPU_COUNT")}
    clean_env["LOKY_MAX_CPU_COUNT"] = "32"
    with patch.dict(os.environ, clean_env, clear=True):
        assert locator_auto._resolve_n_jobs() == 32

    # Verify SMT clamping: 224 logical cores clamped to 112
    clean_env = {k: v for k, v in os.environ.items() if k not in ("SLURM_CPUS_PER_TASK", "LOKY_MAX_CPU_COUNT")}
    with patch.dict(os.environ, clean_env, clear=True):
        with patch("os.sched_getaffinity", return_value=set(range(224))):
            assert locator_auto._resolve_n_jobs() == 112

        # 64 logical cores clamped to 32
        with patch("os.sched_getaffinity", return_value=set(range(64))):
            assert locator_auto._resolve_n_jobs() == 32


def test_dynamic_chunksize():
    """Verify dynamic chunksize computation targeting ~4 chunks per worker."""
    locator = OGSNonLinLoc()

    # Small catalog: 100 events, 112 workers -> clamped to min 10
    assert locator._compute_dynamic_chunksize(100, 112) == 10

    # Medium catalog: 2,000 events, 10 workers -> ceil(2000 / 40) = 50
    assert locator._compute_dynamic_chunksize(2000, 10) == 50

    # Large catalog: 50,000 events, 10 workers -> ceil(50000 / 40) = 1250 -> clamped to 100
    assert locator._compute_dynamic_chunksize(50000, 10) == 100

    # Explicit user chunksize override respected
    locator_explicit = OGSNonLinLoc(chunksize=42)
    assert locator_explicit._compute_dynamic_chunksize(5000, 16) == 42


def test_shm_staging_atomic_flock():
    """Verify /dev/shm atomic travel-time staging with flock."""
    with tempfile.TemporaryDirectory() as temp_lustre:
        lustre_path = Path(temp_lustre)
        (lustre_path / "layer.P.STA01.time.buf").write_bytes(b"dummy_buf_bytes")
        (lustre_path / "layer.P.STA01.time.hdr").write_text("dummy_hdr_text")

        with tempfile.TemporaryDirectory() as temp_shm:
            shm_job_dir = Path(temp_shm)
            locator = OGSNonLinLoc()

            with patch.dict(os.environ, {"NLLOC_SHM_DIR": str(shm_job_dir)}):
                staged_path = locator._get_shm_time_path(lustre_path)
                assert staged_path == shm_job_dir / "time"
                assert (staged_path / "layer.P.STA01.time.buf").read_bytes() == b"dummy_buf_bytes"
                assert (staged_path / "layer.P.STA01.time.hdr").read_text() == "dummy_hdr_text"
                assert (staged_path / ".time_stage.done").is_file()

                # Second call should immediately reuse the staged directory
                reused_path = locator._get_shm_time_path(lustre_path)
                assert reused_path == staged_path


def test_end_to_end_sample_location():
    """Verify end-to-end relocation on sample Alaska 2018 event using NLLoc binary."""
    sample_dir = Path("/leonardo_work/IscrC_AISeism/NonLinLoc/nlloc_sample_test")
    if not (sample_dir / "run" / "nlloc_sample.in").is_file():
        pytest.skip("NonLinLoc sample directory not available")

    env = os.environ.copy()
    env["PATH"] = f"{NLL_BIN_DIR}:{env.get('PATH', '')}"

    with tempfile.TemporaryDirectory() as test_run_dir:
        test_path = Path(test_run_dir)
        # Copy sample run and obs files
        shutil.copytree(sample_dir / "run", test_path / "run")
        shutil.copytree(sample_dir / "obs", test_path / "obs")
        if (sample_dir / "data_geog").is_dir():
            shutil.copytree(sample_dir / "data_geog", test_path / "data_geog")

        (test_path / "model").mkdir()
        (test_path / "time").mkdir()
        (test_path / "loc").mkdir()

        # Step 1: Vel2Grid
        v2g = subprocess.run(
            [str(NLL_BIN_DIR / "Vel2Grid"), "run/nlloc_sample.in"],
            cwd=test_path,
            capture_output=True,
            text=True,
            env=env,
        )
        assert v2g.returncode == 0, f"Vel2Grid failed: {v2g.stderr}"
        assert (test_path / "model" / "layer.P.mod.hdr").is_file()

        # Step 2: Grid2Time
        g2t = subprocess.run(
            [str(NLL_BIN_DIR / "Grid2Time"), "run/nlloc_sample.in"],
            cwd=test_path,
            capture_output=True,
            text=True,
            env=env,
        )
        assert g2t.returncode == 0, f"Grid2Time failed: {g2t.stderr}"

        # Step 3: NLLoc
        nlloc = subprocess.run(
            [str(NLL_BIN_DIR / "NLLoc"), "run/nlloc_sample.in"],
            cwd=test_path,
            capture_output=True,
            text=True,
            env=env,
        )
        assert nlloc.returncode == 0, f"NLLoc failed: {nlloc.stderr}"
        assert "locations completed" in nlloc.stdout

        # Verify output hypocenter file
        hyp_files = list((test_path / "loc").glob("*.grid0.loc.hyp"))
        assert len(hyp_files) > 0, "No .loc.hyp hypocenter file generated"

        first_hyp = hyp_files[0].read_text()
        assert "GEOGRAPHIC" in first_hyp
        assert "HYPOCENTER" in first_hyp
        assert "STATISTICS" in first_hyp

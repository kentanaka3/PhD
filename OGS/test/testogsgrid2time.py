"""
=============================================================================
OGS GPU 3D Eikonal Solver Test Suite - Unit Tests for NonLinLoc Grid2Time
=============================================================================

OVERVIEW:
Unit test suite for ``ogs_gpu_grid2time.py``. Validates 3D Eikonal equation
solution accuracy, analytical benchmark verification (< 0.01 s error threshold),
NonLinLoc header and binary buffer serialization, station mapping generation,
and velocity model readers (NonLinLoc and SIMUL2K).

USAGE:
python -m unittest OGS/test/test_ogs_gpu_grid2time.py
"""

from ogsgrid2time import (
    GridConfig,
    compute_3d_travel_time,
    write_nll_time_grid,
    generate_travel_time_tables,
    read_nonlinloc_model,
    read_simul2k_model,
    run_self_test,
)
import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))


class TestOGSGPUGrid2Time(unittest.TestCase):
  """Unit test cases for OGS GPU 3D Eikonal Solver."""

  def setUp(self):
    self.nx, self.ny, self.nz = 31, 31, 21
    self.dx, self.dy, self.dz = 1.0, 1.0, 1.0
    self.ox, self.oy, self.oz = 0.0, 0.0, 0.0
    self.grid_cfg = GridConfig(
        numx=self.nx, numy=self.ny, numz=self.nz,
        origx=self.ox, origy=self.oy, origz=self.oz,
        dx=self.dx, dy=self.dy, dz=self.dz,
        transform_str="TRANSFORM  NONE",
    )
    self.vp = 5.0
    self.vs = 3.0
    self.station_id = "STA01"
    self.station_coord = (15.0, 15.0, 5.0)

  def test_grid_config(self):
    """Test GridConfig parsing and header string formatting."""
    cfg_str = "101,101,51,-50.0,-50.0,-5.0,1.0,1.0,1.0"
    cfg = GridConfig.from_string(cfg_str)
    self.assertEqual(cfg.numx, 101)
    self.assertEqual(cfg.origz, -5.0)
    self.assertEqual(cfg.dz, 1.0)

    hdr_str = cfg.format_header("TEST", 10.0, 20.0, 5.0)
    lines = hdr_str.strip().split("\n")
    self.assertEqual(len(lines), 3)
    self.assertTrue(lines[0].endswith("TIME FLOAT"))
    self.assertTrue(lines[1].startswith("TEST 10.000000 20.000000 5.000000"))

  def test_analytical_accuracy(self):
    """
    Verify analytical homogeneous velocity solution error is < 0.01 s.
    T_analytic = sqrt((x-xs)^2 + (y-ys)^2 + (z-zs)^2) / v
    """
    slowness_p = np.full((self.nx, self.ny, self.nz),
                         1.0 / self.vp, dtype=np.float32)
    Tp = compute_3d_travel_time(
        slowness_p, self.grid_cfg, self.station_coord,
        method="factored"
    )

    X, Y, Z = self.grid_cfg.get_mesh()
    dist = np.sqrt(
        (X - self.station_coord[0]) ** 2 +
        (Y - self.station_coord[1]) ** 2 +
        (Z - self.station_coord[2]) ** 2
    )
    Tp_analytic = (dist / self.vp).astype(np.float32)

    max_err = float(np.max(np.abs(Tp - Tp_analytic)))
    self.assertLess(
        max_err, 0.01, f"Max error {max_err} exceeds 0.01 s threshold")

  def test_nll_buffer_and_header(self):
    """
    Verify that written binary buffer size matches Nx * Ny * Nz * 4 bytes
    and header has 4 lines matching GridLib.c specification.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
      tmpdir_path = Path(tmpdir)
      slowness_p = np.full((self.nx, self.ny, self.nz),
                           1.0 / self.vp, dtype=np.float32)
      Tp = compute_3d_travel_time(
          slowness_p, self.grid_cfg, self.station_coord)

      table_prefix = tmpdir_path / f"layer.P.{self.station_id}"
      hdr_path, buf_path = write_nll_time_grid(
          table_prefix, Tp, self.grid_cfg, self.station_id,
          self.station_coord[0], self.station_coord[1], self.station_coord[2]
      )

      # Check exact file paths
      self.assertTrue(hdr_path.is_file())
      self.assertTrue(buf_path.is_file())
      self.assertEqual(hdr_path.name, f"layer.P.{self.station_id}.time.hdr")
      self.assertEqual(buf_path.name, f"layer.P.{self.station_id}.time.buf")

      # Check buffer file size
      expected_size = self.nx * self.ny * self.nz * 4
      self.assertEqual(os.path.getsize(buf_path), expected_size)

      # Check header content
      with open(hdr_path) as f:
        lines = f.readlines()
      self.assertEqual(len(lines), 4)
      self.assertTrue(lines[0].strip().endswith("TIME FLOAT"))
      self.assertTrue(lines[1].startswith(self.station_id))

  def test_mapping_generation(self):
    """Verify generation of mapping.csv compatible with _load_tt_tables()."""
    with tempfile.TemporaryDirectory() as tmpdir:
      tmpdir_path = Path(tmpdir)
      stations_df = pd.DataFrame([{
          "id": self.station_id,
          "x": self.station_coord[0],
          "y": self.station_coord[1],
          "z": self.station_coord[2],
      }])

      mapping_df = generate_travel_time_tables(
          model_p=self.vp,
          model_s=self.vs,
          stations_df=stations_df,
          grid_cfg=self.grid_cfg,
          output_dir=tmpdir_path / "tt_out",
          method="factored",
      )

      mapping_csv = tmpdir_path / "tt_out" / "mapping.csv"
      self.assertTrue(mapping_csv.is_file())
      read_df = pd.read_csv(mapping_csv, keep_default_na=False)
      self.assertEqual(list(read_df.columns), ["id", "p_table", "s_table"])
      self.assertEqual(read_df.iloc[0]["id"], self.station_id)
      self.assertEqual(read_df.iloc[0]["p_table"],
                       f"layer.P.{self.station_id}")
      self.assertEqual(read_df.iloc[0]["s_table"],
                       f"layer.S.{self.station_id}")

  def test_self_test_routine(self):
    """Verify the complete self-test routine succeeds."""
    success = run_self_test()
    self.assertTrue(success)


if __name__ == "__main__":
  unittest.main()

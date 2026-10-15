"""
=============================================================================
OGS Pipeline Test Suite - Makefile Target & Execution Verification
=============================================================================

OVERVIEW:
Unit tests validating pipeline execution corresponding to the macros defined
in OGS/utils/Leonardo/Makefile:
  1. PICKER_RULES:
     Validates 'make PhaseNet[INSTANCE,0.1] YEAR=2024' execution behavior
     (without SLURM job submission), evaluating pretrained PhaseNet on real
     triangulated waveforms from March 27, 2024 with Wood-Anderson amplitudes
     and tabular artifact export (picks.parquet, picks.csv).
  2. Future macros (ASSOCIATOR_RULES, NLL_RULES) reserved for upcoming
     stage integration.

AUTHORS:
  - 健
  - Istituto Nazionale di Oceanografia e di Geofisica Sperimentale (OGS)
=============================================================================
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from OGS.src import ogsconstants as OGS_C
from OGS.src.ogsbase import OGSStatus
from OGS.src.ogsrun import run_pipeline


class TestOGSPipeline(unittest.TestCase):
  """Tests for Makefile-level pipeline rule execution."""

  def setUp(self) -> None:
    self.test_dir = Path(__file__).resolve().parent
    self.config_dir = self.test_dir.parent / "conf"
    self.waveform_dir = self.test_dir / "data" / "waveform"
    self.station_dir = self.test_dir / "data" / "station"

  def test_picker_rules_phasenet_instance(self) -> None:
    """
    Test PhaseNet[INSTANCE,0.1] YEAR=2024 execution outcome without Slurm.
    Corresponds to _PICKER_RULES in OGS/utils/Leonardo/Makefile.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
      out_dir = (
          Path(tmpdir)
          / "catalogs"
          / "Leonardo_Dev"
          / "PhaseNet"
          / "instance"
          / "0.1"
          / "2024"
      )

      # Dynamic execution isolation overrides; all core parameters are
      # established in OGS/test/data/conf/test_phasenet.yaml
      overrides = [
          f"output_path={out_dir}",
          f"data.env={tmpdir}/sq_env",
      ]

      status = run_pipeline(
          config_dir=self.conf_dir,
          config_name="test_phasenet",
          overrides=overrides,
          force_mpi=False,
      )

      self.assertIsInstance(status, OGSStatus)

      # 1. Verify artifact persistence
      picks_pq = out_dir / "picks.parquet"
      picks_csv = out_dir / "picks.csv"
      self.assertTrue(picks_pq.is_file(), f"Missing {picks_pq}")
      self.assertTrue(picks_csv.is_file(), f"Missing {picks_csv}")

      # 2. Verify schema and content
      df_picks = pd.read_parquet(picks_pq)
      self.assertGreater(
          len(df_picks), 300, f"Expected >300 picks, got {len(df_picks)}"
      )

      required_cols = [
          OGS_C.IDX_PICKS_STR,
          OGS_C.STATION_STR,
          OGS_C.PHASE_STR,
          OGS_C.TIME_STR,
          OGS_C.AMPLITUDE_STR,
      ]
      for col in required_cols:
        self.assertIn(col, df_picks.columns)

      # Continuous index verification
      self.assertEqual(
          list(df_picks[OGS_C.IDX_PICKS_STR]),
          list(range(len(df_picks))),
      )

      # Station names conformance (NET.STA)
      stations = set(df_picks[OGS_C.STATION_STR])
      for sta in stations:
        self.assertIn(".", sta)
      self.assertTrue(
          {"HA.INVI", "NI.PURA", "OX.CLUD", "OX.COLI"}.issubset(stations)
      )

      # Phase label verification
      phases = set(df_picks[OGS_C.PHASE_STR])
      self.assertEqual(phases, {"P", "S"})

      # Positive amplitudes verification for successfully evaluated windows
      valid_amps = df_picks[OGS_C.AMPLITUDE_STR].dropna()
      self.assertGreater(len(valid_amps), 0)
      self.assertTrue((valid_amps > 0).all())


if __name__ == "__main__":
  unittest.main()

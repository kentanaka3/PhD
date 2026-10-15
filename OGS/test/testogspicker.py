"""
=============================================================================
OGS Picker Test Suite - Unit Tests for Amplitude Extractor & SeisBench Picker
=============================================================================

OVERVIEW:
Unit tests verifying decoupled ML picking and Wood-Anderson amplitude extraction:
  1. OGSAmplitudeExtractor: Wood-Anderson simulation, instrument deconvolution,
     1-40 Hz bandpass filtering, and noise/signal SNR extraction.
  2. In-thread vs. parallel (thread-pool) amplitude extraction equivalence.
  3. OGSSeisBenchPicker: End-to-end execution with real pretrained PhaseNet
     weights ("instance") on synthetic 3-component ObsPy streams.
  4. Output schema validation: Station name normalization ('NET.STA'), continuous
     0..N-1 indexing in IDX_PICKS_STR, and OGS_C column conformance.
  5. DAG integration with OGSCatalogBuilder for artifact export.

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

import dask
import numpy as np
import obspy
from obspy import Stream, Trace, UTCDateTime
from obspy.core.inventory import (
    Channel,
    InstrumentSensitivity,
    Inventory,
    Network,
    PolesZerosResponseStage,
    Response,
    Station,
)
import pandas as pd
import seisbench.models as sbm
import seisbench.util as sbu

from OGS.src import ogsconstants as OGS_C
from OGS.src.ogsbase import OGSGroupStatus, OGSStatus
from OGS.src.ogsbuilder import OGSCatalogBuilder
from OGS.src.ogspicker import (
    OGSAmplitudeExtractor,
    OGSSeisBenchPicker,
    OGS_WOOD_ANDERSON,
)


def _create_synthetic_inventory(
    network: str = "OX",
    station: str = "TEST",
    channels: tuple[str, ...] = ("HHZ", "HHN", "HHE"),
) -> Inventory:
  """Create a synthetic ObsPy Inventory with valid PAZ response stages."""
  resp = Response()
  paz = PolesZerosResponseStage(
      stage_sequence_number=1,
      stage_gain=1.0,
      stage_gain_frequency=1.0,
      input_units="M/S",
      output_units="V",
      pz_transfer_function_type="LAPLACE (RADIANS/SECOND)",
      normalization_frequency=1.0,
      zeros=[0j, 0j],
      poles=[-0.037 - 0.037j, -0.037 + 0.037j],
      normalization_factor=1.0,
  )
  resp.response_stages.append(paz)
  resp.instrument_sensitivity = InstrumentSensitivity(
      value=1.0, frequency=1.0, input_units="M/S", output_units="V"
  )

  chan_objs = [
      Channel(
          code=ch,
          location_code="",
          latitude=46.0,
          longitude=13.0,
          elevation=100.0,
          depth=0.0,
          response=resp,
      )
      for ch in channels
  ]
  sta_obj = Station(
      code=station,
      latitude=46.0,
      longitude=13.0,
      elevation=100.0,
      channels=chan_objs,
  )
  net_obj = Network(code=network, stations=[sta_obj])
  return Inventory(networks=[net_obj])


def _create_synthetic_stream(
    network: str = "OX",
    station: str = "TEST",
    starttime: UTCDateTime = UTCDateTime("2024-01-01T00:00:00"),
    duration_sec: float = 60.0,
    sampling_rate: float = 100.0,
    with_earthquake: bool = True,
) -> Stream:
  """
  Create a 3-component synthetic stream with background noise and optional
  earthquake pulses.
  """
  n_pts = int(duration_sec * sampling_rate)
  rng = np.random.default_rng(42)

  data_z = rng.normal(0, 0.01, n_pts)
  data_n = rng.normal(0, 0.01, n_pts)
  data_e = rng.normal(0, 0.01, n_pts)

  if with_earthquake:
    # Inject P-wave pulse at t = 20s (sample 2000)
    p_idx = int(20.0 * sampling_rate)
    t_p = np.arange(int(5.0 * sampling_rate)) / sampling_rate
    p_wave = np.sin(2 * np.pi * 10 * t_p) * np.exp(-t_p * 2) * 5.0
    data_z[p_idx : p_idx + len(p_wave)] += p_wave

    # Inject S-wave pulse at t = 28s (sample 2800)
    s_idx = int(28.0 * sampling_rate)
    t_s = np.arange(int(8.0 * sampling_rate)) / sampling_rate
    s_wave = np.sin(2 * np.pi * 4 * t_s) * np.exp(-t_s * 1.5) * 8.0
    data_n[s_idx : s_idx + len(s_wave)] += s_wave
    data_e[s_idx : s_idx + len(s_wave)] += s_wave

  traces = [
      Trace(
          data=data_z,
          header={
              "network": network,
              "station": station,
              "channel": "HHZ",
              "sampling_rate": sampling_rate,
              "starttime": starttime,
          },
      ),
      Trace(
          data=data_n,
          header={
              "network": network,
              "station": station,
              "channel": "HHN",
              "sampling_rate": sampling_rate,
              "starttime": starttime,
          },
      ),
      Trace(
          data=data_e,
          header={
              "network": network,
              "station": station,
              "channel": "HHE",
              "sampling_rate": sampling_rate,
              "starttime": starttime,
          },
      ),
  ]
  return Stream(traces=traces)


class TestOGSAmplitudeExtractor(unittest.TestCase):
  """Unit tests for OGSAmplitudeExtractor."""

  def setUp(self) -> None:
    self.inv = _create_synthetic_inventory("OX", "TEST")
    self.st = _create_synthetic_stream("OX", "TEST", with_earthquake=True)
    t0 = UTCDateTime("2024-01-01T00:00:00")
    self.pick = sbu.Pick(
        trace_id="OX.TEST..HHZ",
        start_time=t0 + 19.5,
        end_time=t0 + 20.5,
        peak_time=t0 + 20.0,
        peak_value=0.9,
        phase="P",
    )

  def test_amplitude_extraction_in_thread(self) -> None:
    extractor = OGSAmplitudeExtractor(
        time_before=2.0,
        time_after=5.0,
        slack=5.0,
        parallel=False,
    )
    amps = extractor.extract_amplitudes(self.st, [self.pick], self.inv)
    self.assertEqual(len(amps), 1)
    res = amps[0]

    # Verify peak amplitude and components
    self.assertIn("amplitude", res)
    self.assertFalse(np.isnan(res["amplitude"]))
    self.assertGreater(res["amplitude"], 0.0)

    # Verify horizontal components N and E
    self.assertIn("amplitude_N", res)
    self.assertIn("amplitude_E", res)
    self.assertGreater(res["amplitude_N"], 0.0)
    self.assertGreater(res["amplitude_E"], 0.0)

    # Verify SNR metrics
    self.assertIn("snr_N", res)
    self.assertIn("snr_E", res)
    self.assertGreater(res["snr_N"], 0.0)
    self.assertGreater(res["snr_E"], 0.0)

  def test_amplitude_extraction_parallel_threadpool(self) -> None:
    extractor_serial = OGSAmplitudeExtractor(parallel=False)
    extractor_parallel = OGSAmplitudeExtractor(parallel=True, n_jobs=2)

    res_serial = extractor_serial.extract_amplitudes(
        self.st, [self.pick], self.inv
    )
    res_parallel = extractor_parallel.extract_amplitudes(
        self.st, [self.pick], self.inv
    )

    self.assertAlmostEqual(
        res_serial[0]["amplitude"], res_parallel[0]["amplitude"], places=5
    )

  def test_missing_inventory_graceful_handling(self) -> None:
    extractor = OGSAmplitudeExtractor()
    amps = extractor.extract_amplitudes(self.st, [self.pick], inventory=None)
    self.assertEqual(len(amps), 1)
    self.assertTrue(np.isnan(amps[0]["amplitude"]))


class TestOGSSeisBenchPicker(unittest.TestCase):
  """Unit tests for OGSSeisBenchPicker with real pretrained PhaseNet."""

  @classmethod
  def setUpClass(cls) -> None:
    cls.model = sbm.PhaseNet.from_pretrained("instance")

  def test_seisbench_picker_execution_and_schema_conformance(self) -> None:
    st = _create_synthetic_stream("OX", "TEST", with_earthquake=True)
    inv = _create_synthetic_inventory("OX", "TEST")
    extractor = OGSAmplitudeExtractor(
        time_before=2.0, time_after=5.0, slack=5.0
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      status = OGSGroupStatus(group="2024-01-01", output_path=tmpdir)
      status.set_param(lambda: iter([st]), "data_func")
      status.set_param(inv, "inventory")

      picker = OGSSeisBenchPicker(
          model=self.model,
          classify_args={"P_threshold": 0.1, "S_threshold": 0.1},
          amplitude_extractor=extractor,
      )
      self.assertEqual(picker.output_keys(), ["picks"])
      self.assertTrue(len(picker.citations()) > 0)
      self.assertIn("seisbench", picker.version())

      picker.run(status)

      # Evaluate delayed Dask graph
      delayed_picks = status.get_param("picks")
      self.assertTrue(dask.is_dask_collection(delayed_picks))
      df_picks = dask.compute(delayed_picks)[0]

      self.assertIsInstance(df_picks, pd.DataFrame)
      self.assertGreater(len(df_picks), 0)

      # Verify OGS_C column schemas
      expected_cols = [
          OGS_C.IDX_PICKS_STR,
          OGS_C.STATION_STR,
          OGS_C.TIME_STR,
          OGS_C.PROBABILITY_STR,
          OGS_C.PHASE_STR,
          OGS_C.AMPLITUDE_STR,
          OGS_C.GROUPS_STR,
      ]
      for col in expected_cols:
        self.assertIn(col, df_picks.columns)

      # Verify station normalization to 'NET.STA'
      for sta in df_picks[OGS_C.STATION_STR]:
        self.assertEqual(sta, "OX.TEST")

      # Verify continuous 0..N-1 index
      self.assertEqual(
          list(df_picks[OGS_C.IDX_PICKS_STR]),
          list(range(len(df_picks))),
      )

      # Verify group column
      for g in df_picks[OGS_C.GROUPS_STR]:
        self.assertEqual(g, "2024-01-01")

      # Verify amplitudes are computed and positive
      self.assertTrue((df_picks[OGS_C.AMPLITUDE_STR] > 0).all())

  def test_seisbench_picker_with_catalog_builder_dag(self) -> None:
    st = _create_synthetic_stream("OX", "TEST", with_earthquake=True)
    inv = _create_synthetic_inventory("OX", "TEST")
    extractor = OGSAmplitudeExtractor(
        time_before=2.0, time_after=5.0, slack=5.0
    )

    class SyntheticDataSource:

      def groups(self) -> list[str]:
        return ["2024-01-01"]

      def get_group(self, group: str) -> Any:
        return iter([st])

      def get_inventory(self) -> Inventory:
        return inv

    with tempfile.TemporaryDirectory() as tmpdir:
      output_path = Path(tmpdir) / "catalog"
      data = SyntheticDataSource()
      picker = OGSSeisBenchPicker(
          model=self.model,
          classify_args={"P_threshold": 0.1, "S_threshold": 0.1},
          amplitude_extractor=extractor,
      )

      builder = OGSCatalogBuilder(
          output_path=output_path,
          data=data,
          group_modules=[picker],
          outputs=["picks", "assignments"],
          formats=["parquet", "csv"],
      )

      final_status = builder.run()
      self.assertIsInstance(final_status, OGSStatus)

      picks_pq = output_path / "picks.parquet"
      picks_csv = output_path / "picks.csv"
      assign_pq = output_path / "assignments.parquet"
      assign_csv = output_path / "assignments.csv"

      self.assertTrue(picks_pq.is_file())
      self.assertTrue(picks_csv.is_file())
      self.assertTrue(assign_pq.is_file())
      self.assertTrue(assign_csv.is_file())

      df = pd.read_parquet(picks_pq)
      self.assertGreater(len(df), 0)
      self.assertEqual(list(df[OGS_C.IDX_PICKS_STR]), list(range(len(df))))
      self.assertTrue(all(sta == "OX.TEST" for sta in df[OGS_C.STATION_STR]))


if __name__ == "__main__":
  unittest.main()


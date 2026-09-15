"""
=============================================================================
OGS Trainer Test Suite - Unit Tests for Phase Picking Training & CLI
=============================================================================

OVERVIEW:
Unit test suite for ``ogstrainer.py``. Validates CLI argument parsing, vector
cross-entropy loss calculation, station and event metadata extraction, waveform
trace signal conditioning, SeisBench data augmentation pipeline construction,
checkpoint persistence, and training/validation epoch loops.

TEST CASES & INVARIANTS:
  1. CLI Argument Parsing (TestOGSTrainerCLI):
     - Validates default parameters (catalog, waveforms, model, dataset, batch size).
     - Validates CLI parameter overrides and mutual exclusion.
     - Enforces choices validation on unsupported model architectures and weights.
  2. Loss Function (TestOGSTrainerLossFn):
     - Validates vector cross-entropy on synthetic probability distributions.
     - Verifies numerical stability with probability values near zero (eps=1e-5).
     - Confirms gradient propagation and finite gradients under backward pass.
  3. Station Parsing (TestOGSTrainerStationParsing):
     - Validates network, station, and location parsing for 1-, 2-, and 3-part dotted codes.
  4. Event Parameter Extraction (TestOGSTrainerEventParams):
     - Validates canonical origin parameters extraction from pandas Series.
     - Verifies correct handling of missing or NaN magnitude values.
  5. Trace Signal Conditioning (TestOGSTrainerCleanTrace):
     - Validates detrend, demean, cosine tapering, bandpass filtering, and resampling.
  6. Augmentation Pipeline Builder (TestOGSTrainerAugmentations):
     - Validates window lengths and augmentation chain for PhaseNet (3001) and EQTransformer (6000).
  7. Checkpointing (TestOGSTrainerCheckpoint):
     - Verifies checkpoint serialization, dictionary schema, and state restoration.
  8. Training & Validation Loops (TestOGSTrainerLoops):
     - Validates forward pass, loss calculation, backpropagation, optimizer stepping,
       and train/eval mode toggling.
  9. Lifecycle & Downloader Dispatch (TestOGSTrainerLifecycle):
     - Validates trainer initialization, output directory scaffolding, and download command dispatch.

USAGE:
python -m unittest OGS/test/testogstrainer.py

DEPENDENCIES:
- unittest: standard library testing framework
- numpy / pandas / obspy / torch: scientific and deep learning libraries
- seisbench: seismic deep-learning framework and model abstractions
- ogstrainer / ogsconstants: module under test and constants

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

from ogstrainer import OGSTrainer, loss_fn, parse_arguments
import ogstrainer
import ogsconstants as OGS_C
import seisbench.models as sbm
import seisbench.generate as sbg
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
import obspy as op
from pathlib import Path
from datetime import datetime
import numpy as np
import argparse
import ctypes
import os
import sys
import tempfile
import unittest
import unittest.mock

# Preload Conda environment libstdc++.so.6 if present to avoid GLIBCXX version mismatch
_conda_prefix = os.path.dirname(os.path.dirname(sys.executable))
_libstdcxx = os.path.join(_conda_prefix, "lib", "libstdc++.so.6")
if os.path.exists(_libstdcxx):
  try:
    ctypes.CDLL(_libstdcxx, mode=ctypes.RTLD_GLOBAL)
  except Exception:
    pass


# NumPy 2.0+ compatibility: restore np.trapz for xdas / seisbench
if not hasattr(np, "trapz"):
  np.trapz = getattr(np, "trapezoid", None)


THIS_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(THIS_DIR + "/../src"))


# -----------------------------------------------------------------------------
# Test Helpers & Synthetic Fixtures
# -----------------------------------------------------------------------------

class SyntheticPickerDataset(Dataset):
  """Synthetic dataset providing batch tensors matching SeisBench output."""

  def __init__(self, num_samples=4, windowlen=3001):
    self.num_samples = num_samples
    self.windowlen = windowlen

  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    # Shape: (3 channels, windowlen samples)
    x = torch.randn(3, self.windowlen, dtype=torch.float32)
    # Shape: (3 pick classes, windowlen samples) with normalized class probabilities
    y = torch.softmax(torch.randn(
        3, self.windowlen, dtype=torch.float32), dim=0)
    return {"X": x, "y": y}


class SyntheticPickerModel(torch.nn.Module):
  """Minimal neural network mimicking PhaseNet / EQTransformer API for tests."""

  def __init__(self, in_channels=3, out_classes=3):
    super().__init__()
    self.device = torch.device("cpu")
    self.conv = torch.nn.Conv1d(in_channels, out_classes, kernel_size=1)
    self.labels = ["P", "S", "N"]

  def annotate_batch_pre(self, x, _):
    return x

  def forward(self, x):
    return torch.softmax(self.conv(x), dim=1)


# -----------------------------------------------------------------------------
# Test Suites
# -----------------------------------------------------------------------------

class TestOGSTrainerCLI(unittest.TestCase):
  """Unit tests for CLI argument parsing in ogstrainer.py."""

  def test_cli_defaults(self):
    """Verifies default arguments when only required arguments are provided."""
    test_argv = [
        "ogstrainer.py",
        "-C", "/path/to/catalog",
        "-W", "/path/to/waveforms",
    ]
    with unittest.mock.patch("sys.argv", test_argv):
      args = parse_arguments()

    self.assertEqual(args.catalog, Path("/path/to/catalog"))
    self.assertEqual(args.waveforms, Path("/path/to/waveforms"))
    self.assertEqual(args.batch_size, 256)
    self.assertEqual(args.epochs, 5)
    self.assertEqual(args.learning_rate, 1e-2)
    self.assertEqual(args.workers, 4)
    self.assertEqual(args.model, OGS_C.PHASENET_STR)
    self.assertEqual(args.dataset, OGS_C.INSTANCE_STR)
    self.assertEqual(args.output, Path("./checkpoints"))
    self.assertFalse(args.download)
    self.assertEqual(
        args.dates,
        [
            datetime.strptime("20240320", OGS_C.YYYYMMDD_FMT),
            datetime.strptime("20240620", OGS_C.YYYYMMDD_FMT),
        ]
    )

  def test_cli_missing_required(self):
    """Verifies that missing catalog or waveforms raises SystemExit."""
    # Missing both
    with unittest.mock.patch("sys.argv", ["ogstrainer.py"]), \
            self.assertRaises(SystemExit):
      parse_arguments()

    # Missing waveforms
    with unittest.mock.patch("sys.argv", ["ogstrainer.py", "-C", "/path/to/cat"]), \
            self.assertRaises(SystemExit):
      parse_arguments()

    # Missing catalog
    with unittest.mock.patch("sys.argv", ["ogstrainer.py", "-W", "/path/to/wf"]), \
            self.assertRaises(SystemExit):
      parse_arguments()

  def test_cli_custom_overrides(self):
    """Verifies that custom command-line flags properly override defaults."""
    test_argv = [
        "ogstrainer.py",
        "-C", "/custom/catalog",
        "-W", "/custom/waveforms",
        "-m", "EQTransformer",
        "-s", "stead",
        "-b", "64",
        "-e", "10",
        "-lr", "0.001",
        "-w", "2",
        "-o", "/tmp/test_ckpt",
        "-d",
        "-D", "20240101", "20240115",
    ]
    with unittest.mock.patch("sys.argv", test_argv):
      args = parse_arguments()

    self.assertEqual(args.catalog, Path("/custom/catalog"))
    self.assertEqual(args.waveforms, Path("/custom/waveforms"))
    self.assertEqual(args.model, "EQTransformer")
    self.assertEqual(args.dataset, "stead")
    self.assertEqual(args.batch_size, 64)
    self.assertEqual(args.epochs, 10)
    self.assertEqual(args.learning_rate, 0.001)
    self.assertEqual(args.workers, 2)
    self.assertEqual(args.output, Path("/tmp/test_ckpt"))
    self.assertTrue(args.download)
    self.assertEqual(
        args.dates,
        [
            datetime.strptime("20240101", OGS_C.YYYYMMDD_FMT),
            datetime.strptime("20240115", OGS_C.YYYYMMDD_FMT),
        ]
    )

  def test_cli_choices_validation(self):
    """Verifies that unsupported model architectures or datasets are rejected."""
    # Invalid model
    invalid_model_argv = [
        "ogstrainer.py",
        "-C", "/path/cat",
        "-W", "/path/wf",
        "-m", "UnsupportedPickerModel",
    ]
    with unittest.mock.patch("sys.argv", invalid_model_argv), \
            self.assertRaises(SystemExit):
      parse_arguments()

    # Invalid dataset
    invalid_dataset_argv = [
        "ogstrainer.py",
        "-C", "/path/cat",
        "-W", "/path/wf",
        "-s", "non_existent_dataset",
    ]
    with unittest.mock.patch("sys.argv", invalid_dataset_argv), \
            self.assertRaises(SystemExit):
      parse_arguments()

  def test_cli_all_valid_choices(self):
    """Verifies that all registered models and datasets are accepted."""
    for model_name in [OGS_C.PHASENET_STR, OGS_C.EQTRANSFORMER_STR]:
      for ds_name in [
          OGS_C.INSTANCE_STR, OGS_C.STEAD_STR, OGS_C.SCEDC_STR,
          OGS_C.ORIGINAL_STR, OGS_C.ADRIAARRAY_STR
      ]:
        argv = [
            "ogstrainer.py",
            "-C", "/cat", "-W", "/wf",
            "-m", model_name,
            "-s", ds_name,
        ]
        with unittest.mock.patch("sys.argv", argv):
          args = parse_arguments()
          self.assertEqual(args.model, model_name)
          self.assertEqual(args.dataset, ds_name)


class TestOGSTrainerLossFn(unittest.TestCase):
  """Unit tests for the vector cross-entropy loss function."""

  def test_loss_fn_synthetic_distribution(self):
    """Validates vector cross-entropy on synthetic probability distributions."""
    batch_size, classes, time_steps = 2, 3, 100
    # True one-hot distribution
    y_true = torch.zeros(batch_size, classes, time_steps, dtype=torch.float32)
    y_true[:, 0, :] = 1.0

    # Perfect prediction matches true distribution
    y_pred_perfect = y_true.clone()
    loss_perfect = loss_fn(y_pred_perfect, y_true, eps=1e-5)

    self.assertIsInstance(loss_perfect, torch.Tensor)
    self.assertEqual(loss_perfect.shape, torch.Size([]))
    self.assertGreaterEqual(loss_perfect.item(), 0.0)
    # -log(1.0 + 1e-5) is ~ -1e-5; with negation it is >= 0 and very small
    self.assertAlmostEqual(loss_perfect.item(), 0.0, places=4)

    # Imperfect prediction has higher loss
    y_pred_uniform = torch.ones(
        batch_size, classes, time_steps, dtype=torch.float32) / 3.0
    loss_imperfect = loss_fn(y_pred_uniform, y_true, eps=1e-5)
    self.assertGreater(loss_imperfect.item(), loss_perfect.item())

  def test_loss_fn_numerical_stability_near_zero(self):
    """Verifies non-negativity and stability when prediction is near zero."""
    batch_size, classes, time_steps = 2, 3, 100
    y_true = torch.ones(batch_size, classes, time_steps,
                        dtype=torch.float32) / 3.0
    y_pred_zero = torch.zeros(
        batch_size, classes, time_steps, dtype=torch.float32)

    loss = loss_fn(y_pred_zero, y_true, eps=1e-5)

    self.assertTrue(torch.isfinite(loss).item())
    self.assertGreater(loss.item(), 0.0)
    # Expected value: -log(1e-5) ~ 11.5129
    expected = -np.log(1e-5)
    self.assertAlmostEqual(loss.item(), expected, places=3)

  def test_loss_fn_backpropagation(self):
    """Verifies that loss.backward() produces finite gradients on model parameters."""
    param = torch.nn.Parameter(torch.randn(2, 3, 50, dtype=torch.float32))
    y_pred = torch.softmax(param, dim=1)
    y_true = torch.softmax(torch.randn(2, 3, 50, dtype=torch.float32), dim=1)

    loss = loss_fn(y_pred, y_true)
    loss.backward()

    self.assertIsNotNone(param.grad)
    self.assertTrue(torch.isfinite(param.grad).all().item())
    self.assertEqual(param.grad.shape, param.shape)


class TestOGSTrainerStationParsing(unittest.TestCase):
  """Unit tests for get_station_info dotted network-station-location parsing."""

  def setUp(self):
    self.trainer = OGSTrainer.__new__(OGSTrainer)

  def test_three_part_station(self):
    """Verifies network.station.location extraction for 3-part dotted codes."""
    net, sta, loc = self.trainer.get_station_info("OX.TRI.00")
    self.assertEqual(net, "OX")
    self.assertEqual(sta, "TRI")
    self.assertEqual(loc, "00")

  def test_two_part_station(self):
    """Verifies network.station extraction with empty location for 2-part dotted codes."""
    net, sta, loc = self.trainer.get_station_info("IV.ACOM")
    self.assertEqual(net, "IV")
    self.assertEqual(sta, "ACOM")
    self.assertEqual(loc, "")

  def test_single_part_station(self):
    """Verifies single station name parsing with empty network and location."""
    net, sta, loc = self.trainer.get_station_info("BOB")
    self.assertEqual(net, "")
    self.assertEqual(sta, "BOB")
    self.assertEqual(loc, "")

  def test_multipart_station(self):
    """Verifies multipart station strings take the first 3 tokens."""
    net, sta, loc = self.trainer.get_station_info("OX.TRI.00.01")
    self.assertEqual(net, "OX")
    self.assertEqual(sta, "TRI")
    self.assertEqual(loc, "00")

  def test_empty_station(self):
    """Verifies empty string parsing returns empty strings."""
    net, sta, loc = self.trainer.get_station_info("")
    self.assertEqual(net, "")
    self.assertEqual(sta, "")
    self.assertEqual(loc, "")


class TestOGSTrainerEventParams(unittest.TestCase):
  """Unit tests for event origin parameter extraction."""

  def setUp(self):
    self.trainer = OGSTrainer.__new__(OGSTrainer)

  def test_canonical_event_params(self):
    """Verifies parameter extraction with valid magnitude and origin fields."""
    event_row = pd.Series({
        OGS_C.TIME_STR: "2024-03-20T12:34:56.789000Z",
        OGS_C.LATITUDE_STR: 46.1234,
        OGS_C.LONGITUDE_STR: 13.5678,
        OGS_C.DEPTH_STR: 12.5,
        OGS_C.MAGNITUDE_L_STR: 2.7,
    })

    params = self.trainer.get_event_params(event_row)

    self.assertEqual(params["source_origin_time"],
                     "2024-03-20T12:34:56.789000Z")
    self.assertEqual(params["source_latitude_deg"], 46.1234)
    self.assertEqual(params["source_longitude_deg"], 13.5678)
    self.assertEqual(params["source_depth_km"], 12.5)
    self.assertEqual(params["source_magnitude"], 2.7)
    self.assertEqual(params["source_magnitude_type"], "ML")

  def test_nan_magnitude_event_params(self):
    """Verifies that NaN or missing magnitude omits source_magnitude fields."""
    event_row = pd.Series({
        OGS_C.TIME_STR: "2024-03-20T12:34:56.789000Z",
        OGS_C.LATITUDE_STR: 46.1234,
        OGS_C.LONGITUDE_STR: 13.5678,
        OGS_C.DEPTH_STR: 12.5,
        OGS_C.MAGNITUDE_L_STR: np.nan,
    })

    params = self.trainer.get_event_params(event_row)

    self.assertNotIn("source_magnitude", params)
    self.assertNotIn("source_magnitude_type", params)
    self.assertEqual(params["source_depth_km"], 12.5)

  def test_missing_keys_event_params(self):
    """Verifies behavior when origin fields are absent in the Series."""
    event_row = pd.Series({})
    params = self.trainer.get_event_params(event_row)

    self.assertEqual(params["source_origin_time"], "")
    self.assertTrue(np.isnan(params["source_latitude_deg"]))
    self.assertTrue(np.isnan(params["source_longitude_deg"]))
    self.assertTrue(np.isnan(params["source_depth_km"]))
    self.assertNotIn("source_magnitude", params)


class TestOGSTrainerCleanTrace(unittest.TestCase):
  """Unit tests for waveform signal conditioning."""

  def setUp(self):
    self.trainer = OGSTrainer.__new__(OGSTrainer)

  def test_clean_trace_signal_conditioning(self):
    """Verifies detrend, demean, taper, filter, and resampling on synthetic Trace."""
    # Synthetic trace with linear drift and non-zero mean at 50 Hz for 20 seconds (1000 samples)
    t = np.linspace(0, 20, 1000, endpoint=False)
    # 5 Hz signal + linear drift + DC offset
    data = 10.0 + 2.0 * t + np.sin(2 * np.pi * 5.0 * t)

    trace = op.Trace(data=data, header={
                     "sampling_rate": 50.0, "station": "TRI"})
    cleaned = self.trainer.get_clean_trace(
        trace, freqmin=1.0, freqmax=15.0, fs=100.0)

    # Sampling rate must be resampled to 100.0 Hz
    self.assertEqual(cleaned.stats.sampling_rate, 100.0)
    # Length must be 20 seconds * 100 Hz = 2000 samples
    self.assertEqual(len(cleaned.data), 2000)
    # Detrend & demean: mean should be approximately zero
    self.assertAlmostEqual(float(np.mean(cleaned.data)), 0.0, delta=0.5)
    # Signal must remain finite without NaNs or Infs
    self.assertTrue(np.isfinite(cleaned.data).all())

  def test_clean_trace_stream_support(self):
    """Verifies that get_clean_trace handles an obspy.Stream of multiple components."""
    data_z = np.linspace(5.0, 15.0, 500)
    data_n = np.linspace(-10.0, 10.0, 500)
    data_e = np.sin(np.linspace(0, 10, 500))

    stream = op.Stream([
        op.Trace(data=data_z.copy(), header={
                 "sampling_rate": 50.0, "channel": "HHZ"}),
        op.Trace(data=data_n.copy(), header={
                 "sampling_rate": 50.0, "channel": "HHN"}),
        op.Trace(data=data_e.copy(), header={
                 "sampling_rate": 50.0, "channel": "HHE"}),
    ])

    cleaned_stream = self.trainer.get_clean_trace(
        stream, freqmin=0.5, freqmax=10.0, fs=100.0)

    self.assertEqual(len(cleaned_stream), 3)
    for tr in cleaned_stream:
      self.assertEqual(tr.stats.sampling_rate, 100.0)
      self.assertEqual(len(tr.data), 1000)
      self.assertTrue(np.isfinite(tr.data).all())


class TestOGSTrainerAugmentations(unittest.TestCase):
  """Unit tests for the SeisBench augmentation pipeline builder."""

  def setUp(self):
    self.trainer = OGSTrainer.__new__(OGSTrainer)

  def test_build_augmentations_phasenet(self):
    """Verifies PhaseNet augmentation chain configuration with windowlen=3001."""
    mock_model = unittest.mock.MagicMock(spec=sbm.PhaseNet)
    mock_model.labels = ["P", "S", "N"]
    self.trainer.model = mock_model

    augmentations = self.trainer._build_augmentations()

    self.assertEqual(len(augmentations), 5)
    self.assertIsInstance(augmentations[0], sbg.WindowAroundSample)
    self.assertEqual(augmentations[0].samples_before, 3001)
    self.assertEqual(augmentations[0].windowlen, 6002)

    self.assertIsInstance(augmentations[1], sbg.RandomWindow)
    self.assertEqual(augmentations[1].windowlen, 3001)

    self.assertIsInstance(augmentations[2], sbg.ChangeDtype)
    self.assertIsInstance(augmentations[3], sbg.ProbabilisticLabeller)
    self.assertIsInstance(augmentations[4], sbg.ChangeDtype)

  def test_build_augmentations_eqtransformer(self):
    """Verifies EQTransformer augmentation chain configuration with windowlen=6000."""
    mock_model = unittest.mock.MagicMock(spec=sbm.EQTransformer)
    mock_model.labels = ["P", "S", "N"]
    self.trainer.model = mock_model

    augmentations = self.trainer._build_augmentations()

    self.assertEqual(len(augmentations), 5)
    self.assertIsInstance(augmentations[0], sbg.WindowAroundSample)
    self.assertEqual(augmentations[0].samples_before, 6000)
    self.assertEqual(augmentations[0].windowlen, 12000)

    self.assertIsInstance(augmentations[1], sbg.RandomWindow)
    self.assertEqual(augmentations[1].windowlen, 6000)

  def test_build_augmentations_fallback(self):
    """Verifies fallback window length is 3001 for non-standard model instances."""
    mock_model = unittest.mock.MagicMock(spec=torch.nn.Module)
    mock_model.labels = ["P", "S", "N"]
    self.trainer.model = mock_model

    augmentations = self.trainer._build_augmentations()

    self.assertEqual(augmentations[0].samples_before, 3001)
    self.assertEqual(augmentations[0].windowlen, 6002)
    self.assertEqual(augmentations[1].windowlen, 3001)


class TestOGSTrainerCheckpoint(unittest.TestCase):
  """Unit tests for model checkpoint persistence."""

  def setUp(self):
    self.temp_dir = tempfile.TemporaryDirectory()
    self.trainer = OGSTrainer.__new__(OGSTrainer)
    self.trainer.output_dir = Path(self.temp_dir.name)
    self.trainer.args = argparse.Namespace(
        model=OGS_C.PHASENET_STR,
        dataset=OGS_C.INSTANCE_STR
    )
    self.model = torch.nn.Linear(5, 2)
    self.trainer.model = self.model
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

  def tearDown(self):
    self.temp_dir.cleanup()

  def test_save_checkpoint_schema_and_reload(self):
    """Verifies checkpoint serialization, dictionary schema, and reload integrity."""
    self.trainer._save_checkpoint(
        epoch=1,
        optimizer=self.optimizer,
        train_loss=0.4567,
        val_loss=0.3456
    )

    expected_path = self.trainer.output_dir / "checkpoint_epoch_001.pt"
    self.assertTrue(expected_path.exists())

    checkpoint = torch.load(expected_path, weights_only=False)
    required_keys = {
        "epoch", "model_state_dict", "optimizer_state_dict",
        "train_loss", "val_loss", "model_name", "dataset"
    }
    self.assertTrue(required_keys.issubset(checkpoint.keys()))

    self.assertEqual(checkpoint["epoch"], 1)
    self.assertAlmostEqual(checkpoint["train_loss"], 0.4567)
    self.assertAlmostEqual(checkpoint["val_loss"], 0.3456)
    self.assertEqual(checkpoint["model_name"], OGS_C.PHASENET_STR)
    self.assertEqual(checkpoint["dataset"], OGS_C.INSTANCE_STR)
    self.assertIn("weight", checkpoint["model_state_dict"])
    self.assertIn("bias", checkpoint["model_state_dict"])

  def test_save_checkpoint_multi_epoch_naming(self):
    """Verifies zero-padded epoch naming across multiple checkpoints."""
    for ep in [1, 10, 100]:
      self.trainer._save_checkpoint(ep, self.optimizer, 0.5, 0.4)
      pt_file = self.trainer.output_dir / f"checkpoint_epoch_{ep:03d}.pt"
      self.assertTrue(pt_file.exists())


class TestOGSTrainerLoops(unittest.TestCase):
  """Unit tests for training and validation epoch iteration loops."""

  def setUp(self):
    self.trainer = OGSTrainer.__new__(OGSTrainer)
    self.model = SyntheticPickerModel()
    self.trainer.model = self.model
    self.dataset = SyntheticPickerDataset(num_samples=4, windowlen=50)
    self.dataloader = DataLoader(self.dataset, batch_size=2, shuffle=False)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)

  def test_train_epoch(self):
    """Verifies training loop forward pass, backward pass, loss, and training mode."""
    self.model.eval()  # ensure train_epoch sets model.train()
    loss = self.trainer._train_epoch(self.dataloader, self.optimizer)

    self.assertTrue(self.model.training)
    self.assertIsInstance(loss, float)
    self.assertGreater(loss, 0.0)
    self.assertTrue(np.isfinite(loss))

  def test_validate(self):
    """Verifies validation loop evaluation mode, loss, and absence of gradients."""
    self.model.train()  # ensure validate sets model.eval()
    val_loss = self.trainer._validate(self.dataloader)

    self.assertFalse(self.model.training)
    self.assertIsInstance(val_loss, float)
    self.assertGreater(val_loss, 0.0)
    self.assertTrue(np.isfinite(val_loss))

  def test_optimizer_updates_weights(self):
    """Verifies that model parameters are updated after a training epoch."""
    initial_weights = self.model.conv.weight.detach().clone()
    self.trainer._train_epoch(self.dataloader, self.optimizer)
    updated_weights = self.model.conv.weight.detach().clone()

    self.assertFalse(torch.equal(initial_weights, updated_weights))


class TestOGSTrainerLifecycle(unittest.TestCase):
  """Unit tests for OGSTrainer initialization and helper dispatch."""

  def test_trainer_initialization(self):
    """Verifies directory scaffolding and attribute binding during __init__."""
    with tempfile.TemporaryDirectory() as tmpdir:
      args = argparse.Namespace(
          catalog=Path(tmpdir) / "catalog",
          waveforms=Path(tmpdir) / "waveforms",
          dates=[
              datetime.strptime("20240320", OGS_C.YYYYMMDD_FMT),
              datetime.strptime("20240620", OGS_C.YYYYMMDD_FMT),
          ],
          model=OGS_C.PHASENET_STR,
          dataset=OGS_C.INSTANCE_STR,
          output=Path(tmpdir) / "checkpoints",
      )

      mock_model = unittest.mock.MagicMock()
      mock_model.device = torch.device("cpu")

      with unittest.mock.patch("ogstrainer.OGSCatalog") as mock_cat, \
              unittest.mock.patch.dict(ogstrainer.MODEL_REGISTRY, {OGS_C.PHASENET_STR: unittest.mock.MagicMock()}):
        ogstrainer.MODEL_REGISTRY[OGS_C.PHASENET_STR].from_pretrained.return_value = mock_model

        trainer = OGSTrainer(args)

        self.assertEqual(trainer.waveforms_dir, Path(tmpdir) / "waveforms")
        self.assertEqual(trainer.output_dir, Path(tmpdir) / "checkpoints")
        self.assertTrue(trainer.output_dir.exists())
        self.assertEqual(trainer.metadata_path,
                         trainer.output_dir / "metadata.csv")
        self.assertEqual(trainer.waveforms_path,
                         trainer.output_dir / "waveforms.hdf5")
        mock_model.to_preferred_device.assert_called_once()

  def test_download_waveform_subprocess(self):
    """Verifies subprocess CLI dispatch parameters when waveform downloading is invoked."""
    trainer = OGSTrainer.__new__(OGSTrainer)
    trainer.args = argparse.Namespace(waveforms=Path("/data/waveforms"))

    picktime = op.UTCDateTime("2024-03-20T12:00:00.000000Z")

    with unittest.mock.patch("subprocess.run") as mock_subproc, \
            unittest.mock.patch("time.sleep"):
      mock_subproc.return_value = unittest.mock.MagicMock(returncode=0)

      trainer._download_waveform("ACOM", "IV", picktime)

      mock_subproc.assert_called_once()
      call_cmd = mock_subproc.call_args[0][0]
      self.assertEqual(call_cmd[0], "python")
      self.assertIn("ogsdownloader.py", call_cmd[1])
      self.assertIn("-D", call_cmd)
      self.assertIn("240320", call_cmd)
      self.assertIn("-c", call_cmd)
      self.assertIn("12:00:00", call_cmd)
      self.assertIn("-S", call_cmd)
      self.assertIn("ACOM", call_cmd)
      self.assertIn("-N", call_cmd)
      self.assertIn("IV", call_cmd)


if __name__ == "__main__":
  unittest.main()

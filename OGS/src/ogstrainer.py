"""
===============================================================================
OGS Model Trainer - PyTorch / SeisBench Phase Picking Training CLI
===============================================================================

OVERVIEW:
Command-line training utility for deep-learning seismic phase pickers. Uses
PyTorch and SeisBench to train and fine-tune models on OGS continuous waveform
archives and supplied reference pick catalogs. This utility does not establish
the scientific validity or review status of those catalogs.

CLI ARGUMENTS:
  -C, --catalog         Path to the OGS reference catalog directory (required).
  -W, --waveforms       Path to the continuous waveform directory (required).
  -D, --dates           Date range (YYYYMMDD YYYYMMDD) to select training windows.
  -m, --model           SeisBench model class (default: PhaseNet).
  -s, --dataset         Pretrained weights name (default: instance).
  -b, --batch_size      Training batch size (default: 256).
  -e, --epochs          Number of training epochs (default: 5).
  -lr, --learning_rate  Learning rate for Adam optimizer (default: 1e-2).
  -t, --threads         Number of DataLoader workers (default: 4).
  -o, --output          Path for model checkpoints (default: ./checkpoints).
  -d, --download        Download waveforms if not locally cached.
  --prepare-only        Prepare dataset only without model training.
  --train-only          Train model using existing SeisBench dataset.
  --seed                Seed for event-ID splitting (fallback: 42); this module
                        does not seed all training randomness.
  --det-weight          Weight for EQTransformer detection loss (default: 1.0).
  --p-weight            Weight for EQTransformer P-phase loss (default: 1.0).
  --s-weight            Weight for EQTransformer S-phase loss (default: 1.0).
  --split-ratio         Train/dev split ratio grouped by event (default: 0.8).

USAGE:
python -m OGS.src.ogstrainer -C /path/to/catalog -W /path/to/waveforms -b 256 -e 10
python -m OGS.src.ogstrainer -C /path/to/catalog -W /path/to/waveforms -m EQTransformer
python -m OGS.src.ogstrainer -C /path/to/catalog -W /path/to/waveforms -m PhaseNet \\
  -s instance -e 20 -b 128 -lr 1e-3

DEPENDENCIES:
  - torch / torch.utils.data: neural network training runtime
  - seisbench: benchmark seismic models and waveform datasets
  - obspy: seismological waveform IO
  - pandas / numpy: catalog metadata handling
  - ogsconstants / ogscatalog: OGS catalog management and constants

AUTHORS:
  - 健
  - Istituto Nazionale di Oceanografia e di Geofisica Sperimentale (OGS)
    Centro di Ricerche Sismologiche (CRS)
  - Università degli Studi di Trieste (UniTS)
    Dipartimento di Matematica, Informatica e Geoscienze (MIGe)
    Applied Data Science and Artificial Intelligence (ADSAI)
  - Terabit Network for Research and Academic Big Data in Italy (TeRABIT)
    Consorzio Interuniversitario del Nord-Est per il Calcolo Automatico (CINECA)

===============================================================================
"""

from __future__ import annotations
from . import ogsutils as OGS_U, ogsconstants as OGS_C
from .ogscatalog import OGSCatalog
from torch.utils.data import DataLoader
import torch
from seisbench.util import worker_seeding
import seisbench.models as sbm
import seisbench.generate as sbg
import seisbench.data as sbd
import seisbench
import pandas as pd
import obspy as op
import numpy as np

import argparse
import ctypes
from datetime import datetime, timedelta
import glob
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

# Attempt to preload libstdc++.so.6 from the interpreter prefix if present.
_conda_prefix = os.path.dirname(os.path.dirname(sys.executable))
_libstdcxx = os.path.join(_conda_prefix, "lib", "libstdc++.so.6")
if os.path.exists(_libstdcxx):
  try:
    ctypes.CDLL(_libstdcxx, mode=ctypes.RTLD_GLOBAL)
  except Exception:
    pass


logger = OGS_U.setup_logger(__name__)

LEARNING_RATE = 1e-2
EPOCHS = 5
BATCH_SIZE = 256
NUM_WORKERS = 4

# Mapping from SeisBench model class names to seisbench.models classes
MODEL_REGISTRY = {
    OGS_C.PHASENET_STR: sbm.PhaseNet,
    OGS_C.EQTRANSFORMER_STR: sbm.EQTransformer,
}

# OGS phase dictionary: maps SeisBench label columns to phase types.
PHASE_DICT = {
    "trace_p_arrival_sample": OGS_C.PWAVE,
    "trace_P_arrival_sample": OGS_C.PWAVE,
    "trace_Pg_arrival_sample": OGS_C.PWAVE,
    "trace_Pn_arrival_sample": OGS_C.PWAVE,
    "trace_s_arrival_sample": OGS_C.SWAVE,
    "trace_S_arrival_sample": OGS_C.SWAVE,
    "trace_Sg_arrival_sample": OGS_C.SWAVE,
    "trace_Sn_arrival_sample": OGS_C.SWAVE,
}

# Hoisted flat dataset columnar schemas
INDEX_COLUMNS = [
    "filepath", "source_id", "network", "station", "location", "channel",
    "source_latitude_deg", "source_longitude_deg", "source_depth_km",
    "source_origin_time", "source_magnitude", OGS_C.PHASE_STR,
    OGS_C.TIME_STR, OGS_C.AMPLITUDE_STR, OGS_C.WEIGHT_STR
]

REQUIRED_PICK_COLUMNS = [
    OGS_C.IDX_PICKS_STR, OGS_C.STATION_STR, OGS_C.PHASE_STR,
    OGS_C.WEIGHT_STR, OGS_C.TIME_STR
]

REQUIRED_EVENT_COLUMNS = [
    OGS_C.IDX_EVENTS_STR, OGS_C.TIME_STR, OGS_C.LATITUDE_STR,
    OGS_C.LONGITUDE_STR, OGS_C.DEPTH_STR
]


def phasenet_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
  """Vector cross entropy loss for probabilistic PhaseNet labels (N, P, S).

  Following the SeisBench convention: mean over sample dimension,
  sum over pick dimension, mean over batch.
  """
  h = y_true * torch.log(y_pred + eps)
  h = h.mean(-1).sum(-1)  # Mean along sample dim, sum along pick dim
  h = h.mean()            # Mean over batch axis
  return -h


def eqtransformer_loss(
    y_pred: tuple[torch.Tensor, ...] | list[torch.Tensor],
    y_true: torch.Tensor | dict[str, torch.Tensor],
    det_weight: float = 1.0,
    p_weight: float = 1.0,
    s_weight: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
  """Multi-head loss for EQTransformer (Detection, P, S).

  Parameters
  ----------
  y_pred : tuple of 3 Tensors
    Outputs from EQTransformer forward pass (detection, P, S), each shape
    (B, L).
  y_true : Tensor of shape (B, 3, L) or dict
    Targets for detection (index 0), P (index 1), and S (index 2), or a
    dictionary with keys 'detection', 'P', and 'S'.
  det_weight : float
    Weight scaling for detection binary cross-entropy.
  p_weight : float
    Weight scaling for P-wave binary cross-entropy.
  s_weight : float
    Weight scaling for S-wave binary cross-entropy.
  eps : float
    Numerical stability epsilon.
  """
  pred_det, pred_p, pred_s = y_pred[0], y_pred[1], y_pred[2]
  if isinstance(y_true, dict):
    true_det = y_true["detection"]
    true_p = y_true[OGS_C.PWAVE]
    true_s = y_true[OGS_C.SWAVE]
  else:
    true_det = y_true[:, 0, :]
    true_p = y_true[:, 1, :]
    true_s = y_true[:, 2, :]

  # Detection head: binary cross entropy
  bce_det = -(
      true_det * torch.log(pred_det + eps) +
      (1.0 - true_det) * torch.log(1.0 - pred_det + eps)
  )
  loss_det = bce_det.mean()

  # Phase heads: binary cross entropy, averaged across batch and samples.
  ce_p = -(
      true_p * torch.log(pred_p + eps) +
      (1.0 - true_p) * torch.log(1.0 - pred_p + eps)
  ).mean()
  ce_s = -(
      true_s * torch.log(pred_s + eps) +
      (1.0 - true_s) * torch.log(1.0 - pred_s + eps)
  ).mean()

  return det_weight * loss_det + p_weight * ce_p + s_weight * ce_s


def loss_fn(y_pred: Any, y_true: Any, eps: float = 1e-5) -> torch.Tensor:
  """Default loss function for PhaseNet (backward compatibility)."""
  return phasenet_loss(y_pred, y_true, eps=eps)


class EQTransformerTargetLabeller:
  """Generates 3-channel targets (detection, P, S) for EQTransformer."""

  def __init__(self, sigma: float = 30.0, key: str = "y") -> None:
    self.sigma = sigma
    self.key = key

  def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
    seq_len = state["X"].shape[-1]
    y = np.zeros((3, seq_len), dtype=np.float32)

    meta = state.get("_metadata", {})
    p_sample = meta.get("trace_p_arrival_sample")
    s_sample = meta.get("trace_s_arrival_sample")

    time_axis = np.arange(seq_len, dtype=np.float32)

    # Channel 1: P arrival Gaussian
    if p_sample is not None and not np.isnan(p_sample):
      p_sample_val = float(p_sample)
      if 0 <= p_sample_val < seq_len:
        y[1] = np.exp(-0.5 * ((time_axis - p_sample_val) / self.sigma) ** 2)

    # Channel 2: S arrival Gaussian
    if s_sample is not None and not np.isnan(s_sample):
      s_sample_val = float(s_sample)
      if 0 <= s_sample_val < seq_len:
        y[2] = np.exp(-0.5 * ((time_axis - s_sample_val) / self.sigma) ** 2)

    # Channel 0: Detection interval
    if (p_sample is not None and not np.isnan(p_sample) and
            s_sample is not None and not np.isnan(s_sample)):
      p_val = float(p_sample)
      s_val = float(s_sample)
      det_start = int(max(0, p_val))
      det_end = int(min(seq_len, s_val + 1.4 * (s_val - p_val)))
      if det_end > det_start:
        y[0, det_start:det_end] = 1.0
    elif p_sample is not None and not np.isnan(p_sample):
      p_val = float(p_sample)
      det_start = int(max(0, p_val - self.sigma))
      det_end = int(min(seq_len, p_val + 4 * self.sigma))
      y[0, det_start:det_end] = 1.0
    elif s_sample is not None and not np.isnan(s_sample):
      s_val = float(s_sample)
      det_start = int(max(0, s_val - 4 * self.sigma))
      det_end = int(min(seq_len, s_val + self.sigma))
      y[0, det_start:det_end] = 1.0

    state[self.key] = y
    return state


class OGSTrainer:
  def __init__(self, args: argparse.Namespace) -> None:
    self.args = args
    self.logger = OGS_U.setup_logger(__name__, args.verbose, args.quiet)

    # Date range parsing
    self.start, self.end = args.dates

    # Catalog initialization unless in train-only mode
    if getattr(args, "train_only", False):
      self.catalog = None
    else:
      self.catalog = OGSCatalog(
          args.catalog,
          start=self.start,
          end=self.end,
          name="Training Catalog"
      )

    self.waveforms_dir = Path(args.waveforms)
    self.output_dir = Path(args.output)
    self.output_dir.mkdir(parents=True, exist_ok=True)

    # SeisBench output paths for WaveformDataWriter
    self.metadata_path = self.output_dir / "metadata.csv"
    self.waveforms_path = self.output_dir / "waveforms.hdf5"

    # Lazy model initialization: instantiate pretrained model only when training is active
    self.model = None
    if not getattr(args, "prepare_only", False):
      self.get_model()

  def get_model(self) -> torch.nn.Module:
    """Retrieve or lazily initialize the neural network model."""
    if self.model is None:
      model_cls = MODEL_REGISTRY[self.args.model]
      self.model = model_cls.from_pretrained(self.args.dataset)
      self.model.to_preferred_device(verbose=True)
      self.logger.info("Model: %s (pretrained: %s) on device: %s",
                       self.args.model, self.args.dataset, self.model.device)
    return self.model

  def compute_loss(
      self,
      y_pred: Any,
      y_true: torch.Tensor,
  ) -> torch.Tensor:
    """Compute architecture-specific loss for forward predictions."""
    if isinstance(self.model, sbm.EQTransformer) or isinstance(y_pred, (tuple, list)):
      return eqtransformer_loss(
          y_pred,
          y_true,
          det_weight=getattr(self.args, "det_weight", 1.0),
          p_weight=getattr(self.args, "p_weight", 1.0),
          s_weight=getattr(self.args, "s_weight", 1.0),
      )
    return phasenet_loss(y_pred, y_true)

  def get_station_info(self, station: str) -> tuple[str, str, str]:
    """Parse network.station.location from a dotted station string."""
    parts = station.split(".")
    if len(parts) >= 3:
      return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
      return parts[0], parts[1], ""
    return "", station, ""

  def get_event_params(
      self, event_row: pd.Series | dict[str, Any]
  ) -> dict[str, Any]:
    """Extract event parameters from a catalog event DataFrame row.

    Parameters
    ----------
    event_row : pd.Series or dict
      A row from the events DataFrame containing origin parameters.

    Returns
    -------
    dict
      Event parameters for the SeisBench WaveformDataWriter.
    """
    event_params: dict[str, Any] = {
        "source_origin_time": str(event_row.get(OGS_C.TIME_STR, "")),
        "source_latitude_deg": float(event_row.get(OGS_C.LATITUDE_STR, np.nan)),
        "source_longitude_deg": float(event_row.get(OGS_C.LONGITUDE_STR, np.nan)),
        "source_depth_km": float(event_row.get(OGS_C.DEPTH_STR, np.nan)),
    }

    mag = event_row.get(OGS_C.MAGNITUDE_L_STR, np.nan)
    if pd.notna(mag):
      event_params["source_magnitude"] = float(mag)
      event_params["source_magnitude_type"] = "ML"

    return event_params

  def get_clean_trace(
      self,
      trace_or_stream: op.Trace | op.Stream,
      freqmin: float = 0.1,
      freqmax: float = 20.0,
      fs: float = 100.0,
  ) -> op.Trace | op.Stream:
    """Apply standard preprocessing to a waveform trace or stream.

    Pipeline: detrend(linear) -> detrend(demean) -> taper -> bandpass -> resample.
    """
    trace_or_stream.detrend("linear")
    trace_or_stream.detrend("demean")
    trace_or_stream.taper(max_percentage=0.05, type="cosine")
    trace_or_stream.filter("bandpass", freqmin=freqmin, freqmax=freqmax,
                           corners=4, zerophase=True)
    trace_or_stream.resample(fs)
    return trace_or_stream

  def _download_waveform(
      self,
      sta: str,
      net: str,
      picktime: op.UTCDateTime,
  ) -> bool:
    """Download a missing waveform using ogsdownloader.py."""
    date_ = picktime.strftime(OGS_C.YYYYMMDD_FMT)
    time_ = picktime.strftime(OGS_C.TIME_FMT)
    cmd = [
        sys.executable, str(Path(__file__).parent / "ogsdownloader.py"),
        "-D", date_, date_,
        "-c", time_,
        "-W", str(self.args.waveforms),
        "--station", sta,
    ]
    if net:
      cmd.extend(["--network", net])
    self.logger.info("Downloading waveform: %s", " ".join(cmd))
    try:
      result = subprocess.run(cmd, capture_output=True, text=True, check=False)
      if result.returncode != 0:
        self.logger.warning("Download failed: %s", result.stderr)
        return False
      return True
    except Exception as exc:
      self.logger.warning("Downloader execution error: %s", exc)
      return False

  def build_dataset(self) -> pd.DataFrame:
    """Build a training dataset CSV from the OGS catalog and waveforms.

    Validates catalog schemas and selects non-null phases with numeric weight
    <= 2 from event/station groups containing more than one pick. Matches exact
    pick-centered filename windows, checks pick-time coverage against the
    stream's overall minimum/maximum times, and writes training_dataset.csv.
    This does not verify continuous coverage of every component.

    Returns
    -------
    pd.DataFrame
      Training dataset index with event and pick metadata.
    """
    if self.catalog is None:
      raise ValueError(
          "Catalog is not initialized; cannot build training dataset.")

    start_date = self.start.date() if isinstance(
        self.start, datetime) else self.start
    end_date = self.end.date() if isinstance(self.end, datetime) else self.end
    num_days = (end_date - start_date).days + 1
    days = [start_date + timedelta(days=i) for i in range(max(0, num_days))]

    records: list[dict[str, Any]] = []
    offset_sec = OGS_C.PICK_TRAIN_OFFSET.total_seconds()

    for day_ in days:
      if day_ not in self.catalog.picks_:
        continue
      try:
        df_picks = self.catalog._load_day("picks", day_)
        df_events = self.catalog._load_day("events", day_)
      except Exception as exc:
        self.logger.warning("Failed to load catalog for day %s: %s", day_, exc)
        continue

      if df_picks.empty or df_events.empty:
        continue

      # Validate required schema columns
      if not all(col in df_picks.columns for col in REQUIRED_PICK_COLUMNS):
        self.logger.warning(
            "Missing required pick columns in day %s, skipping", day_)
        continue
      if not all(col in df_events.columns for col in REQUIRED_EVENT_COLUMNS):
        self.logger.warning(
            "Missing required event columns in day %s, skipping", day_)
        continue

      # Pre-index events by identifier once per day
      events_by_id = (
          df_events.drop_duplicates(subset=[OGS_C.IDX_EVENTS_STR])
          .set_index(OGS_C.IDX_EVENTS_STR)
          .to_dict("index")
      )

      # Filter picks: valid phase and numeric weight <= 2
      valid_picks = df_picks[df_picks[OGS_C.PHASE_STR].notna()].copy()
      valid_picks[OGS_C.WEIGHT_STR] = pd.to_numeric(
          valid_picks[OGS_C.WEIGHT_STR], errors="coerce"
      )
      valid_picks = valid_picks[valid_picks[OGS_C.WEIGHT_STR] <= 2]

      for (event_id, station), station_picks in valid_picks.groupby(
          [OGS_C.IDX_PICKS_STR, OGS_C.STATION_STR]
      ):
        if len(station_picks) <= 1:
          continue

        event_data = events_by_id.get(event_id)
        if event_data is None:
          self.logger.debug("Missing event origin for pick ID: %s", event_id)
          continue
        event_params = self.get_event_params(event_data)

        for _, pick in station_picks.iterrows():
          net, sta, loc = self.get_station_info(pick[OGS_C.STATION_STR])
          try:
            picktime = op.UTCDateTime(pick[OGS_C.TIME_STR])
          except Exception:
            continue

          t_start = picktime - offset_sec
          t_end = picktime + offset_sec

          day_dir = OGS_U.day_directory(self.waveforms_dir, picktime.date)
          pattern = (
              f"{net if net else '*'}.{sta}.*.*__"
              f"{t_start.strftime('%Y%m%dT%H%M%SZ')}__"
              f"{t_end.strftime('%Y%m%dT%H%M%SZ')}.mseed"
          )
          filepath = day_dir / pattern
          files_ = glob.glob(str(filepath))

          if not files_ and self.args.download:
            self.logger.info("Missing waveform file: %s", filepath)
            download_ok = self._download_waveform(sta, net, picktime)
            if download_ok:
              files_ = glob.glob(str(filepath))

          if not files_:
            continue

          for wf_file in files_:
            try:
              stream = op.read(wf_file)
              stream = self.get_clean_trace(stream)
            except Exception:
              self.logger.warning("Failed to read waveform: %s", wf_file)
              continue

            if len(stream) == 0:
              continue

            # Validate interval coverage
            st_start = min(tr.stats.starttime for tr in stream)
            st_end = max(tr.stats.endtime for tr in stream)
            if picktime < st_start or picktime > st_end:
              self.logger.debug(
                  "Waveform %s does not cover pick time %s", wf_file, picktime
              )
              continue

            stats = stream[0].stats
            records.append({
                "filepath": wf_file,
                "source_id": event_id,
                "network": stats.network,
                "station": stats.station,
                "location": stats.location,
                "channel": stats.channel,
                "source_latitude_deg": event_params["source_latitude_deg"],
                "source_longitude_deg": event_params["source_longitude_deg"],
                "source_depth_km": event_params["source_depth_km"],
                "source_origin_time": event_params["source_origin_time"],
                "source_magnitude": event_params.get("source_magnitude", np.nan),
                OGS_C.TIME_STR: str(pick[OGS_C.TIME_STR]),
                OGS_C.PHASE_STR: pick[OGS_C.PHASE_STR],
                OGS_C.WEIGHT_STR: pick[OGS_C.WEIGHT_STR],
                OGS_C.AMPLITUDE_STR: pick.get(OGS_C.AMPLITUDE_STR, np.nan),
            })

    dataset = pd.DataFrame(records, columns=INDEX_COLUMNS)
    dataset_path = self.output_dir / "training_dataset.csv"
    dataset.to_csv(dataset_path, index=False)
    self.logger.info("Training dataset: %d samples -> %s",
                     len(dataset), dataset_path)
    return dataset

  def convert_to_seisbench(
      self, dataset_df: pd.DataFrame | None = None
  ) -> tuple[Path, Path]:
    """
    Convert training dataset CSV into SeisBench WaveformDataset
    (HDF5 + metadata).

    Resamples to 100 Hz and stacks Z/N/E components truncated to the shortest
    length, or uses stream order if exactly three unrecognized components are
    present. Component start times are not aligned here. Arrival samples use
    the first trace's start time. Assigns train/dev splits by source_id and
    writes waveforms.hdf5 and metadata.csv via WaveformDataWriter, returning
    their paths (metadata first). Grouping by source_id prevents shared IDs
    across splits but does not check overlapping waveform content.
    """
    if dataset_df is None or dataset_df.empty:
      dataset_path = self.output_dir / "training_dataset.csv"
      if not dataset_path.exists():
        raise FileNotFoundError(
            f"Training dataset CSV not found at {dataset_path}")
      dataset_df = pd.read_csv(dataset_path)

    if dataset_df.empty:
      raise ValueError(
          "Cannot convert empty dataset DataFrame to SeisBench format")

    # Partition unique source IDs into disjoint train and dev sets.
    seed = getattr(self.args, "seed", 42)
    split_ratio = getattr(self.args, "split_ratio", 0.8)
    rng = np.random.default_rng(seed)

    unique_events = list(dataset_df["source_id"].unique())
    rng.shuffle(unique_events)

    n_train = int(round(len(unique_events) * split_ratio))
    if len(unique_events) > 1 and n_train == len(unique_events):
      n_train = len(unique_events) - 1
    elif len(unique_events) > 1 and n_train == 0:
      n_train = 1

    train_events = set(unique_events[:n_train])
    event_split_map = {
        eid: "train" if eid in train_events else "dev"
        for eid in unique_events
    }

    grouped = dataset_df.groupby(["filepath", "source_id", "station"])
    self.logger.info(
        "Converting %d waveform groups to SeisBench format...", len(grouped))

    with sbd.WaveformDataWriter(self.metadata_path, self.waveforms_path) as writer:
      for (wf_file, source_id, sta), group in grouped:
        try:
          st = op.read(wf_file)
          st = self.get_clean_trace(st)
        except Exception as exc:
          self.logger.warning(
              "Skipping unreadable waveform %s: %s", wf_file, exc)
          continue

        comp_map = {}
        for tr in st:
          comp = tr.stats.channel[-1].upper()
          if comp in ("Z", "N", "E"):
            comp_map[comp] = tr

        if len(comp_map) < 3:
          if len(st) == 3:
            comp_data = np.stack([st[0].data, st[1].data, st[2].data], axis=0)
          else:
            self.logger.debug(
                "Waveform %s does not contain 3 components, skipping", wf_file
            )
            continue
        else:
          min_len = min(
              len(comp_map["Z"].data),
              len(comp_map["N"].data),
              len(comp_map["E"].data),
          )
          comp_data = np.stack([
              comp_map["Z"].data[:min_len],
              comp_map["N"].data[:min_len],
              comp_map["E"].data[:min_len],
          ], axis=0).astype(np.float32)

        start_time = st[0].stats.starttime
        first_row = group.iloc[0]

        meta: dict[str, Any] = {
            "source_id": str(source_id),
            "station_network_code": str(first_row.get("network", "")),
            "station_code": str(sta),
            "station_location_code": str(first_row.get("location", "")),
            "source_latitude_deg": float(first_row.get("source_latitude_deg", np.nan)),
            "source_longitude_deg": float(first_row.get("source_longitude_deg", np.nan)),
            "source_depth_km": float(first_row.get("source_depth_km", np.nan)),
            "source_origin_time": str(first_row.get("source_origin_time", "")),
            "source_magnitude": float(first_row.get("source_magnitude", np.nan)),
            "trace_sampling_rate_hz": 100.0,
            "trace_start_time": str(start_time),
            "split": event_split_map.get(source_id, "train"),
        }

        # Calculate arrival samples relative to waveform start
        for _, pick in group.iterrows():
          phase = str(pick.get(OGS_C.PHASE_STR, "")).upper()
          pick_time_str = pick.get(OGS_C.TIME_STR)
          if pd.isna(pick_time_str):
            continue
          try:
            pick_utc = op.UTCDateTime(pick_time_str)
            sample_offset = int(round((pick_utc - start_time) * 100.0))
            if 0 <= sample_offset < comp_data.shape[1]:
              if phase.startswith(OGS_C.PWAVE):
                meta["trace_p_arrival_sample"] = sample_offset
              elif phase.startswith(OGS_C.SWAVE):
                meta["trace_s_arrival_sample"] = sample_offset
          except Exception:
            continue

        writer.add_trace(meta, comp_data)

    self.logger.info(
        "SeisBench dataset generated: %s, %s",
        self.metadata_path, self.waveforms_path
    )
    return self.metadata_path, self.waveforms_path

  def _build_augmentations(self) -> list[Any]:
    """Build the SeisBench augmentation pipeline matching model architecture."""
    model = self.get_model()
    if isinstance(model, sbm.PhaseNet):
      windowlen = 3001
      augmentations = [
          sbg.WindowAroundSample(
              list(PHASE_DICT.keys()),
              samples_before=windowlen,
              windowlen=windowlen * 2,
              selection="random",
              strategy="variable",
          ),
          sbg.RandomWindow(windowlen=windowlen, strategy="pad"),
          sbg.ChangeDtype(np.float32),
          sbg.ProbabilisticLabeller(
              label_columns=PHASE_DICT,
              model_labels=model.labels,
              sigma=30,
              dim=0,
          ),
          sbg.ChangeDtype(np.float32, key="y"),
      ]
      return augmentations
    elif isinstance(model, sbm.EQTransformer):
      windowlen = 6000
      augmentations = [
          sbg.WindowAroundSample(
              list(PHASE_DICT.keys()),
              samples_before=windowlen,
              windowlen=windowlen * 2,
              selection="random",
              strategy="variable",
          ),
          sbg.RandomWindow(windowlen=windowlen, strategy="pad"),
          sbg.ChangeDtype(np.float32),
          EQTransformerTargetLabeller(sigma=30.0),
          sbg.ChangeDtype(np.float32, key="y"),
      ]
      return augmentations
    else:
      windowlen = 3001
      labels = getattr(model, "labels", [OGS_C.PWAVE, OGS_C.SWAVE, "N"])
      augmentations = [
          sbg.WindowAroundSample(
              list(PHASE_DICT.keys()),
              samples_before=windowlen,
              windowlen=windowlen * 2,
              selection="random",
              strategy="variable",
          ),
          sbg.RandomWindow(windowlen=windowlen, strategy="pad"),
          sbg.ChangeDtype(np.float32),
          sbg.ProbabilisticLabeller(
              label_columns=PHASE_DICT,
              model_labels=labels,
              sigma=30,
              dim=0,
          ),
          sbg.ChangeDtype(np.float32, key="y"),
      ]
      return augmentations

  def _create_data_loaders(
      self, data: sbd.WaveformDataset
  ) -> tuple[DataLoader, DataLoader]:
    """Create train/dev data loaders from a SeisBench dataset.

    Enforces non-empty train/dev splits; trusts existing split metadata
    without rechecking event overlap.
    """
    train, dev, _ = data.train_dev_test()
    if len(train) == 0:
      raise ValueError(
          "Training split is empty. Check dataset split configuration.")
    if len(dev) == 0:
      raise ValueError(
          "Development split is empty. Check dataset split configuration.")

    train_generator = sbg.GenericGenerator(train)
    dev_generator = sbg.GenericGenerator(dev)

    augmentations = self._build_augmentations()
    train_generator.add_augmentations(augmentations)
    dev_generator.add_augmentations(augmentations)

    train_loader = DataLoader(
        train_generator,
        batch_size=self.args.batch_size,
        shuffle=True,
        num_workers=self.args.threads,
        worker_init_fn=worker_seeding,
    )
    dev_loader = DataLoader(
        dev_generator,
        batch_size=self.args.batch_size,
        shuffle=False,
        num_workers=self.args.threads,
        worker_init_fn=worker_seeding,
    )
    return train_loader, dev_loader

  def _train_epoch(
      self, dataloader: DataLoader, optimizer: torch.optim.Optimizer
  ) -> float:
    """Run one training epoch."""
    if len(dataloader) == 0 or len(dataloader.dataset) == 0:
      raise ValueError(
          "Training DataLoader is empty; cannot run training epoch.")

    model = self.get_model()
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    size = len(dataloader.dataset)

    for batch_id, batch in enumerate(dataloader):
      x = batch["X"].to(model.device)
      x_preproc = model.annotate_batch_pre(x, {})
      pred = model(x_preproc)
      loss = self.compute_loss(pred, batch["y"].to(model.device))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss += loss.item()

      if batch_id % 5 == 0:
        current = batch_id * batch["X"].shape[0]
        self.logger.info("  loss: %7f  [%5d/%5d]", loss.item(), current, size)

    return total_loss / max(num_batches, 1)

  def _validate(self, dataloader: DataLoader) -> float:
    """Run validation and return average loss."""
    if len(dataloader) == 0 or len(dataloader.dataset) == 0:
      raise ValueError("Validation DataLoader is empty; cannot compute loss.")

    model = self.get_model()
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
      for batch in dataloader:
        x = batch["X"].to(model.device)
        x_preproc = model.annotate_batch_pre(x, {})
        pred = model(x_preproc)
        loss = self.compute_loss(pred, batch["y"].to(model.device))
        total_loss += loss.item()

    avg_loss = total_loss / max(num_batches, 1)
    self.logger.info("  Validation avg loss: %8f", avg_loss)
    return avg_loss

  def _save_checkpoint(
      self,
      epoch: int,
      optimizer: torch.optim.Optimizer,
      train_loss: float,
      val_loss: float,
  ) -> None:
    """
    Save model/optimizer states, losses, and selected configuration/version
    metadata.
    """
    model = self.get_model()
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "model_name": self.args.model,
        "dataset": self.args.dataset,
        "provenance": {
            "torch_version": torch.__version__,
            "seisbench_version": getattr(seisbench, "__version__", "unknown"),
            "obspy_version": op.__version__,
            "seed": getattr(self.args, "seed", 42),
            "split_ratio": getattr(self.args, "split_ratio", 0.8),
            "split_policy": "event_grouped",
            "det_weight": getattr(self.args, "det_weight", 1.0),
            "p_weight": getattr(self.args, "p_weight", 1.0),
            "s_weight": getattr(self.args, "s_weight", 1.0),
            "sampling_rate_hz": 100.0,
            "filter_bandpass": [0.1, 20.0],
            "catalog_path": str(getattr(self.args, "catalog", "")),
            "waveforms_path": str(getattr(self.args, "waveforms", "")),
        },
    }
    path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(checkpoint, path)
    self.logger.info("  Checkpoint saved: %s", path)

  def train(self, data: sbd.WaveformDataset) -> None:
    """Run the full training loop."""
    train_loader, dev_loader = self._create_data_loaders(data)
    model = self.get_model()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=self.args.learning_rate
    )

    best_val_loss = float("inf")

    for epoch in range(self.args.epochs):
      self.logger.info("Epoch %d/%d", epoch + 1, self.args.epochs)
      self.logger.info("-" * 40)

      train_loss = self._train_epoch(train_loader, optimizer)
      val_loss = self._validate(dev_loader)

      self.logger.info(
          "  Train loss: %.6f | Val loss: %.6f", train_loss, val_loss
      )

      self._save_checkpoint(epoch + 1, optimizer, train_loss, val_loss)

      if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_path = self.output_dir / "best_model.pt"
        torch.save(model.state_dict(), best_path)
        self.logger.info("  Best model updated: %.6f -> %s",
                         val_loss, best_path)

    self.logger.info("Training complete. Best val loss: %.6f", best_val_loss)


def main(args: argparse.Namespace) -> None:
  global logger
  logger = OGS_U.setup_logger(__name__, args.verbose, args.quiet)

  trainer = OGSTrainer(args)

  # Step 1: Preparation workflow unless running in train-only mode
  if not getattr(args, "train_only", False):
    logger.info("Building training dataset index...")
    dataset_df = trainer.build_dataset()

    if dataset_df.empty:
      logger.error("No training data found. Check catalog and waveform paths.")
      return

    # Step 2: Convert to SeisBench HDF5/metadata format
    if not (trainer.waveforms_path.exists() and trainer.metadata_path.exists()):
      logger.info("Serializing dataset into SeisBench format...")
      trainer.convert_to_seisbench(dataset_df)

    if getattr(args, "prepare_only", False):
      logger.info("Dataset preparation complete (--prepare-only). Exiting.")
      return

  # Step 3: Load existing SeisBench dataset
  if trainer.waveforms_path.exists() and trainer.metadata_path.exists():
    logger.info("Loading existing SeisBench dataset from %s",
                trainer.output_dir)
    data = sbd.WaveformDataset(trainer.output_dir)
  else:
    logger.error(
        "SeisBench HDF5 dataset not found at %s. "
        "Cannot proceed with training without prepared SeisBench dataset.",
        trainer.waveforms_path
    )
    return

  # Step 4: Execute training
  logger.info(
      "Starting training: model=%s, dataset=%s, epochs=%d, "
      "batch_size=%d, lr=%s",
      args.model, args.dataset, args.epochs,
      args.batch_size, args.learning_rate
  )
  trainer.train(data)


if __name__ == "__main__":
  main(OGS_U.parse_trainer_args())

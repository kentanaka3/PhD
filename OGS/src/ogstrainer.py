"""
=============================================================================
OGS Model Trainer - PyTorch / SeisBench Phase Picking Training CLI
=============================================================================

OVERVIEW:
Command-line training utility for deep-learning seismic phase pickers. Uses
PyTorch and SeisBench to train and fine-tune models on OGS continuous waveform
archives and validated reference pick catalogs.

CLI ARGUMENTS:
  -C, --catalog    Path to the OGS reference catalog directory (required).
  -W, --waveforms  Path to the continuous waveform directory (required).
  -D, --dates      Date range (YYYYMMDD YYYYMMDD) to select training windows.
  -m, --model      SeisBench model class (default: PhaseNet).
  -s, --dataset    Pretrained weights name (default: instance).
  -b, --batch_size Training batch size (default: 256).
  -e, --epochs     Number of training epochs (default: 5).
  -lr, --learning_rate  Learning rate for Adam optimizer (default: 1e-2).
  -w, --workers    Number of DataLoader workers (default: 4).
  -o, --output     Path for model checkpoints (default: ./checkpoints).
  -d, --download   Download waveforms if not locally cached.

USAGE:
python ogstrainer.py -C /path/to/catalog -W /path/to/waveforms -b 256 -e 10
python ogstrainer.py -C /path/to/catalog -W /path/to/waveforms -m EQTransformer
python ogstrainer.py -C /path/to/catalog -W /path/to/waveforms -m PhaseNet \\
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
=============================================================================
"""

import glob
import logging
import argparse
import subprocess
import time
import numpy as np
import obspy as op
import pandas as pd
import torch
import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm

from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from seisbench.util import worker_seeding

import ogsconstants as OGS_C
import ogsutils as OGS_U
from ogscatalog import OGSCatalog

logger = logging.getLogger(__name__)

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
# OGS catalogs use simple "P" and "S" phase annotations.
PHASE_DICT = {
    "trace_p_arrival_sample": "P",
    "trace_P_arrival_sample": "P",
    "trace_Pg_arrival_sample": "P",
    "trace_Pn_arrival_sample": "P",
    "trace_s_arrival_sample": "S",
    "trace_S_arrival_sample": "S",
    "trace_Sg_arrival_sample": "S",
    "trace_Sn_arrival_sample": "S",
}


def parse_arguments():
  parser = argparse.ArgumentParser(description="Train OGS models")
  parser.add_argument(
      "-C", "--catalog", type=Path, required=True,
      help="Path to the catalog directory"
  )
  date_group = parser.add_mutually_exclusive_group(required=False)
  date_group.add_argument(
      '-D', "--dates", required=False, metavar=OGS_C.DATE_STD,
      type=OGS_U.is_date, nargs=2, action=OGS_U.SortDatesAction,
      default=[
          datetime.strptime("20240320", OGS_C.YYYYMMDD_FMT),
          datetime.strptime("20240620", OGS_C.YYYYMMDD_FMT)
      ],
      help="""
          Specify the beginning and ending (inclusive) Gregorian date
          (YYYYMMDD) range to work with.
      """
  )
  parser.add_argument(
      "-W", "--waveforms", type=Path, required=True,
      help="Path to the waveforms directory"
  )
  parser.add_argument(
      "-m", "--model", type=str, default=OGS_C.PHASENET_STR,
      choices=list(MODEL_REGISTRY.keys()),
      help="SeisBench model class name (default: PhaseNet)"
  )
  parser.add_argument(
      "-s", "--dataset", type=str, default=OGS_C.INSTANCE_STR,
      choices=[
          OGS_C.INSTANCE_STR, OGS_C.STEAD_STR, OGS_C.SCEDC_STR,
          OGS_C.ORIGINAL_STR, OGS_C.ADRIAARRAY_STR
      ],
      help="Pretrained weights name to fine-tune from (default: instance)"
  )
  parser.add_argument(
      "-b", "--batch_size", type=int, default=BATCH_SIZE,
      help="Batch size for training"
  )
  parser.add_argument(
      "-d", "--download", action="store_true", help="Enable download mode"
  )
  parser.add_argument(
      "-e", "--epochs", type=int, default=EPOCHS,
      help="Number of training epochs"
  )
  parser.add_argument(
      "-lr", "--learning_rate", type=float, default=LEARNING_RATE,
      help="Learning rate for training"
  )
  parser.add_argument(
      "-w", "--workers", type=int, default=NUM_WORKERS,
      help="Number of workers for data loading"
  )
  parser.add_argument(
      "-o", "--output", type=Path, default=Path("./checkpoints"),
      help="Output directory for model checkpoints"
  )
  return parser.parse_args()


def loss_fn(y_pred, y_true, eps=1e-5):
  """Vector cross entropy loss for probabilistic phase labels.

  Following the SeisBench convention: mean over sample dimension,
  sum over pick dimension, mean over batch.
  """
  h = y_true * torch.log(y_pred + eps)
  h = h.mean(-1).sum(-1)  # Mean along sample dim, sum along pick dim
  h = h.mean()            # Mean over batch axis
  return -h


class OGSTrainer:
  def __init__(self, args):
    self.args = args
    self.catalog = OGSCatalog(
        args.catalog,
        start=args.dates[0],
        end=args.dates[1],
        name="Training Catalog"
    )
    self.start, self.end = args.dates
    self.waveforms_dir = Path(args.waveforms)
    self.waveforms = OGS_U.waveforms(self.waveforms_dir, self.start, self.end)
    self.stations = {
        station.split(".")[1]: station for date in self.waveforms.values()
        for station in date.keys()
    }
    add = {}
    for key, val in self.stations.items():
      if len(key) > 4:
        add[key[:4]] = val
    self.stations.update(add)
    self.download = args.download
    self.dataset()

  def train(self):
    pass

  def get_station_info(self, station):
    sta = station.split(".")[1]  # type: ignore
    return (
        self.stations[sta].split(".") if sta in self.stations
        else ("", sta, "")
    )

  def get_event_params(self, event):
    print(event)
    exit()
    origin = event.preferred_origin()
    mag = event.preferred_magnitude()

    source_id = str(event.resource_id)

    event_params = {
        "source_id": source_id,
        "source_origin_time": str(origin.time),
        "source_origin_uncertainty_sec": origin.time_errors["uncertainty"],
        "source_latitude_deg": origin.latitude,
        "source_latitude_uncertainty_km": origin.latitude_errors["uncertainty"],
        "source_longitude_deg": origin.longitude,
        "source_longitude_uncertainty_km": origin.longitude_errors["uncertainty"],
        "source_depth_km": origin.depth / 1e3,
        "source_depth_uncertainty_km": origin.depth_errors["uncertainty"] / 1e3,
    }

    if mag is not None:
      event_params["source_magnitude"] = mag.mag
      event_params["source_magnitude_uncertainty"] = mag.mag_errors["uncertainty"]
      event_params["source_magnitude_type"] = mag.magnitude_type
      event_params["source_magnitude_author"] = mag.creation_info.agency_id

      if str(origin.time) < "2015-01-07":
        split = "train"
      elif str(origin.time) < "2015-01-08":
        split = "dev"
      else:
        split = "test"
      event_params["split"] = split

    return event_params

  def data_writer(self, catalog):
    with sbd.WaveformDataWriter(
        self.metadata_path, self.waveforms_path, overwrite=True
    ) as writer:
      writer.data_format = {
          "dimension_order": "CW",
          "component_order": "ZNE",
          "measurement": "velocity",
          "unit": "counts",
          "instrument_response": "not restituted",
      }
      for event in catalog:
        event_params = self.get_event_params(event)

  def get_clean_trace(self, trace, freqmin=0.1, freqmax=20.0, fs=100.0):
    trace.detrend("linear")
    trace.detrend("demean")
    trace.taper(max_percentage=0.05, type="cosine")
    trace.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=4,
                 zerophase=True)
    trace.resample(fs)
    return trace

  def dataset(self):
    DIR_FMT = {
        "year": "{:04}",
        "month": "{:02}",
        "day": "{:02}",
    }
    DATASET = pd.DataFrame(columns=[
        "filepath", "source_id", "network", "station", "location", "channel",
        "source_latitude_deg", "source_longitude_deg", "source_depth_km",
        "source_origin_time", "source_magnitude", OGS_C.PHASE_STR,
        OGS_C.TIME_STR, OGS_C.AMPLITUDE_STR, OGS_C.WEIGHT_STR
    ])
    DAYS = np.arange(
        self.start, self.end + OGS_C.ONE_DAY, OGS_C.ONE_DAY,
        dtype='datetime64[D]'
    ).tolist()
    DAYS = [op.UTCDateTime(day).date for day in DAYS]
    for day_ in DAYS:
      if day_ not in self.catalog.picks_:
        continue
      df_picks = self.catalog._load_day("picks", day_)
      df_events = self.catalog._load_day("events", day_)
      if df_picks.empty:
        continue
      for (event_id, station), station_picks in df_picks.groupby(
              [OGS_C.IDX_PICKS_STR, OGS_C.STATION_STR]
      ):
        station_picks = station_picks[station_picks[OGS_C.PHASE_STR].notna()]
        station_picks = station_picks[
            station_picks[OGS_C.WEIGHT_STR].astype(float) <= 2
        ]
        if len(station_picks) <= 1:
          continue
        for _, pick in station_picks.iterrows():
          net, sta, loc = self.get_station_info(pick[OGS_C.STATION_STR])
          picktime = op.UTCDateTime(pick[OGS_C.TIME_STR])
          filepath = Path(
              self.args.waveforms /
              DIR_FMT['year'].format(picktime.year) /
              DIR_FMT['month'].format(picktime.month) /
              DIR_FMT['day'].format(picktime.day) /
              f"{net if net else '*'}.{sta}.*.*__"
              f"{(picktime - OGS_C.PICK_TRAIN_OFFSET
                  ).strftime('%Y%m%dT%H%M%SZ')}__"
              f"{(picktime + OGS_C.PICK_TRAIN_OFFSET
                  ).strftime("%Y%m%dT%H%M%SZ")}.mseed"
          )  # type: ignore
          files_ = glob.glob(str(filepath))
          if not files_:
            print(f"Missing waveform file: {filepath}")
            if self.args.download:
              print("Downloading missing waveform...")
              date_ = picktime.strftime(OGS_C.YYMMDD_FMT)
              time_ = picktime.strftime(OGS_C.TIME_FMT)
              cmd = (
                  # TODO: Improve the command construction
                  f"python {Path(__file__).parent}/ogsdownloader.py -D " +
                  f"{date_} {date_} -c {time_} -d {self.args.waveforms} " +
                  f"-S {sta} "
              ) + ("-N " + net if net else net)  # type: ignore
              print(cmd)
              # TODO: Use subprocess.run() instead of os.system() for better
              # error handling and security.
              os.system(cmd)
              time.sleep(1)
          files_ = glob.glob(str(filepath))
          dataset = []
          for wf_file in files_:
            trace = self.get_clean_trace(op.read(wf_file))
            stats = trace[0].stats
            event = df_events[
                df_events[OGS_C.IDX_EVENTS_STR] == pick[OGS_C.IDX_PICKS_STR]
            ]
            if event.empty:
              print(f"Missing event for pick ID: {pick[OGS_C.IDX_PICKS_STR]}")
              continue
            event_row = event.iloc[0]
            dataset.append({
                "filepath": wf_file,
                "source_id": pick[OGS_C.IDX_PICKS_STR],
                "network": stats.network,
                "station": stats.station,
                "location": stats.location,
                "channel": stats.channel,
                "source_latitude_deg": event_row[OGS_C.LATITUDE_STR],
                "source_longitude_deg": event_row[OGS_C.LONGITUDE_STR],
                "source_depth_km": event_row[OGS_C.DEPTH_STR],
                "source_origin_time": event_row[OGS_C.TIME_STR],
                "source_magnitude": event_row[OGS_C.MAGNITUDE_L_STR],
                OGS_C.TIME_STR: pick[OGS_C.TIME_STR],
                OGS_C.PHASE_STR: pick[OGS_C.PHASE_STR],
                OGS_C.WEIGHT_STR: pick[OGS_C.WEIGHT_STR],
                OGS_C.AMPLITUDE_STR: pick.get(OGS_C.AMPLITUDE_STR, np.nan),
            })
          DATASET = pd.concat([DATASET, pd.DataFrame(dataset)],
                              ignore_index=True)
    DATASET.to_csv("training_dataset.csv", index=False)


def main(args):
  trainer = OGSTrainer(args)

  # Step 1: Build the training dataset from OGS catalog + waveforms
  logger.info("Building training dataset...")
  dataset_df = trainer.build_dataset()

  if dataset_df.empty:
    logger.error("No training data found. Check catalog and waveform paths.")
    return

  # Step 2: Load or create SeisBench dataset
  # If a SeisBench-format dataset already exists, load it directly;
  # otherwise the build_dataset() CSV serves as the data index.
  if trainer.waveforms_path.exists() and trainer.metadata_path.exists():
    logger.info("Loading existing SeisBench dataset from %s",
                trainer.output_dir)
    data = sbd.WaveformDataset(trainer.output_dir)
  else:
    logger.warning(
        "SeisBench HDF5 dataset not found at %s. "
        "The training dataset CSV has been created at %s. "
        "Convert it to SeisBench format using sbd.WaveformDataWriter "
        "before training can proceed.",
        trainer.waveforms_path,
        trainer.output_dir / "training_dataset.csv"
    )
    return

  # Step 3: Train
  logger.info("Starting training: model=%s, dataset=%s, epochs=%d, "
              "batch_size=%d, lr=%s",
              args.model, args.dataset, args.epochs,
              args.batch_size, args.learning_rate)
  trainer.train(data)


if __name__ == "__main__":
  main(parse_arguments())

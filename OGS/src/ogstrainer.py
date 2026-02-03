import os
import glob
import time
import argparse
import numpy as np
import obspy as op
import pandas as pd
import seisbench.data as sbd

from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

import ogsconstants as OGS_C

LEARNING_RATE = 1e-2
EPOCHS = 5
BATCH_SIZE = 256
NUM_WORKERS = 4




def parse_arguments():
  parser = argparse.ArgumentParser(description="Train OGS models")
  parser.add_argument("-C", "--catalog", type=Path, required=True,
                      help="Path to the catalog directory")
  date_group = parser.add_mutually_exclusive_group(required=False)
  date_group.add_argument(
    '-D', "--dates", required=False, metavar=OGS_C.DATE_STD,
    type=OGS_C.is_date, nargs=2, action=OGS_C.SortDatesAction,
    default=[datetime.strptime("240320", OGS_C.YYMMDD_FMT),
             datetime.strptime("240620", OGS_C.YYMMDD_FMT)],
    help="Specify the beginning and ending (inclusive) Gregorian date " \
         "(YYMMDD) range to work with.")
  parser.add_argument("-W", "--waveforms", type=Path, required=True,
                      help="Path to the waveforms directory")
  parser.add_argument("-b", "--batch_size", type=int, default=BATCH_SIZE,
                      help="Batch size for training")
  parser.add_argument("-d", "--download", action="store_true",
                      help="Enable download mode")
  parser.add_argument("-e", "--epochs", type=int, default=EPOCHS,
                      help="Number of training epochs")
  parser.add_argument(
    "-lr", "--learning_rate", type=float, default=LEARNING_RATE,
    help="Learning rate for training")
  parser.add_argument("-w", "--workers", type=int, default=NUM_WORKERS,
                      help="Number of workers for data loading")
  return parser.parse_args()

class OGSTrainer:
  def __init__(self, args):
    self.args = args
    self.catalog: OGS_C.OGSCatalog = OGS_C.OGSCatalog(
      args.catalog,
      start=args.dates[0],
      end=args.dates[1],
      name="Training Catalog"
    )
    self.metadata_path = Path(".") / "metadata.csv"
    self.waveforms_path = Path(".") / "waveforms.hdf5"
    self.start, self.end = args.dates
    self.waveforms = OGS_C.waveforms(args.waveforms, self.start, self.end)
    self.stations = {
      station.split(".")[1] : station for date in self.waveforms.values()
        for station in date.keys()
    }
    add = {}
    for key, val in self.stations.items():
      if len(key) > 4: add[key[:4]] = val
    self.stations.update(add)
    self.download = args.download
    self.dataset()

  def train(self):
    pass

  def get_station_info(self, station):
    sta = station.split(".")[1] # type: ignore
    return self.stations[sta].split(".") \
      if sta in self.stations else ("", sta, "")

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
      "latitude", "longitude", "depth_km", "source_origin_time", "source_magnitude",
      OGS_C.PHASE_STR, OGS_C.TIME_STR, OGS_C.AMPLITUDE_STR, OGS_C.WEIGHT_STR,
      OGS_C.PROBABILITY_STR])
    DAYS = np.arange(self.start, self.end + OGS_C.ONE_DAY, OGS_C.ONE_DAY,
                     dtype='datetime64[D]').tolist()
    DAYS = [op.UTCDateTime(day).date for day in DAYS]
    for day_ in DAYS:
      if day_ not in self.catalog.picks_: continue
      df_picks = self.catalog.get_(day_, "picks")
      df_events = self.catalog.get_(day_, "events")
      if df_picks.empty: continue
      for (event, station), station_picks in df_picks.groupby(
        [OGS_C.IDX_PICKS_STR, OGS_C.STATION_STR]):
        station_picks = station_picks[station_picks[OGS_C.PHASE_STR].notna()]
        station_picks = station_picks[station_picks[OGS_C.WEIGHT_STR].astype(float) <= 2]
        if len(station_picks) <= 1: continue
        print(station_picks)
        for _, pick in station_picks.iterrows():
          net, sta, loc = self.get_station_info(pick[OGS_C.STATION_STR])
          picktime = op.UTCDateTime(pick[OGS_C.TIME_STR])
          # e.g. /Users/admin/Desktop/OGS_Catalog/waveforms/2001/08/08/FV.BAD..HNE__20010808T111251Z__20010808T111451Z.mseed
          filepath = Path(
            self.args.waveforms /
            DIR_FMT['year'].format(picktime.year) /
            DIR_FMT['month'].format(picktime.month) /
            DIR_FMT['day'].format(picktime.day) /
            f"{net if net else '*'}.{sta}.*.*__"
            f"{(picktime - OGS_C.PICK_TRAIN_OFFSET
                ).strftime("%Y%m%dT%H%M%SZ")}__" # type: ignore
            f"{(picktime + OGS_C.PICK_TRAIN_OFFSET
                ).strftime("%Y%m%dT%H%M%SZ")}.mseed"
            ) # type: ignore
          files_ = glob.glob(str(filepath))
          if not files_:
            print(f"Missing waveform file: {filepath}")
            if self.args.download:
              print("Downloading missing waveform...")
              date_ = picktime.strftime(OGS_C.YYMMDD_FMT)
              time_ = picktime.strftime(OGS_C.TIME_FMT)
              cmd = (
                f"python {Path(__file__).parent}/ogsdownloader.py -D " +
                f"{date_} {date_} -c {time_} -d {self.args.waveforms} " +
                f"-S {sta} ") + ("-N " + net if net else net) # type: ignore
              print(cmd)
              os.system(cmd)
              time.sleep(1)
          files_ = glob.glob(str(filepath))
          dataset = []
          for wf_file in files_:
            trace = self.get_clean_trace(op.read(wf_file))
            stats = trace[0].stats
            event = df_events[df_events[OGS_C.IDX_EVENTS_STR] == pick[OGS_C.IDX_PICKS_STR]]
            if event.empty:
              print(f"Missing event for pick ID: {pick[OGS_C.IDX_PICKS_STR]}")
              continue
            event = event.iloc[0]
            dataset.append({
              "filepath": wf_file,
              "source_id": pick[OGS_C.IDX_PICKS_STR],
              "network": stats.network,
              "station": stats.station,
              "location": stats.location,
              "channel": stats.channel,
              "source_latitude_deg": event[OGS_C.LATITUDE_STR],
              "source_longitude_deg": event[OGS_C.LONGITUDE_STR],
              "source_depth_km": event[OGS_C.DEPTH_STR],
              "source_origin_time": event[OGS_C.TIME_STR],
              "source_magnitude": event[OGS_C.MAGNITUDE_L_STR],
              OGS_C.TIME_STR: pick[OGS_C.TIME_STR],
              OGS_C.PHASE_STR: pick[OGS_C.PHASE_STR],
              OGS_C.WEIGHT_STR: pick[OGS_C.WEIGHT_STR],
              OGS_C.AMPLITUDE_STR: pick[OGS_C.AMPLITUDE_STR],
            })
          DATASET = pd.concat([DATASET, pd.DataFrame(dataset)],
                              ignore_index=True)
    DATASET.to_csv("training_dataset.csv", index=False)

def main(args):
  trainer = OGSTrainer(args)
  trainer.train()


if __name__ == "__main__": main(parse_arguments())
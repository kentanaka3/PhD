import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from pathlib import Path
# Set the project folder
PRJ_PATH = Path(os.path.dirname(__file__)).parent
INC_PATH = os.path.join(PRJ_PATH, "inc")
IMG_PATH = os.path.join(PRJ_PATH, "img")
DATA_PATH = os.path.join(PRJ_PATH, "data")
import sys
# Add to path
if INC_PATH not in sys.path: sys.path.append(INC_PATH)
import argparse
import numpy as np
import pandas as pd
# ObsPy
import obspy
from obspy.core.utcdatetime import UTCDateTime
# Seisbench
import seisbench.data as sbd
import seisbench.util as sbu


from constants import *
import parser as prs
import analyzer as ana
import initializer as ini

class TrainConfig:
  def __init__(self, SOURCE : pd.DataFrame, DETECT : pd.DataFrame,
               WAVEFORMS : pd.DataFrame, args : argparse.Namespace):
    self.source = SOURCE
    self.detect = DETECT
    self.waveforms = WAVEFORMS
    self.args = args

def dataset_loader(args : argparse.Namespace):
  assert len(args.weights) == 1
  WEIGHT = args.weights[0]
  DATASET_PATH = Path(DATA_PATH, MODELS_STR, WEIGHT)
  if args.force or (not DATASET_PATH.exists()):
    print(f"Creating dataset for {WEIGHT}")
    dataset_builder(args)
  DATA = sbd.WaveformDataset(DATASET_PATH, WEIGHT)
  if args.verbose and args.force:
    w, m = DATA.get_sample(int(np.random.random() * len(DATA)))

def dataset_builder(args : argparse.Namespace, SOURCE : pd.DataFrame = None,
                    DETECT : pd.DataFrame = None,
                    WAVEFORMS : pd.DataFrame = None) -> Path:
  assert len(args.weights) == 1
  WEIGHT = args.weights[0]
  global DATA_PATH
  DATA_PATH = args.directory.parent
  if WAVEFORMS is None or args.force: WAVEFORMS = ini.waveform_table(args)
  if SOURCE is None or DETECT is None or args.force:
    SOURCE, DETECT = ini.true_loader(args, WAVEFORMS=WAVEFORMS)
  DATASET_PATH = Path(DATA_PATH, MODELS_STR, WEIGHT)
  DATASET_PATH.mkdir(parents=True, exist_ok=True)
  METADATA_PATH = Path(DATASET_PATH, METADATA_STR + CSV_EXT)
  WAVEFORM_PATH = Path(DATASET_PATH, WAVEFORMS_STR + HDF5_EXT)
  with sbd.WaveformDataWriter(METADATA_PATH, WAVEFORM_PATH) as WFW:
    WFW.data_format = {
      "dimension_order": "CW",
      "component_order": "ZNE",
      "measurement": "velocity",
      "unit": "counts",
      "instrument_response": "not restituted",
    }
    for _, SRC in SOURCE.iterrows():
      idx = SRC[ID_STR]
      date = SRC[TIMESTAMP_STR].date
      latitude = SRC[LATITUDE_STR]
      longitude = SRC[LONGITUDE_STR]
      depth = SRC[LOCAL_DEPTH_STR]
      magnitude = SRC[MAGNITUDE_STR]
      event_params = {
        "source_id": idx,
        "source_origin_time": SRC[TIMESTAMP_STR],
        "source_latitude_deg": latitude,
        "source_longitude_deg": longitude,
        "source_depth_km": depth,
        "source_magnitude": magnitude,
        "split": "train"
      }
      waveforms_d = WAVEFORMS[WAVEFORMS[DATE_STR] == date]
      if waveforms_d.empty: continue
      for station, DTC in DETECT[DETECT[ID_STR] == idx].groupby(STATION_STR):
        waveforms = waveforms_d[waveforms_d[STATION_STR] == station]
        if waveforms.empty: continue
        id = waveforms[NETWORK_STR].unique()[0] + PERIOD_STR + station
        STATION_PATH = Path(DATA_PATH, STATION_STR, id + XML_EXT)
        STATION = obspy.read_inventory(STATION_PATH)[0][0]
        if STATION is None: continue
        traces = list(waveforms.index)
        start = DTC[TIMESTAMP_STR].min() - PICK_OFFSET_TRAIN
        end = DTC[TIMESTAMP_STR].max() + PICK_OFFSET_TRAIN
        stream = obspy.Stream()
        for trace in traces:
          if not Path(trace).exists(): continue
          # TODO: Warning msg
          stream += obspy.read(trace, starttime=start, endtime=end,
                               nearest_sample=True)
          stream.resample(SAMPLING_RATE)
        # TODO: If filtered, consider that for TEST must be filtered as well
        stream.detrend(type="linear").filter(type="highpass", freq=1., corners=4,
                                            zerophase=True)
        # TODO: Warning msg
        if len(stream) == 0: continue
        actual_t_start, data, _ = sbu.stream_to_array(
          stream, component_order=WFW.data_format["component_order"])
        trace_params = {
          "station_network_code": stream[-1].stats.network,
          "station_code": stream[-1].stats.station,
          "trace_channel": stream[-1].stats.channel,
          "station_location_code": stream[-1].stats.location,
          "station_latitude_deg":STATION.latitude,
          "station_longitude_deg":STATION.longitude,
          "station_elevation_m":STATION.elevation,
          "trace_sampling_rate_hz": SAMPLING_RATE,
          "trace_start_time": str(actual_t_start)
        }
        for phase, pick in DTC.groupby(PHASE_STR):
          sample = int((pick[TIMESTAMP_STR].iloc[0] - actual_t_start) * SAMPLING_RATE)
          trace_params[f"trace_{phase}_status"] = "manual"
          trace_params[f"trace_{phase}_arrival_sample"] = int(sample)
        WFW.add_trace({**event_params, **trace_params}, data)

def main(args : argparse.Namespace):
  dataset_loader(args)

if __name__ == "__main__": main(ini.parse_arguments())
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from pathlib import Path
# Set the "./../inc" from the script folder
lib_path = os.path.join(Path(os.path.dirname(__file__)).parent, "inc")
import sys
# Add to path
if lib_path not in sys.path: sys.path.append(lib_path)
from constants import *
import re
import json
import torch
import obspy
import pickle
import requests
import argparse
import itertools
import numpy as np
import numba as nb
import pandas as pd
import seisbench.models as sbm
import matplotlib.pyplot as plt
from datetime import timedelta
# TODO: Read Stations XML
import xml.etree.ElementTree as ET
from seisbench.util import PickList
# TODO: Implement downloading data
from obspy.clients.fdsn import Client
from obspy.core.utcdatetime import UTCDateTime

SAMPLING_RATE = 100

DATE_FMT = "%y%m%d"
DATETIME_FMT = "%y%m%d%H%M%S"
ONE_DAY = timedelta(days=1)
PICK_OFFSET = timedelta(seconds=0.5)
ASSOCIATE_OFFSET = timedelta(seconds=1)

# Extensions
PICKLE_EXT = ".pkl"
TORCH_EXT = ".pt"
MSEED_EXT = ".mseed"
JSON_EXT = ".json"
PNG_EXT = ".png"

PRC_MSEED_FMT = "{NETWORK}.{STATION}..{CHANNEL}__{BEGDT}" + MSEED_EXT

MSEED_FMT = "{NETWORK}.{STATION}..{CHANNEL}__{BEGDT}T{BEGTM}Z" \
                                          "__{ENDDT}T{ENDTM}Z" + MSEED_EXT

# Models (Alphabetically Ordered)
EQTRANSFORMER_STR = "EQTransformer"
GPD_STR           = "GPD"
PHASENET_STR      = "PhaseNet"

CLASS_STR = "class"

# Various pre-trained weights for each model (Add if new are available)
MODEL_WEIGHTS_DICT = {
  EQTRANSFORMER_STR : {
    CLASS_STR : sbm.EQTransformer()
  },
  GPD_STR           : {
    CLASS_STR : sbm.GPD()
  },
  PHASENET_STR      : {
    CLASS_STR : sbm.PhaseNet()
  }
}

COLORS = {
  "P": "C0",
  "S": "C1",
  "Detection": "C2"
}

DATA_PATH = "data"
IMG_PATH = "img"
MSEED_STR = "MSEED"

FILENAME_STR = "FILENAME"
NETWORK_STR = "NETWORK"
STATION_STR = "STATION"
CHANNEL_STR = "CHANNEL"
BEG_DATE_STR = "BEGDT"
HEADER = [FILENAME_STR, NETWORK_STR, STATION_STR, CHANNEL_STR, BEG_DATE_STR]

P_TYPE_STR = "P_TYPE"
S_TYPE_STR = "S_TYPE"
P_WEIGHT_STR = "P_WEIGHT"
S_WEIGHT_STR = "S_WEIGHT"
P_TIME_STR = "P_TIME"
S_TIME_STR = "S_TIME"
PHASE_EXTRACTOR = \
  re.compile(fr"^(?P<{STATION_STR}>(\w{{4}}|\w{{3}}\s))"            # Station
             fr"(?P<{P_TYPE_STR}>[ei?]P[cd\s])"                     # P Type
             fr"(?P<{P_WEIGHT_STR}>[0-4])"                          # P Weight
             fr"1(?P<{BEG_DATE_STR}>\d{{10}})"                      # Date
             fr"\s(?P<{P_TIME_STR}>\d{{4}})"                        # P Time
             fr"\s+((?P<{S_TIME_STR}>\d{{4}}|\d{{3}})"              # S Time
             fr"(?P<{S_TYPE_STR}>[ei?]S\s)"                         # S Type
             fr"(?P<{S_WEIGHT_STR}>[0-4]))*")                       # S Weight
EVENT_EXTRACTOR = re.compile(r"^1(\s+D)*\s*$")                      # Event

def event_parser(filename : str) -> dict:
  """
  input  :
    - filename (str)

  output :
    - dictionary (str : value)

  errors :
    - None

  notes  :

  """
  with open(filename, 'r') as fr:
    lines = fr.readlines()
  events = {}
  event = 0
  events.setdefault(event, [])
  for line in [l.strip() for l in lines]:
    if EVENT_EXTRACTOR.match(line):
      event += 1
      events.setdefault(event, [])
      continue
    match = PHASE_EXTRACTOR.match(line)
    if match:
      result = match.groupdict()
      result[BEG_DATE_STR] = UTCDateTime.strptime(result[BEG_DATE_STR],
                                                  "%y%m%d%H%M")
      result[P_WEIGHT_STR] = int(result[P_WEIGHT_STR])
      result[P_TIME_STR] = \
        timedelta(seconds=float(result[P_TIME_STR][:2] + "." + \
                                result[P_TIME_STR][2:]))
      if result[S_TIME_STR]:
        result[S_WEIGHT_STR] = int(result[S_WEIGHT_STR])
        result[S_TIME_STR] = \
          timedelta(seconds=float(result[S_TIME_STR][:2] + "." + \
                                  result[S_TIME_STR][2:]))
      events[event].append(result)
  # with open(os.path.splitext(filename)[0] + JSON_EXT, 'w') as fr:
  #   json.dump(events, fr, indent=2)
  return events

# Pretrained model weights (Alphabetically Ordered)
INSTANCE_STR = "instance"
ORIGINAL_STR = "original"
STEAD_STR = "stead"
SCEDC_STR = "scedc"

# TODO: Run 
# TODO: Study GaMMA associator with folder
# TODO: Colab PyOcto associator to be tested with GaMMA
# TODO: Get Vel Model

MNL_DATA_PATH = os.path.join(DATA_PATH, "manual")
RAW_DATA_PATH = os.path.join(DATA_PATH, "waveforms")
PRC_DATA_PATH = os.path.join(DATA_PATH, "processed")
ANT_DATA_PATH = os.path.join(DATA_PATH, "annotated")
CLF_DATA_PATH = os.path.join(DATA_PATH, "classified")

def is_file_path(string : str) -> str:
  """
  input:
    - string (str)

  output:
    - str

  errors:
    - NotADirectoryError

  notes:

  """
  if os.path.isfile(string): return string
  else: raise NotADirectoryError(string)

def is_date(string) -> UTCDateTime:
  return UTCDateTime.strptime(string, DATE_FMT)

IMG_ANT_OFFSET = 2

def parse_arguments():
  parser = argparse.ArgumentParser(description="Process AdriaArray Dataset")
  parser.add_argument('-C', "--channel", default=None, type=str, nargs='*',
                      help="Specify the Channel to analyze. If file is not "
                           "available, then a key must be provided in order "
                           "to download the data")
  parser.add_argument('-D', "--dates", nargs=2, required=False, type=is_date,
                      metavar="DATE",
                      default=[UTCDateTime.strptime("230601", DATE_FMT),
                               UTCDateTime.strptime("230731", DATE_FMT)],
                      help="Specify the date range to work with. If files are "
                           "not present")
  parser.add_argument('-G', "--groups", nargs='+', required=False, metavar="",
                      default=[BEG_DATE_STR, NETWORK_STR, STATION_STR],
                      help="Analize the data based on a specified list")
  parser.add_argument('-J', "--julian", default=False, action="store_true",
                      help="Transform the selected dates into Julian date.")
  # TODO: Implement data retrieval
  parser.add_argument('-K', "--key", default=None, nargs=1, required=False,
                      type=is_file_path,
                      help="Key to download the data from server.")
  parser.add_argument('-M', "--models", choices=MODEL_WEIGHTS_DICT.keys(),
                      required=False, metavar="", type=str, nargs='+',
                      default=[PHASENET_STR, EQTRANSFORMER_STR],
                      help="Select a specific Machine Learning based model",)
  parser.add_argument('-N', "--network", default=None, type=str, nargs='*',
                      metavar="", required=False,
                      help="Specify the Network to analyze. If file is not "
                           "available, then a key must be provided in order "
                           "to download the data")
  parser.add_argument('-S', "--station", default=None, type=str, nargs='*',
                      metavar="", required=False,
                      help="Specify the Station to analyze. If file is not "
                           "available, then a key must be provided in order "
                           "to download the data")
  parser.add_argument('-T', "--train", default=False, action='store_true')
  parser.add_argument('-W', "--weights", required=False, metavar="", type=str,
                      default=[INSTANCE_STR, ORIGINAL_STR, STEAD_STR,
                               SCEDC_STR], nargs='+',
                      help="Select a specific pretrained weights for the "
                           "selected Machine Learning based model. "
                           "WARNING: Weights which are not available for the "
                           "selected models will not be considered")
  parser.add_argument('-d', "--directory", default=RAW_DATA_PATH, type=str,
                      required=False,
                      help="Directory path to the raw files")
  parser.add_argument('-p', "--pwave", default=0.2, type=float, required=False,
                      help="P wave threshold.")
  parser.add_argument('-s', "--swave", default=0.1, type=float, required=False,
                      help="S wave threshold.")
  # TODO: Add verbose LEVEL
  parser.add_argument('-v', "--verbose", default=False, action='store_true')
  return parser.parse_args()

def waveform_table(args, data_folder = RAW_DATA_PATH):
  """
  input:
    - args        ()
    - data_folder (os.path)

  output:
    - pandas.DataFrame

  errors:
    - None

  notes:
    If the starttime of the trace is 23:00 hrs, then we assume the date to be
    recorded is the next day
  """
  if args.verbose: print("Constructing the Table of Files")
  WAVEFORMS_DATA = []
  for f in os.listdir(data_folder):
    fr = os.path.join(data_folder, f)
    if os.path.isfile(fr):
      try:
        trc = obspy.read(fr, headonly=True, dtype=np.float32)[0].stats
      except:
        continue
      start = UTCDateTime(trc.starttime.date)
      if trc.starttime.hour == 23: start += ONE_DAY
      end = start + ONE_DAY
      outcome = True
      if args.network:
        outcome = outcome and any([n == trc.network for n in args.network])
      if args.station and outcome:
        outcome = outcome and any([n == trc.station for n in args.station])
      if args.channel and outcome:
        outcome = outcome and any([n == trc.channel for n in args.channel])
      if outcome:
        outcome = outcome and (args.dates[0] <= start and end <= args.dates[1])
      if outcome:
        WAVEFORMS_DATA.append([fr, trc.network, trc.station, trc.channel,
                               UTCDateTime.strftime(start, DATE_FMT)])
  return pd.DataFrame(WAVEFORMS_DATA, columns=HEADER).set_index(FILENAME_STR)\
                                                     .groupby(args.groups)

def read_traces(trace_files, data_folder = PRC_DATA_PATH, verbose = False,
                headonly = False) -> obspy.Stream:
  """
  input:
    - trace_files   (pandas.api.typing.DataFrameGroupBy)
    - data_folder   (os.path)
    - verbose       (bool)
    - headonly      (bool)

  output:
    - stream        (obspy.Stream)

  errors:
    - None

  notes:

  """
  if verbose: print("Reading the Traces")
  stream = obspy.Stream()
  for _, row in trace_files.iterrows():
    fpath = os.path.join(data_folder, row[BEG_DATE_STR], row[NETWORK_STR],
                         row[STATION_STR])
    os.makedirs(fpath, exist_ok=True)
    TRC_FILE = os.path.join(fpath,
                            PRC_MSEED_FMT.format(NETWORK=row[NETWORK_STR],
                                                 STATION=row[STATION_STR],
                                                 CHANNEL=row[CHANNEL_STR],
                                                 BEGDT=row[BEG_DATE_STR]))
    if os.path.exists(TRC_FILE):
      if verbose:
        print(f"Found and reading previously processed file {TRC_FILE}")
      stream += obspy.read(TRC_FILE, headonly=headonly, dtype=np.float32)
    else:
      if verbose: print(f"Attempting to read from raw data")
      stream += obspy.read(row.name, headonly=headonly, dtype=np.float32)
      # Clean the stream
      clean_stream(stream)
  return stream

@nb.njit(nogil=True)
def filter_data_(data : np.array) -> bool:
  for d in data:
    if np.isnan(d) or np.isinf(d): return True
  return False

@nb.jit
def filter_data(data : np.array) -> bool:
  # if np.isnan(trc.data).any() or np.isinf(trc.data).any(): return True
  return filter_data_(data)

def clean_stream(stream : obspy.Stream, data_folder = PRC_DATA_PATH,
                 verbose = False) -> None:
  """
  input:
    - stream          (obspy.Stream)
    - data_folder     (os.path)
    - verbose         (bool)

  output:
    - None

  errors:
    - None

  notes:
  
  """
  if verbose: print("Cleaning the Stream")
  for trc in stream:
    start = UTCDateTime(trc.stats.starttime.date)
    if trc.stats.starttime.hour == 23: start += ONE_DAY
    end = start + ONE_DAY
    fpath = os.path.join(data_folder, UTCDateTime.strftime(start, DATE_FMT),
                         trc.stats.network, trc.stats.station)
    os.makedirs(fpath, exist_ok=True)
    TRC_FILE = \
      os.path.join(fpath,
                   PRC_MSEED_FMT.format(NETWORK=trc.stats.network,
                                        STATION=trc.stats.station,
                                        CHANNEL=trc.stats.channel,
                                        BEGDT=UTCDateTime.strftime(start, DATE_FMT)))
    # Remove Stream.Trace if it contains NaN or Inf
    if filter_data(trc.data): stream.remove(trc)
    # Sample has to be 100 Hz
    if trc.stats.sampling_rate != SAMPLING_RATE: trc.resample(SAMPLING_RATE)
    trc.trim(start, end, pad=True, fill_value=0, dtype=np.float32,
             nearest_sample=(trc.stats.starttime.hour != 23))
    trc.write(TRC_FILE, format=MSEED_STR)

def classify_stream(categories : tuple, trace_files : pd.core.frame.DataFrame,
                    model : sbm, x : str, y : str, args,
                    data_folder = CLF_DATA_PATH) -> PickList:
  """
  input:
    - categories    (tuple)
    - trace_files   (pd.core.frame.DataFrame)
    - model         (seisbench.models)
    - x             (str)
    - y             (str)
    - args          ()
    - data_folder   (os.path)
    - verbose       (bool)

  output:
    - output        (seisbench.util.PickList)

  errors:
    - None

  notes:

  """
  fpath = os.path.join(data_folder, *categories, x, y)
  os.makedirs(fpath, exist_ok=True)
  CLF_FILE = os.path.join(fpath, "_".join([*categories, x, y]) + PICKLE_EXT)
  if os.path.isfile(CLF_FILE):
    if args.verbose: print("Found and loading previously classified results")
    output = PickList()
    with open(CLF_FILE, 'rb') as fr:
      while True:
        try:
          output += pickle.load(fr)
        except EOFError:
          break
  else:
    stream = read_traces(trace_files, verbose=args.verbose)
    if args.verbose: print("Classifying the Stream")
    output = model.classify(stream, batch_size=256, P_threshold=args.pwave,
                            S_threshold=args.swave).picks
    pickle.dump(output, open(CLF_FILE, 'wb'))
  if args.verbose:
    print(f"Classification results for model: {x}, with preloaded weight: "
          f"{y}, categorized by {categories}")
    print(output)
  return output

def get_model(x : str, y : str) -> sbm:
  """
  From a given model (x) trained on the dataset (y), return the associated
  testing model.

  input:
    - x (str)
    - y (str)

  output:
    - seisbench.models

  errors:
    - None

  notes:

  """
  try:
    model = MODEL_WEIGHTS_DICT[x][CLASS_STR].from_pretrained(y)
  except:
    if args.verbose:
      print(f"WARNING: Pretrained weights {y} not found for model {x}")
    return None
  print(x, model.weights_docstring)
  return model

def main(args):
  RAW_DATA_PATH = args.directory
  DATA_PATH = os.path.dirname(RAW_DATA_PATH)
  global PRC_DATA_PATH
  PRC_DATA_PATH = os.path.join(DATA_PATH, "processed")
  global ANT_DATA_PATH
  ANT_DATA_PATH = os.path.join(DATA_PATH, "annotated")
  global CLF_DATA_PATH
  CLF_DATA_PATH = os.path.join(DATA_PATH, "classified")
  WAVEFORMS_DATA = waveform_table(args, data_folder=args.directory)
  if not args.train: # Test
    for x, y in list(itertools.product(args.models, args.weights)):
      model = get_model(x, y)
      if model is None: continue
      for categories, trace_files in WAVEFORMS_DATA:
        # Classification
        output = classify_stream(categories, trace_files, model, x, y, args)
        # # Annotation
        # os.makedirs(ANT_DATA_PATH, exist_ok=True)
        # if len(output):
        #   ANT_FILE = os.path.join(ANT_DATA_PATH,
        #                           "_".join([*group[0], x, y]) + PICKLE_EXT)
        #   if not os.path.isfile(ANT_FILE):
        #     annotations = model.annotate(stream)
        #     pickle.dump(annotations, open(ANT_FILE, 'wb'))
        #   else:
        #     if args.verbose:
        #       print("Found and loading previously annotated results for "
        #             f"{x}({y})")
        #     annotations = obspy.Stream()
        #     with open(ANT_FILE, 'rb') as fr:
        #       while True:
        #         try:
        #           annotations += pickle.load(fr)
        #         except EOFError:
        #           break
        #   if args.verbose:
        #     print(f"Annotations results for model: {x}, with preloaded "
        #           f"weight: {y}, grouped by {[*group[0]]}")
        #     print(annotations)
        #   fig = plt.figure(figsize=(15, 10))
        #   axs = fig.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0})
        #   for trc, ant in zip(stream, annotations):
        #     axs[0].plot(trc.times("matplotlib"), trc.data, label=trc.id)
        #     if ant.stats.channel[-1] != "N":  # Do not plot noise curve
        #       axs[1].plot(ant.times("matplotlib"), ant.data, label=ant.id)
        #   axs[0].legend()
        #   axs[1].legend()
        #   fig.suptitle(f"{trc.stats.starttime.date} - {x}({y})")
        #   # Zoom in-out
        #   plt.savefig(os.path.join(IMG_PATH, "_".join([*group[0], x, y]) + \
        #                                      PNG_EXT))
        #   if args.verbose:
        #     plt.show()
        #   plt.close()
  else: # Train
    pass
  return

if __name__ == "__main__":
  args = parse_arguments()
  main(args)
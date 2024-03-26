import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import re
import io
import obspy
import pickle
import requests
import argparse
import itertools
import numpy as np
import pandas as pd
import seisbench.models as sbm
import matplotlib.pyplot as plt

SAMPLING_RATE = 100

DATE_FMT = "{YYYY}{MM:02}{DD:02}"
DAY2SEC = 24 * 60 * 60

# Extensions
MSEED_EXT = ".mseed"
PICKLE_EXT = ".pkl"
JSON_EXT = ".json"

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

DATA_PATH = "data"
IMG_PATH = "img"
MSEED_STR = "MSEED"

FILENAME_STR = "FILENAME"
NETWORK_STR = "NETWORK"
STATION_STR = "STATION"
CHANNEL_STR = "CHANNEL"
BEG_DATE_STR = "BEGDT"
BEG_TIME_STR = "BEGTM"
END_DATE_STR = "ENDDT"
END_TIME_STR = "ENDTM"
MSEED_FILENAME_RGX = re.compile(fr"(?P<{NETWORK_STR}>\w+)\."
                                fr"(?P<{STATION_STR}>\w+)\.\."
                                fr"(?P<{CHANNEL_STR}>\w+)\_\_"
                                fr"(?P<{BEG_DATE_STR}>\d{{8}})T"
                                fr"(?P<{BEG_TIME_STR}>\d{{6}})Z\_\_"
                                fr"(?P<{END_DATE_STR}>\d{{8}})T"
                                fr"(?P<{END_TIME_STR}>\d{{6}})Z{MSEED_EXT}")
HEADER = [NETWORK_STR, STATION_STR, CHANNEL_STR, BEG_DATE_STR, FILENAME_STR]

# Pretrained model weights (Alphabetically Ordered)
INSTANCE_STR = "instance"
ORIGINAL_STR = "original"

RAW_DATA_PATH = os.path.join(DATA_PATH, "waveforms")
PRC_DATA_PATH = os.path.join(DATA_PATH, "processed")
ANT_DATA_PATH = os.path.join(DATA_PATH, "annotated")
CLF_DATA_PATH = os.path.join(DATA_PATH, "classified")
os.makedirs(ANT_DATA_PATH, exist_ok=True)
os.makedirs(CLF_DATA_PATH, exist_ok=True)
os.makedirs(PRC_DATA_PATH, exist_ok=True)
os.makedirs(IMG_PATH, exist_ok=True)

def parse_arguments():
  parser = argparse.ArgumentParser(description="Process AdriaArray Dataset")
  parser.add_argument('-C', "--channel", default=None, type=str, nargs='*',
                      help="Specify the Channel to analyze. If file is not "
                           "available, then a key must be provided in order "
                           "to download the data")
  parser.add_argument('-D', "--dates", nargs=2, required=False, type=str,
                      metavar="DATE", default=["20230601", "20230731"],
                      help="Specify the date range to work with. If files are "
                           "not present")
  parser.add_argument('-G', "--groups", default=[BEG_DATE_STR], nargs='+',
                      required=False, metavar="",
                      help="Analize the data based on a specified "
                           "categorically ordered list")
  parser.add_argument('-K', "--key", default=None, nargs=1, required=False,
                      help="Key to download the data from server.")
  parser.add_argument('-M', "--models", choices=MODEL_WEIGHTS_DICT.keys(),
                      required=False, default=[PHASENET_STR], nargs='+',
                      metavar="", type=str,
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
  parser.add_argument('-W', "--weights", default=[INSTANCE_STR], nargs='+',
                      required=False, metavar="", type=str,
                      help="Select a specific pretrained weights for the "
                           "selected Machine Learning based model. "
                           "WARNING: Weights which are not available for the "
                           "selected models will not be considered")
  parser.add_argument('-v', "--verbose", default=False, action='store_true')
  return parser.parse_args()

def waveform_table(args, data_folder=RAW_DATA_PATH):
  WAVEFORMS_DATA = []
  for f in os.listdir(data_folder):
    if os.path.isfile(os.path.join(data_folder, f)):
      match = MSEED_FILENAME_RGX.match(str(f))
      if match:
        os.makedirs(
          os.path.join(PRC_DATA_PATH, match.group(BEG_DATE_STR),
                       match.group(NETWORK_STR), match.group(STATION_STR)),
          exist_ok=True)
        outcome = True
        if args.network:
          outcome *= True if any([n == match.group(NETWORK_STR)
                                  for n in args.network]) else False
        if args.station:
          outcome *= True if any([n == match.group(STATION_STR)
                                  for n in args.station]) else False
        if args.channel:
          outcome *= True if any([n == match.group(CHANNEL_STR)
                                  for n in args.channel]) else False
        outcome *= True if args.dates[0] <= match.group(BEG_DATE_STR) and \
                           match.group(END_DATE_STR) <= args.dates[1] \
                        else False
        if outcome:
          WAVEFORMS_DATA.append([*match.groups()[:-3], f])
  return pd.DataFrame(WAVEFORMS_DATA,
                      columns=HEADER).sort_values(by=HEADER, ignore_index=True)

def read_traces(group, headonly=False):
  stream = obspy.Stream()
  clean = False
  for _, row in group.iterrows():
    prc_data = os.path.join(PRC_DATA_PATH,
                            PRC_MSEED_FMT.format(NETWORK=row[NETWORK_STR],
                                                 STATION=row[STATION_STR],
                                                 CHANNEL=row[CHANNEL_STR],
                                                 BEGDT=row[BEG_DATE_STR]))
    if os.path.exists(prc_data):
      stream += obspy.read(prc_data, headonly=headonly)
    else:
      stream += obspy.read(os.path.join(RAW_DATA_PATH, row[FILENAME_STR]),
                           headonly=headonly)
      clean = True
  return stream, clean

def main(args):
  WAVEFORMS_DATA = waveform_table(args)
  GROUPS = args.groups
  for x, y in list(itertools.product(args.models, args.weights)):
    try:
      model = MODEL_WEIGHTS_DICT[x][CLASS_STR].from_pretrained(y)
      print(model.weights_docstring)
    except:
      if args.verbose:
        print(f"WARNING: Pretrained weights {y} not found for model {x}")
      continue
    for group in WAVEFORMS_DATA.groupby(GROUPS):
      stream, clean = read_traces(group[1])
      if args.verbose: print(stream, clean)
      # Clean the stream
      if clean:
        for trc in stream:
          start = obspy.UTCDateTime(trc.stats.starttime.date)
          if trc.stats.starttime.hour == 23: start += DAY2SEC
          end = start + DAY2SEC
          d = DATE_FMT.format(YYYY=start.year, MM=start.month, DD=start.day)
          TRC_FILE = os.path.join(PRC_DATA_PATH, d, trc.stats.network,
                                  trc.stats.station,
                                  PRC_MSEED_FMT.format(
                                    NETWORK=trc.stats.network,
                                    STATION=trc.stats.station,
                                    CHANNEL=trc.stats.channel,
                                    BEGDT=start))
          # Remove Stream.Trace if it contains NaN or Inf
          # TODO: Consider optimizing the removal using the following:
          # import numba as nb
          # import numpy as np
          # @nb.njit(nogil=True)
          # def _any_nans(a):
          #   for x in a: if np.isnan(x): return True
          #   return False
          # @nb.jit
          # def any_nans(a):
          #   if not a.dtype.kind == 'f': return False
          #   return _any_nans(a.flat)
          # array1M = np.random.rand(1000000)
          # assert any_nans(array1M) == False
          # %timeit any_nans(array1M)  # 573us
          # array1M[0] = float("nan")
          # assert any_nans(array1M) == True
          # %timeit any_nans(array1M)  # 774ns  (!nanoseconds)
          if np.isnan(trc.data).any() or np.isinf(trc.data).any():
            stream.remove(trc)
          # Sample has to be 100 Hz
          if trc.stats.sampling_rate != SAMPLING_RATE:
            trc.resample(SAMPLING_RATE)
          trc.trim(start, end, pad=True, fill_value=0,
                    nearest_sample=(trc.stats.starttime.hour != 23))
          trc.trim(start, end, pad=True, fill_value=0,
                    nearest_sample=(trc.stats.starttime.hour != 23))
          trc.write(TRC_FILE, format=MSEED_STR)
      stream = stream.merge(method=1, fill_value='interpolate')
      print(stream)
      exit()
      CLF_FILE = os.path.join(CLF_DATA_PATH, "_".join([*group[0], x, y]) + \
                                              PICKLE_EXT)
      if not os.path.isfile(CLF_FILE):
        output = model.classify(stream, batch_size=256, P_threshold=0.2,
                                S_threshold=0.1, parallelism=8).picks
        pickle.dump(output, open(CLF_FILE, 'wb'))
      else:
        output = []
        with open(CLF_FILE, 'rb') as fr:
          while True:
            try:
              output.append(pickle.load(fr))
            except EOFError:
              break
      if not len(output):
        ANT_FILE = os.path.join(ANT_DATA_PATH,
                                "_".join([*group[0], x, y]) + PICKLE_EXT)
        if not os.path.isfile(ANT_FILE):
          annotations = model.annotate(stream, parallelism=8)
          pickle.dump(annotations, open(ANT_FILE, 'wb'))
        else:
          annotations = []
          with open(ANT_FILE, 'rb') as fr:
            while True:
              try:
                annotations.append(pickle.load(fr))
              except EOFError:
                break
        print(annotations)
        fig = plt.figure(figsize=(15, 10))
        axs = fig.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0})
        offset = annotations[0].stats.starttime - stream[0].stats.starttime
        for trc, ant in zip(stream, annotations):
          axs[0].plot(trc.times("matplotlib"), trc.data, label=trc.id)
          #if ant.stats.channel[-1] != "N":  # Do not plot noise curve
          #  axs[1].plot(ant.times() + offset, ant.data, label=ant.id)
        axs[0].legend()
        #axs[1].legend()
        plt.show()
  return

if __name__ == "__main__":
  args = parse_arguments()
  main(args)
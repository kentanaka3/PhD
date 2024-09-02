import os
from pathlib import Path
# Set the project folder
PRJ_PATH = Path(os.path.abspath('')).parent
INC_PATH = os.path.join(PRJ_PATH, "inc")
IMG_PATH = os.path.join(PRJ_PATH, "img")
DATA_PATH = os.path.join(PRJ_PATH, "data")
import sys
# Add to path
if INC_PATH not in sys.path: sys.path.append(INC_PATH)
from constants import *
import AdriaArray as AA
import pickle
import numpy as np
import pandas as pd
from obspy.core.utcdatetime import UTCDateTime

def read_data(path : str):
  # Load the data
  with open(os.path.join(path), 'rb') as f:
    data = pickle.load(f)
  return data

def load_data():
  ANT_PATH = os.path.join(DATA_PATH, ANT_STR)
  DATA = []
  for m in [PHASENET_STR, EQTRANSFORMER_STR]:
    for w in [INSTANCE_STR, ORIGINAL_STR, SCEDC_STR, STEAD_STR]:
      MODEL = AA.get_model(m, w)
      for d in os.listdir(ANT_PATH):
        for n in os.listdir(os.path.join(ANT_PATH, d)):
          for s in os.listdir(os.path.join(ANT_PATH, d, n)):
            f = os.path.join(ANT_PATH, d, n, s, "_".join([d, n, s, m, w]) + "." + PICKLE_EXT)
            data = read_data(f)
            for t in ["P", "S"]:
              for i in np.linspace(0.2, .9, 8):
                DATA.append((m, w, d, n, s, t, i, MODEL.picks_from_annotations(data, threshold=i, phase=t)))
  DATA = pd.DataFrame(DATA, columns=["MODEL", "WEIGHT", "DATE", "NETWORK", "STATION", "WAVE", "THRESHOLD", "DATA"])
  return DATA

def plot_data(DATA):
  x = DATA["DATE"].unique()
  for DAT in DATA.groupby(["MODEL", "WEIGHT", "NETWORK", "STATION", "WAVE"]):
    m, w, n, s, t = DAT[0]
    plt.title(" ".join(DAT[0]))
    for D in DAT[1].groupby(["THRESHOLD"]):
      i = D[0]
      y = np.cumsum([len(res) for res in D[1]["DATA"][:len(x)]])
      plt.plot(x, y, label="_".join([str(i)]))
    plt.yscale("log")
    plt.legend()
    plt.show()
    plt.clf()

def confusion_matrix(DATA):
  for DAT in DATA.groupby(["MODEL", "WEIGHT", "NETWORK", "STATION", "WAVE"]):
    m, w, n, s, t = DAT[0]
    for D in DAT[1].groupby(["THRESHOLD"]):
      i = D[0]
      y = np.cumsum([len(res) for res in D[1]["DATA"]])
      print(m, w, n, s, t, i, y)
  return

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
  with open(filename, 'r') as fr: lines = fr.readlines()
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
      result[P_TIME_STR] = td(seconds=float(result[P_TIME_STR][:2] + "." + \
                                            result[P_TIME_STR][2:]))
      if result[S_TIME_STR]:
        result[S_WEIGHT_STR] = int(result[S_WEIGHT_STR])
        result[S_TIME_STR] = td(seconds=float(result[S_TIME_STR][:2] + "." + \
                                              result[S_TIME_STR][2:]))
      events[event].append(result)
  # with open(os.path.splitext(filename)[0] + "." + JSON_EXT, 'w') as fr:
  #   json.dump(events, fr, indent=2)
  return events

def main():
  DATA = load_data()
  confusion_matrix(DATA)

  return

if __name__ == "__main__": main()
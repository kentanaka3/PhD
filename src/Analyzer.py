import os
from pathlib import Path
# Set the project folder
PRJ_PATH = Path(os.path.dirname(__file__)).parent
INC_PATH = os.path.join(PRJ_PATH, "inc")
IMG_PATH = os.path.join(PRJ_PATH, "img")
DATA_PATH = os.path.join(PRJ_PATH, "data")
import sys
# Add to path
if INC_PATH not in sys.path: sys.path.append(INC_PATH)
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta as td
from obspy.core.utcdatetime import UTCDateTime
from sklearn.metrics import ConfusionMatrixDisplay as ConfMtxDisp

from constants import *
import AdriaArray as AA

def event_merger(ANCHOR : list, ADDITIONAL : list,
                 width : td, PROBABILITIES = None, axis = 0) -> list:
  if PROBABILITIES is None:
    PROBABILITIES = [1.0] * len(ADDITIONAL)
  # Initialize with ANCHOR and no match (False Negative)
  BOXES = [(a, 0.0) for a in ANCHOR]
  i = 0 # Index for ANCHOR
  j = 0 # Index for ADDITIONAL
  while i < len(ANCHOR) and j < len(ADDITIONAL):
    if td(seconds=abs(ANCHOR[i] - ADDITIONAL[j])) < width:
      # ANCHOR[i] and ADDITIONAL[j] are a match (True Positive)
      BOXES[i] = (ANCHOR[i], PROBABILITIES[j])
      i += 1
      j += 1
    elif ADDITIONAL[j] < ANCHOR[i]:
      # ADDITIONAL[j] is not in ANCHOR (False Positive)
      BOXES.append((ADDITIONAL[j], -1 * PROBABILITIES[j]))
      j += 1
    else:
      # ANCHOR[i] is not in ADDITIONAL (False Negative)
      i += 1
  while j < len(ADDITIONAL):
    # ADDITIONAL[j] is not in ANCHOR (False Positive)
    BOXES.append((ADDITIONAL[j], -1 * PROBABILITIES[j]))
    j += 1
  return sorted(BOXES, key=lambda x: x[axis])

def read_data(path : str):
  # Load the data
  with open(os.path.join(path), 'rb') as f:
    data = pickle.load(f)
  return data

def load_data(args : argparse.Namespace) -> pd.DataFrame:
  global DATA_PATH
  DATA_PATH  = Path(args.directory).parent
  CLF_PATH = os.path.join(DATA_PATH, CLF_STR)
  DATA = []
  HEADER = [MODEL_STR, WEIGHT_STR, TIMESTAMP_STR, NETWORK_STR, STATION_STR,
            PHASE_STR, PROBABILITY_STR]
  for model in args.models:
    for weight in args.weights:
      for date in os.listdir(CLF_PATH):
        for network in os.listdir(os.path.join(CLF_PATH, date)):
          for station in os.listdir(os.path.join(CLF_PATH, date, network)):
            f = os.path.join(CLF_PATH, date, network, station,
                             UNDERSCORE_STR.join([date, network, station,
                                                  model, weight]) + PICKLE_EXT)
            for p in read_data(f):
              DATA.append([model, weight, p.peak_time, network, station,
                           p.phase, p.peak_value])
  return pd.DataFrame(DATA, columns=HEADER).sort_values(TIMESTAMP_STR)\
                                           .reset_index(drop=True)

def plot_data(DATA : pd.DataFrame, args : argparse.Namespace) -> None:
  start, end = args.dates
  DATA = DATA[DATA[PHASE_STR] == PWAVE]
  x = [start]
  while x[-1] <= end: x.append(x[-1] + ONE_DAY)
  for DAT in DATA.groupby([MODEL_STR, NETWORK_STR, STATION_STR]):
    _, _axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = _axs.flatten()
    plt.suptitle(SPACE_STR.join(DAT[0]))
    y_max = 0
    for i, DA in enumerate(DAT[1].groupby(WEIGHT_STR)):
      axs[i].set_title(DA[0])
      for threshold in np.linspace(0.2, 0.9, 8):
        D = DA[1]
        y = [D[(D[PROBABILITY_STR] >= threshold) & (D[TIMESTAMP_STR] <= x[i]) &
               (D[TIMESTAMP_STR] >= start)].size for i in range(len(x))]
        axs[i].plot(x, y, label=threshold)
        y_max = max(y_max, max(y))
    for ax in axs:
      ax.set_ylim(0, y_max)
      ax.set_ylabel("Cumulative number of picks")
      ax.set_xlabel("Date")
      ax.grid()
      ax.legend()
    plt.savefig(Path(IMG_PATH, "EC_" + UNDERSCORE_STR.join(DAT[0]) + PNG_EXT))
    if args.verbose: plt.show()

def conf_mtx(TRUE : pd.DataFrame, PRED : pd.DataFrame,
             args : argparse.Namespace) -> pd.DataFrame:
  stations = args.station if (args.station is not None and
                              args.station != ALL_WILDCHAR_STR) else \
             PRED[STATION_STR].unique()
  start, end = args.dates
  N_seconds = int((end - start) / (2 * PICK_OFFSET.total_seconds()))
  TRUE = TRUE[(TRUE[P_TIME_STR] >= start) & (TRUE[P_TIME_STR] <= end) & \
              TRUE[STATION_STR].isin(stations)]
  PRED = PRED[(PRED[TIMESTAMP_STR] >= start) & (PRED[TIMESTAMP_STR] <= end) & \
              PRED[STATION_STR].isin(stations) & (PRED[PHASE_STR] == PWAVE) & \
              (PRED[PROBABILITY_STR] >= args.pwave)]
  T = {}
  for station, dataframe in TRUE.groupby(STATION_STR):
    T[station] = list(dataframe[P_TIME_STR])
  DATA = {}
  # Analyze P waves
  for (station, model, weight), dataframe in \
    PRED.groupby([STATION_STR, MODEL_STR, WEIGHT_STR]):
    DATA.setdefault((model, weight), [])
    DATA[(model, weight)] += \
      event_merger(T[station], dataframe[TIMESTAMP_STR].to_list(),
                   PICK_OFFSET, dataframe[PROBABILITY_STR].to_list())
  DATAFRAME = []
  HEADER = [MODEL_STR, WEIGHT_STR, THRESHOLD_STR, TP_STR, FP_STR, FN_STR,
            TN_STR, ACCURACY_STR, PRECISION_STR, RECALL_STR, F1_STR]
  for (model, weight), values in DATA.items():
    _, axs = plt.subplots(3, 3, figsize=(17, 17))
    plt.suptitle(SPACE_STR.join([model, weight]))
    for threshold, ax in zip(np.linspace(0.2, 1.0, 9), axs.flatten()):
      threshold = round(threshold, 2)
      TP = len([v for v in values if v[1] >= threshold])  # True Positives
      FP = len([v for v in values if v[1] <= -threshold]) # False Positives
      FN = len([v for v in values if v[1] == 0])          # False Negatives
      TN = N_seconds - (TP + FP + FN)                     # True Negatives
      ConfMtxDisp(np.array([[TP, FP], [FN, TN]]),
                  display_labels=[PWAVE, NONE_STR]).plot(ax=ax)
      ax.set_title(str(threshold))
      ACCURACY = (TP + TN) / N_seconds                    # Accuracy
      if TP + FP == 0:
        DATAFRAME.append([model, weight, threshold, TP, FP, FN, TN,
                          ACCURACY, np.nan, np.nan, np.nan])
        continue
      PRECISION = TP / (TP + FP)                          # Precision
      if TP + FN == 0:
        DATAFRAME.append([model, weight, threshold, TP, FP, FN, TN,
                          ACCURACY, PRECISION, np.nan, np.nan])
        continue
      RECALL = TP / (TP + FN)                             # Recall
      if PRECISION + RECALL == 0:
        DATAFRAME.append([model, weight, threshold, TP, FP, FN, TN, ACCURACY,
                          PRECISION, RECALL, np.nan])
        continue
      F1 = 2 * PRECISION * RECALL / (PRECISION + RECALL)  # F1 Score
      DATAFRAME.append([model, weight, threshold, TP, FP, FN, TN,
                        ACCURACY, PRECISION, RECALL, F1])
    IMG_FILE = \
      Path(IMG_PATH, "CM_" + UNDERSCORE_STR.join([model, weight]) + PNG_EXT)
    plt.savefig(IMG_FILE)
    plt.close()
  DATAFRAME = pd.DataFrame(DATAFRAME, columns=HEADER)

  width = 0.35
  TPx = np.arange(len(DATAFRAME[THRESHOLD_STR].unique()))
  FNx = TPx + width
  for (model, weight), dataframe in DATAFRAME.groupby([MODEL_STR, WEIGHT_STR]):
    plt.bar(TPx, dataframe[TP_STR], width, label=TP_STR)
    plt.bar(FNx, dataframe[FN_STR], width, label=FN_STR)
    plt.plot(TPx, dataframe[ACCURACY_STR], label=ACCURACY_STR, color='k')
    plt.plot(TPx, dataframe[PRECISION_STR], label=PRECISION_STR, color='g')
    plt.plot(TPx, dataframe[RECALL_STR], label=RECALL_STR, color='r')
    plt.plot(TPx, dataframe[F1_STR], label=F1_STR, color='b')
    plt.xticks(TPx + width / 2, dataframe[THRESHOLD_STR].unique())
    plt.xlabel(THRESHOLD_STR)
    plt.yscale("log")
    plt.ylabel("Number of picks")
    plt.title(SPACE_STR.join([model, weight]))
    plt.legend()
    plt.savefig(Path(IMG_PATH, "TPFN_" + UNDERSCORE_STR.join([model, weight]) + PNG_EXT))
    plt.close()
  return DATAFRAME

def event_parser(filename : Path) -> pd.DataFrame:
  """
  input  :
    - filename (Path)

  output :
    - pd.DataFrame

  errors :
    - None

  notes  :

  """
  with open(filename, 'r') as fr: lines = fr.readlines()
  HEADER = [EVENT_STR, STATION_STR, P_TYPE_STR, P_WEIGHT_STR, P_TIME_STR,
            S_TYPE_STR, S_WEIGHT_STR, S_TIME_STR]
  DATA = []
  event = 0
  for line in [l.strip() for l in lines]:
    if EVENT_EXTRACTOR.match(line):
      event += 1
      continue
    match = PHASE_EXTRACTOR.match(line)
    if match:
      result = match.groupdict()
      result[BEG_DATE_STR] = UTCDateTime.strptime(result[BEG_DATE_STR],
                                                  "%y%m%d%H%M")
      result[P_WEIGHT_STR] = int(result[P_WEIGHT_STR])
      result[P_TIME_STR] = \
        result[BEG_DATE_STR] + td(seconds=float(result[P_TIME_STR][:2] + \
                                                PERIOD_STR + \
                                                result[P_TIME_STR][2:]))
      if result[S_TIME_STR]:
        result[S_WEIGHT_STR] = int(result[S_WEIGHT_STR])
        result[S_TIME_STR] = \
          result[BEG_DATE_STR] + td(seconds=float(result[S_TIME_STR][:2] + \
                                                  PERIOD_STR + \
                                                  result[S_TIME_STR][2:]))
    DATA.append([event, result[STATION_STR].strip(SPACE_STR),
                 result[P_TYPE_STR], result[P_WEIGHT_STR],
                 result[P_TIME_STR],
                 result[S_TYPE_STR], result[S_WEIGHT_STR],
                 result[S_TIME_STR]])
  # We sort the values by the Primary wave arrival time
  return pd.DataFrame(DATA, columns=HEADER).sort_values(P_TIME_STR)

def main(args : argparse.Namespace):
  TRUE = event_parser(Path(DATA_PATH, "test", "manual", "manual.dat"))
  PRED = load_data(args)
  conf_mtx(TRUE, PRED, args)


if __name__ == "__main__": main(AA.parse_arguments())
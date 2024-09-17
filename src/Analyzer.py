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
import matplotlib.dates as mdates
from datetime import timedelta as td
from obspy.core.utcdatetime import UTCDateTime
from sklearn.metrics import ConfusionMatrixDisplay as ConfMtxDisp

from constants import *
import AdriaArray as AA

def event_merger(ANCHOR : list, ADDITIONAL : list,
                 width : td, PROBABILITIES = None, axis = 0, station = None) \
    -> list:
  if PROBABILITIES is None:
    PROBABILITIES = [1.0] * len(ADDITIONAL)
  if len(ANCHOR) == 0:
    return [(a, -b, None, station) for a, b in zip(ADDITIONAL, PROBABILITIES)]
  # Initialize with ANCHOR and no match (False Negative)
  BOXES = [(a, 0.0, None, station) for a in ANCHOR]
  i = 0 # Index for ANCHOR
  j = 0 # Index for ADDITIONAL
  while i < len(ANCHOR) and j < len(ADDITIONAL):
    if td(seconds=abs(ANCHOR[i] - ADDITIONAL[j])) < width:
      # ANCHOR[i] and ADDITIONAL[j] are a match (True Positive)
      BOXES[i] = (ANCHOR[i], PROBABILITIES[j], ADDITIONAL[j], station)
      i += 1
      j += 1
    elif ADDITIONAL[j] < ANCHOR[i]:
      # ADDITIONAL[j] is not in ANCHOR (False Positive)
      BOXES.append((ADDITIONAL[j], -PROBABILITIES[j], None, station))
      j += 1
    else:
      # ANCHOR[i] is not in ADDITIONAL (False Negative)
      i += 1
  while j < len(ADDITIONAL):
    # ADDITIONAL[j] is not in ANCHOR (False Positive)
    BOXES.append((ADDITIONAL[j], -PROBABILITIES[j], None, station))
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
  start, end = args.dates
  for model in args.models:
    for weight in args.weights:
      for date in os.listdir(CLF_PATH):
        date_obj = UTCDateTime.strptime(date, DATE_FMT)
        if date_obj < start or date_obj > end: continue
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

def plot_data(DATA : pd.DataFrame, args : argparse.Namespace, phase = PWAVE) \
    -> None:
  MSG = f"Cumulative number of {phase} picks"
  if args.verbose: print(MSG)
  start, end = args.dates
  DATA = DATA[(DATA[PHASE_STR] == phase) & (DATA[TIMESTAMP_STR] >= start)]
  x = [start]
  while x[-1] <= end: x.append(x[-1] + ONE_DAY)
  z = [round(t, 2) for t in np.linspace(0.2, 0.9, 8)]
  for model, dataframe in DATA.groupby(MODEL_STR):
    _, _axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = _axs.flatten()
    plt.suptitle(model, fontsize=16)
    axs[0].set(xticklabels=[], xlabel=None, ylabel=MSG)
    axs[1].set(xticklabels=[], xlabel=None, yticklabels=[], ylabel=None)
    axs[2].set(xlabel="Date", ylabel=MSG)
    axs[3].set(xlabel="Date", yticklabels=[], ylabel=None)
    y_max = 0
    for i, (weight, data) in enumerate(dataframe.groupby(WEIGHT_STR)):
      axs[i].set_title(weight)
      for threshold in z:
        y = [data[(data[PROBABILITY_STR] >= threshold) &
                  (data[TIMESTAMP_STR] <= d)].size for d in x]
        axs[i].plot([np.datetime64(t.datetime) for t in x], y, label=threshold)
        y_max = max(y_max, max(y))
    for ax in axs:
      ax.set_ylim(0, y_max)
      ax.grid()
      ax.legend()
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for label in axs[2].get_xticklabels():
      label.set(rotation=30, horizontalalignment='right')
    axs[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for label in axs[3].get_xticklabels():
      label.set(rotation=30, horizontalalignment='right')
    IMG_FILE = Path(IMG_PATH, "CP_" + model + PNG_EXT)
    plt.tight_layout()
    plt.savefig(IMG_FILE)
    plt.close()
    if args.verbose: print(f"Saving {IMG_FILE}")


  for (model, network, station), dataframe in \
    DATA.groupby([MODEL_STR, NETWORK_STR, STATION_STR]):
    _, _axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = _axs.flatten()
    plt.suptitle(SPACE_STR.join([model, network, station]), fontsize=16)
    axs[0].set(xticklabels=[], xlabel=None, ylabel=MSG)
    axs[1].set(xticklabels=[], xlabel=None, yticklabels=[], ylabel=None)
    axs[2].set(xlabel="Date", ylabel=MSG)
    axs[3].set(xlabel="Date", yticklabels=[], ylabel=None)
    y_max = 0
    for i, (weight, data) in enumerate(dataframe.groupby(WEIGHT_STR)):
      axs[i].set_title(weight)
      for threshold in z:
        y = [data[(data[PROBABILITY_STR] >= threshold) &
                  (data[TIMESTAMP_STR] <= d)].size for d in x]
        axs[i].plot([np.datetime64(t.datetime) for t in x], y, label=threshold)
        y_max = max(y_max, max(y))
    for ax in axs:
      ax.set_ylim(0, y_max)
      ax.grid()
      ax.legend()
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for label in axs[2].get_xticklabels():
      label.set(rotation=30, horizontalalignment='right')
    axs[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for label in axs[3].get_xticklabels():
      label.set(rotation=30, horizontalalignment='right')
    IMG_FILE = \
      Path(IMG_PATH,
           "CP_" + UNDERSCORE_STR.join([model, network, station]) + PNG_EXT)
    plt.tight_layout()
    plt.savefig(IMG_FILE)
    plt.close()
    if args.verbose: print(f"Saving {IMG_FILE}")

def conf_mtx(TRUE : pd.DataFrame, PRED : pd.DataFrame,
             args : argparse.Namespace, phase = PWAVE):
  if args.verbose: print("Computing the Confusion Matrix")
  stations = args.station if (args.station is not None and
                              args.station != ALL_WILDCHAR_STR) else \
             PRED[STATION_STR].unique()
  start, end = args.dates
  N_seconds = int((end - start) / (2 * PICK_OFFSET.total_seconds()))
  TRUE = TRUE[(TRUE[P_TIME_STR] >= start) & (TRUE[P_TIME_STR] <= end) & \
              TRUE[STATION_STR].isin(stations)]
  if args.verbose:
    TRUE.to_csv(Path(DATA_PATH, TRUE_STR + CSV_EXT), index=False)
  PRED = PRED[PRED[STATION_STR].isin(stations) & (PRED[PHASE_STR] == phase) & \
              (PRED[PROBABILITY_STR] >= args.pwave)]
  if args.verbose:
    PRED.to_csv(Path(DATA_PATH, PRED_STR + CSV_EXT), index=False)
  T = {}
  for station, dataframe in TRUE.groupby(STATION_STR):
    T[station] = list(dataframe[P_TIME_STR])
  DATA = {}
  # Analyze P waves
  for (station, model, weight), dataframe in \
    PRED.groupby([STATION_STR, MODEL_STR, WEIGHT_STR]):
    # TODO: What if station not in PRED?
    DATA.setdefault((model, weight), [])
    DATA[(model, weight)] += \
      event_merger(T[station] if station in T else [],
                   dataframe[TIMESTAMP_STR].to_list(), PICK_OFFSET,
                   dataframe[PROBABILITY_STR].to_list(), station=station)
  DATAFRAME = []
  HEADER = [MODEL_STR, WEIGHT_STR, THRESHOLD_STR, TP_STR, FP_STR, FN_STR,
            TN_STR, ACCURACY_STR, PRECISION_STR, RECALL_STR, F1_STR]
  z = [round(t, 2) for t in np.linspace(0.2, 0.9, 8)]
  for (model, weight), values in DATA.items():
    _, _axs = plt.subplots(3, 3, figsize=(10, 10))
    axs = _axs.flatten()
    plt.suptitle(SPACE_STR.join([model, weight]), fontsize=16)
    if args.verbose:
      for v in values:
        if v[1] == 0: print("WARNING: ", v[0], v[3], model, weight)
    for threshold, ax in zip(z, axs):
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
    axs[0].set(xticklabels=[], xlabel=None)
    axs[1].set(xticklabels=[], xlabel=None, yticklabels=[], ylabel=None)
    axs[2].set(xticklabels=[], xlabel=None, yticklabels=[], ylabel=None)
    axs[3].set(xticklabels=[], xlabel=None)
    plt.savefig(IMG_FILE)
    plt.close()
  DATAFRAME = pd.DataFrame(DATAFRAME, columns=HEADER)

  width = 0.35
  TPx = np.arange(len(DATAFRAME[THRESHOLD_STR].unique()))
  FNx = TPx + width
  for (model, weight), dataframe in DATAFRAME.groupby([MODEL_STR, WEIGHT_STR]):
    _, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.bar(TPx, dataframe[TP_STR], width, label=TP_STR)
    ax1.bar(FNx, dataframe[FN_STR], width, label=FN_STR)
    ax1.set_xlabel(THRESHOLD_STR)
    ax1.set_ylabel("Number of picks")
    ax1.set_xticks(TPx + width / 2, dataframe[THRESHOLD_STR].unique())
    ax1.legend(loc='lower left')
    # ax2.plot(TPx, dataframe[ACCURACY_STR], label=ACCURACY_STR, color='k')
    # ax2.plot(TPx, dataframe[PRECISION_STR], label=PRECISION_STR, color='g')
    ax2.plot(TPx, dataframe[RECALL_STR], label=RECALL_STR, color='r')
    # ax2.plot(TPx, dataframe[F1_STR], label=F1_STR, color='b')
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')
    plt.title(SPACE_STR.join([model, weight]), fontsize=16)
    IMG_FILE = \
      Path(IMG_PATH, "TPFN_" + UNDERSCORE_STR.join([model, weight]) + PNG_EXT)
    plt.savefig(IMG_FILE)
    plt.close()
  return DATAFRAME, DATA

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

def time_displacement(DATA : pd.DataFrame, args : argparse.Namespace):
  bins = np.linspace(-0.5, 0.5, 21, endpoint=True)
  z = [round(t, 2) for t in np.linspace(0.2, 0.9, 8)]
  for (model, weight), dataframe in DATA.items():
    fig, axs = plt.subplots(figsize=(10, 5))
    plt.title(SPACE_STR.join([model, weight]), fontsize=16)
    for threshold in z:
      x = [v[0] - v[2] for v in dataframe if v[1] >= threshold]
      count, _ = np.histogram(x, bins=bins)
      plt.plot(bins[:-1], count, label=threshold)
    plt.xlim(-0.5, 0.5)
    plt.xlabel("Time Displacement (s)")
    plt.ylim(0)
    plt.ylabel("Number of picks")
    plt.grid()
    plt.legend()
    IMG_FILE = \
      Path(IMG_PATH, "TD_" + UNDERSCORE_STR.join([model, weight]) + PNG_EXT)
    plt.savefig(IMG_FILE)
    plt.close()

def main(args : argparse.Namespace):
  PRED = load_data(args)
  plot_data(PRED, args)
  TRUE = event_parser(Path(DATA_PATH, "manual", "manual.dat"))
  _, DATA = conf_mtx(TRUE, PRED, args)
  time_displacement(DATA, args)

if __name__ == "__main__": main(AA.parse_arguments())
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
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta as td
from obspy.core.utcdatetime import UTCDateTime
from sklearn.metrics import ConfusionMatrixDisplay as ConfMtxDisp

from constants import *
import AdriaArray as AA

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
             args : argparse.Namespace):
  if args.verbose: print("Computing the Confusion Matrix")
  stations = args.station if (args.station is not None and
                              args.station != ALL_WILDCHAR_STR) else \
             PRED[STATION_STR].unique()
  start, end = args.dates
  N_seconds = int((end - start) / (2 * PICK_OFFSET.total_seconds()))
  TRUE = TRUE[(TRUE[TIMESTAMP_STR] >= start) & (TRUE[TIMESTAMP_STR] <= end) & \
              TRUE[STATION_STR].isin(stations)]
  TRUE = TRUE[[STATION_STR, TIMESTAMP_STR, PHASE_STR, WEIGHT_STR]]
  if args.verbose:
    TRUE.to_csv(Path(DATA_PATH, TRUE_STR + CSV_EXT), index=False)
  z = [round(t, 2) for t in np.linspace(0.2, 0.9, 8)]
  PRED = PRED[PRED[STATION_STR].isin(stations)]
  if args.verbose:
    PRED.to_csv(Path(DATA_PATH, PRED_STR + CSV_EXT), index=False)
  TP = []
  FN = []
  for threshold, (model, dataframe_m) in \
    itertools.product(z, PRED.groupby(MODEL_STR)):
    fig, _axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = _axs.flatten()
    plt.suptitle(f"{model} w/ threshold: {threshold}", fontsize=16)
    for ax, (weight, dataframe_w) in zip(axs, dataframe_m.groupby(WEIGHT_STR)):
      ax.set_title(weight)
      CFN_MTX = pd.DataFrame(0, index=[PWAVE, SWAVE, NONE_STR],
                             columns=[PWAVE, SWAVE, NONE_STR], dtype=int)
      for station, dataframe_s in dataframe_w.groupby(STATION_STR):
        TRUE_S = TRUE[TRUE[STATION_STR] == station].reset_index(drop=True)
        PRED_S = dataframe_s[dataframe_s[PROBABILITY_STR] >= threshold]
        PRED_S = PRED_S[[STATION_STR, TIMESTAMP_STR, PHASE_STR,
                         PROBABILITY_STR]].reset_index(drop=True)
        i, j = 0, 0
        while i < TRUE_S.shape[0] and j < PRED_S.shape[0]:
          T = TRUE_S.loc[i]
          P = PRED_S.loc[j]
          if td(seconds=abs(T[TIMESTAMP_STR] -
                            P[TIMESTAMP_STR])) <= PICK_OFFSET:
            # Partial True Positive
            tp = [model, weight, station, threshold,
                  (T[TIMESTAMP_STR], P[TIMESTAMP_STR]),
                  (T[PHASE_STR], P[PHASE_STR]), T[WEIGHT_STR]]
            if T[PHASE_STR] == P[PHASE_STR]: TP.append(tp)
            CFN_MTX.loc[T[PHASE_STR], P[PHASE_STR]] += 1
            i += 1
            j += 1
          elif T[TIMESTAMP_STR] < P[TIMESTAMP_STR]:
            # Partial False Negative
            fn = [model, weight, station, threshold, T[TIMESTAMP_STR],
                  T[PHASE_STR], T[WEIGHT_STR]]
            FN.append(fn)
            if args.verbose:
              print(f"WARNING: Missing prediction by", fn)
            CFN_MTX.loc[T[PHASE_STR], NONE_STR] += 1
            i += 1
          else:
            # Partial False Positive
            CFN_MTX.loc[NONE_STR, P[PHASE_STR]] += 1
            j += 1
        while i < TRUE_S.shape[0]:
          # Partial False Negative
          T = TRUE_S.loc[i]
          fn = [model, weight, station, threshold, T[TIMESTAMP_STR],
                T[PHASE_STR], T[WEIGHT_STR]]
          FN.append(fn)
          if args.verbose:
            print(f"WARNING: Missing prediction by", fn)
          CFN_MTX.loc[T[PHASE_STR], NONE_STR] += 1
          i += 1
        while j < PRED_S.shape[0]:
          # Partial False Positive
          CFN_MTX.loc[NONE_STR, PRED_S.loc[j, PHASE_STR]] += 1
          j += 1
      CFN_MTX.loc[NONE_STR, NONE_STR] = N_seconds - CFN_MTX.sum().sum()
      disp = ConfMtxDisp(CFN_MTX.values, display_labels=CFN_MTX.columns)
      disp.plot(ax=ax, colorbar=False)
      for labels in disp.text_.ravel():
        labels.set(color="#E4007C", fontsize=12, fontweight="bold")
      disp.im_.set(clim=(1, N_seconds), cmap="Blues", norm="log")
    axs[0].set(xlabel=None, xticklabels=[])
    axs[1].set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])
    axs[2].set()
    axs[3].set(ylabel=None, yticklabels=[])
    fig.subplots_adjust(left=0.08, right=1.03, bottom=0.03, top=0.95,
                        wspace=0.08, hspace=0.05)
    fig.colorbar(disp.im_, ax=axs, orientation='vertical',
                 label="Number of Picks")
    disp.im_.set_clim(1, N_seconds)
    plt.rcParams.update({'font.size': 12})
    IMG_FILE = \
      Path(IMG_PATH, "CM_" + UNDERSCORE_STR.join([model, str(threshold)]) + \
           PNG_EXT)
    plt.savefig(IMG_FILE)
    plt.close()
  HEADER = [MODEL_STR, WEIGHT_STR, STATION_STR, THRESHOLD_STR, TIMESTAMP_STR,
            PHASE_STR, PROBABILITY_STR]
  FN = pd.DataFrame(FN, columns=HEADER)
  FN_FILE = Path(DATA_PATH, FN_STR + CSV_EXT)
  FN.to_csv(FN_FILE, index=False)
  HEADER = [MODEL_STR, WEIGHT_STR, STATION_STR, THRESHOLD_STR, TIMESTAMP_STR,
            PHASE_STR, TYPE_STR]
  TP = pd.DataFrame(TP, columns=HEADER)
  # Plot the True Positives, False Negatives histogram and the Recall as a
  # function of the threshold for each model and weight
  m = \
    max(TP.groupby([MODEL_STR, WEIGHT_STR])[THRESHOLD_STR].value_counts().max(),
        FN.groupby([MODEL_STR, WEIGHT_STR])[THRESHOLD_STR].value_counts().max())
  m = (m + 4) // 5 * 5
  for model, weight in itertools.product(args.models, args.weights):
    _, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    plt.suptitle(SPACE_STR.join([model, weight]), fontsize=16)
    tp = TP[(TP[MODEL_STR] == model) & (TP[WEIGHT_STR] == weight)]
    tp = tp[THRESHOLD_STR].value_counts().sort_index()
    fn = FN[(FN[MODEL_STR] == model) & (FN[WEIGHT_STR] == weight)]
    fn = fn[THRESHOLD_STR].value_counts().sort_index()
    RECALL = tp / (tp + fn)
    RECALL.plot(ax=ax2, color='r', label=RECALL_STR, use_index=False)
    TPFN = pd.DataFrame({TP_STR: tp, FN_STR: fn})
    TPFN.plot(kind='bar', ax=ax1, label=[TP_STR, FN_STR])
    ax1.set(xlabel=THRESHOLD_STR, ylabel="Number of Picks")
    ax1.legend(loc='center left')
    ax1.set_ylim(0, m)
    ax2.set(ylabel="Score", ylim=(0, 1))
    ax2.legend(loc='center right')
    IMG_FILE = \
      Path(IMG_PATH, "TPFN_" + UNDERSCORE_STR.join([model, weight]) + PNG_EXT)
    plt.savefig(IMG_FILE)
    plt.close()
  return TP

def event_parser(filename : Path, args : argparse.Namespace) -> pd.DataFrame:
  """
  input  :
    - filename      (Path)
    - args          (argparse.Namespace)

  output :
    - pd.DataFrame

  errors :
    - FileNotFoundError

  notes  :
    | EVENT | STATION | PHASE | BEGDT | WEIGHT |
    --------------------------------------------
  """
  global DATA_PATH
  DATA_PATH = Path(args.directory).parent
  if Path(DATA_PATH, ARGUMENTS_STR + JSON_EXT).exists() and \
     AA.primary_arguments(args) == AA.read_arguments(args):
    # As the arguments are the same, we can use the waveform catalog to search
    # for the waveforms given the events listed
    WAVEFORMS = pd.read_csv(Path(DATA_PATH, WAVEFORMS_STR + CSV_EXT))
  else:
    # As the arguments are different, we need to regenerate the waveform
    # catalog manually taking into consideration the current arguments
    CLF_PATH = Path(DATA_PATH, CLF_STR)
    if args.verbose:
      print("WARNING: Creating catalog based on all existing files in",
            CLF_PATH)
    if not CLF_PATH.exists():
      print(f"ERROR: {CLF_PATH} does not exist")
      raise FileNotFoundError
    WAVEFORMS = []
    for date in os.listdir(CLF_PATH):
      for network in os.listdir(Path(CLF_PATH, date)):
        for station in os.listdir(Path(CLF_PATH, date, network)):
          WAVEFORMS.append([date, station])
    WAVEFORMS = pd.DataFrame(WAVEFORMS, columns=[BEG_DATE_STR, STATION_STR])
  WAVEFORMS[BEG_DATE_STR] = \
    WAVEFORMS[BEG_DATE_STR].apply(lambda x: UTCDateTime.strptime(x, DATE_FMT))
  HEADER = [EVENT_STR, STATION_STR, PHASE_STR, TIMESTAMP_STR, WEIGHT_STR]
  DATA = []
  event = 0
  with open(filename, 'r') as fr: lines = fr.readlines()
  for line in [l.strip() for l in lines]:
    if EVENT_EXTRACTOR.match(line):
      event += 1
      continue
    match = PHASE_EXTRACTOR.match(line)
    if match:
      result = match.groupdict()
      result[BEG_DATE_STR] = UTCDateTime.strptime(result[BEG_DATE_STR],
                                                  "%y%m%d%H%M")
      if WAVEFORMS[
           (WAVEFORMS[STATION_STR] == result[STATION_STR].strip(SPACE_STR)) &
           (WAVEFORMS[BEG_DATE_STR] == result[BEG_DATE_STR].date)].empty:
        if args.verbose:
          print(f"WARNING: {result[STATION_STR]} {result[BEG_DATE_STR]} not "
                "found")
        continue
      result[P_WEIGHT_STR] = int(result[P_WEIGHT_STR])
      result[P_TIME_STR] = \
        result[BEG_DATE_STR] + td(seconds=float(result[P_TIME_STR][:2] + \
                                                PERIOD_STR + \
                                                result[P_TIME_STR][2:]))
      DATA.append([event, result[STATION_STR].strip(SPACE_STR), PWAVE,
                   result[P_TIME_STR], result[P_WEIGHT_STR]])
      if result[S_TIME_STR]:
        result[S_WEIGHT_STR] = int(result[S_WEIGHT_STR])
        result[S_TIME_STR] = \
          result[BEG_DATE_STR] + td(seconds=float(result[S_TIME_STR][:2] + \
                                                  PERIOD_STR + \
                                                  result[S_TIME_STR][2:]))
        DATA.append([event, result[STATION_STR].strip(SPACE_STR), SWAVE,
                     result[S_TIME_STR], result[S_WEIGHT_STR]])
  # We sort the values by the Primary wave arrival time
  return pd.DataFrame(DATA, columns=HEADER).sort_values(TIMESTAMP_STR)

def time_displacement(DATA : pd.DataFrame, args : argparse.Namespace,
                      phase = PWAVE) -> None:
  bins = np.linspace(-0.5, 0.5, 21, endpoint=True)
  z = [round(t, 2) for t in np.linspace(0.2, 0.9, 8)]
  for (model, weight), dataframe in DATA.groupby([MODEL_STR, WEIGHT_STR]):
    fig, axs = plt.subplots(figsize=(10, 5))
    plt.title(SPACE_STR.join([model, weight]), fontsize=16)
    dataframe = dataframe[[THRESHOLD_STR, TIMESTAMP_STR]]
    dataframe[TIMESTAMP_STR] = \
      dataframe[TIMESTAMP_STR].map(lambda x: x[0] - x[1])
    for threshold in z:
      data = dataframe[dataframe[THRESHOLD_STR] == threshold][TIMESTAMP_STR]
      counts, _ = np.histogram(data, bins=bins)
      plt.plot(bins[:-1], counts, label=threshold)
    plt.xlim(-0.5, 0.5)
    plt.xlabel("Time Displacement (s)")
    plt.ylim(0)
    plt.ylabel(f"Number of {phase} picks")
    plt.grid()
    plt.legend()
    IMG_FILE = \
      Path(IMG_PATH, "TD_" + UNDERSCORE_STR.join([model, weight]) + PNG_EXT)
    plt.savefig(IMG_FILE)
    plt.close()

def main(args : argparse.Namespace):
  PRED = load_data(args)
  plot_data(PRED, args)
  TRUE = event_parser(Path(DATA_PATH, "manual", "manual.dat"), args)
  TP = conf_mtx(TRUE, PRED, args)
  print(TP)
  time_displacement(TP, args)

if __name__ == "__main__": main(AA.parse_arguments())
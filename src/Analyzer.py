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
import copy
import pickle
import argparse
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta as td
from obspy.core.utcdatetime import UTCDateTime
from sklearn.metrics import ConfusionMatrixDisplay as ConfMtxDisp

from constants import *
import Picker as Pkr

def plot_data(TRUE : pd.DataFrame, PRED : pd.DataFrame,
              args : argparse.Namespace, phase = PWAVE) -> None:
  """
  input  :
    - TRUE          (pd.DataFrame)
    - PRED          (pd.DataFrame)
    - args          (argparse.Namespace)
    - phase         (str)

  output :
    - None

  errors :
    - AttributeError

  notes  :
    The data is plotted for each model and weight. The plots are saved in the
    img directory.
  """
  MSG = f"Cumulative number of {phase} picks"
  if args.verbose: print(MSG)
  start, end = args.dates
  PRED = PRED[(PRED[PHASE_STR] == phase) & (PRED[TIMESTAMP_STR] >= start)]
  TRUE = TRUE[(TRUE[PHASE_STR] == phase)].reset_index(drop=True)
  x = [start]
  while x[-1] <= end: x.append(x[-1] + ONE_DAY)
  z = [round(t, 2) for t in np.linspace(0.2, 0.9, 8)]

  y_true = [len(TRUE[TRUE[TIMESTAMP_STR] <= d].index) for d in x]

  for model, dataframe in PRED.groupby(MODEL_STR):
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
        y = [len(data[(data[PROBABILITY_STR] >= threshold) &
                      (data[TIMESTAMP_STR] <= d)].index) for d in x]
        axs[i].plot([np.datetime64(t.datetime) for t in x], y, label=threshold)
        y_max = max(y_max, max(y))
      axs[i].plot([np.datetime64(t.datetime) for t in x], y_true, label="True",
                  color="k")
      y_max = max(y_max, max(y_true))
    for ax in axs:
      ax.set(xlim=(x[0], x[-1]), ylim=(1, y_max), yscale="log")
      ax.grid()
      ax.legend()
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for label in axs[2].get_xticklabels():
      label.set(rotation=30, horizontalalignment='right')
    axs[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for label in axs[3].get_xticklabels():
      label.set(rotation=30, horizontalalignment='right')
    IMG_FILE = \
      Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
           UNDERSCORE_STR.join([CMTV_PICKS_STR, model]) + PNG_EXT)
    plt.tight_layout()
    plt.savefig(IMG_FILE)
    plt.close()
    if args.verbose: print(f"Saving {IMG_FILE}")

  for (model, network, station), dataframe in \
    PRED.groupby([MODEL_STR, NETWORK_STR, STATION_STR]):
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
        y = [len(data[(data[PROBABILITY_STR] >= threshold) &
                      (data[TIMESTAMP_STR] <= d)].index) for d in x]
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
      Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
           UNDERSCORE_STR.join([CMTV_PICKS_STR, model, network, station]) + \
           PNG_EXT)
    plt.tight_layout()
    plt.savefig(IMG_FILE)
    plt.close()
    if args.verbose: print(f"Saving {IMG_FILE}")

def dist_balanced(T : pd.Series, P : pd.Series) -> float:
  return (dist_time(T, P) + dist_phase(T, P)) / 2.

def dist_phase(T : pd.Series, P : pd.Series) -> float:
  return int(P[PHASE_STR] == T[PHASE_STR])

def dist_time(T : pd.Series, P : pd.Series) -> float:
  return 1. - (P[TEMPORAL_STR] / PICK_OFFSET)

def dist_default(T : pd.Series, P : pd.Series) -> float:
  return dist_balanced(T, P)

def recall(TRUE : pd.DataFrame, PRED : pd.DataFrame, model_name : str,
           dataset_name : str, threshold : float, args : argparse.Namespace) \
      -> pd.DataFrame:
  """
  input  :
    - TRUE          (pd.DataFrame)
    - PRED          (pd.DataFrame)
    - model_name    (str)
    - dataset_name  (str)
    - threshold     (float)
    - args          (argparse.Namespace)

  output :
    - pd.DataFrame
    - list
    - list
    - list

  errors :
    - AttributeError

  notes  :

  """
  N = len(TRUE.index)
  G = nx.Graph()
  # All Predictions are initialized as False Positives
  G.add_nodes_from([(i + N, {PHASE_STR : P[PHASE_STR],
                             STATION_STR : P[STATION_STR],
                             STATUS_STR : FP_STR})
                    for i, P in PRED.iterrows()], bipartite=1)
  start, _ = args.dates
  pos = {i + N : (P[TIMESTAMP_STR] - start, 1) for i, P in PRED.iterrows()}
  for i, T in TRUE.iterrows():
    # All True are initialized as False Negatives
    G.add_nodes_from([(i, {PHASE_STR : T[PHASE_STR],
                           STATION_STR : T[STATION_STR],
                           STATUS_STR : FN_STR})], bipartite=0)
    pos[i] = (T[TIMESTAMP_STR] - start, 0)
    PRED[TEMPORAL_STR] = (PRED[TIMESTAMP_STR] - T[TIMESTAMP_STR])\
                           .apply(lambda x : td(seconds=abs(x)))
    # TODO: Consider H71 error interval
    # PICKS = PRED[PRED[TEMPORAL_STR] < H71_OFFSET[T[WEIGHT_STR]]]
    PICKS = PRED[PRED[TEMPORAL_STR] <= PICK_OFFSET]
    if PICKS.empty: continue
    # If there are picks within the PICK_OFFSET, we change the status of the
    # True and Predicted picks to True Positives and we add the corresponding
    # edges to the graph
    G.nodes[i][STATUS_STR] = TP_STR
    for j, P in PICKS.iterrows():
      G.add_edge(i, j + N, weight=dist_default(T, P))
      G.nodes[j + N][STATUS_STR] = TP_STR
  LINKS = nx.max_weight_matching(G)
  for node in G.nodes:
    # As there are more Predicted picks than True picks, we only traverse the
    # True picks of the graph and remove the edges that are not part of
    # the maximum weight matching
    if node >= N: continue
    for neighbor in copy.deepcopy(G.neighbors(node)):
      #       TRUE, PRED
      edge = (node, neighbor)
      if edge not in LINKS: G.remove_edge(*edge)
  TP, FN, FP = [], [], []
  tags = [PWAVE, SWAVE, NONE_STR]
  CFN_MTX = pd.DataFrame(0, index=tags, columns=tags, dtype=int)
  for node in G.nodes:
    deg = nx.degree(G, node)
    if not deg: G.nodes[node][STATUS_STR] = FN_STR if node < N else FP_STR
    elif deg > 1: raise AttributeError("The graph is not a matching")
    # TRUE picks
    if node < N:
      t = TRUE.iloc[node]
      if G.nodes[node][STATUS_STR] == TP_STR:
        p = PRED.iloc[list(G.neighbors(node))[0] - N]
        CFN_MTX.loc[t[PHASE_STR], p[PHASE_STR]] += 1
        if t[PHASE_STR] == p[PHASE_STR]:
          TP.append([model_name, dataset_name, t[STATION_STR], t[PHASE_STR],
                     threshold, (t[TIMESTAMP_STR], p[TIMESTAMP_STR]),
                     t[WEIGHT_STR]])
      elif G.nodes[node][STATUS_STR] == FN_STR:
        FN.append([model_name, dataset_name, t[STATION_STR], t[PHASE_STR],
                   threshold, t[TIMESTAMP_STR], t[WEIGHT_STR]])
        CFN_MTX.loc[TRUE.iloc[node][PHASE_STR], NONE_STR] += 1
      else:
        raise AttributeError(f"The TRUE node is not a {TP_STR} or {FN_STR}")
    # PRED picks
    else:
      p = PRED.iloc[node - N]
      # As the graph is a matching, the Predicted picks that are not part of
      # the matching are False Positives
      if G.nodes[node][STATUS_STR] == FP_STR:
        FP.append([model_name, dataset_name, p[STATION_STR], p[PHASE_STR],
                   threshold, p[TIMESTAMP_STR], p[PROBABILITY_STR]])
        CFN_MTX.loc[NONE_STR, PRED.iloc[node - N][PHASE_STR]] += 1
  if False and args.verbose:
    node_color = [COLOR_ENCODING[G.nodes[node][STATUS_STR]]\
                    [G.nodes[node][PHASE_STR]] for node in G.nodes]
    fig, ax = plt.subplots(figsize=(15, 2))
    nx.draw(G, pos=pos, ax=ax, node_color=node_color, edge_color='black',
            width=2, node_size=10)
    ax.axis('on')
    ax.tick_params(bottom=True, labelbottom=True)
    for node in G.nodes:
      if node < N:
        ax.axvline(x=pos[node][0] + PICK_OFFSET.seconds, color='k', linestyle='--')
        ax.axvline(x=pos[node][0] - PICK_OFFSET.seconds, color='k', linestyle='--')
    ax.grid()
    ax.set_title(SPACE_STR.join([model_name, dataset_name]))
    fig.tight_layout()
    plt.show()
    plt.close(fig=fig)
  return CFN_MTX, TP, FN, FP

def conf_mtx(TRUE : pd.DataFrame, PRED : pd.DataFrame,
             args : argparse.Namespace) -> pd.DataFrame:
  """
  input  :
    - TRUE          (pd.DataFrame)
    - PRED          (pd.DataFrame)
    - args          (argparse.Namespace)

  output :
    - pd.DataFrame

  errors :
    - AttributeError

  notes  :
    | MODEL | WEIGHT | STATION | THRESHOLD | PHASE | TIMESTAMP | TYPE |
    -------------------------------------------------------------------
  """
  if args.verbose: print("Computing the Confusion Matrix")
  start, end = args.dates
  N_seconds = int((end - start) / (2 * PICK_OFFSET.total_seconds()))
  z = [round(t, 2) for t in np.linspace(0.2, 0.9, 8)]
  TP, FN, FP = [], [], []
  for threshold, (model, dataframe_m) in \
    itertools.product(z, PRED.groupby(MODEL_STR)):
    fig, _axs = plt.subplots(2, 2, figsize=(10, 9))
    axs = _axs.flatten()
    plt.rcParams.update({'font.size': 12})
    for ax, (weight, dataframe_w) in zip(axs, dataframe_m.groupby(WEIGHT_STR)):
      ax.set_title(weight)
      tags = [PWAVE, SWAVE, NONE_STR]
      CFN_MTX = pd.DataFrame(0, index=tags, columns=tags, dtype=int)
      for station, dataframe_s in dataframe_w.groupby(STATION_STR):
        TRUE_S = TRUE[TRUE[STATION_STR] == station].reset_index(drop=True)
        PRED_S = dataframe_s[dataframe_s[PROBABILITY_STR] >= threshold]\
                  .reset_index(drop=True)
        cfn_mtx, tp, fn, fp = recall(TRUE_S, PRED_S, model, weight, threshold,
                                     args)
        CFN_MTX += cfn_mtx
        TP.extend(tp)
        FN.extend(fn)
        FP.extend(fp)
      CFN_MTX.loc[NONE_STR, NONE_STR] = N_seconds - CFN_MTX.sum().sum()
      disp = ConfMtxDisp(CFN_MTX.values, display_labels=CFN_MTX.columns)
      disp.plot(ax=ax, colorbar=False)
      for labels in disp.text_.ravel():
        labels.set(color="#E4007C", fontsize=12, fontweight="bold")
      disp.im_.set(clim=(1, N_seconds), cmap="Blues", norm="log")
    axs[0].set()
    axs[1].set(ylabel=None, yticklabels=[])
    axs[2].set(title=None)
    axs[2].set_xlabel(args.weights[2], fontsize=14)
    axs[2].xaxis.tick_top()
    axs[3].set(ylabel=None, yticklabels=[], title=None)
    axs[3].set_xlabel(args.weights[3], fontsize=14)
    axs[3].xaxis.tick_top()
    fig.subplots_adjust(left=0.08, right=1.08, top=.95, bottom=0.05,
                        wspace=0.1, hspace=0.2)
    fig.colorbar(disp.im_, ax=axs, orientation='vertical',
                 label="Number of Picks", aspect=50, shrink=0.8)
    disp.im_.set_clim(1, N_seconds)
    IMG_FILE = Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                    UNDERSCORE_STR.join([CFN_MTX_STR, model, str(threshold)]) +
                    PNG_EXT)
    plt.savefig(IMG_FILE)
    plt.close()
  HEADER = [MODEL_STR, WEIGHT_STR, STATION_STR, PHASE_STR, THRESHOLD_STR,
            TIMESTAMP_STR, TYPE_STR]
  # True Positives
  TP = pd.DataFrame(TP, columns=HEADER)
  HEADER = [MODEL_STR, WEIGHT_STR, STATION_STR, PHASE_STR, THRESHOLD_STR,
            TIMESTAMP_STR, PROBABILITY_STR]
  # False Negatives
  FN = pd.DataFrame(FN, columns=HEADER)
  FN_FILE = Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                 FN_STR + CSV_EXT)
  FN.to_csv(FN_FILE, index=False)
  # False Positives
  FP = pd.DataFrame(FP, columns=HEADER)
  FP_FILE = Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                 FP_STR + CSV_EXT)
  FP.to_csv(FP_FILE, index=False)
  # Plot the True Positives, False Negatives histogram and the Recall as a
  # function of the threshold for each model and weight
  groups = [MODEL_STR, WEIGHT_STR, PHASE_STR]
  m = max(TP.groupby(groups)[THRESHOLD_STR].value_counts().max(),
          FN.groupby(groups)[THRESHOLD_STR].value_counts().max())
  m = (m + 9) // 10 * 10
  for model in args.models:
    _, _axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = _axs.flatten()
    for ax1, weight in zip(axs, args.weights):
      ax2 = ax1.twinx()
      ax1.set_title(weight, fontsize=16)
      # True Positives
      tp = TP[(TP[MODEL_STR] == model) & (TP[WEIGHT_STR] == weight)]
      # P True Positives
      ptp = tp[tp[PHASE_STR] == PWAVE]
      ptp = ptp[THRESHOLD_STR].value_counts().sort_index()
      # S True Positives
      stp = tp[tp[PHASE_STR] == SWAVE]
      stp = stp[THRESHOLD_STR].value_counts().sort_index()
      # False Negatives
      fn = FN[(FN[MODEL_STR] == model) & (FN[WEIGHT_STR] == weight)]
      # P False Negatives
      pfn = fn[fn[PHASE_STR] == PWAVE]
      pfn = pfn[THRESHOLD_STR].value_counts().sort_index()
      # S False Negatives
      sfn = fn[fn[PHASE_STR] == SWAVE]
      sfn = sfn[THRESHOLD_STR].value_counts().sort_index()
      pRECALL = ptp / (ptp + pfn)
      sRECALL = stp / (stp + sfn)
      RECALL = (ptp + stp) / ((ptp + pfn) + (stp + sfn))
      pRECALL.plot(ax=ax2, label=PWAVE, use_index=False, color="r")
      sRECALL.plot(ax=ax2, label=SWAVE, use_index=False, color="b")
      RECALL.plot(ax=ax2, label=f"{PWAVE} + {SWAVE}", use_index=False,
                  color="k")
      TPFN = pd.DataFrame({
        SPACE_STR.join([PWAVE, TP_STR]): ptp,
        SPACE_STR.join([SWAVE, TP_STR]): stp,
        SPACE_STR.join([PWAVE, FN_STR]): pfn,
        SPACE_STR.join([SWAVE, FN_STR]): sfn
      })
      TPFN.plot(kind='bar', ax=ax1, width=0.7)
      ax1.set(ylabel="Number of Picks", ylim=(0, m))
      ax2.set(ylim=(0, 1))
      yticks, yticklabels = ax2.get_yticks(), ax2.get_yticklabels()
      ax2.set(yticks=[], yticklabels=[])
      ax1.grid()
      ax1.get_legend().remove()
    axs[0].set(xlabel=None, xticklabels=[])
    axs[0].legend()
    axs[1].set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])
    ax1 = axs[1].twinx()
    ax1.set(ylabel=RECALL_STR)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabels)
    axs[2].set()
    axs[3].set(xlabel=THRESHOLD_STR, ylabel=None, yticklabels=[])
    ax2.set_ylabel(RECALL_STR)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticklabels)
    ax2.legend()
    IMG_FILE = Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                    UNDERSCORE_STR.join(["TPFN", model]) + PNG_EXT)
    plt.tight_layout()
    plt.savefig(IMG_FILE)
    plt.close()

  m = max(m, FP.groupby(groups)[THRESHOLD_STR].value_counts().max())
  m = (m + 9) // 10 * 10
  for model, phase in itertools.product(args.models, [PWAVE, SWAVE]):
    _, _axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = _axs.flatten()
    for ax1, weight in zip(axs, args.weights):
      ax2 = ax1.twinx()
      ax1.set_title(weight, fontsize=16)
      tp = TP[(TP[PHASE_STR] == phase) & (TP[WEIGHT_STR] == weight)]
      tp = tp[THRESHOLD_STR].value_counts().sort_index()
      fn = FN[(FN[PHASE_STR] == phase) & (FN[WEIGHT_STR] == weight)]
      fn = fn[THRESHOLD_STR].value_counts().sort_index()
      fp = FP[(FP[PHASE_STR] == phase) & (FP[WEIGHT_STR] == weight)]
      fp = fp[THRESHOLD_STR].value_counts().sort_index()
      PRECISION = tp / (tp + fp)
      RECALL = tp / (tp + fn)
      F1 = 2 * (PRECISION * RECALL) / (PRECISION + RECALL)
      RECALL.plot(ax=ax2, label=RECALL_STR, use_index=False, color="b")
      F1.plot(ax=ax2, label=F1_STR, use_index=False, color="r")
      TPFNFP = pd.DataFrame({
        TP_STR: tp,
        FN_STR: fn,
        FP_STR: fp
      })
      TPFNFP.plot(kind='bar', ax=ax1, width=0.7)
      ax1.set(ylabel="Number of Picks", ylim=(1, m), yscale="log")
      ax2.set(ylim=(0, 1))
      yticks, yticklabels = ax2.get_yticks(), ax2.get_yticklabels()
      ax2.set(yticks=[], yticklabels=[])
      ax1.grid()
      ax1.get_legend().remove()
    axs[0].set(xlabel=None, xticklabels=[])
    axs[0].legend()
    axs[1].set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])
    ax1 = axs[1].twinx()
    ax1.set_ylabel("Score")
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabels)
    axs[2].set()
    axs[3].set(xlabel=THRESHOLD_STR, ylabel=None, yticklabels=[])
    ax2.set_ylabel("Score")
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticklabels)
    ax2.legend()
    IMG_FILE = \
      Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
           UNDERSCORE_STR.join(["TPFNFP", model, phase]) + PNG_EXT)
    plt.tight_layout()
    plt.savefig(IMG_FILE)
  return TP

def event_parser(filename : Path, stations : list, args : argparse.Namespace) \
      -> pd.DataFrame:
  """
  input  :
    - filename      (Path)
    - stations      (list)
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
     Pkr.primary_arguments(args) == Pkr.read_arguments(args):
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
    for date_path in CLF_PATH.iterdir():
      date = UTCDateTime.strptime(date_path.name, DATE_FMT)
      for network_path in date_path.iterdir():
        for station_path in network_path.iterdir():
          WAVEFORMS.append([date, station_path.name])
    WAVEFORMS = pd.DataFrame(WAVEFORMS, columns=[BEG_DATE_STR, STATION_STR])
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
  DATA = pd.DataFrame(DATA, columns=HEADER).sort_values(TIMESTAMP_STR)
  start, end = args.dates
  DATA = DATA[(DATA[TIMESTAMP_STR] >= start) &
              (DATA[TIMESTAMP_STR] < end + ONE_DAY) & \
              DATA[STATION_STR].isin(stations)]
  return DATA

def time_displacement(DATA : pd.DataFrame, args : argparse.Namespace,
                      phase = PWAVE) -> None:
  """
  input  :
    - DATA          (pd.DataFrame)
    - args          (argparse.Namespace)
    - phase         (str)

  output :
    - None

  errors :
    - AttributeError

  notes  :
    | MODEL | WEIGHT | PHASE | THRESHOLD | TIMESTAMP |
    --------------------------------------------------
  """
  DATA[TIMESTAMP_STR] = DATA[TIMESTAMP_STR].map(lambda x: x[0] - x[1])
  bins = np.linspace(-0.5, 0.5, 21, endpoint=True)
  z = [round(t, 2) for t in np.linspace(0.2, 0.9, 8)]
  groups = [MODEL_STR, WEIGHT_STR, PHASE_STR, THRESHOLD_STR]
  m = 0
  for groups, dataframe in DATA.groupby(groups):
    counts, _ = np.histogram(dataframe[TIMESTAMP_STR], bins=bins)
    m = max(m, max(counts))
  m = (m + 9) // 10 * 10
  for model, phase in itertools.product(args.models, [PWAVE, SWAVE]):
    fig, _axs = plt.subplots(2, 2, figsize=(10, 7))
    axs = _axs.flatten()
    dataframe_mp = DATA[(DATA[MODEL_STR] == model) &
                        (DATA[PHASE_STR] == phase)]
    for ax, (weight, dataframe) in zip(axs, dataframe_mp.groupby(WEIGHT_STR)):
      ax.set_title(weight)
      for threshold in z:
        data = dataframe[dataframe[THRESHOLD_STR] == threshold][TIMESTAMP_STR]
        counts, _ = np.histogram(data, bins=bins)
        ax.plot(bins[:-1], counts, label=threshold)
      ax.set(xlim=(-0.5, 0.5), ylim=(0, m))
      ax.grid()
      ax.legend()
    xlabel = "Time Displacement (s)"
    ylabel = f"Number of {phase} picks"
    axs[0].set(xlabel=None, xticklabels=[], ylabel=ylabel)
    axs[1].set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])
    axs[2].set(xlabel=xlabel, ylabel=ylabel)
    axs[3].set(xlabel=xlabel, ylabel=None, yticklabels=[])
    IMG_FILE = \
      Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
           UNDERSCORE_STR.join([TIME_DSPLCMT_STR, model, phase]) + PNG_EXT)
    plt.tight_layout()
    plt.savefig(IMG_FILE)
    plt.close()

def main(args : argparse.Namespace):
  PRED = Pkr.load_data(args)
  stations = args.station if (args.station is not None and
                              args.station != ALL_WILDCHAR_STR) else \
             PRED[STATION_STR].unique()
  TRUE = event_parser(Path(DATA_PATH, "manual.dat"), stations, args)
  if args.verbose:
    TRUE.to_csv(Path(DATA_PATH, TRUE_STR + CSV_EXT), index=False)
    PRED.to_csv(Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                     PRED_STR + CSV_EXT), index=False)
  plot_data(copy.deepcopy(TRUE), copy.deepcopy(PRED), args)
  TP = conf_mtx(copy.deepcopy(TRUE), copy.deepcopy(PRED), args)
  time_displacement(TP, args)

if __name__ == "__main__": main(Pkr.parse_arguments())
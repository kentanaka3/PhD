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
import initializer as ini
import parser as prs

THRESHOLDS : list[float] = [round(t, 2) for t in np.linspace(0.1, 0.9, 9)]
THRESHOLDER_STR = PWAVE + "{p:0.2}" + SWAVE + "{s:0.2}"
DATES = None

def plot_cluster(PICK : pd.DataFrame, RECD : pd.DataFrame,
                 args : argparse.Namespace) -> None:
  """
  input  :
    - PICK          (pd.DataFrame)
    - RECD          (pd.DataFrame)
    - args          (argparse.Namespace)

  output :
    - None

  errors :
    - AttributeError

  notes  :
    The data is plotted for each model and weight. The plots are saved in the
    img directory.
  """
  PICK = PICK[PICK[MODEL_STR].isin(args.models) &
              PICK[WEIGHT_STR].isin(args.weights)].reset_index(drop=True)
  RECD = RECD[(RECD[MODEL_STR].isin(args.models)) &
              (RECD[WEIGHT_STR].isin(args.weights)) &
              (RECD[THRESHOLD_STR] >= min(args.pwave,
                                          args.swave))].reset_index(drop=True)
  Ws : int = len(args.weights)
  x : int = int(np.sqrt(Ws))
  y : int = Ws // x + int((Ws % x) != 0)
  for model, dataframe_m in PICK.groupby(MODEL_STR):
    fig, _axs = plt.subplots(x, y, figsize=(int(y * Ws) * 1.5,
                                            int(x * Ws - 1) * 1.5))
    axs = _axs.flatten()
    fig.suptitle(model)
    plt.rcParams.update({'font.size': 12})
    for ax, (weight, dataframe_w) in zip(axs, dataframe_m.groupby(WEIGHT_STR)):
      X = list()
      Y = list()
      Z = list()
      C = list()
      for station, P in dataframe_w.groupby(STATION_STR):
        R = RECD[(RECD[MODEL_STR] == model) & (RECD[WEIGHT_STR] == weight) &
                 (RECD[STATION_STR] == station)].reset_index(drop=True)
        X.append(P.size)
        Y.append(R.size)
        Z.append(station)
        C.append(R[THRESHOLD_STR].to_list())
      c = [np.asarray(i).mean() if i else 0 for i in C]
      disp = ax.scatter(X, Y, c=c)
      disp.set(clim=(0.1, 1), cmap="turbo", norm="log")
      for i, txt in enumerate(Z): ax.annotate(txt, (X[i] + .5, Y[i] + .5))
      ax.set(title=weight + SPACE_STR + "({:0.2})".format(np.asarray(c).mean()),
             xlabel="Picks", xlim=0, ylabel="Events", ylim=0)
      ax.grid()
    axs[0].set()
    axs[1].set(ylabel=None, yticklabels=[])
    if len(args.weights) > 2:
      axs[2].set(title=None)
      axs[2].set_xlabel(args.weights[2] if len(args.weights) > 2
                                      else EMPTY_STR, fontsize=14)
      axs[2].xaxis.tick_top()
      axs[3].set(ylabel=None, yticklabels=[], title=None)
      axs[3].set_xlabel(args.weights[3] if len(args.weights) > 3
                                      else EMPTY_STR, fontsize=14)
      axs[3].xaxis.tick_top()
    fig.subplots_adjust(left=0.08, right=1.08, top=.95, bottom=0.05,
                        wspace=0.1, hspace=0.2)
    fig.colorbar(disp, ax=axs, orientation='vertical',
                 label="Mean probability", aspect=50, shrink=0.8)
    IMG_FILE = \
      Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
           UNDERSCORE_STR.join([
             CLSTR_PLOT_STR, model, THRESHOLDER_STR.format(
               p=args.pwave, s=args.swave)]) + PNG_EXT)
    plt.savefig(IMG_FILE)
    plt.close()

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
  global DATES
  MSG = f"Cumulative number of {phase} picks"
  if args.verbose: print(MSG)
  start, end = args.dates
  PRED = PRED[(PRED[PHASE_STR] == phase) &
              (PRED[TIMESTAMP_STR] >= start.datetime)]
  TRUE = TRUE[(TRUE[PHASE_STR] == phase)].reset_index(drop=True)
  if DATES is None:
    DATES = [start.datetime]
    while DATES[-1] <= end.datetime: DATES.append(DATES[-1] + ONE_DAY)

  y_true = [len(TRUE[TRUE[TIMESTAMP_STR] <= d].index) for d in DATES]
  # Plot a 2x2 grid for each model and weight
  for model, dtfrm in PRED.groupby(MODEL_STR):
    _, _axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = _axs.flatten()
    plt.suptitle(model, fontsize=16)
    axs[0].set(xticklabels=[], xlabel=None, ylabel=MSG)
    axs[1].set(xticklabels=[], xlabel=None, yticklabels=[], ylabel=None)
    axs[2].set(xlabel="Date", ylabel=MSG)
    axs[3].set(xlabel="Date", yticklabels=[], ylabel=None)
    y_max = 0
    for i, (weight, data) in enumerate(dtfrm.groupby(WEIGHT_STR)):
      axs[i].set_title(weight)
      for threshold in THRESHOLDS:
        y = [len(data[(data[PROBABILITY_STR] >= threshold) &
                      (data[TIMESTAMP_STR] <= d)].index) for d in DATES]
        axs[i].plot([np.datetime64(t) for t in DATES], y,
                    label=rf"$\geq$ {threshold}")
        y_max = max(y_max, max(y))
      axs[i].plot([np.datetime64(t) for t in DATES], y_true, label="True",
                  color="k")
      y_max = max(y_max, max(y_true))
    for ax in axs:
      ax.set(xlim=(DATES[0], DATES[-1]), ylim=(1, y_max), yscale="log")
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
           UNDERSCORE_STR.join([
             CMTV_PICKS_STR, model, phase, THRESHOLDER_STR.format(
               p=args.pwave, s=args.swave)]) + PNG_EXT)
    plt.tight_layout()
    plt.savefig(IMG_FILE)
    plt.close()
    if args.verbose: print(f"Saving {IMG_FILE}")
  # Plot a 2x2 grid for each model, network and station
  for (model, network, station), dtfrm in \
    PRED.groupby([MODEL_STR, NETWORK_STR, STATION_STR]):
    _, _axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = _axs.flatten()
    plt.suptitle(SPACE_STR.join([model, network, station]), fontsize=16)
    axs[0].set(xticklabels=[], xlabel=None, ylabel=MSG)
    axs[1].set(xticklabels=[], xlabel=None, yticklabels=[], ylabel=None)
    axs[2].set(xlabel="Date", ylabel=MSG)
    axs[3].set(xlabel="Date", yticklabels=[], ylabel=None)
    y_max = 0
    for i, (weight, data) in enumerate(dtfrm.groupby(WEIGHT_STR)):
      axs[i].set_title(weight)
      for threshold in THRESHOLDS:
        y = [len(data[(data[PROBABILITY_STR] >= threshold) &
                      (data[TIMESTAMP_STR] <= d)].index) for d in DATES]
        axs[i].plot([np.datetime64(t) for t in DATES], y,
                    label=rf"$\geq$ {threshold}")
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
           UNDERSCORE_STR.join([CMTV_PICKS_STR, model, network,
                                station, phase]) + PNG_EXT)
    plt.tight_layout()
    plt.savefig(IMG_FILE)
    plt.close()
    if args.verbose: print(f"Saving {IMG_FILE}")

def dist_balanced(T : pd.Series, P : pd.Series) -> float:
  return (dist_time(T, P) + 9. * dist_phase(T, P)) / 10.

def dist_phase(T : pd.Series, P : pd.Series) -> float:
  return int(P[PHASE_STR] == T[PHASE_STR])

def dist_time(T : pd.Series, P : pd.Series,
              offset : td = PICK_OFFSET) -> float:
  return 1. - (diff_time(T, P) / offset)

def diff_time(T : pd.Series, P : pd.Series) -> float:
  return td(seconds=abs(P[TIMESTAMP_STR] - T[TIMESTAMP_STR]))

def dist_default(T : pd.Series, P : pd.Series) -> float:
  return (99. * dist_balanced(T, P) + P[PROBABILITY_STR]) / 100.


class myBPGraph():
  def __init__(self, T : pd.DataFrame, P : pd.DataFrame, weight):
    self.W : function = weight
    self.M : int = len(P.index)
    self.N : int = 0
    self.P : dict[str, dict[int, ]] = P.to_dict()
    self.T : dict[str, dict[int, ]] = {key : {} for key in T.columns.tolist()}
    self.G : list[dict[int, dict[int, float]], dict[int, dict[int, float]]] = \
               [{}, {i : {} for i in range(self.M)}]
    for _, t in T.iterrows(): self.incNode(t, 0)

  def int2bool(self, i : int) -> bool: return i == 1
  def bool2int(self, b : bool) -> int: return 1 if b else 0
  def intflip(self, i : int) -> int: return 1 - i

  def addNode(self, u : int, bipartite : int, neighbours = dict())  -> None:
    if u not in self.G[bipartite]: self.G[bipartite][u] = neighbours

  def incNode(self, u : pd.Series, bipartite : int = 1) -> None:
    if bipartite == 1:
      for key, val in u.to_dict().items(): self.P[key][self.M] = val
      x = self.cnxNode(u, bipartite)
      self.addNode(self.M, bipartite, x)
      for node, weight in x.items(): self.addEdge(node, self.M, 0, weight)
      self.M += 1
    else:
      for key, val in u.to_dict().items(): self.T[key][self.N] = val
      x = self.cnxNode(u, bipartite)
      self.addNode(self.N, bipartite, x)
      for node, weight in x.items(): self.addEdge(node, self.N, 1, weight)
      self.N += 1
    return

  def cnxNode(self, u : pd.Series, bipartite : int = 1) -> dict[int, float]:
    v = pd.DataFrame(self.T if bipartite == 1 else self.P)
    v = v[(v[TIMESTAMP_STR] - u[TIMESTAMP_STR])\
          .apply(lambda x: td(seconds=abs(x))) <= PICK_OFFSET]
    if v.empty: return dict()
    return {i : self.W(u, v.loc[i]) for i in v.index}

  def addEdge(self, u : int, v : int, bipartite: int, weight : float) -> None:
    self.G[bipartite].setdefault(u, dict())
    self.G[bipartite][u][v] = weight

  def rmvEdge(self, u : int, v : int, bipartite : int = 0) -> None:
    if u in self.G[bipartite]: self.G[bipartite][u].remove(v)

  def getNeighbours(self, u : int, b : int = 0) -> set[int, int, float]:
    return set([(k, u, v) if b else (u, k, v)
                for k, v in self.G[b][u].items()])

  def adjMtx(self) -> np.ndarray:
    A = np.zeros((self.N, self.M))
    for t in self.G[0]:
      for p in self.G[0][t]: A[t][p] = self.G[0][t][p]
    return A

  def maxWmatch(self) -> list[tuple[int, int, float]]:
    MATCH = {}
    for i in range(self.N):
      x = self.getNeighbours(i, 0)
      if len(x) == 0: continue
      t, p, w = max(x, key=lambda x: x[-1])
      MATCH.setdefault(p, (t, w))
      if MATCH[p][-1] < w: MATCH[p] = (t, w)
    return [(t, p, w) for p, (t, w) in MATCH.items()]

  def makeMatch(self) -> None:
    LINKS : list[tuple[int, int, float]] = self.maxWmatch()
    if len(LINKS) == 0: return
    for i in range(2):
      for u in self.G[i].keys(): self.G[i][u] = dict()
    for t, p, w in LINKS:
      self.G[0][t] = {p : w}
      self.G[1][p] = {t : w}

  def confMtx(self) -> pd.DataFrame:
    TP, FN, FP = set(), [], set()
    CFN_MTX = pd.DataFrame(0, index=HEADER_CFMX, columns=HEADER_CFMX,
                           dtype=int)
    # We traverse the TRUE nodes of the graph to extract relevant information
    # to the True Positives and False Negatives lists
    nodesT = pd.DataFrame(self.T)
    nodesP = pd.DataFrame(self.P)
    for t, val in self.G[0].items():
      T = nodesT.iloc[t]
      x = list(val.keys())
      if len(x) == 0:
        CFN_MTX.loc[T[PHASE_STR], NONE_STR] += 1
        FN.append([T[ID_STR], T[TIMESTAMP_STR].__str__(), T[PROBABILITY_STR],
                   T[PHASE_STR], None, T[STATION_STR]])
        continue
      assert len(x) == 1
      P = nodesP.iloc[x[0]]
      CFN_MTX.loc[T[PHASE_STR], P[PHASE_STR]] += 1
      if T[PHASE_STR] == P[PHASE_STR]:
        TP.add(((T[ID_STR], P[ID_STR]),
                (str(T[TIMESTAMP_STR]), str(P[TIMESTAMP_STR])),
                (T[PROBABILITY_STR], P[PROBABILITY_STR]), T[PHASE_STR],
                P[NETWORK_STR], T[STATION_STR]))
    for p, val in self.G[1].items():
      if len(list(val.items())) == 0:
        P = nodesP.iloc[p]
        CFN_MTX.loc[NONE_STR, P[PHASE_STR]] += 1
        FP.add((P[ID_STR], P[TIMESTAMP_STR].__str__(), P[PROBABILITY_STR],
                P[PHASE_STR], P[NETWORK_STR], P[STATION_STR]))
    return CFN_MTX, TP, FN, FP

def conf_mtx(TRUE : pd.DataFrame, PRED : pd.DataFrame, model_name : str,
             dataset_name : str, args : argparse.Namespace)\
      -> list[pd.DataFrame, list, list, list]:
  """
  input  :
    - TRUE          (pd.DataFrame)
    - PRED          (pd.DataFrame)
    - model_name    (str)
    - dataset_name  (str)
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
  bpg = myBPGraph(TRUE, PRED, dist_default)
  bpg.makeMatch()
  # if args.interactive: plot_timeline(G, pos, N, model_name, dataset_name)
  CFN_MTX, TP, FN, FP = bpg.confMtx()
  TP = set([tuple([model_name, dataset_name, None, *x]) for x in TP])
  FN = [[model_name, dataset_name,
         THRESHOLDER_STR.format(p=args.pwave, s=args.swave), *x] for x in FN]
  FP = set([tuple([model_name, dataset_name, None, *x]) for x in FP])
  return CFN_MTX, TP, FN, FP

def stat_test(TRUE : pd.DataFrame, PRED : pd.DataFrame,
              args : argparse.Namespace, method = "Picker") -> pd.DataFrame:
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
  N_seconds = int((end + ONE_DAY - start) / (2 * PICK_OFFSET.total_seconds()))
  TP, FN, FP = set(), [], set()
  PRED = PRED[((PRED[PHASE_STR] == PWAVE) &
               (PRED[PROBABILITY_STR] >= args.pwave)) |
              ((PRED[PHASE_STR] == SWAVE) &
               (PRED[PROBABILITY_STR] >= args.swave))]
  Ws : int = len(args.weights)
  x : int = int(np.sqrt(Ws))
  y : int = Ws // x + int((Ws % x) != 0)
  for model, dataframe_m in PRED.groupby(MODEL_STR):
    fig, _axs = plt.subplots(x, y, figsize=(int(y * Ws) * 1.5,
                                            int(x * Ws - 1) * 1.5))
    axs = _axs.flatten()
    plt.rcParams.update({'font.size': 12})
    fig.suptitle(model)
    for ax, (weight, dataframe_w) in zip(axs, dataframe_m.groupby(WEIGHT_STR)):
      ax.set_title(weight)
      CFN_MTX = pd.DataFrame(0, index=HEADER_CFMX, columns=HEADER_CFMX,
                             dtype=int)
      for station, PRED_S in dataframe_w.groupby(STATION_STR):
        TRUE_S = TRUE[TRUE[STATION_STR] == station].reset_index(drop=True)
        cfn_mtx, tp, fn, fp = conf_mtx(TRUE_S, PRED_S.reset_index(drop=True),
                                       model, weight, args)
        CFN_MTX += cfn_mtx
        TP = TP.union(tp)
        FN.extend(fn)
        FP = FP.union(fp)
      CFN_MTX.loc[NONE_STR, NONE_STR] = N_seconds - CFN_MTX.sum().sum()
      disp = ConfMtxDisp(CFN_MTX.values, display_labels=CFN_MTX.columns)
      disp.plot(ax=ax, colorbar=False)
      for labels in disp.text_.ravel():
        labels.set(color="#E4007C", fontsize=12, fontweight="bold")
      disp.im_.set(clim=(1, N_seconds), cmap="Blues", norm="log")
    # TODO: Implement the properties of the plot as a function of the number of
    #       weights and coordinates
    axs[0].set()
    axs[1].set(ylabel=None, yticklabels=[])
    if len(args.weights) > 2:
      axs[2].set(title=None)
      axs[2].set_xlabel(args.weights[2] if len(args.weights) > 2
                                      else EMPTY_STR, fontsize=14)
      axs[2].xaxis.tick_top()
      axs[3].set(ylabel=None, yticklabels=[], title=None)
      axs[3].set_xlabel(args.weights[3] if len(args.weights) > 3
                                      else EMPTY_STR, fontsize=14)
      axs[3].xaxis.tick_top()
    fig.subplots_adjust(left=0.08, right=1.08, top=.95, bottom=0.05,
                        wspace=0.1, hspace=0.2)
    fig.colorbar(disp.im_, ax=axs, orientation='vertical',
                label="Number of Picks", aspect=50, shrink=0.8)
    disp.im_.set_clim(1, N_seconds)
    IMG_FILE = \
      Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
           UNDERSCORE_STR.join([
             method, CFN_MTX_STR, model, THRESHOLDER_STR.format(
               p=args.pwave, s=args.swave)]) + PNG_EXT)
    plt.savefig(IMG_FILE)
    plt.close()
  # True Positives
  TP = pd.DataFrame(TP, columns=HEADER_PRED).sort_values(SORT_HIERARCHY_PRED)
  TP_FILE = Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                  UNDERSCORE_STR.join([method, TP_STR]) + CSV_EXT)
  TP.to_csv(TP_FILE, index=False)
  # False Negatives
  FN = pd.DataFrame(FN, columns=HEADER_PRED).sort_values(SORT_HIERARCHY_PRED)
  FN_FILE = Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                 UNDERSCORE_STR.join([method, FN_STR]) + CSV_EXT)
  FN.to_csv(FN_FILE, index=False)
  # False Positives
  FP = pd.DataFrame(FP, columns=HEADER_PRED).sort_values(SORT_HIERARCHY_PRED)
  FP_FILE = Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                 UNDERSCORE_STR.join([method, FP_STR]) + CSV_EXT)
  FP.to_csv(FP_FILE, index=False)
  # False Negative Pie plot
  for (model, weight), dtfrm in FN.groupby([MODEL_STR, WEIGHT_STR]):
    fig, _ax = plt.subplots(1, 2, figsize=(10, 5))
    plt.suptitle(SPACE_STR.join([model, weight]), fontsize=16)
    for ax, phase in zip(_ax, [PWAVE, SWAVE]):
      ax.set_title(phase)
      dtfrm[dtfrm[PHASE_STR] == phase][PROBABILITY_STR].value_counts()\
        .plot(kind='pie', ax=ax, autopct='%1.1f%%')
    IMG_FILE = \
      Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
           UNDERSCORE_STR.join([
             method, FN_STR, model, weight, THRESHOLDER_STR.format(
               p=args.pwave, s=args.swave)]) + PNG_EXT)
    plt.savefig(IMG_FILE)
    plt.close()
  return TP
  # TODO: Redo the plots for the True Positives, False Negatives and False
  #       Positives
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
  DATA_PATH = args.directory.parent
  WAVEFORMS_DATA = ini.waveform_table(args)
  # TODO: Stations are not considered due to the low amount of data
  SOURCE, DETECT = prs.event_parser(filename, *args.dates, None)
  TRUE_S = pd.DataFrame(columns=HEADER_SRC)
  TRUE_D = pd.DataFrame(columns=HEADER_MANL)
  for date, dataframe_d in WAVEFORMS_DATA.groupby(DATE_STR):
    start = UTCDateTime.strptime(date, DATE_FMT)
    end = start + ONE_DAY
    if SOURCE is not None:
      source = SOURCE[SOURCE[TIMESTAMP_STR].between(start, end,
                                                    inclusive='left')]
      if not source.empty:
        TRUE_S = pd.concat([TRUE_S, source.sort_values(by=TIMESTAMP_STR)]) \
                   if not TRUE_S.empty else source
    station = dataframe_d[STATION_STR].unique().tolist()
    TRUE_D = pd.concat([TRUE_D, DETECT[
      (DETECT[TIMESTAMP_STR].between(start, end, inclusive='left')) &
      (DETECT[STATION_STR].isin(station))].sort_values(by=TIMESTAMP_STR)])
  if args.verbose:
    TRUE_S.to_csv(Path(DATA_PATH,
                       UNDERSCORE_STR.join([TRUE_STR, SOURCE_STR]) + CSV_EXT),
                       index=False)
    TRUE_D.to_csv(Path(DATA_PATH,
                       UNDERSCORE_STR.join([TRUE_STR, DETECT_STR]) + CSV_EXT),
                       index=False)
  return TRUE_S, TRUE_D

def time_displacement(DATA : pd.DataFrame, args : argparse.Namespace,
                      phase : str = PWAVE, method : str = "Picker") -> None:
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
  global THRESHOLDS
  if args.verbose: print("Plotting the Time Displacement")
  DATA[TIMESTAMP_STR] = \
    DATA[TIMESTAMP_STR].map(lambda x: UTCDateTime(x[0]) - UTCDateTime(x[1]))
  DATA[PROBABILITY_STR] = DATA[PROBABILITY_STR].map(lambda x: x[1])
  bins = np.linspace(-0.5, 0.5, 21, endpoint=True)
  groups = [MODEL_STR, WEIGHT_STR, PHASE_STR]
  m = 0
  for (model, weight, phase), dtfrm in DATA.groupby(groups):
    counts, _ = np.histogram(dtfrm[TIMESTAMP_STR], bins=bins)
    m = max(m, max(counts))
  m = (m + 9) // 10 * 10
  for model, phase in itertools.product(args.models, [PWAVE, SWAVE]):
    fig, _axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = _axs.flatten()
    dataframe_mp = DATA[(DATA[MODEL_STR] == model) &
                        (DATA[PHASE_STR] == phase)].reset_index(drop=True)
    for ax, (weight, dtfrm) in zip(axs, dataframe_mp.groupby(WEIGHT_STR)):
      ax.set_title(weight)
      counts, _ = np.histogram(dtfrm[TIMESTAMP_STR], bins=bins)
      mu = np.mean(dtfrm[TIMESTAMP_STR])
      std = np.std(dtfrm[TIMESTAMP_STR])
      ax.bar(bins[:-1], counts, label=rf"$\mu$={mu:.2f},$\sigma$={std:.2f}",
             alpha=0.5, width=0.05)
      for t_i, t_f in zip(THRESHOLDS[:-1], THRESHOLDS[1:]):
        data = dtfrm[dtfrm[PROBABILITY_STR].between(t_i, t_f, inclusive='left')
                    ][TIMESTAMP_STR]
        # TODO: Consider a KDE plot
        counts, _ = np.histogram(data, bins=bins)
        ax.step(bins[:-1], counts, where='mid', label=rf"[{t_i},{t_f})")
      data = dtfrm[
               dtfrm[PROBABILITY_STR] >= THRESHOLDS[-1]][TIMESTAMP_STR]
      counts, _ = np.histogram(data, bins=bins)
      ax.step(bins[:-1], counts, where='mid', label=rf"[{THRESHOLDS[-1]},1)")
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
           UNDERSCORE_STR.join([method, TIME_DSPLCMT_STR, model, phase,
             THRESHOLDER_STR.format(p=args.pwave, s=args.swave)]) + PNG_EXT)
    plt.tight_layout()
    plt.savefig(IMG_FILE)
    plt.close()

def main(args : argparse.Namespace):
  global DATA_PATH, DATES
  DATA_PATH = args.directory.parent
  # Picker
  ANALYSIS = "Picker"
  PICK : pd.DataFrame = ini.classified_loader(args)
  if args.verbose:
    PICK.to_csv(Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                     UNDERSCORE_STR.join([ANALYSIS, PRED_STR]) + CSV_EXT),
                     index=False)
  if not args.file: raise ValueError("No event file given")
  if len(args.file) > 1: raise NotImplementedError("Multiple event files")
  TRUE_S, TRUE_D = event_parser(args.file[0], args)
  TRUE_D = TRUE_D[TRUE_D[STATION_STR].isin(PICK[STATION_STR].unique())]
  plot_data(copy.deepcopy(TRUE_D), copy.deepcopy(PICK), args)
  plot_data(copy.deepcopy(TRUE_D), copy.deepcopy(PICK), args, phase=SWAVE)
  TP = stat_test(copy.deepcopy(TRUE_D), copy.deepcopy(PICK), args, ANALYSIS)
  if args.verbose:
    TP.to_csv(Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                   UNDERSCORE_STR.join([ANALYSIS, TP_STR]) + CSV_EXT),
                   index=False)
  time_displacement(copy.deepcopy(TP), args)
  # Associator
  ANALYSIS = "GaMMA"
  RECD : pd.DataFrame = ini.associated_loader(args)
  if args.verbose:
    RECD.to_csv(Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                     UNDERSCORE_STR.join([ANALYSIS, PRED_STR]) + CSV_EXT),
                     index=False)
  plot_cluster(PICK, RECD, args)
  del PICK
  start, end = args.dates
  TP = pd.DataFrame([], columns=HEADER_PRED)
  if DATES is None:
    DATES = [start]
    while DATES[-1] <= end: DATES.append(DATES[-1] + ONE_DAY)
  for s, e in zip(DATES[:-1], DATES[1:]):
    REC = RECD[RECD[TIMESTAMP_STR].between(s, e, inclusive='left')]
    if REC.empty: continue
    TP = pd.concat([TP, stat_test(copy.deepcopy(TRUE_D), copy.deepcopy(REC),
                                  args, ANALYSIS)])
  if args.verbose:
    TP.to_csv(Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                   UNDERSCORE_STR.join([ANALYSIS, TP_STR]) + CSV_EXT),
                   index=False)
  time_displacement(copy.deepcopy(TP), args, method=ANALYSIS)

if __name__ == "__main__": main(ini.parse_arguments())
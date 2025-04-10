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
import argparse
import itertools
import obspy as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from copy import deepcopy as dcpy
from datetime import timedelta as td
from obspy.geodetics import gps2dist_azimuth
from obspy.core.utcdatetime import UTCDateTime
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import ConfusionMatrixDisplay as ConfMtxDisp

from constants import *
import initializer as ini
import parser as prs

THRESHOLDS : list[float] = [round(t, 2) for t in np.linspace(0.2, 0.9, 8)]
THRESHOLDER_STR = PWAVE + "{p:0.2}" + SWAVE + "{s:0.2}"
DATES = None

def plot_cluster(PICK : pd.DataFrame, GMMA : pd.DataFrame,
                 args : argparse.Namespace) -> None:
  """
  input  :
    - PICK          (pd.DataFrame)
    - GMMA          (pd.DataFrame)
    - args          (argparse.Namespace)

  output :
    - None

  errors :
    - AttributeError

  notes  :
    The data is plotted for each model and weight. The plots are saved in the
    img directory.
  """
  if args.verbose: print("Plotting the Cluster")
  PICK = PICK[(PICK[MODEL_STR].isin(args.models)) &
              (PICK[WEIGHT_STR].isin(args.weights))].reset_index(drop=True)
  GMMA = GMMA[(GMMA[MODEL_STR].isin(args.models)) &
              (GMMA[WEIGHT_STR].isin(args.weights)) &
              (GMMA[THRESHOLD_STR] >= min(args.pwave,
                                          args.swave))].reset_index(drop=True)
  Ws : int = len(args.weights)
  x : int = int(np.sqrt(Ws))
  y : int = Ws // x + int((Ws % x) != 0)
  for model, dataframe_m in PICK.groupby(MODEL_STR):
    fig, _axs = plt.subplots(x, y, figsize=(int(y * Ws) * 1.5,
                                            int(x * Ws - 1.5) * 1.5))
    axs = _axs.flatten()
    fig.suptitle(model)
    plt.rcParams.update({'font.size': 12})
    max_p, max_r = 0, 0
    min_p, min_r = len(PICK.index), len(GMMA.index)
    for ax, (weight, df_w) in zip(axs, dataframe_m.groupby(WEIGHT_STR)):
      X = list()
      Y = list()
      Z = list()
      C = list()
      for station, P in df_w.groupby(STATION_STR):
        R = GMMA[(GMMA[MODEL_STR] == model) & (GMMA[WEIGHT_STR] == weight) &
                 (GMMA[STATION_STR] == station)].reset_index(drop=True)
        p, r = len(P.index), len(R.index)
        if p == 0 or r == 0:
          print(f"({model},{weight},{station}) was not plotted")
          continue
        X.append(len(P.index))
        Y.append(len(R.index))
        Z.append(station)
        C.append(R[THRESHOLD_STR].to_numpy().mean())
      if len(X) == 0:
        print(f"({model},{weight}) was not plotted")
        continue
      disp = ax.scatter(X, Y, c=C, cmap="turbo", clim=(0.1, 1))
      max_p, min_p = max(max_p, max(X)), min(min_p, min(X))
      max_r, min_r = max(max_r, max(Y)), min(min_r, min(Y))
      ax.set(title="{} ({:0.2})".format(weight, np.asarray(C).mean()),
             xlabel="Picks", xscale="log", ylabel="Events", yscale="log")
      ax.grid()
    if max_p == 0 or max_r == 0:
      print(f"({model}) was not plotted")
      continue
    max_p = np.power(10, int(np.log10(max_p)) + 1)
    min_p = np.power(10, int(np.log10(min_p)))
    max_r = np.power(10, int(np.log10(max_r)) + 1)
    min_r = np.power(10, int(np.log10(min_r)))
    for ax in axs: ax.set(xlim=(min_p, max_p), ylim=(min_r, max_r))
    axs[0].set()
    axs[1].set(ylabel=None, yticklabels=[])
    if len(args.weights) > 2:
      axs[2].set_xlabel(axs[2].get_title(), fontsize=14)
      axs[2].set(title=None)
      axs[2].xaxis.tick_top()
      axs[3].set_xlabel(axs[3].get_title(), fontsize=14)
      axs[3].set(ylabel=None, yticklabels=[], title=None)
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
  if args.verbose:
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
      print(f"Saving {IMG_FILE}")

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
MATCH_CNFG[CLSSFD_STR].update({DISTANCE_STR : dist_default})
MATCH_CNFG[DETECT_STR].update({DISTANCE_STR : dist_default})

def dist_space(T : pd.Series, P : pd.Series, offset : float = 1.) -> float:
  return 1. - (float(format(diff_space(T, P) / 1000., ".4f")) / offset)

def diff_space(T : pd.Series, P : pd.Series) -> float:
  return gps2dist_azimuth(T[LATITUDE_STR], T[LONGITUDE_STR],
                          P[LATITUDE_STR], P[LONGITUDE_STR])[0]

def dist_event(T : pd.Series, P : pd.Series,
               time_offset_sec : td = ASSOCIATE_TIME_OFFSET,
               space_offset_km : float = ASSOCIATE_DIST_OFFSET) -> float:
  return (dist_time(T, P, time_offset_sec) +
          dist_space(T, P, space_offset_km)) / 2.
MATCH_CNFG[SOURCE_STR].update({DISTANCE_STR : dist_event})

class myBPGraph():
  def __init__(self, T : pd.DataFrame, P : pd.DataFrame,
               config : dict[str, any] = MATCH_CNFG[CLSSFD_STR]):
    self.W : function = config[DISTANCE_STR]
    self.M : int = len(P.index)
    self.N : int = 0
    self.P : dict[str, dict[int, ]] = P.to_dict()
    self.T : dict[str, dict[int, ]] = {key : {} for key in T.columns.tolist()}
    self.G : list[dict[int, dict[int, float]], dict[int, dict[int, float]]] = \
               [{}, {i : {} for i in range(self.M)}]
    self.C : dict[str, any] = config
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
          .apply(lambda x: td(seconds=abs(x))) <= self.C[TIME_DSPLCMT_STR]]
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
    match_vals = self.C[CATEGORY_STR]
    CFN_MTX = pd.DataFrame(0, index=match_vals, columns=match_vals, dtype=int)
    # We traverse the TRUE nodes of the graph to extract relevant information
    # to the True Positives and False Negatives lists
    nodesT = pd.DataFrame(self.T).reset_index(drop=True)
    nodesP = pd.DataFrame(self.P).reset_index(drop=True)
    for t, val in self.G[0].items():
      T = nodesT.iloc[t]
      x = list(val.keys())
      if len(x) == 0:
        if self.C[METHOD_STR] in [CLSSFD_STR, DETECT_STR]:
          CFN_MTX.loc[T[PHASE_STR], NONE_STR] += 1
          FN.append([T[ID_STR], T[TIMESTAMP_STR].__str__(), T[PROBABILITY_STR],
                     T[PHASE_STR], None, T[STATION_STR]])
        else:
          CFN_MTX.loc[EVENT_STR, NONE_STR] += 1
          FN.append([T[ID_STR], str(T[TIMESTAMP_STR]), T[LATITUDE_STR],
                     T[LONGITUDE_STR], T[LOCAL_DEPTH_STR], T[MAGNITUDE_STR],
                     None, None, None, None, None, None, None, None])
        continue
      assert len(x) == 1
      P = nodesP.iloc[x[0]]
      if self.C[METHOD_STR] in [CLSSFD_STR, DETECT_STR]:
        CFN_MTX.loc[T[PHASE_STR], P[PHASE_STR]] += 1
        if T[PHASE_STR] == P[PHASE_STR]:
          TP.add((P[THRESHOLD_STR], (T[ID_STR], str(P[ID_STR])),
                  (str(T[TIMESTAMP_STR]), str(P[TIMESTAMP_STR])),
                  (T[PROBABILITY_STR], P[PROBABILITY_STR]), T[PHASE_STR],
                  P[NETWORK_STR], T[STATION_STR]))
      else:
        CFN_MTX.loc[EVENT_STR, EVENT_STR] += 1
        TP.add((P[THRESHOLD_STR], (T[ID_STR], str(P[ID_STR])),
                (str(T[TIMESTAMP_STR]), str(P[TIMESTAMP_STR])),
                (T[LATITUDE_STR], P[LATITUDE_STR]),
                (T[LONGITUDE_STR], P[LONGITUDE_STR]),
                (T[LOCAL_DEPTH_STR], P[LOCAL_DEPTH_STR]),
                (T[MAGNITUDE_STR], P[MAGNITUDE_STR]), None, None, None, None,
                None, None, None, (SOURCE_STR, P[NOTES_STR])))
    result = []
    for p, val in self.G[1].items():
      if len(list(val.items())) == 0:
        P = nodesP.iloc[p]
        if self.C[METHOD_STR] in [CLSSFD_STR, DETECT_STR]:
          CFN_MTX.loc[NONE_STR, P[PHASE_STR]] += 1
          result.append((P[ID_STR], str(P[TIMESTAMP_STR]), P[PROBABILITY_STR],
                         P[PHASE_STR], P[NETWORK_STR], P[STATION_STR]))
        else:
          CFN_MTX.loc[NONE_STR, EVENT_STR] += 1
          result.append((P[ID_STR], str(P[TIMESTAMP_STR]), P[LATITUDE_STR],
                         P[LONGITUDE_STR], P[LOCAL_DEPTH_STR],
                         P[MAGNITUDE_STR], None, None, None, None, None, None,
                         None, P[NOTES_STR]))
    FP.update(set(result))
    return CFN_MTX, TP, FN, FP

def conf_mtx(TRUE : pd.DataFrame, PRED : pd.DataFrame, model_name : str,
             dataset_name : str, args : argparse.Namespace,
             method : str = CLSSFD_STR) \
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
  bpg = myBPGraph(TRUE.reset_index(drop=True),
                  PRED.reset_index(drop=True), config=MATCH_CNFG[method])
  bpg.makeMatch()
  # if args.interactive: plot_timeline(G, pos, N, model_name, dataset_name)
  CFN_MTX, TP, FN, FP = bpg.confMtx()
  TP = set([tuple([model_name, dataset_name, *x]) for x in TP])
  FN = [[model_name, dataset_name, None, *x] for x in FN]
  FP = set([tuple([model_name, dataset_name, None, *x]) for x in FP])
  return CFN_MTX, TP, FN, FP

def stat_test(TRUE : pd.DataFrame, PRED : pd.DataFrame,
              args : argparse.Namespace,
              method : str = CLSSFD_STR) -> pd.DataFrame:
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
  N_seconds = int((end + ONE_DAY - start) / \
                  (2 * MATCH_CNFG[method][TIME_DSPLCMT_STR].total_seconds()))
  global DATES
  if DATES is None:
    DATES = [start.datetime]
    while DATES[-1] <= end.datetime: DATES.append(DATES[-1] + ONE_DAY)
  TP, FN, FP = set(), [], set()
  if method in [CLSSFD_STR, DETECT_STR]:
    PRED = PRED[((PRED[PHASE_STR] == PWAVE) &
                 (PRED[PROBABILITY_STR] >= args.pwave)) |
                ((PRED[PHASE_STR] == SWAVE) &
                 (PRED[PROBABILITY_STR] >= args.swave))].reset_index(drop=True)
  Ws : int = len(args.weights)
  x : int = int(np.sqrt(Ws))
  y : int = Ws // x + int((Ws % x) != 0)
  for model, dataframe_m in PRED.groupby(MODEL_STR):
    fig, _axs = plt.subplots(x, y, figsize=(int(y * Ws) * 1.5,
                                            int(x * Ws - 1) * 1.5))
    axs = _axs.flatten()
    plt.rcParams.update({'font.size': 12})
    fig.suptitle(model)
    for ax, (weight, PRED_W) in zip(axs, dataframe_m.groupby(WEIGHT_STR)):
      ax.set_title(weight)
      CFN_MTX = pd.DataFrame(0, index=MATCH_CNFG[method][CATEGORY_STR],
                             columns=MATCH_CNFG[method][CATEGORY_STR],
                             dtype=int)
      tp = set()
      fn = []
      fp = set()
      print(f"Processing {model} {weight}...")
      if method in [CLSSFD_STR, DETECT_STR]:
        for station, PRED_S in PRED_W.groupby(STATION_STR):
          cfn_mtx, tp, fn, fp = conf_mtx(
            TRUE[(TRUE[STATION_STR] == station)].reset_index(drop=True),
            PRED_S.reset_index(drop=True), model, weight, args, method)
          CFN_MTX += cfn_mtx
          TP = TP.union(tp)
          FN.extend(fn)
          FP = FP.union(fp)
      else:
        CFN_MTX, tp, fn, fp = conf_mtx(
          TRUE.reset_index(drop=True), PRED_W.reset_index(drop=True), model,
          weight, args, method)
        TP = TP.union(tp)
        FN.extend(fn)
        FP = FP.union(fp)
      CFN_MTX.loc[NONE_STR, NONE_STR] = N_seconds - CFN_MTX.sum().sum()
      disp = ConfMtxDisp(CFN_MTX.values, display_labels=CFN_MTX.columns)
      disp.plot(ax=ax, colorbar=False)
      disp.im_.set(clim=(1, N_seconds), cmap="Blues", norm="log")
      for labels in disp.text_.ravel():
        labels.set(color="#E4007C", fontsize=12, fontweight="bold")
    # TODO: Implement the properties of the plot as a function of the number of
    #       weights and coordinates
    axs[0].set()
    axs[1].set(ylabel=None, yticklabels=[])
    if len(args.weights) > 2:
      axs[2].set_xlabel(axs[2].get_title(), fontsize=14)
      axs[2].set(title=None)
      axs[2].xaxis.tick_top()
      axs[3].set_xlabel(axs[3].get_title(), fontsize=14)
      axs[3].set(ylabel=None, yticklabels=[], title=None)
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

  HEADER = HEADER_MODL + MATCH_CNFG[method][HEADER_STR]
  print(HEADER)
  # True Positives
  TP = pd.DataFrame(TP, columns=HEADER).sort_values(SORT_HIERARCHY_PRED)
  print(TP)
  # TODO: Implement the threshold for the True Positives
  for (m, w), df in TP.groupby([MODEL_STR, WEIGHT_STR]):
    print(m, w)
    if df.empty: continue
    tp = len(df.index)
    ids = {id[0] : id[1] for id in df[ID_STR].to_list()}
    if method in [CLSSFD_STR, DETECT_STR]:
      tp_s = len(df[df[PHASE_STR] == SWAVE].index)
      print(f"cTP ({PWAVE}): {tp - tp_s}")
      print(f"cTP ({SWAVE}): {tp_s}")
      for k, v in ids.items():
        TP.loc[(TP[MODEL_STR] == m) & (TP[WEIGHT_STR] == w) &
               (TP[ID_STR] == (k, v)), [THRESHOLD_STR, ID_STR]] = [
          float("{:0.1}".format(min([x[1] for x in df.loc[
            df[ID_STR] == (k, v), PROBABILITY_STR].tolist()]))),
          tuple([k, v])]
    print(f"TP: {tp}")
  if args.verbose: TP.to_csv(Path(
    DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
    UNDERSCORE_STR.join([method, TP_STR]) + CSV_EXT), index=False)

  # False Negatives
  FN = pd.DataFrame(FN, columns=HEADER).sort_values(SORT_HIERARCHY_PRED)
  for (m, w), df in FN.groupby([MODEL_STR, WEIGHT_STR]):
    print(m, w)
    fn = len(df.index)
    if method in [CLSSFD_STR, DETECT_STR]:
      fn_s = len(df[df[PHASE_STR] == SWAVE].index)
      print(f"FN ({PWAVE}): {fn - fn_s}")
      print(f"FN ({SWAVE}): {fn_s}")
    print(f"FN: {fn}")
  FN_FILE = Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                 UNDERSCORE_STR.join([method, FN_STR]) + CSV_EXT)
  if args.verbose: FN.to_csv(FN_FILE, index=False)

  # False Positives
  FP = pd.DataFrame(FP, columns=HEADER).sort_values(SORT_HIERARCHY_PRED)
  for (m, w), df in FP.groupby([MODEL_STR, WEIGHT_STR]):
    print(m, w)
    fp = len(df.index)
    if method in [CLSSFD_STR, DETECT_STR]:
      fp_s = len(df[df[PHASE_STR] == SWAVE].index)
      print(f"FP ({PWAVE}): {fp - fp_s}")
      print(f"FP ({SWAVE}): {fp_s}")
    print(f"FP: {len(df.index)}")
  FP_FILE = Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                 UNDERSCORE_STR.join([method, FP_STR]) + CSV_EXT)
  if args.verbose: FP.to_csv(FP_FILE, index=False)

  # False Negative Pie plot
  if method in [CLSSFD_STR, DETECT_STR]:
    for (model, weight), df in FN.groupby([MODEL_STR, WEIGHT_STR]):
      fig, _ax = plt.subplots(1, 2, figsize=(10, 5))
      plt.suptitle(SPACE_STR.join([model, weight]), fontsize=16)
      for ax, phase in zip(_ax, [PWAVE, SWAVE]):
        ax.set_title(phase)
        df[df[PHASE_STR] == phase][PROBABILITY_STR].value_counts()\
          .plot(kind='pie', ax=ax, autopct='%1.1f%%')
      IMG_FILE = \
        Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
            UNDERSCORE_STR.join([
              method, FN_STR, model, weight, THRESHOLDER_STR.format(
                p=args.pwave, s=args.swave)]) + PNG_EXT)
      plt.savefig(IMG_FILE)
      plt.close()

  # TP FN
  if method not in [CLSSFD_STR, DETECT_STR]: return TP
  groups = [MODEL_STR, WEIGHT_STR, PHASE_STR]
  max_threshold = max(TP.groupby(groups)[THRESHOLD_STR].value_counts().max(),
                      FN.groupby(groups)[THRESHOLD_STR].value_counts().max())
  max_threshold = (max_threshold + 9) // 10 * 10
  Ws : int = len(args.weights)
  x : int = int(np.sqrt(Ws))
  y : int = Ws // x + int((Ws % x) != 0)
  for model in args.models:
    TP_M = TP[TP[MODEL_STR] == model]
    FN_M = FN[FN[MODEL_STR] == model]
    fig, _axs = plt.subplots(x, y, figsize=(int(y * Ws) * 1.5,
                                            int(x * Ws - 1) * 1.5))
    axs = _axs.flatten()
    fig.suptitle(model)
    plt.rcParams.update({'font.size': 12})
    for ax1, w in zip(axs, args.weights):
      TP_W = TP_M[TP_M[WEIGHT_STR] == w]
      FN_W = FN_M[FN_M[WEIGHT_STR] == w]
      ax1.set_title(w, fontsize=16)
      ax2 = ax1.twinx()
      ax1.set(ylabel="Number of Picks", ylim=(0, max_threshold))
      RECALL = {
        PWAVE : (0., 0.),
        SWAVE : (0., 0.),
        RECALL_STR : 0.
      }
      for p in [PWAVE, SWAVE]:
        tp = TP_W.loc[TP_W[PHASE_STR] == p, THRESHOLD_STR].value_counts() \
                 .sort_index()
        fn = FN_W.loc[FN_W[PHASE_STR] == p, THRESHOLD_STR].value_counts() \
                 .sort_index()
        RECALL[p] = (tp, fn)
      RECALL[RECALL_STR] = (RECALL[PWAVE][0] + RECALL[SWAVE][0]) / \
                           (RECALL[PWAVE][0] + RECALL[PWAVE][1] + \
                            RECALL[SWAVE][0] + RECALL[SWAVE][1])
      RECALL[RECALL_STR].plot(ax=ax2, label=f"{PWAVE} + {SWAVE}", color="k")
      for p in [PWAVE, SWAVE]:
        (RECALL[p][0] / (RECALL[p][0] + RECALL[p][1])).plot(
          ax=ax2, label=p, color="r" if p == PWAVE else "b")
      TPFN = pd.DataFrame({
        SPACE_STR.join([PWAVE, TP_STR]): RECALL[PWAVE][0],
        SPACE_STR.join([SWAVE, TP_STR]): RECALL[SWAVE][0],
        SPACE_STR.join([PWAVE, FN_STR]): RECALL[PWAVE][1],
        SPACE_STR.join([SWAVE, FN_STR]): RECALL[SWAVE][1]
      })
      TPFN.plot(kind='bar', ax=ax1, width=0.7, legend=True)
      ax1.set(ylabel="Number of Picks", ylim=(0, max_threshold))
      ax2.set(ylim=(0, 1))
      yticks, yticklabels = ax2.get_yticks(), ax2.get_yticklabels()
      ax2.set(yticks=[], yticklabels=[])
      ax1.grid()
    axs[0].set()
    axs[1].set(ylabel=None, yticklabels=[])
    if len(args.weights) > 2:
      axs[2].set_xlabel(axs[2].get_title(), fontsize=14)
      axs[2].set(title=None)
      axs[2].xaxis.tick_top()
      axs[3].set_xlabel(axs[3].get_title(), fontsize=14)
      axs[3].set(ylabel=None, yticklabels=[], title=None)
      axs[3].xaxis.tick_top()
    fig.subplots_adjust(left=0.08, right=1.08, top=.95, bottom=0.05,
                        wspace=0.1, hspace=0.2)
    IMG_FILE = Path(
      IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
      UNDERSCORE_STR.join([method, "TPFN", model, THRESHOLDER_STR.format(
        p=args.pwave, s=args.swave)]) + PNG_EXT)
    plt.tight_layout()
    plt.savefig(IMG_FILE)
    plt.close()
  return TP
  # TODO: Redo the plots for the True Positives, False Negatives and False
  #       Positives
  # Plot the True Positives, False Negatives histogram and the Recall as a
  # function of the threshold for each model and weight
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

def event_parser(filename : Path, args : argparse.Namespace,
                 stations : dict[str, set[str]] = None) -> pd.DataFrame:
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
  # TODO: Stations are not considered due to the low amount of data
  SOURCE, DETECT = prs.event_parser(filename, *args.dates, stations)
  if args.verbose:
    SOURCE.to_csv(Path(DATA_PATH,
                       UNDERSCORE_STR.join([TRUE_STR, SOURCE_STR]) + CSV_EXT),
                       index=False)
    DETECT.to_csv(Path(DATA_PATH,
                       UNDERSCORE_STR.join([TRUE_STR, DETECT_STR]) + CSV_EXT),
                       index=False)
  SOURCE = SOURCE[SOURCE[NOTES_STR].isnull() &
                  SOURCE[LATITUDE_STR].notna()].reset_index(drop=True)
  DETECT = DETECT[DETECT[ID_STR].isin(SOURCE[ID_STR])].reset_index(drop=True)
  print("Picks Detections")
  print(f"True (P): {len(DETECT[DETECT[PHASE_STR] == PWAVE].index)}",
        f"True (S): {len(DETECT[DETECT[PHASE_STR] == SWAVE].index)}",
        f"True: {len(DETECT.index)}")
  print("Events Sources")
  print(f"True: {len(SOURCE.index)}")
  return SOURCE, DETECT

def time_displacement(DATA : pd.DataFrame, args : argparse.Namespace,
                      method : str = CLSSFD_STR) -> None:
  """
  input  :
    - DATA          (pd.DataFrame)
    - args          (argparse.Namespace)
    - method        (str)

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
  sec = MATCH_CNFG[method][TIME_DSPLCMT_STR].total_seconds()
  bins = np.linspace(-sec, sec, 41, endpoint=True)
  width = bins[1] - bins[0]
  groups = [MODEL_STR, WEIGHT_STR]
  if method in [CLSSFD_STR, DETECT_STR]: groups.append(PHASE_STR)
  m = 0
  for _, dtfrm in DATA.groupby(groups):
    counts, _ = np.histogram(dtfrm[TIMESTAMP_STR], bins=bins)
    m = max(m, max(counts))
  m = (m + 9) // 10 * 10
  Ws : int = len(args.weights)
  x : int = int(np.sqrt(Ws))
  y : int = Ws // x + int((Ws % x) != 0)
  if method in [CLSSFD_STR, DETECT_STR]:
    DATA[PROBABILITY_STR] = DATA[PROBABILITY_STR].map(lambda x: x[1])
    for phase, df_p in DATA.groupby(PHASE_STR):
      for model, df_m in df_p.groupby(MODEL_STR):
        fig, _axs = plt.subplots(x, y, figsize=(int(y * Ws) * 1.5,
                                                int(x * Ws - 1) * 1.5))
        axs = _axs.flatten()
        plt.rcParams.update({'font.size': 12})
        fig.suptitle(model)
        for ax, (weight, df_w) in zip(axs, df_m.groupby(WEIGHT_STR)):
          counts, _ = np.histogram(df_w[TIMESTAMP_STR], bins=bins)
          mu = np.mean(df_w[TIMESTAMP_STR])
          std = np.std(df_w[TIMESTAMP_STR])
          ax.bar(bins[:-1], counts, label=rf"$\mu$={mu:.2f},$\sigma$={std:.2f}",
                 alpha=0.5, width=width)
          for t_i, t_f in zip(THRESHOLDS[:-1], THRESHOLDS[1:]):
            # TODO: Consider a KDE plot
            counts, _ = np.histogram(df_w[df_w[PROBABILITY_STR].between(
              t_i, t_f, inclusive='left')][TIMESTAMP_STR], bins=bins)
            ax.step(bins[:-1], counts, where='mid', label=rf"[{t_i},{t_f})")
          counts, _ = np.histogram(
            df_w[df_w[PROBABILITY_STR] >= THRESHOLDS[-1]][TIMESTAMP_STR],
            bins=bins)
          ax.step(bins[:-1], counts, where='mid',
                  label=rf"[{THRESHOLDS[-1]},1)")
          ax.set(title=weight, xlabel="Time Displacement (s)",
                 xlim=(-sec, sec), ylabel=f"Number of {phase} picks",
                 ylim=(0, m))
          ax.grid()
          ax.legend()
        axs[0].set()
        axs[1].set(ylabel=None, yticklabels=[])
        if len(args.weights) > 2:
          axs[2].set_xlabel(axs[2].get_title(), fontsize=14)
          axs[2].set(title=None)
          axs[2].xaxis.tick_top()
          axs[3].set_xlabel(axs[3].get_title(), fontsize=14)
          axs[3].set(ylabel=None, yticklabels=[], title=None)
          axs[3].xaxis.tick_top()
        IMG_FILE = \
          Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
              UNDERSCORE_STR.join([method, TIME_DSPLCMT_STR, model, phase,
                THRESHOLDER_STR.format(p=args.pwave, s=args.swave)]) + PNG_EXT)
        plt.tight_layout()
        plt.savefig(IMG_FILE)
        plt.close()
  else:
    for model, df_m in DATA.groupby(MODEL_STR):
      fig, _axs = plt.subplots(x, y, figsize=(int(y * Ws) * 1.5,
                                              int(x * Ws - 1) * 1.5))
      axs = _axs.flatten()
      plt.rcParams.update({'font.size': 12})
      fig.suptitle(model)
      for ax, (weight, df_w) in zip(axs, df_m.groupby(WEIGHT_STR)):
        print(df_w)
        counts, _ = np.histogram(df_w[TIMESTAMP_STR], bins=bins)
        mu = np.mean(df_w[TIMESTAMP_STR])
        std = np.std(df_w[TIMESTAMP_STR])
        ax.bar(bins[:-1], counts, label=rf"$\mu$={mu:.2f},$\sigma$={std:.2f}",
               alpha=0.5, width=width)
        for t_i in THRESHOLDS:
          # TODO: Consider a KDE plot
          counts, _ = np.histogram(
            df_w[df_w[THRESHOLD_STR] == t_i][TIMESTAMP_STR], bins=bins)
          ax.step(bins[:-1], counts, where='mid', label=rf"$\geq${t_i}")
        ax.set(title=weight, xlabel="Time Displacement (s)", xlim=(-sec, sec),
               ylabel=f"Number of events", ylim=(0, m))
        ax.grid()
        ax.legend()
      axs[0].set()
      axs[1].set(ylabel=None, yticklabels=[])
      if len(args.weights) > 2:
        axs[2].set_xlabel(axs[2].get_title(), fontsize=14)
        axs[2].set(title=None)
        axs[2].xaxis.tick_top()
        axs[3].set_xlabel(axs[3].get_title(), fontsize=14)
        axs[3].set(ylabel=None, yticklabels=[], title=None)
        axs[3].xaxis.tick_top()
      IMG_FILE = \
        Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
            UNDERSCORE_STR.join([method, TIME_DSPLCMT_STR, model,
              THRESHOLDER_STR.format(p=args.pwave, s=args.swave)]) + PNG_EXT)
      plt.tight_layout()
      plt.savefig(IMG_FILE)
      plt.close()

def _Analysis(args : argparse.Namespace,
              method : str = CLSSFD_STR) -> pd.DataFrame:
  DF : pd.DataFrame = pd.DataFrame(columns=HEADER_MODL +
                                           MATCH_CNFG[method][HEADER_STR])
  FILEPATH = Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + method +\
                  CSV_EXT)
  if (not args.force and FILEPATH.exists() and
      ini.read_args(args, False) == ini.dump_args(args, True)):
    print(f"Loading {FILEPATH}...")
    DF = ini.data_loader(FILEPATH)
  else:
    DF = ini.classified_loader(args) if method == CLSSFD_STR else \
         ini.associated_loader(args)
  DF[TIMESTAMP_STR] = DF[TIMESTAMP_STR].apply(lambda x: UTCDateTime(x))
  start, end = args.dates
  DF = DF[DF[TIMESTAMP_STR].between(start, end + ONE_DAY, inclusive='left')]
  if args.verbose: DF.to_csv(FILEPATH, index=False)
  else: return DF
  if method != CLSSFD_STR: return DF
  global DATES
  if DATES is None:
    DATES = [start]
    while DATES[-1] <= end: DATES.append(DATES[-1] + ONE_DAY)
  dates = [np.datetime64(d) for d in DATES]
  Ws : int = len(args.weights)
  x : int = int(np.sqrt(Ws))
  y : int = Ws // x + int((Ws % x) != 0)
  for (model, phase), dataframe_m in DF.groupby([MODEL_STR, PHASE_STR]):
    fig, _axs = plt.subplots(x, y, figsize=(int(y * Ws) * 1.5,
                                            int(x * Ws - 1) * 1.5))
    axs = _axs.flatten()
    plt.rcParams.update({'font.size': 12})
    fig.suptitle(model)
    max_h = 0
    for ax, (weight, df_w) in zip(axs, dataframe_m.groupby(WEIGHT_STR)):
      ax.set_title(weight)
      for threshold in THRESHOLDS:
        h = [len(df_w[(df_w[PROBABILITY_STR] >= threshold) &
                      (df_w[TIMESTAMP_STR] <= d)].index) for d in DATES]
        ax.plot(dates, h, label=rf"$\geq$ {threshold}")
        max_h = max(max_h, max(h))
      ax.set(xlabel=None, yscale="log", ylabel="Number of Picks")
      ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
      for label in ax.get_xticklabels():
        label.set(rotation=30, horizontalalignment='right')
      ax.grid()
      ax.legend()
    for ax in axs: ax.set(ylim=(1, max_h))
    axs[0].set(xticklabels=[])
    axs[1].set(xticklabels=[], ylabel=None, yticklabels=[])
    if len(args.weights) > 2:
      axs[2].set()
      axs[3].set(ylabel=None, yticklabels=[])
    plt.tight_layout()
    IMG_FILE = \
      Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
           UNDERSCORE_STR.join([CMTV_PICKS_STR, method, model, phase]) + \
             PNG_EXT)
    plt.savefig(IMG_FILE)
    plt.close()
  return DF

def _Stations(args : argparse.Namespace) -> dict[str, set[str]]:
  WAVEFORMS = ini.waveform_table(args)
  WAVEFORMS[DATE_STR] = WAVEFORMS[DATE_STR].apply(
    lambda x: UTCDateTime.strptime(x, DATE_FMT))
  start, end = args.dates
  global DATES
  if DATES is None:
    DATES = [start.datetime]
    while DATES[-1] <= end.datetime: DATES.append(DATES[-1] + ONE_DAY)
  """
  STATIONS = {
    s.strftime(DATE_FMT) : set(WAVEFORMS.loc[WAVEFORMS[DATE_STR].between(
                             s, e, inclusive='left'), STATION_STR])
    for s, e in zip(DATES[:-1], DATES[1:])
  }"""
  min_s, max_s = np.inf, 0
  stations = set()
  STATIONS = dict()
  for s, e in zip(DATES[:-1], DATES[1:]):
    S = WAVEFORMS.loc[WAVEFORMS[DATE_STR].between(s, e, inclusive='left'),
                      STATION_STR]
    if S.empty: continue
    S = S.unique()
    min_s = min(min_s, len(S))
    max_s = max(max_s, len(S))
    STATIONS[s.strftime(DATE_FMT)] = set(S)
    stations.update(S)
    if args.verbose: print(f"Stations {s.strftime(DATE_FMT)}: {len(S)}")
  print(f"Min Stations: {min_s}, Max Stations: {max_s}")
  print(f"Total Stations: {len(stations)}")
  return STATIONS

def main(args : argparse.Namespace):
  global DATA_PATH
  DATA_PATH = args.directory.parent
  stream = op.read([f for f in args.directory.iterdir()][0])[0]
  stream = stream.resample(SAMPLING_RATE)
  # TODO: Consider using the Stream.detrend() method
  start = UTCDateTime(stream.stats.starttime.date)
  if stream.stats.starttime.hour == 23: start += ONE_DAY
  stream.trim(starttime=start, endtime=start + ONE_DAY, pad=True, fill_value=0,
              nearest_sample=False)
  fig, ax = plt.subplots(figsize=(10, 5))
  stream.plot(fig=fig, type='dayplot', show=True)
  STATIONS = _Stations(args)
  if not args.file: raise ValueError("No event file given")
  if len(args.file) > 1: raise NotImplementedError("Multiple event files")
  TRUE_S, TRUE_D = event_parser(args.file[0], args, STATIONS)
  if args.option in [CLSSFD_STR, ALL_WILDCHAR_STR]:
    print(CLSSFD_STR)
    PRED = _Analysis(args, CLSSFD_STR)
    time_displacement(stat_test(dcpy(TRUE_D), dcpy(PRED), args, CLSSFD_STR),
                      args, CLSSFD_STR)
    pass
  if args.option in [CLSSFD_STR, DETECT_STR, ALL_WILDCHAR_STR]:
    print(DETECT_STR)
    PRED_D = _Analysis(args, DETECT_STR)
    time_displacement(stat_test(dcpy(TRUE_D), dcpy(PRED_D), args, DETECT_STR),
                      args, DETECT_STR)
    pass
  if args.option == ALL_WILDCHAR_STR:
    plot_cluster(PRED, PRED_D, args)
    #plot_cluster(TRUE_D, PRED_D, args)
    del PRED, PRED_D
  if args.option in [SOURCE_STR, ALL_WILDCHAR_STR]:
    print(SOURCE_STR)
    PRED_S = ini.data_loader(Path(
      DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + SOURCE_STR + 
      CSV_EXT))
    PRED_S[TIMESTAMP_STR] = PRED_S[TIMESTAMP_STR].apply(lambda x:
                                                        UTCDateTime(x))
    time_displacement(stat_test(TRUE_S, PRED_S, args, SOURCE_STR), args,
                      SOURCE_STR)

if __name__ == "__main__": main(ini.parse_arguments())
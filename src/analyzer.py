import argparse

from sklearn.metrics import ConfusionMatrixDisplay as ConfMtxDisp
from obspy.core.utcdatetime import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from datetime import timedelta as td
from copy import deepcopy as dcpy
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
# Set the project folder
PRJ_PATH = Path(os.path.dirname(__file__)).parent
INC_PATH = os.path.join(PRJ_PATH, "inc")
IMG_PATH = os.path.join(PRJ_PATH, "img")
DATA_PATH = os.path.join(PRJ_PATH, "data")
# Add to path
if INC_PATH not in sys.path:
  sys.path.append(INC_PATH)
  import initializer as ini
  from resources.constants import *
else:
  from resources.constants import *
  import inc.initializer as ini


THRESHOLDER_STR = PWAVE + "{p:0.2}" + SWAVE + "{s:0.2}"

def plot_cluster(PICK: pd.DataFrame, GMMA: pd.DataFrame,
                 args: argparse.Namespace) -> None:
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
  if args.verbose:
    print("Plotting the Cluster")
  PICK = PICK[(PICK[MODEL_STR].isin(args.models)) &
              (PICK[WEIGHT_STR].isin(args.weights))].reset_index(drop=True)
  GMMA = GMMA[(GMMA[MODEL_STR].isin(args.models)) &
              (GMMA[WEIGHT_STR].isin(args.weights)) &
              (GMMA[THRESHOLD_STR] >= min(args.pwave,
                                          args.swave))].reset_index(drop=True)
  Ws: int = len(args.weights)
  x: int = int(np.sqrt(Ws))
  y: int = Ws // x + int((Ws % x) != 0)
  figsize = (int(y * Ws) * 1.5, int(x * Ws - 1) * 1.5)
  for model, dataframe_m in PICK.groupby(MODEL_STR):
    fig, _axs = plt.subplots(x, y, figsize=figsize, layout='constrained')
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
    for ax in axs:
      ax.set(xlim=(min_p, max_p), ylim=(min_r, max_r))
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
    IMG_FILE = Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) +
                    UNDERSCORE_STR.join([
                        CLSTR_PLOT_STR, model, THRESHOLDER_STR.format(
                            p=args.pwave, s=args.swave)]) + PNG_EXT)
    plt.tight_layout()
    plt.savefig(IMG_FILE, bbox_inches='tight')
    plt.close()


def dist_balanced(T: pd.Series, P: pd.Series) -> float:
  return (dist_time(T, P) + 9. * dist_phase(T, P)) / 10.


def dist_phase(T: pd.Series, P: pd.Series) -> float:
  return int(P[PHASE_STR] == T[PHASE_STR])


def dist_time(T: pd.Series, P: pd.Series,
              offset: td = PICK_OFFSET) -> float:
  return 1. - (diff_time(T, P) / offset)


def diff_time(T: pd.Series, P: pd.Series) -> float:
  return td(seconds=abs(P[TIMESTAMP_STR] - T[TIMESTAMP_STR]))


def dist_default(T: pd.Series, P: pd.Series) -> float:
  return (99. * dist_balanced(T, P) + P[PROBABILITY_STR]) / 100.


MATCH_CNFG[CLSSFD_STR].update({DISTANCE_STR: dist_default})
MATCH_CNFG[DETECT_STR].update({DISTANCE_STR: dist_default})


def dist_space(T: pd.Series, P: pd.Series, offset: float = 1.) -> float:
  return 1. - (float(format(diff_space(T, P) / 1000., ".4f")) / offset)


def diff_space(T: pd.Series, P: pd.Series) -> float:
  return gps2dist_azimuth(T[LATITUDE_STR], T[LONGITUDE_STR],
                          P[LATITUDE_STR], P[LONGITUDE_STR])[0]


def dist_event(T: pd.Series, P: pd.Series,
               time_offset_sec: td = ASSOCIATE_TIME_OFFSET,
               space_offset_km: float = ASSOCIATE_DIST_OFFSET) -> float:
  return (dist_time(T, P, time_offset_sec) +
          dist_space(T, P, space_offset_km)) / 2.


MATCH_CNFG[SOURCE_STR].update({DISTANCE_STR: dist_event})


class myBPGraph():
  def __init__(self, T: pd.DataFrame, P: pd.DataFrame,
               config=MATCH_CNFG[CLSSFD_STR]):
    self.W: function = config[DISTANCE_STR]
    self.M: int = len(P.index)
    self.N: int = 0
    self.P = P.to_dict()
    self.T = {key: dict() for key in T.columns.tolist()}
    self.G: list[dict[int, dict[int, float]],
                 dict[int, dict[int, float]]] = [
        dict(), {i: dict() for i in range(self.M)}]
    self.C = config
    for _, t in T.iterrows():
      self.incNode(t, 0)

  def addNode(self, u: int, bipartite: int, neighbours=dict()) -> None:
    if u not in self.G[bipartite]:
      self.G[bipartite][u] = neighbours

  def incNode(self, u: pd.Series, bipartite: int = 1) -> None:
    if bipartite == 1:
      for key, val in u.to_dict().items():
        self.P[key][self.M] = val
      x = self.cnxNode(u, bipartite)
      self.addNode(self.M, bipartite, x)
      for node, weight in x.items():
        self.addEdge(node, self.M, 0, weight)
      self.M += 1
    else:
      for key, val in u.to_dict().items():
        self.T[key][self.N] = val
      x = self.cnxNode(u, bipartite)
      self.addNode(self.N, bipartite, x)
      for node, weight in x.items():
        self.addEdge(node, self.N, 1, weight)
      self.N += 1
    return

  def cnxNode(self, u: pd.Series, bipartite: int = 1) -> dict[int, float]:
    v = pd.DataFrame(self.T if bipartite == 1 else self.P)
    v = v[(v[TIMESTAMP_STR] - u[TIMESTAMP_STR])
          .apply(lambda x: td(seconds=abs(x))) <= self.C[TIME_DSPLCMT_STR]]
    if self.C[METHOD_STR] == SOURCE_STR:
      v[DISTANCE_STR] = 0.0
      for i, row in v.iterrows():
        v.loc[i, DISTANCE_STR] = diff_space(u, row) / 1000.
      v = v[v[DISTANCE_STR] <= ASSOCIATE_DIST_OFFSET]
    if v.empty:
      return dict()
    return {i: self.W(u, v.loc[i]) for i in v.index}

  def addEdge(self, u: int, v: int, bipartite: int, weight: float) -> None:
    self.G[bipartite].setdefault(u, dict())
    self.G[bipartite][u][v] = weight

  def rmvEdge(self, u: int, v: int, bipartite: int = 0) -> None:
    if u in self.G[bipartite]:
      self.G[bipartite][u].remove(v)

  def getNeighbours(self, u: int, b: int = 0):
    return set([(k, u, v) if b else (u, k, v)
                for k, v in self.G[b][u].items()])

  def adjMtx(self) -> np.ndarray:
    A = np.zeros((self.N, self.M))
    for t in self.G[0]:
      for p in self.G[0][t]:
        A[t][p] = self.G[0][t][p]
    return A

  def maxWmatch(self) -> list[tuple[int, int, float]]:
    MATCH = {}
    for i in range(self.N):
      x = self.getNeighbours(i, 0)
      if len(x) == 0:
        continue
      t, p, w = max(x, key=lambda x: x[-1])
      MATCH.setdefault(p, (t, w))
      if MATCH[p][-1] < w:
        MATCH[p] = (t, w)
    return [(t, p, w) for p, (t, w) in MATCH.items()]

  def makeMatch(self) -> None:
    LINKS: list[tuple[int, int, float]] = self.maxWmatch()
    if len(LINKS) == 0:
      return
    for i in range(2):
      for u in self.G[i].keys():
        self.G[i][u] = dict()
    for t, p, w in LINKS:
      self.G[0][t] = {p: w}
      self.G[1][p] = {t: w}

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
          FN.append([T[ID_STR], str(T[TIMESTAMP_STR]), T[PROBABILITY_STR],
                     T[PHASE_STR], None, T[STATION_STR]])
        else:
          CFN_MTX.loc[EVENT_STR, NONE_STR] += 1
          FN.append([T[INDEX_STR], str(T[TIMESTAMP_STR]), T[LATITUDE_STR],
                     T[LONGITUDE_STR], T[LOCAL_DEPTH_STR], T[MAGNITUDE_STR],
                     None, None, None, None, None, None, None, None])
        continue
      assert len(x) == 1
      P = nodesP.iloc[x[0]]
      if self.C[METHOD_STR] in [CLSSFD_STR, DETECT_STR]:
        CFN_MTX.loc[T[PHASE_STR], P[PHASE_STR]] += 1
        if T[PHASE_STR] == P[PHASE_STR]:
          TP.add((None, (T[ID_STR], str(P[INDEX_STR])),
                  (str(T[TIMESTAMP_STR]), str(P[TIMESTAMP_STR])),
                  (T[PROBABILITY_STR], P[PROBABILITY_STR]), T[PHASE_STR],
                  P[NETWORK_STR], T[STATION_STR]))
      else:
        CFN_MTX.loc[EVENT_STR, EVENT_STR] += 1
        TP.add((None, (T[INDEX_STR], str(P[INDEX_STR])),
                (str(T[TIMESTAMP_STR]), str(P[TIMESTAMP_STR])),
                (T[LATITUDE_STR], P[LATITUDE_STR]),
                (T[LONGITUDE_STR], P[LONGITUDE_STR]),
                (T[LOCAL_DEPTH_STR], P[LOCAL_DEPTH_STR]),
                (T[MAGNITUDE_STR], P[MAGNITUDE_STR]), None, None, None, None,
                None, None, None, (SOURCE_STR, str(P[NOTES_STR]).rstrip())))
    result = []
    for p, val in self.G[1].items():
      P = nodesP.iloc[p]
      x = list(val.keys())
      if len(x) == 0:
        if self.C[METHOD_STR] in [CLSSFD_STR, DETECT_STR]:
          CFN_MTX.loc[NONE_STR, P[PHASE_STR]] += 1
          result.append((None, P[INDEX_STR], str(P[TIMESTAMP_STR]),
                         P[PROBABILITY_STR], P[PHASE_STR], P[NETWORK_STR],
                         P[STATION_STR]))
        else:
          CFN_MTX.loc[NONE_STR, EVENT_STR] += 1
          result.append((None, P[INDEX_STR], str(P[TIMESTAMP_STR]),
                         P[LATITUDE_STR], P[LONGITUDE_STR], P[LOCAL_DEPTH_STR],
                         P[MAGNITUDE_STR], None, None, None, None, None, None,
                         None, str(P[NOTES_STR]).rstrip()))
        continue
      assert len(x) == 1
    FP.update(set(result))
    return CFN_MTX, TP, FN, FP


def conf_mtx(TRUE: pd.DataFrame, PRED: pd.DataFrame, model_name: str,
             dataset_name: str, args: argparse.Namespace,
             method: str = CLSSFD_STR) -> list[pd.DataFrame, list, list, list]:
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
  FP = set([tuple([model_name, dataset_name, *x]) for x in FP])
  return CFN_MTX, TP, FN, FP


def stat_test(TRUE: pd.DataFrame, PRED: pd.DataFrame,
              args: argparse.Namespace,
              method: str = CLSSFD_STR) -> pd.DataFrame:
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
  if args.verbose:
    print("Computing the Confusion Matrix")
  start, end = args.dates
  N_seconds = int((end + ONE_DAY - start) /
                  (2 * MATCH_CNFG[method][TIME_DSPLCMT_STR].total_seconds()))
  if method in [CLSSFD_STR, DETECT_STR]:
    PRED = PRED[((PRED[PHASE_STR] == PWAVE) &
                 (PRED[PROBABILITY_STR] >= args.pwave)) |
                ((PRED[PHASE_STR] == SWAVE) &
                 (PRED[PROBABILITY_STR] >= args.swave))].reset_index(drop=True)
  categories = MATCH_CNFG[method][CATEGORY_STR]
  Ws: int = len(args.weights)
  x: int = int(np.sqrt(Ws))
  y: int = Ws // x + int((Ws % x) != 0)
  figsize = (int(y * Ws) * 1.5, int(x * Ws - 1) * 1.5)
  TP_FILE = Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) +
                 UNDERSCORE_STR.join([method, TP_STR]) + CSV_EXT)
  FN_FILE = Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) +
                 UNDERSCORE_STR.join([method, FN_STR]) + CSV_EXT)
  FP_FILE = Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) +
                 UNDERSCORE_STR.join([method, FP_STR]) + CSV_EXT)
  if TP_FILE.exists() and FN_FILE.exists() and FP_FILE.exists() and \
     not args.force:
    if args.verbose:
      print(f"Loading {TP_FILE}")
    TP = pd.read_csv(TP_FILE)
    if args.verbose:
      print(f"Loading {FN_FILE}")
    FN = pd.read_csv(FN_FILE)
    if args.verbose:
      print(f"Loading {FP_FILE}")
    FP = pd.read_csv(FP_FILE)
    TP[ID_STR] = TP[ID_STR].apply(lambda x: eval(x))
    TP[TIMESTAMP_STR] = TP[TIMESTAMP_STR].apply(lambda x: eval(x))
    TP[TIMESTAMP_STR] = TP[TIMESTAMP_STR].apply(
        lambda x: (UTCDateTime(x[0]), UTCDateTime(x[1])))
    if method in [CLSSFD_STR, DETECT_STR]:
      TP[PROBABILITY_STR] = TP[PROBABILITY_STR].apply(lambda x: eval(x))
    else:
      TP[LATITUDE_STR] = TP[LATITUDE_STR].apply(lambda x: eval(x))
      TP[LONGITUDE_STR] = TP[LONGITUDE_STR].apply(lambda x: eval(x))
      TP[LOCAL_DEPTH_STR] = TP[LOCAL_DEPTH_STR].apply(lambda x: eval(x))
      TP[MAGNITUDE_STR] = TP[MAGNITUDE_STR].apply(lambda x: eval(x))
    TP = TP.astype({THRESHOLD_STR: str})
  else:
    TP, FN, FP = set(), [], set()
    for model, dataframe_m in PRED.groupby(MODEL_STR):
      fig, _axs = plt.subplots(x, y, figsize=figsize, layout='constrained')
      axs = _axs.flatten()
      plt.rcParams.update({'font.size': 12})
      fig.suptitle(model)
      for ax, (weight, PRED_W) in zip(axs, dataframe_m.groupby(WEIGHT_STR)):
        ax.set_title(weight)
        CFN_MTX = pd.DataFrame(0, index=categories, columns=categories,
                               dtype=int)
        cfn_mtx = pd.DataFrame(0, index=categories, columns=categories,
                               dtype=int)
        tp, fn, fp = set(), [], set()
        print(f"Processing {model} {weight}...")
        if method in [CLSSFD_STR, DETECT_STR]:
          tmp_cfn_mtx = pd.DataFrame(0, index=categories, columns=categories,
                                     dtype=int)
          tmp_tp, tmp_fn, tmp_fp = set(), [], set()
          for station, PRED_S in PRED_W.groupby(STATION_STR):
            tmp_cfn_mtx, tmp_tp, tmp_fn, tmp_fp = conf_mtx(
                TRUE[(TRUE[STATION_STR] == station)].reset_index(drop=True),
                PRED_S.reset_index(drop=True), model, weight, args, method)
            cfn_mtx += tmp_cfn_mtx
            tp = tp.union(tmp_tp)
            fn.extend(tmp_fn)
            fp = fp.union(tmp_fp)
        else:
          cfn_mtx, tp, fn, fp = conf_mtx(
              TRUE.reset_index(drop=True), PRED_W.reset_index(
                  drop=True), model,
              weight, args, method)
        CFN_MTX += cfn_mtx
        CFN_MTX.loc[NONE_STR, NONE_STR] = N_seconds - CFN_MTX.sum().sum()
        TP = TP.union(tp)
        FN.extend(fn)
        FP = FP.union(fp)
        disp = ConfMtxDisp(CFN_MTX.values, display_labels=CFN_MTX.columns)
        disp.plot(ax=ax, colorbar=False)
        disp.im_.set(clim=(1, N_seconds), cmap="Blues", norm="log")
        for labels in disp.text_.ravel():
          labels.set(color=MEX_PINK, fontsize=12, fontweight="bold")
      # TODO: Implement the properties of the plot as a function of the number
      #       of weights and coordinates
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
          Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) +
               UNDERSCORE_STR.join([
                   method, CFN_MTX_STR, model, THRESHOLDER_STR.format(
                       p=args.pwave, s=args.swave)]) + PNG_EXT)
      plt.savefig(IMG_FILE, bbox_inches='tight')
      plt.close()
    HEADER = HEADER_MODL + MATCH_CNFG[method][HEADER_STR]
    # True Positives
    TP = pd.DataFrame(TP, columns=HEADER).sort_values(
        SORT_HIERARCHY_PRED).reset_index(drop=True)
    # False Negatives
    FN = pd.DataFrame(FN, columns=HEADER).sort_values(SORT_HIERARCHY_PRED)
    # False Positives
    FP = pd.DataFrame(FP, columns=HEADER).sort_values(SORT_HIERARCHY_PRED)
  if args.verbose:
    TP.to_csv(TP_FILE, index=False)
    FN.to_csv(FN_FILE, index=False)
    FP.to_csv(FP_FILE, index=False)

  if method in [CLSSFD_STR, DETECT_STR]:
    print("True Positives")
    tp = pd.DataFrame([*zip(
        TP[[MODEL_STR, WEIGHT_STR]].agg(SPACE_STR.join, axis=1),
        TP[PHASE_STR].astype(str))], columns=[ID_STR, PHASE_STR])
    MTX = pd.DataFrame(0, index=tp[ID_STR].unique(),
                       columns=tp[PHASE_STR].unique(), dtype=int)
    for id, val in tp[[ID_STR, PHASE_STR]].value_counts().items():
      MTX.loc[id[0], id[1]] = val
    MTX["Total"] = MTX.sum(axis=1)
    MTX.sort_index(inplace=True)
    print(MTX.to_string(), end="\n\n")
    print("False Negatives")
    fn = pd.DataFrame([*zip(
        FN[[MODEL_STR, WEIGHT_STR]].agg(SPACE_STR.join, axis=1),
        FN[PHASE_STR].astype(str))], columns=[ID_STR, PHASE_STR])
    MTX = pd.DataFrame(0, index=fn[ID_STR].unique(),
                       columns=fn[PHASE_STR].unique(), dtype=int)
    for id, val in fn[[ID_STR, PHASE_STR]].value_counts().items():
      MTX.loc[id[0], id[1]] = val
    MTX["Total"] = MTX.sum(axis=1)
    MTX.sort_index(inplace=True)
    print(MTX.to_string(), end="\n\n")
    print("False Positives")
    fp = pd.DataFrame([*zip(
        FP[[MODEL_STR, WEIGHT_STR]].agg(SPACE_STR.join, axis=1),
        FP[PHASE_STR].astype(str))], columns=[ID_STR, PHASE_STR])
    MTX = pd.DataFrame(0, index=fp[ID_STR].unique(),
                       columns=fp[PHASE_STR].unique(), dtype=int)
    for id, val in fp[[ID_STR, PHASE_STR]].value_counts().items():
      MTX.loc[id[0], id[1]] = val
    MTX["Total"] = MTX.sum(axis=1)
    MTX.sort_index(inplace=True)
    print(MTX.to_string(), end="\n\n")
    tp = [float(t[1]) for t in TP[ID_STR].unique()]

  # True Positive Probability distribution plot
  if method in [CLSSFD_STR, DETECT_STR]:
    for model, tp_m in TP.groupby(MODEL_STR):
      fig, _axs = plt.subplots(x, y, figsize=figsize, layout='tight')
      axs = _axs.flatten()
      plt.rcParams.update({'font.size': 12})
      fig.suptitle(model)
      for ax, (weight, tp_w) in zip(axs, tp_m.groupby(WEIGHT_STR)):
        ax.set_title(weight)
        tp_w[TRUE_STR] = tp_w[PROBABILITY_STR].apply(lambda x: int(x[0]))
        tp_w[PRED_STR] = tp_w[PROBABILITY_STR].apply(lambda x: float(x[1]))
        tmp = dict()
        for (phase, prob), tp in tp_w.groupby([PHASE_STR, TRUE_STR]):
          key = PERIOD_STR.join([phase, str(prob)])
          tmp.setdefault(key, list())
          hist, bins = np.histogram(tp[PRED_STR], bins=NUM_BINS, range=(.1, 1))
          tmp[key].append(hist)
        tmp = pd.DataFrame([list(*hist) for hist in tmp.values()],
                           index=tmp.keys(), columns=bins[:-1], dtype=int) \
            .T.sort_index(ascending=False)
        ax.imshow(tmp, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(tmp.columns)))
        ax.set_xticklabels(tmp.columns)
        ax.set_yticks(range(0, len(tmp.index), 5))
        ax.set_yticklabels(["{:0.2f}".format(x) for x in tmp.index[::5]])
        ax.set(xlabel="Operator Weight", ylabel="Prediction Probability")
      axs[0].set()
      axs[1].set(ylabel=None, yticklabels=[])
      if len(args.weights) > 2:
        axs[2].set_xlabel(axs[2].get_title(), fontsize=14)
        axs[2].set(title=None)
        axs[2].xaxis.tick_top()
        axs[3].set_xlabel(axs[3].get_title(), fontsize=14)
        axs[3].set(ylabel=None, yticklabels=[], title=None)
        axs[3].xaxis.tick_top()
      IMG_FILE = Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) +
                      UNDERSCORE_STR.join([
                          method, TP_STR, model, THRESHOLDER_STR.format(
                              p=args.pwave, s=args.swave)]) + PNG_EXT)
      plt.tight_layout()
      plt.savefig(IMG_FILE, bbox_inches='tight')
      plt.close()

  # False Negative Pie plot
  if method in [CLSSFD_STR, DETECT_STR]:
    print("Plotting FN Pie")
    for (model, weight), df in FN.groupby([MODEL_STR, WEIGHT_STR]):
      fig, _ax = plt.subplots(1, 2, figsize=(10, 5))
      plt.suptitle(SPACE_STR.join([model, weight]), fontsize=16)
      for ax, phase in zip(_ax, [PWAVE, SWAVE]):
        ax.set_title(phase)
        df[df[PHASE_STR] == phase][PROBABILITY_STR].value_counts()\
            .plot(kind='pie', ax=ax, autopct='%1.1f%%')
      IMG_FILE = Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) +
                      UNDERSCORE_STR.join([
                          method, FN_STR, model, weight,
                          THRESHOLDER_STR.format(p=args.pwave,
                                                 s=args.swave)]) + PNG_EXT)
      plt.tight_layout()
      plt.savefig(IMG_FILE, bbox_inches='tight')
      plt.close()
  else:
    print("Plotting FN Pie")
    pass

  if method in [CLSSFD_STR, DETECT_STR]:
    # TP FN
    groups = [MODEL_STR, WEIGHT_STR, PHASE_STR]
    max_threshold = max(TP.groupby(groups)[THRESHOLD_STR].value_counts().max(),
                        FN.groupby(groups)[THRESHOLD_STR].value_counts().max())
    max_threshold = (max_threshold + 9) // 10 * 10
    for model in args.models:
      TP_M = TP[TP[MODEL_STR] == model]
      FN_M = FN[FN[MODEL_STR] == model]
      fig, _axs = plt.subplots(x, y, figsize=figsize, layout='constrained')
      axs = _axs.flatten()
      fig.suptitle(model)
      plt.rcParams.update({'font.size': 12})
      for ax1, w in zip(axs, args.weights):
        TP_W = TP_M[TP_M[WEIGHT_STR] == w]
        FN_W = FN_M[FN_M[WEIGHT_STR] == w]
        if len(TP_W.index) == 0 and len(FN_W.index) == 0:
          continue
        ax1.set_title(w, fontsize=16)
        ax2 = ax1.twinx()
        ax1.set(ylabel="Number of Picks", ylim=(0, max_threshold))
        RECALL = {
            PWAVE: (0., 0.),
            SWAVE: (0., 0.),
            RECALL_STR: 0.
        }
        for p in [PWAVE, SWAVE]:
          tp = TP_W.loc[TP_W[PHASE_STR] == p, THRESHOLD_STR].value_counts() \
              .sort_index()
          fn = FN_W.loc[FN_W[PHASE_STR] == p, THRESHOLD_STR].value_counts() \
              .sort_index()
          RECALL[p] = (tp, fn)
        RECALL[RECALL_STR] = (RECALL[PWAVE][0] + RECALL[SWAVE][0]) / \
                             (RECALL[PWAVE][0] + RECALL[PWAVE][1] +
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
          IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) +
          UNDERSCORE_STR.join([method, "TPFN", model, THRESHOLDER_STR.format(
              p=args.pwave, s=args.swave)]) + PNG_EXT)
      plt.tight_layout()
      plt.savefig(IMG_FILE, bbox_inches='tight')
      plt.close()
  else:
    if args.rectdomain:
      pm = 0.5
      xy = [args.rectdomain[0], args.rectdomain[2]]
      w = args.rectdomain[1] - args.rectdomain[0]
      h = args.rectdomain[3] - args.rectdomain[2]
      extent = [args.rectdomain[0] - pm, args.rectdomain[1] + pm,
                args.rectdomain[2] - pm, args.rectdomain[3] + pm]
    if args.circdomain:
      raise NotImplementedError
      max_r = args.circdomain[3] + 0.5
      lat, lon = args.circdomain[:2]
      extent = [lon - max_r, lon + max_r, lat - max_r, lat + max_r]
    # True Positive Radii distribution plot
    for model, tp_m in TP.groupby(MODEL_STR):
      fig, _axs = plt.subplots(x, y, figsize=figsize, layout='constrained',)
      axs = _axs.flatten()
      plt.rcParams.update({'font.size': 12})
      fig.suptitle(model)
      for ax, (weight, tp_w) in zip(axs, tp_m.groupby(WEIGHT_STR)):
        if len(tp_w.index) == 0:
          continue
        tp_w["X"] = tp_w[LONGITUDE_STR].apply(lambda x: x[0] - x[1])
        tp_w["Y"] = tp_w[LATITUDE_STR].apply(lambda x: x[0] - x[1])
        r = np.sqrt(tp_w["X"]**2 + tp_w["Y"]**2)
        r.plot(ax=ax, kind='hist',
               bins=np.linspace(0., ASSOCIATE_DIST_OFFSET, NUM_BINS))
        ax.set(title=weight, xlabel="Distance (km)",
               ylabel="Number of Picks")
      axs[0].set()
      axs[1].set(ylabel=None, yticklabels=[])
      if len(args.weights) > 2:
        axs[2].set_xlabel(axs[2].get_title(), fontsize=14)
        axs[2].set(title=None)
        axs[2].xaxis.tick_top()
        axs[3].set_xlabel(axs[3].get_title(), fontsize=14)
        axs[3].set(ylabel=None, yticklabels=[], title=None)
        axs[3].xaxis.tick_top()
      IMG_FILE = Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) +
                      UNDERSCORE_STR.join([
                          method, TRUE_STR, "R", PRED_STR, model,
                          THRESHOLDER_STR.format(p=args.pwave,
                                                 s=args.swave)]) + PNG_EXT)
      plt.tight_layout()
      plt.savefig(IMG_FILE, bbox_inches='tight')
      plt.close()

    # True Positive Spatial plot
    for model, tp_m in TP.groupby(MODEL_STR):
      proj = ccrs.PlateCarree()
      fig, _axs = plt.subplots(x, y, figsize=figsize, layout='constrained',
                               subplot_kw={'projection': proj})
      axs = _axs.flatten()
      plt.rcParams.update({'font.size': 12})
      fig.suptitle(model)
      for ax, (weight, tp_w) in zip(axs, tp_m.groupby(WEIGHT_STR)):
        if len(tp_w.index) == 0:
          continue
        ax.add_patch(mpatches.Polygon(
            OGS_POLY_REGION, closed=True, linewidth=1, color='red', fill=False,
            label="OGS Catalog"))
        ax.add_patch(mpatches.Rectangle(xy, w, h, linewidth=1, color='blue',
                                        fill=False, label=OGS_STUDY_STR))
        ax.plot(tp_w[LONGITUDE_STR].apply(lambda x: x[1]),
                tp_w[LATITUDE_STR].apply(lambda x: x[1]), "o", markersize=2,
                color=MEX_PINK, alpha=0.5)
        ax.legend(fontsize=16)
        ax.add_feature(cfeature.OCEAN, facecolor=("lightblue"))
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor=MEX_PINK)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
        ax.set_extent(extent, crs=proj)
        ax.set_aspect('equal', adjustable='box')
        ax.set(title=weight)
        ax.legend()
      gl = axs[0].gridlines()
      gl.left_labels = True
      gl.bottom_labels = True
      gl = axs[1].gridlines()
      gl.bottom_labels = True
      if len(args.weights) > 2:
        axs[2].set_title(axs[2].get_title(), y=0, pad=-14)
        gl = axs[2].gridlines()
        gl.left_labels = True
        axs[3].set_title(axs[3].get_title(), y=0, pad=-14)
        gl = axs[3].gridlines()
      IMG_FILE = Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) +
                      UNDERSCORE_STR.join([
                          method, TP_STR, model, THRESHOLDER_STR.format(
                              p=args.pwave, s=args.swave)]) + PNG_EXT)
      plt.tight_layout()
      plt.savefig(IMG_FILE, bbox_inches='tight')
      plt.close()

  # File all the results for the date range
  STAT = list()
  for model in args.models:
    for weight in args.weights:
      for stat in [TP_STR, FP_STR, FN_STR]:
        if method in [CLSSFD_STR, DETECT_STR]:
          for phase in [PWAVE, SWAVE]:
            STAT.append([model, weight, stat + phase, *([0.]*len(THRESHOLDS))])
        else:
          STAT.append([model, weight, stat, *([0.] * len(THRESHOLDS))])
  STAT = pd.DataFrame(STAT, columns=HEADER_STAT)
  STAT_FILEPATH = Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) +
                       UNDERSCORE_STR.join([
                           method, STAT_STR, start.strftime(DATE_FMT),
                           end.strftime(DATE_FMT)]) + CSV_EXT)
  if STAT_FILEPATH.exists():
    stat = pd.read_csv(STAT_FILEPATH)
    head = [MODEL_STR, WEIGHT_STR, STAT_STR]
    for _, row in stat.iterrows():
      r = STAT.loc[(STAT[head] == row[head]).all(axis=1)]
      # If the row is not in the dataframe, add it, update it, otherwise
      if r.empty:
        STAT.loc[len(STAT.index)] = row.tolist()
      else:
        STAT.loc[(STAT[head] == row[head]).all(axis=1)] = row.tolist()
  if method in [CLSSFD_STR, DETECT_STR]:
    groups = [MODEL_STR, WEIGHT_STR, PHASE_STR]
    for (model, weight, phase), df in TP.groupby(groups):
      tp = len(df[df[PHASE_STR] == phase].index)
      STAT.loc[(STAT[MODEL_STR] == model) & (STAT[WEIGHT_STR] == weight) &
               (STAT[STAT_STR] == TP_STR + phase), "{:.1f}".format(
          args.pwave if phase == PWAVE else args.swave)] = tp
    for (model, weight, phase), df in FN.groupby(groups):
      fn = len(df[df[PHASE_STR] == phase].index)
      STAT.loc[(STAT[MODEL_STR] == model) & (STAT[WEIGHT_STR] == weight) &
               (STAT[STAT_STR] == FN_STR + phase), "{:.1f}".format(
          args.pwave if phase == PWAVE else args.swave)] = fn
    for (model, weight, phase), df in FP.groupby(groups):
      fp = len(df[df[PHASE_STR] == phase].index)
      STAT.loc[(STAT[MODEL_STR] == model) & (STAT[WEIGHT_STR] == weight) &
               (STAT[STAT_STR] == FP_STR + phase), "{:.1f}".format(
          args.pwave if phase == PWAVE else args.swave)] = fp
  else:
    for (model, weight), df in TP.groupby([MODEL_STR, WEIGHT_STR]):
      STAT.loc[(STAT[MODEL_STR] == model) & (STAT[WEIGHT_STR] == weight) &
               (STAT[STAT_STR] == TP_STR), "{:.1f}".format(args.pwave)] = \
          len(df.index)
    for (model, weight), df in FN.groupby([MODEL_STR, WEIGHT_STR]):
      STAT.loc[(STAT[MODEL_STR] == model) & (STAT[WEIGHT_STR] == weight) &
               (STAT[STAT_STR] == FN_STR), "{:.1f}".format(args.pwave)] = \
          len(df.index)
    for (model, weight), df in FP.groupby([MODEL_STR, WEIGHT_STR]):
      STAT.loc[(STAT[MODEL_STR] == model) & (STAT[WEIGHT_STR] == weight) &
               (STAT[STAT_STR] == FP_STR), "{:.1f}".format(args.pwave)] = \
          len(df.index)
  STAT.to_csv(STAT_FILEPATH, index=False)
  # TODO: Redo the plots for the True Positives, False Negatives and False
  #       Positives
  # Plot the True Positives, False Negatives histogram and the Recall as a
  # function of the threshold for each model and weight
  return TP


def time_displacement(DATA: pd.DataFrame, args: argparse.Namespace,
                      method: str = CLSSFD_STR) -> None:
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
  if args.verbose:
    print("Plotting the Time Displacement")
  DATA[TIMESTAMP_STR] = DATA[TIMESTAMP_STR].map(
      lambda x: UTCDateTime(x[0]) - UTCDateTime(x[1]))
  sec = MATCH_CNFG[method][TIME_DSPLCMT_STR].total_seconds()
  bins = np.linspace(-sec, sec, NUM_BINS, endpoint=True)
  width = bins[1] - bins[0]
  groups = [MODEL_STR, WEIGHT_STR]
  if method in [CLSSFD_STR, DETECT_STR]:
    groups.append(PHASE_STR)
  m = 0
  for _, dtfrm in DATA.groupby(groups):
    counts, _ = np.histogram(dtfrm[TIMESTAMP_STR], bins=bins)
    m = max(m, max(counts))
  m = (m + 9) // 10 * 10
  Ws: int = len(args.weights)
  x: int = int(np.sqrt(Ws))
  y: int = Ws // x + int((Ws % x) != 0)
  figsize = (int(y * Ws) * 1.5, int(x * Ws - 1) * 1.5)
  if method in [CLSSFD_STR, DETECT_STR]:
    DATA[PROBABILITY_STR] = DATA[PROBABILITY_STR].map(lambda x: x[1])
    DATA = DATA.astype({THRESHOLD_STR: str})
    for phase, df_p in DATA.groupby(PHASE_STR):
      for model, df_m in df_p.groupby(MODEL_STR):
        fig, _axs = plt.subplots(x, y, figsize=figsize, layout='constrained')
        axs = _axs.flatten()
        plt.rcParams.update({'font.size': 12})
        fig.suptitle(model)
        for ax, (weight, df_w) in zip(axs, df_m.groupby(WEIGHT_STR)):
          counts, _ = np.histogram(df_w[TIMESTAMP_STR], bins=bins)
          mu = np.mean(df_w[TIMESTAMP_STR])
          std = np.std(df_w[TIMESTAMP_STR])
          ax.bar(bins[:-1], counts, alpha=0.5, width=width,
                 label=rf"$\mu$={mu:.2f},$\sigma$={std:.2f}")
          for th in THRESHOLDS:
            # TODO: Consider a KDE plot
            counts, _ = np.histogram(
                df_w[df_w[THRESHOLD_STR] == th][TIMESTAMP_STR], bins=bins)
            ax.step(bins[:-1], counts, where='mid', label=rf"[{th})")
          counts, _ = np.histogram(
              df_w[df_w[PROBABILITY_STR] == THRESHOLDS[-1]][TIMESTAMP_STR],
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
        IMG_FILE = Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) +
                        UNDERSCORE_STR.join([
                            method, TIME_DSPLCMT_STR, model, phase,
                            THRESHOLDER_STR.format(p=args.pwave,
                                                   s=args.swave)]) + PNG_EXT)
        plt.tight_layout()
        plt.savefig(IMG_FILE, bbox_inches='tight')
        plt.close()
  else:
    for model, df_m in DATA.groupby(MODEL_STR):
      fig, _axs = plt.subplots(x, y, figsize=figsize, layout='constrained')
      axs = _axs.flatten()
      plt.rcParams.update({'font.size': 12})
      fig.suptitle(model)
      for ax, (weight, df_w) in zip(axs, df_m.groupby(WEIGHT_STR)):
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
      IMG_FILE = Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) +
                      UNDERSCORE_STR.join([
                          method, TIME_DSPLCMT_STR, model,
                          THRESHOLDER_STR.format(p=args.pwave,
                                                 s=args.swave)]) + PNG_EXT)
      plt.tight_layout()
      plt.savefig(IMG_FILE, bbox_inches='tight')
      plt.close()


def _Analysis(args: argparse.Namespace,
              method: str = CLSSFD_STR) -> pd.DataFrame:
  DF: pd.DataFrame = pd.DataFrame(columns=HEADER_MODL +
                                  MATCH_CNFG[method][HEADER_STR])
  FILEPATH = Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + method +
                  CSV_EXT)
  if (not args.force and FILEPATH.exists() and
          ini.read_args(args, False) == ini.dump_args(args, True)):
    print(f"Loading {FILEPATH}")
    DF = ini.data_loader(FILEPATH)
  else:
    DF = ini.classified_loader(args) if method == CLSSFD_STR else \
        ini.associated_loader(args)
  DF[TIMESTAMP_STR] = DF[TIMESTAMP_STR].apply(lambda x: UTCDateTime(x))
  start, end = args.dates
  DF = DF[DF[TIMESTAMP_STR].between(start, end + ONE_DAY, inclusive='left')]
  if args.verbose:
    DF.to_csv(FILEPATH, index=False)
  else:
    return DF
  if method == CLSSFD_STR:
    return DF
  global DATES
  if DATES is None:
    DATES = [start]
    while DATES[-1] <= end:
      DATES.append(DATES[-1] + ONE_DAY)
  dates = [np.datetime64(d) for d in DATES]
  Ws: int = len(args.weights)
  x: int = int(np.sqrt(Ws))
  y: int = Ws // x + int((Ws % x) != 0)
  figsize = (int(y * Ws) * 1.5, int(x * Ws - 1) * 1.5)
  for (model, phase), dataframe_m in DF.groupby([MODEL_STR, PHASE_STR]):
    fig, _axs = plt.subplots(x, y, figsize=figsize, layout='constrained')
    axs = _axs.flatten()
    plt.rcParams.update({'font.size': 12})
    fig.suptitle(model)
    max_h = 0
    for ax, (weight, df_w) in zip(axs, dataframe_m.groupby(WEIGHT_STR)):
      ax.set_title(weight)
      for th in THRESHOLDS:
        h = [len(df_w[(df_w[THRESHOLD_STR] == th) &
                      (df_w[TIMESTAMP_STR] <= d)].index) for d in DATES]
        ax.plot(dates, h, label=rf"$\geq$ {th}")
        max_h = max(max_h, max(h))
      ax.set(xlabel=None, yscale="log", ylabel="Number of Picks")
      ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
      for label in ax.get_xticklabels():
        label.set(rotation=30, horizontalalignment='right')
      ax.grid()
      ax.legend()
    for ax in axs:
      ax.set(ylim=(1, max_h))
    axs[0].set(xticklabels=[])
    axs[1].set(xticklabels=[], ylabel=None, yticklabels=[])
    if len(args.weights) > 2:
      axs[2].set()
      axs[3].set(ylabel=None, yticklabels=[])
    plt.tight_layout()
    IMG_FILE = Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) +
                    UNDERSCORE_STR.join([
                        CMTV_PICKS_STR, method, model, phase]) + PNG_EXT)
    plt.savefig(IMG_FILE, bbox_inches='tight')
    plt.close()
  return DF


def main(args: argparse.Namespace):
  global DATA_PATH
  DATA_PATH = args.directory.parent
  TRUE_S, TRUE_D = ini.true_loader(args)
  if args.option in [CLSSFD_STR, ALL_WILDCHAR_STR]:
    print(CLSSFD_STR)
    PRED = _Analysis(args, CLSSFD_STR)
    time_displacement(stat_test(dcpy(TRUE_D), dcpy(PRED), args, CLSSFD_STR),
                      args, CLSSFD_STR)
    pass
  if args.option in [DETECT_STR, ALL_WILDCHAR_STR]:
    print(DETECT_STR)
    PRED_D = _Analysis(args, DETECT_STR)
    time_displacement(stat_test(dcpy(TRUE_D), dcpy(PRED_D), args, DETECT_STR),
                      args, DETECT_STR)
    pass
  if args.option == ALL_WILDCHAR_STR:
    plot_cluster(PRED, PRED_D, args)
    # plot_cluster(TRUE_D, PRED_D, args)
  del PRED
  del PRED_D
  if args.option in [SOURCE_STR, ALL_WILDCHAR_STR]:
    print(SOURCE_STR)
    PRED_S = ini.data_loader(Path(
        DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + SOURCE_STR +
        CSV_EXT))
    PRED_S[TIMESTAMP_STR] = PRED_S[TIMESTAMP_STR].apply(lambda x:
                                                        UTCDateTime(x))
    time_displacement(stat_test(TRUE_S, PRED_S, args, SOURCE_STR), args,
                      SOURCE_STR)


if __name__ == "__main__":
  main(ini.parse_arguments())

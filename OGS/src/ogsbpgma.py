import numpy as np
import pandas as pd
from datetime import timedelta as td, datetime
from obspy.geodetics import gps2dist_azimuth
try:
  # Optional dependency; used only when Hungarian matching is requested
  from scipy.optimize import linear_sum_assignment as _lsa
  _HAS_SCIPY = True
except Exception:
  _HAS_SCIPY = False

import ogsconstants as OGS_C

def dist_balanced(T: pd.Series, P: pd.Series) -> float:
  return (dist_time(T, P) + 9. * dist_phase(T, P)) / 10.

def dist_phase(T: pd.Series, P: pd.Series) -> float:
  return int(P[OGS_C.PHASE_STR] == T[OGS_C.PHASE_STR])

def dist_time(T: pd.Series, P: pd.Series,
              offset: td = OGS_C.PICK_OFFSET) -> float:
  return 1. - (diff_time(T, P) / offset)

def diff_time(T: pd.Series, P: pd.Series) -> float:
  return td(seconds=abs(P[OGS_C.TIMESTAMP_STR] - T[OGS_C.TIMESTAMP_STR]))

def dist_default(T: pd.Series, P: pd.Series) -> float:
  return (99. * dist_balanced(T, P) + P[OGS_C.PROBABILITY_STR]) / 100.

OGS_C.MATCH_CNFG[OGS_C.CLSSFD_STR].update({OGS_C.DISTANCE_STR: dist_default})
OGS_C.MATCH_CNFG[OGS_C.DETECT_STR].update({OGS_C.DISTANCE_STR: dist_default})

def dist_space(T: pd.Series, P: pd.Series, offset: float = 1.) -> float:
  return 1. - (float(format(diff_space(T, P) / 1000., ".4f")) / offset)

def diff_space(T: pd.Series, P: pd.Series) -> float:
  return gps2dist_azimuth(T[OGS_C.LATITUDE_STR], T[OGS_C.LONGITUDE_STR],
                          P[OGS_C.LATITUDE_STR], P[OGS_C.LONGITUDE_STR])[0]

def dist_event(T: pd.Series, P: pd.Series,
               time_offset_sec: td = OGS_C.ASSOCIATE_TIME_OFFSET,
               space_offset_km: float = OGS_C.ASSOCIATE_DIST_OFFSET) -> float:
  return (dist_time(T, P, time_offset_sec) +
          dist_space(T, P, space_offset_km)) / 2.

OGS_C.MATCH_CNFG[OGS_C.SOURCE_STR].update({OGS_C.DISTANCE_STR: dist_event})

class myBPGraph():
  def __init__(self, T: pd.DataFrame, P: pd.DataFrame,
               config=OGS_C.MATCH_CNFG[OGS_C.CLSSFD_STR]):
    self.W: function = config[OGS_C.DISTANCE_STR]
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
    v = v[(v[OGS_C.TIMESTAMP_STR] - u[OGS_C.TIMESTAMP_STR])
          .apply(lambda x: td(seconds=abs(x))) <= self.C[OGS_C.TIME_DSPLCMT_STR]]
    if self.C[OGS_C.METHOD_STR] == OGS_C.SOURCE_STR:
      v[OGS_C.DISTANCE_STR] = 0.0
      for i, row in v.iterrows():
        v.loc[i, OGS_C.DISTANCE_STR] = diff_space(u, row) / 1000.
      v = v[v[OGS_C.DISTANCE_STR] <= OGS_C.ASSOCIATE_DIST_OFFSET]
    if v.empty:
      return dict()
    return {i: self.W(u, v.loc[i]) for i in v.index}

  def addEdge(self, u: int, v: int, bipartite: int, weight: float) -> None:
    self.G[bipartite].setdefault(u, dict())
    self.G[bipartite][u][v] = weight

  def rmvEdge(self, u: int, v: int, bipartite: int = 0) -> None:
    if u in self.G[bipartite]:
      self.G[bipartite][u].pop(v, None)

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

  def _hungarian_match(self, min_weight: float = 0.0) \
      -> list[tuple[int, int, float]]:
    """
    Compute a maximum-weight (approx via cost minimization) bipartite matching
    using the Hungarian algorithm. Requires SciPy. Falls back to greedy if
    SciPy is not available.

    min_weight: discard pairs whose edge weight is below this threshold or
                that are non-edges (weight implicitly 0.0 in adjacency).
    """
    try:
      from scipy.optimize import linear_sum_assignment as _lsa_local
    except Exception:
      # Graceful fallback when SciPy is unavailable
      return self.maxWmatch()
    # Build adjacency matrix (weights in [0,1])
    A = self.adjMtx()
    if A.size == 0:
      return []
    # Convert to costs for minimization; prefer high weights
    C = 1.0 - A
    rows, cols = _lsa_local(C)
    LINKS: list[tuple[int, int, float]] = []
    for t, p in zip(rows, cols):
      w = float(A[t, p])
      # Keep only if there was an original edge and above threshold
      if p in self.G[0].get(t, {}) and w >= min_weight:
        LINKS.append((t, p, w))
    return LINKS

  def makeMatch(self, use_hungarian: bool = False, min_weight: float = 0.0) \
      -> None:
    LINKS: list[tuple[int, int, float]] = (
      self._hungarian_match(min_weight=min_weight) if use_hungarian
      else self.maxWmatch()
    )
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
    match_vals = self.C[OGS_C.CATEGORY_STR]
    CFN_MTX = pd.DataFrame(0, index=match_vals, columns=match_vals, dtype=int)
    # We traverse the TRUE nodes of the graph to extract relevant information
    # to the True Positives and False Negatives lists
    nodesT = pd.DataFrame(self.T).reset_index(drop=True)
    nodesP = pd.DataFrame(self.P).reset_index(drop=True)
    for t, val in self.G[0].items():
      T = nodesT.iloc[t]
      x = list(val.keys())
      if len(x) == 0:
        if self.C[OGS_C.METHOD_STR] in [OGS_C.CLSSFD_STR, OGS_C.DETECT_STR]:
          CFN_MTX.loc[T[OGS_C.PHASE_STR], OGS_C.NONE_STR] += 1
          FN.append([T[OGS_C.INDEX_STR], str(T[OGS_C.TIMESTAMP_STR]),
                     T[OGS_C.PHASE_STR], T[OGS_C.STATION_STR],
                     T[OGS_C.ERT_STR]])
        else:
          CFN_MTX.loc[OGS_C.EVENT_STR, OGS_C.NONE_STR] += 1
          FN.append([T[OGS_C.INDEX_STR], str(T[OGS_C.TIMESTAMP_STR]),
                     T[OGS_C.LATITUDE_STR], T[OGS_C.LONGITUDE_STR],
                     T[OGS_C.DEPTH_STR], T[OGS_C.MAGNITUDE_L_STR],
                     T[OGS_C.ERZ_STR], T[OGS_C.ERH_STR], None])
        continue
      assert len(x) == 1
      P = nodesP.iloc[x[0]]
      if self.C[OGS_C.METHOD_STR] in [OGS_C.CLSSFD_STR, OGS_C.DETECT_STR]:
        CFN_MTX.loc[T[OGS_C.PHASE_STR], P[OGS_C.PHASE_STR]] += 1
        if T[OGS_C.PHASE_STR] == P[OGS_C.PHASE_STR]:
          TP.add((None, (T[OGS_C.INDEX_STR], str(P[OGS_C.INDEX_STR])),
                  (str(T[OGS_C.TIMESTAMP_STR]), str(P[OGS_C.TIMESTAMP_STR])),
                  T[OGS_C.PHASE_STR], T[OGS_C.STATION_STR],
                  (T[OGS_C.ERT_STR], P[OGS_C.PROBABILITY_STR])))
      else:
        CFN_MTX.loc[OGS_C.EVENT_STR, OGS_C.EVENT_STR] += 1
        TP.add((None, (T[OGS_C.INDEX_STR], str(P[OGS_C.ID_STR])),
                (str(T[OGS_C.TIMESTAMP_STR]), str(P[OGS_C.TIMESTAMP_STR])),
                (T[OGS_C.LATITUDE_STR], P[OGS_C.LATITUDE_STR]),
                (T[OGS_C.LONGITUDE_STR], P[OGS_C.LONGITUDE_STR]),
                (T[OGS_C.DEPTH_STR], P[OGS_C.DEPTH_STR]),
                (T[OGS_C.MAGNITUDE_L_STR], P[OGS_C.MAGNITUDE_L_STR]),
                (T[OGS_C.ERZ_STR], P[OGS_C.ERZ_STR]),
                (T[OGS_C.ERH_STR], P[OGS_C.ERH_STR]),
                (OGS_C.SOURCE_STR, str(P[OGS_C.NOTES_STR]).rstrip())))
    result = []
    for p, val in self.G[1].items():
      P = nodesP.iloc[p]
      x = list(val.keys())
      if len(x) == 0:
        if self.C[OGS_C.METHOD_STR] in [OGS_C.CLSSFD_STR, OGS_C.DETECT_STR]:
          CFN_MTX.loc[OGS_C.NONE_STR, P[OGS_C.PHASE_STR]] += 1
          result.append((None, P[OGS_C.INDEX_STR], str(P[OGS_C.TIMESTAMP_STR]),
                         P[OGS_C.PHASE_STR], P[OGS_C.STATION_STR],
                         P[OGS_C.PROBABILITY_STR]))
        else:
          CFN_MTX.loc[OGS_C.NONE_STR, OGS_C.EVENT_STR] += 1
          result.append((None, P[OGS_C.ID_STR], str(P[OGS_C.TIMESTAMP_STR]),
                         P[OGS_C.LATITUDE_STR], P[OGS_C.LONGITUDE_STR],
                         P[OGS_C.DEPTH_STR], P[OGS_C.MAGNITUDE_L_STR],
                         P[OGS_C.ERZ_STR], P[OGS_C.ERH_STR],
                         str(P[OGS_C.NOTES_STR]).rstrip()))
        continue
      assert len(x) == 1
    FP.update(set(result))
    return CFN_MTX, TP, FN, FP

def conf_mtx(TRUE: pd.DataFrame, PRED: pd.DataFrame, model_name: str,
       dataset_name: str, method: str = OGS_C.CLSSFD_STR,
       use_hungarian: bool = False, min_weight: float = 0.0) \
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
                  PRED.reset_index(drop=True), config=OGS_C.MATCH_CNFG[method])
  bpg.makeMatch(use_hungarian=use_hungarian, min_weight=min_weight)
  # if args.interactive: plot_timeline(G, pos, N, model_name, dataset_name)
  CFN_MTX, TP, FN, FP = bpg.confMtx()
  TP = set([tuple([model_name, dataset_name, *x]) for x in TP])
  FN = [[model_name, dataset_name, None, *x] for x in FN]
  FP = set([tuple([model_name, dataset_name, *x]) for x in FP])
  return CFN_MTX, TP, FN, FP
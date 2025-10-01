import os
import re
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import itertools as it
from pathlib import Path
from obspy import UTCDateTime, Inventory, read_inventory
from matplotlib.path import Path as mplPath
from obspy.geodetics import gps2dist_azimuth
from datetime import datetime, timedelta as td

from matplotlib.cbook import flatten as flatten_list

THIS_FILE = Path(__file__)

EPSILON = 1e-6

MPI_RANK = 0
MPI_SIZE = 1
MPI_COMM = None

GPU_SIZE = 0
GPU_RANK = -1

THRESHOLDS: list[str] = ["{:.1f}".format(t) for t in np.linspace(0.1, 0.9, 9)]
DATES = None

# DateTime, TimeDelta and Format constants
DATE_STD = "YYMMDD"
DATE_FMT = "%Y-%m-%d"
TIME_FMT = "%H%M%S"
YYMMDD_FMT = "%y%m%d"
DATETIME_FMT = YYMMDD_FMT + TIME_FMT
ONE_DAY = td(days=1)
PICK_TIME_OFFSET = td(seconds=.5) # TODO: Change to .5 sec
PICK_TRAIN_OFFSET = td(seconds=60)
H71_OFFSET = {
    0: 0.01,
    1: 0.04,
    2: 0.2,
    3: 1,
    4: 5,
    5: 25
}
EVENT_TIME_OFFSET = td(seconds=1.5)
EVENT_DIST_OFFSET = 8  # km

# Strings
EMPTY_STR = ''
ALL_WILDCHAR_STR = '*'
ONE_MORECHAR_STR = '+'
PERIOD_STR = '.'
UNDERSCORE_STR = '_'
DASH_STR = '-'
SPACE_STR = ' '
COMMA_STR = ','
SEMICOL_STR = ';'
ZERO_STR = "0"
NONE_STR = "None"
CLF_STR = "SeisBenchPicker"
AST_STR = "GammaAssociator"
FILE_STR = "file"
TEMPORAL_STR = "tmp"
DURATION_STR = "duration"
STATUS_STR = "status"
SECONDS_STR = "seconds"
COMPRESSIONAL_STR = "compressional"
DILATATIONAL_STR = "dilatational"
CLSSFD_STR = "CLSSFD"
SOURCE_STR = "SOURCE"
DETECT_STR = "DETECT"
UNKNOWN_STR = "UNKNOWN"
LEVEL_STR = "LEVEL"
WARNING_STR = "WARNING"
FATAL_STR = "FATAL"
NOTABLE_STR = "NOTABLE"
ASSIGN_STR = "ASSIGN"
UNABLE_STR = "UNABLE"

TRUE_STR = "TRUE"
PRED_STR = "PRED"
ASCT_STR = "ASCT"
STAT_STR = "STAT"
FALSE_STR = "FALSE"

GMMA_STR = "GaMMA"
OCTO_STR = "PyOcto"

# Metrics
TP_STR = "TP"
FP_STR = "FP"
FN_STR = "FN"
TN_STR = "TN"
ACCURACY_STR = "AC"
PRECISION_STR = "PC"
RECALL_STR = "RC"
F1_STR = "F1"
DISTANCE_STR = "Distance"

# Phases
PWAVE = "P"
SWAVE = "S"

# Thresholds
PWAVE_THRESHOLD = SWAVE_THRESHOLD = 0.1

SEED_ID_FMT = "{NETWORK}.{STATION}..{CHANNEL}"

CFN_MTX_STR = "CM"
CMTV_PICKS_STR = "CP"
CLSTR_PLOT_STR = "CT"
TIME_DSPLCMT_STR = "TD"

MEX_PINK = "#E4007C"
OGS_BLUE = "#163771"
ALN_GREEN = "#00e468"
LIP_ORANGE = "#FF8C00"
SUN_YELLOW = "#e4da00"

# TODO: Add Tabular data for relational databases for future development

# Extensions
BLT_STR = "blt"
CSV_STR = "csv"
DAT_STR = "dat"
EPS_STR = "eps"
HDF5_STR = "hdf5"
HPC_STR = "hpc"
HPL_STR = "hpl"
JSON_STR = "json"
LD_STR = "ld"
MOD_STR = "mod"
MSEED_STR = "mseed"
PDF_STR = "pdf"
PICKLE_STR = "pkl"
PNG_STR = "png"
PRT_STR = "prt"
PUN_STR = "pun"
QML_STR = "qml"
TORCH_STR = "pt"
TXT_STR = "txt"
XML_STR = "xml"

BLT_EXT = PERIOD_STR + BLT_STR
CSV_EXT = PERIOD_STR + CSV_STR
DAT_EXT = PERIOD_STR + DAT_STR
EPS_EXT = PERIOD_STR + EPS_STR
HDF5_EXT = PERIOD_STR + HDF5_STR
HPC_EXT = PERIOD_STR + HPC_STR
HPL_EXT = PERIOD_STR + HPL_STR
JSON_EXT = PERIOD_STR + JSON_STR
LD_EXT = PERIOD_STR + LD_STR
MOD_EXT = PERIOD_STR + MOD_STR
MSEED_EXT = PERIOD_STR + MSEED_STR
PDF_EXT = PERIOD_STR + PDF_STR
PICKLE_EXT = PERIOD_STR + PICKLE_STR
PNG_EXT = PERIOD_STR + PNG_STR
PRT_EXT = PERIOD_STR + PRT_STR
PUN_EXT = PERIOD_STR + PUN_STR
QML_EXT = PERIOD_STR + QML_STR
TORCH_EXT = PERIOD_STR + TORCH_STR
TXT_EXT = PERIOD_STR + TXT_STR
XML_EXT = PERIOD_STR + XML_STR

PRC_FMT = SEED_ID_FMT + ".{BEGDT}.{EXT}"

# Models
EQTRANSFORMER_STR = "EQTransformer"
PHASENET_STR = "PhaseNet"

OGS_PROJECTION = "+proj=sterea +lon_0={lon} +lat_0={lat} +units=km"
OGS_MAX_MAGNITUDE = 3.5

# Data components
ID_STR = "idx"
INDEX_STR = "index"
TIMESTAMP_STR = "time"
TIME_STR = "time"
METADATA_STR = "metadata"
PROBABILITY_STR = "probability"
TYPE_STR = "type"
LONGITUDE_STR = "longitude"
LATITUDE_STR = "latitude"
DEPTH_STR = "depth"
ELEVATION_STR = "elevation"     # Elevation in meters
X_COORD_STR = "x(km)"           # X coordinate in kilometers
Y_COORD_STR = "y(km)"           # Y coordinate in kilometers
Z_COORD_STR = "z(km)"           # Z coordinate in kilometers
MAGNITUDE_STR = "magnitude"
MAGNITUDE_L_STR = "ML"
MAGNITUDE_D_STR = "MD"
AMPLITUDE_STR = "amplitude"
VELOCITY_STR = "vel"
METHOD_STR = "method"
DIMENSIONS_STR = "dims"
GAUSS_MIX_MODEL_STR = "GMM"
BAYES_GAUSS_MIX_MODEL_STR = "B" + GAUSS_MIX_MODEL_STR

ARGUMENTS_STR = "arguments"
WAVEFORMS_STR = "waveforms"
DATASETS_STR = "datasets"
MODELS_STR = "models"

BASE_STR = "Base"
TARGET_STR = "Target"

PHASE_STR = "phase"
EVENT_STR = "EVENT"
MODEL_STR = "MODEL"
WEIGHT_STR = "WEIGHT"
GROUPS_STR = "GROUPS"
DIRECTORY_STR = "DIRECTORY"
JULIAN_STR = "JULIAN"
DENOISER_STR = "DENOISER"
DOMAIN_STR = "DOMAIN"
CLIENT_STR = "CLIENT"
RESULTS_STR = "RESULTS"
FILENAME_STR = "FILENAME"
THRESHOLD_STR = "THRESHOLD"
NETWORK_STR = "NETWORK"
STATION_STR = "station"
CHANNEL_STR = "CHANNEL"
DATE_STR = "DATE"

# Labelled Data components
P_TIME_STR = "P_TIME"
P_TYPE_STR = "P_TYPE"
P_ONSET_STR = "P_ONSET"
P_POLARITY_STR = "P_POLARITY"
P_WEIGHT_STR = "P_WEIGHT"
S_TIME_STR = "S_TIME"
S_TYPE_STR = "S_TYPE"
S_ONSET_STR = "S_ONSET"
S_POLARITY_STR = "S_POLARITY"
S_WEIGHT_STR = "S_WEIGHT"
ORIGIN_STR = "ORIGIN"
NO_STR = "NO"
GAP_STR = "GAP"
DMIN_STR = "DMIN"
RMS_STR = "RMS"
ERH_STR = "ERH"
ERZ_STR = "ERZ"
ERT_STR = "ERT"
QM_STR = "QM"
ONSET_STR = "ONSET"
POLARITY_STR = "POLARITY"
GEO_ZONE_STR = "GEOZONE"
EVENT_TYPE_STR = "E_TYPE"
EVENT_LOCAL_EQ_STR = "local_eq"
EVENT_EXPLD_STR = "explosion"
EVENT_BOMB_STR = "bomb"
EVENT_LNDSLD_STR = "landslide"
EVENT_UNKNOWN_STR = UNKNOWN_STR
EVENT_LOCALIZATION_STR = "E_LOC"
LOC_NAME_STR = "LOC_NAME"
NOTES_STR = "NOTES"

# Pretrained model weights
ADRIAARRAY_STR = "adriaarray"
INSTANCE_STR = "instance"
ORIGINAL_STR = "original"
SCEDC_STR = "scedc"
STEAD_STR = "stead"

# Clients
INGV_CLIENT_STR = "INGV"
IRIS_CLIENT_STR = "IRIS"
GFZ_CLIENT_STR = "GFZ"
ETH_CLIENT_STR = "ETH"
ORFEUS_CLIENT_STR = "ORFEUS"
GEOFON_CLIENT_STR = "GEOFON"
RESIF_CLIENT_STR = "RESIF"
LMU_CLIENT_STR = "LMU"
USGS_CLIENT_STR = "USGS"
EMSC_CLIENT_STR = "EMSC"
ODC_CLIENT_STR = "ODC"
GEONET_CLIENT_STR = "GEONET"
OGS_CLIENT_STR = "http://158.110.30.217:8080"
RASPISHAKE_CLIENT_STR = "RASPISHAKE"
COLLALTO_CLIENT_STR = "http://scp-srv.core03.ogs.it:8080"

OGS_CLIENTS_DEFAULT = [OGS_CLIENT_STR, INGV_CLIENT_STR, GFZ_CLIENT_STR,
                       IRIS_CLIENT_STR, ETH_CLIENT_STR, ORFEUS_CLIENT_STR,
                       COLLALTO_CLIENT_STR]

# Headers
CATEGORY_STR = "CATEGORY"
HEADER_STR = "HEADER"

HEADER_MODL = [MODEL_STR, WEIGHT_STR, THRESHOLD_STR]
HEADER_FSYS = [FILENAME_STR, MODEL_STR, WEIGHT_STR, TIMESTAMP_STR, NETWORK_STR,
               STATION_STR]
HEADER_MANL = [INDEX_STR, TIMESTAMP_STR, PHASE_STR, STATION_STR, GROUPS_STR, GROUPS_STR]
HEADER_PRED = HEADER_MODL + HEADER_MANL
HEADER_SNSR = [STATION_STR, LATITUDE_STR, LONGITUDE_STR, DEPTH_STR,
               TIMESTAMP_STR]
HEADER_STAT = [MODEL_STR, WEIGHT_STR, STAT_STR] + THRESHOLDS
SORT_HIERARCHY_PRED = [MODEL_STR, WEIGHT_STR, INDEX_STR, TIME_STR]


# SPECULATIVE
MAX_PICKS_YEAR = 1e6
NUM_BINS = 41
OGS_POLY_REGION = [
    (10.0, 45.5),
    (10.0, 46.5),
    (11.5, 47.0),
    (12.5, 47.0),
    (14.5, 46.5),
    (14.5, 45.5),
    (12.5, 44.5),
    (11.5, 44.5)]
OGS_STUDY_REGION = [9.5, 15.0, 44.3, 47.5]
OGS_ITALY_STR = "Italy"
DESCRIPTION_STR = "Description"

OGS_LABEL_CATEGORY = "{GEO_ZONE_STR}{EVENT_TYPE_STR}{EVENT_LOCALIZATION_STR}"
OGS_GEO_ZONES = {
    "A": "Alto Adige",
    "C": "Croatia",
    "E": "Emilia",
    "F": "Friuli",
    "G": "Venezia Giulia",
    "L": "Lombardia",
    "O": "Austria",
    "R": "Romagna",
    "S": "Slovenia",
    "T": "Trentino",
    "V": "Veneto"
}
OGS_EVENT_TYPES = {
    "B": EVENT_BOMB_STR,
    "E": EVENT_EXPLD_STR,
    "F": EVENT_LNDSLD_STR,
    "L": EVENT_LOCAL_EQ_STR,
    "U": EVENT_UNKNOWN_STR
}
HEADER_EVENTS = [INDEX_STR, TIMESTAMP_STR, LATITUDE_STR, LONGITUDE_STR,
                 DEPTH_STR, ERH_STR, ERZ_STR, GAP_STR]
HEADER_PICKS = [INDEX_STR, TIMESTAMP_STR, PHASE_STR, STATION_STR, ONSET_STR,
                POLARITY_STR, WEIGHT_STR]

"""
def dist_balanced(B: pd.Series, T: pd.Series) -> float:
  return (dist_time(T, P) + 9. * dist_phase(T, P)) / 10.
"""

"""
def dist_default(B: pd.Series, T: pd.Series) -> float:
  return (99. * dist_balanced(T, P) + P[PROBABILITY_STR]) / 100.
"""

def dist_prob(B: pd.Series, T: pd.Series) -> float:
  return T[PROBABILITY_STR]/B[PROBABILITY_STR]

def dist_phase(B: pd.Series, T: pd.Series) -> float:
  return int(T[PHASE_STR] == B[PHASE_STR])

def diff_time(B: pd.Series, T: pd.Series) -> float:
  return B[TIMESTAMP_STR] - T[TIMESTAMP_STR]
def dist_time(B: pd.Series, T: pd.Series,
              offset: td = PICK_TIME_OFFSET) -> float:
  return 1. - (diff_time(B, T) / offset.total_seconds())

def diff_space(B: pd.Series, T: pd.Series) -> float:
  return gps2dist_azimuth(B[LATITUDE_STR], B[LONGITUDE_STR],
                          T[LATITUDE_STR], T[LONGITUDE_STR])[0]
def dist_space(B: pd.Series, T: pd.Series,
               offset: float = EVENT_DIST_OFFSET) -> float:
  return 1. - (float(format(diff_space(B, T) / 1000., ".4f")) / offset)

def dist_pick(B: pd.Series, T: pd.Series,
              time_offset_sec: td = PICK_TIME_OFFSET) -> float:
  return (
   3 * dist_time(T, B, time_offset_sec) +
   1 * dist_phase(T, B) +
   1 * dist_prob(T, B)
  ) / 5.

def dist_event(T: pd.Series, P: pd.Series,
               time_offset_sec: td = EVENT_TIME_OFFSET,
               space_offset_km: float = EVENT_DIST_OFFSET) -> float:
  return (dist_time(T, P, time_offset_sec) +
          dist_space(T, P, space_offset_km)) / 2.

def is_date(string: str) -> datetime:
  return datetime.strptime(string, YYMMDD_FMT)

def is_julian(string: str) -> datetime:
  # TODO: Define and convert Julian date to Gregorian date
  raise NotImplementedError
  return datetime.strptime(string, YYMMDD_FMT)._set_julday(string)

def is_file_path(string: str) -> Path:
  if os.path.isfile(string):
    return Path(os.path.abspath(string))
  else:
    raise FileNotFoundError(string)

def is_dir_path(string: str) -> Path:
  if os.path.isdir(string):
    return Path(os.path.abspath(string))
  else:
    raise NotADirectoryError(string)

def inventory(stations: Path) -> dict[str, tuple[float, float, float, str]]:
  import ogsplotter as OGS_P
  from matplotlib import pyplot as plt
  INVENTORY = Inventory()
  for st in stations.glob("*.xml"):
    try:
      S = read_inventory(str(st))
    except Exception as e:
      print(f"WARNING: Unable to read {st}")
      print(e)
      continue
    INVENTORY.extend(S)
  INVENTORY = {
    sta.code: (sta.longitude, sta.latitude, sta.elevation, net.code)
    for net in INVENTORY.networks for sta in net.stations
  }
  inv = pd.DataFrame.from_dict(
    INVENTORY,
    orient='index',
    columns=[LONGITUDE_STR, LATITUDE_STR, DEPTH_STR, NETWORK_STR])
  mystations = OGS_P.map_plotter(OGS_STUDY_REGION, legend=True,
                                  marker='^', output="OGS_Stations.png")
  cmap = plt.get_cmap("turbo")
  colors = cmap(np.linspace(0, 1, inv[NETWORK_STR].nunique()))
  for i, (net, sta) in enumerate(inv.groupby(NETWORK_STR)):
    mystations.add_plot(sta[LONGITUDE_STR], sta[LATITUDE_STR],
                        label=net, color=None, facecolors='none',
                        edgecolors=colors[i], legend=True)
  mystations.savefig()
  plt.close()
  return INVENTORY

class SortDatesAction(argparse.Action):
  def __call__(self, parser, namespace, values, option_string=None):
    setattr(namespace, self.dest, sorted(values)) # type: ignore

class OGSBPGraph():
  def __init__(self, Base: pd.DataFrame, Target: pd.DataFrame):
    self.Base = Base.reset_index(drop=True)
    self.Target = Target.reset_index(drop=True)
    self.G = nx.Graph()
    self.E : set[tuple[int, int]] = set()
    if not self.Base.empty and not self.Target.empty:
      self.makeMatch()

  def makeMatch(self) -> None:
    raise NotImplementedError

class OGSBPGraphPicks(OGSBPGraph):
  def __init__(self, Base: pd.DataFrame, Target: pd.DataFrame):
    if PROBABILITY_STR not in Base.columns:
      Base[PROBABILITY_STR] = 1.0
    if TIMESTAMP_STR in Base.columns:
      Base[TIMESTAMP_STR] = Base[TIMESTAMP_STR].apply(
        lambda x: UTCDateTime(x)) # type: ignore
    if "time" in Target.columns:
      Target[TIMESTAMP_STR] = Target["time"].apply(
        lambda x: UTCDateTime(x)) # type: ignore
    super().__init__(Base, Target)

  def makeMatch(self) -> None:
    I = len(self.Base)
    J = len(self.Target)
    i = 0
    j = 0
    while i < I:
      while j < J:
        if dist_time(self.Base.iloc[i], self.Target.iloc[j]) <= \
            PICK_TIME_OFFSET.total_seconds():
          d = dist_pick(self.Base.iloc[i], self.Target.iloc[j])
          if d >= PWAVE_THRESHOLD: self.G.add_edge(i, j + I, weight=d)
          j += 1
        else:
          if (self.Base.iloc[i][TIMESTAMP_STR] <
              self.Target.iloc[j][TIMESTAMP_STR]):
            break
          j += 1
      i += 1
    self.E = nx.max_weight_matching(self.G, maxcardinality=False, weight='weight')

class OGSBPGraphEvents(OGSBPGraph):
  def __init__(self, Base: pd.DataFrame, Target: pd.DataFrame):
    if "event_time" in Target.columns:
      Target[TIMESTAMP_STR] = UTCDateTime(Target["event_time"])
    if TIMESTAMP_STR in Base.columns:
      Base[TIMESTAMP_STR] = Base[TIMESTAMP_STR].apply(lambda x: UTCDateTime(x)) # type: ignore
    if TIMESTAMP_STR in Base.columns:
      Base[TIMESTAMP_STR] = Base[TIMESTAMP_STR].apply(lambda x: UTCDateTime(x)) # type: ignore
    if "time" in Target.columns:
      Target[TIMESTAMP_STR] = Target["time"].apply(lambda x: UTCDateTime(x)) # type: ignore
    super().__init__(Base, Target)

  def makeMatch(self):
    I = len(self.Base)
    J = len(self.Target)
    i = j = 0
    while i < I:
      while j < J:
        if dist_time(self.Base.iloc[i], self.Target.iloc[j]) <= EVENT_TIME_OFFSET.total_seconds():
          self.G.add_edge(
            i,
            j + I,
            weight=dist_event(self.Base.iloc[i], self.Target.iloc[j])
          )
          j += 1
        else:
          if self.Base.iloc[i][TIMESTAMP_STR] < self.Target.iloc[j][TIMESTAMP_STR]:
            break
          j += 1
      i += 1
    self.E = nx.max_weight_matching(self.G, maxcardinality=False, weight='weight')


# OGS Catalog
class OGSCatalog:
  def __init__(self, filepath: Path,
               start: datetime = datetime.max,
               end: datetime = datetime.min,
               verbose: bool = False,
               polygon : mplPath = mplPath(OGS_POLY_REGION, closed=True),
               output : Path = THIS_FILE.parent / "data" / "OGSCatalog",
               name: str = EMPTY_STR) -> None:
    assert filepath.exists(), f"Filepath {filepath} does not exist."
    self.name = output.name if name == EMPTY_STR else name
    self.filepath = filepath
    self.start = start
    self.end = end
    self.polygon : mplPath = polygon
    self.verbose = verbose
    self.output = output
    self.picks_ : dict[datetime, Path] = dict() # raw file paths
    self.events_ : dict[datetime, Path] = dict() # raw file paths
    self.picks : dict[datetime, pd.DataFrame] = dict()
    self.events : dict[datetime, pd.DataFrame] = dict()
    self.PICKS : pd.DataFrame = pd.DataFrame(columns=[
      INDEX_STR, TIMESTAMP_STR, PHASE_STR, STATION_STR, ERT_STR, NOTES_STR,
      NETWORK_STR, GROUPS_STR])
    self.EVENTS : pd.DataFrame = pd.DataFrame(columns=[
      INDEX_STR, TIMESTAMP_STR, LATITUDE_STR, LONGITUDE_STR, DEPTH_STR, NO_STR,
      GAP_STR, DMIN_STR, RMS_STR, ERH_STR, ERZ_STR, QM_STR, MAGNITUDE_L_STR,
      MAGNITUDE_D_STR, NOTES_STR,])
    self.preload()

  def load_(self, filepath : Path) -> pd.DataFrame:
    if filepath.suffix == ".csv":
      df = pd.read_csv(filepath)
      return df
    else:
      try:
        df = pd.read_parquet(filepath)
        return df
      except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame(columns=[])

  def get_(self, date: datetime, key: str) -> pd.DataFrame:
    if key == "events":
      if date in self.events: return self.events[date]
      else:
        if date not in self.events_: return pd.DataFrame(columns=[])
        df = self.load_(self.events_[date])
        self.events[date] = df
        return df
    elif key == "picks":
      if date in self.picks: return self.picks[date]
      else:
        if date not in self.picks_: return pd.DataFrame(columns=[])
        df = self.load_(self.picks_[date])
        self.picks[date] = df
        return df
    else: raise ValueError(f"Unknown key: {key}")

  def get(self, key: str) -> pd.DataFrame:
    if key == "EVENTS":
      if self.EVENTS.empty: self.postload()
      return self.EVENTS
    elif key == "PICKS":
      if self.PICKS.empty: self.postload()
      return self.PICKS
    else: raise ValueError(f"Unknown key: {key}")

  def postload(self):
    # Any post-processing after loading all data
    if self.events != {}:
      self.EVENTS = pd.concat(self.events.values(), axis=0)
    if self.picks != {}:
      self.PICKS = pd.concat(self.picks.values(), axis=0)

  def preload(self):
    for filepath in self.filepath.glob("events/*"):
      if filepath.is_file():
        date = UTCDateTime(filepath.stem).date
        if self.start.date() <= date <= self.end.date():
          self.events_[date] = filepath
    for filepath in self.filepath.glob("assignments/*"):
      if filepath.is_file():
        date = UTCDateTime(filepath.stem).date
        if self.start.date() <= date <= self.end.date():
          self.picks_[date] = filepath

  def load(self):
    print(f"Loading picks from {self.filepath} files...")

  def plot(self, other = None):
    if not self.get("EVENTS").empty:
      self.plot_events()
      self.plot_events(other)
      self.plot_magnitude_histogram(NUM_BINS, other)
      self.plot_depth_histogram(NUM_BINS, other)
      self.plot_ert_histogram(NUM_BINS, other)
      self.plot_erh_histogram(NUM_BINS)
      self.plot_erz_histogram(NUM_BINS)
      self.plot_cumulative_events(other)
    if not self.get("PICKS").empty:
      self.plot_cumulative_picks(other)

  def plot_cumulative_picks(self, other = None):
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    picks = self.get("PICKS")
    if picks.empty or TIMESTAMP_STR not in picks.columns:
      print("No Date data available for histogram.")
      return
    if (other is not None and isinstance(other, OGSCatalog) and
        GROUPS_STR in other.get("PICKS").columns):
      hist = OGS_P.day_plotter(
        picks=other.get("PICKS")[GROUPS_STR],
      )
      hist.add_plot(
        picks=picks[GROUPS_STR],
        title=f"Cumulative Picks for {self.name}",
        output=f"{self.filepath.name}_{other.filepath.name}_CumulativePicks.png"
      )
    else:
      hist = OGS_P.day_plotter(
        picks=picks[GROUPS_STR],
        title=f"Cumulative Picks for {self.name}",
        output=f"{self.filepath.name}_CumulativePicks.png"
      )
    plt.close()

  def plot_cumulative_events(self, other = None):
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    events = self.get("EVENTS")
    if events.empty or TIMESTAMP_STR not in events.columns:
      print("No Date data available for histogram.")
      return
    if (other is not None and isinstance(other, OGSCatalog) and
        GROUPS_STR in other.get("EVENTS").columns):
      hist = OGS_P.day_plotter(
        picks=other.get("EVENTS").sort_values(GROUPS_STR)[GROUPS_STR],
      )
      hist.add_plot(
        picks=events.sort_values(GROUPS_STR)[GROUPS_STR],
        title=f"Cumulative Event for {self.name}",
        output=f"{self.filepath.name}_{other.filepath.name}_CumulativeEvent.png"
      )
    else:
      hist = OGS_P.day_plotter(
        picks=events.sort_values(GROUPS_STR)[GROUPS_STR],
        title=f"Cumulative Event for {self.name}",
        output=f"{self.filepath.name}_CumulativeEvent.png"
      )
    plt.close()

  def plot_erz_histogram(self, bins: int = NUM_BINS, other = None):
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    events = self.get("EVENTS")
    if events.empty or ERZ_STR not in events.columns:
      print("No ERZ data available for histogram.")
      return
    if (other is not None and isinstance(other, OGSCatalog) and
        ERZ_STR in other.get("EVENTS").columns):
      other_events = other.get("EVENTS")
      hist = OGS_P.histogram_plotter(
        data=other_events[ERZ_STR].dropna(),
        label=other.name,
        color=MEX_PINK,
      )
      hist.add_plot(
        data=events[ERZ_STR].dropna(),
        xlabel="ERZ (km)",
        ylabel="Number of Events",
        title=f"ERZ Histogram for {self.name}",
        label=self.name,
        legend=True,
        alpha=1,
        color=OGS_BLUE,
        facecolor=OGS_BLUE,
        output=f"{self.filepath.name}_{other.filepath.name}_ERZ.png"
      )
    else:
      hist = OGS_P.histogram_plotter(
        data=events[ERZ_STR].dropna(),
        bins=bins,
        xlabel="ERZ (km)",
        ylabel="Number of Events",
        title=f"ERZ Histogram for {self.name}",
        output=f"{self.filepath.name}_ERZ.png"
      )
    plt.close()

  def plot_erh_histogram(self, bins: int = NUM_BINS, other = None):
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    events = self.get("EVENTS")
    if events.empty or ERH_STR not in events.columns:
      print("No ERH data available for histogram.")
      return
    if (other is not None and isinstance(other, OGSCatalog) and
        ERH_STR in other.get("EVENTS").columns):
      other_events = other.get("EVENTS")
      hist = OGS_P.histogram_plotter(
        data=other_events[ERH_STR].dropna(),
        label=other.name,
        color=MEX_PINK,
      )
      hist.add_plot(
        data=events[ERH_STR].dropna(),
        xlabel="ERH (km)",
        ylabel="Number of Events",
        title=f"ERH Histogram for {self.name}",
        label=self.name,
        legend=True,
        alpha=1,
        color=OGS_BLUE,
        facecolor=OGS_BLUE,
        output=f"{self.filepath.name}_{other.filepath.name}_ERH.png"
      )
    else:
      hist = OGS_P.histogram_plotter(
        data=events[ERH_STR].dropna(),
        bins=bins,
        xlabel="ERH (km)",
        ylabel="Number of Events",
        title=f"ERH Histogram for {self.name}",
        output=f"{self.filepath.name}_ERH.png"
      )
    plt.close()

  def plot_ert_histogram(self, bins: int = NUM_BINS, other = None):
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    events = self.get("EVENTS")
    if events.empty or ERT_STR not in events.columns:
      print("No ERT data available for histogram.")
      return
    if (other is not None and isinstance(other, OGSCatalog) and
        ERT_STR in other.get("EVENTS").columns):
      other_events = other.get("EVENTS")
      hist = OGS_P.histogram_plotter(
        data=other_events[ERT_STR].dropna(),
        label=other.name,
        color=MEX_PINK,
      )
      hist.add_plot(
        data=events[ERT_STR].dropna(),
        xlabel="ERT (s)",
        ylabel="Number of Events",
        title=f"ERT Histogram for {self.name}",
        label=self.name,
        legend=True,
        alpha=1,
        color=OGS_BLUE,
        facecolor=OGS_BLUE,
        output=f"{self.filepath.name}_{other.filepath.name}_ERT.png"
      )
    else:
      hist = OGS_P.histogram_plotter(
        data=events[ERT_STR].dropna(),
        bins=bins,
        xlabel="ERT (s)",
        ylabel="Number of Events",
        title=f"ERT Histogram for {self.name}",
        output=f"{self.filepath.name}_ERT.png"
      )
    plt.close()

  def plot_events(self, other = None):
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    events = self.get("EVENTS")
    if other is not None and isinstance(other, OGSCatalog):
      other_events = other.get("EVENTS")
      eventsMap = OGS_P.map_plotter(
        OGS_STUDY_REGION,
        x=other_events[LONGITUDE_STR],
        y=other_events[LATITUDE_STR],
        legend=True,
        marker='o',
        facecolors='none',
        edgecolors=MEX_PINK,
        label=other.name,
      )
      eventsMap.add_plot(
        x=events[LONGITUDE_STR],
        y=events[LATITUDE_STR],
        legend=True,
        marker='o',
        color="none",
        facecolors='none',
        edgecolors=OGS_BLUE,
        label=self.name,
        output=f"{self.filepath.name}_{other.filepath.name}_Events.png"
      )
    else:
      eventsMap = OGS_P.map_plotter(
        OGS_STUDY_REGION,
        x=events[LONGITUDE_STR],
        y=events[LATITUDE_STR],
        legend=True,
        marker='o',
        color="none",
        facecolors='none',
        edgecolors=OGS_BLUE,
        label=self.name,
        output=f"{self.filepath.name}_Events.png"
      )
    plt.close()

  def plot_depth_histogram(self, bins: int = NUM_BINS, other = None):
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    events = self.get("EVENTS")
    if events.empty or DEPTH_STR not in events.columns:
      print("No depth data available for histogram.")
      return
    if (other is not None and isinstance(other, OGSCatalog) and
        DEPTH_STR in other.get("EVENTS").columns):
      other_events = other.get("EVENTS")
      hist = OGS_P.histogram_plotter(
        data=other_events[DEPTH_STR],
        label=other.name,
        color=MEX_PINK,
      )
      hist.add_plot(
        data=events[DEPTH_STR],
        xlabel="Depth (km)",
        ylabel="Number of Events",
        title=f"Depth Histogram for {self.name}",
        label=self.name,
        legend=True,
        alpha=1,
        color=OGS_BLUE,
        facecolor=OGS_BLUE,
        output=f"{self.filepath.name}_{other.filepath.name}_Depth.png"
      )
    else:
      hist = OGS_P.histogram_plotter(
        data=events[DEPTH_STR],
        xlabel="Depth (km)",
        ylabel="Number of Events",
        title=f"Depth Histogram for {self.name}",
        output=f"{self.filepath.name}_Depth.png"
      )
    plt.close()

  def plot_magnitude_histogram(self, bins: int = NUM_BINS, other = None):
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    events = self.get("EVENTS")
    if events.empty or MAGNITUDE_L_STR not in events.columns:
      print("No magnitude data available for histogram.")
      return
    if (other is not None and isinstance(other, OGSCatalog) and
        MAGNITUDE_L_STR in other.get("EVENTS").columns):
      other_events = other.get("EVENTS")
      hist = OGS_P.histogram_plotter(
        data=other_events[MAGNITUDE_L_STR].dropna(),
        label=other.name,
        color=MEX_PINK,
      )
      hist.add_plot(
        data=events[MAGNITUDE_L_STR].dropna(),
        xlabel="Magnitude $M_L$", ylabel="Number of Events",
        title=f"Magnitude Histogram for {self.name}", label=self.name,
        legend=True, alpha=1, color=OGS_BLUE, facecolor=OGS_BLUE,
        output=f"{self.filepath.name}_{other.filepath.name}_Magnitude.png"
      )
    else:
      hist = OGS_P.histogram_plotter(
        data=events[MAGNITUDE_L_STR].dropna(),
        bins=bins, xlabel="Magnitude $M_L$", ylabel="Number of Events",
        title=f"Magnitude Histogram for {self.name}",
        output=f"{self.filepath.name}_Magnitude.png"
      )
    plt.close()

  def bpgma(self, other: "OGSCatalog") -> None:
    if not isinstance(other, OGSCatalog):
      raise ValueError("Can only perform bpgma on OGSCatalog")
    EVENTS_CFN_MTX = pd.DataFrame(0, index=[EVENT_STR, NONE_STR],
                                  columns=[EVENT_STR, NONE_STR], dtype=int)
    for date, _ in self.events_.items():
      BASE = self.get_(date, "events").reset_index(drop=True)
      TARGET = other.get_(date, "events").reset_index(drop=True)
      I = len(BASE)
      J = len(TARGET)
      bpgEvents = OGSBPGraphEvents(BASE, TARGET)
      baseIDs = set(range(I))
      targetIDs = set(range(J))
      for i, j in bpgEvents.E:
        a = min(i, j)
        b = max(i, j) - I
        EVENTS_CFN_MTX.at[EVENT_STR, EVENT_STR] += 1 # type: ignore
        baseIDs.remove(a)
        targetIDs.remove(b)
      for i in baseIDs:
        EVENTS_CFN_MTX.at[EVENT_STR, NONE_STR] += 1 # type: ignore
      for j in targetIDs:
        EVENTS_CFN_MTX.at[NONE_STR, EVENT_STR] += 1 # type: ignore
    PICKS_CFN_MTX = pd.DataFrame(0, index=[PWAVE, SWAVE, NONE_STR],
                                 columns=[PWAVE, SWAVE, NONE_STR], dtype=int)
    for date, _ in self.picks_.items():
      BASE = self.get_(date, "picks").reset_index(drop=True)
      TARGET = other.get_(date, "picks").reset_index(drop=True)
      I = len(BASE)
      J = len(TARGET)
      if BASE.empty or TARGET.empty:
        continue
      bpgPicks = OGSBPGraphPicks(BASE, TARGET)
      baseIDs = set(range(I))
      targetIDs = set(range(J))
      for i, j in bpgPicks.E:
        a, b = sorted((i, j))
        b -= I
        PICKS_CFN_MTX.at[BASE.at[a, PHASE_STR],
                         TARGET.at[b, PHASE_STR]] += 1 # type: ignore
        baseIDs.remove(a)
        targetIDs.remove(b)
      for i in baseIDs:
        PICKS_CFN_MTX.at[BASE.at[i, PHASE_STR], NONE_STR] += 1 # type: ignore
      for j in targetIDs:
        PICKS_CFN_MTX.at[NONE_STR, TARGET.at[j, PHASE_STR]] += 1 # type: ignore
    print(EVENTS_CFN_MTX)
    print(PICKS_CFN_MTX)

  def __add__(self, other):
    if not isinstance(other, OGSCatalog):
      raise ValueError("Can only add OGSCatalog to OGSCatalog")
    self.picks_ = {**self.picks_, **other.picks_}
    self.PICKS = pd.concat([self.PICKS, other.PICKS], ignore_index=True)
    self.events_ = {**self.events_, **other.events_}
    self.EVENTS = pd.concat([self.EVENTS, other.EVENTS], ignore_index=True)
    return self

  def __sub__(self, other):
    if not isinstance(other, OGSCatalog):
      raise ValueError("Can only subtract OGSCatalog from OGSCatalog")
    self.picks_ = {k: v for k, v in self.picks_.items()
                   if k not in other.picks_}
    self.PICKS = self.PICKS[~self.PICKS[INDEX_STR].isin(
      other.PICKS[INDEX_STR])]
    self.events_ = {k: v for k, v in self.events_.items()
                    if k not in other.events_}
    self.EVENTS = self.EVENTS[~self.EVENTS[INDEX_STR].isin(
      other.EVENTS[INDEX_STR])]
    return self

  def __div__(self, other):
    if not isinstance(other, OGSCatalog):
      raise ValueError("Can only divide OGSCatalog by OGSCatalog")
    [(PICKS_CFN_MTX, PICKS_TP, PICKS_FN, PICKS_FP),
     (EVENTS_CFN_MTX, EVENTS_TP, EVENTS_FN, EVENTS_FP)] = self.bpgma(
       other)
    self.EVENTS = self.EVENTS[self.EVENTS[INDEX_STR].isin(
      other.EVENTS[INDEX_STR])]
    return self

class OGSDataFile(OGSCatalog):
  RECORD_EXTRACTOR_LIST : list = [] # TBD in subclasses
  EVENT_EXTRACTOR_LIST : list = [] # TBD in subclasses
  GROUP_PATTERN = re.compile(r"\(\?P<(\w+)>[\[\]\w\d\{\}\-\\\?\+]+\)(\w)*")
  def __init__(self, filepath: Path, start: datetime = datetime.max,
               end: datetime = datetime.min, verbose: bool = False,
               polygon : mplPath = mplPath(OGS_POLY_REGION, closed=True),
               output : Path = THIS_FILE.parent / "data" / "OGSCatalog"):
    super().__init__(filepath, start, end, verbose, polygon, output)
    self.RECORD_EXTRACTOR : re.Pattern = re.compile(EMPTY_STR.join(
      list(flatten_list(self.RECORD_EXTRACTOR_LIST)))) # TBD in subclasses
    self.EVENT_EXTRACTOR : re.Pattern = re.compile(EMPTY_STR.join(
      list(flatten_list(self.EVENT_EXTRACTOR_LIST)))) # TBD in subclasses
    self.name = self.filepath.suffix.lstrip(PERIOD_STR).upper()

  def read(self):
    raise NotImplementedError

  def log(self):
    log = self.output / self.filepath.suffix
    # Picks
    if self.picks != {}:
      for date, df in self.picks.items():
        date = UTCDateTime(date).date
        dir_path = log / "assignments" / "-".join([
          f"{date.year}", f"{date.month:02}", f"{date.day:02}"])
        dir_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dir_path, index=False)
    # Events
    if self.events != {}:
      for date, df in self.events.items():
        dir_path = log / "events" / "-".join([
          f"{date.year}", f"{date.month:02}", f"{date.day:02}"])
        dir_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dir_path, index=False)
    print(self.EVENTS)
    print(self.PICKS)

  def debug(self, line, EXTRACTOR_LIST):
    RECORD_EXTRACTOR_DEBUG = list(reversed(list(it.accumulate(
      EXTRACTOR_LIST[:-1],
      lambda x, y: x + (y if isinstance(y, str) else
                        EMPTY_STR.join(list(flatten_list(y))))))))
    bug = self.GROUP_PATTERN.findall(EXTRACTOR_LIST[0])
    for i, extractor in enumerate(RECORD_EXTRACTOR_DEBUG):
      match_extractor = re.match(extractor, line)
      if match_extractor:
        match_group = self.GROUP_PATTERN.findall(RECORD_EXTRACTOR_DEBUG[i - 1])
        match_compare = self.GROUP_PATTERN.findall(extractor)
        bug = match_group[-1][match_group[-1][1] != match_compare[-1][1]]
        print(f"{self.filepath.suffix} {bug} : {line}")
        break
    return bug
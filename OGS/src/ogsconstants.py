import os
import re
import argparse
import numpy as np
import obspy as op
import pandas as pd
import networkx as nx
import itertools as it
from pathlib import Path
from obspy import UTCDateTime
from matplotlib.path import Path as mplPath
from obspy.geodetics import gps2dist_azimuth
from datetime import datetime, timedelta as td

THIS_FILE = Path(__file__)

EPSILON = 1e-6

MPI_RANK = 0
MPI_SIZE = 1
MPI_COMM = None

GPU_SIZE = 0
GPU_RANK = -1

THRESHOLDS: list[str] = ["{:.1f}".format(t) for t in np.linspace(0.1, 0.9, 9)]

# DateTime, TimeDelta and Format constants
DATE_STD = "YYMMDD"
DATE_FMT = "%Y-%m-%d"
TIME_FMT = "%H%M%S"
YYMMDD_FMT = "%y%m%d"
YYYYMMDD_FMT = "%Y%m%d"
DATETIME_FMT = YYMMDD_FMT + TIME_FMT
DATETIME_STR = "DATETIME"
TIMESTAMP_STR = "TIMESTAMP"
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
EVENT_DIST_OFFSET = 3  # km

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
NETCOLOR_STR = "NC"
STACOLOR_STR = "SC"
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
PLOT_COLORS = [OGS_BLUE, MEX_PINK, ALN_GREEN, LIP_ORANGE, SUN_YELLOW]

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

IDX_PICKS_STR = "index"
GROUPS_STR = "group"
TIME_STR = "time"
STATION_STR = "station"
PHASE_STR = "phase"
PROBABILITY_STR = "probability"
AMPLITUDE_STR = "amplitude"
EPICENTRAL_DISTANCE_STR = "epicentral_distance"
DEPTH_STR = "depth"
STATION_ML_STR = "station_ML"
NUMBER_P_PICKS_STR = "number_p_picks"
NUMBER_S_PICKS_STR = "number_s_picks"
NUMBER_P_AND_S_PICKS_STR = "number_p_and_s_picks"
ML_STR = "ML"
ML_MEDIAN_STR = "ML_median"
ML_UNC_STR = "ML_unc"
ML_STATIONS_STR = "ML_stations"
IDX_EVENTS_STR = "idx"
INDEX_STR = "idx"
TIME_STR = "time"
METADATA_STR = "metadata"
TYPE_STR = "type"
LONGITUDE_STR = "longitude"
LATITUDE_STR = "latitude"
ELEVATION_STR = "elevation"     # Elevation in meters
X_COORD_STR = "x(km)"           # X coordinate in kilometers
Y_COORD_STR = "y(km)"           # Y coordinate in kilometers
Z_COORD_STR = "z(km)"           # Z coordinate in kilometers
MAGNITUDE_STR = "magnitude"
MAGNITUDE_L_STR = "ML"
MAGNITUDE_D_STR = "MD"
PLACE_STR = "place"
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
NO_STR = "number_picks"
GAP_STR = "azimuthal_gap"
DMIN_STR = "DMIN"
RMS_STR = "RMS"
ERH_STR = "max_horizontal_uncertainty"
ERZ_STR = "vertical_uncertainty"
ERT_STR = "weight"
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

OGS_CLIENTS_DEFAULT = [
  OGS_CLIENT_STR,
  INGV_CLIENT_STR,
  GFZ_CLIENT_STR,
  IRIS_CLIENT_STR,
  ETH_CLIENT_STR,
  ORFEUS_CLIENT_STR,
  COLLALTO_CLIENT_STR
]

# Headers
CATEGORY_STR = "CATEGORY"
HEADER_STR = "HEADER"

HEADER_MODL = [MODEL_STR, WEIGHT_STR, THRESHOLD_STR]
HEADER_FSYS = [FILENAME_STR, MODEL_STR, WEIGHT_STR, TIME_STR, NETWORK_STR,
               STATION_STR]
HEADER_MANL = [INDEX_STR, TIME_STR, PHASE_STR, STATION_STR, GROUPS_STR, GROUPS_STR]
HEADER_PRED = HEADER_MODL + HEADER_MANL
HEADER_SNSR = [STATION_STR, LATITUDE_STR, LONGITUDE_STR, DEPTH_STR,
               TIME_STR]
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
HEADER_EVENTS = [INDEX_STR, TIME_STR, LATITUDE_STR, LONGITUDE_STR,
                 DEPTH_STR, ERH_STR, ERZ_STR, GAP_STR]
HEADER_PICKS = [INDEX_STR, TIME_STR, PHASE_STR, STATION_STR, ONSET_STR,
                POLARITY_STR, WEIGHT_STR]

def dist_prob(B: pd.Series, T: pd.Series) -> float:
  return T[PROBABILITY_STR] / B[PROBABILITY_STR]

def dist_phase(B: pd.Series, T: pd.Series) -> float:
  return int(T[PHASE_STR] == B[PHASE_STR])

def diff_time(B: pd.Series, T: pd.Series) -> float:
  return abs(T[TIME_STR] - B[TIME_STR])
def dist_time(B: pd.Series, T: pd.Series,
              offset: td = PICK_TIME_OFFSET) -> float:
  return 1. - (diff_time(B, T) / offset.total_seconds())

def diff_space(
    B: pd.Series,
    T: pd.Series,
    ndim: int = 2,
    p: float = 2.
  ) -> float:
  return float(format(np.sqrt((gps2dist_azimuth(
    B[LATITUDE_STR], B[LONGITUDE_STR],
    T[LATITUDE_STR], T[LONGITUDE_STR])[0] / 1000.) ** p +
    ((B[DEPTH_STR] - T[DEPTH_STR]) / 1000.) ** p if ndim == 3 else 0.), ".4f"))
def dist_space(B: pd.Series, T: pd.Series,
               offset: float = EVENT_DIST_OFFSET) -> float:
  return 1. - diff_space(B, T) / offset

def dist_pick(B: pd.Series, T: pd.Series,
              time_offset_sec: td = PICK_TIME_OFFSET) -> float:
  return (
   97. * dist_time(T, B, time_offset_sec) +
   2. * dist_phase(T, B) +
   1. * dist_prob(T, B)
  ) / 100.

def dist_event(T: pd.Series, P: pd.Series,
               time_offset_sec: td = EVENT_TIME_OFFSET,
               space_offset_km: float = EVENT_DIST_OFFSET) -> float:
  return (99. * dist_time(T, P, time_offset_sec) +
          1. * dist_space(T, P, space_offset_km)) / 100.

def is_date(string: str) -> datetime:
  return datetime.strptime(string, YYYYMMDD_FMT)

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

def decimeter(value, scale = 'normal') -> int:
  base = np.floor(np.log10(abs(value)))
  if scale == 'normal': return ((value // 10 ** base) + 1) * 10 ** base
  elif scale == 'log': return int(10 ** (base + 1))
  return np.ceil(value / 10) * 10

def inventory(
    stations: Path
  ) -> dict[str, tuple[float, float, float, str, str, str, str]]:
  import ogsplotter as OGS_P
  from matplotlib import pyplot as plt
  from obspy import Inventory, read_inventory
  myInventory = Inventory()
  for st in stations.glob("*.xml"):
    try:
      S = read_inventory(str(st))
    except Exception as e:
      print(f"WARNING: Unable to read {st}")
      print(e)
      continue
    myInventory.extend(S)
  cmap = plt.get_cmap("turbo")
  colors = cmap(np.linspace(0, 1, len(myInventory)))
  INVENTORY: dict[str, tuple[float, float, float, str, str, str, str]] = {
    f"{net.code}.{sta.code}.": (sta.longitude, sta.latitude, sta.elevation,
                                net.code, sta.code, colors[i], "")
    for i, net in enumerate(sorted(myInventory.networks, key=lambda x: x.code))
    for sta in net.stations
  }
  colors = cmap(np.linspace(0, 1, len(INVENTORY)))
  for i, (key, val) in enumerate(INVENTORY.items()):
    INVENTORY[key] = (*val[:6], colors[i])
  inv = pd.DataFrame.from_dict(
    INVENTORY,
    orient='index',
    columns=[LONGITUDE_STR, LATITUDE_STR, DEPTH_STR, NETWORK_STR, STATION_STR,
             NETCOLOR_STR, STACOLOR_STR])
  mystations = OGS_P.map_plotter(OGS_STUDY_REGION, legend=True,
                                 marker='^', output="OGSStations.png")
  for i, (net, sta) in enumerate(inv.groupby(NETWORK_STR)):
    mystations.add_plot(sta[LONGITUDE_STR], sta[LATITUDE_STR],
                        label=net, color=None, facecolors='none',
                        edgecolors=sta[NETCOLOR_STR], legend=True)
  mystations.savefig()
  plt.close()
  return INVENTORY

def waveforms(
    directory: Path,
    start: datetime,
    end: datetime) -> dict[str, dict[str, list[Path]]]:
  """
  Scans the given directory for waveform files within the specified date range.
  Args:
      directory (Path): Path to the waveforms directory.
      start (datetime): Start date (inclusive).
      end (datetime): End date (inclusive).
  Returns:
      dict[str, dict[str, list[Path]]]: Nested dictionary with dates as keys,
      station IDs as sub-keys, and lists of waveform file paths as values.
  """
  import ogsplotter as OGS_P
  from matplotlib import pyplot as plt
  WAVEFORMS: dict[str, dict[str, list[Path]]] = dict()
  DAYS = np.arange(start, end + ONE_DAY, ONE_DAY,
                   dtype='datetime64[D]').tolist()
  DAYS = [UTCDateTime(day).date for day in DAYS]
  for wf in directory.glob("**/*.mseed"):
    stid, dateinitid, _ = wf.stem.split(UNDERSCORE_STR + UNDERSCORE_STR)
    stid = PERIOD_STR.join(stid.split(PERIOD_STR)[:3])
    dateinitid = UTCDateTime(dateinitid).date
    if dateinitid not in WAVEFORMS: WAVEFORMS[dateinitid] = dict()
    if stid not in WAVEFORMS[dateinitid]: WAVEFORMS[dateinitid][stid] = list()
    WAVEFORMS[dateinitid][stid].append(wf)
  x, y = zip(*sorted([
    (date, len(WAVEFORMS[date].keys())) for date in DAYS if date in WAVEFORMS
  ], key=lambda x: x[0]))
  OGS_P.line_plotter(
    x, y,
    xlabel="Date",
    ylabel="Number of Stations",
    title="Availability",
    output="OGSAvailability.png",
    ylim=(0, decimeter(max(y)))
  )
  plt.close()
  return WAVEFORMS

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
  """
  Bipartite graph for pick assignment between Base and Target datasets.
  Uses NetworkX for graph representation and maximum weight matching.
  Args:
      Base (pd.DataFrame): DataFrame containing base picks.
      Target (pd.DataFrame): DataFrame containing target picks.

  Note:
    - Assumes Base and Target DataFrames have necessary columns:
      PROBABILITY_STR, TIME_STR, STATION_STR, PHASE_STR.
    - Uses optimized vectorized operations for time conversion and
      pre-filtering by station to improve performance.
  """
  def __init__(self, Base: pd.DataFrame, Target: pd.DataFrame):
    # Ensure PROBABILITY_STR column exists, defaulting to 1.0 if absent
    if PROBABILITY_STR not in Base.columns:
      Base[PROBABILITY_STR] = 1.0

    # Optimization: Vectorized UTCDateTime conversion using list comprehension
    # Faster than apply(lambda) for large datasets
    if TIME_STR in Base.columns:
      Base[TIME_STR] = [UTCDateTime(x) for x in Base[TIME_STR]]
    if TIME_STR in Target.columns:
      Target[TIME_STR] = [UTCDateTime(x) for x in Target[TIME_STR]]

    super().__init__(Base, Target)

  def makeMatch(self) -> None:
    """
    Constructs the bipartite graph and computes maximum weight matching.
    Edges are added between Base and Target picks based on time proximity
    and station matching, with weights calculated using dist_pick function.
    """
    I = len(self.Base)
    J = len(self.Target)
    self.G = nx.Graph()
    self.Target[NETWORK_STR] = self.Target[STATION_STR].str.split(".").str[0]
    self.Target[STATION_STR] = self.Target[STATION_STR].str.split(".").str[1]
    self.Base[STATION_STR] = self.Base[STATION_STR].astype(str)

    target_by_station = {
      station: group for station, group in self.Target.groupby(STATION_STR)
    }

    for idxBase, rowBase in self.Base.iterrows():
      station = rowBase[STATION_STR]
      # Only iterate over targets at the same station
      if station not in target_by_station:
        continue

      target_candidates = target_by_station[station]
      for idxTarget, rowTarget in target_candidates.iterrows():
        if diff_time(rowBase, rowTarget) <= PICK_TIME_OFFSET.total_seconds():
          self.G.add_edge(
            idxBase, int(idxTarget) + I,
            weight=dist_pick(rowBase, rowTarget)
          )
    self.E = nx.max_weight_matching(self.G, maxcardinality=False, weight='weight')

class OGSBPGraphEvents(OGSBPGraph):
  """
  Bipartite graph for event assignment between Base and Target datasets.
  Uses NetworkX for graph representation and maximum weight matching.
  Args:
      Base (pd.DataFrame): DataFrame containing base events.
      Target (pd.DataFrame): DataFrame containing target events.
  Note:
    - Assumes Base and Target DataFrames have necessary columns:
      TIME_STR, LATITUDE_STR, LONGITUDE_STR, DEPTH_STR.
    - Uses optimized vectorized operations for time conversion and
      pre-filtering by time to improve performance.
  """
  def __init__(self, Base: pd.DataFrame, Target: pd.DataFrame):
    # Optimization: Vectorized UTCDateTime conversion using list comprehension
    if "event_time" in Target.columns:
      Target[TIME_STR] = UTCDateTime(Target["event_time"])
    if TIME_STR in Base.columns:
      Base[TIME_STR] = [UTCDateTime(x) for x in Base[TIME_STR]]
    if "time" in Target.columns:
      Target[TIME_STR] = [UTCDateTime(x) for x in Target["time"]]

    super().__init__(Base, Target)

  def makeMatch(self):
    """
    Constructs the bipartite graph and computes maximum weight matching.
    Edges are added between Base and Target events based on time and spatial
    proximity, with weights calculated using dist_event function.
    """
    I = len(self.Base)
    # Optimization: Vectorized time filtering before nested loops
    # Create time bounds for efficient filtering
    base_times = self.Base[TIME_STR].values
    target_times = self.Target[TIME_STR].values

    for idxBase, rowBase in self.Base.iterrows():
      # Pre-filter targets by time window (reduces candidates significantly)
      time_mask = np.abs(target_times - rowBase[TIME_STR]) <= EVENT_TIME_OFFSET.total_seconds()
      target_candidates = self.Target[time_mask]

      for idxTarget, rowTarget in target_candidates.iterrows():
        # Only check spatial distance if time constraint is met
        if diff_space(rowBase, rowTarget) <= EVENT_DIST_OFFSET:
          self.G.add_edge(
            idxBase, int(idxTarget) + I,
            weight=dist_event(rowBase, rowTarget)
          )
    self.E = nx.max_weight_matching(self.G, maxcardinality=False, weight='weight')
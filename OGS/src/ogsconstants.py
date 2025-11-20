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

from matplotlib.cbook import flatten as flatten_list

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

OGS_CLIENTS_DEFAULT = [OGS_CLIENT_STR, INGV_CLIENT_STR, GFZ_CLIENT_STR,
                       IRIS_CLIENT_STR, ETH_CLIENT_STR, ORFEUS_CLIENT_STR,
                       COLLALTO_CLIENT_STR]

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
  def __init__(self, Base: pd.DataFrame, Target: pd.DataFrame):
    if PROBABILITY_STR not in Base.columns:
      Base[PROBABILITY_STR] = 1.0
    
    # Optimization: Vectorized UTCDateTime conversion using list comprehension
    # Faster than apply(lambda) for large datasets
    if TIME_STR in Base.columns:
      Base[TIME_STR] = [UTCDateTime(x) for x in Base[TIME_STR]]
    if "time" in Target.columns:
      Target[TIME_STR] = [UTCDateTime(x) for x in Target["time"]]
    
    super().__init__(Base, Target)

  def makeMatch(self) -> None:
    I = len(self.Base)
    J = len(self.Target)
    self.G = nx.Graph()
    self.Target[NETWORK_STR] = self.Target[STATION_STR].str.split(".").str[0]
    self.Target[STATION_STR] = self.Target[STATION_STR].str.split(".").str[1]
    self.Base[STATION_STR] = self.Base[STATION_STR].astype(str)

    # Optimization: Pre-filter by station to reduce nested loop iterations
    # Group target picks by station for O(1) lookup instead of O(n) search
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


# OGS Catalog
class OGSCatalog:
  def __init__(self, filepath: Path,
               start: datetime = datetime.max,
               end: datetime = datetime.min,
               verbose: bool = False,
               polygon : mplPath = mplPath(OGS_POLY_REGION, closed=True),
               output : Path = THIS_FILE.parent / "data" / "OGSCatalog",
               name: str = EMPTY_STR) -> None:
    assert filepath.exists(follow_symlinks=True), f"Filepath {filepath} does not exist."
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
      IDX_PICKS_STR, GROUPS_STR, TIME_STR, STATION_STR, PHASE_STR,
      PROBABILITY_STR, EPICENTRAL_DISTANCE_STR, DEPTH_STR, AMPLITUDE_STR,
      STATION_ML_STR
    ])
    self.EVENTS : pd.DataFrame = pd.DataFrame(columns=[
      IDX_EVENTS_STR, TIME_STR, LATITUDE_STR, LONGITUDE_STR, DEPTH_STR,
      GAP_STR, ERZ_STR, ERH_STR, GROUPS_STR, NO_STR,
      NUMBER_P_PICKS_STR, NUMBER_S_PICKS_STR, NUMBER_P_AND_S_PICKS_STR,
      ML_STR, ML_MEDIAN_STR, ML_UNC_STR, ML_STATIONS_STR
    ])
    self.preload()

  def load_(self, filepath : Path) -> pd.DataFrame:
    if filepath.suffix == ".csv":
      return pd.read_csv(filepath)
    else:
      try:
        return pd.read_parquet(filepath)
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
    elif key == "events":
      self.EVENTS[TIME_STR] = self.EVENTS[TIME_STR].apply(
        lambda x: UTCDateTime(x)) # type: ignore
      start = UTCDateTime(min(self.EVENTS[TIME_STR])).date
      end = UTCDateTime(max(self.EVENTS[TIME_STR])).date
      DAYS = np.arange(start, end + ONE_DAY, ONE_DAY,
                   dtype='datetime64[D]').tolist()
      DAYS = [UTCDateTime(day).date for day in DAYS]
      for day in DAYS:
        self.events[day] = self.EVENTS[self.EVENTS[TIME_STR].between(
          UTCDateTime(day), UTCDateTime(day) + ONE_DAY, inclusive='left' # type: ignore
        )]
        self.events[day][TIME_STR] = self.events[day][TIME_STR].apply(
          lambda x: UTCDateTime(x).__str__()) # type: ignore
      return self.EVENTS
    elif key == "PICKS":
      if self.PICKS.empty: self.postload()
      return self.PICKS
    elif key == "picks":
      self.PICKS[TIME_STR] = self.PICKS[TIME_STR].apply(
        lambda x: UTCDateTime(x)) # type: ignore
      start = UTCDateTime(min(self.PICKS[TIME_STR])).date
      end = UTCDateTime(max(self.PICKS[TIME_STR])).date
      DAYS = np.arange(start, end + ONE_DAY, ONE_DAY,
                   dtype='datetime64[D]').tolist()
      DAYS = [UTCDateTime(day).date for day in DAYS]
      for day in DAYS:
        self.picks[day] = self.PICKS[self.PICKS[TIME_STR].between(
          UTCDateTime(day), # type: ignore
          UTCDateTime(day) + ONE_DAY, # type: ignore
          inclusive='left'
        )]
        self.picks[day][TIME_STR] = self.picks[day][TIME_STR].apply(
          lambda x: UTCDateTime(x).__str__()) # type: ignore
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
    for filepath in self.filepath.glob("picks/*"):
      if filepath.is_file():
        date = UTCDateTime(filepath.stem).date
        if self.start.date() <= date <= self.end.date():
          self.picks_[date] = filepath

  def load(self):
    print(f"Loading picks from {self.filepath} files...")

  def plot(self,
        other = None,
        waveforms: dict[str, dict[str, list[Path]]] = dict()
      ) -> None:
    i = 0
    self.waveforms = waveforms
    if not self.get("EVENTS").empty:
      self.plot_events()
      self.plot_events(other)
      self.plot_magnitude_histogram(NUM_BINS, other)
      self.plot_depth_histogram(NUM_BINS, other)
      self.plot_ert_histogram(NUM_BINS, other)
      self.plot_erh_histogram(NUM_BINS)
      self.plot_erz_histogram(NUM_BINS)
      self.plot_cumulative_events(other)
      i += 1
    if not self.get("PICKS").empty:
      i += 1
      self.plot_cumulative_picks(other)
    if i == 2:
      if self.waveforms != {}:
        for idx, picks in self.PICKS[self.PICKS[IDX_PICKS_STR].isin(
          self.EventsFN["idx"])].groupby([IDX_PICKS_STR,]): # type: ignore
          event: pd.Series = self.EVENTS[
            self.EVENTS[IDX_EVENTS_STR] == idx
          ].iloc[0]
          event[TIME_STR] = UTCDateTime(event[TIME_STR]).datetime
          waveforms_day: dict[str, list[Path]] = self.waveforms.get(
            event[TIME_STR].date(),
            dict()
          )
          if waveforms_day != dict():
            self.plot_events_fn_waveforms(picks, event, waveforms_day)

  def plot_events_fn_waveforms(self,
        picks: pd.DataFrame,
        event: pd.Series,
        waveforms: dict[str, list[Path]],
      ) -> None:
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    myfnplot = OGS_P.event_plotter(
      picks=picks,
      event=event,
      waveforms=waveforms,
      inventory=self.stations,
      title=f"Missed (MS) Event {event[IDX_EVENTS_STR]} ({event[MAGNITUDE_L_STR] if MAGNITUDE_L_STR in event else EMPTY_STR}) | Proposed (PS) Picks {event[TIME_STR] - td(seconds=1)}",
    )
    myfppicks = self.PicksFP[
      self.PicksFP[TIME_STR].between( # type: ignore
        event[TIME_STR] - td(seconds=1),
        event[TIME_STR] + td(seconds=30)
      )
    ]
    myfnplot.add_plot(picks=myfppicks, flip=True,
      output=f"{self.filepath.name}_Event{event[IDX_EVENTS_STR]}_MSPS.png"
    )
    plt.close()

  def plot_cumulative_picks(self, other = None):
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    picks = self.get("PICKS")
    if picks.empty or TIME_STR not in picks.columns:
      print("No Date data available for histogram.")
      return
    if (other is not None and isinstance(other, OGSCatalog) and
        GROUPS_STR in other.get("PICKS").columns):
      hist = OGS_P.day_plotter(
        picks=other.get("PICKS")[GROUPS_STR],
      )
      hist.add_plot(
        picks=picks[GROUPS_STR],
        title=f"Cumulative Picks",
        output=f"{self.filepath.name}_{other.filepath.name}_CumulativePicks.png"
      )
    else:
      hist = OGS_P.day_plotter(
        picks=picks[GROUPS_STR],
        title=f"Cumulative Picks",
        output=f"{self.filepath.name}_CumulativePicks.png"
      )
    plt.close()

  def plot_cumulative_events(self, other = None):
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    events = self.get("EVENTS")
    if events.empty or TIME_STR not in events.columns:
      print("No Date data available for histogram.")
      return
    if (other is not None and isinstance(other, OGSCatalog) and
        GROUPS_STR in other.get("EVENTS").columns):
      hist = OGS_P.day_plotter(
        picks=other.get("EVENTS").sort_values(GROUPS_STR)[GROUPS_STR],
      )
      hist.add_plot(
        picks=events.sort_values(GROUPS_STR)[GROUPS_STR],
        title=f"Cumulative Event",
        output=f"{self.filepath.name}_{other.filepath.name}_CumulativeEvent.png"
      )
    else:
      hist = OGS_P.day_plotter(
        picks=events.sort_values(GROUPS_STR)[GROUPS_STR],
        title=f"Cumulative Event",
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
        title=f"ERZ Histogram",
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
        title=f"ERZ Histogram",
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
        title=f"ERH Histogram",
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
        title=f"ERH Histogram",
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
        data=other_events["probability"].dropna(),
        label=other.name,
        color=MEX_PINK,
      )
      hist.add_plot(
        data=events[ERT_STR].dropna(),
        xlabel="ERT (s)",
        ylabel="Number of Events",
        title=f"ERT Histogram",
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
        title=f"ERT Histogram",
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
        title=f"Depth Histogram",
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
        title=f"Depth Histogram",
        output=f"{self.filepath.name}_Depth.png"
      )
    plt.close()

  def plot_magnitude_histogram(self, bins: int = NUM_BINS, other = None):
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    if (other is not None and isinstance(other, OGSCatalog) and
        MAGNITUDE_L_STR in other.get("EVENTS").columns):
      other_events = other.get("EVENTS")
      hist = OGS_P.histogram_plotter(
        data=other_events[MAGNITUDE_L_STR].dropna(),
        label=other.name,
        color=MEX_PINK,
        output=f"{other.filepath.name}_Magnitude.png"
      )
      events = self.get("EVENTS")
      if events.empty or MAGNITUDE_L_STR not in events.columns:
        print("No magnitude data available for histogram.")
        return
      hist.add_plot(
        data=events[MAGNITUDE_L_STR].dropna(),
        xlabel="Magnitude $M_L$", ylabel="Number of Events",
        title=f"Magnitude comparison",
        label=self.name, legend=True, alpha=1, color=OGS_BLUE,
        facecolor=OGS_BLUE,
        output=f"{self.filepath.name}_{other.filepath.name}_Magnitude.png"
      )
    else:
      events = self.get("EVENTS")
      if events.empty or MAGNITUDE_L_STR not in events.columns:
        print("No magnitude data available for histogram.")
        return
      hist = OGS_P.histogram_plotter(
        data=events[MAGNITUDE_L_STR].dropna(),
        bins=bins, xlabel="Magnitude $M_L$", ylabel="Number of Events",
        title=f"Magnitude Histogram",
        output=f"{self.filepath.name}_Magnitude.png"
      )
    plt.close()

  def bgmaEvents(self, other: "OGSCatalog") -> None:
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    if not isinstance(other, OGSCatalog):
      raise ValueError("Can only perform bgmaEvents on OGSCatalog")
    EVENTS_CFN_MTX = pd.DataFrame(0, index=[EVENT_STR, NONE_STR],
                                  columns=[EVENT_STR, NONE_STR], dtype=int)
    self.EventsTP = list()
    self.EventsFN = list()
    self.EventsFP = list()
    columns = [INDEX_STR, TIME_STR, LATITUDE_STR, LONGITUDE_STR,
               DEPTH_STR, ERH_STR, ERZ_STR, GAP_STR, MAGNITUDE_L_STR]
    for date, _ in self.events_.items():
      BASE = self.get_(date, "events").reset_index(drop=True)
      TARGET = other.get_(date, "events").reset_index(drop=True)
      I = len(BASE)
      bpgEvents = OGSBPGraphEvents(BASE, TARGET)
      baseIDs = set(range(I))
      targetIDs = set(range(len(TARGET)))
      for i, j in bpgEvents.E:
        a, b = sorted((i, j))
        b -= I
        EVENTS_CFN_MTX.at[EVENT_STR, EVENT_STR] += 1 # type: ignore
        baseIDs.remove(a)
        targetIDs.remove(b)

        self.EventsTP.append([
          (BASE.at[a, col] if col in BASE.columns else None,
           TARGET.at[b, col] if col in TARGET.columns else None)
           for col in columns
        ])
      for i in baseIDs:
        EVENTS_CFN_MTX.at[EVENT_STR, NONE_STR] += 1 # type: ignore
        self.EventsFN.append([
          BASE.at[i, col] if col in BASE.columns else None for col in columns
        ])
      if not TARGET.empty:
        fp_target = TARGET[TARGET[[LONGITUDE_STR, LATITUDE_STR]].apply(
          lambda x: self.polygon.contains_point(
            (x[LONGITUDE_STR], x[LATITUDE_STR])), axis=1)]
        for j in targetIDs:
          if j not in fp_target.index: continue
          EVENTS_CFN_MTX.at[NONE_STR, EVENT_STR] += 1 # type: ignore
          self.EventsFP.append([
            fp_target.at[j, col] if col in fp_target.columns else None
            for col in columns
          ])
    recall = \
      EVENTS_CFN_MTX.at[EVENT_STR, EVENT_STR] / ( # type: ignore
        EVENTS_CFN_MTX.at[EVENT_STR, EVENT_STR] +
        EVENTS_CFN_MTX.at[EVENT_STR, NONE_STR]
      )
    print("Recall: ", recall)
    print("MH: ", EVENTS_CFN_MTX.at[EVENT_STR, EVENT_STR],
          "PS: ", EVENTS_CFN_MTX.at[NONE_STR, EVENT_STR],
          "MS: ", EVENTS_CFN_MTX.at[EVENT_STR, NONE_STR])
    print(EVENTS_CFN_MTX)
    self.EventsTP = pd.DataFrame(self.EventsTP, columns=columns)
    self.EventsFN = pd.DataFrame(self.EventsFN, columns=columns).sort_values(
      by=TIME_STR)
    self.EventsFP = pd.DataFrame(self.EventsFP, columns=columns).sort_values(
      by=TIME_STR)
    filepath = f"{self.filepath.name}_{other.filepath.name}_EventsMH.csv"
    self.EventsTP.to_csv(filepath, index=False)
    print(f"{filepath} written.")
    filepath = f"{self.filepath.name}_{other.filepath.name}_EventsMS.csv"
    self.EventsFN.to_csv(filepath, index=False)
    print(f"{filepath} written.")
    filepath = f"{self.filepath.name}_{other.filepath.name}_EventsPS.csv"
    self.EventsFP.to_csv(filepath, index=False)
    print(f"{filepath} written.")
    filepath = f"{self.filepath.name}_" + \
               f"{other.filepath.name}_EventsConfMtx.png"
    OGS_P.ConfMtx_plotter(
      EVENTS_CFN_MTX.values,
      title="Recall: {:.4f}".format(recall),
      label=EVENTS_CFN_MTX.columns.tolist(),
      output=filepath
    )
    plt.close()
    # Time Difference Histogram
    OGS_P.histogram_plotter(
      self.EventsTP[TIME_STR].apply(lambda x: x[1] - x[0]),
      xlabel="Time Difference (s)",
      title="Event Time Difference",
      xlim=(-EVENT_TIME_OFFSET.total_seconds(),
            EVENT_TIME_OFFSET.total_seconds()),
      output=f"{self.filepath.name}_{other.filepath.name}_EventsTimeDiff.png",
      legend=True)
    plt.close()
    # Missed (MS) Map
    myplot = OGS_P.map_plotter(
      domain=OGS_STUDY_REGION,
      x=self.EventsTP[LONGITUDE_STR].apply(lambda x: x[0]),
      y=self.EventsTP[LATITUDE_STR].apply(lambda x: x[0]),
      facecolors="none", edgecolors=OGS_BLUE, legend=True,
      label=self.name)
    myplot.add_plot(
      self.EventsTP[LONGITUDE_STR].apply(lambda x: x[1]),
      self.EventsTP[LATITUDE_STR].apply(lambda x: x[1]), color=None,
        label="Missed (MS) [SBC]", legend=True, facecolors="none",
        edgecolors=MEX_PINK,
        output=f"{self.filepath.name}_{other.filepath.name}_MS.png")
    plt.close()
    # Missed (MS) and Proposed (PS) Map
    myplot = OGS_P.map_plotter(
      domain=OGS_STUDY_REGION,
      x=self.EventsFN[LONGITUDE_STR], y=self.EventsFN[LATITUDE_STR],
      label="Missed (MS) [OGS]", legend=True,)
    myplot.add_plot(
      self.EventsFP[LONGITUDE_STR], self.EventsFP[LATITUDE_STR], color=None,
        label="Proposed (PS) [SBC]", legend=True, facecolors="none",
        edgecolors=MEX_PINK,
        output=f"{self.filepath.name}_{other.filepath.name}_False.png")
    plt.close()
    # Depth Difference Histogram
    OGS_P.histogram_plotter(
      self.EventsTP[DEPTH_STR].apply(lambda x: x[1] - x[0]),
      xlabel="Depth Difference (km) [OGS - SBC]",
      title="Event Depth Difference",
      xlim=(-EVENT_DIST_OFFSET, EVENT_DIST_OFFSET),
      output=f"{self.filepath.name}_{other.filepath.name}_DepthDiff.png",
      legend=True)
    plt.close()
    # Event Location Scatter Plot
    OGS_P.histogram_plotter(
      OGS_P.v_lat_long_to_distance(
        self.EventsTP[LONGITUDE_STR].apply(lambda x: x[0]),
        self.EventsTP[LATITUDE_STR].apply(lambda x: x[0]),
        self.EventsTP[DEPTH_STR].apply(lambda x: 0),
        self.EventsTP[LONGITUDE_STR].apply(lambda x: x[1]),
        self.EventsTP[LATITUDE_STR].apply(lambda x: x[1]),
        self.EventsTP[DEPTH_STR].apply(lambda x: x[1]),
        dim=2
      ),
      xlim=(0, EVENT_DIST_OFFSET),
      xlabel="Epicentral Distance Difference (km)",
      title="Event Epicentral Distance Difference",
      output=f"{self.filepath.name}_{other.filepath.name}_EpiDistDiff.png",
      legend=True)
    plt.close()
    if MAGNITUDE_L_STR in self.EventsTP.columns:
      # Magnitude Scatter Plot
      mymags = OGS_P.scatter_plotter(
        self.EventsTP[MAGNITUDE_L_STR].apply(lambda x: x[1]),
        self.EventsTP[MAGNITUDE_L_STR].apply(lambda x: x[0]),
        xlabel="SBC Magnitude ($M_L$)",
        ylabel="OGS Magnitude ($M_L$)",
        title="Magnitude Prediction",
        color=OGS_BLUE,
        legend=True
      )
      x_min = min(
        self.EventsTP[MAGNITUDE_L_STR].apply(lambda x: x[0]).min(),
        self.EventsTP[MAGNITUDE_L_STR].apply(lambda x: x[1]).min()
      )
      x_max = max(
        self.EventsTP[MAGNITUDE_L_STR].apply(lambda x: x[0]).max(),
        self.EventsTP[MAGNITUDE_L_STR].apply(lambda x: x[1]).max()
      )
      mymags.ax.plot([x_min, x_max], [x_min, x_max], color=MEX_PINK,
                     linestyle='--')
      mymags.ax.set_aspect('equal', adjustable='box')
      mymags.ax.grid(True)
      mymags.savefig(
        f"{self.filepath.name}_{other.filepath.name}_MagDist.png")
      plt.close()
      if self.filepath.name in (TXT_EXT, ".all") and \
         other.filepath.name in ("OGSLocalMagnitude"):
        # Magnitude Difference Histogram
        OGS_P.histogram_plotter(
          self.EventsTP[MAGNITUDE_L_STR].apply(lambda x: x[1] - x[0]),
          xlabel="Magnitude Difference ($M_L$) [OGS - SBC]",
        title="Event Magnitude Difference",
        xlim=(-1, 1),
        bins=21,
        output=f"{self.filepath.name}_{other.filepath.name}_MagDiff.png",
        legend=True)
        plt.close()
        # Event Magnitude Histogram
        mymags = OGS_P.histogram_plotter(
          self.EventsFP[MAGNITUDE_L_STR].dropna(),
          xlabel="Magnitude ($M_L$)",
          title="Event Magnitude",
          color=MEX_PINK,
          yscale='log',
          label="Proposed (PS) [SBC]",
        )
        mymags.add_plot(
          self.EventsFN[MAGNITUDE_L_STR],
          label="Missed (MS) [OGS]",
          color=OGS_BLUE,
          legend=True,
          output=f"{self.filepath.name}_{other.filepath.name}_MSPSMagDist.png",
        )
        plt.close()
      else:
        events = self.EventsFP[MAGNITUDE_L_STR].dropna()
        if not events.empty:
          # Event Magnitude Histogram
          OGS_P.histogram_plotter(
            events,
            xlabel="Magnitude ($M_L$)",
            title="Event Magnitude",
            color=MEX_PINK,
            label="Proposed (PS) [SBC]",
            output=f"{other.filepath.name}_PSMagDist.png",
          )
          plt.close()
    self.EVENTS = self.get("EVENTS")
    other.EVENTS = other.get("EVENTS")

  def bgmaPicks(self, other: "OGSCatalog",) -> None:
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    if not isinstance(other, OGSCatalog):
      raise ValueError("Can only perform bgmaPicks on OGSCatalog")
    PICKS_CFN_MTX = pd.DataFrame(0, index=[PWAVE, SWAVE, NONE_STR],
                                 columns=[PWAVE, SWAVE, NONE_STR], dtype=int)
    INVENTORY = [key.split(".")[1] for key in self.stations.keys()]
    self.PicksTP = list()
    self.PicksFN = list()
    self.PicksFP = list()
    columns = [IDX_PICKS_STR, TIME_STR, PHASE_STR, STATION_STR,
               PROBABILITY_STR]
    for date, _ in self.picks_.items():
      BASE = self.get_(date, "picks")
      BASE[NETWORK_STR] = BASE[STATION_STR].str.split(".").str[0]
      BASE[STATION_STR] = BASE[STATION_STR].str.split(".").str[1]
      BASE = BASE[BASE[STATION_STR].isin(INVENTORY)].reset_index(drop=True)
      TARGET = other.get_(date, "picks").reset_index(drop=True)
      I = len(BASE)
      bpgPicks = OGSBPGraphPicks(BASE, TARGET)
      baseIDs = set(range(I))
      targetIDs = set(range(len(TARGET)))
      for i, j in bpgPicks.E:
        a, b = sorted((i, j))
        b -= I
        PICKS_CFN_MTX.at[BASE.at[a, PHASE_STR],
                         TARGET.at[b, PHASE_STR]] += 1 # type: ignore
        if BASE.at[a, PHASE_STR] == TARGET.at[b, PHASE_STR]:
          self.PicksTP.append([
            (BASE.at[a, IDX_PICKS_STR], TARGET.at[b, IDX_PICKS_STR]),
            (str(BASE.at[a, TIME_STR]), str(TARGET.at[b, TIME_STR])),
            (BASE.at[a, PHASE_STR]),
            (TARGET.at[b, STATION_STR]),
            (BASE.at[a, PROBABILITY_STR], TARGET.at[b, PROBABILITY_STR])
          ])
        baseIDs.remove(a)
        targetIDs.remove(b)
      for i in baseIDs:
        PICKS_CFN_MTX.at[BASE.at[i, PHASE_STR], NONE_STR] += 1 # type: ignore
        self.PicksFN.append([BASE.at[i, col] for col in columns])
      for j in targetIDs:
        PICKS_CFN_MTX.at[NONE_STR,
                         TARGET.at[j, PHASE_STR]] += 1 # type: ignore
        self.PicksFP.append([
          TARGET.at[j, IDX_PICKS_STR],
          TARGET.at[j, TIME_STR],
          TARGET.at[j, PHASE_STR],
          TARGET.at[j, STATION_STR],
          TARGET.at[j, PROBABILITY_STR]
        ])
    recall = \
      (PICKS_CFN_MTX.at[PWAVE, PWAVE] + # type: ignore
      PICKS_CFN_MTX.at[SWAVE, SWAVE]) / (
        PICKS_CFN_MTX.at[PWAVE, PWAVE] + # type: ignore
        PICKS_CFN_MTX.at[SWAVE, SWAVE] +
        PICKS_CFN_MTX.at[PWAVE, SWAVE] + PICKS_CFN_MTX.at[SWAVE, PWAVE] +
        PICKS_CFN_MTX.at[PWAVE, NONE_STR] + PICKS_CFN_MTX.at[SWAVE, NONE_STR]
      )
    print("Recall: ", recall)
    p_recall = PICKS_CFN_MTX.at[PWAVE, PWAVE] / ( # type: ignore
      PICKS_CFN_MTX.at[PWAVE, PWAVE] + # type: ignore
      PICKS_CFN_MTX.at[PWAVE, SWAVE] +
      PICKS_CFN_MTX.at[PWAVE, NONE_STR]
    )
    print("Recall P-wave: ", p_recall)
    s_recall = PICKS_CFN_MTX.at[SWAVE, SWAVE] / ( # type: ignore
      PICKS_CFN_MTX.at[SWAVE, SWAVE] + # type: ignore
      PICKS_CFN_MTX.at[SWAVE, PWAVE] +
      PICKS_CFN_MTX.at[SWAVE, NONE_STR]
    )
    print("Recall S-wave: ", s_recall)
    print(PICKS_CFN_MTX)
    print("MH: ", PICKS_CFN_MTX.at[PWAVE, PWAVE] + # type: ignore
                  PICKS_CFN_MTX.at[SWAVE, SWAVE],
          "PS: ", PICKS_CFN_MTX.at[NONE_STR, PWAVE] +
          PICKS_CFN_MTX.at[NONE_STR, SWAVE], # type: ignore
          "MS: ", PICKS_CFN_MTX.at[PWAVE, NONE_STR] +
                  PICKS_CFN_MTX.at[SWAVE, NONE_STR]) # type: ignore
    self.PicksTP = pd.DataFrame(self.PicksTP, columns=columns).sort_values(
      by=TIME_STR
    )
    self.PicksFN = pd.DataFrame(self.PicksFN, columns=columns).sort_values(
      by=TIME_STR
    ).sort_values(by=TIME_STR)
    self.PicksFP = pd.DataFrame(self.PicksFP, columns=columns).sort_values(
      by=TIME_STR
    ).sort_values(by=TIME_STR)
    filepath = f"{self.filepath.name}_{other.filepath.name}_PicksMH.csv"
    self.PicksTP.to_csv(filepath, index=False)
    print(f"{filepath} written.")
    filepath = f"{self.filepath.name}_{other.filepath.name}_PicksMS.csv"
    self.PicksFN.to_csv(filepath, index=False)
    print(f"{filepath} written.")
    filepath = f"{self.filepath.name}_{other.filepath.name}_PicksPS.csv"
    self.PicksFP.to_csv(filepath, index=False)
    print(f"{filepath} written.")
    filepath = f"{self.filepath.name}_" + \
               f"{other.filepath.name}_PicksConfMtx.png"
    OGS_P.ConfMtx_plotter(
      PICKS_CFN_MTX.values,
      title="Recall: {:.4f}, Recall P: {:.4f}, Recall S: {:.4f}".format(
        recall, p_recall, s_recall
      ),
      label=PICKS_CFN_MTX.columns.tolist(),
      output=filepath
    )
    plt.close()
    # Time Difference Histogram
    pickdiff = OGS_P.histogram_plotter(
      self.PicksTP[TIME_STR].apply(
        lambda x: UTCDateTime(x[1]) - UTCDateTime(x[0]) # type: ignore
      ),
      xlabel="Time Difference (s)",
      title=f"Pick Time Difference between {self.name} and {other.name}",
      legend=True,
      label="Matched (MH)",
      color=MEX_PINK,
      xlim=(-PICK_TIME_OFFSET.total_seconds(),
            PICK_TIME_OFFSET.total_seconds()))
    data = self.PicksTP.loc[
      self.PicksTP[PHASE_STR] == PWAVE,
      TIME_STR
    ].apply(
      lambda x: UTCDateTime(x[1]) - UTCDateTime(x[0]) # type: ignore
    )
    pickdiff.add_plot(
      data,
      alpha=1,
      step=True,
      color=OGS_BLUE,
      label=f"P Picks: $\mu$ = {data.mean():.3E}, $\sigma$ = {data.std():.3E}",
    )
    data = self.PicksTP.loc[
      self.PicksTP[PHASE_STR] == SWAVE,
      TIME_STR
    ].apply(
      lambda x: UTCDateTime(x[1]) - UTCDateTime(x[0]) # type: ignore
    )
    pickdiff.add_plot(
      data,
      alpha=1,
      color=ALN_GREEN,
      step=True,
      label=f"S Picks: $\mu$ = {data.mean():.3E}, $\sigma$ = {data.std():.3E}",
      legend=True,
      output=f"{self.filepath.name}_{other.filepath.name}_PicksTimeDiff.png",
    )
    plt.close()
    # Confidence Histogram
    myconf = OGS_P.histogram_plotter(
      self.PicksTP[PROBABILITY_STR].apply(lambda x: x[1]),
      xlabel="Pick Confidence",
      title="Pick Confidence Distribution",
      label="Matched (MH)",
      xlim=(0, 1),
    )
    myconf.add_plot(
      self.PicksTP.loc[
        self.PicksTP[PHASE_STR] == PWAVE,
        PROBABILITY_STR
      ].apply(lambda x: x[1]),
      alpha=1,
      step=True,
      color=MEX_PINK,
      label="MH P Picks",
    )
    myconf.add_plot(
      self.PicksTP.loc[
        self.PicksTP[PHASE_STR] == SWAVE,
        PROBABILITY_STR
      ].apply(lambda x: x[1]),
      alpha=1,
      color=ALN_GREEN,
      step=True,
      label="MH S Picks",
    )
    myconf.add_plot(
      self.PicksFP[PROBABILITY_STR],
      alpha=1,
      color=LIP_ORANGE,
      step=True,
      label="Proposed (PS)",
      legend=True,
      yscale='log',
      output=f"{other.filepath.name}_PicksConfDist.png",
    )
    plt.close()
    self.PICKS = self.get("PICKS")
    other.PICKS = other.get("PICKS")

  def bpgma(self,
        other: "OGSCatalog",
        stations: dict[str, tuple[float, float, float, str, str, str, str]]
      ) -> None:
    if not isinstance(other, OGSCatalog):
      raise ValueError("Can only perform bpgma on OGSCatalog")
    self.stations = stations
    if self.events_ != {}:
      if other.events_ == {} and other.get("EVENTS").empty:
        print(f"{other.name} catalog has no events to compare.")
      else:
        self.bgmaEvents(other)
    if self.picks_ != {}:
      if other.picks_ == {} and other.get("PICKS").empty:
        print(f"{other.name} catalog has no picks to compare.")
      else:
        self.bgmaPicks(other)

  def plotFNWaveforms(self, other: "OGSCatalog", output: Path) -> None:
    pass

  def __iadd__(self, other):
    if not isinstance(other, OGSCatalog):
      raise ValueError("Can only add OGSCatalog to OGSCatalog")
    self.picks_ = {**self.picks_, **other.picks_}
    if not self.PICKS.empty and not other.PICKS.empty:
      self.PICKS = pd.concat([self.PICKS, other.PICKS], ignore_index=True)
    elif self.PICKS.empty:
      self.PICKS = other.PICKS.copy()
    self.events_ = {**self.events_, **other.events_}
    if not self.EVENTS.empty and not other.EVENTS.empty:
      self.EVENTS = pd.concat([self.EVENTS, other.EVENTS], ignore_index=True)
    elif self.EVENTS.empty:
      self.EVENTS = other.EVENTS.copy()
    return self

  def __isub__(self, other):
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
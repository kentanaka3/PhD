import re
import numpy as np
import pandas as pd
import itertools as it
from pathlib import Path
from datetime import datetime, timedelta as td
from matplotlib.path import Path as mplPath

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
PICK_OFFSET = td(seconds=.5) # TODO: Change to .5 sec
PICK_OFFSET_TRAIN = td(seconds=60)
H71_OFFSET = {
    0: 0.01,
    1: 0.04,
    2: 0.2,
    3: 1,
    4: 5,
    5: 25
}
ASSOCIATE_TIME_OFFSET = td(seconds=1.5)
ASSOCIATE_DIST_OFFSET = 8  # km

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
MATCH_CNFG = {
    CLSSFD_STR: {
        CATEGORY_STR: [PWAVE, SWAVE, NONE_STR],
        METHOD_STR: CLSSFD_STR,
        TIME_DSPLCMT_STR: PICK_OFFSET,
        DISTANCE_STR: None,
        HEADER_STR: HEADER_MANL,
        DIRECTORY_STR: CLF_STR
    },
    DETECT_STR: {
        CATEGORY_STR: [PWAVE, SWAVE, NONE_STR],
        METHOD_STR: DETECT_STR,
        TIME_DSPLCMT_STR: ASSOCIATE_TIME_OFFSET,
        DISTANCE_STR: None,
        HEADER_STR: HEADER_MANL,
        DIRECTORY_STR: AST_STR
    },
    SOURCE_STR: {
        CATEGORY_STR: [EVENT_STR, NONE_STR],
        METHOD_STR: SOURCE_STR,
        TIME_DSPLCMT_STR: ASSOCIATE_TIME_OFFSET,
        DISTANCE_STR: 1.5,
    },
}

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

# OGS Catalog
class OGSDataFile:
  RECORD_EXTRACTOR_LIST : list = [] # TBD in subclasses
  EVENT_EXTRACTOR_LIST : list = [] # TBD in subclasses
  GROUP_PATTERN = re.compile(r"\(\?P<(\w+)>[\[\]\w\d\{\}\-\\\?\+]+\)(\w)*")
  def __init__(self, filepath: Path, start: datetime = datetime.max,
               end: datetime = datetime.min, verbose: bool = False,
               polygon : mplPath = mplPath(OGS_POLY_REGION, closed=True),
               name : Path = THIS_FILE.parent / "data" / "OGSCatalog"):
    self.filepath = filepath
    self.start = start
    self.end = end
    self.polygon : mplPath = polygon
    self.verbose = verbose
    self.name = name
    self.picks = pd.DataFrame(columns=[
      INDEX_STR, TIMESTAMP_STR, PHASE_STR, STATION_STR,
      ERT_STR, NOTES_STR, NETWORK_STR, GROUPS_STR])
    self.events = pd.DataFrame(columns=[
      INDEX_STR, TIMESTAMP_STR, LATITUDE_STR,
      LONGITUDE_STR, DEPTH_STR, NO_STR,
      GAP_STR, DMIN_STR, RMS_STR,
      ERH_STR, ERZ_STR, QM_STR, MAGNITUDE_L_STR,
      MAGNITUDE_D_STR, NOTES_STR,])
    self.RECORD_EXTRACTOR : re.Pattern = re.compile(EMPTY_STR.join(
      list(flatten_list(self.RECORD_EXTRACTOR_LIST)))) # TBD in subclasses
    self.EVENT_EXTRACTOR : re.Pattern = re.compile(EMPTY_STR.join(
      list(flatten_list(self.EVENT_EXTRACTOR_LIST)))) # TBD in subclasses
    print(f"Processing file: {self.filepath}")

  def read(self):
    raise NotImplementedError
  DIR_FMT = {
    "year": "{:04}",
    "month": "{:02}",
    "day": "{:02}",
  }
  def log(self):
    log = self.name / self.filepath.suffix
    # Picks
    if not self.picks.empty:
      for date, df in self.picks.groupby(
          self.picks[TIMESTAMP_STR].dt.date):
        dir_path = log / "assignments" / f"{date.year}" / f"{date.month:02}" /\
                   f"{date.day:02}.csv"
        dir_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dir_path, index=False)
    print(self.picks)
    # Events
    if not self.events.empty:
      for date, df in self.events.groupby(
          self.events[TIMESTAMP_STR].dt.date):
        dir_path = log / "events" / f"{date.year}" / f"{date.month:02}" /\
                   f"{date.day:02}.csv"
        dir_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dir_path, index=False)
    print(self.events)

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
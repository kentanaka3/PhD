import numpy as np
from pathlib import Path
import seisbench.models as sbm
from datetime import timedelta as td


SAMPLING_RATE = 100

EPSILON = 1e-6

DENOISER = None

MPI_RANK = 0
MPI_SIZE = 1
MPI_COMM = None

GPU_SIZE = 0
GPU_RANK = -1

THRESHOLDS: list[str] = ["{:.1f}".format(t) for t in np.linspace(0.1, 0.9, 9)]
DATES = None

NORM = "std"  # "peak" or "std"

# DateTime, TimeDelta and Format constants
DATE_FMT = "%y%m%d"
TIME_FMT = "%H%M%S"
DATETIME_FMT = DATE_FMT + TIME_FMT
ONE_DAY = td(days=1)
PICK_OFFSET = td(seconds=0.5)
PICK_OFFSET_TRAIN = td(seconds=60)
H71_OFFSET = {
    0: td(seconds=0.01),
    1: td(seconds=0.04),
    2: td(seconds=0.2),
    3: td(seconds=1),
    4: td(seconds=5),
    5: td(seconds=25)
}
ASSOCIATE_TIME_OFFSET = td(seconds=1.5)
ASSOCIATE_DIST_OFFSET = .5  # km

# Strings
EMPTY_STR = ''
ALL_WILDCHAR_STR = '*'
ONE_MORECHAR_STR = '+'
PERIOD_STR = '.'
UNDERSCORE_STR = '_'
DASH_STR = '-'
SPACE_STR = ' '
COMMA_STR = ','
ZERO_STR = "0"
NONE_STR = "None"
CLF_STR = "classified"
AST_STR = "associated"
FILE_STR = "file"
TEMPORAL_STR = "tmp"
STATUS_STR = "status"
SECONDS_STR = "seconds"
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

PHASES_DICT = {
    "trace_p_arrival_sample": PWAVE,
    "trace_pP_arrival_sample": PWAVE,
    "trace_P_arrival_sample": PWAVE,
    "trace_P1_arrival_sample": PWAVE,
    "trace_Pg_arrival_sample": PWAVE,
    "trace_Pn_arrival_sample": PWAVE,
    "trace_PmP_arrival_sample": PWAVE,
    "trace_pwP_arrival_sample": PWAVE,
    "trace_pwPm_arrival_sample": PWAVE,
    "trace_s_arrival_sample": SWAVE,
    "trace_S_arrival_sample": SWAVE,
    "trace_S1_arrival_sample": SWAVE,
    "trace_Sg_arrival_sample": SWAVE,
    "trace_SmS_arrival_sample": SWAVE,
    "trace_Sn_arrival_sample": SWAVE,
}

# Thresholds
PWAVE_THRESHOLD = SWAVE_THRESHOLD = 0.2

SEED_ID_FMT = "{NETWORK}.{STATION}..{CHANNEL}"

CFN_MTX_STR = "CM"
CMTV_PICKS_STR = "CP"
CLSTR_PLOT_STR = "CT"
TIME_DSPLCMT_STR = "TD"

MEX_PINK = "#E4007C"

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
XML_EXT = PERIOD_STR + XML_STR

PRC_FMT = SEED_ID_FMT + ".{BEGDT}.{EXT}"

# Models
EQTRANSFORMER_STR = "EQTransformer"
PHASENET_STR = "PhaseNet"

# Various pre-trained weights for each model (Add if new are available)
MODEL_WEIGHTS_DICT = {
    EQTRANSFORMER_STR: sbm.EQTransformer(phases=PWAVE + SWAVE,
                                         sampling_rate=SAMPLING_RATE,
                                         norm=NORM),
    PHASENET_STR: sbm.PhaseNet(phases=PWAVE + SWAVE,
                               sampling_rate=SAMPLING_RATE,
                               norm=NORM)
}


class DIRSTRUCT():
  def __init__(self, root_dir: Path, categories: list[str] = None,
               file: any = None):
    self.root_dir = root_dir
    self.categories = categories
    self.path = Path(root_dir, *categories) if categories is not None \
        else Path(root_dir)
    self.path.mkdir(parents=True, exist_ok=True)
    if type(file) == str:
      self.file = Path(self.path, file)
    # elif type(file) == list(str):
    #  self.file = [Path(self.path, f) for f in file]
    else:
      self.file = None

  def __str__(self):
    return str(self.path) if self.file is None else str(self.file)

  def update(self, file: str): self.file = Path(self.path, file)

  def update(self, categories: list[str]):
    self.categories = categories
    self.path = Path(self.root_dir, *categories)
    self.path.mkdir(parents=True, exist_ok=True)
    self.file = None

  def update(self, root_dir: Path):
    self.root_dir = root_dir
    self.path = Path(root_dir, *self.categories)
    self.path.mkdir(parents=True, exist_ok=True)
    self.file = None


# Colors
COLORS = {
    PWAVE: "C0",
    SWAVE: "C1",
    "Detection": "C2"
}
COLOR_ENCODING = {
    TP_STR: {
        PWAVE: "red",
        SWAVE: "blue"
    },
    FP_STR: {
        PWAVE: "orange",
        SWAVE: "green"
    },
    FN_STR: {
        PWAVE: "pink",
        SWAVE: "purple"
    }
}

OGS_PROJECTION = "+proj=sterea +lon_0={lon} +lat_0={lat} +units=km"
OGS_MAX_MAGNITUDE = 3.5

# Data components
ID_STR = "id"
TIMESTAMP_STR = "timestamp"
METADATA_STR = "metadata"
PROBABILITY_STR = "prob"
TYPE_STR = "type"
LONGITUDE_STR = "longitude"
LATITUDE_STR = "latitude"
LOCAL_DEPTH_STR = "local_depth"
ELEVATION_STR = "elevation"     # Elevation in meters
X_COORD_STR = "x(km)"           # X coordinate in kilometers
Y_COORD_STR = "y(km)"           # Y coordinate in kilometers
Z_COORD_STR = "z(km)"           # Z coordinate in kilometers
MAGNITUDE_STR = "magnitude"
VELOCITY_STR = "vel"
METHOD_STR = "method"
DIMENSIONS_STR = "dims"
GAUSS_MIX_MODEL_STR = "GMM"
BAYES_GAUSS_MIX_MODEL_STR = "B" + GAUSS_MIX_MODEL_STR

ARGUMENTS_STR = "arguments"
WAVEFORMS_STR = "waveforms"
DATASETS_STR = "datasets"
MODELS_STR = "models"

PHASE_STR = "PHASE"
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
P_WEIGHT_STR = "P_WEIGHT"
S_TIME_STR = "S_TIME"
S_TYPE_STR = "S_TYPE"
S_WEIGHT_STR = "S_WEIGHT"
ORIGIN_STR = "ORIGIN"
NO_STR = "NO"
GAP_STR = "GAP"
DMIN_STR = "DMIN"
RMS_STR = "RMS"
ERH_STR = "ERH"
ERZ_STR = "ERZ"
QM_STR = "QM"
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
ED_CLIENT_STR = "http://scp-srv.core03.ogs.it:8080"

# Headers
CATEGORY_STR = "CATEGORY"
HEADER_STR = "HEADER"

HEADER_MODL = [MODEL_STR, WEIGHT_STR, THRESHOLD_STR]
HEADER_FSYS = [FILENAME_STR, MODEL_STR, WEIGHT_STR, TIMESTAMP_STR, NETWORK_STR,
               STATION_STR]
HEADER_MANL = [ID_STR, TIMESTAMP_STR, PROBABILITY_STR, PHASE_STR, NETWORK_STR,
               STATION_STR]
HEADER_PRED = HEADER_MODL + HEADER_MANL
HEADER_SRC = [ID_STR, TIMESTAMP_STR, LATITUDE_STR, LONGITUDE_STR,
              LOCAL_DEPTH_STR, MAGNITUDE_STR, NO_STR, GAP_STR, DMIN_STR,
              RMS_STR, ERH_STR, ERZ_STR, QM_STR, NOTES_STR]
HEADER_ASCT = HEADER_MODL + HEADER_SRC
HEADER_SNSR = [STATION_STR, LATITUDE_STR, LONGITUDE_STR, LOCAL_DEPTH_STR,
               TIMESTAMP_STR]
HEADER_STAT = [MODEL_STR, WEIGHT_STR, STAT_STR] + THRESHOLDS
SORT_HIERARCHY_PRED = [MODEL_STR, WEIGHT_STR, ID_STR, TIMESTAMP_STR]
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
        HEADER_STR: HEADER_SRC
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
OGS_STUDY_REGION = [44.5, 47, 10, 14.5]  # [10.0, 14.5, 44.5, 47.0]
OGS_ITALY_STR = "Italy"
OGS_STUDY_STR = "Study"
DESCRIPTION_STR = "Description"

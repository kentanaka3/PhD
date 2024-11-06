import re
from collections import deque
import seisbench.models as sbm
from datetime import timedelta as td

STREAM_STACK = deque()

SAMPLING_RATE = 100

EPSILON = 1e-6

DENOISER = None

MPI_RANK = 0
MPI_SIZE = 1
MPI_COMM = None

GPU_SIZE = 0
GPU_RANK = -1

NORM = "peak" # "peak" or "std"

# DateTime, TimeDelta and Format constants
DATE_FMT = "%y%m%d"
DATETIME_FMT = DATE_FMT + "%H%M%S"
ONE_DAY = td(days=1)
PICK_OFFSET = td(seconds=0.5)
H71_OFFSET = {
  0 : td(seconds=0.01),
  1 : td(seconds=0.04),
  2 : td(seconds=0.2),
  3 : td(seconds=1),
  4 : td(seconds=5)
}
ASSOCIATE_OFFSET = td(seconds=1)

# Strings
EMPTY_STR = ''
ALL_WILDCHAR_STR = '*'
PERIOD_STR = '.'
UNDERSCORE_STR = '_'
SPACE_STR = ' '
COMMA_STR = ','
NONE_STR = "None"
CLF_STR = "classified"
AST_STR = "associated"
FILE_STR = "file"
TEMPORAL_STR = "tmp"
STATUS_STR = "status"

TRUE_STR = "TRUE"
PRED_STR = "PRED"

# Metrics
TP_STR = "True Positive"
FP_STR = "False Positive"
FN_STR = "False Negative"
TN_STR = "True Negative"
ACCURACY_STR = "Accuracy"
PRECISION_STR = "Precision"
RECALL_STR = "Recall"
F1_STR = "F1 Score"

# Phases
PWAVE = "P"
SWAVE = "S"

# Thresholds
PWAVE_THRESHOLD = 0.2
SWAVE_THRESHOLD = 0.1

SEED_ID_FMT = "{NETWORK}.{STATION}..{CHANNEL}"

CFN_MTX_STR = "CM"
CMTV_PICKS_STR = "CP"
TIME_DSPLCMT_STR = "TD"

# TODO: Add Tabular data for relational databases for future development

# Extensions
CSV_STR       = "csv"
DAT_STR       = "dat"
EPS_STR       = "eps"
HDF5_STR      = "h5"
JSON_STR      = "json"
MSEED_STR     = "mseed"
PDF_STR       = "pdf"
PICKLE_STR    = "pkl"
PNG_STR       = "png"
PUN_STR       = "pun"
TORCH_STR     = "pt"
XML_STR       = "xml"

CSV_EXT       = PERIOD_STR + CSV_STR
DAT_EXT       = PERIOD_STR + DAT_STR
EPS_EXT       = PERIOD_STR + EPS_STR
HDF5_EXT      = PERIOD_STR + HDF5_STR
JSON_EXT      = PERIOD_STR + JSON_STR
MSEED_EXT     = PERIOD_STR + MSEED_STR
PDF_EXT       = PERIOD_STR + PDF_STR
PICKLE_EXT    = PERIOD_STR + PICKLE_STR
PNG_EXT       = PERIOD_STR + PNG_STR
PUN_EXT       = PERIOD_STR + PUN_STR
TORCH_EXT     = PERIOD_STR + TORCH_STR
XML_EXT       = PERIOD_STR + XML_STR

PRC_FMT = SEED_ID_FMT + ".{BEGDT}.{EXT}"

# Models
DEEPDENOISER_STR  = "DeepDenoiser"
EQTRANSFORMER_STR = "EQTransformer"
PHASENET_STR      = "PhaseNet"

# Various pre-trained weights for each model (Add if new are available)
MODEL_WEIGHTS_DICT = {
  DEEPDENOISER_STR  : sbm.DeepDenoiser(sampling_rate=SAMPLING_RATE),
  EQTRANSFORMER_STR : sbm.EQTransformer(phases=PWAVE + SWAVE,
                                        sampling_rate=SAMPLING_RATE,
                                        norm=NORM),
  PHASENET_STR      : sbm.PhaseNet(phases=PWAVE + SWAVE,
                                   sampling_rate=SAMPLING_RATE,
                                   norm=NORM)
}

# Colors
COLORS = {
  PWAVE: "C0",
  SWAVE: "C1",
  "Detection": "C2"
}
COLOR_ENCODING = {
  TP_STR : {
    PWAVE: "red",
    SWAVE: "blue"
  },
  FP_STR : {
    PWAVE: "orange",
    SWAVE: "green"
  },
  FN_STR : {
    PWAVE: "pink",
    SWAVE: "purple"
  }
}

# Data components
ID_STR = "id"
TIMESTAMP_STR = "timestamp"
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
BEG_DATE_STR = "BEGDT"

# Labelled Data components
P_TIME_STR      = "P_TIME"
P_TYPE_STR      = "P_TYPE"
P_WEIGHT_STR    = "P_WEIGHT"
S_TIME_STR      = "S_TIME"
S_TYPE_STR      = "S_TYPE"
S_WEIGHT_STR    = "S_WEIGHT"
# TODO: Implement polarity
PHASE_EXTRACTOR = \
  re.compile(fr"^(?P<{STATION_STR}>(\w{{4}}|\w{{3}}\s))"            # Station
             fr"(?P<{P_TYPE_STR}>[ei?]{PWAVE}[cd\s])"               # P Type
             fr"(?P<{P_WEIGHT_STR}>[0-4])"                          # P Weight
             fr"1(?P<{BEG_DATE_STR}>\d{{10}})"                      # Date
             fr"\s(?P<{P_TIME_STR}>\d{{4}})"                        # P Time
             fr"\s+((?P<{S_TIME_STR}>\d{{4}}|\d{{3}})"              # S Time
             fr"(?P<{S_TYPE_STR}>[ei?]{SWAVE}\s)"                   # S Type
             fr"(?P<{S_WEIGHT_STR}>[0-4]))*")                       # S Weight
EVENT_EXTRACTOR = re.compile(r"^1(\s+D)*\s*$")                      # Event

# Pretrained model weights
ADRIAARRAY_STR  = "adriaarray"
INSTANCE_STR    = "instance"
ORIGINAL_STR    = "original"
SCEDC_STR       = "scedc"
STEAD_STR       = "stead"

# Clients
INGV_CLIENT_STR   = "INGV"
IRIS_CLIENT_STR   = "IRIS"
GFZ_CLIENT_STR    = "GFZ"
ETH_CLIENT_STR    = "ETH"
ORFEUS_CLIENT_STR = "ORFEUS"
GEOFON_CLIENT_STR = "GEOFON"
RESIF_CLIENT_STR  = "RESIF"
LMU_CLIENT_STR    = "LMU"
USGS_CLIENT_STR   = "USGS"
EMSC_CLIENT_STR   = "EMSC"
GEONET_CLIENT_STR = "GEONET"
ODC_CLIENT_STR    = "ODC"
GEONET_CLIENT_STR = "GEONET"
GEONET_CLIENT_STR = "GEONET"
OGS_CLIENT_STR    = "http://158.110.30.217:8080"
RASPISHAKE_CLIENT_STR = "RASPISHAKE"

ASSOCIATION_CONFIG = {
  DIMENSIONS_STR : [X_COORD_STR, Y_COORD_STR, Z_COORD_STR],
  "use_dbscan" : True,
  "use_amplitude" : False,
  X_COORD_STR : (250, 600),
  Y_COORD_STR : (7200, 8000),
  Z_COORD_STR : (0, 20),
  VELOCITY_STR : {
    PWAVE.lower(): 5.85,
    SWAVE.lower(): 5.85 / 1.78
  },
  METHOD_STR : BAYES_GAUSS_MIX_MODEL_STR,
  "oversample_factor" : 4,
  "dbscan_eps" : 0.5,
  "dbscan_min_samples" : 3,
  "min_picks_per_eq" : 5,
  "min_p_picks_per_eq" : 0,
  "min_s_picks_per_eq" : 0,
  "max_sigma11" : 3.0,
  "max_sigma22" : 1.0,
  "max_sigma12" : 1.0,
}

# DBSCAN
ASSOCIATION_CONFIG["bfgs_bounds"] = (
  (ASSOCIATION_CONFIG[X_COORD_STR][0] - 1, ASSOCIATION_CONFIG[X_COORD_STR][1] + 1),  # x
  (ASSOCIATION_CONFIG[Y_COORD_STR][0] - 1, ASSOCIATION_CONFIG[X_COORD_STR][1] + 1),  # y
  (0, ASSOCIATION_CONFIG[Z_COORD_STR][1] + 1),  # x
  (None, None),  # t
)

EIKONAL_P = [3.0, 3.59,  4.0,  4.8, 5.59, 6.5, 8.0]

ASSOCIATION_CONFIG["eikonal"] = {
  VELOCITY_STR : {
    PWAVE.lower(): EIKONAL_P,
    SWAVE.lower(): [v / 1.75 for v in EIKONAL_P],
    "z" : [0.0, 0.5, 2.0, 4.0, 6.0, 12.0, 30.0]
  },
  "h": 1.0,
  "xlim": ASSOCIATION_CONFIG[X_COORD_STR],
  "ylim": ASSOCIATION_CONFIG[Y_COORD_STR],
  "zlim": ASSOCIATION_CONFIG[Z_COORD_STR]
}
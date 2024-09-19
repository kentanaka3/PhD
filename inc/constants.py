import re
import seisbench.models as sbm
from datetime import timedelta as td

SAMPLING_RATE = 100

EPSILON = 1e-6

NORM = "peak" # "peak" or "std"

# DateTime, TimeDelta and Format constants
DATE_FMT = "%y%m%d"
DATETIME_FMT = DATE_FMT + "%H%M%S"
ONE_DAY = td(days=1)
PICK_OFFSET = td(seconds=0.5)
ASSOCIATE_OFFSET = td(seconds=1)

# Strings
EMPTY_STR = ''
ALL_WILDCHAR_STR = '*'
PERIOD_STR = '.'
UNDERSCORE_STR = '_'
SPACE_STR = ' '
COMMA_STR = ','
NONE_STR = "None"
PRC_STR = "processed"
ANT_STR = "annotated"
CLF_STR = "classified"

TRUE_STR = "TRUE"
PRED_STR = "PRED"

# Metrics
TP_STR = "True_Positive"
FP_STR = "False_Positive"
FN_STR = "False_Negative"
TN_STR = "True_Negative"
ACCURACY_STR = "Accuracy"
PRECISION_STR = "Precision"
RECALL_STR = "Recall"
F1_STR = "F1_Score"

# Phases
PWAVE = "P"
SWAVE = "S"

# Thresholds
PWAVE_THRESHOLD = 0.2
SWAVE_THRESHOLD = 0.1

SEED_ID_FMT = "{NETWORK}.{STATION}..{CHANNEL}"

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

COLORS = {
  PWAVE: "C0",
  SWAVE: "C1",
  "Detection": "C2"
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
STATION_STR = "STATION"
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
INGV_STR = "INGV"
IRIS_STR = "IRIS"
OGS_STR  = "http://158.110.30.217:8080"
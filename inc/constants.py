import re
import seisbench.models as sbm
from datetime import timedelta as td

SAMPLING_RATE = 100

EPSILON = 1e-6

NORM = "peak" # "peak" or "std"

# DateTime, TimeDelta and Format constants
DATE_FMT = "%y%m%d"
DATETIME_FMT = "%y%m%d%H%M%S"
ONE_DAY = td(days=1)
PICK_OFFSET = td(seconds=0.5)
ASSOCIATE_OFFSET = td(seconds=1)

EMPTY_STR = ''
ALL_WILDCHAR_STR = '*'
PERIOD_STR = '.'
UNDERSCORE_STR = '_'
SPACE_STR = ' '
PRC_STR = "processed"
ANT_STR = "annotated"
CLF_STR = "classified"

PWAVE = "P"
SWAVE = "S"

PWAVE_THRESHOLD = 0.2
SWAVE_THRESHOLD = 0.1

# Extensions
CSV_EXT       = "csv"
DAT_EXT       = "dat"
EPS_EXT       = "eps"
HDF5_EXT      = "h5"
JSON_EXT      = "json"
MSEED_EXT     = "mseed"
PDF_EXT       = "pdf"
PICKLE_EXT    = "pkl"
PNG_EXT       = "png"
PUN_EXT       = "pun"
TORCH_EXT     = "pt"

PRC_FMT = "{NETWORK}.{STATION}.{CHANNEL}.{BEGDT}.{EXT}"

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

MSEED_STR = "MSEED"

MODEL_STR = "MODEL"
WEIGHT_STR = "WEIGHT"
PHASE_STR = "PHASE"
THRESHOLD_STR = "THRESHOLD"
RESULTS_STR = "RESULTS"
FILENAME_STR = "FILENAME"
NETWORK_STR = "NETWORK"
STATION_STR = "STATION"
CHANNEL_STR = "CHANNEL"
BEG_DATE_STR = "BEGDT"
TIMESTAMP_STR = "TIMESTAMP"
PROBABILITY_STR = "PROBABILITY"
HEADER = [FILENAME_STR, NETWORK_STR, STATION_STR, CHANNEL_STR, BEG_DATE_STR]

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
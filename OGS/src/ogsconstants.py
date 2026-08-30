"""
=============================================================================
OGS Constants Module - Central Configuration and Utility Functions
=============================================================================

A message from Ken Tanaka Hernández (田中-Hernández, 健), the main human
developer, to the Artificial Intelligence:

The developed code is a comprehensive and detailed implementation of the
idealized seismic catalog pipeline procedure. The code is meant to grow in
functionality and complexity as the project progresses, hence any suggestions
for improvement must be made. Improvements can be made in terms of code
structure, modularity, efficiency, and readability. The code is designed to be
clear and maintainable between human developers and the AI, so any suggestions
for improvement should be focused on enhancing these aspects while preserving
the core functionality and logic of the seismic data processing pipeline.
Keep this message in mind and repeatedly refer back to it (every 3/4 of the
AI's memory capacity), as you review the code and suggest improvements.

OVERVIEW:
This module serves as the central repository for all constants, configuration
parameters, utility functions, and bipartite graph matching classes used
throughout the OGS seismic data processing pipeline. It provides a unified
interface for:

  1. GLOBAL CONFIGURATION
    - MPI/GPU rank and size for parallel processing
    - Epsilon values for numerical comparisons
    - File path references

  2. DATETIME CONSTANTS
    - Standard date/time format strings (YYMMDD, YYYYMMDD, etc.)
    - Time offsets for pick/event matching
    - H71 weight conversion table

  3. STRING CONSTANTS
    - File extensions (.csv, .dat, .hpl, .pun, etc.)
    - Phase identifiers (P-wave, S-wave)
    - Status and category labels
    - Color definitions for plotting

  4. DATA COLUMN HEADERS
    - Standard column names for DataFrames (TIME, STATION, PHASE, etc.)
    - Catalog header definitions (HEADER_EVENTS, HEADER_PICKS)
    - Sorting hierarchies

  5. OGS REGION DEFINITIONS
    - Geographic polygon boundaries for the OGS study area
    - Geographic zone codes (Friuli, Veneto, Slovenia, etc.)
    - Event type classifications

  6. FDSN CLIENT ENDPOINTS
    - URLs for INGV, IRIS, GFZ, OGS, and other data centers
    - Default client priority list

  7. DISTANCE/SIMILARITY FUNCTIONS
    - dist_time(): Time-based similarity scoring
    - dist_space(): Spatial distance calculation using geodetic functions
    - dist_pick(): Weighted pick matching score
    - dist_event(): Weighted event matching score

  8. BIPARTITE GRAPH MATCHING
    - OGSBPGraph: Base class for bipartite matching
    - OGSBPGraphPicks: Maximum weight matching for phase picks
    - OGSBPGraphEvents: Maximum weight matching for seismic events

ARCHITECTURE:
                    ┌─────────────────────────────────────┐
                    │         ogsconstants.py             │
                    ├─────────────────────────────────────┤
                    │  Constants     │  Utility Functions │
                    │  ────────────  │  ────────────────  │
                    │  • Formats     │  • is_date()       │
                    │  • Extensions  │  • is_file_path()  │
                    │  • Headers     │  • decimeter()     │
                    │  • Colors      │  • inventory()     │
                    │  • Thresholds  │  • waveforms()     │
                    ├─────────────────────────────────────┤
                    │        Bipartite Graph Classes      │
                    │  ─────────────────────────────────  │
                    │  OGSBPGraph (base)                  │
                    │    ├── OGSBPGraphPicks              │
                    │    └── OGSBPGraphEvents             │
                    └─────────────────────────────────────┘

USAGE:
  from ogsconstants import (
    PWAVE, SWAVE,           # Phase identifiers
    DATE_FMT, TIME_FMT,     # Format strings
    HEADER_PICKS,           # Column headers
    dist_pick, dist_event,  # Matching functions
    OGSBPGraphPicks         # Bipartite matching
  )

DEPENDENCIES:
  - numpy: Numerical operations
  - pandas: DataFrame handling
  - obspy: Seismological utilities (UTCDateTime, geodetics)
  - networkx: Graph algorithms for bipartite matching
  - matplotlib: Plotting utilities

AUTHOR: AI2Seism Project
=============================================================================
"""

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import re                                 # Regular expression operations
from pathlib import Path                  # Object-oriented filesystem paths
from datetime import timedelta as td      # Time handling

# =============================================================================
# THIRD-PARTY LIBRARY IMPORTS
# =============================================================================
import numpy as np                        # Numerical computing


# =============================================================================
# MODULE-LEVEL CONFIGURATION
# =============================================================================

# Reference to this file's path for relative imports
THIS_FILE = Path(__file__)

# =============================================================================
# NUMERICAL CONSTANTS
# =============================================================================

# Small epsilon value for floating-point comparisons to avoid division by zero
EPSILON = 1e-6

# =============================================================================
# MPI PARALLEL PROCESSING CONFIGURATION
# =============================================================================
# These values are initialized at startup and modified by MPI initialization
# when running in parallel mode on HPC clusters

MPI_RANK = 0      # Current MPI process rank (0 = master, default for serial)
MPI_SIZE = 1      # Total number of MPI processes (1 = serial execution)
MPI_COMM = None   # MPI communicator object (None when not using MPI)

# =============================================================================
# GPU CONFIGURATION
# =============================================================================
# GPU allocation for CUDA-accelerated processing (e.g., ML inference)

GPU_SIZE = 0      # Total number of available GPUs
GPU_RANK = -1     # Assigned GPU device ID (-1 = no GPU assigned)

# =============================================================================
# PROBABILITY THRESHOLDS
# =============================================================================
# Threshold values for ML model confidence scoring (0.1 to 0.9 in 0.1 steps)
# Used for pick probability filtering and performance evaluation

THRESHOLDS: list[str] = ["{:.1f}".format(t) for t in np.linspace(0.1, 0.9, 9)]

# =============================================================================
# DATE/TIME FORMAT CONSTANTS
# =============================================================================
# Standard format strings for parsing and formatting dates/times throughout
# the OGS pipeline. Uses Python strftime/strptime conventions.

DATE_STD = "YYMMDD"                   # Standard date representation string
DATE_FMT = "%Y-%m-%d"                 # ISO date format (2022-01-15)
TIME_FMT = "%H%M%S"                   # Compact time format (143052)
YYMMDD_FMT = "%y%m%d"                 # 2-digit year date (220115)
YYYYMMDD_FMT = "%Y%m%d"               # 4-digit year date (20220115)
DATETIME_FMT = YYMMDD_FMT + TIME_FMT  # Combined datetime (220115143052)
DATETIME_STR = "DATETIME"             # Column name for datetime fields
TIMESTAMP_STR = "TIMESTAMP"           # Column name for Unix timestamps

# =============================================================================
# TIME DELTA CONSTANTS
# =============================================================================
# Time intervals used for event detection, pick matching, and data segmentation

ONE_DAY = td(days=1)                  # One day interval for date iteration

# Maximum time difference for matching predicted picks to manual picks
# Picks within this window are considered potential matches
PICK_TIME_OFFSET = td(seconds=.5)     # 0.5 second tolerance for pick matching

# Time window for training data extraction around picks
PICK_TRAIN_OFFSET = td(seconds=60)    # 60 second window for ML training

# =============================================================================
# H71 WEIGHT CONVERSION TABLE
# Mapping of H71 weight classes to numerical offsets for event matching
# Used in the dist_event() function to convert H71 weights to time offsets
H71_OFFSET: dict[int, float] = {
    0: 0.01,
    1: 0.04,
    2: 0.2,
    3: 1,
    4: 5,
    5: 25
}
"""
===============================================================================
H71 WEIGHT CONVERSION TABLE
===============================================================================
Hypo71 standard weight codes mapped to uncertainty in seconds
These represent picking precision: 0 = most precise, 5 = least precise
Weight | Uncertainty (sec) | Interpretation
-------|-------------------|----------------
  0    |       0.01        | Impulsive onset, very clear
  1    |       0.04        | Clear onset
  2    |       0.2         | Fairly clear onset
  3    |       1.0         | Emergent onset
  4    |       5.0         | Poor quality pick
  5    |      25.0         | Very uncertain (often unused)
"""

# =============================================================================
# EVENT MATCHING TOLERANCES
# =============================================================================
# Thresholds for matching detected events to catalog events

EVENT_TIME_OFFSET = td(seconds=2)
"""
Max time difference for event matching: 2 seconds\n
This is used in the dist_event() function to determine if a detected event is
close enough in time to a catalog event to be considered a match. This
threshold accounts for uncertainties in pick timing, association, and event
location.
"""
EVENT_DIST_OFFSET = 8                 # Max spatial distance (km) for matching
"""
Max spatial distance for event matching: 8 km\n
This is used in the dist_event() function to determine if a detected event is
close enough to a catalog event to be considered a match.
"""

# Commonly used string literals to ensure consistency and avoid typos
EMPTY_STR = ''                          # Empty string for initialization
ALL_WILDCHAR_STR = '*'                  # Wildcard for matching all entries
ONE_MORECHAR_STR = '+'                  # Regex: one or more characters
PERIOD_STR = '.'                        # Period (used in SEED IDs, extensions)
UNDERSCORE_STR = '_'                    # Underscore (filename separator)
DASH_STR = '-'                          # Dash
SPACE_STR = ' '                         # Space character
COMMA_STR = ','                         # Comma character
SEMICOL_STR = ';'                       # Semicolon character
ZERO_STR = "0"                          # String representation of zero
NAN_STR = "NaN"                         # String representation of Not-a-Number
NONE_STR = "None"                       # String representation of None

# =============================================================================
# PIPELINE COMPONENT IDENTIFIERS
# =============================================================================
# String identifiers for various pipeline stages and components

DEFAULT_PICKER = "SeisBenchPicker"      # ML-based phase picker identifier
DEFAULT_ASSOCIATOR = "GammaAssociator"  # GaMMA phase associator identifier
FILE_STR = "file"                       # Generic file reference
TEMPORAL_STR = "tmp"                    # Temporary file prefix
DURATION_STR = "duration"               # Duration field name
STATUS_STR = "status"                   # Status field name
SECONDS_STR = "seconds"                 # Seconds unit label

# =============================================================================
# POLARITY IDENTIFIERS
# =============================================================================
# First-motion polarity labels for focal mechanism analysis

COMPRESSIONAL_STR = "compressional"    # Upward first motion (compression)
DILATATIONAL_STR = "dilatational"      # Downward first motion (dilation)

# =============================================================================
# CLASSIFICATION AND LOGGING LABELS
# =============================================================================
# Labels used for categorization and log message formatting

CLSSFD_STR = "CLSSFD"                  # Classified status marker
SOURCE_STR = "SOURCE"                  # Data source identifier
DETECT_STR = "DETECT"                  # Detection status
UNKNOWN_STR = "UNKNOWN"                # Unknown/unclassified label
LEVEL_STR = "LEVEL"                    # Log level indicator
WARNING_STR = "WARNING"                # Warning log level
FATAL_STR = "FATAL"                    # Fatal error log level
NOTABLE_STR = "NOTABLE"                # Notable event marker
ASSIGN_STR = "ASSIGN"                  # Assignment status
UNABLE_STR = "UNABLE"                  # Unable to process marker

# =============================================================================
# DATA CATEGORY LABELS
# =============================================================================
# Labels for distinguishing between manual (TRUE) and predicted data

TRUE_STR = "TRUE"                      # Manual/ground truth data
PRED_STR = "PRED"                      # Predicted/ML-generated data
ASCT_STR = "ASCT"                      # Associated data marker
STAT_STR = "STAT"                      # Statistics marker
FALSE_STR = "FALSE"                    # Negative/false marker

# =============================================================================
# ASSOCIATOR ALGORITHM IDENTIFIERS
# =============================================================================
# Names for phase association algorithms

GMMA_STR = "GaMMA"                     # GaMMA (Gaussian Mixture Model Assoc.)
OCTO_STR = "PyOcto"                    # PyOcto (Octree-based associator)

# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================
# String constants for confusion matrix and performance evaluation metrics

TP_STR = "TP"                          # True Positive count
FP_STR = "FP"                          # False Positive count
FN_STR = "FN"                          # False Negative count
TN_STR = "TN"                          # True Negative count
ACCURACY_STR = "AC"                    # Accuracy metric
PRECISION_STR = "PC"                   # Precision metric
RECALL_STR = "RC"                      # Recall metric
NETCOLOR_STR = "NC"                    # Network color for plotting
STACOLOR_STR = "SC"                    # Station color for plotting
F1_STR = "F1"                          # F1 score metric
DISTANCE_STR = "Distance"              # Distance metric label

# =============================================================================
# SEISMIC PHASE IDENTIFIERS
# =============================================================================
# Standard phase type labels for P and S waves

PWAVE = "P"                            # Primary (compressional) wave
SWAVE = "S"                            # Secondary (shear) wave

# =============================================================================
# DEFAULT PHASE THRESHOLDS
# =============================================================================
# Minimum probability thresholds for accepting ML-detected phases

PWAVE_THRESHOLD = SWAVE_THRESHOLD = 0.1  # 10% minimum confidence

# =============================================================================
# SEED IDENTIFIER FORMAT
# =============================================================================
# FDSN SEED naming convention for seismic channels
# Format: NETWORK.STATION.LOCATION.CHANNEL (e.g., IV.ACER..HHZ)

SEED_ID_FMT = "{NETWORK}.{STATION}..{CHANNEL}"

# =============================================================================
# OUTPUT FILE IDENTIFIERS
# =============================================================================
# Prefixes/suffixes for various output file types

CFN_MTX_STR = "CM"                     # Confusion matrix output
CMTV_PICKS_STR = "CP"                  # Cumulative picks output
CLSTR_PLOT_STR = "CT"                  # Cluster plot output
TIME_DSPLCMT_STR = "TD"                # Time displacement output

# =============================================================================
# COLOR PALETTE FOR PLOTTING
# =============================================================================
# Hex color codes for consistent visualization across the project

MEX_PINK = "#E4007C"                  # Bright pink (accent color)
OGS_BLUE = "#163771"                  # OGS institutional blue (primary)
ALN_GREEN = "#00e468"                 # Bright green (positive/success)
LIP_ORANGE = "#FF8C00"                # Orange (warning/highlight)
SUN_YELLOW = "#e4da00"                # Yellow (tertiary accent)

# Standard color sequence for multi-series plots
PLOT_COLORS = [OGS_BLUE, MEX_PINK, ALN_GREEN, LIP_ORANGE, SUN_YELLOW]

# TODO: Add Tabular data for relational databases for future development

# =============================================================================
# FILE EXTENSION CONSTANTS
# =============================================================================
# String constants for file type extensions (without leading period)

BLT_STR = "blt"                        # Bulletin file format
CSV_STR = "csv"                        # Comma-separated values
DAT_STR = "dat"                        # OGS phase data format
EPS_STR = "eps"                        # Encapsulated PostScript (vector)
HDF5_STR = "hdf5"                      # Hierarchical Data Format 5
HPC_STR = "hpc"                        # HPC-specific format
HPL_STR = "hpl"                        # OGS hypocenter location format
JSON_STR = "json"                      # JavaScript Object Notation
LD_STR = "ld"                          # Linked data format
MOD_STR = "mod"                        # Model/velocity model format
MSEED_STR = "mseed"                    # MiniSEED waveform format
PDF_STR = "pdf"                        # Portable Document Format
PICKLE_STR = "pkl"                     # Python pickle serialization
PNG_STR = "png"                        # Portable Network Graphics (raster)
PRT_STR = "prt"                        # Print/report file format
PUN_STR = "pun"                        # OGS punch card output format
QML_STR = "qml"                        # QuakeML seismic data exchange
TORCH_STR = "pt"                       # PyTorch model weights
TXT_STR = "txt"                        # Plain text format
XML_STR = "xml"                        # Extensible Markup Language

# =============================================================================
# FILE EXTENSION CONSTANTS (WITH PERIOD)
# =============================================================================
# Full file extensions including the leading period for direct use

BLT_EXT = PERIOD_STR + BLT_STR         # .blt
CSV_EXT = PERIOD_STR + CSV_STR         # .csv
DAT_EXT = PERIOD_STR + DAT_STR         # .dat
EPS_EXT = PERIOD_STR + EPS_STR         # .eps
HDF5_EXT = PERIOD_STR + HDF5_STR       # .hdf5
HPC_EXT = PERIOD_STR + HPC_STR         # .hpc
HPL_EXT = PERIOD_STR + HPL_STR         # .hpl
JSON_EXT = PERIOD_STR + JSON_STR       # .json
LD_EXT = PERIOD_STR + LD_STR           # .ld
MOD_EXT = PERIOD_STR + MOD_STR         # .mod
MSEED_EXT = PERIOD_STR + MSEED_STR     # .mseed
PDF_EXT = PERIOD_STR + PDF_STR         # .pdf
PICKLE_EXT = PERIOD_STR + PICKLE_STR   # .pkl
PNG_EXT = PERIOD_STR + PNG_STR         # .png
PRT_EXT = PERIOD_STR + PRT_STR         # .prt
PUN_EXT = PERIOD_STR + PUN_STR         # .pun
QML_EXT = PERIOD_STR + QML_STR         # .qml
TORCH_EXT = PERIOD_STR + TORCH_STR     # .pt
TXT_EXT = PERIOD_STR + TXT_STR         # .txt
XML_EXT = PERIOD_STR + XML_STR         # .xml

# =============================================================================
# WAVEFORM FILE NAMING FORMAT
# =============================================================================
# Template for constructing waveform filenames following SEED conventions

PRC_FMT = SEED_ID_FMT + ".{BEGDT}.{EXT}"  # NETWORK.STATION..CHANNEL.DATE.EXT

# =============================================================================
# ML MODEL IDENTIFIERS
# =============================================================================
# Names of supported machine learning models for phase picking

EQTRANSFORMER_STR = "EQTransformer"    # EQTransformer deep learning model
PHASENET_STR = "PhaseNet"              # PhaseNet deep learning model

# =============================================================================
# OGS PROJECTION SYSTEM
# =============================================================================
# Stereographic projection parameters for local coordinate transformation
# Uses PROJ4 format string with placeholder for center coordinates

OGS_PROJECTION = "+proj=sterea +lon_0={lon} +lat_0={lat} +units=km"

# Maximum magnitude threshold for OGS catalog (filter out larger events)
OGS_MAX_MAGNITUDE = 3.5
OGS_MAGNITUDE_SIZE = {
    # Magnitude : [Marker Size]
    (-1., 0.): 10,
    (0., 1.): 20,
    (1., 2.): 40,
    (2., 3.): 80,
    (3., OGS_MAX_MAGNITUDE): 160,
}

# =============================================================================
# DATAFRAME COLUMN NAME CONSTANTS
# =============================================================================
# Standardized column names for pandas DataFrames throughout the pipeline

# Pick-related columns
IDX_PICKS_STR = "index"                # Pick index identifier
GROUPS_STR = "group"                   # Group/cluster identifier
TIME_STR = "time"                      # Timestamp column
STATION_STR = "station"                # Station identifier
PHASE_STR = "phase"                    # Phase type (P or S)
PROBABILITY_STR = "probability"        # ML confidence score
AMPLITUDE_STR = "amplitude"            # Waveform amplitude
EPICENTRAL_DISTANCE_STR = "epicentral_distance"  # Distance from epicenter
DEPTH_STR = "depth"                    # Event depth (km)
STATION_ML_STR = "station_ML"          # Station-specific magnitude
NUMBER_P_PICKS_STR = "number_p_picks"  # Count of P-wave picks
NUMBER_S_PICKS_STR = "number_s_picks"  # Count of S-wave picks
NUMBER_P_AND_S_PICKS_STR = "number_p_and_s_picks"  # Count of P+S picks

# Magnitude-related columns
ML_STR = "ML"                          # Local magnitude
ML_MEDIAN_STR = "ML_median"            # Median local magnitude
ML_UNC_STR = "ML_unc"                  # Magnitude uncertainty
ML_STATIONS_STR = "ML_stations"        # Number of stations for ML

# Event identification columns
IDX_EVENTS_STR = "idx"                 # Event index identifier
INDEX_STR = "idx"                      # Generic index column
METADATA_STR = "metadata"              # Metadata container column
TYPE_STR = "type"                      # Type classification column

# Geographic coordinate columns
LONGITUDE_STR = "longitude"            # Longitude (degrees)
LATITUDE_STR = "latitude"              # Latitude (degrees)
ELEVATION_STR = "elevation"            # Elevation in meters
X_COORD_STR = "x(km)"                  # X coordinate in kilometers (local)
Y_COORD_STR = "y(km)"                  # Y coordinate in kilometers (local)
Z_COORD_STR = "z(km)"                  # Z coordinate in kilometers (depth)

# Additional event attributes
MAGNITUDE_STR = "magnitude"            # Generic magnitude column
MAGNITUDE_L_STR = "ML"                 # Local magnitude type
MAGNITUDE_D_STR = "MD"                 # Duration magnitude type
PLACE_STR = "place"                    # Location description
VELOCITY_STR = "vel"                   # Velocity model reference
METHOD_STR = "method"                  # Processing method used
DIMENSIONS_STR = "dims"                # Dimensionality (2D/3D)

# Clustering method identifiers
GAUSS_MIX_MODEL_STR = "GMM"            # Gaussian Mixture Model
BAYES_GAUSS_MIX_MODEL_STR = "B" + GAUSS_MIX_MODEL_STR  # Bayesian GMM

# =============================================================================
# CONFIGURATION AND PATH COLUMN NAMES
# =============================================================================
# Column names for configuration DataFrames and file management

ARGUMENTS_STR = "arguments"            # Command-line arguments
WAVEFORMS_STR = "waveforms"            # Waveform data reference
DATASETS_STR = "datasets"              # Dataset identifiers
MODELS_STR = "models"                  # Model identifiers

# Comparison labels for base vs. target analysis
BASE_STR = "Base"                      # Reference/ground truth dataset
TARGET_STR = "Target"                  # Comparison/predicted dataset

# =============================================================================
# UPPERCASE COLUMN NAMES FOR HEADERS
# =============================================================================
# Uppercase versions for header rows and configuration files

EVENT_STR = "EVENT"                    # Event identifier (uppercase)
MODEL_STR = "MODEL"                    # Model name column
WEIGHT_STR = "WEIGHT"                  # Weight/pretrained weights
DIRECTORY_STR = "DIRECTORY"            # Directory path column
JULIAN_STR = "JULIAN"                  # Julian day column
DENOISER_STR = "DENOISER"              # Denoising model reference
DOMAIN_STR = "DOMAIN"                  # Domain/region identifier
CLIENT_STR = "CLIENT"                  # FDSN client identifier
RESULTS_STR = "RESULTS"                # Results directory
FILENAME_STR = "FILENAME"              # Filename column
THRESHOLD_STR = "THRESHOLD"            # Probability threshold column
NETWORK_STR = "NETWORK"                # Seismic network code
CHANNEL_STR = "CHANNEL"                # Channel code
DATE_STR = "DATE"                      # Date column

# =============================================================================
# LABELLED DATA COLUMN NAMES (P AND S WAVE)
# =============================================================================
# Column names for manually labeled phase data with P and S wave attributes

# P-wave pick attributes
P_TIME_STR = "P_TIME"                  # P-wave arrival time
P_TYPE_STR = "P_TYPE"                  # P-wave type (e.g., Pg, Pn)
P_ONSET_STR = "P_ONSET"                # P-wave onset quality (I/E)
P_POLARITY_STR = "P_POLARITY"          # P-wave first motion (U/D)
P_WEIGHT_STR = "P_WEIGHT"              # P-wave pick weight (0-4)

# S-wave pick attributes
S_TIME_STR = "S_TIME"                  # S-wave arrival time
S_TYPE_STR = "S_TYPE"                  # S-wave type (e.g., Sg, Sn)
S_ONSET_STR = "S_ONSET"                # S-wave onset quality
S_POLARITY_STR = "S_POLARITY"          # S-wave polarity (if measurable)
S_WEIGHT_STR = "S_WEIGHT"              # S-wave pick weight

# =============================================================================
# EVENT QUALITY INDICATORS
# =============================================================================
# Column names for event location quality metrics

ORIGIN_STR = "ORIGIN"                  # Origin time column
NO_STR = "number_picks"                # Number of picks used
GAP_STR = "azimuthal_gap"              # Azimuthal gap in degrees
DMIN_STR = "DMIN"                      # Distance to nearest station
RMS_STR = "RMS"                        # RMS travel time residual
ERH_STR = "max_horizontal_uncertainty"  # Horizontal error (km)
ERZ_STR = "vertical_uncertainty"       # Vertical error (km)
ERT_STR = "weight"                     # Overall location weight
QM_STR = "QM"                          # Quality metric
ONSET_STR = "ONSET"                    # Onset type (I=impulsive, E=emergent)
POLARITY_STR = "POLARITY"              # First motion polarity (U/D)

# =============================================================================
# OGS GEOGRAPHIC CLASSIFICATION
# =============================================================================
# Column names and values for OGS regional earthquake classification

GEO_ZONE_STR = "GEOZONE"               # Geographic zone code column
EVENT_TYPE_STR = "E_TYPE"              # Event type column

# Event type classification values
EVENT_LOCAL_EQ_STR = "local_eq"        # Local tectonic earthquake
EVENT_EXPLD_STR = "explosion"          # Industrial explosion
EVENT_BOMB_STR = "bomb"                # Military detonation (historical)
EVENT_LNDSLD_STR = "landslide"         # Landslide-induced event
EVENT_UNKNOWN_STR = UNKNOWN_STR        # Unknown/unclassified event

# Location metadata
EVENT_LOCALIZATION_STR = "E_LOC"       # Localization method/status
LOC_NAME_STR = "LOC_NAME"              # Location place name
NOTES_STR = "NOTES"                    # Analyst notes field

# =============================================================================
# PRETRAINED MODEL WEIGHT IDENTIFIERS
# =============================================================================
# Names of pretrained weight variants for SeisBench models

ADRIAARRAY_STR = "adriaarray"          # Trained on AdriaArray data
INSTANCE_STR = "instance"              # Trained on INSTANCE dataset
ORIGINAL_STR = "original"              # Original author weights
SCEDC_STR = "scedc"                    # Southern California Earthquake DC
STEAD_STR = "stead"                    # STanford EArthquake Dataset

# =============================================================================
# FDSN WEB SERVICE CLIENT IDENTIFIERS
# =============================================================================
# Standard FDSN data center names and OGS-specific endpoints

# Major international FDSN data centers
INGV_CLIENT_STR = "INGV"               # Italian National Institute
IRIS_CLIENT_STR = "IRIS"               # US IRIS Data Management Center
GFZ_CLIENT_STR = "GFZ"                 # German Research Centre, Potsdam
ETH_CLIENT_STR = "ETH"                 # Swiss Seismological Service
ORFEUS_CLIENT_STR = "ORFEUS"           # European ORFEUS Data Center
GEOFON_CLIENT_STR = "GEOFON"           # GFZ GEOFON program
RESIF_CLIENT_STR = "RESIF"             # French RESIF network
LMU_CLIENT_STR = "LMU"                 # Ludwig Maximilian University
USGS_CLIENT_STR = "USGS"               # US Geological Survey
EMSC_CLIENT_STR = "EMSC"               # Euro-Mediterranean Seismological
ODC_CLIENT_STR = "ODC"                 # ORFEUS Data Center
GEONET_CLIENT_STR = "GEONET"           # New Zealand GeoNet
RASPISHAKE_CLIENT_STR = "RASPISHAKE"   # Raspberry Shake citizen network

# OGS-specific FDSN endpoints (internal servers)
OGS_CLIENT_STR = "http://158.110.30.217:8080"  # OGS main FDSN server
COLLALTO_CLIENT_STR = "http://scp-srv.core03.ogs.it:8080"  # Collalto array

# OGS-specific stations to reject (e.g., noisy or unreliable stations)
OGS_REJECT_STATIONS = ["SP", "OL", "ED"]

# =============================================================================
# DEFAULT CLIENT PRIORITY LIST
# =============================================================================
# Ordered list of FDSN clients to query (first available wins)

OGS_CLIENTS_DEFAULT = [
    OGS_CLIENT_STR,                      # OGS internal (highest priority)
    INGV_CLIENT_STR,                     # Italian national network
    GFZ_CLIENT_STR,                      # German stations in region
    IRIS_CLIENT_STR,                     # Global backup
    ETH_CLIENT_STR,                      # Swiss border stations
    ORFEUS_CLIENT_STR,                   # European federation
    COLLALTO_CLIENT_STR                  # Collalto dense array
]

# =============================================================================
# DATAFRAME HEADER DEFINITIONS
# =============================================================================
# Predefined column lists for creating consistent DataFrames

CATEGORY_STR = "CATEGORY"              # Category column name
HEADER_STR = "HEADER"                  # Header identifier

# Model configuration header (3 columns)
HEADER_MODL = [MODEL_STR, WEIGHT_STR, THRESHOLD_STR]

# File system tracking header (6 columns)
HEADER_FSYS = [FILENAME_STR, MODEL_STR, WEIGHT_STR, TIME_STR, NETWORK_STR,
               STATION_STR]

# Manual pick data header (5 columns)
HEADER_MANL = [INDEX_STR, TIME_STR, PHASE_STR, STATION_STR, GROUPS_STR]

# Predicted pick header (model info + pick info)
HEADER_PRED = HEADER_MODL + HEADER_MANL

# Station metadata header (5 columns)
HEADER_SNSR = [STATION_STR, LATITUDE_STR, LONGITUDE_STR, DEPTH_STR,
               TIME_STR]

# Statistics header (model info + thresholds)
HEADER_STAT = [MODEL_STR, WEIGHT_STR, STAT_STR] + THRESHOLDS

# Sorting priority for prediction DataFrames
SORT_HIERARCHY_PRED = [MODEL_STR, WEIGHT_STR, INDEX_STR, TIME_STR]

# =============================================================================
# SPECULATIVE/EXPERIMENTAL CONSTANTS
# =============================================================================
# Values used for capacity estimation and histogram binning

MAX_PICKS_YEAR = 1e6                   # Maximum expected picks per year
NUM_BINS = 41                          # Default histogram bin count

# =============================================================================
# OGS STUDY REGION DEFINITIONS
# =============================================================================
# Geographic boundaries for the OGS monitoring area in NE Italy

# Polygon vertices defining the OGS operational region (lon, lat pairs)
# Used for filtering events to the region of interest
OGS_POLY_REGION = [
    (10.0, 45.5),                      # SW corner (Trentino)
    (10.0, 46.5),                      # NW corner (Alto Adige)
    (11.5, 47.0),                      # N edge (Austria border)
    (12.5, 47.0),                      # NE corner (Austria)
    (14.5, 46.5),                      # E edge (Slovenia)
    (14.5, 45.5),                      # SE corner (Friuli-Venezia Giulia)
    (12.5, 44.5),                      # S edge (Emilia-Romagna)
    (11.5, 44.5)                       # SW return (Veneto/Emilia)
]
"""
Polygon vertices defining the OGS operational region in NE Italy:
- (10.0, 45.5): Southwest corner near Trentino
- (10.0, 46.5): Northwest corner near Alto Adige (South Tyrol)
- (11.5, 47.0): Northern edge along Austria border
- (12.5, 47.0): Northeast corner in Austria
- (14.5, 46.5): Eastern edge along Slovenia border
- (14.5, 45.5): Southeast corner in Friuli-Venezia Giulia
- (12.5, 44.5): Southern edge along Emilia-Romagna
- (11.5, 44.5): Southwest return point near Veneto/Emilia border
"""

# Bounding box for the extended study region
# [lon_min, lon_max, lat_min, lat_max]
# Slightly larger than the polygon to include border areas
OGS_STUDY_REGION = [9.5, 15.0, 44.3, 47.5]
"""
Bounding box for OGS study region: [lon_min, lon_max, lat_min, lat_max]
- lon_min: 9.5 (western boundary)
- lon_max: 15.0 (eastern boundary)
- lat_min: 44.3 (southern boundary)
- lat_max: 47.5 (northern boundary)

This box encompasses the polygon defined by OGS_POLY_REGION and includes border
areas of interest in NE Italy, Austria, Slovenia, and Croatia.
"""

# Place name strings
OGS_ITALY_STR = "Italy"                # Country identifier
DESCRIPTION_STR = "Description"        # Description field label


# =============================================================================
# OGS EVENT LABEL FORMAT
# =============================================================================
# Template for constructing event category labels from components

OGS_LABEL_CATEGORY = "{GEO_ZONE_STR}{EVENT_TYPE_STR}{EVENT_LOCALIZATION_STR}"

# =============================================================================
# GEOGRAPHIC ZONE CODE MAPPING
# =============================================================================
# Single-letter codes used in OGS catalog to identify geographic regions

OGS_GEO_ZONES = {
    "A": "Alto Adige",                 # Northern Italy (South Tyrol)
    "C": "Croatia",                    # Croatia (cross-border events)
    "E": "Emilia",                     # Emilia region
    "F": "Friuli",                     # Friuli region (main OGS focus)
    "G": "Venezia Giulia",             # Venezia Giulia region
    "L": "Lombardia",                  # Lombardy region
    "O": "Austria",                    # Austria (cross-border events)
    "R": "Romagna",                    # Romagna region
    "S": "Slovenia",                   # Slovenia (cross-border events)
    "T": "Trentino",                   # Trentino region
    "V": "Veneto"                      # Veneto region
}
"""
===============================================================================
GEOGRAPHIC ZONE CODE MAPPING
===============================================================================
Single-letter codes used in OGS catalog to identify geographic regions:
- A: Alto Adige (South Tyrol)
- C: Croatia (cross-border events)
- E: Emilia region
- F: Friuli region (main OGS focus)
- G: Venezia Giulia region
- L: Lombardy region
- O: Austria (cross-border events)
- R: Romagna region
- S: Slovenia (cross-border events)
- T: Trentino region
- V: Veneto region
"""

# =============================================================================
# EVENT TYPE CODE MAPPING
# =============================================================================
# Single-letter codes used in OGS catalog to classify event types

OGS_EVENT_TYPES = {
    "B": EVENT_BOMB_STR,               # Military detonation (historical)
    "E": EVENT_EXPLD_STR,              # Industrial explosion/quarry blast
    "F": EVENT_LNDSLD_STR,             # Landslide-induced seismic event
    "L": EVENT_LOCAL_EQ_STR,           # Local tectonic earthquake
    "U": EVENT_UNKNOWN_STR             # Unknown/unclassified source
}
"""
===============================================================================
EVENT TYPE CODE MAPPING
===============================================================================
Single-letter codes used in OGS catalog to classify event types:
- B: Military detonation (historical)
- E: Industrial explosion/quarry blast
- F: Landslide-induced seismic event
- L: Local tectonic earthquake
- U: Unknown/unclassified source
"""

# =============================================================================
# CATALOG OUTPUT HEADER DEFINITIONS
# =============================================================================
# Standard column order for event and pick output files

# Event catalog header (8 columns: ID, time, location, uncertainties, gap)
HEADER_EVENTS = [INDEX_STR, TIME_STR, LATITUDE_STR, LONGITUDE_STR,
                 DEPTH_STR, ERH_STR, ERZ_STR, GAP_STR]
"""
EVENT CATALOG HEADER DEFINITIONS\n
Standard column order for event output files:
- idx: Unique event identifier
- time: Origin time of the event
- latitude: Event latitude in degrees
- longitude: Event longitude in degrees
- depth: Event depth in kilometers
- ERH: Maximum horizontal uncertainty in kilometers
- ERZ: Vertical uncertainty in kilometers
- GAP: Azimuthal gap in degrees (measure of station coverage)
"""

# Pick catalog header (7 columns: ID, time, phase info, quality)
HEADER_PICKS = [INDEX_STR, TIME_STR, PHASE_STR, STATION_STR, ONSET_STR,
                POLARITY_STR, WEIGHT_STR]
"""
PICK CATALOG HEADER DEFINITIONS\n
Standard column order for pick output files:
- idx: Unique pick identifier
- time: Pick arrival time
- phase: Seismic phase type (e.g., P, S)
- station: Station code where pick was made
- onset: Onset quality (I=impulsive, E=emergent)
- polarity: First motion polarity (U=up, D=down)
- weight: Pick weight (0-4, with 0 being most precise)
"""

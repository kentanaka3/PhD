"""
===============================================================================
OGS Constants Module - Shared Configuration Values and Labels
===============================================================================

OVERVIEW:
This module defines shared constants and configuration defaults for the OGS
seismic data processing pipeline. This module provides:

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

  5. OGS REGION DEFINITIONS
    - Geographic polygon boundaries for the OGS study area
    - Geographic zone codes (Friuli, Veneto, Slovenia, etc.)
    - Event type classifications

  6. FDSN CLIENT ENDPOINTS
    - URLs for INGV, IRIS, GFZ, OGS, and other data centers
    - Default client priority list

ARCHITECTURE:
  ``ogsconstants`` supplies formats, extensions, headers, colors, thresholds,
  region definitions, endpoint identifiers, and mutable MPI/GPU defaults to
  the other OGS modules. It defines no functions or classes.

USAGE:
  from OGS.src.ogsconstants import (
    PWAVE, SWAVE,           # Phase identifiers
    DATE_FMT, TIME_FMT,     # Format strings
  )

DEPENDENCIES:
  - Python standard library: os, pathlib, datetime.timedelta

AUTHORS:
  - 健
  - Istituto Nazionale di Oceanografia e di Geofisica Sperimentale (OGS)
    Centro di Ricerche Sismologiche (CRS)
  - Università degli Studi di Trieste (UniTS)
    Dipartimento di Matematica, Informatica e Geoscienze (MIGe)
    Applied Data Science and Artificial Intelligence (ADSAI)
  - Terabit Network for Research and Academic Big Data in Italy (TeRABIT)
    Consorzio Interuniversitario del Nord-Est per il Calcolo Automatico (CINECA)

===============================================================================
"""

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
from pathlib import Path                  # Object-oriented filesystem paths
import os                                 # Operating system interfaces
from datetime import timedelta as td      # Time handling

# =============================================================================
# MODULE-LEVEL CONFIGURATION
# =============================================================================

# Reference to this file's path for deriving package-local paths
THIS_FILE = Path(__file__)

DEFAULT_CORES_COUNT = int(
    os.environ.get("CORES", os.environ.get("SLURM_CPUS_PER_TASK", "1"))
)

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
# DATE/TIME FORMAT CONSTANTS
# =============================================================================
# Standard format strings for parsing and formatting dates/times throughout
# the OGS pipeline. Uses Python strftime/strptime conventions.

DATE_STD = "YYYYMMDD"                 # Standard date representation string
DATE_JUL = "YYYYJJJ"                  # Julian date representation string
DATE_FMT = "%Y-%m-%d"                 # ISO date format (2022-01-15)
TIME_FMT = "%H%M%S"                   # Compact time format (143052)
YYMMDD_FMT = "%y%m%d"                 # 2-digit year date (220115)
YYYYMMDD_FMT = "%Y%m%d"               # 4-digit year date (20220115)
DATETIME_FMT = YYMMDD_FMT + TIME_FMT  # Combined datetime (220115143052)
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
# Project mapping of H71 pick weight classes to offsets in seconds.
H71_OFFSET: dict[int, float] = {
    0: 0.01,
    1: 0.04,
    2: 0.2,
    3: 1,
    4: 5,
}
"""
===============================================================================
H71 WEIGHT CONVERSION TABLE
===============================================================================
Hypo71 standard weight codes mapped to uncertainty in seconds
These represent picking precision: 0 = most precise, 4 = least precise
Weight | Uncertainty (sec) | Interpretation
-------|-------------------|----------------
  0    |       0.01        | Impulsive onset, very clear
  1    |       0.04        | Clear onset
  2    |       0.2         | Fairly clear onset
  3    |       1.0         | Emergent onset
  4    |       5.0         | Poor quality pick
These configured values do not establish measured picking uncertainty or
onset quality.
"""

# =============================================================================
# EVENT MATCHING TOLERANCES
# =============================================================================
# Thresholds for matching detected events to catalog events

EVENT_TIME_OFFSET = td(seconds=2)
"""
Max time difference for event matching: 2 seconds\n
"""
EVENT_DIST_OFFSET = 8
"""
Max spatial distance for event matching: 8 km\n
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
NONE_STR = "None"                       # String representation of None

# =============================================================================
# PIPELINE COMPONENT IDENTIFIERS
# =============================================================================
# String identifiers for various pipeline stages and components

DEFAULT_PICKER = "SeisBenchPicker"      # ML-based phase picker identifier
DEFAULT_ASSOCIATOR = "GammaAssociator"  # GaMMA phase associator identifier
DURATION_STR = "duration"               # Duration field name
SECONDS_STR = "seconds"                 # Seconds unit label

# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================
# String constants for confusion matrix and performance evaluation metrics

TP_STR = "TP"                           # True Positive count
MH_STR = "MH"                           # Matched count
FP_STR = "FP"                           # False Positive count
PS_STR = "PS"                           # Proposed count
SP_STR = "SP"                           # Skipped count
FN_STR = "FN"                           # False Negative count
MS_STR = "MS"                           # Missed count
SM_STR = "SM"                           # Skimmed count
SW_STR = "SW"                           # Swapped count
TN_STR = "TN"                           # True Negative count

NETCOLOR_STR = "NC"                     # Network color for plotting
STACOLOR_STR = "SC"                     # Station color for plotting

# =============================================================================
# SEISMIC PHASE IDENTIFIERS
# =============================================================================
# Standard phase type labels for P and S waves

PWAVE = "P"                             # Primary (compressional) wave
SWAVE = "S"                             # Secondary (shear) wave

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
# COLOR PALETTE FOR SCIENTIFIC VISUALIZATION
# =============================================================================
# Hex color codes for consistent visualization across the project

# --- Tier-0: Base / Primary Accent (Standard Markers & Main Traces) ---
MEX_PINK = "#E4007C"                   # Bright pink (accent color)
OGS_BLUE = "#163771"                   # OGS institutional blue (primary)
ALN_GREEN = "#00e468"                  # Bright green (positive/success)
LIP_ORANGE = "#FF8C00"                 # Orange (warning/highlight)
SUN_YELLOW = "#e4da00"                 # Yellow (tertiary accent)

# --- Tier-1: Pastel / Tints (Fills, Density Valleys & Confidence Bands) ---
MEX_PINK_PASTEL = "#F0A8CF"            # Soft carnation pink
OGS_BLUE_PASTEL = "#A0B9E3"            # Soft periwinkle blue
ALN_GREEN_PASTEL = "#A2EBC4"           # Soft seafoam mint green
LIP_ORANGE_PASTEL = "#F5D0A3"          # Soft peach apricot
SUN_YELLOW_PASTEL = "#F4F1AF"          # Soft butter cream yellow

# --- Tier-2: Dark / Shades (Outlines, Text Labels, Axes & High Density) ---
MEX_PINK_DARK = "#8F004E"              # Deep berry wine
OGS_BLUE_DARK = "#0A1E43"              # Deep midnight navy
ALN_GREEN_DARK = "#007A38"             # Deep forest pine green
LIP_ORANGE_DARK = "#995400"            # Deep burnt rust orange
SUN_YELLOW_DARK = "#8F8900"            # Deep golden amber / olive

# Standard color sequence for multi-series plots
PLOT_COLORS = [OGS_BLUE, MEX_PINK, ALN_GREEN, LIP_ORANGE, SUN_YELLOW]

# TODO: Add Tabular data for relational databases for future development

# =============================================================================
# FILE EXTENSION CONSTANTS
# =============================================================================
# String constants for file type extensions (without leading period)

BLT_STR = "blt"                         # Bulletin file format
CSV_STR = "csv"                         # Comma-separated values
DAT_STR = "dat"                         # OGS phase data format
EPS_STR = "eps"                         # Encapsulated PostScript (vector)
HDF5_STR = "hdf5"                       # Hierarchical Data Format 5
HPC_STR = "hpc"                         # HPC-specific format
HPL_STR = "hpl"                         # OGS hypocenter location format
JSON_STR = "json"                       # JavaScript Object Notation
LD_STR = "ld"                           # Linked data format
MOD_STR = "mod"                         # Model/velocity model format
MSEED_STR = "mseed"                     # MiniSEED waveform format
PDF_STR = "pdf"                         # Portable Document Format
PICKLE_STR = "pkl"                      # Python pickle serialization
PNG_STR = "png"                         # Portable Network Graphics (raster)
PRT_STR = "prt"                         # Print/report file format
PUN_STR = "pun"                         # OGS punch card output format
QML_STR = "qml"                         # QuakeML seismic data exchange
TORCH_STR = "pt"                        # PyTorch model weights
TXT_STR = "txt"                         # Plain text format
XML_STR = "xml"                         # Extensible Markup Language

# =============================================================================
# FILE EXTENSION CONSTANTS (WITH PERIOD)
# =============================================================================
# Full file extensions including the leading period for direct use

BLT_EXT = PERIOD_STR + BLT_STR          # .blt
CSV_EXT = PERIOD_STR + CSV_STR          # .csv
DAT_EXT = PERIOD_STR + DAT_STR          # .dat
EPS_EXT = PERIOD_STR + EPS_STR          # .eps
HDF5_EXT = PERIOD_STR + HDF5_STR        # .hdf5
HPC_EXT = PERIOD_STR + HPC_STR          # .hpc
HPL_EXT = PERIOD_STR + HPL_STR          # .hpl
JSON_EXT = PERIOD_STR + JSON_STR        # .json
LD_EXT = PERIOD_STR + LD_STR            # .ld
MOD_EXT = PERIOD_STR + MOD_STR          # .mod
MSEED_EXT = PERIOD_STR + MSEED_STR      # .mseed
PDF_EXT = PERIOD_STR + PDF_STR          # .pdf
PICKLE_EXT = PERIOD_STR + PICKLE_STR    # .pkl
PNG_EXT = PERIOD_STR + PNG_STR          # .png
PRT_EXT = PERIOD_STR + PRT_STR          # .prt
PUN_EXT = PERIOD_STR + PUN_STR          # .pun
QML_EXT = PERIOD_STR + QML_STR          # .qml
TORCH_EXT = PERIOD_STR + TORCH_STR      # .pt
TXT_EXT = PERIOD_STR + TXT_STR          # .txt
XML_EXT = PERIOD_STR + XML_STR          # .xml

# =============================================================================
# WAVEFORM FILE NAMING FORMAT
# =============================================================================
# SEED channel ID followed by start/end timestamps (YYYYMMDDTHHMMSSZ).
# LOCATION may be empty; EXT does not include the leading period.
PRC_FMT = (
    "{NETWORK}.{STATION}.{LOCATION}.{CHANNEL}__{BEGDT}__{ENDDT}.{EXT}"
)

# =============================================================================
# ML MODEL IDENTIFIERS
# =============================================================================
# Names of supported machine learning models for phase picking

EQTRANSFORMER_STR = "EQTransformer"     # EQTransformer deep learning model
PHASENET_STR = "PhaseNet"               # PhaseNet deep learning model

# =============================================================================
# OGS PROJECTION SYSTEM
# =============================================================================
# Stereographic projection parameters for local coordinate transformation
# Uses PROJ4 format string with placeholder for center coordinates

OGS_PROJECTION = "+proj=sterea +lon_0={lon} +lat_0={lat} +units=km"

# Plotting threshold: events at or above this magnitude use star markers.
OGS_MAX_MAGNITUDE = 4.0
OGS_MAGNITUDE_SIZE = {
    # Magnitude interval [lower, upper) : (marker size, color)
    (-1., 0.): (10, MEX_PINK_DARK),
    (0., 1.): (20, OGS_BLUE),
    (1., 2.): (40, ALN_GREEN),
    (2., 3.): (80, SUN_YELLOW_DARK),
    (3., OGS_MAX_MAGNITUDE): (160, LIP_ORANGE_DARK),
}

# =============================================================================
# DATAFRAME COLUMN NAME CONSTANTS
# =============================================================================
# Standardized column names for pandas DataFrames throughout the pipeline

# Pick-related columns
IDX_PICKS_STR = "index"                 # Pick index identifier
GROUPS_STR = "group"                    # Group/cluster identifier
TIME_STR = "time"                       # Timestamp column
STATION_STR = "station"                 # Station identifier
PHASE_STR = "phase"                     # Phase type (P or S)
PROBABILITY_STR = "probability"         # ML confidence score
AMPLITUDE_STR = "amplitude"             # Waveform amplitude
EPICENTRAL_DISTANCE_STR = "epicentral_distance"  # Distance from epicenter
# Event depth (km); inventory elevation (m)
DEPTH_STR = "depth"
STATION_ML_STR = "station_ML"           # Station-specific magnitude
NUMBER_P_PICKS_STR = "number_p_picks"   # Count of P-wave picks
NUMBER_S_PICKS_STR = "number_s_picks"   # Count of S-wave picks
NUMBER_P_AND_S_PICKS_STR = "number_p_and_s_picks"  # Count of P+S picks

# Magnitude-related columns
ML_STR = "ML"                           # Local magnitude
ML_MEDIAN_STR = "ML_median"             # Median local magnitude
ML_UNC_STR = "ML_unc"                   # Magnitude uncertainty
ML_STATIONS_STR = "ML_stations"         # Number of stations for ML

# Duration magnitude quality columns
MD_STR = "MD"                           # Duration magnitude
MD_MEDIAN_STR = "MD_median"             # Median duration magnitude
MD_UNC_STR = "MD_unc"                   # Duration magnitude uncertainty
MD_STATIONS_STR = "MD_stations"         # Number of stations for MD

# Continuous Hypo71 solution magnitude (Column 6)
HYPO71_MAG_STR = "hypo71_mag"
HPL_AUX_FLOAT_STR = "hpl_aux_float"     # auxiliary float 0.00
VELOCITY_MODEL_ID_STR = "vel_model_id"  # velocity model ID
PHASES_USED_STR = "num_phases"          # number of phases used in solution
MEAN_RESIDUAL_STR = "mean_residual"     # mean travel-time residual
STD_RESIDUAL_STR = "std_residual"       # standard deviation of residuals
M3_STATIONS_STR = "m3_stations"         # 3rd magnitude station count
M3_MAGNITUDE_STR = "m3_magnitude"       # 3rd magnitude value
M3_UNC_STR = "m3_unc"                   # 3rd magnitude uncertainty

# Event identification columns
IDX_EVENTS_STR = "idx"                  # Event index identifier
LEGACY_ID_STR = "legacy_id"             # Legacy catalog event identifier
METADATA_STR = "metadata"               # Metadata container column

# Geographic coordinate columns
LONGITUDE_STR = "longitude"             # Longitude (degrees)
LATITUDE_STR = "latitude"               # Latitude (degrees)
ELEVATION_STR = "elevation"             # Elevation in meters
X_COORD_STR = "x(km)"                   # X coordinate in kilometers (local)
Y_COORD_STR = "y(km)"                   # Y coordinate in kilometers (local)
Z_COORD_STR = "z(km)"                   # Z coordinate in kilometers (depth)

# Additional event attributes
MAGNITUDE_L_STR = "ML"                  # Local magnitude type
MAGNITUDE_D_STR = "MD"                  # Duration magnitude type
VELOCITY_STR = "vel"                    # Velocity model reference

# Clustering method identifiers
GAUSS_MIX_MODEL_STR = "GMM"             # Gaussian Mixture Model
BAYES_GAUSS_MIX_MODEL_STR = "B" + GAUSS_MIX_MODEL_STR  # Bayesian GMM

# =============================================================================
# CONFIGURATION AND PATH COLUMN NAMES
# =============================================================================
# Column names for configuration DataFrames and file management

WAVEFORM_STR = "waveform"               # Waveform data reference
STATION_STR = "station"                 # Station metadata reference

# Comparison labels for base vs. target analysis
BASE_STR = "Base"                       # Reference-side label
TARGET_STR = "Target"                   # Comparison-side label

# =============================================================================
# UPPERCASE COLUMN NAMES FOR HEADERS
# =============================================================================
# Uppercase versions for header rows and configuration files

EVENT_STR = "EVENT"                     # Event identifier (uppercase)
WEIGHT_STR = "WEIGHT"                   # Weight/pretrained weights
FILENAME_STR = "FILENAME"               # Filename column
NETWORK_STR = "NETWORK"                 # Seismic network code
CHANNEL_STR = "CHANNEL"                 # Channel code
DATE_STR = "DATE"                       # Date column

# =============================================================================
# LABELLED DATA COLUMN NAMES (P AND S WAVE)
# =============================================================================
# Column names for manually labeled phase data with P and S wave attributes

# P-wave pick attributes
P_TIME_STR = "P_TIME"                   # P-wave arrival time
P_TYPE_STR = "P_TYPE"                   # P-wave type (e.g., Pg, Pn)
P_ONSET_STR = "P_ONSET"                 # P-wave onset quality (I/E)
P_POLARITY_STR = "P_POLARITY"           # P-wave first motion (U/D)
P_WEIGHT_STR = "P_WEIGHT"               # P-wave pick weight (0-4)

# S-wave pick attributes
S_TIME_STR = "S_TIME"                   # S-wave arrival time
S_TYPE_STR = "S_TYPE"                   # S-wave type (e.g., Sg, Sn)
S_ONSET_STR = "S_ONSET"                 # S-wave onset quality
S_POLARITY_STR = "S_POLARITY"           # S-wave polarity (if measurable)
S_WEIGHT_STR = "S_WEIGHT"               # S-wave pick weight

# =============================================================================
# EVENT QUALITY INDICATORS
# =============================================================================
# Column names for event location quality metrics

NO_STR = "number_picks"                 # Number of picks used
GAP_STR = "azimuthal_gap"               # Azimuthal gap in degrees
DMIN_STR = "dmin"                       # Distance to nearest station
RMS_STR = "rms"                         # RMS travel time residual
ERH_STR = "max_horizontal_uncertainty"  # Horizontal error (km)
ERZ_STR = "vertical_uncertainty"        # Vertical error (km)
ERT_STR = "weight"                      # Overall location weight
QM_STR = "qm"                           # Quality metric
ONSET_STR = "onset"                     # Onset type (I=impulsive, E=emergent)
POLARITY_STR = "polarity"               # First motion polarity (U/D)

# =============================================================================
# OGS GEOGRAPHIC CLASSIFICATION
# =============================================================================
# Column names and values for OGS regional earthquake classification

GEO_ZONE_STR = "GEOZONE"                # Geographic zone code column
EVENT_TYPE_STR = "E_TYPE"               # Event type column

# Event type classification values
EVENT_LOCAL_EQ_STR = "local_eq"         # Local tectonic earthquake
EVENT_REGIONAL_STR = "regional"         # Regional tectonic earthquake
EVENT_EXPLD_STR = "explosion"           # Industrial explosion
EVENT_BOMB_STR = "bomb"                 # Military detonation (historical)
EVENT_LNDSLD_STR = "landslide"          # Landslide-induced event
EVENT_UNKNOWN_STR = "UNKNOWN"           # Unknown/unclassified event

# Location metadata
EVENT_LOCALIZATION_STR = "E_LOC"        # Localization method/status
LOC_NAME_STR = "LOC_NAME"               # Location place name
NOTES_STR = "NOTES"                     # Analyst notes field

# =============================================================================
# PRETRAINED MODEL WEIGHT IDENTIFIERS
# =============================================================================
# Names of pretrained weight variants for SeisBench models

ADRIAARRAY_STR = "adriaarray"           # Trained on AdriaArray data
INSTANCE_STR = "instance"               # Trained on INSTANCE dataset
ORIGINAL_STR = "original"               # Original author weights
SCEDC_STR = "scedc"                     # Southern California Earthquake DC
STEAD_STR = "stead"                     # STanford EArthquake Dataset

# =============================================================================
# FDSN WEB SERVICE CLIENT IDENTIFIERS
# =============================================================================
# Standard FDSN data center names and OGS-specific endpoints

# Major international FDSN data centers
BGR_CLIENT_STR = "BGR"                  # BGR
EIDA_CLIENT_STR = "EIDA"                # EIDA federator / routing service
ETH_CLIENT_STR = "ETH"                  # Swiss Seismological Service
GEONET_CLIENT_STR = "GEONET"            # New Zealand GeoNet
GFZ_CLIENT_STR = "GFZ"                  # German Research Centre, Potsdam
INGV_CLIENT_STR = "INGV"                # Italian National Institute
IRIS_CLIENT_STR = "IRIS"                # US IRIS Data Management Center
LMU_CLIENT_STR = "LMU"                  # Ludwig Maximilian University
ORFEUS_CLIENT_STR = "https://www.orfeus-eu.org"  # European ORFEUS Data Center
RASPISHAKE_CLIENT_STR = "RASPISHAKE"    # Raspberry Shake citizen network
RESIF_CLIENT_STR = "RESIF"              # French RESIF network

# Catalog-only FDSN services (No waveform dataselect)
EMSC_CLIENT_STR = "EMSC"                # Euro-Mediterranean Seismological
USGS_CLIENT_STR = "USGS"                # US Geological Survey

# OGS-specific FDSN endpoints (internal servers)
OGS_CLIENT_STR = "http://158.110.30.217:8080"
# Collalto array (deprecated/unreachable)
COLLALTO_CLIENT_STR = "http://scp-srv.core03.ogs.it:8080"

# OGS-specific stations to reject (e.g., noisy or unreliable stations)
OGS_REJECT_STATIONS = ["SP", "OL", "ED", "VNZE"]

# =============================================================================
# DEFAULT CLIENT PRIORITY LIST
# =============================================================================
# Ordered default list of FDSN clients; query behavior belongs to callers.

OGS_CLIENTS_DEFAULT = [
    # Tier-0: OGS internal servers (highest priority)
    OGS_CLIENT_STR,                     # OGS internal (highest priority)
    INGV_CLIENT_STR,                    # Italian national network
    # Tier-1: Major international FDSN data centers
    EIDA_CLIENT_STR,                    # EIDA federator / routing service
    ETH_CLIENT_STR,                     # Swiss Seismological Service
    LMU_CLIENT_STR,                     # Ludwig Maximilian University
    ORFEUS_CLIENT_STR,                  # European ORFEUS Data Center
    RASPISHAKE_CLIENT_STR,              # Raspberry Shake citizen network
    RESIF_CLIENT_STR,                   # French RESIF network
]

# =============================================================================
# SPECULATIVE/EXPERIMENTAL CONSTANTS
# =============================================================================
# Values used for capacity estimation and histogram binning

MAX_PICKS_YEAR = 1e6                    # Maximum expected picks per year
MAX_EVENTS_YEAR = 1e4                   # Maximum expected events per year
NUM_BINS: int = 41                      # Default histogram bin count

# =============================================================================
# OGS STUDY REGION DEFINITIONS
# =============================================================================
# Geographic boundaries for the OGS monitoring area in NE Italy

# Polygon vertices defining the OGS operational region (lon, lat pairs)
# Used for filtering events to the region of interest
OGS_POLY_REGION = [
    (10.0, 45.5),                       # SW corner (Trentino)
    (10.0, 46.5),                       # NW corner (Alto Adige)
    (11.5, 47.0),                       # N edge (Austria border)
    (12.5, 47.0),                       # NE corner (Austria)
    (14.5, 46.5),                       # E edge (Slovenia)
    (14.5, 45.5),                       # SE corner (Friuli-Venezia Giulia)
    (12.5, 44.5),                       # S edge (Emilia-Romagna)
    (11.5, 44.5)                        # SW return (Veneto/Emilia)
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
OGS_STUDY_REGION = (9.5, 15.0, 44.3, 47.5)
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
OGS_ITALY_STR = "Italy"                 # Country identifier

OGS_TIMEOUT = 30                        # Timeout (seconds)
OGS_RETRY = 3                           # Retry attempts

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
    "A": "Alto Adige",                  # Northern Italy (South Tyrol)
    "C": "Croatia",                     # Croatia (cross-border events)
    "E": "Emilia",                      # Emilia region
    "F": "Friuli",                      # Friuli region (main OGS focus)
    "G": "Venezia Giulia",              # Venezia Giulia region
    "L": "Lombardia",                   # Lombardy region
    "O": "Austria",                     # Austria (cross-border events)
    "R": "Romagna",                     # Romagna region
    "S": "Slovenia",                    # Slovenia (cross-border events)
    "T": "Trentino",                    # Trentino region
    "V": "Veneto"                       # Veneto region
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
    "B": EVENT_BOMB_STR,                # Military detonation (historical)
    "E": EVENT_EXPLD_STR,               # Industrial explosion/quarry blast
    "F": EVENT_LNDSLD_STR,              # Landslide-induced seismic event
    "L": EVENT_LOCAL_EQ_STR,            # Local tectonic earthquake
    "R": EVENT_REGIONAL_STR,            # Regional tectonic earthquake
    "U": EVENT_UNKNOWN_STR              # Unknown/unclassified source
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
- R: Regional tectonic earthquake
- U: Unknown/unclassified source
"""

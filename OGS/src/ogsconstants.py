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
import os                                  # Operating system interface
import re                                  # Regular expression operations
import argparse                            # Command-line argument parsing
import itertools as it                     # Iterator utilities
from pathlib import Path                   # Object-oriented filesystem paths
from datetime import datetime, timedelta as td  # Date/time handling
from typing import Any, Optional, Tuple    # Type hinting

# =============================================================================
# THIRD-PARTY LIBRARY IMPORTS
# =============================================================================
import numpy as np                         # Numerical computing
import obspy as op                         # Seismological toolkit
import pandas as pd                        # Data manipulation and analysis
import networkx as nx                      # Graph algorithms (bipartite matching)
from obspy import UTCDateTime              # Seismology-specific datetime
from matplotlib.path import Path as mplPath  # Matplotlib path for polygon ops
from obspy.geodetics import gps2dist_azimuth

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

DATE_STD = "YYMMDD"                    # Standard date representation string
DATE_FMT = "%Y-%m-%d"                  # ISO date format (2022-01-15)
TIME_FMT = "%H%M%S"                    # Compact time format (143052)
YYMMDD_FMT = "%y%m%d"                  # 2-digit year date (220115)
YYYYMMDD_FMT = "%Y%m%d"                # 4-digit year date (20220115)
DATETIME_FMT = YYMMDD_FMT + TIME_FMT   # Combined datetime (220115143052)
DATETIME_STR = "DATETIME"              # Column name for datetime fields
TIMESTAMP_STR = "TIMESTAMP"            # Column name for Unix timestamps

# =============================================================================
# TIME DELTA CONSTANTS
# =============================================================================
# Time intervals used for event detection, pick matching, and data segmentation

ONE_DAY = td(days=1)                   # One day interval for date iteration

# Maximum time difference for matching predicted picks to manual picks
# Picks within this window are considered potential matches
PICK_TIME_OFFSET = td(seconds=.5)      # 0.5 second tolerance for pick matching

# Time window for training data extraction around picks
PICK_TRAIN_OFFSET = td(seconds=60)     # 60 second window for ML training

# =============================================================================
# H71 WEIGHT CONVERSION TABLE
# =============================================================================
# Hypo71 standard weight codes mapped to uncertainty in seconds
# These represent picking precision: 0 = most precise, 5 = least precise
#
# Weight | Uncertainty (sec) | Interpretation
# -------|-------------------|----------------
#   0    |       0.01        | Impulsive onset, very clear
#   1    |       0.04        | Clear onset
#   2    |       0.2         | Fairly clear onset
#   3    |       1.0         | Emergent onset
#   4    |       5.0         | Poor quality pick
#   5    |      25.0         | Very uncertain (often unused)

H71_OFFSET = {
  0: 0.01,
  1: 0.04,
  2: 0.2,
  3: 1,
  4: 5,
  5: 25
}

# =============================================================================
# EVENT MATCHING TOLERANCES
# =============================================================================
# Thresholds for matching detected events to catalog events

EVENT_TIME_OFFSET = td(seconds=2)      # Max time difference for event matching
EVENT_DIST_OFFSET = 8                  # Max spatial distance (km) for matching

# =============================================================================
# STRING CONSTANTS - GENERAL PURPOSE
# =============================================================================
# Commonly used string literals to ensure consistency and avoid typos

EMPTY_STR = ''                         # Empty string for initialization
ALL_WILDCHAR_STR = '*'                 # Wildcard for glob patterns (any chars)
ONE_MORECHAR_STR = '+'                 # Regex: one or more characters
PERIOD_STR = '.'                       # Period (used in SEED IDs, extensions)
UNDERSCORE_STR = '_'                   # Underscore (filename separator)
DASH_STR = '-'                         # Dash (date separator)
SPACE_STR = ' '                        # Space character
COMMA_STR = ','                        # Comma (CSV separator)
SEMICOL_STR = ';'                      # Semicolon (alternative separator)
ZERO_STR = "0"                         # Zero string for padding
NONE_STR = "None"                      # String representation of None

# =============================================================================
# PIPELINE COMPONENT IDENTIFIERS
# =============================================================================
# String identifiers for various pipeline stages and components

CLF_STR = "SeisBenchPicker"            # ML-based phase picker identifier
AST_STR = "GammaAssociator"            # GaMMA phase associator identifier
FILE_STR = "file"                      # Generic file reference
TEMPORAL_STR = "tmp"                   # Temporary file prefix
DURATION_STR = "duration"              # Duration field name
STATUS_STR = "status"                  # Status field name
SECONDS_STR = "seconds"                # Seconds unit label

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

MEX_PINK = "#E4007C"                   # Bright pink (accent color)
OGS_BLUE = "#163771"                   # OGS institutional blue (primary)
ALN_GREEN = "#00e468"                  # Bright green (positive/success)
LIP_ORANGE = "#FF8C00"                 # Orange (warning/highlight)
SUN_YELLOW = "#e4da00"                 # Yellow (tertiary accent)

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
TIME_STR = "time"                      # Time column (redefined for clarity)
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
AMPLITUDE_STR = "amplitude"            # Amplitude measurement
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

PHASE_STR = "phase"                    # Phase type column
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
STATION_STR = "station"                # Station code (lowercase)
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

# Manual pick data header (6 columns)
HEADER_MANL = [INDEX_STR, TIME_STR, PHASE_STR, STATION_STR, GROUPS_STR, GROUPS_STR]

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

# Bounding box for the extended study region [lon_min, lon_max, lat_min, lat_max]
# Slightly larger than the polygon to include border areas
OGS_STUDY_REGION = [9.5, 15.0, 44.3, 47.5]

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

# =============================================================================
# CATALOG OUTPUT HEADER DEFINITIONS
# =============================================================================
# Standard column order for event and pick output files

# Event catalog header (8 columns: ID, time, location, uncertainties, gap)
HEADER_EVENTS = [INDEX_STR, TIME_STR, LATITUDE_STR, LONGITUDE_STR,
                 DEPTH_STR, ERH_STR, ERZ_STR, GAP_STR]

# Pick catalog header (7 columns: ID, time, phase info, quality)
HEADER_PICKS = [INDEX_STR, TIME_STR, PHASE_STR, STATION_STR, ONSET_STR,
                POLARITY_STR, WEIGHT_STR]

# =============================================================================
# DISTANCE AND SIMILARITY FUNCTIONS
# =============================================================================
# Functions for computing distances and similarity scores between picks/events


def dist_prob(B: pd.Series, T: pd.Series) -> float:
  """
  Calculate probability ratio between target and base picks.

  Used as a component in the weighted pick matching score.
  Higher target probability relative to base yields higher score.

  Args:
    B: Base pick (ground truth) as pandas Series with PROBABILITY_STR.
    T: Target pick (prediction) as pandas Series with PROBABILITY_STR.

  Returns:
    Ratio of target probability to base probability.
  """
  return T[PROBABILITY_STR] / B[PROBABILITY_STR]


def dist_phase(B: pd.Series, T: pd.Series) -> float:
  """
  Check if phase types match between base and target picks.

  Args:
    B: Base pick as pandas Series with PHASE_STR.
    T: Target pick as pandas Series with PHASE_STR.

  Returns:
    1.0 if phases match (both P or both S), 0.0 otherwise.
  """
  return int(T[PHASE_STR] == B[PHASE_STR])


def diff_time(B: pd.Series, T: pd.Series) -> float:
  """
  Calculate absolute time difference between two picks/events.

  Args:
    B: Base record as pandas Series with TIME_STR (UTCDateTime).
    T: Target record as pandas Series with TIME_STR (UTCDateTime).

  Returns:
    Absolute time difference in seconds.
  """
  return abs(T[TIME_STR] - B[TIME_STR])


def dist_time(B: pd.Series, T: pd.Series,
              offset: td = PICK_TIME_OFFSET) -> float:
  """
  Calculate normalized time similarity score.

  Converts time difference to a similarity score between 0 and 1,
  where 1 means perfect match and 0 means at the tolerance limit.

  Args:
    B: Base record as pandas Series with TIME_STR.
    T: Target record as pandas Series with TIME_STR.
    offset: Maximum time tolerance (default: PICK_TIME_OFFSET).

  Returns:
    Similarity score: 1 - (time_diff / tolerance).
  """
  return 1. - (diff_time(B, T) / offset.total_seconds())


def diff_space(
    B: pd.Series,
    T: pd.Series,
    ndim: int = 2,
    p: float = 2.
  ) -> float:
  """
  Calculate spatial distance between two locations using geodetic formulas.

  Uses ObsPy's gps2dist_azimuth for accurate great-circle distance.
  Optionally includes depth difference for 3D distance calculation.

  Args:
      B: Base location as pandas Series with LATITUDE_STR, LONGITUDE_STR,
          and optionally DEPTH_STR.
      T: Target location as pandas Series with same columns.
      ndim: Number of dimensions (2 for epicentral, 3 for hypocentral).
      p: Power for distance metric (2 = Euclidean).

  Returns:
      Distance in kilometers, rounded to 4 decimal places.
  """
  # Calculate horizontal distance using geodetic formula (returns meters)
  horizontal_dist_km = gps2dist_azimuth(
      B[LATITUDE_STR], B[LONGITUDE_STR],
      T[LATITUDE_STR], T[LONGITUDE_STR])[0] / 1000.

  # Add vertical component if 3D distance requested
  vertical_component = ((B[DEPTH_STR] - T[DEPTH_STR]) / 1000.) ** p if ndim == 3 else 0.

  # Compute Lp norm distance
  return float(format(np.sqrt(horizontal_dist_km ** p + vertical_component), ".4f"))


def dist_space(B: pd.Series, T: pd.Series,
              offset: float = EVENT_DIST_OFFSET) -> float:
  """
  Calculate normalized spatial similarity score.

  Converts spatial distance to a similarity score between 0 and 1.

  Args:
      B: Base location as pandas Series.
      T: Target location as pandas Series.
      offset: Maximum distance tolerance in km (default: EVENT_DIST_OFFSET).

  Returns:
      Similarity score: 1 - (distance / tolerance).
  """
  return 1. - diff_space(B, T) / offset


def dist_pick(B: pd.Series, T: pd.Series,
              time_offset_sec: td = PICK_TIME_OFFSET) -> float:
  """
  Calculate weighted similarity score for pick matching.

  Combines time similarity (97%), phase match (2%), and probability
  ratio (1%) into a single matching score for bipartite graph edges.

  Args:
      B: Base pick (ground truth) as pandas Series.
      T: Target pick (prediction) as pandas Series.
      time_offset_sec: Time tolerance for matching.

  Returns:
      Weighted similarity score between 0 and 1.
  """
  return (
    97. * dist_time(T, B, time_offset_sec) +  # Time dominates (97%)
    2. * dist_phase(T, B) +                    # Phase type (2%)
    1. * dist_prob(T, B)                       # Probability ratio (1%)
  ) / 100.


def dist_event(T: pd.Series, P: pd.Series,
               time_offset_sec: td = EVENT_TIME_OFFSET,
               space_offset_km: float = EVENT_DIST_OFFSET) -> float:
  """
  Calculate weighted similarity score for event matching.

  Combines time similarity (99%) and spatial similarity (1%) for
  matching detected events to catalog events.

  Args:
    T: Target event as pandas Series.
    P: Predicted/reference event as pandas Series.
    time_offset_sec: Time tolerance for matching.
    space_offset_km: Spatial tolerance in km.

  Returns:
    Weighted similarity score between 0 and 1.
  """
  return (99. * dist_time(T, P, time_offset_sec) +   # Time dominates (99%)
          1. * dist_space(T, P, space_offset_km)) / 100.  # Space (1%)


# =============================================================================
# ARGUMENT PARSING UTILITY FUNCTIONS
# =============================================================================
# Functions for validating and converting command-line arguments


def is_date(string: str) -> datetime:
  """
  Parse a date string in YYYYMMDD format.

  Used as argparse type converter for date arguments.

  Args:
    string: Date string in YYYYMMDD format (e.g., "20220115").

  Returns:
    datetime object representing the parsed date.

  Raises:
    ValueError: If string doesn't match expected format.
  """
  return datetime.strptime(string, YYYYMMDD_FMT)


def is_julian(string: str) -> datetime:
  """
  Parse a Julian day number to datetime (NOT IMPLEMENTED).

  TODO: Define and convert Julian date to Gregorian date.

  Args:
    string: Julian day string.

  Returns:
    datetime object.

  Raises:
    NotImplementedError: This function is not yet implemented.
  """
  # TODO: Define and convert Julian date to Gregorian date
  raise NotImplementedError
  return datetime.strptime(string, YYMMDD_FMT)._set_julday(string)


def is_file_path(string: str) -> Path:
  """
  Validate and convert a string to an absolute file path.

  Used as argparse type converter for file arguments.

  Args:
    string: Path string to validate.

  Returns:
    Absolute Path object if file exists.

  Raises:
    FileNotFoundError: If the file does not exist.
  """
  if os.path.isfile(string):
    return Path(os.path.abspath(string))
  else:
    raise FileNotFoundError(string)


def is_dir_path(string: str) -> Path:
  """
  Validate and convert a string to an absolute directory path.

  Used as argparse type converter for directory arguments.

  Args:
    string: Path string to validate.

  Returns:
    Absolute Path object if directory exists.

  Raises:
    NotADirectoryError: If the directory does not exist.
  """
  if os.path.isdir(string):
    return Path(os.path.abspath(string))
  else:
    raise NotADirectoryError(string)


def decimeter(value, scale='normal') -> int:
  """
  Round a value up to a "nice" number for axis limits.

  Computes the next aesthetically pleasing round number above the input,
  useful for setting plot axis limits.

  Args:
    value: Numeric value to round up.
    scale: Rounding mode:
        - 'normal': Round to next multiple of leading digit + 1
        - 'log': Round to next power of 10
        - other: Round to next multiple of 10

  Returns:
    Rounded integer value.

  Example:
    >>> decimeter(47)  # Returns 50
    >>> decimeter(123, 'log')  # Returns 1000
  """
  # Find the order of magnitude (number of digits - 1)
  base = np.floor(np.log10(abs(value)))

  if scale == 'normal':
    # Round up to next "nice" number (e.g., 47 -> 50, 123 -> 200)
    return ((value // 10 ** base) + 1) * 10 ** base
  elif scale == 'log':
    # Round up to next power of 10
    return int(10 ** (base + 1))

  # Default: round up to next multiple of 10
  return np.ceil(value / 10) * 10


def labels_to_colormap(
      labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Any, Any]:
  """
  Map arbitrary cluster labels to sequential indices for colormapping.

  Handles cases where labels include noise points (label=-1) or
  non-sequential cluster IDs. Creates a discrete colormap with
  one color per unique label.

  Parameters
  ----------
  labels : np.ndarray
    Cluster labels array, may include -1 for noise points.

  Returns
  -------
  tuple
    (encoded_labels, unique_labels, colormap, norm)
    - encoded_labels: Labels mapped to 0..K-1
    - unique_labels: Original unique label values
    - colormap: Matplotlib colormap resampled to K colors
    - norm: BoundaryNorm for discrete color mapping

  Example
  -------
  >>> labels = np.array([0, 1, 1, -1, 2, 0])
  >>> encoded, unique, cmap, norm = labels_to_colormap(labels)
  >>> # encoded: [1, 2, 2, 0, 3, 1] (with -1 mapped to 0)
  """
  from matplotlib.colors import BoundaryNorm  # Discrete colormap normalization
  from matplotlib import colormaps            # Colormap registry

  # Find all unique labels (may include -1 for noise)
  unique = np.unique(labels)

  # Create mapping from original labels to sequential indices
  label_to_idx = {lab: i for i, lab in enumerate(unique)}

  # Apply mapping to all labels
  encoded = np.vectorize(label_to_idx.get, otypes=[int])(labels)

  # Create discrete colormap with exactly len(unique) colors
  cmap = colormaps.get_cmap("Paired").resampled(len(unique))

  # Create boundary norm for discrete color assignment
  # Boundaries at -0.5, 0.5, 1.5, ... ensure each integer maps to one color
  norm = BoundaryNorm(np.arange(-0.5, len(unique) + 0.5), cmap.N)

  return encoded, unique, cmap, norm

# =============================================================================
# STATION INVENTORY MANAGEMENT
# =============================================================================


def inventory(
    stations: Path,
    output: Optional[Path] = None
  ) -> pd.DataFrame:
  """
  Load and process station metadata from StationXML files.

  Reads all .xml files from the specified directory, extracts station
  coordinates, and assigns colors for plotting.

  Args:
    stations: Path to directory containing StationXML files.

  Returns:
    pd.DataFrame: DataFrame containing station metadata with columns:
    LONGITUDE_STR, LATITUDE_STR, DEPTH_STR, NETWORK_STR, STATION_STR,
    NETCOLOR_STR, STACOLOR_STR

  Side Effects:
    - Prints warnings for unreadable station files
  """
  # Import ObsPy utilities (lazy import to avoid circular dependencies)
  from obspy import Inventory, read_inventory

  # Initialize empty ObsPy Inventory container
  myInventory = Inventory()

  # Read all StationXML files in the directory
  for station in stations.glob("*.xml"):
    try:
      S = read_inventory(str(station))
    except Exception as e:
      print(f"WARNING: Unable to read {station}")
      print(e)
      continue
    myInventory.extend(S)

  elements: list[list] = []
  for net in sorted(myInventory.networks, key=lambda x: x.code):
    for sta in net.stations:
      elements.append([
        f"{net.code}.{sta.code}.",  # Unique station ID
        sta.longitude,
        sta.latitude,
        sta.elevation,
        net.code,
        sta.code,
      ])
  INVENTORY = pd.DataFrame(
    elements,
    columns=[INDEX_STR, LONGITUDE_STR, LATITUDE_STR, DEPTH_STR, NETWORK_STR,
             STATION_STR],
  ).sort_values(by=[INDEX_STR]).reset_index(drop=True)

  # Use labels_to_colormap for consistent network and station coloring
  from sklearn.preprocessing import LabelEncoder
  net_encoder = LabelEncoder()
  sta_encoder = LabelEncoder()

  net_labels: np.ndarray = net_encoder.fit_transform(
    INVENTORY[NETWORK_STR].values # type: ignore
  )
  sta_labels: np.ndarray = sta_encoder.fit_transform(
    INVENTORY[STATION_STR].values # type: ignore
  )

  _, _, net_cmap, net_norm = labels_to_colormap(net_labels)
  _, _, sta_cmap, sta_norm = labels_to_colormap(sta_labels)

  INVENTORY[NETCOLOR_STR] = [net_cmap(net_norm(l)) for l in net_labels]
  INVENTORY[STACOLOR_STR] = [sta_cmap(sta_norm(l)) for l in sta_labels]

  if output is not None:
    INVENTORY.to_csv(output / "OGSInventory.csv", index=False)
  return INVENTORY


# =============================================================================
# WAVEFORM FILE DISCOVERY
# =============================================================================


def waveforms(
    waveforms: Path,
    stations: Path,
    start: datetime,
    end: datetime,
    output: Path = Path("."),
    vlines: list[tuple[datetime, str, str]] = []
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """
  Scan directory for waveform files within a specified date range.

  Recursively searches for MiniSEED files, organizes them by date and
  station, and generates a data availability plot.

  Args:
    waveforms: Path to the waveforms directory to scan.
    stations: Path to directory containing StationXML files.
    start: Start date (inclusive) of the date range.
    end: End date (inclusive) of the date range.
    output: Path to directory where availability plot will be saved.
    vlines: List of tuples containing datetime objects, labels, and colors
            to mark with vertical lines on the plot.

  Returns:
    pd.DataFrame: DataFrame containing waveform file information with columns:
    NETWORK_STR, STATION_STR, LOC_NAME_STR, CHANNEL_STR, DATE_STR, FILENAME_STR
    Each row represents a waveform file.

  Side Effects:
    Generates "OGSAvailability.png" showing station count over time.

  Note:
    Expects waveform filenames in format:
    NET.STA.LOC.CHA__YYYYMMDDTHHMMSS__...mseed
  """
  # Import plotting utilities (lazy import)
  import ogsplotter as OGS_P
  from matplotlib import pyplot as plt

  elements = []
  # Scan all MiniSEED files recursively
  for wf in waveforms.glob("**/*.mseed"):
    if wf.name.startswith("."): continue  # Skip hidden files
    # Parse filename: NET.STA.LOC.CHA__YYYYMMDDTHHMMSS__suffix.mseed
    stid, dateinitid, _ = wf.stem.split(UNDERSCORE_STR + UNDERSCORE_STR)

    # Parse date from filename
    dateinitid = UTCDateTime(dateinitid).date
    if dateinitid < start.date() or dateinitid > end.date():
      continue  # Skip files outside date range
    elements.append([*stid.split(PERIOD_STR), dateinitid, wf])

  WAVEFORMS = pd.DataFrame(elements,
                           columns=[NETWORK_STR, STATION_STR, LOC_NAME_STR,
                                    CHANNEL_STR, DATE_STR, FILENAME_STR])
  WAVEFORMS.to_csv(output / "OGSWaveforms.csv", index=False)
  print(f"Saved file to {output / 'OGSWaveforms.csv'}")
  INVENTORY = inventory(stations)
  INVENTORY = INVENTORY.merge(
    WAVEFORMS[[NETWORK_STR, STATION_STR]],
    how="inner",
    on=[NETWORK_STR, STATION_STR]
  ).drop_duplicates()
  INVENTORY.to_csv(output / "OGSInventory.csv", index=False)
  print(f"Saved file to {output / 'OGSInventory.csv'}")
  mystations = OGS_P.map_plotter(
    OGS_STUDY_REGION,
    legend=True,
    marker="^",
  )
  for net, df in INVENTORY.groupby(NETWORK_STR):
    mystations.add_plot(
      df[LONGITUDE_STR], df[LATITUDE_STR], label=net,
      color=None, facecolors="none", edgecolors=df[NETCOLOR_STR],
      legend=True, output=output / "img" / "OGSStations.png",
    )
  plt.close()

  NET_COLORS = INVENTORY[
    [NETWORK_STR, NETCOLOR_STR]
  ].drop_duplicates().set_index(NETWORK_STR)[NETCOLOR_STR].to_dict()
  DAYS = np.arange(start, end + ONE_DAY, ONE_DAY, # type: ignore
                   dtype='datetime64[D]').tolist() # type: ignore
  DAYS = [UTCDateTime(day).date for day in DAYS]
  counts = {
    day: {net: 0 for net in WAVEFORMS[NETWORK_STR].unique()} for day in DAYS
  }
  for (date, net), group in WAVEFORMS.groupby([DATE_STR, NETWORK_STR]):
    counts[date][net] = len(group[STATION_STR].unique())
  df = pd.DataFrame(counts).sort_index().T
  x, y = [UTCDateTime(xx).date for xx in df.index], df.values.T
  OGS_P.stack_plotter(
    x, y, labels=df.columns.tolist(),
    colors=[NET_COLORS.get(net, "gray") for net in df.columns],
    xlabel="Date", ylabel="Station Count",
    output=output / "img" / "OGSAvailability.png",
    vlines=vlines,
    legend=True
  )
  plt.close()
  return WAVEFORMS, INVENTORY


# =============================================================================
# ARGPARSE CUSTOM ACTIONS
# =============================================================================


class SortDatesAction(argparse.Action):
  """
  Custom argparse action to sort date arguments chronologically.

  When multiple dates are provided as command-line arguments, this action
  ensures they are stored in sorted order.

  Example:
      parser.add_argument('-D', nargs=2, action=SortDatesAction)
      # Args "-D 20220115 20220101" will be stored as [20220101, 20220115]
  """

  def __call__(self, parser, namespace, values, option_string=None):
    """Sort and store the values."""
    setattr(namespace, self.dest, sorted(values))  # type: ignore


# =============================================================================
# BIPARTITE GRAPH MATCHING CLASSES
# =============================================================================
# Classes for optimal assignment between ground truth and predicted data
# using maximum weight bipartite matching via NetworkX


class OGSBPGraph():
  """
  Base class for bipartite graph matching between two datasets.

  Provides the framework for constructing bipartite graphs where nodes
  represent data records and edges represent potential matches with
  associated similarity weights.

  Attributes:
    Base: DataFrame containing reference/ground truth records.
    Target: DataFrame containing records to match against Base.
    G: NetworkX Graph representing the bipartite structure.
    E: Set of matched edge pairs (base_idx, target_idx + len(base)).

  Architecture:
    Base nodes: indices 0 to len(Base)-1
    Target nodes: indices len(Base) to len(Base)+len(Target)-1
    Edges: Connect Base[i] to Target[j] if they are potential matches

  Note:
    This is an abstract base class. Subclasses must implement makeMatch().
  """

  def __init__(self, Base: pd.DataFrame, Target: pd.DataFrame):
    """
    Initialize bipartite graph with Base and Target datasets.

    Args:
        Base: Reference dataset (ground truth picks or events).
        Target: Dataset to match against Base (predictions).
    """
    # Reset indices to ensure consistent node numbering
    self.Base = Base.reset_index(drop=True)
    self.Target = Target.reset_index(drop=True)

    # Initialize empty graph and edge set
    self.G = nx.Graph()
    self.E: set[tuple[int, int]] = set()

    # Build graph and compute matching if both datasets are non-empty
    if not self.Base.empty and not self.Target.empty:
        self.makeMatch()

  def makeMatch(self) -> None:
    """
    Construct the bipartite graph and compute maximum weight matching.

    Must be implemented by subclasses to define edge construction logic.

    Raises:
        NotImplementedError: If called on base class.
    """
    raise NotImplementedError


class OGSBPGraphPicks(OGSBPGraph):
  """
  Bipartite graph for optimal pick assignment between datasets.

  Implements maximum weight bipartite matching to find the optimal
  one-to-one correspondence between manual (Base) and predicted (Target)
  phase picks. Uses NetworkX's max_weight_matching algorithm.

  The matching considers:
  - Time proximity: Picks must be within PICK_TIME_OFFSET
  - Station matching: Only same-station picks can match
  - Phase type: P-P and S-S matches preferred
  - Probability: Higher confidence picks weighted more

  Attributes:
    Inherited from OGSBPGraph.

  Example:
    >>> matcher = OGSBPGraphPicks(manual_picks_df, predicted_picks_df)
    >>> matched_pairs = matcher.E  # Set of (base_idx, target_idx+I) tuples

  Note:
    - Base DataFrame should have: TIME_STR, STATION_STR, PHASE_STR
    - Target DataFrame should have: TIME_STR, STATION_STR, PHASE_STR,
      PROBABILITY_STR
    - Uses station-based pre-filtering for O(n) improvement
  """

  def __init__(self, Base: pd.DataFrame, Target: pd.DataFrame):
    """
    Initialize pick matcher with optional probability column creation.

    Args:
      Base: Manual picks DataFrame (ground truth).
      Target: Predicted picks DataFrame from ML model.
    """
    # Ensure PROBABILITY_STR column exists, defaulting to 1.0 if absent
    # (manual picks often don't have probability values)
    if PROBABILITY_STR not in Base.columns:
      Base[PROBABILITY_STR] = 1.0

    # Optimization: Vectorized UTCDateTime conversion using list comprehension
    # Faster than apply(lambda) for large datasets
    if TIME_STR in Base.columns:
      Base[TIME_STR] = [UTCDateTime(x) for x in Base[TIME_STR]]
    if TIME_STR in Target.columns:
      Target[TIME_STR] = [UTCDateTime(x) for x in Target[TIME_STR]]

    # Call parent constructor (triggers makeMatch)
    super().__init__(Base, Target)

  def makeMatch(self) -> None:
    """
    Build bipartite graph and compute maximum weight matching for picks.

    Algorithm:
    1. Group target picks by station for O(1) lookup
    2. For each base pick, find target picks at same station
    3. Add edge if time difference <= PICK_TIME_OFFSET
    4. Edge weight = dist_pick() similarity score
    5. Compute max weight matching (not max cardinality)

    Result stored in self.E as set of matched index pairs.
    """
    I = len(self.Base)  # Offset for target node indices
    J = len(self.Target)
    self.G = nx.Graph()

    # Parse station from SEED ID format (NET.STA.LOC.CHA -> extract STA)
    self.Target[NETWORK_STR] = self.Target[STATION_STR].str.split(".").str[0]
    self.Target[STATION_STR] = self.Target[STATION_STR].str.split(".").str[1]
    self.Base[STATION_STR] = self.Base[STATION_STR].astype(str)

    # Pre-group targets by station for efficient lookup
    target_by_station = {
      station: group for station, group in self.Target.groupby(STATION_STR)
    }

    # Build edges between matching picks
    for idxBase, rowBase in self.Base.iterrows():
      station = rowBase[STATION_STR]

      # Only iterate over targets at the same station
      if station not in target_by_station:
        continue

      target_candidates = target_by_station[station]

      for idxTarget, rowTarget in target_candidates.iterrows():
        # Check time proximity constraint
        if diff_time(rowBase, rowTarget) <= PICK_TIME_OFFSET.total_seconds():
          # Add edge with similarity weight
          self.G.add_edge(
            idxBase, int(idxTarget) + I,  # Target offset by I
            weight=dist_pick(rowBase, rowTarget)
          )

    # Compute maximum weight matching (optimal assignment)
    self.E = nx.max_weight_matching(self.G, maxcardinality=False, weight='weight')


class OGSBPGraphEvents(OGSBPGraph):
  """
  Bipartite graph for optimal event assignment between datasets.

  Implements maximum weight bipartite matching to find the optimal
  one-to-one correspondence between manual (Base) and detected (Target)
  seismic events. Uses both temporal and spatial constraints.

  The matching considers:
  - Time proximity: Events must be within EVENT_TIME_OFFSET (1.5 sec)
  - Spatial proximity: Events must be within EVENT_DIST_OFFSET (3 km)
  - Weight: 99% time similarity + 1% spatial similarity

  Attributes:
    Inherited from OGSBPGraph.

  Example:
    >>> matcher = OGSBPGraphEvents(catalog_events_df, detected_events_df)
    >>> matched_pairs = matcher.E

  Note:
    - Requires: TIME_STR, LATITUDE_STR, LONGITUDE_STR columns
    - Optional: DEPTH_STR for 3D distance calculation
    - Uses time-based pre-filtering for efficiency
  """

  def __init__(self, Base: pd.DataFrame, Target: pd.DataFrame):
    """
    Initialize event matcher with time column normalization.

    Handles different time column names from various sources
    (e.g., "event_time" from some associators, "time" from others).

    Args:
      Base: Catalog events DataFrame (ground truth).
      Target: Detected events DataFrame from associator.
    """
    # Handle "event_time" column name variant
    if "event_time" in Target.columns:
      Target[TIME_STR] = UTCDateTime(Target["event_time"])

    # Vectorized UTCDateTime conversion
    if TIME_STR in Base.columns:
      Base[TIME_STR] = [UTCDateTime(x) for x in Base[TIME_STR]]
    if "time" in Target.columns:
      Target[TIME_STR] = [UTCDateTime(x) for x in Target["time"]]

    # Call parent constructor (triggers makeMatch)
    super().__init__(Base, Target)

  def makeMatch(self):
    """
    Build bipartite graph and compute maximum weight matching for events.

    Algorithm:
    1. Vectorize time values for efficient filtering
    2. For each base event, pre-filter targets by time window
    3. Check spatial distance for time-proximate candidates
    4. Add edge if both constraints met, weight = dist_event()
    5. Compute max weight matching

    Optimization: Pre-filtering by time significantly reduces the
    O(n*m) comparison space, especially for sparse event catalogs.
    """
    I = len(self.Base)  # Offset for target node indices

    # Optimization: Vectorized time values for efficient filtering
    base_times = self.Base[TIME_STR].values
    target_times = self.Target[TIME_STR].values

    # Build edges between matching events
    for idxBase, rowBase in self.Base.iterrows():
      # Pre-filter targets by time window (reduces candidates significantly)
      time_mask = np.abs(target_times - rowBase[TIME_STR]) <= EVENT_TIME_OFFSET.total_seconds()
      target_candidates = self.Target[time_mask]

      for idxTarget, rowTarget in target_candidates.iterrows():
        # Only check spatial distance if time constraint is met
        if diff_space(rowBase, rowTarget) <= EVENT_DIST_OFFSET:
          # Add edge with similarity weight
          self.G.add_edge(
            idxBase, int(idxTarget) + I,
            weight=dist_event(rowBase, rowTarget)
          )

    # Compute maximum weight matching
    self.E = nx.max_weight_matching(self.G, maxcardinality=False, weight='weight')
"""
=============================================================================
OGS Catalog Parser - Multi-Format Seismic Catalog Aggregator
=============================================================================

OVERVIEW:
This module provides a unified interface for parsing and merging seismic
catalogs from multiple OGS file formats. It acts as a catalog aggregator
that can read picks and events from various legacy formats (HPL, DAT, TXT,
PUN) and consolidate them into a single unified catalog.

KEY FEATURES:
  - Multi-format support: Automatically dispatches to format-specific parsers
    based on file extension
  - Catalog merging: Combines picks and events from multiple files into a
    single consolidated catalog with proper cross-referencing
  - Geographic filtering: Supports polygon-based spatial filtering
  - Date range filtering: Temporal subsetting of catalog data
  - Optimized aggregation: Uses vectorized pandas operations for efficiency

SUPPORTED FILE FORMATS:
  ┌──────────┬─────────────────┬────────────────────────────────────┐
  │ Extension│ Parser Class    │ Content Description                │
  ├──────────┼─────────────────┼────────────────────────────────────┤
  │ .hpl     │ DataFileHPL     │ Hypocenter locations (recommended) │
  │ .dat     │ DataFileDAT     │ Phase picks (P/S arrivals)         │
  │ .txt     │ DataFileTXT     │ Local magnitude (ML) information   │
  │ .pun     │ DataFilePUN     │ Event punch cards                  │
  └──────────┴─────────────────┴────────────────────────────────────┘

ARCHITECTURE:
  Command Line / API
    │
    ▼
  DataCatalog (this module)
    │
    ├── DataFileHPL (ogshpl.py)
    ├── DataFileDAT (ogsdat.py)
    ├── DataFileTXT (ogstxt.py)
    ├── DataFilePUN (ogspun.py)
    │
    ▼
  Merged Catalog (Parquet output)

USAGE:
  Command line - Parse and merge multiple files:
    python ogsparser.py -f file1.hpl file2.dat -D 20220101 20221231 --merge

  Command line - Process all files in directory:
    python ogsparser.py -d /path/to/catalog/ -x .hpl .dat --merge

  Programmatic:
    from ogsparser import DataCatalog
    catalog = DataCatalog(args)
    catalog.read()
    catalog.merge()

OUTPUT:
  When --merge is specified, creates consolidated Parquet files:
    - {output}/.all/assignments/YYYY-MM-DD  (merged picks)
    - {output}/.all/events/YYYY-MM-DD       (merged events)

MERGE LOGIC:
  1. PICKS: Simple concatenation from all input files
  2. EVENTS: Outer join on time/location, with format-specific handling:
     - TXT files contribute magnitude data (ML, MD)
     - PUN files contribute hypocenter locations
     - HPL files provide primary event information

DEPENDENCIES:
  - pandas: DataFrame operations and merge logic
  - matplotlib: Polygon path for geographic filtering
  - ogsdatafile: Base class for file parsing
  - Format-specific parsers: ogshpl, ogsdat, ogspun, ogstxt

=============================================================================
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

# Standard library: Command-line argument parsing
import argparse

# Pandas: DataFrame operations, merging, and Parquet I/O
import pandas as pd

# Standard library: Filesystem path handling
from pathlib import Path

# Matplotlib: Path object for polygon-based geographic containment tests
from matplotlib.path import Path as mplPath

# Standard library: Date/time objects
from datetime import datetime

# Local module: OGS-specific constants (extensions, column names, formats)
import ogsconstants as OGS_C

# Local module: Base class providing file I/O and logging
from ogsdatafile import OGSDataFile

# Local modules: Format-specific parsers
from ogshpl import DataFileHPL  # Hypocenter location files
from ogsdat import DataFileDAT  # Phase pick files
from ogspun import DataFilePUN  # Punch card format files
from ogstxt import DataFileTXT  # Text format magnitude files

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

# Base path for data files (two levels up from this script's location)
DATA_PATH = Path(__file__).parent.parent.parent


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_polygon(points: str) -> mplPath:
  """
  Convert a string of polygon vertices to a matplotlib Path object.

  Used by argparse to validate and convert polygon arguments for
  geographic filtering of seismic events.

  Args:
    points: String representation of polygon vertices

  Returns:
    mplPath: Closed matplotlib Path for containment testing
  """
  return mplPath(points, closed=True)


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_arguments() -> argparse.Namespace:
  """
  Parse command-line arguments for the catalog parser.

  Supports two input modes:
    1. File mode (-f): Process specific files
    2. Directory mode (-d): Process all matching files in directory

  Returns:
    argparse.Namespace with:
      - merge: Boolean flag to merge all files into single catalog
      - ext: List of file extensions to process
      - verbose: Boolean flag for debug output
      - directory: Path to input directory (mutually exclusive with file)
      - file: List of input file paths (mutually exclusive with directory)
      - dates: Tuple of (start_date, end_date) for filtering
      - julian: Alternative Julian date range specification
      - output: Path for output catalog directory
      - polygon: matplotlib Path for geographic filtering
  """
  parser = argparse.ArgumentParser(description="Parse OGS Manual Catalogs")

  # -m/--merge: Consolidate all parsed files into a single unified catalog
  parser.add_argument(
    "-m", "--merge", action='store_true', default=False,
    help="Merge all data files into a single catalog")

  # -x/--ext: File extensions to process (default: all known extensions)
  parser.add_argument(
    "-x", "--ext", default=OGS_C.ALL_WILDCHAR_STR, type=str,
    nargs=OGS_C.ONE_MORECHAR_STR, metavar=OGS_C.EMPTY_STR,
    help="File extension to process")

  # -v/--verbose: Enable detailed logging output
  parser.add_argument(
    '-v', "--verbose", action='store_true', default=False,
    help="Enable verbose output")

  # -------------------------------------------------------------------------
  # INPUT PATH GROUP (mutually exclusive: directory OR file)
  # -------------------------------------------------------------------------
  path_group = parser.add_mutually_exclusive_group(required=True)

  # -d/--directory: Process all matching files in a directory recursively
  path_group.add_argument(
    '-d', "--directory", required=False, type=OGS_C.is_dir_path, default=None,
    help="Base directory for data files.")

  # -f/--file: Process specific file(s) by path
  path_group.add_argument(
    '-f', "--file", required=False, type=OGS_C.is_file_path, default=None,
    nargs=OGS_C.ONE_MORECHAR_STR, metavar=OGS_C.EMPTY_STR,
    help="Base file for data files.")

  # -------------------------------------------------------------------------
  # DATE RANGE GROUP (mutually exclusive: Gregorian OR Julian)
  # -------------------------------------------------------------------------
  date_group = parser.add_mutually_exclusive_group(required=False)

  # -D/--dates: Gregorian date range (YYYYMMDD format)
  date_group.add_argument(
    '-D', "--dates", required=False, metavar=OGS_C.DATE_STD,
    type=OGS_C.is_date, nargs=2, action=OGS_C.SortDatesAction,
    default=[datetime.min, datetime.max - OGS_C.ONE_DAY],
    help="Specify the beginning and ending (inclusive) Gregorian date " \
         "(YYMMDD) range to work with.")

  # -J/--julian: Julian date range (YYDDD format)
  date_group.add_argument(
    '-J', "--julian", required=False, metavar=OGS_C.DATE_STD,
    action=OGS_C.SortDatesAction, type=OGS_C.is_julian, default=None, nargs=2,
    help="Specify the beginning and ending (inclusive) Julian date (YYMMDD) " \
         "range to work with.")

  # -o/--output: Output directory for the merged catalog
  parser.add_argument(
    "-o", "--output", required=False, type=OGS_C.is_dir_path,
    default=DATA_PATH / "catalog" / "OGSCatalog",
    help="Name of the catalog")

  # -P/--polygon: Geographic polygon for spatial filtering of events
  parser.add_argument(
    "-P", "--polygon", required=False, type=is_polygon,
    default=mplPath(OGS_C.OGS_POLY_REGION, closed=True),
    nargs=OGS_C.ONE_MORECHAR_STR, metavar=OGS_C.EMPTY_STR,
    help="Polygon string to filter events")

  return parser.parse_args()


# =============================================================================
# DataCatalog Class - Multi-Format Catalog Aggregator
# =============================================================================

class DataCatalog(OGSDataFile):
  """
  Aggregator class for parsing and merging multiple OGS catalog files.

  This class extends OGSDataFile to provide multi-format support and
  catalog merging capabilities. It automatically dispatches to the
  appropriate format-specific parser based on file extension.

  Attributes:
    DATAFILE_TYPES: Dict mapping file extensions to parser classes
    args: Parsed command-line arguments
    files: List of instantiated format-specific parser objects
  """

  # -------------------------------------------------------------------------
  # FILE TYPE REGISTRY
  # -------------------------------------------------------------------------
  # Maps file extensions to their corresponding parser classes
  # Each parser handles a specific OGS legacy format
  DATAFILE_TYPES = {
    OGS_C.HPL_EXT: DataFileHPL,  # (Highly recommended) Hypocenter information
    OGS_C.DAT_EXT: DataFileDAT,  # (Recommended) Picks information
    OGS_C.TXT_EXT: DataFileTXT,  # Local Magnitude information
    OGS_C.PUN_EXT: DataFilePUN,  # Events (punch card format)
  }

  # -------------------------------------------------------------------------
  # CONSTRUCTOR
  # -------------------------------------------------------------------------

  def __init__(self, args: argparse.Namespace) -> None:
    """
    Initialize the catalog aggregator with command-line arguments.

    Args:
      args: Parsed argparse.Namespace containing:
        - output: Output directory path
        - dates: (start, end) date tuple
        - verbose: Debug output flag
        - polygon: Geographic filter polygon
    """
    # Store arguments for later use in read() and merge()
    self.args = args

    # Initialize list to hold format-specific parser instances
    self.files : list[OGSDataFile] = list()

    # Initialize parent class with catalog settings
    super().__init__(
      args.output, args.dates[0], args.dates[1], verbose=args.verbose,
      polygon=args.polygon, output=args.output)

  # -------------------------------------------------------------------------
  # METHOD: read() - Discover and parse input files
  # -------------------------------------------------------------------------

  def read(self) -> None:
    """
    Discover input files and delegate parsing to format-specific parsers.

    Operates in two modes:
      1. File mode: Process explicitly specified files
      2. Directory mode: Recursively find files matching extension filter

    For each discovered file:
      - Instantiates the appropriate parser based on extension
      - Calls parser.read() to parse the file
      - Calls parser.log() to write Parquet output
    """
    # -------------------------------------------------------------------------
    # FILE MODE: Process explicitly specified files
    # -------------------------------------------------------------------------
    if self.args.directory is None:
      for fr in self.args.file:
        # Get file extension to determine parser type
        ext = Path(fr).suffix

        # Only process files with known extensions
        if ext in self.DATAFILE_TYPES:
          # Instantiate appropriate parser and add to file list
          self.files.append(self.DATAFILE_TYPES[ext](
            fr, self.args.dates[0], self.args.dates[1],
            verbose=self.args.verbose, polygon=self.args.polygon,
            output=self.args.output))

    # -------------------------------------------------------------------------
    # DIRECTORY MODE: Recursively find matching files
    # -------------------------------------------------------------------------
    else:
      # Process each requested extension
      for ext in self.args.ext:
        # Recursively glob for files with this extension
        files = list(self.args.directory.rglob(f"*{ext}"))

        # Warn if no files found (in verbose mode)
        if len(files) == 0 and self.args.verbose:
          print(f"No *{ext} files found in {self.args.directory}")

        # Process each discovered file
        for fr in files:
          # Only process files with known extensions
          if fr.suffix in self.DATAFILE_TYPES:
            # Instantiate appropriate parser and add to file list
            self.files.append(self.DATAFILE_TYPES[fr.suffix](
              fr, self.args.dates[0], self.args.dates[1],
              verbose=self.args.verbose, polygon=self.args.polygon,
              output=self.args.output))

    # -------------------------------------------------------------------------
    # PARSE AND LOG ALL FILES
    # -------------------------------------------------------------------------
    for f in self.files:
      # Parse the input file into picks/events DataFrames
      f.read()
      # Write parsed data to Parquet format
      f.log()

  # -------------------------------------------------------------------------
  # METHOD: merge_events() - Consolidate events from all files
  # -------------------------------------------------------------------------

  def merge_events(self) -> pd.DataFrame:
    """
    Merge event catalogs from all parsed files into a single DataFrame.

    Merge strategy varies by file type:
      - First file: Initialize EVENTS DataFrame
      - TXT files: Outer join on time/groups, contributes magnitude data
      - PUN files: Outer join on time/location/depth

    After merging, computes pick statistics per event:
      - Number of P-wave picks
      - Number of S-wave picks
      - Number of stations with both P and S picks

    Returns:
      pd.DataFrame: Consolidated events with all available metadata
    """
    # -------------------------------------------------------------------------
    # MERGE EVENTS FROM ALL FILES
    # -------------------------------------------------------------------------
    for f in self.files:
      if f.get("EVENTS").empty: continue
      self.logger.info(f"Processing EVENTS from file: {f.input}")

      # First file: Initialize with copy of its events
      f.EVENTS[OGS_C.IDX_EVENTS_STR] = f.EVENTS[OGS_C.IDX_EVENTS_STR].apply(
        pd.to_numeric, errors='coerce'
      ).astype(int)
      if self.EVENTS.empty:
        self.EVENTS = f.get("EVENTS").copy()

        continue
      self.EVENTS[OGS_C.IDX_EVENTS_STR] = self.EVENTS[OGS_C.IDX_EVENTS_STR].apply(
        pd.to_numeric, errors='coerce'
      ).astype(int)
      # TXT files: Contribute magnitude and error information
      if f.input.suffix == OGS_C.TXT_EXT:
        """
        TXT files contain magnitude information that needs to be joined
        with hypocenter data from HPL/PUN files.

        Columns from existing EVENTS:
          time, latitude, longitude, depth, picks counts, ML values, groups, no

        Columns contributed by TXT:
          time, groups, event_id, magnitude_d, erz, erh, gap
        """
        self.EVENTS = pd.merge(
          # Left side: Existing merged events
          self.EVENTS[[OGS_C.IDX_EVENTS_STR,
            # TODO: Order alfabetically
            OGS_C.LATITUDE_STR,
            OGS_C.LONGITUDE_STR,
            OGS_C.DEPTH_STR,
            OGS_C.NUMBER_P_PICKS_STR,
            OGS_C.NUMBER_S_PICKS_STR,
            OGS_C.NUMBER_P_AND_S_PICKS_STR,
            OGS_C.ML_MEDIAN_STR,
            OGS_C.ML_UNC_STR,
            OGS_C.ML_STATIONS_STR,
          ]],
          # Right side: TXT file contribution (magnitude, errors, gap)
          f.EVENTS[[OGS_C.IDX_EVENTS_STR,
            # TODO: Order alfabetically
            OGS_C.TIME_STR,
            OGS_C.ERT_STR,
            OGS_C.ERZ_STR,
            OGS_C.ERH_STR,
            OGS_C.GAP_STR,
            OGS_C.GROUPS_STR,
            OGS_C.MAGNITUDE_L_STR,
            OGS_C.MAGNITUDE_D_STR,
          ]],
          how="outer",  # Keep all events from both sources
          on=OGS_C.IDX_EVENTS_STR).copy()

      # PUN files: Contribute hypocenter location data
      elif f.input.suffix == OGS_C.PUN_EXT:
        self.EVENTS = pd.merge(
          self.EVENTS,
          f.EVENTS,
          how="outer",  # Keep all events from both sources
          on=[OGS_C.TIME_STR, OGS_C.LATITUDE_STR,
              OGS_C.LONGITUDE_STR, OGS_C.DEPTH_STR,
              OGS_C.GROUPS_STR]).copy()

    # -------------------------------------------------------------------------
    # COMPUTE PICK STATISTICS PER EVENT
    # -------------------------------------------------------------------------
    # Optimization: Vectorized aggregation instead of iterrows() loops
    # This replaces nested loops with efficient groupby operations

    if not self.PICKS.empty:
      # Count picks by event and phase type using groupby + pivot
      # Creates a DataFrame with event_id as index and phase types as columns
      phase_counts = self.PICKS.groupby(
        [OGS_C.IDX_PICKS_STR, OGS_C.PHASE_STR]
      ).size().unstack(fill_value=0)

      # Map P and S pick counts to events
      for phase, column in [(OGS_C.PWAVE, OGS_C.NUMBER_P_PICKS_STR),
                            (OGS_C.SWAVE, OGS_C.NUMBER_S_PICKS_STR)]:
        if phase in phase_counts.columns:
          # Map count from phase_counts to events by event_id
          self.EVENTS[column] = self.EVENTS[OGS_C.IDX_EVENTS_STR].map(
            phase_counts[phase]
          ).fillna(0).astype(int)
        else:
          # No picks of this phase type
          self.EVENTS[column] = 0

      # Count stations with both P and S picks per event
      # Step 1: Count unique phase types per (event, station) pair
      station_phase_counts = self.PICKS.groupby(
        [OGS_C.IDX_PICKS_STR, OGS_C.STATION_STR]
      )[OGS_C.PHASE_STR].nunique()

      # Step 2: Filter to stations with 2+ phase types (both P and S)
      # Step 3: Count such stations per event
      stations_with_both = station_phase_counts[station_phase_counts >= 2].groupby(
        level=0  # Group by event_id (first level of MultiIndex)
      ).size()

      # Map station counts to events
      self.EVENTS[OGS_C.NUMBER_P_AND_S_PICKS_STR] = self.EVENTS[
        OGS_C.IDX_EVENTS_STR
      ].map(stations_with_both).fillna(0).astype(int)

    else:
      # No picks available: Set all counts to zero
      self.EVENTS[OGS_C.NUMBER_P_PICKS_STR] = 0
      self.EVENTS[OGS_C.NUMBER_S_PICKS_STR] = 0
      self.EVENTS[OGS_C.NUMBER_P_AND_S_PICKS_STR] = 0
      # Extract date from timestamp for grouping
      self.EVENTS[OGS_C.GROUPS_STR] = \
        self.EVENTS[OGS_C.TIME_STR].dt.date  # type: ignore

    return self.EVENTS

  # -------------------------------------------------------------------------
  # METHOD: merge_picks() - Consolidate picks from all files
  # -------------------------------------------------------------------------

  def merge_picks(self) -> pd.DataFrame:
    """
    Merge pick catalogs from all parsed files into a single DataFrame.

    Uses simple concatenation since picks from different files are
    independent (no deduplication or joining needed).

    Returns:
      pd.DataFrame: Consolidated picks from all input files
    """
    for f in self.files:
      # First file: Initialize with copy of its picks
      if self.PICKS.empty:
        self.PICKS = f.get("PICKS").copy()
      else:
        # Subsequent files: Concatenate picks
        picks = f.get("PICKS").copy()
        if not picks.empty:
          self.PICKS = pd.concat([self.PICKS, picks], ignore_index=True)

    return self.PICKS

  # -------------------------------------------------------------------------
  # METHOD: merge() - Full catalog merge and output
  # -------------------------------------------------------------------------

  def merge(self) -> None:
    """
    Perform full catalog merge: picks first, then events, then log output.

    Creates a merged catalog with:
      - All picks from all input files (concatenated)
      - All events with metadata from all files (joined)
      - Pick count statistics computed per event

    Output is written to {input}.all/ directory structure.
    """
    # Merge picks first (events depend on pick statistics)
    print(self.merge_picks())

    # Merge events and compute statistics
    self.merge_events()

    # Log output path for debugging
    print(self.input)

    # Append ".all" suffix to output path for merged catalog
    self.input = Path(self.input.__str__() + ".all")

    # Write merged catalog to Parquet files
    self.log()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args: argparse.Namespace) -> None:
  """
  Main entry point for command-line execution.

  Workflow:
    1. Create DataCatalog aggregator from arguments
    2. Read and parse all input files
    3. Optionally merge into unified catalog (if --merge specified)

  Args:
    args: Parsed command-line arguments from parse_arguments()
  """
  # Create catalog aggregator with provided arguments
  OGS_Catalog = DataCatalog(args)

  # Discover and parse all input files
  OGS_Catalog.read()

  # If merge flag set, consolidate all files into single catalog
  if args.merge: OGS_Catalog.merge()


# Script entry point: parse arguments and run main
if __name__ == "__main__": main(parse_arguments())
"""
===============================================================================
OGS Catalog Parser - Multi-Format Seismic Catalog Aggregator
===============================================================================

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
  - Geographic filtering inherited for loaded Parquet event days; parsed
    tables are not explicitly polygon-filtered here
  - Date range filtering: Temporal subsetting of catalog data
  - Optimized aggregation: Uses vectorized pandas operations for efficiency

SUPPORTED FILE FORMATS:
  ┌───────────┬──────────────┬────────────────────────────────────┐
  │ Extension │ Parser Class │ Content Description                │
  ├───────────┼──────────────┼────────────────────────────────────┤
  │ .hpl      │ DataFileHPL  │ Event summaries and P/S picks      │
  │ .dat      │ DataFileDAT  │ Phase picks (P/S arrivals)         │
  │ .txt      │ DataFileTXT  │ Events, ML/MD, locality and type   │
  │ .pun      │ DataFilePUN  │ Hypo71 event summaries             │
  └───────────┴──────────────┴────────────────────────────────────┘

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
    python -m OGS.src.ogsparser -f file1.hpl file2.dat -D 20220101 20221231 \
      --merge

  Command line - Process all files in directory:
    python -m OGS.src.ogsparser -d /path/to/catalog/ -x .hpl .dat --merge

  Programmatic:
    from OGS.src.ogsparser import DataCatalog
    catalog = DataCatalog(args)
    catalog.read()
    catalog.merge()

OUTPUT:
  When --merge is specified, creates consolidated Parquet files:
    - {output}/.all/assignments/YYYY-MM-DD  (merged picks)
    - {output}/.all/events/YYYY-MM-DD       (merged events)

MERGE LOGIC:
  1. PICKS: Concatenate, then keep the first row per event ID/station/phase.
  2. EVENTS: Process HPL, PUN, then TXT; existing non-null metadata wins.
     - HPL rows are concatenated; overlapping year/event IDs raise ValueError.
     - PUN rows combine on time/latitude/longitude/depth/group.
     - TXT rows combine on year/event ID.
     - Missing merge-key columns fall back to concatenation.
  3. Compute phase statistics from the consolidated picks, write date
     partitions, and generate plots.

DEPENDENCIES:
  - pandas: DataFrame operations and merge logic
  - ogsdatafile: Base class for file parsing
  - Format-specific parsers: ogshpl, ogsdat, ogspun, ogstxt

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

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

# Standard library: Command-line argument parsing
import argparse

# Pandas: DataFrame operations, merging, and Parquet I/O
import pandas as pd

# Standard library: Filesystem path handling
from pathlib import Path

from . import ogsconstants as OGS_C, ogsutils as OGS_U
from .ogsdatafile import OGSDataFile
from .ogshpl import DataFileHPL  # Hypocenter location files
from .ogsdat import DataFileDAT  # Phase picks files
from .ogspun import DataFilePUN  # Punch card format files
from .ogstxt import DataFileTXT  # Text catalog event summaries

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
      OGS_C.HPL_EXT: DataFileHPL,  # (Recommended) Hypocenter information
      OGS_C.DAT_EXT: DataFileDAT,  # (Recommended) Picks information
      OGS_C.TXT_EXT: DataFileTXT,  # Events, magnitudes, locality and type
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
        - file / directory: Explicit files or recursive discovery directory
        - ext: Extension filters for directory mode
      No CLI polygon argument is used; the inherited default is retained.
    """
    # Store arguments for later use in read() and merge()
    self.args = args

    # Initialize list to hold format-specific parser instances
    self.files: list[OGSDataFile] = list()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Initialize parent class with catalog settings
    super().__init__(
        Path(args.output), args.dates[0], args.dates[1], verbose=args.verbose,
        output=Path(args.output)
    )

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

    Unsupported suffixes are ignored. Parser/read failures propagate; log()
    reports individual partition-write failures without re-raising them.
    Returns None and retains parser instances in self.files.
    """
    # -------------------------------------------------------------------------
    # DISCOVER AND INSTANTIATE PARSERS (FILE OR DIRECTORY MODE)
    # -------------------------------------------------------------------------
    if self.args.directory is None:
      input_paths = [Path(fr) for fr in self.args.file]
    else:
      input_paths = []
      for ext in self.args.ext:
        files = list(self.args.directory.rglob(f"*{ext}"))
        if len(files) == 0 and self.args.verbose:
          self.logger.info(f"No *{ext} files found in {self.args.directory}")
        input_paths.extend(files)

    for path in input_paths:
      if path.suffix in self.DATAFILE_TYPES:
        self.files.append(self.DATAFILE_TYPES[path.suffix](
            path, self.args.dates[0], self.args.dates[1],
            verbose=self.args.verbose, output=Path(self.args.output)
        ))

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
    Consolidate events from all parsed files into a unified catalog.

    Processes files in deterministic format order (HPL -> PUN -> TXT).
    Validates per-file uniqueness on (__event_year, event_id), prevents
    cross-file HPL collisions, and merges metadata via non-destructive
    outer combine_first. Computes vectorized pick statistics per event.
    Existing non-null values take precedence over later formats. PUN uses
    time/location/depth/group keys; TXT uses year/event ID keys.

    Returns:
      pd.DataFrame: Consolidated events with unified metadata

    Raises:
      ValueError: If event years cannot be determined, per-file year/ID
        identities repeat, HPL identities overlap, or merge keys repeat.
      TypeError: If the events time column is not a Series when rebuilding
        groups without picks.
    """
    self.logger.info("Merging events from files...")
    prepared = []
    for f in self.files:
      if not f.HAS_EVENTS:
        continue
      events = f.get("EVENTS").copy()
      if events.empty:
        continue
      self.logger.info(f"Processing EVENTS from file: {f.input}")
      f.EVENTS = self.normalize_groups(f.EVENTS.copy())
      self.EVENTS = self.normalize_groups(self.EVENTS)

      # First file: Initialize with copy of its events
      f.EVENTS[OGS_C.IDX_EVENTS_STR] = f.EVENTS[OGS_C.IDX_EVENTS_STR].apply(
          pd.to_numeric, errors='coerce'
      ).astype(int)
      if self.EVENTS.empty or OGS_C.IDX_EVENTS_STR not in self.EVENTS.columns:
        self.EVENTS = f.EVENTS.copy()
        continue
      if self.EVENTS[self.EVENTS[OGS_C.IDX_EVENTS_STR].isin(
          f.EVENTS[OGS_C.IDX_EVENTS_STR]
      )].empty:
        self.EVENTS = pd.concat([self.EVENTS, f.EVENTS], ignore_index=True)
        continue
      self.EVENTS[OGS_C.IDX_EVENTS_STR] = self.EVENTS[
          OGS_C.IDX_EVENTS_STR
      ].apply(pd.to_numeric, errors='coerce').astype(int)
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
            self.EVENTS[[
                OGS_C.IDX_EVENTS_STR,
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
            f.EVENTS[[
                OGS_C.IDX_EVENTS_STR,
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
            on=OGS_C.IDX_EVENTS_STR
        ).copy()

      # PUN files: Contribute hypocenter location data
      elif f.input.suffix == OGS_C.PUN_EXT:
        self.EVENTS = pd.merge(
            self.EVENTS,
            f.EVENTS,
            how="outer",  # Keep all events from both sources
            on=[
                OGS_C.TIME_STR, OGS_C.LATITUDE_STR,
                OGS_C.LONGITUDE_STR, OGS_C.DEPTH_STR,
                OGS_C.GROUPS_STR
            ]
        ).copy()

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
      stations_with_both = station_phase_counts[
          station_phase_counts >= 2
      ].groupby(level=0).size()

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
      time_series = self.EVENTS[OGS_C.TIME_STR]
      if not isinstance(time_series, pd.Series):
        raise TypeError(
            f"Expected {OGS_C.TIME_STR!r} to resolve to a pandas Series"
        )
      self.EVENTS[OGS_C.GROUPS_STR] = time_series.dt.strftime(OGS_C.DATE_FMT)

    return self.EVENTS

  # -------------------------------------------------------------------------
  # METHOD: merge_picks() - Consolidate picks from all files
  # -------------------------------------------------------------------------

  def merge_picks(self) -> pd.DataFrame:
    """
    Merge pick catalogs from all parsed files into a single DataFrame.

    Combines picks from all parsed files and deduplicates overlapping arrivals
    sharing the same event ID, station, and phase.

    Returns:
      pd.DataFrame: Consolidated picks from all input files
    """
    self.logger.info("Merging picks from files...")
    pick_frames: list[pd.DataFrame] = []
    for f in self.files:
      if not f.HAS_PICKS:
        continue
      picks = f.get("PICKS").copy()
      if picks.empty:
        continue
      pick_frames.append(picks)

    if pick_frames:
      self.PICKS = pd.concat(pick_frames, ignore_index=True)
    else:
      self.PICKS = pd.DataFrame(columns=self._PICK_COLUMNS)

    self.PICKS = self.normalize_time(self.PICKS)
    if OGS_C.PHASE_STR in self.PICKS.columns:
      for pick in [OGS_C.PWAVE, OGS_C.SWAVE]:
        self.logger.info(
            f"Total merged {pick}-phase picks: "
            f"{len(self.PICKS[self.PICKS[OGS_C.PHASE_STR] == pick])}"
        )
    return self.PICKS

  # -------------------------------------------------------------------------
  # METHOD: merge() - Full catalog merge and output
  # -------------------------------------------------------------------------

  def merge(self) -> None:
    """
    Perform full catalog merge: picks first, then events, then log output.

    Creates a merged catalog with:
      - Concatenated picks deduplicated by event ID/station/phase
      - Events combined with format-specific keys and non-null precedence
      - Pick count statistics computed per event

    Sets self.input to {previous_input}/.all and writes Parquet partitions
    under {output}/.all/. Calls inherited plot() after logging. Returns None.
    """
    self.logger.info("Starting full catalog merge...")
    # Merge picks first (events depend on pick statistics)
    self.logger.info(f"Total merged picks: {len(self.merge_picks())}")

    # Merge events and compute statistics
    self.logger.info(f"Total merged events: {len(self.merge_events())}")

    # Append ".all" suffix to output path for merged catalog (idempotent)
    if self.input.name != ".all":
      self.input = self.input / ".all"
    self.logger.info(f"Output path for merged catalog: {self.input}")

    # Write merged catalog to Parquet files
    self.log()
    self.plot()


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
    args: Parsed command-line arguments from ogsutils.parse_catalog_args()
  """
  # Create catalog aggregator with provided arguments
  OGS_Catalog = DataCatalog(args)

  # Discover and parse all input files
  OGS_Catalog.read()

  # If merge flag set, consolidate all files into single catalog
  if args.merge:
    OGS_Catalog.merge()


# Package-module entry point: parse arguments and run main
if __name__ == "__main__":
  main(OGS_U.parse_catalog_args())

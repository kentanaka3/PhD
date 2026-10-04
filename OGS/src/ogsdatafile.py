"""
===============================================================================
OGS Data File Abstractions and Logging Helpers
===============================================================================

OVERVIEW:
This module provides the OGSDataFile class, a subclass-oriented base for
parsing and processing seismic data files from OGS (Istituto Nazionale di
Oceanografia e di Geofisica Sperimentale). It extends OGSCatalog to add file
I/O and regex-based record extraction capabilities.

KEY FEATURES:
  - Regex-based parsing: Uses configurable regex patterns to extract seismic
    picks (phase arrivals) and events (earthquakes) from text-based data files
  - Extensible design: Subclasses define RECORD_EXTRACTOR_LIST and
    EVENT_EXTRACTOR_LIST to handle different file formats
  - Geographic filtering inherited for loaded Parquet event days; builders
    normalize coordinates but do not apply the polygon to parsed rows
  - Parquet output: Persists parsed data in efficient columnar format
  - Debug utilities: Helps identify which regex group fails during parsing

ARCHITECTURE:
  OGSCatalog (base)
  │
  └── OGSDataFile (this class)
      │
      ├── DataFileDAT / DataFileHPL (phase picks)
      ├── DataFilePUN / DataFileTXT (event summaries)
      └── ...

USAGE:
  Format subclasses define the relevant RECORD_EXTRACTOR_LIST and/or
  EVENT_EXTRACTOR_LIST fragments and implement read() to populate aggregate
  PICKS/EVENTS DataFrames and their per-date picks/events dictionaries.

DEPENDENCIES:
  - obspy: Seismological Python library for time handling (UTCDateTime)
  - matplotlib: Used for polygon path operations (geographic filtering)
  - pandas: DataFrames for structured data (via OGSCatalog parent)

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
from .ogscatalog import OGSCatalog
from . import ogsconstants as OGS_C

from abc import ABC, abstractmethod

# Standard library: Regular expressions for pattern matching
import re

# Standard library: Filesystem path handling
from pathlib import Path

# Standard library: Date and time objects for temporal filtering
from datetime import datetime, timedelta as td

# Standard library: Threaded writes for independent date partitions
from concurrent.futures import ThreadPoolExecutor, as_completed

# ObsPy: Seismological library - UTCDateTime for precise earthquake timing
from obspy import UTCDateTime

# Matplotlib: Path object for polygon-based geographic containment tests
from matplotlib.path import Path as mplPath

# Pandas: DataFrame operations, merging, and Parquet I/O
import pandas as pd

# NumPy: Numerical arrays and NaN representation
import numpy as np


# =============================================================================
# OGSDataFile Class
# =============================================================================

class OGSDataFile(OGSCatalog, ABC):
  """
  Abstract base class for parsing OGS seismic data files.

  Provides regex-based extraction of seismic picks and events from various
  text-based file formats. Subclasses define format-specific regex patterns
  and implement the read() method.

  Attributes:
    RECORD_EXTRACTOR_LIST: Regex patterns for individual pick records
    EVENT_EXTRACTOR_LIST: Regex patterns for event header lines
    RECORD_EXTRACTOR: Compiled regex from RECORD_EXTRACTOR_LIST
    EVENT_EXTRACTOR: Compiled regex from EVENT_EXTRACTOR_LIST
    name: File format identifier (uppercase extension, e.g., "HPL")
  """

  # -------------------------------------------------------------------------
  # CLASS ATTRIBUTES (to be overridden by subclasses)
  # -------------------------------------------------------------------------

  # Expected file extension for format validation
  # (e.g. ".dat", ".hpl", ".pun", ".txt")
  EXTENSION: str = ""

  # Default pick table schema columns
  _PICK_COLUMNS = [
      OGS_C.IDX_PICKS_STR, OGS_C.GROUPS_STR, OGS_C.TIME_STR, OGS_C.STATION_STR,
      OGS_C.PHASE_STR, OGS_C.WEIGHT_STR, OGS_C.EPICENTRAL_DISTANCE_STR,
      OGS_C.DEPTH_STR, OGS_C.AMPLITUDE_STR, OGS_C.STATION_ML_STR,
      OGS_C.PROBABILITY_STR
  ]

  # Default event table schema columns
  # (unified 28-column superset across all catalog formats)
  _EVENT_COLUMNS = [
      OGS_C.IDX_EVENTS_STR,            # 0: idx
      OGS_C.TIME_STR,                  # 1: time
      OGS_C.LATITUDE_STR,              # 2: latitude
      OGS_C.LONGITUDE_STR,             # 3: longitude
      OGS_C.DEPTH_STR,                 # 4: depth
      OGS_C.GAP_STR,                   # 5: azimuthal_gap
      OGS_C.ERZ_STR,                   # 6: max_vertical_uncertainty
      OGS_C.ERH_STR,                   # 7: max_horizontal_uncertainty
      OGS_C.ERT_STR,                   # 8: max_time_uncertainty
      OGS_C.GROUPS_STR,                # 9: group
      OGS_C.NO_STR,                    # 10: number_picks
      OGS_C.NUMBER_P_PICKS_STR,        # 11: number_p_picks
      OGS_C.NUMBER_S_PICKS_STR,        # 12: number_s_picks
      OGS_C.NUMBER_P_AND_S_PICKS_STR,  # 13: number_p_and_s_picks
      OGS_C.MAGNITUDE_D_STR,           # 14: Duration magnitude (MD)
      OGS_C.MAGNITUDE_L_STR,           # 15: Local magnitude (ML)
      OGS_C.ML_MEDIAN_STR,             # 16: ML_median
      OGS_C.ML_UNC_STR,                # 17: ML_unc
      OGS_C.ML_STATIONS_STR,           # 18: ML_stations
      OGS_C.DMIN_STR,                  # 19: Minimum distance (D)
      OGS_C.RMS_STR,                   # 20: Root mean square (RMS)
      OGS_C.QM_STR,                    # 21: Quality measure (QM)
      OGS_C.LOC_NAME_STR,              # 22: LOC_NAME
      OGS_C.EVENT_TYPE_STR,            # 23: Event type
      OGS_C.NOTES_STR,                 # 24: NOTES
      OGS_C.MD_UNC_STR,                # 25: MD_unc
      OGS_C.MD_STATIONS_STR,           # 26: MD_stations
      OGS_C.MD_MEDIAN_STR,             # 27: MD_median
  ]

  # List of regex pattern fragments for parsing individual pick/phase records
  # Subclasses populate this with format-specific patterns
  RECORD_EXTRACTOR_LIST: list = []  # TBD in subclasses

  # List of regex pattern fragments for parsing event header lines
  # Subclasses populate this with format-specific patterns
  EVENT_EXTRACTOR_LIST: list = []   # TBD in subclasses

  # Regex to extract named group identifiers from regex patterns
  # Used by debug() to select a suspected capture-group failure
  # Matches patterns like: (?P<station>[\w]+) and extracts "station"
  GROUP_PATTERN = re.compile(r"\(\?P<(\w+)>[\[\]\w\d\{\}\-\\\?\+]+\)(\w)*")

  @staticmethod
  def _flatten(iterable):
    """Recursively flatten nested iterables of strings into a flat generator."""
    for item in iterable:
      if isinstance(item, str):
        yield item
      else:
        yield from OGSDataFile._flatten(item)

  # -------------------------------------------------------------------------
  # CONSTRUCTOR
  # -------------------------------------------------------------------------

  def __init__(
      self, input: Path, start: datetime = datetime.max,
      end: datetime = datetime.min, verbose: bool = False,
      polygon: mplPath = mplPath(OGS_C.OGS_POLY_REGION, closed=True),
      output: Path = OGS_C.THIS_FILE.parent / "data" / "OGSCatalog"
  ):
    """
    Initialize the data file wrapper and compile regex extractors.

    Args:
      input: Path to the input data file to parse
      start: Start datetime for temporal filtering (default: datetime.max)
      end: End datetime for temporal filtering (default: datetime.min)
      verbose: Enable verbose logging output (default: False)
      polygon: matplotlib Path defining geographic region of interest
                (default: OGS regional polygon from constants)
      output: Directory for output files (default: OGS/src/data/OGSCatalog)

    The parent checks input existence and creates output/img directories.
    Default start/end bounds are reversed; pass a valid range to retain rows.
    """
    # Initialize parent class with catalog management capabilities
    super().__init__(input, start, end, verbose, polygon, output)

    # Compile the record extractor regex from the list of pattern fragments
    # _flatten handles nested lists, join concatenates all fragments
    self.RECORD_EXTRACTOR: re.Pattern = re.compile(OGS_C.EMPTY_STR.join(
        list(self._flatten(self.RECORD_EXTRACTOR_LIST))
    ))  # Patterns supplied by subclasses

    # Compile the event extractor regex from the list of pattern fragments
    self.EVENT_EXTRACTOR: re.Pattern = re.compile(OGS_C.EMPTY_STR.join(
        list(self._flatten(self.EVENT_EXTRACTOR_LIST))
    ))   # Patterns supplied by subclasses

    # Extract file format name from extension (e.g., ".hpl" -> "HPL")
    self.name = self.input.suffix.lstrip(OGS_C.PERIOD_STR).upper()

  # -------------------------------------------------------------------------
  # ABSTRACT METHOD: read()
  # -------------------------------------------------------------------------

  @abstractmethod
  def read(self):
    """
    Read and parse the input data file into picks and events.

    Subclasses must implement this abstract method to be instantiated.
    The implementation should:
      1. Open and read the input file
      2. Use RECORD_EXTRACTOR to parse pick/phase records
      3. Use EVENT_EXTRACTOR to parse event headers
      4. Populate self.PICKS / self.EVENTS and postload per-date dictionaries

    Raises:
      NotImplementedError: Always, as subclasses must override this method
    """
    raise NotImplementedError

  def log_file(self, file_type: str, date: datetime, df: pd.DataFrame):
    """
    Log a single DataFrame to the appropriate Parquet file based on file type
    and date.

    Args:
      file_type (str): Type of data being logged ("picks" or "events").
      date (datetime): Date associated with the data.
      df (pd.DataFrame): DataFrame containing the data to be logged.
    """
    # Convert date key to Python date object for path construction
    date = UTCDateTime(date).date

    # Construct base output path using file extension as subdirectory
    log = self.output / self.input.suffix

    # Preserve the historical picks output directory name
    subdirectory = "assignments" if file_type == "picks" else file_type

    # Build date-based directory path
    dir_path = log / subdirectory / OGS_C.DASH_STR.join([
        f"{date.year}", f"{date.month:02}", f"{date.day:02}"
    ])

    # Create parent directories if they don't exist
    dir_path.parent.mkdir(parents=True, exist_ok=True)

    # Write DataFrame to Parquet format
    df.to_parquet(dir_path, index=False)
    self.logger.debug(f"Saved {file_type.upper()} for {date} to {dir_path}")

  # -------------------------------------------------------------------------
  # METHOD: log()
  # -------------------------------------------------------------------------
  def log(self):
    """
    Persist parsed picks and events to the output directory as Parquet files.

    Organizes output in a date-based directory structure:
      {output}/{extension}/assignments/{year}-{month}-{day}  (for picks)
      {output}/{extension}/events/{year}-{month}-{day}       (for events)

    Uses Parquet format for efficient columnar storage and fast I/O.
    Rebuilds daily caches from non-empty aggregates before threaded writes.
    Individual worker failures are logged rather than re-raised. Returns None.
    """
    tasks = [
        (key, date, df)
        for key in ("picks", "events")
        for date, df in self.postload(key, update=True).items()
    ]
    if not tasks:
      return

    max_workers = max(1, min(OGS_C.DEFAULT_CORES_COUNT, len(tasks)))
    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
      futures = {
          executor.submit(self.log_file, key, date, df): (key, date)
          for key, date, df in tasks
      }
      for future in as_completed(futures):
        key, date = futures[future]
        try:
          future.result()
        except Exception as exc:
          failures.append(exc)
          self.logger.exception(f"Failed to save {key.upper()} for {date}")

    for failure in failures:
      self.logger.error(f"Failure encountered: {failure}")

  # -------------------------------------------------------------------------
  # METHOD: debug()
  # -------------------------------------------------------------------------

  def debug(self, line, EXTRACTOR_LIST):
    """
    Identify which regex capture group fails to match a given input line.

    Flatten nested fragments and test syntactically valid cumulative prefixes.
    Partial prefixes may leave groups open; their compilation errors are logged
    and only those prefixes are skipped. The full extractor must compile.
    Named groups added by the next longer valid prefix identify a suspected
    field, not a proven cause of the mismatch.

    Args:
      line: The input line that failed to match the full regex
      EXTRACTOR_LIST: The list of regex pattern fragments to debug

    Returns:
      str or list: A suspected named field, or [] when no field is
        identifiable. The input line and diagnosis are always logged.

    Raises:
      re.error: If the full extractor is invalid.
    """
    # Build reversed cumulative list of regex patterns for progressive testing
    # This creates patterns of decreasing length to isolate the failure point
    # Example: [full_pattern, pattern_minus_last, pattern_minus_last_two, ...]
    RECORD_EXTRACTOR_DEBUG = list(reversed(list(it.accumulate(
        EXTRACTOR_LIST[:-1],
        lambda x, y: x + (
            y if isinstance(y, str)
            else OGS_C.EMPTY_STR.join(list(self._flatten(y)))
        )
    ))))

    # Default to first group as the suspected failure point
    bug = self.GROUP_PATTERN.findall(EXTRACTOR_LIST[0])

    # Iterate through progressively shorter patterns
    for i, extractor in enumerate(RECORD_EXTRACTOR_DEBUG):
      # Try to match current (shorter) pattern against input line
      match_extractor = re.match(extractor, line)

      if match_extractor:
        # Compare captures with the previously tested pattern as a heuristic
        match_group = self.GROUP_PATTERN.findall(RECORD_EXTRACTOR_DEBUG[i - 1])
        match_compare = self.GROUP_PATTERN.findall(extractor)

        # Select a tuple element from the final named-group capture
        bug = match_group[-1][match_group[-1][1] != match_compare[-1][1]]

        # Log the failure for debugging purposes
        self.logger.warning(f"{self.input.suffix} {bug} : {line}")
        break

    return bug

  # -------------------------------------------------------------------------
  # GROUP, TIMESTAMP, AND INDEX NORMALIZATION HELPERS
  # -------------------------------------------------------------------------

  @staticmethod
  def normalize_time(
      dataframe: pd.DataFrame, time_col: str = OGS_C.TIME_STR
  ) -> pd.DataFrame:
    """
    Coerce the timestamp column with pandas.to_datetime(errors="coerce").

    Handles strings, Python datetime objects, and missing values. Invalid
    values become NaT; timezone/dtype behavior follows pandas. Updates the
    supplied DataFrame and returns it.
    """
    if dataframe.empty or time_col not in dataframe.columns:
      return dataframe
    dataframe[time_col] = pd.to_datetime(dataframe[time_col], errors='coerce')
    return dataframe
  # -------------------------------------------------------------------------
  # SHARED PARSING UTILITIES
  # -------------------------------------------------------------------------

  def validate_input(self) -> None:
    """
    Validate that the input file exists and matches the expected format
    extension.
    """
    if not self.input.exists():
      raise FileNotFoundError(f"File {self.input} does not exist")
    if self.EXTENSION:
      if self.input.suffix != self.EXTENSION:
        raise ValueError(f"File extension must be {self.EXTENSION}")

  def _build_picks_dataframe(self, picks_data: list) -> pd.DataFrame:
    return pd.DataFrame(picks_data, columns=self._PICK_COLUMNS).astype({
        OGS_C.IDX_PICKS_STR: int
    })

  @staticmethod
  def _build_pick_row(
      event_index, pick_time: datetime, station: str, phase: str, weight: int
  ) -> list:
    """Create a standardized pick row for the output DataFrame."""
    return [
        event_index,                          # OGS_C.IDX_PICKS_STR
        pick_time.strftime(OGS_C.DATE_FMT),   # OGS_C.GROUPS_STR
        pick_time,                            # OGS_C.TIME_STR
        f".{station}.",                       # OGS_C.STATION_STR
        phase,                                # OGS_C.PHASE_STR
        weight,                               # OGS_C.WEIGHT_STR
        None,                                 # OGS_C.EPICENTRAL_DISTANCE_STR
        None,                                 # OGS_C.DEPTH_STR
        None,                                 # OGS_C.AMPLITUDE_STR
        None,                                 # OGS_C.STATION_ML_STR
        1.0,                                  # OGS_C.PROBABILITY_STR
    ]

  def _is_before_start(self, value: datetime) -> bool:
    """Check if the given datetime is before the configured start date."""
    return self.start is not None and value < self.start

  def _is_after_end(self, value: datetime) -> bool:
    """Check the exclusive end + ONE_DAY bound, with overflow fallback."""
    if self.end is None:
      return False
    try:
      return value >= self.end + OGS_C.ONE_DAY
    except OverflowError:
      return value > self.end

  @staticmethod
  def _parse_seconds(value: str) -> td:
    """Convert a fixed-width seconds field into a timedelta."""
    return td(seconds=float(value.replace(OGS_C.SPACE_STR, OGS_C.ZERO_STR)))

  @staticmethod
  def _parse_float(value: str, default_value: float | None = None):
    """Convert to float; blank or invalid strings return default_value."""
    if not value or value.strip(OGS_C.SPACE_STR) == OGS_C.EMPTY_STR:
      return default_value
    return float(value)

  @staticmethod
  def _parse_zero_padded_float(value: str) -> float:
    """Convert a numeric field, mapping placeholder-only values to NaN."""
    normalized = value.strip()
    if not any(character.isdigit() for character in normalized):
      return float("nan")
    return float(normalized.replace(OGS_C.SPACE_STR, OGS_C.ZERO_STR))

  @staticmethod
  def _parse_zero_padded_int(value: str, default_value: int | None = None):
    """Convert a fixed-width integer field, preserving blank values as default."""
    if not value or value.strip(OGS_C.SPACE_STR) == OGS_C.EMPTY_STR:
      return default_value
    return int(value.replace(OGS_C.SPACE_STR, OGS_C.ZERO_STR))

  @staticmethod
  def _parse_coordinate(value: str, round_decimals: int | None = None):
    """Convert a degree-minute coordinate string (DD-MM.MM) to decimal degrees."""
    if not value:
      return OGS_C.NONE_STR

    normalized = value.replace(OGS_C.SPACE_STR, OGS_C.ZERO_STR)
    if OGS_C.DASH_STR not in normalized:
      return OGS_C.NONE_STR

    degrees, minutes = normalized.split(OGS_C.DASH_STR, maxsplit=1)
    coord = float(degrees) + float(minutes) / 60.0
    if round_decimals is not None:
      return float(f"{coord:.{round_decimals}f}")
    return coord

  @staticmethod
  def _parse_weight(value: str, default_value: int = 0) -> int:
    """Convert a weight field, using default_value for blanks."""
    if not value or value.strip(OGS_C.SPACE_STR) == OGS_C.EMPTY_STR:
      return default_value
    return int(value)

  @staticmethod
  def _parse_index(
      value: str,
      year: int,
      year_stride: int | float = OGS_C.MAX_PICKS_YEAR,
  ):
    """Build a globally unique index using the configured yearly stride."""
    if not value:
      return None
    return (
        int(value.replace(OGS_C.SPACE_STR, OGS_C.ZERO_STR)) +
        year * year_stride
    )

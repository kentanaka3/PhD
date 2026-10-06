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
  text-based file formats. Subclasses define format-specific regex patterns and
  implement the read() method.

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

  @staticmethod
  def normalize_index(
      value: str | int | None,
      year: int,
      year_stride: int | float = OGS_C.MAX_PICKS_YEAR,
  ) -> int | None:
    """
    Combine a calendar year and local index using the configured stride.

    Encodes the year into the most significant digits using a unified stride
    (default: MAX_PICKS_YEAR = 1,000,000). Values already at or above
    year * stride are returned unchanged; None/blank inputs return None.
    Uniqueness depends on caller indices staying within their year's stride.
    """
    if value is None:
      return None
    if isinstance(value, str):
      cleaned = value.strip()
      if not cleaned:
        return None
      val_int = int(cleaned.replace(OGS_C.SPACE_STR, OGS_C.ZERO_STR))
    else:
      val_int = int(value)

    year_stride_int = int(year_stride)
    if year > 0 and val_int >= year * year_stride_int:
      return val_int

    return int(val_int + year * year_stride_int)

  @staticmethod
  def _vectorized_to_decimal(series: pd.Series) -> pd.Series:
    """
    Vectorized conversion of coordinates (degrees-minutes 'DD-MM.mm' or
    decimal).
    """
    if series.empty or pd.api.types.is_numeric_dtype(series):
      return pd.to_numeric(series, errors="coerce")

    numeric = pd.to_numeric(series, errors="coerce")
    unparsed_mask = numeric.isna() & series.notna()
    if not unparsed_mask.any():
      return numeric

    s_unparsed = series[unparsed_mask].astype(str).str.strip()
    valid_unparsed = s_unparsed[~s_unparsed.isin(("", "None", "nan", "NaN"))]
    if valid_unparsed.empty:
      return numeric

    is_neg = valid_unparsed.str.startswith("-")
    clean = valid_unparsed.str.lstrip("-")
    parts = clean.str.split("-", n=1, expand=True)

    if parts.shape[1] == 2:
      deg = pd.to_numeric(parts[0], errors="coerce")
      minutes = pd.to_numeric(parts[1], errors="coerce")
      converted = deg + (minutes / 60.0)
      converted = converted.where(~is_neg, -converted)
      numeric = numeric.combine_first(converted)

    return numeric

  @staticmethod
  def normalize_coordinates(
      dataframe: pd.DataFrame,
      lat_col: str = OGS_C.LATITUDE_STR,
      lon_col: str = OGS_C.LONGITUDE_STR,
      depth_col: str = OGS_C.DEPTH_STR,
      round_decimals: int = 4,
      error_cols: tuple[str, ...] = (
          OGS_C.ERH_STR, OGS_C.ERZ_STR, OGS_C.ERT_STR,
          OGS_C.RMS_STR, OGS_C.GAP_STR, OGS_C.DMIN_STR,
      ),
  ) -> pd.DataFrame:
    """
    Normalize hypocenter coordinates and error metrics to standard numeric
    types.

    Operations:
      - Converts coordinates (handling degree-minute strings 'DD-MM.MM' or
        decimal values) to numeric values, rounded to `round_decimals`
        (default: 4).
      - Validates physical bounds (-90 <= lat <= 90, -180 <= lon <= 180),
        setting out-of-bounds to NaN.
      - Coerces depth to a pandas numeric dtype.
      - Coerces error metrics (ERH, ERZ, ERT, RMS, GAP, DMIN) to numeric
        values, converting empty strings, dash sequences, and string 'None'
        into np.nan.

    Returns:
      pd.DataFrame: The supplied frame, updated in place with normalized
        coordinates and numeric errors.
    """
    if dataframe.empty:
      return dataframe

    # Normalize Latitude
    if lat_col in dataframe.columns:
      lat_series = OGSDataFile._vectorized_to_decimal(dataframe[lat_col])
      # Bounds validation: [-90, 90]
      lat_series = lat_series.where(
          (lat_series >= -90.0) & (lat_series <= 90.0), np.nan
      )
      dataframe[lat_col] = lat_series.round(round_decimals)

    # Normalize Longitude
    if lon_col in dataframe.columns:
      lon_series = OGSDataFile._vectorized_to_decimal(dataframe[lon_col])
      # Bounds validation: [-180, 180]
      lon_series = lon_series.where(
          (lon_series >= -180.0) & (lon_series <= 180.0), np.nan
      )
      dataframe[lon_col] = lon_series.round(round_decimals)

    # Normalize Depth
    if depth_col in dataframe.columns:
      dataframe[depth_col] = pd.to_numeric(
          dataframe[depth_col], errors='coerce'
      )

    # Normalize Error Metrics
    for col in error_cols:
      if col in dataframe.columns:
        dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')

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
    try:
      return float(value.replace(OGS_C.SPACE_STR, OGS_C.ZERO_STR))
    except ValueError:
      return default_value

  @staticmethod
  def _parse_int(value: str, default_value: int | None = None):
    """Zero-fill spaces and parse int; blank/invalid strings return default."""
    if not value or value.strip(OGS_C.SPACE_STR) == OGS_C.EMPTY_STR:
      return default_value
    try:
      return int(value.replace(OGS_C.SPACE_STR, OGS_C.ZERO_STR))
    except ValueError:
      return default_value

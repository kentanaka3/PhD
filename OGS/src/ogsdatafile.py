"""
=============================================================================
OGS Data File Abstractions and Logging Helpers
=============================================================================

OVERVIEW:
This module provides the OGSDataFile class, an abstract base class for parsing
and processing seismic data files from OGS (Istituto Nazionale di Oceanografia
e di Geofisica Sperimentale). It extends OGSCatalog to add file I/O and regex-
based record extraction capabilities.

KEY FEATURES:
  - Regex-based parsing: Uses configurable regex patterns to extract seismic
    picks (phase arrivals) and events (earthquakes) from text-based data files
  - Extensible design: Subclasses define RECORD_EXTRACTOR_LIST and
    EVENT_EXTRACTOR_LIST to handle different file formats
  - Geographic filtering: Supports polygon-based spatial filtering of events
  - Parquet output: Persists parsed data in efficient columnar format
  - Debug utilities: Helps identify which regex group fails during parsing

ARCHITECTURE:
  OGSCatalog (base)
  │
  └── OGSDataFile (this class)
      │
      ├── Subclass for format A (e.g., .hyp files)
      ├── Subclass for format B (e.g., .cnv files)
      └── ...

USAGE:
  Subclasses must:
    1. Define RECORD_EXTRACTOR_LIST: List of regex patterns for pick records
    2. Define EVENT_EXTRACTOR_LIST: List of regex patterns for event headers
    3. Implement read(): Parse input file into picks/events DataFrames

DEPENDENCIES:
  - obspy: Seismological Python library for time handling (UTCDateTime)
  - matplotlib: Used for polygon path operations (geographic filtering)
  - pandas: DataFrames for structured data (via OGSCatalog parent)

=============================================================================
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

# Standard library: Regular expressions for pattern matching
import re

# Standard library: Functional programming utilities (accumulate for debugging)
import itertools as it

# Standard library: Filesystem path handling
from pathlib import Path

# Standard library: Date and time objects for temporal filtering
from datetime import datetime

# ObsPy: Seismological library - UTCDateTime for precise earthquake timing
from obspy import UTCDateTime

# Matplotlib: Path object for polygon-based geographic containment tests
from matplotlib.path import Path as mplPath

# Utility to flatten nested lists (for regex pattern assembly)
def _flatten(iterable):
  """Recursively flatten nested iterables of strings into a flat generator."""
  for item in iterable:
    if isinstance(item, str):
      yield item
    else:
      yield from _flatten(item)

# Local module: OGS-specific constants (strings, default paths, regions)
import ogsconstants as OGS_C

# Local module: Parent class providing catalog data structures and methods
from ogscatalog import OGSCatalog


# =============================================================================
# OGSDataFile Class
# =============================================================================

class OGSDataFile(OGSCatalog):
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
    name: File format identifier (uppercase extension, e.g., "HYP")
  """

  # -------------------------------------------------------------------------
  # CLASS ATTRIBUTES (to be overridden by subclasses)
  # -------------------------------------------------------------------------

  # List of regex pattern fragments for parsing individual pick/phase records
  # Subclasses populate this with format-specific patterns
  RECORD_EXTRACTOR_LIST : list = []  # TBD in subclasses

  # List of regex pattern fragments for parsing event header lines
  # Subclasses populate this with format-specific patterns
  EVENT_EXTRACTOR_LIST : list = []   # TBD in subclasses

  # Regex to extract named group identifiers from regex patterns
  # Used by debug() to identify which capture group failed matching
  # Matches patterns like: (?P<station>[\w]+) and extracts "station"
  GROUP_PATTERN = re.compile(r"\(\?P<(\w+)>[\[\]\w\d\{\}\-\\\?\+]+\)(\w)*")

  # -------------------------------------------------------------------------
  # CONSTRUCTOR
  # -------------------------------------------------------------------------

  def __init__(self, input: Path, start: datetime = datetime.max,
               end: datetime = datetime.min, verbose: bool = False,
               polygon : mplPath = mplPath(OGS_C.OGS_POLY_REGION, closed=True),
               output : Path = OGS_C.THIS_FILE.parent / "data" / "OGSCatalog"):
    """
    Initialize the data file wrapper and compile regex extractors.

    Args:
      input: Path to the input data file to parse
      start: Start datetime for temporal filtering (default: datetime.max)
      end: End datetime for temporal filtering (default: datetime.min)
      verbose: Enable verbose logging output (default: False)
      polygon: matplotlib Path defining geographic region of interest
                (default: OGS regional polygon from constants)
      output: Directory path for output files (default: data/OGSCatalog)
    """
    # Initialize parent class with catalog management capabilities
    super().__init__(input, start, end, verbose, polygon, output)

    # Compile the record extractor regex from the list of pattern fragments
    # _flatten handles nested lists, join concatenates all fragments
    self.RECORD_EXTRACTOR : re.Pattern = re.compile(OGS_C.EMPTY_STR.join(
      list(_flatten(self.RECORD_EXTRACTOR_LIST))))  # TBD in subclasses

    # Compile the event extractor regex from the list of pattern fragments
    self.EVENT_EXTRACTOR : re.Pattern = re.compile(OGS_C.EMPTY_STR.join(
      list(_flatten(self.EVENT_EXTRACTOR_LIST))))   # TBD in subclasses

    # Extract file format name from extension (e.g., ".hyp" -> "HYP")
    self.name = self.input.suffix.lstrip(OGS_C.PERIOD_STR).upper()

  # -------------------------------------------------------------------------
  # ABSTRACT METHOD: read()
  # -------------------------------------------------------------------------

  def read(self):
    """
    Read and parse the input data file into picks and events.

    This is an abstract method that must be implemented by subclasses.
    The implementation should:
      1. Open and read the input file
      2. Use RECORD_EXTRACTOR to parse pick/phase records
      3. Use EVENT_EXTRACTOR to parse event headers
      4. Populate self.picks and self.events DataFrames

    Raises:
      NotImplementedError: Always, as subclasses must override this method
    """
    raise NotImplementedError

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
    """
    # Construct base output path using file extension as subdirectory
    log = self.output / self.input.suffix
    self.logger.info(f"Logging data to: {log}")

    # ----- SAVE PICKS (phase arrival assignments) -----
    # Iterate over picks grouped by date from postload() method
    for date, df in self.postload("picks").items():
      # Convert date key to Python date object for path construction
      date = UTCDateTime(date).date

      # Build date-based directory path: {log}/assignments/YYYY-MM-DD
      dir_path = log / "assignments" / OGS_C.DASH_STR.join([
        f"{date.year}", f"{date.month:02}", f"{date.day:02}"])

      # Create parent directories if they don't exist
      dir_path.parent.mkdir(parents=True, exist_ok=True)

      # Write DataFrame to Parquet format (efficient columnar storage)
      df.to_parquet(dir_path, index=False)
      self.logger.debug(f"Saved PICKS for {date} to {dir_path}")

    # ----- SAVE EVENTS (earthquake catalog entries) -----
    # Iterate over events grouped by date from postload() method
    for date, df in self.postload("events").items():
      # Build date-based directory path: {log}/events/YYYY-MM-DD
      dir_path = log / "events" / OGS_C.DASH_STR.join([
        f"{date.year}", f"{date.month:02}", f"{date.day:02}"])

      # Create parent directories if they don't exist
      dir_path.parent.mkdir(parents=True, exist_ok=True)

      # Write DataFrame to Parquet format
      df.to_parquet(dir_path, index=False)
      self.logger.debug(f"Saved EVENTS for {date} to {dir_path}")

  # -------------------------------------------------------------------------
  # METHOD: debug()
  # -------------------------------------------------------------------------

  def debug(self, line, EXTRACTOR_LIST):
    """
    Identify which regex capture group fails to match a given input line.

    This debugging utility helps diagnose parsing failures by progressively
    testing truncated versions of the regex pattern to find the exact point
    of failure. Useful when adding support for new file formats or handling
    malformed input data.

    Algorithm:
      1. Build a list of progressively shorter regex patterns (reversed accumulation)
      2. Test each pattern against the input line
      3. When a match succeeds, the previous (longer) pattern's last group
         is the one that failed

    Args:
      line: The input line that failed to match the full regex
      EXTRACTOR_LIST: The list of regex pattern fragments to debug

    Returns:
      str: The name of the regex capture group that caused the match failure
    """
    # Build reversed cumulative list of regex patterns for progressive testing
    # This creates patterns of decreasing length to isolate the failure point
    # Example: [full_pattern, pattern_minus_last, pattern_minus_last_two, ...]
    RECORD_EXTRACTOR_DEBUG = list(reversed(list(it.accumulate(
      EXTRACTOR_LIST[:-1],
      lambda x, y: x + (y if isinstance(y, str) else
                        OGS_C.EMPTY_STR.join(list(_flatten(y))))))))

    # Default to first group as the suspected failure point
    bug = self.GROUP_PATTERN.findall(EXTRACTOR_LIST[0])

    # Iterate through progressively shorter patterns
    for i, extractor in enumerate(RECORD_EXTRACTOR_DEBUG):
      # Try to match current (shorter) pattern against input line
      match_extractor = re.match(extractor, line)

      if match_extractor:
        # Match succeeded! The failure is in the next group (previous pattern)
        # Extract named groups from the pattern that just worked
        match_group = self.GROUP_PATTERN.findall(RECORD_EXTRACTOR_DEBUG[i - 1])
        match_compare = self.GROUP_PATTERN.findall(extractor)

        # Identify the differing group between the two patterns
        # This is the group that caused the failure
        bug = match_group[-1][match_group[-1][1] != match_compare[-1][1]]

        # Log the failure for debugging purposes
        self.logger.warning(f"{self.input.suffix} {bug} : {line}")
        break

    return bug
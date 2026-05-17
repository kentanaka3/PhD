"""
=============================================================================
OGS TXT File Parser - Catalog Event Summary Extractor
=============================================================================

OVERVIEW:
This module parses OGS .txt catalog exports containing event-level summaries.
Each data row represents one located event with origin time, hypocenter,
uncertainty estimates, magnitudes, locality name, and an event-type label.

FILE FORMAT DESCRIPTION:
  The .txt format stores one fixed-width summary record per event with:
  - Event index and legacy catalog identifier
  - ISO origin time (YYYY-MM-DDTHH:MM:SS.mmm)
  - ERT, latitude, longitude, ERH, depth, ERZ, and GAP quality fields
  - Local magnitude (ML) and duration magnitude (MD)
  - Human-readable locality name
  - Bracketed event type label used for downstream filtering
  - A header line that is skipped before parsing begins

KEY FEATURES:
  - Regex-based extraction with named capture groups
  - Date range filtering for temporal subsetting
  - Post-processing of placeholder dashes into numeric NaN values
  - Event type filtering for suspected explosions
  - Parquet output via the shared OGSDataFile logging pipeline

USAGE:
  Command line:
    python ogstxt.py -f input.txt -D 20240320 20240620 -v

  Programmatic:
    from ogstxt import DataFileTXT
    parser = DataFileTXT(Path("input.txt"), start_date, end_date)
    parser.read()
    parser.log()

OUTPUT:
  - self.EVENTS: DataFrame with event origin, geometry, magnitudes, and labels
  - self.events: Dict mapping dates to grouped event DataFrames

DEPENDENCIES:
  - pandas: DataFrame operations and Parquet I/O
  - obspy: UTCDateTime for seismological time handling
  - ogsconstants: OGS-specific constants and regex fragments
  - ogsdatafile: Base class providing regex extraction and logging helpers

=============================================================================
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

# Standard library: command-line argument parsing
import argparse

# Standard library: filesystem path handling
from pathlib import Path

# ObsPy: seismological time conversions
from obspy import UTCDateTime

# Standard library: date/time objects
from datetime import datetime

# Pandas: tabular data manipulation
import pandas as pd

# Local module: OGS-specific constants and formatting strings
import ogsconstants as OGS_C

# Local module: OGS-specific argument parsing helpers
import ogsutils as OGS_U

# Local module: base parser for extraction and logging
from ogsdatafile import OGSDataFile

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

# Base path for data files (two levels up from this script's location)
DATA_PATH = Path(__file__).parent.parent.parent


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_arguments():
  """
  Parse command-line arguments for the TXT file processor.

  Returns:
    argparse.Namespace with:
      - file: List of Path objects to input .txt files
      - dates: Tuple of (start_date, end_date) for filtering
      - verbose: Boolean flag for debug output
  """
  parser = argparse.ArgumentParser(description="Run OGS TXT quality checks")

  # -f/--file: Input file path(s), required, accepts multiple files
  parser.add_argument(
    "-f", "--file", type=Path, required=True, nargs=OGS_C.ONE_MORECHAR_STR,
    help="Path to the input file")

  # -D/--dates: Date range filter, optional, format YYYYMMDD
  parser.add_argument(
    '-D', "--dates", required=False, metavar=OGS_C.DATE_STD,
    type=OGS_U.is_date, nargs=2, action=OGS_U.SortDatesAction,
    default=[datetime.strptime("20240320", OGS_C.YYYYMMDD_FMT),
             datetime.strptime("20240620", OGS_C.YYYYMMDD_FMT)],
    help="Specify the beginning and ending (inclusive) Gregorian date " \
         "(YYYYMMDD) range to work with.")

  # -v/--verbose: Enable detailed logging output
  parser.add_argument(
    '-v', "--verbose", action='store_true', default=False,
    help="Enable verbose output")

  return parser.parse_args()


# =============================================================================
# DataFileTXT Class - TXT Format Parser
# =============================================================================

class DataFileTXT(OGSDataFile):
  """
  Parser for OGS .txt catalog event summary files.

  Extends OGSDataFile to parse text exports where each line describes a single
  event with origin time, location, magnitude estimates, and a classification
  label used by later filtering stages.

  Attributes:
    RECORD_EXTRACTOR_LIST: Regex patterns for individual event summary lines
  """

  # -------------------------------------------------------------------------
  # RECORD EXTRACTOR: fixed-width event summary line
  # -------------------------------------------------------------------------
  RECORD_EXTRACTOR_LIST = [
    fr"^(?P<{OGS_C.INDEX_STR}>\d{{5}})\s",                        # Index
    fr"\d{{4}}_\d{{5}}\s",                                        # Legacy
    fr"(?P<{OGS_C.TIME_STR}>\d{{4}}-\d{{2}}-\d{{2}}T",            # Date
    fr"\d{{2}}:\d{{2}}:\d{{2}}\.\d{{3}})\s",                      # Time
    fr"(?P<{OGS_C.ERT_STR}>[\s\d\.\-]{{5}})\s",                   # ERT
    fr"(?P<{OGS_C.LATITUDE_STR}>[\s\d\-\.]{{7}})\s",              # Latitude
    fr"(?P<{OGS_C.LONGITUDE_STR}>[\s\d\-\.]{{7}})\s",             # Longitude
    fr"(?P<{OGS_C.ERH_STR}>[\s\d\.\-]{{5}})\s",                   # ERH
    fr"(?P<{OGS_C.DEPTH_STR}>[\s\d\.\-]{{5}})\s",                 # Depth
    fr"(?P<{OGS_C.ERZ_STR}>[\s\d\.\-]{{5}})\s",                   # ERZ
    fr"(?P<{OGS_C.GAP_STR}>([\s\d\-]{{3}}))\s",                   # GAP
    fr"(?P<{OGS_C.MAGNITUDE_L_STR}>([\-\s\d\.]{{4}}))\s",         # ML
    fr"(?P<{OGS_C.MAGNITUDE_D_STR}>([\-\s\d\.]{{4}}))\s",         # MD
    fr"(?P<{OGS_C.LOC_NAME_STR}>['\.\-\w\s\(\)]+)\s",             # Place
    fr"(?P<{OGS_C.EVENT_TYPE_STR}>\[.*\])$",                      # Event Type
  ]

  @staticmethod
  def _parse_numeric_series(series: pd.Series, missing_marker: str) -> pd.Series:
    """Convert placeholder-marked numeric strings to float values."""
    return series.replace(missing_marker, "NaN").apply(float)

  def read(self):
    """
    Read and parse a .txt catalog file into an event summary DataFrame.

    The parser skips the header row, extracts one event per remaining line,
    filters by date range, normalizes placeholder values, and stores grouped
    results in self.EVENTS and self.events.

    Raises:
      FileNotFoundError: If the input file does not exist.
      ValueError: If the input file does not use the .txt extension.
    """
    # -----------------------------------------------------------------------
    # INPUT VALIDATION
    # -----------------------------------------------------------------------
    if not self.input.exists():
      raise FileNotFoundError(f"File {self.input} does not exist")

    if self.input.suffix != OGS_C.TXT_EXT:
      raise ValueError(f"File extension must be {OGS_C.TXT_EXT}")

    # -----------------------------------------------------------------------
    # FILE READING
    # -----------------------------------------------------------------------
    records = list()

    # The first row is a header line, so parsing starts from the second line.
    with open(self.input, 'r') as fr:
      lines = fr.readlines()[1:]
    self.logger.info(f"Reading TXT file: {self.input}")

    # -----------------------------------------------------------------------
    # LINE-BY-LINE PARSING
    # -----------------------------------------------------------------------
    for line in [raw_line.strip() for raw_line in lines]:
      if line == OGS_C.EMPTY_STR:
        continue

      match = self.RECORD_EXTRACTOR.match(line)
      if not match:
        self.logger.error(f"ERROR: (TXT) Could not parse line: {line}")
        self.debug(line, self.RECORD_EXTRACTOR_LIST)
        continue

      result: dict = match.groupdict()
      result[OGS_C.TIME_STR] = datetime.fromisoformat(result[OGS_C.TIME_STR])

      # ---------------------------------------------------------------------
      # DATE RANGE FILTERING
      # ---------------------------------------------------------------------
      if self.start is not None and result[OGS_C.TIME_STR] < self.start:
        self.logger.debug(f"Skipping event before start date: {self.start}")
        self.logger.debug(line)
        continue

      if (self.end is not None and
          result[OGS_C.TIME_STR] > self.end + OGS_C.ONE_DAY):
        self.logger.debug(f"Stopping read at event after end date: {self.end}")
        self.logger.debug(line)
        break

      # ---------------------------------------------------------------------
      # APPEND RAW EVENT SUMMARY TO RESULTS
      # ---------------------------------------------------------------------
      records.append([
        result[OGS_C.INDEX_STR],
        result[OGS_C.TIME_STR],
        result[OGS_C.ERT_STR],
        result[OGS_C.LATITUDE_STR],
        result[OGS_C.LONGITUDE_STR],
        result[OGS_C.ERH_STR],
        result[OGS_C.DEPTH_STR],
        result[OGS_C.ERZ_STR],
        result[OGS_C.GAP_STR],
        result[OGS_C.MAGNITUDE_L_STR],
        result[OGS_C.MAGNITUDE_D_STR],
        result[OGS_C.LOC_NAME_STR],
        result[OGS_C.EVENT_TYPE_STR],
      ])

    # -----------------------------------------------------------------------
    # BUILD OUTPUT DATAFRAME
    # -----------------------------------------------------------------------
    self.EVENTS = pd.DataFrame(records, columns=[
      OGS_C.INDEX_STR,
      OGS_C.TIME_STR,
      OGS_C.ERT_STR,
      OGS_C.LATITUDE_STR,
      OGS_C.LONGITUDE_STR,
      OGS_C.ERH_STR,
      OGS_C.DEPTH_STR,
      OGS_C.ERZ_STR,
      OGS_C.GAP_STR,
      OGS_C.MAGNITUDE_L_STR,
      OGS_C.MAGNITUDE_D_STR,
      OGS_C.LOC_NAME_STR,
      OGS_C.EVENT_TYPE_STR,
    ])

    if self.EVENTS.empty:
      self.logger.warning(f"No valid TXT records found in {self.input}")
      return

    time_series = pd.to_datetime(self.EVENTS[OGS_C.TIME_STR])
    self.EVENTS[OGS_C.TIME_STR] = time_series
    self.EVENTS[OGS_C.INDEX_STR] = self.EVENTS[OGS_C.INDEX_STR].apply(int) + \
      time_series.dt.year * OGS_C.MAX_PICKS_YEAR
    self.EVENTS[OGS_C.GROUPS_STR] = time_series.dt.date

    numeric_columns = [
      (OGS_C.ERT_STR, OGS_C.DASH_STR * 5),
      (OGS_C.LONGITUDE_STR, OGS_C.DASH_STR * 7),
      (OGS_C.LATITUDE_STR, OGS_C.DASH_STR * 7),
      (OGS_C.ERH_STR, OGS_C.DASH_STR * 5),
      (OGS_C.DEPTH_STR, OGS_C.DASH_STR * 5),
      (OGS_C.ERZ_STR, OGS_C.DASH_STR * 5),
      (OGS_C.GAP_STR, OGS_C.DASH_STR * 3),
      (OGS_C.MAGNITUDE_L_STR, OGS_C.DASH_STR * 4),
      (OGS_C.MAGNITUDE_D_STR, OGS_C.DASH_STR * 4),
    ]
    for column, missing_marker in numeric_columns:
      self.EVENTS[column] = self._parse_numeric_series(
        self.EVENTS[column], missing_marker)

    self.EVENTS[OGS_C.NOTES_STR] = None
    self.EVENTS = self.EVENTS.astype({OGS_C.INDEX_STR: int})

    # Final filtering is retained after normalization so callers see the same
    # post-processed event table as the original implementation.
    event_mask = self.EVENTS[OGS_C.EVENT_TYPE_STR] != "[suspected explosion]"
    if self.start is not None:
      event_mask &= self.EVENTS[OGS_C.TIME_STR] >= self.start
    if self.end is not None:
      event_mask &= self.EVENTS[OGS_C.TIME_STR] <= self.end + OGS_C.ONE_DAY
    self.EVENTS = self.EVENTS[event_mask]

    self.logger.info(f"Total events read: {len(self.EVENTS)}")

    for date, dataframe in self.EVENTS.groupby(OGS_C.GROUPS_STR):
      self.events[UTCDateTime(date).date] = dataframe


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args):
  """
  Main entry point for command-line execution.

  Processes each input file specified on the command line:
    1. Creates a DataFileTXT parser instance
    2. Reads and parses the file
    3. Logs output through the shared OGS pipeline

  Args:
    args: Parsed command-line arguments from parse_arguments()
  """
  for file in args.file:
    datafile = DataFileTXT(file, args.dates[0], args.dates[1],
                           verbose=args.verbose)
    datafile.read()
    datafile.log()


if __name__ == "__main__":
  main(parse_arguments())
"""
=============================================================================
OGS DAT File Parser - Seismic Phase Pick Extractor
=============================================================================

OVERVIEW:
This module parses OGS .dat format files containing seismic phase picks
(P and S wave arrival times) recorded by the OGS seismic network. The DAT
format is a legacy fixed-width text format used for manual analyst picks.

FILE FORMAT DESCRIPTION:
  The .dat format uses fixed-width columns with the following structure:
  - Columns 1-4:   Station code (4 chars, right-padded)
  - Column 5:      P-wave onset quality (e/i/?)
  - Column 6:      P-wave polarity (+/-/c/d)
  - Column 7:      P-wave weight (0-4, quality indicator)
  - Column 8:      Fixed "1" marker
  - Columns 9-18:  Date-time (YYMMDDHHMM format)
  - Columns 19-22: P-wave arrival time (SSCC, seconds.centiseconds)
  - Columns 23-30: Reserved/unknown
  - Columns 31-38: Optional S-wave data (time, onset, polarity, weight)
  - Columns 39-60: Padding
  - Column 61:     Geographic zone code
  - Column 62:     Event type code
  - Column 63:     Event localization flag (D = distant)
  - Columns 64-68: Padding
  - Columns 69-73: Signal duration (samples or seconds)
  - Columns 74-77: Event index number

KEY FEATURES:
  - Regex-based parsing with named capture groups
  - P and S wave pick extraction from the same record
  - Date range filtering for temporal subsetting
  - Weight quality indicator preservation
  - Event type filtering (local earthquakes only by default)
  - Parquet output for efficient storage

USAGE:
  Command line:
    python ogsdat.py -f input.dat -D 20220101 20221231 -v

  Programmatic:
    from ogsdat import DataFileDAT
    parser = DataFileDAT(Path("input.dat"), start_date, end_date)
    parser.read()
    parser.log()

OUTPUT:
  - self.PICKS: DataFrame with station-level P and S picks
  - self.picks: Dict mapping dates to grouped pick DataFrames

DEPENDENCIES:
  - pandas: DataFrame operations and Parquet I/O
  - obspy: UTCDateTime for seismological time handling
  - ogsconstants: OGS-specific constants and patterns
  - ogsdatafile: Base class for file parsing

=============================================================================
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

# Standard library: Regular expressions for pattern matching
import re

# Standard library: Command-line argument parsing
import argparse

# Pandas: DataFrame operations and data manipulation
import pandas as pd

# Standard library: Filesystem path handling
from pathlib import Path

# ObsPy: Seismological library - precise time handling
from obspy import UTCDateTime

# Standard library: Date/time objects and time deltas
from datetime import datetime, timedelta as td

# Local module: OGS-specific constants (column names, patterns, formats)
import ogsconstants as OGS_C

# Local module: OGS-specific utility functions (date parsing, path validation)
import ogsutils as OGS_U

# Local module: Base class providing regex extraction and logging
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
  Parse command-line arguments for the DAT file processor.

  Returns:
    argparse.Namespace with:
      - file: List of Path objects to input .dat files
      - dates: Tuple of (start_date, end_date) for filtering
      - verbose: Boolean flag for debug output
  """
  parser = argparse.ArgumentParser(description="Run OGS DAT quality checks")

  # -f/--file: Input file path(s), required, accepts multiple files
  parser.add_argument(
      "-f", "--file", type=Path, required=True, nargs=OGS_C.ONE_MORECHAR_STR,
      help="Path to the input file"
  )

  # -D/--dates: Date range filter, optional, format YYMMDD
  # Uses custom SortDatesAction to ensure start <= end.
  parser.add_argument(
      '-D', "--dates", required=False, metavar=OGS_C.DATE_STD,
      type=OGS_U.is_date, nargs=2, action=OGS_U.SortDatesAction,
      default=[datetime.min, datetime.max - OGS_C.ONE_DAY],
      help=(
          "Specify the beginning and ending (inclusive) Gregorian date "
          "(YYMMDD) range to work with."
      )
  )

  # -v/--verbose: Enable detailed logging output
  parser.add_argument(
      '-v', "--verbose", default=False, action='store_true', required=False,
      help="Enable verbose output"
  )

  return parser.parse_args()


# =============================================================================
# DataFileDAT Class - DAT Format Parser
# =============================================================================

class DataFileDAT(OGSDataFile):
  """
  Parser for OGS .dat format seismic phase pick files.

  Extends OGSDataFile to provide format-specific regex patterns and parsing
  logic for the legacy DAT fixed-width text format. Each record contains a
  P-wave pick and may optionally append a paired S-wave pick for the same
  station.

  Attributes:
    RECORD_EXTRACTOR_LIST: Regex patterns for individual pick records
    EVENT_EXTRACTOR_LIST: Regex patterns for event summary lines
  """

  # -------------------------------------------------------------------------
  # RECORD EXTRACTOR: Regex fragments for station pick records
  # -------------------------------------------------------------------------
  # Each fragment matches one field in the fixed-width DAT layout. Named
  # capture groups allow the parser to build a dictionary directly from the
  # regex match.
  RECORD_EXTRACTOR_LIST = [
      # Station
      fr"^(?P<{OGS_C.STATION_STR}>[A-Z0-9\s]{{4}})",
      # P-wave onset quality: e=emergent, i=impulsive, ?=uncertain, space=unknown
      fr"(?P<{OGS_C.P_ONSET_STR}>[ei\s\?]){OGS_C.PWAVE}",
      # P-wave polarity: c/C/+=compression(up), d/D/-=dilatation(down), space=unknown
      fr"(?P<{OGS_C.P_POLARITY_STR}>[cC\+dD\-\s])",
      # P-wave weight: 0=best, 4=worst quality, space=unweighted
      fr"(?P<{OGS_C.P_WEIGHT_STR}>[0-4\s])",
      # Fixed marker "1" (format identifier)
      fr"1",
      # Date-time: YYMMDDHHMM format (10 digits) followed by space or zero
      fr"(?P<{OGS_C.DATE_STR}>\d{{10}})[\s0]",
      # P-wave arrival time: SSCC (seconds.centiseconds, 4 digits)
      fr"(?P<{OGS_C.P_TIME_STR}>[\s\d]{{4}})",
      # Reserved/unknown field: 8 characters (ignored)
      fr".{{8}}",
      # Optional S-wave data block (may be 8 spaces if no S pick)
      [
          # S-wave arrival time: SSCC (seconds.centiseconds, 4 digits)
          fr"(((?P<{OGS_C.S_TIME_STR}>[\s\d]{{4}})",
          # S-wave onset quality: e=emergent, i=impulsive, ?=uncertain, space=unknown
          fr"(?P<{OGS_C.S_ONSET_STR}>[ei\s\?]){OGS_C.SWAVE}",
          # S-wave polarity: c/C/+=compression(up), d/D/-=dilatation(down), space=unknown
          fr"(?P<{OGS_C.S_POLARITY_STR}>[cC\+dD\-\s])",
          # S-wave weight: 0=best, 5=worst quality, space=unweighted
          fr"(?P<{OGS_C.S_WEIGHT_STR}>[0-5\s]))|\s{{8}})"
      ],
      # Padding: 22 spaces to align with fixed-width format
      fr"\s{{22}}",
      # Geographic zone code
      fr"(?P<{OGS_C.GEO_ZONE_STR}>" + (
          fr"[{OGS_C.EMPTY_STR.join(OGS_C.OGS_GEO_ZONES.keys())}\s]"
      ) + fr")",
      # Event type code: Single character classifying the seismic event (L=local, R=regional, T=teleseismic, Q=quarry blast, etc.)
      fr"(?P<{OGS_C.EVENT_TYPE_STR}>" + (
          fr"[{OGS_C.EMPTY_STR.join(OGS_C.OGS_EVENT_TYPES.keys())}\s]"
      ) + fr")",
      # Event localization flag: D=distant event, space=local/regional
      fr"(?P<{OGS_C.EVENT_LOCALIZATION_STR}>[D\s])",
      # Padding: 5 spaces
      fr"\s{{5}}",
      # Signal duration: 5 digits (in samples or deciseconds)
      fr"(?P<{OGS_C.DURATION_STR}>[\s\d]{{5}})",
      # Event index: 4-digit sequential event number within the year
      fr"(?P<{OGS_C.INDEX_STR}>[\s\d]{{4}})",
      fr""
  ]

  # -------------------------------------------------------------------------
  # EVENT EXTRACTOR: Metadata-only lines without pick data
  # -------------------------------------------------------------------------
  EVENT_EXTRACTOR_LIST = [
      # Event type code: Single character classifying the seismic event (L=local, R=regional, T=teleseismic, Q=quarry blast, etc.)
      fr"(?P<{OGS_C.EVENT_TYPE_STR}>" + (
          fr"[{OGS_C.EMPTY_STR.join(OGS_C.OGS_EVENT_TYPES.keys())}\s]"
      ) + fr")",
      # Event localization flag: D=distant event, space=local/regional
      fr"(?P<{OGS_C.EVENT_LOCALIZATION_STR}>[D\s])",
      # Padding: 5 spaces
      fr"\s{{5}}",
      # Signal duration: 5 digits (in samples or deciseconds)
      fr"(?P<{OGS_C.DURATION_STR}>[\s\d]{{5}})",
      # Event index: 4-digit sequential event number within the year
      fr"(?P<{OGS_C.INDEX_STR}>[\s\d]{{4}})",
  ]

  @staticmethod
  def _parse_event_datetime(value: str) -> datetime:
    """Convert the DAT date field to a datetime, handling minute rollover."""
    if int(value[-2:]) >= 60:
      return datetime.strptime(
          value[:-2], OGS_C.DATETIME_FMT[:-4]
      ) + td(hours=1)
    return datetime.strptime(value, OGS_C.DATETIME_FMT[:-2])

  @staticmethod
  def _parse_pick_time(base_time: datetime, value: str) -> datetime:
    """Convert a SSCC field to an absolute pick time."""
    offset = float(value.replace(OGS_C.SPACE_STR, OGS_C.ZERO_STR)) / 100.
    return base_time + td(seconds=offset)

  @staticmethod
  def _build_pick_row(event_index: int, pick_time: datetime, station: str,
                      phase: str, weight: int):
    """Create a standardized pick row for the output DataFrame."""
    return [
        event_index,
        pick_time.strftime(OGS_C.DATE_FMT),
        pick_time,
        f".{station}.",
        phase,
        weight,
        None,
        None,
        None,
        None,
        1.0,
    ]

  def read(self):
    """
    Read and parse a .dat format file into P and S wave picks.

    The parser walks through the input file line by line, skips metadata-only
    lines, extracts station records with regex patterns, filters by date range
    and event type, and stores grouped picks in self.PICKS and self.picks.

    Raises:
      FileNotFoundError: If the input file does not exist.
      ValueError: If the input file does not use the .dat extension.
    """
    # -----------------------------------------------------------------------
    # INPUT VALIDATION
    # -----------------------------------------------------------------------
    if not self.input.exists():
      raise FileNotFoundError(f"File {self.input} does not exist")

    if self.input.suffix != OGS_C.DAT_EXT:
      raise ValueError(f"File extension must be {OGS_C.DAT_EXT}")

    # -----------------------------------------------------------------------
    # FILE READING
    # -----------------------------------------------------------------------
    pick_records = list()
    default_weight = 0

    self.logger.info(f"Reading DAT file: {self.input}")
    try:
      with open(self.input, 'r') as fr:
        lines = fr.readlines()
    except Exception as exc:
      corrupt_path = self.input.with_suffix('.dat.corrupt')
      self.logger.error(f"Failed to read file. Backing up to {corrupt_path}")
      try:
        import shutil
        shutil.copy2(self.input, corrupt_path)
      except Exception:
        pass
      raise RuntimeError(f"Parsing failed for {self.input}: {exc}") from exc

    # -----------------------------------------------------------------------
    # LINE-BY-LINE PARSING
    # -----------------------------------------------------------------------
    for raw_line in lines:
      line = raw_line.strip()

      if line == OGS_C.EMPTY_STR:
        continue

      # Event summary lines carry only metadata already repeated in pick rows.
      if self.EVENT_EXTRACTOR.match(line):
        continue

      match = self.RECORD_EXTRACTOR.match(line)
      if not match:
        if re.match(r"1\s*D?\s*.?$", line):
          continue
        self.logger.error(f"ERROR: (DAT) Could not parse line: {line}")
        self.debug(line, self.RECORD_EXTRACTOR_LIST)
        continue

      result: dict = match.groupdict()

      # -----------------------------------------------------------------------
      # EVENT TYPE FILTERING
      # -----------------------------------------------------------------------
      # Keep local earthquakes and distant events, mirroring the legacy
      # filtering behavior in the original parser.
      if (
          result[OGS_C.EVENT_LOCALIZATION_STR] != "D"
          and result[OGS_C.EVENT_TYPE_STR] != OGS_C.SPACE_STR
          and OGS_C.OGS_EVENT_TYPES[result[OGS_C.EVENT_TYPE_STR]] !=
              OGS_C.EVENT_LOCAL_EQ_STR
      ):
        continue

      try:
        event_time = self._parse_event_datetime(result[OGS_C.DATE_STR])
      except ValueError as exc:
        self.logger.error(exc)
        continue

      # ---------------------------------------------------------------------
      # DATE RANGE FILTERING
      # ---------------------------------------------------------------------
      if self.start is not None and event_time < self.start:
        self.logger.debug(f"Skipping pick before start date: {self.start}")
        self.logger.debug(line)
        continue

      if self.end is not None and event_time >= self.end + OGS_C.ONE_DAY:
        self.logger.debug(f"Stopping read at pick after end date: {self.end}")
        self.logger.debug(line)
        break

      # ---------------------------------------------------------------------
      # FIELD PROCESSING
      # ---------------------------------------------------------------------
      station = result[OGS_C.STATION_STR].strip(OGS_C.SPACE_STR)

      try:
        event_index = self._parse_index(
            result[OGS_C.INDEX_STR], event_time.year
        )
      except ValueError as exc:
        event_index = None
        self.logger.error(exc)

      try:
        p_pick_time = self._parse_pick_time(
            event_time, result[OGS_C.P_TIME_STR]
        )
      except ValueError as exc:
        self.logger.error(exc)
        continue

      try:
        p_weight = self._parse_weight(
            result[OGS_C.P_WEIGHT_STR], default_weight
        )
      except ValueError as exc:
        self.logger.error(exc)
        continue

      # ---------------------------------------------------------------------
      # APPEND P-WAVE PICK TO RESULTS
      # ---------------------------------------------------------------------
      pick_records.append(self._build_pick_row(
          event_index,
          p_pick_time,
          station,
          OGS_C.PWAVE,
          p_weight
      ))

      # ---------------------------------------------------------------------
      # S-WAVE PROCESSING (if present)
      # ---------------------------------------------------------------------
      if result[OGS_C.S_TIME_STR]:
        try:
          s_weight = self._parse_weight(
              result[OGS_C.S_WEIGHT_STR], default_weight
          )
        except ValueError as exc:
          self.logger.error(exc)
          continue

        try:
          s_pick_time = self._parse_pick_time(
              event_time, result[OGS_C.S_TIME_STR]
          )
        except ValueError as exc:
          self.logger.error(exc)
          continue

        pick_records.append(self._build_pick_row(
            event_index,
            s_pick_time,
            station,
            OGS_C.SWAVE,
            s_weight
        ))

    # -----------------------------------------------------------------------
    # BUILD OUTPUT DATAFRAME
    # -----------------------------------------------------------------------
    self.PICKS = pd.DataFrame(pick_records, columns=[
        OGS_C.IDX_PICKS_STR, OGS_C.GROUPS_STR, OGS_C.TIME_STR,
        OGS_C.STATION_STR, OGS_C.PHASE_STR, OGS_C.WEIGHT_STR,
        OGS_C.EPICENTRAL_DISTANCE_STR, OGS_C.DEPTH_STR, OGS_C.AMPLITUDE_STR,
        OGS_C.STATION_ML_STR, OGS_C.PROBABILITY_STR
    ]).astype({OGS_C.IDX_PICKS_STR: int})

    # Extract the Gregorian date from the timestamp for downstream grouping.
    self.PICKS[OGS_C.GROUPS_STR] = self.PICKS[OGS_C.TIME_STR].apply(
        lambda value: value.date()
    )

    for date, dataframe in self.PICKS.groupby(OGS_C.GROUPS_STR):
      self.picks[UTCDateTime(date).date] = dataframe


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args):
  """
  Main entry point for command-line execution.

  Processes each input file specified on the command line:
    1. Creates a DataFileDAT parser instance
    2. Reads and parses the file
    3. Logs output through the shared OGS pipeline

  Args:
    args: Parsed command-line arguments from parse_arguments()
  """
  for file in args.file:
    datafile = DataFileDAT(file, args.dates[0], args.dates[1],
                           verbose=args.verbose)
    datafile.read()
    datafile.log()


if __name__ == "__main__":
  main(parse_arguments())

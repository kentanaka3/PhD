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
  Parquet files organized by date containing:
    - Event index, date, timestamp, station, phase type (P/S)
    - Weight, distances, amplitudes, ML estimates, probability

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
  parser = argparse.ArgumentParser(description="Run OGS HPL quality checks")

  # -f/--file: Input file path(s), required, accepts multiple files
  parser.add_argument(
    "-f", "--file", type=Path, required=True, nargs=OGS_C.ONE_MORECHAR_STR,
    help="Path to the input file")

  # -D/--dates: Date range filter, optional, format YYMMDD
  # Uses custom SortDatesAction to ensure start <= end
  parser.add_argument(
    '-D', "--dates", required=False, metavar=OGS_C.DATE_STD,
    type=OGS_C.is_date, nargs=2, action=OGS_C.SortDatesAction,
    default=[datetime.min, datetime.max - OGS_C.ONE_DAY],
    help="Specify the beginning and ending (inclusive) Gregorian date " \
         "(YYMMDD) range to work with.")

  # -v/--verbose: Enable detailed logging output
  parser.add_argument(
    '-v', "--verbose", default=False, action='store_true', required=False,
    help="Enable verbose output")

  return parser.parse_args()


# =============================================================================
# DataFileDAT Class - DAT Format Parser
# =============================================================================

class DataFileDAT(OGSDataFile):
  """
  Parser for OGS .dat format seismic phase pick files.

  Extends OGSDataFile to provide format-specific regex patterns and
  parsing logic for the legacy DAT fixed-width text format.

  The DAT format contains one line per phase pick, with optional S-wave
  data appended to P-wave records. Each record includes timing, quality
  weights, and event classification metadata.

  Attributes:
    RECORD_EXTRACTOR_LIST: Regex patterns for individual pick records
    EVENT_EXTRACTOR_LIST: Regex patterns for event summary lines
  """

  # -------------------------------------------------------------------------
  # RECORD EXTRACTOR: Regex patterns for parsing individual pick lines
  # -------------------------------------------------------------------------
  # Each pattern fragment matches a specific column or field in the DAT format
  # Named capture groups (?P<name>...) allow direct extraction to dict keys
  RECORD_EXTRACTOR_LIST = [
    # Station code: 4 alphanumeric characters (right-padded with spaces)
    fr"^(?P<{OGS_C.STATION_STR}>[A-Z0-9\s]{{4}})",                # Station

    # P-wave onset quality: e=emergent, i=impulsive, ?=uncertain, space=unknown
    fr"(?P<{OGS_C.P_ONSET_STR}>[ei\s\?]){OGS_C.PWAVE}",           # P Onset

    # P-wave polarity: c/C/+=compression(up), d/D/-=dilatation(down), space=unknown
    fr"(?P<{OGS_C.P_POLARITY_STR}>[cC\+dD\-\s])",                 # P Polarity

    # P-wave weight: 0=best, 4=worst quality, space=unweighted
    fr"(?P<{OGS_C.P_WEIGHT_STR}>[0-4\s])",                        # P Weight

    # Fixed marker "1" (format identifier)
    fr"1",                                                        # 1

    # Date-time: YYMMDDHHMM format (10 digits) followed by space or zero
    fr"(?P<{OGS_C.DATE_STR}>\d{{10}})[\s0]",                         # Date

    # P-wave arrival time: SSCC (seconds.centiseconds, 4 digits)
    fr"(?P<{OGS_C.P_TIME_STR}>[\s\d]{{4}})",                      # P Time

    # Reserved/unknown field: 8 characters (ignored)
    fr".{{8}}",                                                   # Unknown

    # Optional S-wave data block (may be 8 spaces if no S pick)
    [
      # Either S-wave data OR 8 spaces (no S pick)
      fr"(((?P<{OGS_C.S_TIME_STR}>[\s\d]{{4}})",                  # S Time
      fr"(?P<{OGS_C.S_ONSET_STR}>[ei\s\?]){OGS_C.SWAVE}",         # S Onset
      fr"(?P<{OGS_C.S_POLARITY_STR}>[cC\+dD\-\s])",               # S Polarity
      fr"(?P<{OGS_C.S_WEIGHT_STR}>[0-5\s]))|\s{{8}})"             # S Weight
    ],

    # Padding: 22 spaces before metadata
    fr"\s{{22}}",                                                 # SPACE

    # Geographic zone code: Single character from predefined zone list
    # Zones define regional seismotectonic areas (e.g., Alps, Friuli, etc.)
    fr"(?P<{OGS_C.GEO_ZONE_STR}>[{OGS_C.EMPTY_STR.join(
      OGS_C.OGS_GEO_ZONES.keys())}\s])",

    # Event type code: Single character classifying the seismic event
    # Types include: L=local, R=regional, T=teleseismic, Q=quarry blast, etc.
    fr"(?P<{OGS_C.EVENT_TYPE_STR}>[{OGS_C.EMPTY_STR.join(
      OGS_C.OGS_EVENT_TYPES.keys())}\s])",

    # Event localization flag: D=distant event, space=local/regional
    fr"(?P<{OGS_C.EVENT_LOCALIZATION_STR}>[D\s])",                # Event Loc

    # Padding: 5 spaces
    fr"\s{{5}}",                                                  # SPACE

    # Signal duration: 5 digits (in samples or deciseconds)
    fr"(?P<{OGS_C.DURATION_STR}>[\s\d]{{5}})",                    # Duration

    # Event index: 4-digit sequential event number within the year
    fr"(?P<{OGS_C.INDEX_STR}>[\s\d]{{4}})",                       # Event

    # End of line (empty pattern to terminate regex)
    fr""
  ]

  # -------------------------------------------------------------------------
  # EVENT EXTRACTOR: Regex patterns for event summary lines
  # -------------------------------------------------------------------------
  # Event lines contain only the metadata portion (no station/phase data)
  EVENT_EXTRACTOR_LIST = [
    # Event type code
    fr"(?P<{OGS_C.EVENT_TYPE_STR}>[{OGS_C.EMPTY_STR.join(
      OGS_C.OGS_EVENT_TYPES.keys())}\s])",

    # Event localization flag
    fr"(?P<{OGS_C.EVENT_LOCALIZATION_STR}>[D\s])",                # Event Loc

    # Padding
    fr"\s{{5}}",                                                  # SPACE

    # Signal duration
    fr"(?P<{OGS_C.DURATION_STR}>[\s\d]{{5}})",                    # Duration

    # Event index
    fr"(?P<{OGS_C.INDEX_STR}>[\s\d]{{4}})",                       # Event

    # End of line
    fr""
  ]

  # -------------------------------------------------------------------------
  # METHOD: read() - Parse DAT file into picks DataFrame
  # -------------------------------------------------------------------------

  def read(self):
    """
    Read and parse a .dat format file into P and S wave picks.

    Processes each line of the input file, extracting phase arrival data
    using regex patterns. Filters by date range and event type (local
    earthquakes only). Handles edge cases like minute=60 (rollover).

    The method populates:
      - self.PICKS: DataFrame with all extracted picks
      - self.picks: Dict mapping dates to pick DataFrames

    Raises:
      AssertionError: If file doesn't exist or has wrong extension
    """
    # -------------------------------------------------------------------------
    # INPUT VALIDATION
    # -------------------------------------------------------------------------

    # Verify input file exists on filesystem
    if not self.input.exists():
      raise FileNotFoundError(f"File {self.input} does not exist")

    # Verify correct file extension (.dat)
    if self.input.suffix != OGS_C.DAT_EXT:
      raise ValueError(f"File extension must be {OGS_C.DAT_EXT}")

    # TODO: Attemp restoration before SHUTDOWN

    # -------------------------------------------------------------------------
    # FILE READING
    # -------------------------------------------------------------------------

    # Initialize list to collect parsed pick records
    DETECT = list()

    # Read all lines from input file
    with open(self.input, 'r') as fr: lines = fr.readlines()
    self.logger.info(f"Reading DAT file: {self.input}")

    # -------------------------------------------------------------------------
    # LINE-BY-LINE PARSING
    # -------------------------------------------------------------------------

    for line in [l.strip() for l in lines]:

      # Skip event summary lines (matched by EVENT_EXTRACTOR)
      if self.EVENT_EXTRACTOR.match(line): continue

      # Attempt to match line against RECORD_EXTRACTOR pattern
      match = self.RECORD_EXTRACTOR.match(line)

      if match:
        # Extract all named capture groups into dictionary
        result: dict = match.groupdict()

        # -----------------------------------------------------------------------
        # EVENT TYPE FILTERING
        # -----------------------------------------------------------------------
        # Only process local earthquakes (not distant events, not non-seismic)
        # Skip if: localization is not "D" AND event type is defined AND
        #          event type is not local earthquake
        if (result[OGS_C.EVENT_LOCALIZATION_STR] != "D" and
            result[OGS_C.EVENT_TYPE_STR] != OGS_C.SPACE_STR and
            OGS_C.OGS_EVENT_TYPES[result[OGS_C.EVENT_TYPE_STR]] != \
              OGS_C.EVENT_LOCAL_EQ_STR):
          # print("WARNING: (DAT) Ignoring line:", line)
          continue

        # -----------------------------------------------------------------------
        # DATE PARSING WITH EDGE CASE HANDLING
        # -----------------------------------------------------------------------
        try:
          # Handle minute=60 edge case (some systems record 60 instead of 00+1hr)
          if int(result[OGS_C.DATE_STR][-2:]) >= 60:
            # Parse without minutes, then add 1 hour
            result[OGS_C.DATE_STR] = \
                datetime.strptime(result[OGS_C.DATE_STR][:-2],
                                  OGS_C.DATETIME_FMT[:-4]) + td(hours=1)
          else:
            # Standard parsing: YYMMDDHHMM format
            result[OGS_C.DATE_STR] = datetime.strptime(
              result[OGS_C.DATE_STR], OGS_C.DATETIME_FMT[:-2])
        except ValueError as e:
          # Skip records with unparseable dates
          print(e)
          continue

        # -----------------------------------------------------------------------
        # DATE RANGE FILTERING
        # -----------------------------------------------------------------------

        # Skip picks before the specified start date
        if self.start is not None and result[OGS_C.DATE_STR] < self.start:
          self.logger.debug(f"Skipping pick before start date: {self.start}")
          self.logger.debug(line)
          continue

        # Stop processing if we've passed the end date (assumes sorted input)
        if (self.end is not None and
            result[OGS_C.DATE_STR] >= self.end + OGS_C.ONE_DAY):
          self.logger.debug(f"Stopping read at pick after end date: {self.end}")
          self.logger.debug(line)
          break

        # -----------------------------------------------------------------------
        # FIELD PROCESSING
        # -----------------------------------------------------------------------

        # Clean station name: remove padding spaces
        result[OGS_C.STATION_STR] = \
          result[OGS_C.STATION_STR].strip(OGS_C.SPACE_STR)

        # Format date string for grouping (YYMMDD format)
        date = result[OGS_C.DATE_STR].strftime(OGS_C.YYMMDD_FMT)

        # -----------------------------------------------------------------------
        # P-WAVE TIME CALCULATION
        # -----------------------------------------------------------------------
        try:
          # Convert SSCC (seconds.centiseconds) to timedelta and add to base time
          # Replace spaces with zeros for numeric conversion
          result[OGS_C.P_TIME_STR] = result[OGS_C.DATE_STR] + \
            td(seconds=float(result[OGS_C.P_TIME_STR].replace(
              OGS_C.SPACE_STR, OGS_C.ZERO_STR)) / 100.)
        except ValueError as e:
          self.logger.error(e)
          continue

        # -----------------------------------------------------------------------
        # EVENT INDEX PROCESSING
        # -----------------------------------------------------------------------
        if result[OGS_C.INDEX_STR]:
          try:
            # Convert to integer, add year offset for global uniqueness
            # MAX_PICKS_YEAR ensures non-overlapping indices across years
            result[OGS_C.INDEX_STR] = int(result[OGS_C.INDEX_STR].replace(
              OGS_C.SPACE_STR, OGS_C.ZERO_STR)) + \
                result[OGS_C.DATE_STR].year * OGS_C.MAX_PICKS_YEAR
          except ValueError as e:
            result[OGS_C.INDEX_STR] = None
            self.logger.error(e)

        # Default weight value for missing/blank weights
        DEFAULT_VALUE = 0

        # -----------------------------------------------------------------------
        # P-WAVE WEIGHT PROCESSING
        # -----------------------------------------------------------------------
        try:
          # Convert weight to integer, use default if blank
          if result[OGS_C.P_WEIGHT_STR] == OGS_C.SPACE_STR:
            result[OGS_C.P_WEIGHT_STR] = DEFAULT_VALUE
          else:
            result[OGS_C.P_WEIGHT_STR] = int(result[OGS_C.P_WEIGHT_STR])
        except ValueError as e:
          self.logger.error(e)
          continue

        # -----------------------------------------------------------------------
        # APPEND P-WAVE PICK TO RESULTS
        # -----------------------------------------------------------------------
        # Record format: [event_id, date, time, station, phase, weight,
        #                 distance, depth, amplitude, ML, probability]
        DETECT.append([
          result[OGS_C.INDEX_STR],
          result[OGS_C.P_TIME_STR].strftime(OGS_C.DATE_FMT),
          result[OGS_C.P_TIME_STR],
          f".{result[OGS_C.STATION_STR]}.",  # Station with delimiters
          OGS_C.PWAVE, int(result[OGS_C.P_WEIGHT_STR]),
          None, None, None, None, 1.0  # Placeholders for computed fields
        ])

        # -----------------------------------------------------------------------
        # S-WAVE PROCESSING (if present)
        # -----------------------------------------------------------------------
        if result[OGS_C.S_TIME_STR]:

          # S-wave weight processing
          try:
            if result[OGS_C.S_WEIGHT_STR] == OGS_C.SPACE_STR:
              result[OGS_C.S_WEIGHT_STR] = DEFAULT_VALUE
            else:
              result[OGS_C.S_WEIGHT_STR] = int(result[OGS_C.S_WEIGHT_STR])
          except ValueError as e:
            self.logger.error(e)
            continue

          # S-wave time calculation (same method as P-wave)
          try:
            result[OGS_C.S_TIME_STR] = result[OGS_C.DATE_STR] + \
              td(seconds=float(result[OGS_C.S_TIME_STR].replace(
                OGS_C.SPACE_STR, OGS_C.ZERO_STR)) / 100.)
          except ValueError as e:
            self.logger.error(e)
            continue

          # Append S-wave pick to results
          DETECT.append([
            result[OGS_C.INDEX_STR],
            result[OGS_C.S_TIME_STR].strftime(OGS_C.DATE_FMT),
            result[OGS_C.S_TIME_STR],
            f".{result[OGS_C.STATION_STR]}.",
            OGS_C.SWAVE, int(result[OGS_C.S_WEIGHT_STR]),
            None, None, None, None, 1.0
          ])
        continue

      # -------------------------------------------------------------------------
      # UNMATCHED LINE HANDLING
      # -------------------------------------------------------------------------

      # Skip known non-data lines (format markers, blank lines)
      if re.match(r"1\s*D?\s*.?$", line): continue
      if line == OGS_C.EMPTY_STR: continue

      # Log and debug unrecognized lines
      self.logger.error(f"ERROR: (DAT) Could not parse line: {line}")
      self.debug(line, self.RECORD_EXTRACTOR_LIST)

    # -------------------------------------------------------------------------
    # BUILD OUTPUT DATAFRAME
    # -------------------------------------------------------------------------

    # Create DataFrame from collected picks with proper column names
    self.PICKS = pd.DataFrame(DETECT, columns=[
      OGS_C.IDX_PICKS_STR, OGS_C.GROUPS_STR, OGS_C.TIME_STR, OGS_C.STATION_STR,
      OGS_C.PHASE_STR, OGS_C.WEIGHT_STR, OGS_C.EPICENTRAL_DISTANCE_STR,
      OGS_C.DEPTH_STR, OGS_C.AMPLITUDE_STR, OGS_C.STATION_ML_STR,
      OGS_C.PROBABILITY_STR
    ]).astype({ OGS_C.IDX_PICKS_STR: int})

    # Extract date from timestamp for grouping
    self.PICKS[OGS_C.GROUPS_STR] = self.PICKS[OGS_C.TIME_STR].apply(
      lambda x: x.date())

    # Populate picks dictionary: date -> DataFrame
    for date, df in self.PICKS.groupby(OGS_C.GROUPS_STR):
      self.picks[UTCDateTime(date).date] = df


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args):
  """
  Main entry point for command-line execution.

  Processes each input file specified on the command line:
    1. Creates DataFileDAT parser instance
    2. Reads and parses the file
    3. Logs output to Parquet format

  Args:
    args: Parsed command-line arguments from parse_arguments()
  """
  for file in args.file:
    # Create parser with date range and verbosity settings
    datafile = DataFileDAT(file, args.dates[0], args.dates[1],
                           verbose=args.verbose)
    # Parse the input file
    datafile.read()
    # Write output to Parquet files
    datafile.log()


# Script entry point: parse arguments and run main
if __name__ == "__main__": main(parse_arguments())
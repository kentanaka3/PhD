"""
===============================================================================
OGS TXT File Parser - Catalog Event Summary Extractor
===============================================================================

OVERVIEW:
This module parses OGS .txt catalog exports containing event-level summaries.
Each data row represents one located event with origin time, hypocenter,
uncertainty estimates, magnitudes, locality name, and an event-type label.

FILE FORMAT DESCRIPTION:
  The .txt format stores one fixed-width summary record per event with:
  - Event index and legacy catalog identifier
  - ISO origin time (YYYY-MM-DDTHH:MM:SS.mmm)
  - RMS, latitude, longitude, ERH, depth, ERZ, and GAP quality fields
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


DEFAULT_FILTERED_EVENT_TYPES = [
    "[suspected explosion]",
    "[chemical explosion]",
    "[suspected slide]",
]


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
    EVENT_EXTRACTOR_LIST: Regex patterns for individual event summary lines
  """

  # Expected file extension for format validation
  EXTENSION: str = OGS_C.TXT_EXT

  # -------------------------------------------------------------------------
  # EVENT EXTRACTOR: fixed-width event summary line
  # -------------------------------------------------------------------------
  EVENT_EXTRACTOR_LIST = [
      fr"^(?P<{OGS_C.IDX_EVENTS_STR}>\d{{5}})\s",                 # Index
      fr"(?P<{OGS_C.LEGACY_ID_STR}>\d{{4}}_\d{{5}})\s",           # Legacy ID
      fr"(?P<{OGS_C.TIME_STR}>\d{{4}}-\d{{2}}-\d{{2}}T",          # Date
      fr"\d{{2}}:\d{{2}}:\d{{2}}\.\d{{3}})\s",                    # Time
      fr"(?P<{OGS_C.RMS_STR}>[\s\d\.\-]{{5}})\s",                 # RMS
      fr"(?P<{OGS_C.LATITUDE_STR}>[\s\d\-\.]{{7}})\s",            # Latitude
      fr"(?P<{OGS_C.LONGITUDE_STR}>[\s\d\-\.]{{7}})\s",           # Longitude
      fr"(?P<{OGS_C.ERH_STR}>[\s\d\.\-]{{5}})\s",                 # ERH
      fr"(?P<{OGS_C.DEPTH_STR}>[\s\d\.\-]{{5}})\s",               # Depth
      fr"(?P<{OGS_C.ERZ_STR}>[\s\d\.\-]{{5}})\s",                 # ERZ
      fr"(?P<{OGS_C.GAP_STR}>([\s\d\-]{{3}}))\s",                 # GAP
      fr"(?P<{OGS_C.MAGNITUDE_L_STR}>([\-\s\d\.]{{4}}))\s",       # ML
      fr"(?P<{OGS_C.MAGNITUDE_D_STR}>([\-\s\d\.]{{4}}))\s",       # MD
      fr"(?P<{OGS_C.LOC_NAME_STR}>['\.\-\w\s\(\)]+)\s",           # Place
      fr"(?P<{OGS_C.EVENT_TYPE_STR}>\[.*\])$",                    # Event Type
  ]

  def _build_event_row(
      self,
      result: dict,
      event_index: int | None = None,
  ) -> list:
    """Create a standardized event row for the output DataFrame."""
    event_time = result[OGS_C.TIME_STR]
    if event_index is None:
      event_index = self.normalize_index(
          result[OGS_C.IDX_EVENTS_STR], event_time.year
      )
    return [
        event_index,                            # 0: idx
        event_time,                             # 1: time
        result[OGS_C.LATITUDE_STR],             # 2: latitude
        result[OGS_C.LONGITUDE_STR],            # 3: longitude
        result[OGS_C.DEPTH_STR],                # 4: depth
        result[OGS_C.GAP_STR],                  # 5: azimuthal_gap
        result[OGS_C.ERZ_STR],                  # 6: max_vertical_uncertainty
        result[OGS_C.ERH_STR],                  # 7: max_horizontal_uncertainty
        None,                                   # 8: max_time_uncertainty
        event_time.strftime(OGS_C.DATE_FMT),    # 9: group
        None,                                   # 10: number_picks
        0,                                      # 11: number_p_picks
        0,                                      # 12: number_s_picks
        0,                                      # 13: number_p_and_s_picks
        result[OGS_C.MAGNITUDE_D_STR],          # 14: MD
        result[OGS_C.MAGNITUDE_L_STR],          # 15: ML
        None,                                   # 16: ML_median
        None,                                   # 17: ML_unc
        None,                                   # 18: ML_stations
        None,                                   # 19: DMIN
        result[OGS_C.RMS_STR],                  # 20: RMS
        None,                                   # 21: QM
        result[OGS_C.LOC_NAME_STR],             # 22: LOC_NAME
        result[OGS_C.EVENT_TYPE_STR],           # 23: E_TYPE
        None,                                   # 24: NOTES
        None,                                   # 25: MD_unc
        None,                                   # 26: MD_stations
        None,                                   # 27: MD_median
    ]

  def _parse_event_datetime(self, value: str) -> datetime:
    """Convert the TXT date field to a datetime."""
    return datetime.fromisoformat(value)

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
    self.validate_input()

    # -----------------------------------------------------------------------
    # FILE READING & LINE-BY-LINE PARSING
    # -----------------------------------------------------------------------
    records = list()

    self.logger.info(f"Reading TXT file: {self.input}")
    # The first row is a header line, so parsing starts from the second line.
    with open(self.input, 'r', encoding='utf-8') as fr:
      next(fr, None)  # Skip header line lazily
      for raw_line in fr:
        line = raw_line.strip()
        if not line:
          continue

        match = self.EVENT_EXTRACTOR.match(line)
        if not match:
          self.logger.error(f"ERROR: (TXT) Could not parse line: {line}")
          self.debug(line, self.EVENT_EXTRACTOR_LIST)
          continue

        result: dict = match.groupdict()
        result[OGS_C.TIME_STR] = self._parse_event_datetime(
            result[OGS_C.TIME_STR]
        )

        # ---------------------------------------------------------------------
        # DATE RANGE FILTERING
        # ---------------------------------------------------------------------
        if self._is_before_start(result[OGS_C.TIME_STR]):
          self.logger.debug(f"Skipping event before start date: {self.start}")
          self.logger.debug(line)
          continue

        if self._is_after_end(result[OGS_C.TIME_STR]):
          self.logger.debug(
              f"Skipping event after end date: {self.end}"
          )
          self.logger.debug(line)
          continue

        if result.get(OGS_C.EVENT_TYPE_STR) in DEFAULT_FILTERED_EVENT_TYPES:
          self.logger.debug(f"Skipping filtered event type: {line}")
          continue

        # ---------------------------------------------------------------------
        # APPEND RAW EVENT SUMMARY TO RESULTS
        # ---------------------------------------------------------------------
        records.append(self._build_event_row(result))

    # -----------------------------------------------------------------------
    # BUILD OUTPUT DATAFRAME
    # -----------------------------------------------------------------------
    self.EVENTS = self._build_events_dataframe(records)

    if self.EVENTS.empty:
      self.logger.warning(f"No valid TXT records found in {self.input}")
      return

    self.EVENTS = self.normalize_time(self.EVENTS)
    self.EVENTS[OGS_C.INDEX_STR] = [
        self.normalize_index(index, year)
        for index, year in zip(
            self.EVENTS[OGS_C.INDEX_STR], self.EVENTS[OGS_C.TIME_STR].dt.year
        )
    ]
    self.EVENTS = self.normalize_groups(self.EVENTS)

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
          self.EVENTS[column], missing_marker
      )

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
    self.postload("events", update=True)


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
    datafile = DataFileTXT(
        file, args.dates[0], args.dates[1], verbose=args.verbose
    )
    datafile.read()
    datafile.log()


if __name__ == "__main__":
  main(OGS_U.parse_txt_args())

"""
=============================================================================
OGS HPL File Parser - Hypo71 Event and Pick Extractor
=============================================================================

OVERVIEW:
This module parses OGS .hpl format files produced by legacy Hypo71 workflows.
An HPL file stores event summary rows together with station-level P and
optional S picks, plus optional analyst notes and locality descriptions.

FILE FORMAT DESCRIPTION:
  The .hpl format is organized as fixed-width text blocks:
  - One event summary line with origin time, hypocenter, and quality metrics
  - A variable number of station records containing phase picks
  - Optional location and free-text note lines after the station block

KEY FEATURES:
  - Regex-based extraction of event, station, location, and notes lines
  - Separate DataFrame construction for picks and event metadata
  - Date range filtering while streaming through the file
  - Preservation of Hypo71 quality metrics and analyst annotations
  - Parquet output via the shared OGSDataFile logging pipeline

USAGE:
  Command line:
    python ogshpl.py -f input.hpl -D 240320 240620 -v

  Programmatic:
    from ogshpl import DataFileHPL
    parser = DataFileHPL(Path("input.hpl"), start_date, end_date)
    parser.read()
    parser.log()

OUTPUT:
  - self.PICKS: station-level P and S arrival picks
  - self.EVENTS: event-level origin, location, and quality metadata
  - self.picks / self.events: date-indexed dictionaries for downstream use

DEPENDENCIES:
  - pandas: DataFrame operations and Parquet I/O
  - obspy: UTCDateTime for seismological time handling
  - ogsconstants: OGS-specific constants and regex fragments
  - ogsdatafile: Base class providing compiled extractors and logging helpers

=============================================================================
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

# Standard library: regular expressions for pattern matching
import re

# Standard library: command-line argument parsing
import argparse

# Pandas: tabular data manipulation
import pandas as pd

# Standard library: filesystem path handling
from pathlib import Path

# ObsPy: seismological time conversions
from obspy import UTCDateTime

# Standard library: date/time objects and time deltas
from datetime import datetime, timedelta as td

# Local module: OGS-specific constants and formatting strings
import ogsconstants as OGS_C

# Local module: OGS-specific argument parsing helpers
import ogsutils as OGS_U

# Local module: base parser and regex list flattener
from ogsdatafile import OGSDataFile, _flatten

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
  Parse command-line arguments for the HPL file processor.

  Returns:
    argparse.Namespace with:
      - file: List of Path objects to input .hpl files
      - dates: Tuple of (start_date, end_date) for filtering
      - verbose: Boolean flag for debug output
  """
  parser = argparse.ArgumentParser(description="Run OGS HPL quality checks")

  # -f/--file: Input file path(s), required, accepts multiple files
  parser.add_argument(
    "-f", "--file", type=Path, required=True, nargs=OGS_C.ONE_MORECHAR_STR,
    help="Path to the input file")

  # -D/--dates: Date range filter, optional, format YYMMDD
  parser.add_argument(
    '-D', "--dates", required=False, metavar=OGS_C.DATE_STD,
    type=OGS_U.is_date, nargs=2, action=OGS_U.SortDatesAction,
    default=[datetime.strptime("240320", OGS_C.YYMMDD_FMT),
             datetime.strptime("240620", OGS_C.YYMMDD_FMT)],
    help="Specify the beginning and ending (inclusive) Gregorian date " \
          "(YYMMDD) range to work with.")

  # -v/--verbose: Enable detailed logging output
  parser.add_argument(
    '-v', "--verbose", action='store_true', default=False,
    help="Enable verbose output")

  return parser.parse_args()


# =============================================================================
# DataFileHPL Class - HPL Format Parser
# =============================================================================

class DataFileHPL(OGSDataFile):
  """
  Parser for OGS .hpl format event summaries and phase picks.

  Extends OGSDataFile to parse legacy Hypo71 fixed-width output. HPL files
  contain event summary records followed by a configurable number of station
  records, with optional locality and notes lines after each event block.

  Attributes:
    RECORD_EXTRACTOR_LIST: Regex patterns for station phase-pick records
    EVENT_EXTRACTOR_LIST: Regex patterns for event summary records
    LOCATION_EXTRACTOR_LIST: Regex pattern for locality description lines
    NOTES_EXTRACTOR_LIST: Regex pattern for analyst note lines
  """

  # -------------------------------------------------------------------------
  # RECORD EXTRACTOR: station-level phase pick lines
  # -------------------------------------------------------------------------
  # Many fixed-width columns are still not mapped to domain names, so the
  # unknown fields remain intentionally positional until the format is decoded.
  RECORD_EXTRACTOR_LIST = [
    fr"^(?P<{OGS_C.INDEX_STR}>[\d\s]{{6}})\s",                    # Event
    fr"(?P<{OGS_C.STATION_STR}>[A-Z0-9\s]{{4}})\s",               # Station
    fr"([\d\s\.]{{5}})\s",                                        # Unknown
    fr"([\d\s]{{3}})\s",                                          # Unknown
    fr"([\d\s]{{3}})\s",                                          # Unknown
    fr"(?P<{OGS_C.P_ONSET_STR}>[ei?\s]){OGS_C.PWAVE}",            # P Onset
    fr"(?P<{OGS_C.P_POLARITY_STR}>[cC\+dD\-\s])",                 # P Polarity
    fr"(?P<{OGS_C.P_WEIGHT_STR}>[0-4])\s",                        # P Weight
    # P Time [hhmm]
    fr"(?P<{OGS_C.P_TIME_STR}>[\s\d]{{4}})\s",
    # Seconds [ss.ss]
    fr"(?P<{OGS_C.SECONDS_STR}>[\s\d\.]{{5}})",
    fr"(?P<A>[\s\d\-\.]{{6}})\s",                                 # Unknown
    fr"(?P<B>[\s\d\-\.]{{5}})\s",                                 # Unknown
    fr"(?P<C>[\s\d\-\.]{{5}})",                                   # Unknown
    fr"(?P<D>[\s\d\-\.]{{6}})\s",                                 # Unknown
    fr"(?P<E>[\s\d\-\.]{{5}})\s",                                 # Unknown
    fr"(?P<F>[\s\d\-\.]{{3}})\s",                                 # Unknown
    fr"(?P<G>[\s\d\-\.]{{2}})\s",                                 # Unknown
    fr"(?P<H>[\s\d\-\.]{{5}})\s",                                 # Unknown
    fr"(?P<I>[\s\d])\s{{6}}",                                     # Unknown
    fr"(?P<{OGS_C.GEO_ZONE_STR}>[{OGS_C.EMPTY_STR.join(
      OGS_C.OGS_GEO_ZONES.keys())}\s])",                          # Geo Zone
    # Event Type
    fr"(?P<{OGS_C.EVENT_TYPE_STR}>[{OGS_C.EMPTY_STR.join(
      OGS_C.OGS_EVENT_TYPES.keys())}\s])",
    fr"(?P<{OGS_C.EVENT_LOCALIZATION_STR}>[D\s])",                # Event Loc
    fr"(?P<J>[\s\d\*]{{4}})",                                     # Unknown
    fr"(?P<K>[\s\d\-\.\*]{{5}})\s",
    [
      fr"(((?P<{OGS_C.S_ONSET_STR}>[ei\s\?]){OGS_C.SWAVE}\s",     # S Onset
      fr"(?P<{OGS_C.S_WEIGHT_STR}>[0-5\s])\s",                    # S Weight
      fr"(?P<{OGS_C.S_TIME_STR}>[\s\d\.]{{5}})",                  # S Time
      fr"(?P<P>[\s\d\-\.]{{6}})",                                 # Unknown
      fr"(?P<Q>[\s\d\-\.]{{6}})\s{{2}}",                          # Unknown
      fr"(?P<R>[\s\d\.]{{4}})\s{{5}})|\s{{33}})\s"                # Unknown
    ],
    fr"(?P<S>[A-Z0-9\s]{{4}})\s{{4}}[\sgn][\sgn]*",               # Station
  ]

  # -------------------------------------------------------------------------
  # EVENT EXTRACTOR: event summary lines with hypocentral metadata
  # -------------------------------------------------------------------------
  EVENT_EXTRACTOR_LIST = [
    fr"^(?P<{OGS_C.INDEX_STR}>[\d\s]{{6}})1",                     # Event
    # Date [yymmdd hhmm]
    fr"(?P<{OGS_C.DATE_STR}>\d{{6}}\s[\s\d]{{4}})\s",
    # Seconds [ss.ss]
    fr"(?P<{OGS_C.SECONDS_STR}>[\s\d\.]{{5}})\s",
    fr"(?P<{OGS_C.LATITUDE_STR}>[\s\d\-\.]{{8}})\s{{2}}",         # Latitude
    fr"(?P<{OGS_C.LONGITUDE_STR}>[\s\d\-\.]{{8}})\s{{2}}",        # Longitude
    fr"(?P<{OGS_C.DEPTH_STR}>[\s\d\.]{{5}})\s",                   # Depth
    fr"(?P<{OGS_C.MAGNITUDE_D_STR}>[\s\-\d\.]{{6}})\s",           # Magnitude
    fr"(?P<{OGS_C.NO_STR}>[\s\d]{{2}})",                          # NO
    fr"(?P<{OGS_C.DMIN_STR}>[\s\d]{{3}})\s",                      # DMIN
    fr"(?P<{OGS_C.GAP_STR}>[\s\d]{{3}})\s1\s",                    # GAP
    fr"(?P<{OGS_C.ERT_STR}>[\s\d\.]{{4}})\s",                     # ERT
    fr"(?P<{OGS_C.ERH_STR}>[\s\d\.]{{4}})",                       # ERH
    fr"(?P<{OGS_C.ERZ_STR}>[\s\d\.]{{5}})\s",                     # ERZ
    fr"(?P<{OGS_C.QM_STR}>[A-D\s])\s",                            # QM
    fr"(([A-D]/[A-D])|\s{{3}})",                                  # Unknown
    fr"(?P<A>[\s\d\.]{{5}})\s",                                   # Unknown
    fr"(?P<B>[\s\d]{{2}})\s",                                     # Unknown
    fr"(?P<C>[\s\d]{{2}})",                                       # Unknown
    fr"(?P<D>[\-\s\d\.]{{5}})",                                   # Unknown
    fr"(?P<E>[\s\d\.]{{5}})\s",                                   # Unknown
    fr"(?P<F>[\s\d]{{2}})\s",                                     # Unknown
    fr"(?P<G>[\s\d\.]{{4}})\s",                                   # Unknown
    fr"(?P<H>[\s\d\.]{{4}})\s",                                   # Unknown
    fr"(?P<I>[\s\d]{{2}})\s",                                     # Unknown
    fr"(?P<J>[\s\d\-\.]{{4}})\s",                                 # Unknown
    fr"(?P<K>[\s\d\.]{{4}})",                                     # Unknown
    fr"(?P<L>[\s\d]{{2}})",                                       # Unknown
    fr"(?P<M>[\s\d\.]{{5}})\s",                                   # Unknown
    fr"(?P<N>[\s\d\.]{{4}})\s{{9}}",                              # Unknown
    fr"(?P<{OGS_C.NOTES_STR}>[\s\d]\d)",                          # Notes
  ]

  # -------------------------------------------------------------------------
  # OPTIONAL AUXILIARY LINES: location labels and analyst notes
  # -------------------------------------------------------------------------
  LOCATION_EXTRACTOR_LIST = [
    fr"^\^(?P<{OGS_C.LOC_NAME_STR}>[A-Z\s\.']+(\s\([A-Z\-\s]+\))?)"
  ]
  LOCATION_EXTRACTOR = re.compile(OGS_C.EMPTY_STR.join(
    list(_flatten(LOCATION_EXTRACTOR_LIST))))

  NOTES_EXTRACTOR_LIST = [
    fr"^\*\s+(?P<{OGS_C.NOTES_STR}>.*)"
  ]
  NOTES_EXTRACTOR = re.compile(OGS_C.EMPTY_STR.join(
    list(_flatten(NOTES_EXTRACTOR_LIST))))

  _PICK_COLUMNS = [
    OGS_C.IDX_PICKS_STR, OGS_C.GROUPS_STR, OGS_C.TIME_STR, OGS_C.STATION_STR,
    OGS_C.PHASE_STR, OGS_C.WEIGHT_STR, OGS_C.EPICENTRAL_DISTANCE_STR,
    OGS_C.DEPTH_STR, OGS_C.AMPLITUDE_STR, OGS_C.STATION_ML_STR,
    OGS_C.PROBABILITY_STR
  ]

  _EVENT_COLUMNS = [
    OGS_C.IDX_EVENTS_STR, OGS_C.TIME_STR, OGS_C.LONGITUDE_STR,
    OGS_C.LATITUDE_STR, OGS_C.DEPTH_STR, OGS_C.GAP_STR, OGS_C.ERZ_STR,
    OGS_C.ERH_STR, OGS_C.GROUPS_STR, OGS_C.NO_STR,
    OGS_C.NUMBER_P_PICKS_STR, OGS_C.NUMBER_S_PICKS_STR,
    OGS_C.NUMBER_P_AND_S_PICKS_STR, OGS_C.MAGNITUDE_D_STR,
    OGS_C.MAGNITUDE_L_STR, OGS_C.ML_MEDIAN_STR, OGS_C.ML_UNC_STR,
    OGS_C.ML_STATIONS_STR
  ]

  @staticmethod
  def _parse_seconds(value: str) -> td:
    """Convert a fixed-width seconds field into a timedelta."""
    return td(seconds=float(value.replace(OGS_C.SPACE_STR, OGS_C.ZERO_STR)))

  @staticmethod
  def _parse_zero_padded_int(value: str, default_value=OGS_C.NONE_STR):
    """Convert a fixed-width integer field, preserving blank-as-zero HPL use."""
    if not value:
      return default_value
    return int(value.replace(OGS_C.SPACE_STR, OGS_C.ZERO_STR))

  @staticmethod
  def _parse_float(value: str, default_value=OGS_C.NONE_STR):
    """Convert a fixed-width float field, allowing fully blank values."""
    if not value or value.strip(OGS_C.SPACE_STR) == OGS_C.EMPTY_STR:
      return default_value
    return float(value)

  @staticmethod
  def _parse_weight(value: str, default_value: int = 0) -> int:
    """Convert a weight field, using the DAT default for blanks."""
    if not value or value.strip(OGS_C.SPACE_STR) == OGS_C.EMPTY_STR:
      return default_value
    return int(value)

  @staticmethod
  def _parse_index(value: str, year: int):
    """Build a globally unique event index from the yearly counter."""
    if not value:
      return None
    return int(value.replace(OGS_C.SPACE_STR, OGS_C.ZERO_STR)) + \
      year * OGS_C.MAX_PICKS_YEAR

  @staticmethod
  def _parse_clock_time(base_time: datetime, hhmm: str) -> datetime:
    """Convert a HHMM field into a datetime anchored to an event day."""
    normalized = hhmm.replace(OGS_C.SPACE_STR, OGS_C.ZERO_STR)
    return datetime(base_time.year, base_time.month, base_time.day) + \
      td(hours=int(normalized[:2]), minutes=int(normalized[2:]))

  @staticmethod
  def _parse_coordinate(value: str):
    """Convert Hypo71 degree-minute coordinates to decimal degrees."""
    if not value:
      return OGS_C.NONE_STR

    normalized = value.replace(OGS_C.SPACE_STR, OGS_C.ZERO_STR)
    if OGS_C.DASH_STR not in normalized:
      return OGS_C.NONE_STR

    degrees, minutes = normalized.split(OGS_C.DASH_STR, maxsplit=1)
    return float(f"{float(degrees) + float(minutes) / 60.:.4f}")

  def _parse_event_datetime(self, result: dict) -> datetime:
    """Build the event origin datetime from HPL date and seconds fields."""
    event_time = datetime.strptime(
      result[OGS_C.DATE_STR].replace(OGS_C.SPACE_STR, OGS_C.ZERO_STR),
      f"{OGS_C.YYMMDD_FMT}0%H%M")
    return event_time + self._parse_seconds(result[OGS_C.SECONDS_STR])

  def _is_before_start(self, value: datetime) -> bool:
    return self.start is not None and value < self.start

  def _is_after_end(self, value: datetime) -> bool:
    return self.end is not None and value >= self.end + OGS_C.ONE_DAY

  @staticmethod
  def _is_blank_line(line: str) -> bool:
    return line.strip() == OGS_C.EMPTY_STR

  @staticmethod
  def _is_supported_event_record(result: dict) -> bool:
    """Keep distant, blank-type, and local-earthquake records."""
    if result[OGS_C.EVENT_LOCALIZATION_STR] == "D":
      return True

    event_type = result[OGS_C.EVENT_TYPE_STR]
    if event_type == OGS_C.SPACE_STR:
      return True

    return OGS_C.OGS_EVENT_TYPES.get(event_type) == OGS_C.EVENT_LOCAL_EQ_STR

  @staticmethod
  def _build_pick_row(event_index, pick_time: datetime, station: str,
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

  def _build_event_context(self, result: dict, event_time: datetime):
    context_time = datetime(
      event_time.year,
      event_time.month,
      event_time.day,
      event_time.hour,
      event_time.minute,
    ) + self._parse_seconds(result[OGS_C.SECONDS_STR])
    return (
      context_time,
      self._parse_coordinate(result[OGS_C.LONGITUDE_STR]),
      self._parse_coordinate(result[OGS_C.LATITUDE_STR]),
      self._parse_float(result[OGS_C.DEPTH_STR]),
    )

  def _build_event_row(self, result: dict, event_context):
    event_time = event_context[0]
    return [
      self._parse_index(result[OGS_C.INDEX_STR], event_time.year),
      *event_context,
      self._parse_zero_padded_int(result[OGS_C.GAP_STR]),
      result[OGS_C.ERZ_STR],
      result[OGS_C.ERH_STR],
      event_time.strftime(OGS_C.DATE_FMT),
      self._parse_zero_padded_int(result[OGS_C.NO_STR]),
      0,
      0,
      None,
      result[OGS_C.MAGNITUDE_D_STR],
      None,
      None,
      None,
      None,
    ]

  def _parse_event_summary(self, result: dict, line: str):
    event_time = self._parse_event_datetime(result)

    if self._is_before_start(event_time):
      self.logger.debug("Skipping event before start date")
      self.logger.debug(line)
      return False, None

    if self._is_after_end(event_time):
      self.logger.debug("Stopping read at event after end date")
      self.logger.debug(line)
      return True, None

    event_context = self._build_event_context(result, event_time)
    event_row = self._build_event_row(result, event_context)
    return False, (event_context, int(result[OGS_C.NOTES_STR]), event_row)

  def _event_index_from_record(self, result: dict, event_context):
    try:
      return self._parse_index(result[OGS_C.INDEX_STR], event_context[0].year)
    except ValueError as exc:
      self.logger.error(exc)
      return None

  def _parse_pick_record(self, line: str, event_context):
    match = self.RECORD_EXTRACTOR.match(line)

    if not match:
      self.logger.error(f"ERROR: (HPL) Could not parse line: {line}")
      self.debug(line, self.RECORD_EXTRACTOR_LIST)
      return False, []

    result: dict = match.groupdict()
    if not self._is_supported_event_record(result):
      self.logger.warning(f"WARNING: (HPL) Ignoring line: {line}")
      return False, []

    p_time = self._parse_clock_time(event_context[0], result[OGS_C.P_TIME_STR])
    if self._is_before_start(p_time):
      self.logger.debug(f"Skipping event before start date: {self.start}")
      self.logger.debug(line)
      return False, []

    if self._is_after_end(p_time):
      return True, []

    station = result[OGS_C.STATION_STR].strip(OGS_C.SPACE_STR)
    event_index = self._event_index_from_record(result, event_context)
    rows = [self._build_pick_row(
      event_index,
      p_time + self._parse_seconds(result[OGS_C.SECONDS_STR]),
      station,
      OGS_C.PWAVE,
      self._parse_weight(result[OGS_C.P_WEIGHT_STR]))]

    if result[OGS_C.S_TIME_STR]:
      rows.append(self._build_pick_row(
        event_index,
        p_time + self._parse_seconds(result[OGS_C.S_TIME_STR]),
        station,
        OGS_C.SWAVE,
        self._parse_weight(result[OGS_C.S_WEIGHT_STR])))

    return False, rows

  def _apply_metadata_line(self, line: str, events_data: list) -> bool:
    if self.LOCATION_EXTRACTOR.match(line):
      return True

    match = self.NOTES_EXTRACTOR.match(line)
    if not match:
      return False

    if events_data:
      events_data[-1][-2] = match.groupdict()[OGS_C.NOTES_STR].rstrip(
        OGS_C.SPACE_STR)
    return True

  def _build_picks_dataframe(self, picks_data: list) -> pd.DataFrame:
    return pd.DataFrame(picks_data, columns=self._PICK_COLUMNS).astype({
      OGS_C.IDX_PICKS_STR: int
    })

  def _build_events_dataframe(self, events_data: list) -> pd.DataFrame:
    dataframe = pd.DataFrame(events_data, columns=self._EVENT_COLUMNS)
    dataframe[OGS_C.GROUPS_STR] = pd.to_datetime(
      dataframe[OGS_C.TIME_STR], format=OGS_C.DATE_FMT)
    dataframe[OGS_C.ERH_STR] = \
      dataframe[OGS_C.ERH_STR].replace(" " * 4, "NaN").apply(float)
    dataframe[OGS_C.ERZ_STR] = \
      dataframe[OGS_C.ERZ_STR].replace(" " * 5, "NaN").apply(float)
    return dataframe

  def _apply_pick_counts(self):
    if self.PICKS.empty or self.EVENTS.empty:
      return

    event_indexes = set(self.EVENTS[OGS_C.IDX_EVENTS_STR].values)
    for idx, dataframe in self.PICKS.groupby(OGS_C.IDX_PICKS_STR):
      if idx not in event_indexes:
        continue
      for phase, phase_dataframe in dataframe.groupby(OGS_C.PHASE_STR):
        column = OGS_C.NUMBER_P_PICKS_STR if phase == OGS_C.PWAVE \
          else OGS_C.NUMBER_S_PICKS_STR
        self.EVENTS.loc[
          self.EVENTS[OGS_C.IDX_EVENTS_STR] == idx,
          column
        ] = len(phase_dataframe.index)

  def _group_pick_dataframes(self):
    for date, dataframe in self.PICKS.groupby(OGS_C.GROUPS_STR):
      self.picks[UTCDateTime(date).date] = dataframe

  def _group_event_dataframes(self):
    if self.EVENTS.empty:
      return
    for date, dataframe in self.EVENTS.groupby(OGS_C.GROUPS_STR):
      self.events[UTCDateTime(date).date] = dataframe

  def _build_dataframes(self, events_data: list, picks_data: list):
    self.PICKS = self._build_picks_dataframe(picks_data)
    self._group_pick_dataframes()

    self.EVENTS = self._build_events_dataframe(events_data)
    self.logger.info(f"Total events read: {len(self.EVENTS)}")
    self._apply_pick_counts()
    self._group_event_dataframes()

  def read(self):
    """
    Read and parse an .hpl format file into event and pick tables.

    The parser walks through the file sequentially, alternating between event
    summary lines and the following station records declared by the summary.
    Parsed picks are stored in self.PICKS / self.picks, while event-level
    metadata is stored in self.EVENTS / self.events.

    Raises:
      FileNotFoundError: If the input file does not exist.
      ValueError: If the input file does not use the .hpl extension.
    """
    # -----------------------------------------------------------------------
    # INPUT VALIDATION
    # -----------------------------------------------------------------------
    if not self.input.exists():
      raise FileNotFoundError(f"File {self.input} does not exist")

    if self.input.suffix != OGS_C.HPL_EXT:
      raise ValueError(f"File extension must be {OGS_C.HPL_EXT}")

    # -----------------------------------------------------------------------
    # STATE INITIALIZATION
    # -----------------------------------------------------------------------
    events_data = list()
    picks_data = list()
    record_lines_remaining = 0
    event_context = (datetime.min, 0, 0, 0)

    with open(self.input, 'r') as fr:
      lines = fr.readlines()
    self.logger.info(f"Reading HPL file: {self.input}")

    for raw_line in lines:
      line = raw_line.strip("\n")

      if record_lines_remaining > 0:
        record_lines_remaining -= 1
        stop_reading, pick_rows = self._parse_pick_record(line, event_context)
        if stop_reading:
          break
        picks_data.extend(pick_rows)
        continue

      match = self.EVENT_EXTRACTOR.match(line)
      if match:
        stop_reading, event_summary = self._parse_event_summary(
          match.groupdict(), line)
        if stop_reading:
          break
        if event_summary is None:
          continue
        event_context, record_lines_remaining, event_row = event_summary
        events_data.append(event_row)
        continue

      if self._apply_metadata_line(line, events_data):
        continue

      if self._is_blank_line(line):
        continue

    self._build_dataframes(events_data, picks_data)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args):
  """
  Main entry point for command-line execution.

  Processes each input file specified on the command line:
    1. Creates a DataFileHPL parser instance
    2. Reads and parses the file
    3. Logs output through the shared OGS pipeline

  Args:
    args: Parsed command-line arguments from parse_arguments()
  """
  for file in args.file:
    datafile = DataFileHPL(file, args.dates[0], args.dates[1],
                           verbose=args.verbose)
    datafile.read()
    datafile.log()


if __name__ == "__main__":
  main(parse_arguments())
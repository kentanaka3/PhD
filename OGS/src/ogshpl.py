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

  @staticmethod
  def _parse_seconds(value: str) -> td:
    """Convert a fixed-width seconds field into a timedelta."""
    return td(seconds=float(value.replace(OGS_C.SPACE_STR, OGS_C.ZERO_STR)))

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
    for line in [l.strip("\n") for l in lines]:
      if event_detect > 0:
        event_detect -= 1
        match = self.RECORD_EXTRACTOR.match(line)
        if match:
          result : dict = match.groupdict()
          if (result[OGS_C.EVENT_LOCALIZATION_STR] != "D" and
              result[OGS_C.EVENT_TYPE_STR] != "L"):
            self.logger.warning(f"WARNING: (HPL) Ignoring line: {line}")
            continue
          result[OGS_C.STATION_STR] = \
            result[OGS_C.STATION_STR].strip(OGS_C.SPACE_STR)
          # Event
          if result[OGS_C.INDEX_STR]:
            try:
              result[OGS_C.INDEX_STR] = int(result[OGS_C.INDEX_STR].replace(
                OGS_C.SPACE_STR, OGS_C.ZERO_STR)) + \
                event_spacetime[0].year * OGS_C.MAX_PICKS_YEAR
            except ValueError as e:
              result[OGS_C.INDEX_STR] = None
              self.logger.error(e)
          result[OGS_C.P_WEIGHT_STR] = int(result[OGS_C.P_WEIGHT_STR])
          result[OGS_C.SECONDS_STR] = td(seconds=float(
            result[OGS_C.SECONDS_STR].replace(
              OGS_C.SPACE_STR,
              OGS_C.ZERO_STR
            )
          ))
          result[OGS_C.P_TIME_STR] = result[OGS_C.P_TIME_STR].replace(
            OGS_C.SPACE_STR,
            OGS_C.ZERO_STR
          )
          date = event_spacetime[0]
          min_td = td(minutes=int(result[OGS_C.P_TIME_STR][2:]))
          hrs = td(hours=int(result[OGS_C.P_TIME_STR][:2]))
          result[OGS_C.P_TIME_STR] = datetime(
            date.year, date.month, date.day
          ) + hrs + min_td
          if (self.start is not None and
              result[OGS_C.P_TIME_STR] < self.start):
            event_detect = -1 # Error
            self.logger.debug("Skipping event before start date:"
                              f"{self.start}")
            self.logger.debug(line)
            continue
          if (self.end is not None and
              result[OGS_C.P_TIME_STR] >= self.end + OGS_C.ONE_DAY): break
          DETECT.append([
            result[OGS_C.INDEX_STR],
            result[OGS_C.P_TIME_STR].strftime(OGS_C.DATE_FMT),
            result[OGS_C.P_TIME_STR] + result[OGS_C.SECONDS_STR],
            f".{result[OGS_C.STATION_STR]}.",
            OGS_C.PWAVE, result[OGS_C.P_WEIGHT_STR],
            None, None, None, None, 1.0
          ])
          if result[OGS_C.S_TIME_STR]:
            result[OGS_C.S_WEIGHT_STR] = int(result[OGS_C.S_WEIGHT_STR])
            result[OGS_C.S_TIME_STR] = td(seconds=float(
              result[OGS_C.S_TIME_STR].replace(OGS_C.SPACE_STR,
                                                OGS_C.ZERO_STR)))
            DETECT.append([
              result[OGS_C.INDEX_STR],
              result[OGS_C.P_TIME_STR].strftime(OGS_C.DATE_FMT),
              result[OGS_C.P_TIME_STR] + result[OGS_C.S_TIME_STR],
              f".{result[OGS_C.STATION_STR]}.",
              OGS_C.SWAVE, result[OGS_C.S_WEIGHT_STR],
              None, None, None, None, 1.0
              #OGS_C.diff_space(), None, None, None, 1.0
            ])
          continue
      else:
        match = self.EVENT_EXTRACTOR.match(line)
        if match:
          result = match.groupdict()
          result[OGS_C.SECONDS_STR] = td(seconds=float(
            result[OGS_C.SECONDS_STR].replace(OGS_C.SPACE_STR,
                                              OGS_C.ZERO_STR)))
          result[OGS_C.DATE_STR] = datetime.strptime(
            result[OGS_C.DATE_STR].replace(OGS_C.SPACE_STR, OGS_C.ZERO_STR),
            f"{OGS_C.YYMMDD_FMT}0%H%M") + result[OGS_C.SECONDS_STR]
          if self.start is not None and result[OGS_C.DATE_STR] < self.start:
            event_detect = 0
            self.logger.debug("Skipping event before start date")
            self.logger.debug(line)
            continue
          if (self.end is not None and
              result[OGS_C.DATE_STR] >= self.end + OGS_C.ONE_DAY):
            self.logger.debug("Stopping read at event after end date")
            self.logger.debug(line)
            break
          # Event
          # # Index
          result[OGS_C.INDEX_STR] = int(int(result[OGS_C.INDEX_STR].replace(
            OGS_C.SPACE_STR, OGS_C.ZERO_STR)) + \
              result[OGS_C.DATE_STR].year * OGS_C.MAX_PICKS_YEAR)
          # # Latitude
          result[OGS_C.LATITUDE_STR] = result[OGS_C.LATITUDE_STR].replace(
            OGS_C.SPACE_STR, OGS_C.ZERO_STR) \
              if result[OGS_C.LATITUDE_STR] else OGS_C.NONE_STR
          if result[OGS_C.LATITUDE_STR] != OGS_C.NONE_STR:
            splt = result[OGS_C.LATITUDE_STR].split(OGS_C.DASH_STR)
            result[OGS_C.LATITUDE_STR] = float("{:.4f}".format(
                float(splt[0]) + float(splt[1]) / 60.))
          # # Longitude
          result[OGS_C.LONGITUDE_STR] = result[OGS_C.LONGITUDE_STR].replace(
            OGS_C.SPACE_STR, OGS_C.ZERO_STR) \
              if result[OGS_C.LONGITUDE_STR] else OGS_C.NONE_STR
          if result[OGS_C.LONGITUDE_STR] != OGS_C.NONE_STR:
            splt = result[OGS_C.LONGITUDE_STR].split(OGS_C.DASH_STR)
            result[OGS_C.LONGITUDE_STR] = float("{:.4f}".format(
                float(splt[0]) + float(splt[1]) / 60.))
          # # Depth
          result[OGS_C.DEPTH_STR] = float(result[OGS_C.DEPTH_STR]) \
            if result[OGS_C.DEPTH_STR] else OGS_C.NONE_STR
          event_spacetime = (
            datetime(
              result[OGS_C.DATE_STR].year,
              result[OGS_C.DATE_STR].month,
              result[OGS_C.DATE_STR].day,
              result[OGS_C.DATE_STR].hour,
              result[OGS_C.DATE_STR].minute
            ) + result[OGS_C.SECONDS_STR],
            result[OGS_C.LONGITUDE_STR], result[OGS_C.LATITUDE_STR],
            result[OGS_C.DEPTH_STR]
          )
          # # Number of Observations
          result[OGS_C.NO_STR] = int(result[OGS_C.NO_STR].replace(
            OGS_C.SPACE_STR,
            OGS_C.ZERO_STR
          )) if result[OGS_C.NO_STR] else OGS_C.NONE_STR
          # # Gap
          result[OGS_C.GAP_STR] = int(result[OGS_C.GAP_STR].replace(
            OGS_C.SPACE_STR,
            OGS_C.ZERO_STR
          )) if result[OGS_C.GAP_STR] else OGS_C.NONE_STR
          # # DMIN
          result[OGS_C.DMIN_STR] = float(result[OGS_C.DMIN_STR].replace(
            OGS_C.SPACE_STR,
            OGS_C.ZERO_STR
          )) if result[OGS_C.DMIN_STR] else OGS_C.NONE_STR
          # # ERT
          result[OGS_C.ERT_STR] = float(result[OGS_C.ERT_STR].replace(
            OGS_C.SPACE_STR,
            OGS_C.ZERO_STR
          )) if result[OGS_C.ERT_STR] else OGS_C.NONE_STR
          # # Quality Metric
          result[OGS_C.QM_STR] = result[OGS_C.QM_STR].strip(OGS_C.SPACE_STR) \
            if result[OGS_C.QM_STR] else OGS_C.NONE_STR
          event_detect = int(result[OGS_C.NOTES_STR])
          SOURCE.append([
            result[OGS_C.INDEX_STR], *event_spacetime, result[OGS_C.GAP_STR],
            result[OGS_C.ERZ_STR], result[OGS_C.ERH_STR],
            event_spacetime[0].strftime(OGS_C.DATE_FMT), result[OGS_C.NO_STR],
            0, 0, None, result[OGS_C.MAGNITUDE_D_STR], None, None, None, None
          ])
          continue
        match = self.LOCATION_EXTRACTOR.match(line)
        if match:
          result = match.groupdict()
          continue
        if event_detect == 0:
          match = self.NOTES_EXTRACTOR.match(line)
          if match:
            result = match.groupdict()
            event_notes = result[OGS_C.NOTES_STR].rstrip(OGS_C.SPACE_STR)
            if len(SOURCE) > 0:
              SOURCE[-1][-2] = event_notes
            continue
        if re.match(r"^\s*$", line): continue
      if event_detect < 0:
        self.logger.error(f"ERROR: (HPL) Could not parse line: {line}")
        self.debug(line, self.EVENT_EXTRACTOR_LIST if event_detect == 0
                   else self.RECORD_EXTRACTOR_LIST)
    self.PICKS = pd.DataFrame(DETECT, columns=[
      OGS_C.IDX_PICKS_STR, OGS_C.GROUPS_STR, OGS_C.TIME_STR, OGS_C.STATION_STR,
      OGS_C.PHASE_STR, OGS_C.WEIGHT_STR, OGS_C.EPICENTRAL_DISTANCE_STR,
      OGS_C.DEPTH_STR, OGS_C.AMPLITUDE_STR, OGS_C.STATION_ML_STR,
      OGS_C.PROBABILITY_STR
    ]).astype({OGS_C.IDX_PICKS_STR: int})

    for date, dataframe in self.PICKS.groupby(OGS_C.GROUPS_STR):
      self.picks[UTCDateTime(date).date] = dataframe

    self.EVENTS = pd.DataFrame(events_data, columns=[
      OGS_C.IDX_EVENTS_STR, OGS_C.TIME_STR, OGS_C.LONGITUDE_STR,
      OGS_C.LATITUDE_STR, OGS_C.DEPTH_STR, OGS_C.GAP_STR, OGS_C.ERZ_STR,
      OGS_C.ERH_STR, OGS_C.GROUPS_STR, OGS_C.NO_STR,
      OGS_C.NUMBER_P_PICKS_STR, OGS_C.NUMBER_S_PICKS_STR,
      OGS_C.NUMBER_P_AND_S_PICKS_STR, OGS_C.MAGNITUDE_D_STR,
      OGS_C.MAGNITUDE_L_STR, OGS_C.ML_MEDIAN_STR, OGS_C.ML_UNC_STR,
      OGS_C.ML_STATIONS_STR
    ])
    self.EVENTS[OGS_C.GROUPS_STR] = pd.to_datetime(
      self.EVENTS[OGS_C.TIME_STR], format=OGS_C.DATE_FMT)
    self.EVENTS[OGS_C.ERH_STR] = \
      self.EVENTS[OGS_C.ERH_STR].replace(" " * 4, "NaN").apply(float)
    self.EVENTS[OGS_C.ERZ_STR] = \
      self.EVENTS[OGS_C.ERZ_STR].replace(" " * 5, "NaN").apply(float)

    self.logger.info(f"Total events read: {len(self.EVENTS)}")

    for idx, dataframe in self.PICKS.groupby(OGS_C.IDX_PICKS_STR):
      for phase, phase_dataframe in dataframe.groupby(OGS_C.PHASE_STR):
        if idx in self.EVENTS[OGS_C.IDX_EVENTS_STR].values:
          self.EVENTS.loc[
            self.EVENTS[OGS_C.IDX_EVENTS_STR] == idx,
            OGS_C.NUMBER_P_PICKS_STR if phase == OGS_C.PWAVE
            else OGS_C.NUMBER_S_PICKS_STR
          ] = len(phase_dataframe.index)
      # TODO: Compute NUMBER_P_AND_S_PICKS_STR

    if not self.EVENTS.empty:
      for date, dataframe in self.EVENTS.groupby(OGS_C.GROUPS_STR):
        self.events[UTCDateTime(date).date] = dataframe


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
"""
===============================================================================
OGS HPL File Parser - Hypo71 Event and Pick Extractor
===============================================================================

OVERVIEW:
This module parses OGS .hpl format files produced by legacy Hypo71 workflows.
An HPL file stores event summary rows together with station-level P and
optional S picks, plus optional analyst notes and locality lines. Locality
lines are recognized and ignored; notes update the last retained event.

FILE FORMAT DESCRIPTION:
  The .hpl format is organized as fixed-width text blocks:
  - One event summary line with origin time, hypocenter, and quality metrics
  - A variable number of station records containing phase picks
  - Optional location and free-text note lines after the station block

KEY FEATURES:
  - Regex-based extraction of event, station, location, and notes lines
  - Separate DataFrame construction for picks and event metadata
  - Date range filtering after reading the file into memory
  - Selected Hypo71 quality metrics and the last note per retained event
  - Parquet output via the shared OGSDataFile logging pipeline

USAGE:
  Command line:
    python -m OGS.src.ogshpl -f input.hpl -D 20240320 20240620 -v

  Programmatic:
    from pathlib import Path
    from OGS.src.ogshpl import DataFileHPL
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

# Standard library: regular expressions for pattern matching
import re

# Standard library: date/time objects and time deltas
from datetime import datetime, timedelta as td

from . import ogsconstants as OGS_C, ogsutils as OGS_U
from .ogsdatafile import OGSDataFile

# =============================================================================
# DataFileHPL Class - HPL Format Parser
# =============================================================================

class DataFileHPL(OGSDataFile):
  """
  Parser for OGS .hpl format event summaries and phase picks.

  Extends OGSDataFile to parse legacy Hypo71 fixed-width output. HPL files
  contain event summary records followed by a summary-declared number of
  station records. Locality lines are ignored and note lines replace the
  last retained event's notes. Not every captured summary field is exported.

  Attributes:
    RECORD_EXTRACTOR_LIST: Regex patterns for station phase-pick records
    EVENT_EXTRACTOR_LIST: Regex patterns for event summary records
    LOCATION_EXTRACTOR_LIST: Regex pattern for locality description lines
    NOTES_EXTRACTOR_LIST: Regex pattern for analyst note lines
  """

  EXTENSION: str = OGS_C.HPL_EXT

  # -------------------------------------------------------------------------
  # RECORD EXTRACTOR: station-level phase pick lines
  # -------------------------------------------------------------------------
  # Many fixed-width columns are still not mapped to domain names, so the
  # unknown fields remain intentionally positional until the format is decoded.
  RECORD_EXTRACTOR_LIST = [
      # Event index: six whitespace/digit characters, normalized with year
      fr"^(?P<{OGS_C.IDX_EVENTS_STR}>[\d\s]{{6}})\s",
      # Station
      fr"(?P<{OGS_C.STATION_STR}>[A-Z0-9\s]{{4}})\s",
      # Unknown field: five whitespace/digit/period characters
      fr"([\d\s\.]{{5}})\s",
      # Unknown field: three whitespace/digit characters
      fr"([\d\s]{{3}})\s",
      # Unknown field: three whitespace/digit characters
      fr"([\d\s]{{3}})\s",
      # P-wave onset quality: e=emergent, i=impulsive, ?=uncertain, space=unknown
      fr"(?P<{OGS_C.P_ONSET_STR}>[ei?\s]){OGS_C.PWAVE}",
      # P-wave polarity: c/C/+=compression(up), d/D/-=dilatation(down), space=unknown
      fr"(?P<{OGS_C.P_POLARITY_STR}>[cC\+dD\-\s])",
      # P-wave weight: one required digit from 0 through 4
      fr"(?P<{OGS_C.P_WEIGHT_STR}>[0-4])\s",
      # P-wave clock base: four whitespace/digit characters (HHMM)
      fr"(?P<{OGS_C.P_TIME_STR}>[\s\d]{{4}})\s",
      # P-wave seconds: five whitespace/digit/period characters
      fr"(?P<{OGS_C.SECONDS_STR}>[\s\d\.]{{5}})",
      # Unknown field: six whitespace/digit/minus/period characters
      fr"(?P<A>[\s\d\-\.]{{6}})\s",
      # Unknown field: five whitespace/digit/minus/period characters
      fr"(?P<B>[\s\d\-\.]{{5}})\s",
      # Unknown field: five whitespace/digit/minus/period characters
      fr"(?P<C>[\s\d\-\.]{{5}})",
      # Unknown field: six whitespace/digit/minus/period characters
      fr"(?P<D>[\s\d\-\.]{{6}})\s",
      # Unknown field: five whitespace/digit/minus/period characters
      fr"(?P<E>[\s\d\-\.]{{5}})\s",
      # Unknown field: three whitespace/digit/minus/period characters
      fr"(?P<F>[\s\d\-\.]{{3}})\s",
      # Unknown field: two whitespace/digit/minus/period characters
      fr"(?P<G>[\s\d\-\.]{{2}})\s",
      # Unknown field: five whitespace/digit/minus/period characters
      fr"(?P<H>[\s\d\-\.]{{5}})\s",
      # Unknown field: one whitespace/digit character, then six whitespace chars
      fr"(?P<I>[\s\d])\s{{6}}",
      # Geographical zone code
      fr"(?P<{OGS_C.GEO_ZONE_STR}>" + (
          fr"[{OGS_C.EMPTY_STR.join(OGS_C.OGS_GEO_ZONES.keys())}\s]"
      ) + fr")",
      # Event type: B/E/F/L/R/U from OGS_EVENT_TYPES, or whitespace
      fr"(?P<{OGS_C.EVENT_TYPE_STR}>" + (
          fr"[{OGS_C.EMPTY_STR.join(OGS_C.OGS_EVENT_TYPES.keys())}\s]"
      ) + fr")",
      # Event localization flag: D or whitespace
      fr"(?P<{OGS_C.EVENT_LOCALIZATION_STR}>[D\s])",
      # Unknown field: four whitespace/digit/asterisk characters
      fr"(?P<J>[\s\d\*]{{4}})",
      # Unknown field: five whitespace/digit/minus/period/asterisk characters
      fr"(?P<K>[\s\d\-\.\*]{{5}})\s",
      # Optional S-wave pick block (may be 33 spaces if no S pick)
      [
          # S-wave onset quality: e=emergent, i=impulsive, ?=uncertain, space=unknown
          fr"(((?P<{OGS_C.S_ONSET_STR}>[ei\s\?]){OGS_C.SWAVE}\s",
          # S-wave weight: digit 0-5 or whitespace (blank defaults to zero)
          fr"(?P<{OGS_C.S_WEIGHT_STR}>[0-5\s])\s",
          # S-wave seconds offset from the P-wave HHMM base: five characters
          fr"(?P<{OGS_C.S_TIME_STR}>[\s\d\.]{{5}})",
          # Unknown field: six whitespace/digit/minus/period characters
          fr"(?P<P>[\s\d\-\.]{{6}})",
          # Unknown field: six whitespace/digit/minus/period characters
          fr"(?P<Q>[\s\d\-\.]{{6}})\s{{2}}",
          # Unknown field: four whitespace/digit/period characters
          fr"(?P<R>[\s\d\.]{{4}})\s{{5}})|\s{{33}})\s"
      ],
      # Unknown field: four uppercase-letter/digit/whitespace characters
      fr"(?P<S>[A-Z0-9\s]{{4}})\s{{4}}"
      # Unknown suffix: one or more whitespace/g/n characters
      fr"[\sgn][\sgn]*",
      # End of line anchor to ensure full-line match
      fr"$"
  ]
  # print(OGS_C.EMPTY_STR.join(OGSDataFile._flatten(RECORD_EXTRACTOR_LIST)))

  # -------------------------------------------------------------------------
  # EVENT EXTRACTOR: event summary lines with hypocentral metadata
  # -------------------------------------------------------------------------
  EVENT_EXTRACTOR_LIST = [
      fr"^(?P<{OGS_C.IDX_EVENTS_STR}>[\d\s]{{6}})1",          # Event
      # Date [yymmdd hhmm]
      fr"(?P<{OGS_C.DATE_STR}>\d{{6}}\s[\s\d]{{4}})\s",
      fr"(?P<{OGS_C.SECONDS_STR}>[\s\d\.]{{5}})\s",           # Seconds [ss.ss]
      fr"(?P<{OGS_C.LATITUDE_STR}>[\s\d\-\.]{{8}})\s{{2}}",   # Latitude
      fr"(?P<{OGS_C.LONGITUDE_STR}>[\s\d\-\.]{{8}})\s{{2}}",  # Longitude
      fr"(?P<{OGS_C.DEPTH_STR}>[\s\d\.]{{5}})\s",             # Depth
      fr"(?P<{OGS_C.MAGNITUDE_D_STR}>[\s\-\d\.]{{6}})",       # Hypo71 mag
      fr"(?P<{OGS_C.NO_STR}>[\s\d]{{3}})",                    # NO
      fr"(?P<{OGS_C.DMIN_STR}>[\s\d]{{3}})\s",                # DMIN
      fr"(?P<{OGS_C.GAP_STR}>[\s\d]{{3}})\s1",                # GAP
      fr"(?P<{OGS_C.RMS_STR}>[\s\d\.]{{5}})",                 # RMS residual
      fr"(?P<{OGS_C.ERH_STR}>[\s\d\.]{{5}})",                 # ERH
      fr"(?P<{OGS_C.ERZ_STR}>[\s\d\.]{{5}})\s",               # ERZ
      fr"(?P<{OGS_C.QM_STR}>[A-D\s])\s",                      # QM
      fr"(([A-D]/[A-D])|\s{{3}})",                            # Unknown
      fr"(?P<{OGS_C.HPL_AUX_FLOAT_STR}>[\s\d\.]{{5}})\s",     # Aux float
      fr"(?P<{OGS_C.VELOCITY_MODEL_ID_STR}>[\s\d]{{2}})",     # Velocity Model
      fr"(?P<{OGS_C.PHASES_USED_STR}>[\s\d]{{3}})",           # Num Picks
      fr"(?P<{OGS_C.MEAN_RESIDUAL_STR}>[\-\s\d\.]{{5,6}})",   # Mean Residual
      fr"(?P<{OGS_C.STD_RESIDUAL_STR}>[\s\d\.]{{5}})\s",      # StDev Residual
      fr"(?P<{OGS_C.ML_STATIONS_STR}>[\s\d]{{2}})\s",         # ML stations
      fr"(?P<{OGS_C.MAGNITUDE_L_STR}>[\s\d\-\.]{{4}})\s",     # ML
      fr"(?P<{OGS_C.ML_UNC_STR}>[\s\d\.]{{4}})\s",            # ML_unc
      fr"(?P<{OGS_C.MD_STATIONS_STR}>[\s\d]{{2}})\s",         # MD stations
      fr"(?P<{OGS_C.HYPO71_MAG_STR}>[\s\d\-\.]{{4}})\s",      # MD
      fr"(?P<{OGS_C.MD_UNC_STR}>[\s\d\.]{{4}})",              # MD_unc
      # 3rd magnitude station count
      fr"(?P<{OGS_C.M3_STATIONS_STR}>[\s\d]{{2}})",
      # 3rd magnitude value
      fr"(?P<{OGS_C.M3_MAGNITUDE_STR}>[\s\d\.]{{5}})\s",
      # 3rd magnitude uncertainty
      fr"(?P<{OGS_C.M3_UNC_STR}>[\s\d\.]{{4}})\s{{9}}",
      # Number of pick lines remaining
      fr"(?P<{OGS_C.NO_STR}_picks>[\s\d]\d)",
  ]

  # -------------------------------------------------------------------------
  # OPTIONAL AUXILIARY LINES: location labels and analyst notes
  # -------------------------------------------------------------------------
  LOCATION_EXTRACTOR_LIST = [
      fr"^\^(?P<{OGS_C.LOC_NAME_STR}>[A-Z\s\.']+(\s\([A-Z\-\s]+\))?)"
  ]
  LOCATION_EXTRACTOR = re.compile(OGS_C.EMPTY_STR.join(
      list(OGSDataFile._flatten(LOCATION_EXTRACTOR_LIST))
  ))

  NOTES_EXTRACTOR_LIST = [
      fr"^\*\s+(?P<{OGS_C.NOTES_STR}>.*)"
  ]
  NOTES_EXTRACTOR = re.compile(OGS_C.EMPTY_STR.join(
      list(OGSDataFile._flatten(NOTES_EXTRACTOR_LIST))
  ))

  @staticmethod
  def _parse_clock_time(base_time: datetime, hhmm: str) -> datetime:
    """Convert a HHMM field into a datetime anchored to an event day."""
    normalized = hhmm.replace(OGS_C.SPACE_STR, OGS_C.ZERO_STR)
    return datetime(base_time.year, base_time.month, base_time.day) + \
        td(hours=int(normalized[:2]), minutes=int(normalized[2:]))

  def _parse_event_datetime(self, result: dict) -> datetime:
    """Build the event origin datetime from HPL date and seconds fields."""
    event_time = datetime.strptime(
        result[OGS_C.DATE_STR].replace(OGS_C.SPACE_STR, OGS_C.ZERO_STR),
        f"{OGS_C.YYMMDD_FMT}0%H%M")
    return event_time + self._parse_seconds(result[OGS_C.SECONDS_STR])

  @staticmethod
  def _is_supported_event_record(result: dict) -> bool:
    """Keep distant, blank-type, and local-earthquake records."""
    if result[OGS_C.EVENT_LOCALIZATION_STR] == "D":
      return True

    event_type = result[OGS_C.EVENT_TYPE_STR]
    if event_type == OGS_C.SPACE_STR:
      return True

    return OGS_C.OGS_EVENT_TYPES.get(event_type) == OGS_C.EVENT_LOCAL_EQ_STR

  def _apply_metadata_line(
      self, line: str, events_data: list[dict[str, object]],
  ) -> bool:
    if self.LOCATION_EXTRACTOR.match(line):
      return True

    match = self.NOTES_EXTRACTOR.match(line)
    if not match:
      return False

    if events_data:
      events_data[-1][OGS_C.NOTES_STR] = (
          match.groupdict()[OGS_C.NOTES_STR].rstrip(OGS_C.SPACE_STR)
      )
    return True

  def _build_events_dataframe(self, events_data: list) -> pd.DataFrame:
    dataframe = pd.DataFrame(events_data, columns=self._EVENT_COLUMNS)
    dataframe = self.normalize_groups(dataframe)
    dataframe[OGS_C.ERH_STR] = \
        dataframe[OGS_C.ERH_STR].replace(" " * 5, "NaN").apply(float)
    dataframe[OGS_C.ERZ_STR] = \
        dataframe[OGS_C.ERZ_STR].replace(" " * 5, "NaN").apply(float)
    for col in [
        OGS_C.MAGNITUDE_D_STR, OGS_C.MAGNITUDE_L_STR, OGS_C.ML_MEDIAN_STR,
        OGS_C.ML_UNC_STR, OGS_C.ML_STATIONS_STR
    ]:
      if col in dataframe.columns:
        dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
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
    self.postload("picks", update=True)

    self.EVENTS = self._build_events_dataframe(events_data)
    self.logger.info(f"Total events read: {len(self.EVENTS)}")
    self.EVENTS = self.normalize_pick_stats(self.EVENTS, self.PICKS)
    self.postload("events", update=True)

  def read(self):
    """
    Read and parse an .hpl format file into event and pick tables.

    The parser walks through the file sequentially, alternating between event
    summary lines and the following station records declared by each retained
    summary. Recognized unlocated events skip their declared following lines
    (falling back to one line). Event summaries use origin-time filtering;
    station records use their HHMM base before adding P/S seconds.
    Parsed picks are stored in self.PICKS / self.picks, while event-level
    metadata is stored in self.EVENTS / self.events.

    Returns:
      None: Results are stored on the instance.

    Raises:
      FileNotFoundError: If the input file does not exist.
      ValueError: If the suffix is not .hpl or a field conversion fails.
      OSError: If opening or reading the file fails.
      Other processing failures propagate; regex-mismatched station lines
      are logged and skipped.
    """
    # -----------------------------------------------------------------------
    # INPUT VALIDATION
    # -----------------------------------------------------------------------
    self.validate_input()

    # -----------------------------------------------------------------------
    # STATE INITIALIZATION
    # -----------------------------------------------------------------------
    events_data: list[dict[str, object]] = []
    picks_data = list()
    record_lines_remaining = 0
    event_time = datetime.min

    with open(self.input, 'r') as fr:
      lines = fr.readlines()
    self.logger.info(f"Reading HPL file: {self.input}")

    for raw_line in lines:
      line = raw_line.strip("\n")

      if record_lines_remaining > 0:
        record_lines_remaining -= 1
        match = self.RECORD_EXTRACTOR.match(line)
        if not match:
          self.logger.error(f"ERROR: (HPL) Could not parse line: {line}")
          self.debug(line, self.RECORD_EXTRACTOR_LIST)
          continue

        result = match.groupdict()
        if not self._is_supported_event_record(result):
          self.logger.warning(f"WARNING: (HPL) Ignoring line: {line}")
          continue

        result[OGS_C.P_TIME_STR] = self._parse_clock_time(
            event_time, result[OGS_C.P_TIME_STR]
        )
        if self._is_before_start(result[OGS_C.P_TIME_STR]):
          self.logger.debug(f"Skipping event before start date: {self.start}")
          self.logger.debug(line)
          continue

        if self._is_after_end(result[OGS_C.P_TIME_STR]):
          continue

        result[OGS_C.STATION_STR] = result[OGS_C.STATION_STR].strip(
            OGS_C.SPACE_STR
        )
        result[OGS_C.SECONDS_STR] = self._parse_seconds(
            result[OGS_C.SECONDS_STR]
        )
        result[OGS_C.P_WEIGHT_STR] = self._parse_weight(
            result[OGS_C.P_WEIGHT_STR]
        )
        result[OGS_C.IDX_EVENTS_STR] = self.normalize_index(
            result[OGS_C.IDX_EVENTS_STR], event_time.year
        )
        picks_data.append(self._build_pick_row(
            result[OGS_C.IDX_EVENTS_STR],
            result[OGS_C.P_TIME_STR] + result[OGS_C.SECONDS_STR],
            result[OGS_C.STATION_STR],
            OGS_C.PWAVE,
            result[OGS_C.P_WEIGHT_STR],
        ))
        if result[OGS_C.S_TIME_STR]:
          result[OGS_C.S_TIME_STR] = self._parse_seconds(
              result[OGS_C.S_TIME_STR]
          )
          result[OGS_C.S_WEIGHT_STR] = self._parse_weight(
              result[OGS_C.S_WEIGHT_STR]
          )
          picks_data.append(self._build_pick_row(
              result[OGS_C.IDX_EVENTS_STR],
              result[OGS_C.P_TIME_STR] + result[OGS_C.S_TIME_STR],
              result[OGS_C.STATION_STR],
              OGS_C.SWAVE,
              result[OGS_C.S_WEIGHT_STR],
          ))
        continue

      match = self.EVENT_EXTRACTOR.match(line)
      if match:
        result: dict = match.groupdict()
        event_time = self._parse_event_datetime(result)

        if self._is_before_start(event_time):
          self.logger.debug("Skipping event before start date")
          self.logger.debug(line)
          continue

        if self._is_after_end(event_time):
          self.logger.debug("Stopping read at event after end date")
          self.logger.debug(line)
          continue

        result[OGS_C.TIME_STR] = event_time
        result[OGS_C.IDX_EVENTS_STR] = self.normalize_index(
            result[OGS_C.IDX_EVENTS_STR], event_time.year
        )
        result[OGS_C.GAP_STR] = self._parse_zero_padded_int(
            result[OGS_C.GAP_STR]
        )
        result[OGS_C.NO_STR] = self._parse_zero_padded_int(
            result[OGS_C.NO_STR]
        )
        result[OGS_C.ML_STATIONS_STR] = self._parse_zero_padded_int(
            result[OGS_C.ML_STATIONS_STR]
        )
        result[OGS_C.MD_STATIONS_STR] = self._parse_zero_padded_int(
            result[OGS_C.MD_STATIONS_STR]
        )
        result[OGS_C.MD_UNC_STR] = (
            self._parse_float(result[OGS_C.MD_UNC_STR])
            if result[OGS_C.MAGNITUDE_D_STR] is not None
            else None
        )
        result[OGS_C.MAGNITUDE_L_STR] = (
            self._parse_float(result[OGS_C.MAGNITUDE_L_STR])
            if result[OGS_C.ML_STATIONS_STR]
            else None
        )
        result[OGS_C.ML_UNC_STR] = (
            self._parse_float(result[OGS_C.ML_UNC_STR])
            if result[OGS_C.MAGNITUDE_L_STR] is not None
            else None
        )
        record_lines_remaining = int(result[f"{OGS_C.NO_STR}_picks"])
        events_data.append(result)
        continue

      if self._apply_metadata_line(line, events_data):
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
    args: Parsed command-line arguments from ogsutils.parse_hpl_args()
  """
  for file in args.file:
    datafile = DataFileHPL(file, args.dates[0], args.dates[1],
                           verbose=args.verbose)
    datafile.read()
    datafile.log()


if __name__ == "__main__":
  main(OGS_U.parse_hpl_args())

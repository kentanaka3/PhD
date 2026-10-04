"""
===============================================================================
OGS DAT File Parser - Seismic Phase Pick Extractor
===============================================================================

OVERVIEW:
This module parses OGS .dat format files containing seismic phase picks
(P and S wave arrival times) recorded by the OGS seismic network. The DAT
format is a legacy fixed-width text format used for manual analyst picks.

FILE FORMAT DESCRIPTION:
  The .dat format uses fixed-width columns with the following structure:
  - Columns 1-4:   Station code (4 chars, right-padded)
  - Column 5:      P-wave onset quality (e/i/?)
  - Column 6:      P marker or whitespace
  - Column 7:      P-wave polarity (+/-/c/C/d/D or whitespace)
  - Column 8:      P-wave weight (0-4 or whitespace)
  - Column 9:      "1" marker or whitespace
  - Columns 10-19: Date-time (YYMMDDHHMM format)
  - Column 20:     Whitespace or zero separator
  - Columns 21-24: P-wave seconds/centiseconds (SSCC)
  - Columns 25-32: Ignored field (any eight characters)
  - Columns 33-40: S seconds/centiseconds, onset, S marker, polarity, weight;
                  alternatively eight whitespace characters
  - Columns 41-62: Whitespace padding
  - Column 63:     Geographic zone code or whitespace
  - Column 64:     Event type code (B/E/F/L/U or whitespace)
  - Column 65:     Event localization flag (D or whitespace)
  - Columns 66-70: Whitespace padding
  - Columns 71-75: Captured duration field (not converted or exported)
  - Columns 76-79: Event index number
  These positions describe the regex input after stripping each raw line.
  Records are prefix-matched; trailing text is not rejected.

KEY FEATURES:
  - Regex-based parsing with named capture groups
  - P and S wave pick extraction from the same record
  - Date range filtering for temporal subsetting
  - Numeric pick weights, defaulting to zero for blanks
  - Retention of distant, blank-type, and local-earthquake records
  - Parquet output for efficient storage

USAGE:
  Command line:
    python -m OGS.src.ogsdat -f input.dat -D 20220101 20221231 -v

  Programmatic:
    from pathlib import Path
    from OGS.src.ogsdat import DataFileDAT
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

# Standard library: Regular expressions for pattern matching
import re

# Standard library: Date/time objects and time deltas
from datetime import datetime, timedelta as td

from . import ogsconstants as OGS_C, ogsutils as OGS_U
from .ogsdatafile import OGSDataFile


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

  # Expected file extension for format validation
  EXTENSION: str = OGS_C.DAT_EXT

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
      fr"(?P<{OGS_C.P_ONSET_STR}>[ei\s\?])[{OGS_C.PWAVE}\s]",
      # P-wave polarity: c/C/+=compression(up), d/D/-=dilatation(down), space=unknown
      fr"(?P<{OGS_C.P_POLARITY_STR}>[cC\+dD\-\s])",
      # P-wave weight: digit 0-4 or whitespace (blank defaults to zero)
      fr"(?P<{OGS_C.P_WEIGHT_STR}>[0-4\s])",
      # Marker "1" or whitespace
      fr"[1\s]",
      # Date-time: YYMMDDHHMM format (10 digits) followed by space or zero
      fr"(?P<{OGS_C.DATE_STR}>\d{{10}})[\s0]",
      # P-wave seconds/centiseconds: four whitespace/digit characters
      fr"(?P<{OGS_C.P_TIME_STR}>[\s\d]{{4}})",
      # Reserved/unknown field: 8 characters (ignored)
      fr".{{8}}",
      # Optional S-wave data block (may be 8 spaces if no S pick)
      [
          # S-wave seconds/centiseconds: four whitespace/digit characters
          fr"(((?P<{OGS_C.S_TIME_STR}>[\s\d]{{4}})",
          # S-wave onset quality: e=emergent, i=impulsive, ?=uncertain, space=unknown
          fr"(?P<{OGS_C.S_ONSET_STR}>[ei\s\?]){OGS_C.SWAVE}",
          # S-wave polarity: c/C/+=compression(up), d/D/-=dilatation(down), space=unknown
          fr"(?P<{OGS_C.S_POLARITY_STR}>[cC\+dD\-\s])",
          # S-wave weight: digit 0-5 or whitespace (blank defaults to zero)
          fr"(?P<{OGS_C.S_WEIGHT_STR}>[0-5\s]))|\s{{8}})"
      ],
      # Padding: 22 spaces to align with fixed-width format
      fr"\s{{22}}",
      # Geographic zone code
      fr"(?P<{OGS_C.GEO_ZONE_STR}>" + (
          fr"[{OGS_C.EMPTY_STR.join(OGS_C.OGS_GEO_ZONES.keys())}\s]"
      ) + fr")",
      # Event type: B/E/F/L/U from OGS_EVENT_TYPES, or whitespace
      fr"(?P<{OGS_C.EVENT_TYPE_STR}>" + (
          fr"[{OGS_C.EMPTY_STR.join(OGS_C.OGS_EVENT_TYPES.keys())}\s]"
      ) + fr")",
      # Event localization flag: D or whitespace
      fr"(?P<{OGS_C.EVENT_LOCALIZATION_STR}>[D\s])",
      # Padding: 5 spaces
      fr"\s{{5}}",
      # Duration: five whitespace/digit characters, captured but not exported
      fr"(?P<{OGS_C.DURATION_STR}>[\s\d]{{5}})",
      # Event index: four whitespace/digit characters, normalized with year
      fr"(?P<{OGS_C.IDX_EVENTS_STR}>[\s\d]{{4}})",
      fr""
  ]

  # -------------------------------------------------------------------------
  # EVENT EXTRACTOR: Metadata-only lines without pick data
  # -------------------------------------------------------------------------
  EVENT_EXTRACTOR_LIST = [
      # Event type: B/E/F/L/U from OGS_EVENT_TYPES, or whitespace
      fr"(?P<{OGS_C.EVENT_TYPE_STR}>" + (
          fr"[{OGS_C.EMPTY_STR.join(OGS_C.OGS_EVENT_TYPES.keys())}\s]"
      ) + fr")",
      # Event localization flag: D or whitespace
      fr"(?P<{OGS_C.EVENT_LOCALIZATION_STR}>[D\s])",
      # Padding: 5 spaces
      fr"\s{{5}}",
      # Duration: five whitespace/digit characters
      fr"(?P<{OGS_C.DURATION_STR}>[\s\d]{{5}})",
      # Event index: four whitespace/digit characters
      fr"(?P<{OGS_C.IDX_EVENTS_STR}>[\s\d]{{4}})",
  ]

  @staticmethod
  def _parse_event_datetime(value: str) -> datetime:
    """Parse YYMMDDHHMM; minutes >= 60 become the next hour at minute zero."""
    if int(value[-2:]) >= 60:
      return datetime.strptime(
          value[:-2], OGS_C.DATETIME_FMT[:-4]
      ) + td(hours=1)
    return datetime.strptime(value, OGS_C.DATETIME_FMT[:-2])

  @staticmethod
  def _parse_pick_time(base_time: datetime, value: str) -> datetime:
    """Add SSCC / 100 seconds to base_time, replacing spaces with zeros."""
    offset = float(value.replace(OGS_C.SPACE_STR, OGS_C.ZERO_STR)) / 100.
    return base_time + td(seconds=offset)

  def read(self):
    """
    Read and parse a .dat format file into P and S wave picks.

    The parser walks through the input file line by line, skips metadata-only
    lines, extracts station records with regex patterns, filters by date range
    and event type, and stores grouped picks in self.PICKS and self.picks.
    Invalid regex records and handled field-conversion failures are logged and
    skipped. An invalid event index is retained as None before the DataFrame
    builder converts it to zero.

    Returns:
      None: Results are stored on the instance.

    Raises:
      FileNotFoundError: If the input file does not exist.
      ValueError: If the input file does not use the .dat extension.
      RuntimeError: If opening or reading fails; a .dat.corrupt copy is
        attempted before raising. Other processing failures may propagate.
    """
    # -----------------------------------------------------------------------
    # INPUT VALIDATION
    # -----------------------------------------------------------------------
    self.validate_input()

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
      if self._is_before_start(event_time):
        self.logger.debug(f"Skipping pick before start date: {self.start}")
        self.logger.debug(line)
        continue

      if self._is_after_end(event_time):
        self.logger.debug(
            f"Stopping read at pick after end date: {self.end}"
        )
        self.logger.debug(line)
        continue

      # ---------------------------------------------------------------------
      # FIELD PROCESSING
      # ---------------------------------------------------------------------
      station = result[OGS_C.STATION_STR].strip(OGS_C.SPACE_STR)

      try:
        event_index = self.normalize_index(
            result[OGS_C.IDX_EVENTS_STR], event_time.year
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
    self.PICKS = self._build_picks_dataframe(pick_records)
    self.postload("picks", update=True)


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
    args: Parsed command-line arguments from ogsutils.parse_dat_args()
  """
  for file in args.file:
    datafile = DataFileDAT(file, args.dates[0], args.dates[1],
                           verbose=args.verbose)
    datafile.read()
    datafile.log()


if __name__ == "__main__":
  main(OGS_U.parse_dat_args())

import re
import pandas as pd
import itertools as it
from pathlib import Path
from obspy import UTCDateTime
from datetime import timedelta as td
from matplotlib.cbook import flatten as flatten_list
from obspy.core.event import read_events as obspy_read_events
from concurrent.futures import ThreadPoolExecutor

from constants import *
from errors import ERRORS

DEBUG = False


def event_merger_l(NEW: pd.DataFrame, OLD: pd.DataFrame, on: list) \
        -> pd.DataFrame:
  if NEW.empty:
    return OLD
  if OLD.empty:
    return NEW
  cols_n = NEW.columns
  rows_o = len(OLD.index)
  off = [col for col in cols_n if col not in on]
  # if DEBUG:
  #   cols_o = OLD.columns
  #   assert all([col in cols_n and col in cols_o for col in on])
  #   assert all([col_a == col_b for col_a, col_b in zip(cols_n, cols_o)])
  idx_r: int = 0
  for idx_l, row in NEW.iterrows():
    if idx_r < rows_o and all([row[col] == OLD.loc[idx_r][col] for col in on]):
      for col in off:
        row[col] = OLD.loc[idx_r, col] if OLD.loc[idx_r, col] else row[col]
      NEW.loc[idx_l] = row
      idx_r += 1
  return NEW


# TODO: Implement polarity
RECORD_EXTRACTOR_DAT_LIST = [
    fr"^(?P<{STATION_STR}>[A-Z0-9\s]{{4}})",                       # Station
    fr"(?P<{P_ONSET_STR}>[ei\s\?]){PWAVE}",                        # P Onset
    fr"(?P<{P_POLARITY_STR}>[cC\+dD\-\s])",                        # P Polarity
    fr"(?P<{P_WEIGHT_STR}>[0-4\s])",                               # P Weight
    fr"1",                                                         # 1
    fr"(?P<{DATE_STR}>\d{{10}})\s",                                # Date
    fr"(?P<{P_TIME_STR}>[\s\d]{{4}})",                             # P Time
    fr".{{8}}",                                                    # Unknown
    [
        fr"(((?P<{S_TIME_STR}>[\s\d]{{4}})",                       # S Time
        fr"(?P<{S_ONSET_STR}>[ei\s\?]){SWAVE}",                    # S Onset
        fr"(?P<{S_POLARITY_STR}>[cC\+dD\-\s])",                    # S Polarity
        fr"(?P<{S_WEIGHT_STR}>[0-5\s]))|\s{{8}})"                  # S Weight
    ],
    fr"\s{{22}}",                                                  # SPACE
    # Geo Zone
    fr"(?P<{GEO_ZONE_STR}>[{EMPTY_STR.join(OGS_GEO_ZONES.keys())}\s])",
    # Event Type
    fr"(?P<{EVENT_TYPE_STR}>[{EMPTY_STR.join(OGS_EVENT_TYPES.keys())}\s])",
    fr"(?P<{EVENT_LOCALIZATION_STR}>[D\s])",                       # Event Loc
    fr"\s{{5}}",                                                   # SPACE
    fr"(?P<{DURATION_STR}>[\s\d]{{5}})",                           # Duration
    fr"(?P<{EVENT_STR}>[\s\d]{{4}})",                              # Event
    fr""
]
RECORD_EXTRACTOR_DAT = re.compile(EMPTY_STR.join(
    list(flatten_list(RECORD_EXTRACTOR_DAT_LIST))))
# print(RECORD_EXTRACTOR_DAT.pattern)
EVENT_EXTRACTOR_DAT = re.compile(r"^1.*$")                         # Event
EVENT_CONTRIVER_DAT = \
    "{STATION_STR}" + ALL_WILDCHAR_STR + PWAVE + ALL_WILDCHAR_STR + \
    "{P_WEIGHT_STR}" + ALL_WILDCHAR_STR + "{DATE_STR}" + SPACE_STR + \
    "{P_TIME_STR}" + ALL_WILDCHAR_STR * 8 + "{S_TIME_STR}" + \
    ALL_WILDCHAR_STR + SWAVE + ALL_WILDCHAR_STR + "{S_WEIGHT_STR}" + \
    ALL_WILDCHAR_STR * 35 + "{EVENT_STR}"


def event_parser_dat(filename: Path, start: UTCDateTime = None,
                     end: UTCDateTime = None, level: str = WARNING_STR,
                     stations: dict[str, set[str]] = None) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
  """
  Parse the DAT file and return the DataFrame of the manual picks.

  input:
    - filename  (Path)        : The path to the DAT file
    - start     (UTCDateTime) : The start date of the picks
    - end       (UTCDateTime) : The end date of the picks
    - stations  (set[str])    : The set of stations to consider
    - level     (str)         : The level of the error message

  output:
    - pd.DataFrame : Empty DataFrame
    - pd.DataFrame : The DataFrame of the manual picks

  exceptions:
    - FileNotFoundError : If the file does not exist

  notes:
    The DAT file contains the manual picks of the earthquake events by
    providing the station, the type of the P pick, the weight of the P pick,
    the date of the pick, the time of the pick, the type of the S pick, the
    weight of the S pick, and the time of the S pick.
  """
  unble_msg = ERRORS[level][UNABLE_STR].format(
      pre=f"{filename}, ", verb="parse", type="({type})",
      post="from line: {line}")
  notbl_msg = ERRORS[level][NOTABLE_STR].format(
      pre=f"{str(filename)}, ", value="{value}", key="{key}",
      post="from line: {line}")
  assgn_msg = ERRORS[level][ASSIGN_STR].format(
      pre=f"{str(filename)}, ", value="{value}", key="{key}", post=EMPTY_STR)
  # TODO: Attemp restoration before SHUTDOWN
  if not filename.exists():
    raise FileNotFoundError(filename)
  DETECT = list()
  with open(filename, 'r') as fr:
    lines = fr.readlines()
  for line in [l.strip() for l in lines]:
    if EVENT_EXTRACTOR_DAT.match(line):
      continue
    match = RECORD_EXTRACTOR_DAT.match(line)
    if match:
      result = match.groupdict()
      if (result[EVENT_LOCALIZATION_STR] != "D" and
              OGS_EVENT_TYPES[result[EVENT_TYPE_STR]] != EVENT_LOCAL_EQ_STR):
        # print("WARNING: (DAT) Ignoring line:", line)
        continue
      # Date
      try:
        if int(result[DATE_STR][-2:]) >= 60:
          # print(notbl_msg.format(value="60", key=DATE_STR, line=line))
          result[DATE_STR] = \
              UTCDateTime.strptime(result[DATE_STR][:-2],
                                   DATETIME_FMT[:-4]) + td(hours=1)
        else:
          result[DATE_STR] = UTCDateTime.strptime(result[DATE_STR],
                                                  DATETIME_FMT[:-2])
      except ValueError as e:
        print(unble_msg.format(type=DATE_STR, line=line))
        print(e)
        continue
      # We only consider the picks from the date range (if specified)
      if start is not None and result[DATE_STR] < start:
        continue
      if end is not None and result[DATE_STR] >= end + ONE_DAY:
        break
      # Station, We only consider the picks from the stations (if specified)
      result[STATION_STR] = result[STATION_STR].strip(SPACE_STR)
      date = result[DATE_STR].strftime(YYMMDD_FMT)
      if (stations is not None and date in stations and
              result[STATION_STR] not in stations[date]):
        continue
      # P Time
      try:
        result[P_TIME_STR] = result[DATE_STR] + td(seconds=float(
            result[P_TIME_STR].replace(SPACE_STR, ZERO_STR)) / 100.)
      except ValueError as e:
        print(unble_msg.format(type=P_TIME_STR, line=line))
        print(e)
        continue
      # Event
      if result[EVENT_STR]:
        try:
          result[EVENT_STR] = \
              int(result[EVENT_STR].replace(SPACE_STR, ZERO_STR)) + \
              result[DATE_STR].year * MAX_PICKS_YEAR
        except ValueError as e:
          result[EVENT_STR] = None
          print(unble_msg.format(type=EVENT_STR, line=line))
          print(e)
      DEFAULT_VALUE = 0
      # P Weight
      try:
        if result[P_WEIGHT_STR] == SPACE_STR:
          print(notbl_msg.format(value=SPACE_STR, key=P_WEIGHT_STR, line=line))
          print(assgn_msg.format(value=DEFAULT_VALUE, key=P_WEIGHT_STR))
          result[P_WEIGHT_STR] = DEFAULT_VALUE
        else:
          result[P_WEIGHT_STR] = int(result[P_WEIGHT_STR])
      except ValueError as e:
        print(unble_msg.format(type=P_WEIGHT_STR, line=line))
        print(e)
        continue
      DETECT.append([result[EVENT_STR], result[P_TIME_STR],
                     result[P_WEIGHT_STR], PWAVE, None, result[STATION_STR]])
      # S Type
      if result[S_TIME_STR]:
        # S Weight
        try:
          if result[S_WEIGHT_STR] == SPACE_STR:
            print(notbl_msg.format(value=SPACE_STR, key=S_WEIGHT_STR,
                                   line=line))
            print(assgn_msg.format(value=DEFAULT_VALUE, key=S_WEIGHT_STR))
            result[S_WEIGHT_STR] = DEFAULT_VALUE
          else:
            result[S_WEIGHT_STR] = int(result[S_WEIGHT_STR])
        except ValueError as e:
          print(unble_msg.format(type=S_WEIGHT_STR, line=line))
          print(e)
          continue
        # S Time
        try:
          result[S_TIME_STR] = result[DATE_STR] + td(seconds=float(
              result[S_TIME_STR].replace(SPACE_STR, ZERO_STR)) / 100.)
        except ValueError as e:
          print(unble_msg.format(type=S_TIME_STR, line=line))
          print(e)
          continue
        DETECT.append([result[EVENT_STR], result[S_TIME_STR],
                       result[S_WEIGHT_STR], SWAVE, None, result[STATION_STR]])
      # TODO: Add debug method
      # if verbose:
      #   print(line)
      #   with open()
      #   print(EVENT_CONTRIVER_DAT.format(
      #       STATION_STR=result[STATION_STR].lfill(
      #           4, SPACE_STR),
      #                                    P_WEIGHT_STR=result[STATION_STR],
      #                                    DATE_STR=
      #                                    "{P_TIME_STR}"
      #                                    "{S_TIME_STR}"
      #                                    "{S_WEIGHT_STR}"
      #                                     "{EVENT_STR}"))
      continue
    if line == EMPTY_STR:
      continue
    RECORD_EXTRACTOR_DAT_DEBUG = list(reversed(list(it.accumulate(
        RECORD_EXTRACTOR_DAT_LIST[:-1],
        lambda x, y: x + (y if isinstance(y, str)
                          else EMPTY_STR.join(list(flatten_list(y))))))))
    bug = STATION_STR
    for i, extractor in enumerate(RECORD_EXTRACTOR_DAT_DEBUG):
      match_extractor = re.match(extractor, line)
      if match_extractor:
        group = re.compile(r"\(\?P<(\w+)>[\[\]\w\d\{\}\-\\\?\+]+\)(\w)*")
        match_group = group.findall(RECORD_EXTRACTOR_DAT_DEBUG[i - 1])
        match_compare = group.findall(extractor)
        bug = match_group[-1][match_group[-1][1] != match_compare[-1][1]]
        break
    print(unble_msg.format(type=bug, line=line))
  return (pd.DataFrame(columns=HEADER_SRC),
          pd.DataFrame(DETECT, columns=HEADER_MANL).astype({ID_STR: int},
                                                           errors='ignore'))


RECORD_EXTRACTOR_PUN = re.compile(
    fr"^1(?P<{DATE_STR}>\d{{6}}[\s\d]\d[\s\d]\d)\s"                 # Date
    fr"(?P<{SECONDS_STR}>[\s\d]\d\.\d{{2}})\s"                      # Seconds
    fr"(?P<{LATITUDE_STR}>[\s\d]\d-[\s\d]\d\.\d{{2}})\s{{2}}"       # Latitude
    fr"(?P<{LONGITUDE_STR}>[\s\d]\d-[\s\d]\d\.\d{{2}})\s{{2}}"      # Longitude
    fr"(?P<{LOCAL_DEPTH_STR}>[\s\d]\d\.\d{{2}})\s{{2}}"             # Depth
    fr"(?P<{MAGNITUDE_STR}>[\-\s](\d\.\d{{2}}|\s{{4}}))\s"          # Magnitude
    fr"(?P<{NO_STR}>[\s\d]\d)\s"                                    # NO
    fr"(?P<{GAP_STR}>[\s\d]{{3}})"                                  # GAP
    fr"(?P<{DMIN_STR}>[\s\d]{{2}}\d\.\d)"                           # DMIN
    fr"(?P<{RMS_STR}>[\s\d]\d\.\d{{2}})"                            # RMS
    fr"(?P<{ERH_STR}>([\s\d]{{2}}\d\.\d|\s{{5}}))"                  # ERH
    fr"(?P<{ERZ_STR}>([\s\d]{{2}}\d\.\d|\s{{5}}))\s"                # ERZ
    fr"(?P<{QM_STR}>[A-Z][0-9])")                                   # QM
RECORD_CONTRIVER_PUN = \
    "1{DATE_STR}" + SPACE_STR + "{SECONDS_STR}" + SPACE_STR + \
    "{LATITUDE_STR}" + SPACE_STR * 2 + "{LONGITUDE_STR}" + SPACE_STR * 2 + \
    "{LOCAL_DEPTH_STR}" + SPACE_STR * 3 + "{MAGNITUDE_STR}" + SPACE_STR + \
    "{NO_STR}" + SPACE_STR + "{GAP_STR}" + SPACE_STR + \
    "{DMIN_STR}" + SPACE_STR + "{RMS_STR}" + SPACE_STR * 2 + \
    "{ERH_STR}" + SPACE_STR * 2 + "{ERZ_STR}" + SPACE_STR + "{QM_STR}"


def event_parser_pun(filename: Path, start: UTCDateTime = None,
                     end: UTCDateTime = None, level: str = WARNING_STR) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
  """
  Parse the PUN file and return the DataFrame of the source.

  input:
    - filename  (Path)        : The path to the DAT file
    - start     (UTCDateTime) : The start date of the picks
    - end       (UTCDateTime) : The end date of the picks
    - stations  (set[str])    : The set of stations to consider
    - level     (str)         : The level of the error message

  output:
    - pd.DataFrame : The DataFrame of the source
    - pd.DataFrame : Empty DataFrame

  exceptions:
    - FileNotFoundError : If the file does not exist

  notes:
    The PUN file contains the source metadata of the earthquake events by
    providing the date and time of the event, the latitude and longitude of the
    event, the depth of the event, the magnitude of the event, the number of
    the event, the gap of the event, the DMIN of the event, the RMS of the
    event, the ERH of the event, the ERZ of the event, and the QM of the event.
  """
  unble_msg: str = ERRORS[level][UNABLE_STR].format(
      pre="PUN, ", verb="parse", type="{type}", post="from line: {line}")
  if not filename.exists():
    raise FileNotFoundError(filename)
  SOURCE = list()
  event: int = 0
  with open(filename, 'r') as fr:
    lines = fr.readlines()[1:]
  for line in [l.strip() for l in lines]:
    match = RECORD_EXTRACTOR_PUN.match(line)
    if match:
      result: dict[str] = match.groupdict()
      result[SECONDS_STR] = td(seconds=float(result[SECONDS_STR]))
      result[DATE_STR] = UTCDateTime.strptime(
          result[DATE_STR].replace(SPACE_STR, ZERO_STR), DATETIME_FMT[:-2]
      ) + result[SECONDS_STR]
      # We only consider the picks from the date range (if specified)
      if start is not None and result[DATE_STR] < start:
        continue
      if end is not None and result[DATE_STR] >= end + ONE_DAY:
        break
      result[LATITUDE_STR] = result[LATITUDE_STR].replace(SPACE_STR,
                                                          ZERO_STR)\
          if result[LATITUDE_STR] else NONE_STR
      if result[LATITUDE_STR] != NONE_STR:
        splt = result[LATITUDE_STR].split(DASH_STR)
        result[LATITUDE_STR] = float(splt[0]) + float(splt[1]) / 60.
      result[LONGITUDE_STR] = result[LONGITUDE_STR].replace(SPACE_STR,
                                                            ZERO_STR) \
          if result[LONGITUDE_STR] else NONE_STR
      if result[LONGITUDE_STR] != NONE_STR:
        splt = result[LONGITUDE_STR].split(DASH_STR)
        result[LONGITUDE_STR] = float(splt[0]) + float(splt[1]) / 60.
      result[LOCAL_DEPTH_STR] = float(result[LOCAL_DEPTH_STR]) \
          if result[LOCAL_DEPTH_STR] else NONE_STR
      result[MAGNITUDE_STR] = float(result[MAGNITUDE_STR].replace(SPACE_STR,
                                                                  ZERO_STR))
      result[NO_STR] = int(result[NO_STR].replace(SPACE_STR, ZERO_STR)) \
          if result[NO_STR] else NONE_STR
      result[GAP_STR] = int(result[GAP_STR].replace(SPACE_STR, ZERO_STR))
      result[DMIN_STR] = float(result[DMIN_STR].replace(SPACE_STR, ZERO_STR))
      result[RMS_STR] = float(result[RMS_STR].replace(SPACE_STR, ZERO_STR))
      result[ERH_STR] = float(result[ERH_STR].replace(SPACE_STR, ZERO_STR))
      result[ERZ_STR] = float(result[ERZ_STR].replace(SPACE_STR, ZERO_STR))
      SOURCE.append([None, result[DATE_STR], result[LATITUDE_STR],
                     result[LONGITUDE_STR], result[LOCAL_DEPTH_STR],
                     result[MAGNITUDE_STR], result[NO_STR], result[GAP_STR],
                     result[DMIN_STR], result[RMS_STR], result[ERH_STR],
                     result[ERZ_STR], result[QM_STR], None])
      event += 1
      continue
    print(unble_msg.format(type=EMPTY_STR, line=line))
  return (pd.DataFrame(SOURCE, columns=HEADER_SRC),
          pd.DataFrame(columns=HEADER_MANL))


RECORD_EXTRACTOR_HPC = re.compile(
    fr"^(?P<{STATION_STR}>[A-Z0-9\s]{{4}})"                         # Station
    fr"(?P<{P_TYPE_STR}>[ei]{PWAVE}[cC\+dD\-\s])"                   # P Type
    fr"(?P<{P_WEIGHT_STR}>[0-4])1"                                  # P Weight
    fr"(?P<{DATE_STR}>\d{{10}})\s"                                  # Date
    fr"(?P<{P_TIME_STR}>[\s\d]{{4}})\s+"                            # P Time
    fr"(?P<{S_TIME_STR}>\d{{4}}|\d{{3}})"                           # S Time
    fr"(?P<{S_TYPE_STR}>[ei]{SWAVE}\s)"                             # S Type
    fr"(?P<{S_WEIGHT_STR}>[0-4])")                                  # S Weight
# print(RECORD_EXTRACTOR_HPC.pattern)


def event_parser_hpc(filename: Path, start: UTCDateTime = None,
                     end: UTCDateTime = None,
                     stations: dict[str, set[str]] = None) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
  if not filename.exists():
    raise FileNotFoundError(filename)
  DETECT = list()
  event = 0
  with open(filename, 'r') as fr:
    lines = fr.readlines()
  for line in [l.strip() for l in lines]:
    match = RECORD_EXTRACTOR_HPC.match(line)
    if match:
      result = match.groupdict()
      result[STATION_STR] = result[STATION_STR].strip(SPACE_STR)
      result[DATE_STR] = UTCDateTime.strptime(result[DATE_STR],
                                              DATETIME_FMT[:-2])
      date = result[DATE_STR].strftime(YYMMDD_FMT)
      if (stations is not None and date in stations and
              result[STATION_STR] not in stations[date]):
        continue
      if start is not None and result[DATE_STR] < start:
        continue
      if end is not None and result[DATE_STR] >= end + ONE_DAY:
        break
      result[P_TIME_STR] = result[DATE_STR] + td(seconds=float(
          result[P_TIME_STR][:2].replace(SPACE_STR, ZERO_STR) + PERIOD_STR +
          result[P_TIME_STR][2:].replace(SPACE_STR, ZERO_STR)))
      result[S_TIME_STR] = result[DATE_STR] + td(seconds=float(
          result[S_TIME_STR].replace(SPACE_STR, ZERO_STR)))
      result[P_WEIGHT_STR] = int(result[P_WEIGHT_STR])
      result[S_WEIGHT_STR] = int(result[S_WEIGHT_STR])
      DETECT.append([event, result[P_TIME_STR], result[P_WEIGHT_STR], PWAVE,
                     None, result[STATION_STR]])
      DETECT.append([event, result[S_TIME_STR], result[S_WEIGHT_STR], SWAVE,
                     None, result[STATION_STR]])
      event += 1
      continue
    print("WARNING: (HPC) Unable to parse line:", line)
  return (pd.DataFrame(columns=HEADER_SRC),
          pd.DataFrame(DETECT, columns=HEADER_MANL))


RECORD_EXTRACTOR_HPL = re.compile(
    fr"^(?P<{EVENT_STR}>[\d\s]{{6}})\s"                            # Event
    fr"(?P<{STATION_STR}>[A-Z0-9\s]{{4}})\s"                       # Station
    fr"(([\d\s]{{3}}\.\d)|\s{{5}})\s"                              # Unknown
    fr"([\d\s]{{3}})\s"                                            # Unknown
    fr"([\d\s]{{3}})\s"                                            # Unknown
    fr"(?P<{P_ONSET_STR}>[ei?\s]){PWAVE}"                          # P Onset
    fr"(?P<{P_POLARITY_STR}>[cC\+dD\-\s])"                         # P Polarity
    fr"(?P<{P_WEIGHT_STR}>[0-4])\s"                                # P Weight
    # P Time [hhmm]
    fr"(?P<{P_TIME_STR}>[\s\d]{{4}})\s"
    # Seconds [ss.ss]
    fr"(?P<{SECONDS_STR}>([\s\d]\d\.\d{{2}})|[\s\*]{{5}})"
    fr"(([\s\d]{{2}}\d\.\d{{2}})|[\s\*]{{6}})\s"                   # Unknown
    fr"(([\s\d]\d\.\d{{2}})|\s{{5}})\s"                            # Unknown
    fr"(([\s\d\-]\d\.\d{{2}})|\s{{5}})"                            # Unknown
    fr"(([\s\d\-]{{2}}\d\.\d{{2}})|[\s\*]{{6}})\s"                 # Unknown
    fr"(([\s\-]\d\.\d{{2}})|[\s\*]{{5}})\s"                        # Unknown
    fr"([\d\s]{{3}})\s"                                            # Unknown
    fr"([\d\s]{{2}})\s"                                            # Unknown
    fr"(([\d\s]\d\.\d{{2}})|\s{{5}})\s"                            # Unknown
    fr"(\d|\s)\s"                                                  # Unknown
    fr"(.{{4}})\s"                                                 # Unknown
    # Geo Zone
    fr"(?P<{GEO_ZONE_STR}>[{EMPTY_STR.join(OGS_GEO_ZONES.keys())}\s])"
    # Event Type
    fr"(?P<{EVENT_TYPE_STR}>[{EMPTY_STR.join(OGS_EVENT_TYPES.keys())}\s])"
    fr"(?P<{EVENT_LOCALIZATION_STR}>[D\s])"                        # Event Loc
    fr"([\d\s\*]{{4}})"                                            # Unknown
    fr"([\s\-](\d\.\d)|\s{{4}})[\*\s]\s"                           # Unknown
    fr"((?P<{S_ONSET_STR}>[ei\s]){SWAVE}\s"                        # S Type
    fr"(?P<{S_WEIGHT_STR}>[0-4])?\s"                               # S Weight
    fr"(?P<{S_TIME_STR}>[\s\d]\d\.\d{{2}}))?"                      # S Time
)
# print(RECORD_EXTRACTOR_HPL.pattern)
EVENT_EXTRACTOR_HPL = re.compile(
    fr"^(?P<{EVENT_STR}>[\d\s]{{6}})1"                          # Event
    # Date [yymmdd hhmm]
    fr"(?P<{DATE_STR}>\d{{6}}\s[\s\d]{{4}})\s"
    # Seconds [ss.ss]
    fr"((?P<{SECONDS_STR}>[\s\d]{{2}}\.\d{{2}})|\s{{5}})\s"
    fr"((?P<{LATITUDE_STR}>[\s\d]{{2}}-[\s\d]{{2}}\.\d{{2}})|"  # Latitude
    fr"\s{{8}})\s{{2}}"
    fr"((?P<{LONGITUDE_STR}>[\s\d]{{2}}-[\s\d]{{2}}\.\d{{2}})|"  # Longitude
    fr"\s{{8}})\s{{2}}"
    fr"((?P<{LOCAL_DEPTH_STR}>[\s\d]{{2}}\.\d{{2}})|"           # Depth
    fr"\s{{5}})\s{{2}}"
    fr"((?P<{MAGNITUDE_STR}>[\s\-]\d\.\d{{2}})|\s{{5}})\s"      # Magnitude
    fr"((?P<{NO_STR}>[\s\d]\d)|\s{{2}})"                        # NO
    fr"((?P<{DMIN_STR}>[\s\d]{{2}}\d)|\s{{3}})\s"               # DMIN
    fr"(?P<{GAP_STR}>[\s\d]{{3}})\s"                            # GAP
    fr"([\d\s])"                                                # Unknown
    fr"((?P<{RMS_STR}>[\s\d]{{2}}\.\d{{2}})|\s{{5}})"           # RMS
    fr"((?P<{ERH_STR}>[\s\d]{{3}}\.\d)|[\s\*]{{5}})"            # ERH
    fr"((?P<{ERZ_STR}>[\s\d]{{3}}\.\d)|[\s\*]{{5}})\s"          # ERZ
    fr"((?P<{QM_STR}>[A-D])|\s)\s"                              # QM
    fr"(([A-D]/[A-D])|\s{{3}})"                                 # Unknown
    fr"(([\s\d]{{2}}\.\d{{2}})|\s{{5}})\s"                      # Unknown
    fr"([\s\d]{{2}})\s"                                         # Unknown
    fr"([\s\d]{{2}})"                                           # Unknown
    fr"(([\-\s]\d\.\d{{2}})|\s{{5}})"                           # Unknown
    fr"(([\s\d]\d\.\d{{2}})|\s{{5}})\s"                         # Unknown
    fr"([\s\d]{{2}})\s"                                         # Unknown
    fr"(([\s\d]{{2}}\.\d)|\s{{4}})\s"                           # Unknown
    fr"(([\s\d]{{2}}\.\d)|\s{{4}})\s"                           # Unknown
    fr"([\s\d]{{2}})\s"                                         # Unknown
    fr"(([\s\d\-]{{2}}\.\d)|\s{{4}})\s"                         # Unknown
    fr"(([\s\d]{{2}}\.\d)|\s{{4}})"                             # Unknown
    fr"([\s\d]{{2}})"                                           # Unknown
    fr"(([\s\d]{{3}}\.\d)|\s{{5}})\s"                           # Unknown
    fr"(([\s\d]{{2}}\.\d)|\s{{4}})\s{{9}}"                      # Unknown
    fr"(?P<{NOTES_STR}>[\s\d]\d)"                               # Notes
)
LOCATION_EXTRACTOR_HPL = re.compile(
    fr"^\^(?P<{LOC_NAME_STR}>[A-Z\s\.']+(\s\([A-Z\-\s]+\))?)"
)
# print(LOCATION_EXTRACTOR_HPL.pattern)
NOTES_EXTRACTOR_HPL = re.compile(fr"^\*\s+(?P<{NOTES_STR}>.*)")


def event_parser_hpl(filename: Path, start: UTCDateTime = None,
                     end: UTCDateTime = None, level: str = WARNING_STR,
                     stations: dict[str, set[str]] = None) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
  """
  Parse the HPL file and return the DataFrame of the source and the manual
  picks.

  input:
    - filename  (Path)        : The path to the DAT file
    - start     (UTCDateTime) : The start date of the picks
    - end       (UTCDateTime) : The end date of the picks
    - stations  (set[str])    : The set of stations to consider
    - level     (str)         : The level of the error message

  output:
    - pd.DataFrame : The DataFrame of the source
    - pd.DataFrame : The DataFrame of the manual picks

  exceptions:
    - FileNotFoundError : If the file does not exist

  notes:
    The HPL file contains the source metadata of the earthquake events by
    providing the event number, the date and time of the event, the latitude
    and longitude of the event, the depth of the event, the magnitude of the
    event, the NO of the event, the GAP of the event, the DMIN of the event,
    the RMS of the event, the ERH of the event, the ERZ of the event, the QM of
    the event, the notes of the event, the station, the type of the pick, the
    weight of the pick, the time of the pick, the seconds of the pick, the type
    of the S pick, the weight of the S pick, and the time of the S pick.
  """
  unble_msg: str = ERRORS[level][UNABLE_STR].format(
      pre=f"{filename}, ", verb="parse", type="{type}",
      post="from line: {line}")
  notbl_msg: str = ERRORS[level][NOTABLE_STR].format(
      pre=f"{filename}, ", value="{value}", key="{key}",
      post="from line: {line}")
  if not filename.exists():
    raise FileNotFoundError(filename)
  SOURCE = list()
  DETECT = list()
  event_notes: str = ""
  event_detect: int = 0
  event_spacetime = (UTCDateTime(0), 0, 0, 0)
  with open(filename, 'r') as fr:
    lines = fr.readlines()
  for line in [l.strip("\n") for l in lines]:
    if event_detect > 0:
      event_detect -= 1
      match = RECORD_EXTRACTOR_HPL.match(line)
      if match:
        result = match.groupdict()
        if (result[EVENT_LOCALIZATION_STR] != "D" and
                result[EVENT_TYPE_STR] != "L"):
          # print("WARNING: (HPL) Ignoring line:", line)
          continue
        date = UTCDateTime(event_spacetime[0].date)
        result[STATION_STR] = result[STATION_STR].strip(SPACE_STR)
        if (stations is not None and date.strftime(YYMMDD_FMT) in stations and
                result[STATION_STR] not in stations[date.strftime(YYMMDD_FMT)]):
          continue
        # Event
        if result[EVENT_STR]:
          try:
            result[EVENT_STR] = \
                int(result[EVENT_STR].replace(SPACE_STR, ZERO_STR)) + \
                date.year * MAX_PICKS_YEAR
          except ValueError as e:
            result[EVENT_STR] = None
            print(unble_msg.format(type=EVENT_STR, line=line))
            print(e)
        result[P_WEIGHT_STR] = int(result[P_WEIGHT_STR])
        if result[SECONDS_STR] == SPACE_STR*5:
          result[SECONDS_STR] = td(0)
          msg = notbl_msg.format(value=SPACE_STR*5, key=SECONDS_STR,
                                 line=line.strip())
          if level == WARNING_STR:
            print(msg)
          else:
            raise ValueError(msg)
        else:
          result[SECONDS_STR] = td(seconds=float(result[SECONDS_STR]))
        result[P_TIME_STR] = result[P_TIME_STR].replace(SPACE_STR, ZERO_STR)
        min = td(minutes=int(result[P_TIME_STR][2:]))
        if min >= td(minutes=60):
          print(notbl_msg.format(value=">= 60", key=P_TIME_STR,
                                 line=line.strip()))
        hrs = td(hours=int(result[P_TIME_STR][:2]))
        if hrs >= td(hours=24):
          print(notbl_msg.format(value=">= 24", key=P_TIME_STR,
                                 line=line.strip()))
        result[P_TIME_STR] = date + hrs + min
        DETECT.append([result[EVENT_STR],
                       result[P_TIME_STR] + result[SECONDS_STR],
                       result[P_WEIGHT_STR], PWAVE, None, result[STATION_STR]])
        if result[S_TIME_STR]:
          result[S_WEIGHT_STR] = int(result[S_WEIGHT_STR])
          result[S_TIME_STR] = td(seconds=float(result[S_TIME_STR]))
          DETECT.append([result[EVENT_STR],
                         result[P_TIME_STR] + result[S_TIME_STR],
                         result[S_WEIGHT_STR], SWAVE, None,
                         result[STATION_STR]])
        continue
    else:
      match = EVENT_EXTRACTOR_HPL.match(line)
      if match:
        result = match.groupdict()
        result[SECONDS_STR] = td(seconds=float(result[SECONDS_STR])) \
            if result[SECONDS_STR] else td(0)
        result[DATE_STR] = UTCDateTime.strptime(
            result[DATE_STR].replace(SPACE_STR, ZERO_STR), "%y%m%d0%H%M") + \
            result[SECONDS_STR]
        # Event
        if result[EVENT_STR]:
          try:
            result[EVENT_STR] = \
                int(result[EVENT_STR].replace(SPACE_STR, ZERO_STR)) + \
                result[DATE_STR].year * MAX_PICKS_YEAR
          except ValueError as e:
            result[EVENT_STR] = None
            print(unble_msg.format(type=EVENT_STR, line=line))
            print(e)
        if start is not None and result[DATE_STR] < start:
          event_detect = -1
          continue
        if end is not None and result[DATE_STR] >= end + ONE_DAY:
          break
        result[LATITUDE_STR] = result[LATITUDE_STR].replace(SPACE_STR,
                                                            ZERO_STR)\
            if result[LATITUDE_STR] else NONE_STR
        if result[LATITUDE_STR] != NONE_STR:
          splt = result[LATITUDE_STR].split(DASH_STR)
          result[LATITUDE_STR] = float("{:.4f}".format(
              float(splt[0]) + float(splt[1]) / 60.))
        result[LONGITUDE_STR] = result[LONGITUDE_STR].replace(SPACE_STR,
                                                              ZERO_STR) \
            if result[LONGITUDE_STR] else NONE_STR
        if result[LONGITUDE_STR] != NONE_STR:
          splt = result[LONGITUDE_STR].split(DASH_STR)
          result[LONGITUDE_STR] = float("{:.4f}".format(
              float(splt[0]) + float(splt[1]) / 60.))
        result[LOCAL_DEPTH_STR] = float(result[LOCAL_DEPTH_STR]) \
            if result[LOCAL_DEPTH_STR] else NONE_STR
        event_spacetime = (result[DATE_STR], result[LATITUDE_STR],
                           result[LONGITUDE_STR], result[LOCAL_DEPTH_STR])
        result[MAGNITUDE_STR] = float(result[MAGNITUDE_STR]) \
            if result[MAGNITUDE_STR] else float("NaN")
        result[NO_STR] = int(result[NO_STR].replace(SPACE_STR, ZERO_STR)) \
            if result[NO_STR] else NONE_STR
        result[GAP_STR] = int(result[GAP_STR].replace(SPACE_STR, ZERO_STR)) \
            if result[GAP_STR] else NONE_STR
        result[DMIN_STR] = float(result[DMIN_STR].replace(SPACE_STR, ZERO_STR)) \
            if result[DMIN_STR] else NONE_STR
        result[RMS_STR] = float(result[RMS_STR].replace(SPACE_STR, ZERO_STR)) \
            if result[RMS_STR] else NONE_STR
        result[ERH_STR] = float(result[ERH_STR].replace(SPACE_STR, ZERO_STR)) \
            if result[ERH_STR] else float("NaN")
        result[ERZ_STR] = float(result[ERZ_STR].replace(SPACE_STR, ZERO_STR)) \
            if result[ERZ_STR] else float("NaN")
        result[QM_STR] = result[QM_STR].strip(SPACE_STR) \
            if result[QM_STR] else NONE_STR
        event_detect = int(result[NOTES_STR])
        SOURCE.append([result[EVENT_STR], *event_spacetime,
                       result[MAGNITUDE_STR], result[NO_STR], result[GAP_STR],
                       result[DMIN_STR], result[RMS_STR], result[ERH_STR],
                       result[ERZ_STR], result[QM_STR], None])
        continue
      match = LOCATION_EXTRACTOR_HPL.match(line)
      if match:
        result = match.groupdict()
        continue
      if event_detect == 0:
        match = NOTES_EXTRACTOR_HPL.match(line)
        if match:
          result = match.groupdict()
          event_notes = result[NOTES_STR].rstrip(SPACE_STR)
          SOURCE[-1][-1] = event_notes
          continue
      if re.match(r"^\s*$", line):
        continue
    if event_detect >= 0:
      print(unble_msg.format(type=EMPTY_STR, line=line))
  SOURCE = pd.DataFrame(SOURCE, columns=HEADER_SRC)
  SOURCE = SOURCE[SOURCE[LATITUDE_STR] != NONE_STR]
  DETECT = pd.DataFrame(DETECT, columns=HEADER_MANL)
  DETECT = DETECT[DETECT[ID_STR].isin(SOURCE[ID_STR])]
  return SOURCE, DETECT

# TODO: Implement ObsPy Catalog for OGS files
# TODO: Implement ObsPy Event for OGS files
# TODO: Implement ObsPy Origin for OGS files


def event_parser_qml(filename: Path, start: UTCDateTime = None,
                     end: UTCDateTime = None,
                     stations: dict[str, set[str]] = None) -> pd.DataFrame:
  if not filename.exists():
    raise FileNotFoundError(filename)
  SOURCE: list[list] = list()
  DETECT: list[list] = list()
  for event in obspy_read_events(filename):
    if len(event.origins):
      e_origin = event.origins[0]
    else:
      continue
    e_time = e_origin.time
    if start is not None and e_time < start:
      continue
    if end is not None and e_time >= end + ONE_DAY:
      break
    DETECT.extend([[None, pick.time, pick.time_errors.uncertainty,
                    pick.phase_hint, pick.waveform_id.network_code,
                    pick.waveform_id.station_code] for pick in event.picks])
    SOURCE.append([None, e_time, e_origin.latitude, e_origin.longitude,
                   e_origin.depth, None, len(event.picks), *([None] * 6),
                   event.event_type])
  SOURCE = pd.DataFrame(SOURCE, columns=HEADER_SRC)
  DETECT = pd.DataFrame(DETECT, columns=HEADER_MANL)
  return SOURCE, DETECT


STATION_EXTRACTOR_MOD = re.compile(
    fr"^(?P<{STATION_STR}>[A-Z0-9\s]{{4}})"                         # Station
    fr"(?P<{LONGITUDE_STR}>[\s\d]{{4}}\.\d{{2}}[NS])\s"             # Longitude
    fr"(?P<{LATITUDE_STR}>[\s\d]{{4}}\.\d{{2}}[EW])"                # Latitude
    fr"(?P<{LOCAL_DEPTH_STR}>[\s\d]{{4}}).*"                        # Depth
    fr"[\.\s](?P<{TIMESTAMP_STR}>[A-Z\d]+)?\s*$"                    # Unknown
)


def event_parser_mod(filename: Path,
                     stations: dict[str, set[str]] = None) -> dict[str, str]:
  if not filename.exists():
    raise FileNotFoundError(filename)
  DATA = list()
  with open(filename, 'r') as fr:
    lines = fr.readlines()
  for line in [l.strip() for l in lines]:
    match = STATION_EXTRACTOR_MOD.match(line)
    if match:
      result = match.groupdict()
      result[STATION_STR] = result[STATION_STR].strip(SPACE_STR)
      # TODO: Review MOD file
      date = result[DATE_STR].strftime(YYMMDD_FMT)
      if (stations is not None and date in stations and
              result[STATION_STR] not in stations[date]):
        continue
      if len(result[TIMESTAMP_STR]) <= 4:
        result[TIMESTAMP_STR] = result[STATION_STR]
      result[LONGITUDE_STR] = \
          result[LONGITUDE_STR][:2] + DASH_STR + \
          result[LONGITUDE_STR][2:-1].replace(SPACE_STR, ZERO_STR)
      result[LATITUDE_STR] = \
          result[LATITUDE_STR][:2] + DASH_STR + \
          result[LATITUDE_STR][2:-1].replace(SPACE_STR, ZERO_STR)
      result[LOCAL_DEPTH_STR] = float(result[LOCAL_DEPTH_STR]) \
          if result[LOCAL_DEPTH_STR] else NONE_STR
      DATA.append([
          result[STATION_STR], result[LONGITUDE_STR], result[LATITUDE_STR],
          result[LOCAL_DEPTH_STR], None])
      continue
    print("WARNING: (MOD) Unable to parse line:", line)


def event_parser_(filename: Path, start: UTCDateTime = None,
                  end: UTCDateTime = None,
                  stations: dict[str, set[str]] = None) -> pd.DataFrame:
  if not filename.exists():
    raise FileNotFoundError(filename)
  sfx = filename.suffix
  if sfx == BLT_EXT:
    pass
  elif sfx == DAT_EXT:
    return event_parser_dat(filename, start, end,
                            stations=stations)
  elif sfx == HPC_EXT:
    pass  # return event_parser_hpc(filename, start, end,
    #                    stations=stations)
  elif sfx == HPL_EXT:
    return event_parser_hpl(filename, start, end,
                            stations=stations)
  elif sfx == MOD_EXT:
    pass  # return event_parser_mod(filename, start, end,
    #                    stations=stations)
  elif sfx == PRT_EXT:
    pass
  elif sfx == PUN_EXT:
    pass  # return event_parser_pun(filename, start, end)
  elif sfx == QML_EXT:
    pass  # return event_parser_qml(filename, start, end,
    #                         stations=stations)
  print(ValueError(f"WARNING: Unknown file extension: {sfx}"))
  return pd.DataFrame(columns=HEADER_SRC), pd.DataFrame(columns=HEADER_MANL)


def event_parser(filename: Path, start: UTCDateTime = None,
                 end: UTCDateTime = None,
                 stations: dict[str, set[str]] = None) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
  if not filename.exists():
    raise FileNotFoundError(filename)
  SOURCE = pd.DataFrame(columns=HEADER_SRC)
  DETECT = pd.DataFrame(columns=HEADER_MANL)
  if filename.is_dir():
    def process_file(file):
      return file.suffix, (event_parser(file, start, end, stations))
    with ThreadPoolExecutor() as executor:
      results = list(executor.map(process_file, filename.iterdir()))
    FIND_SRC = [TIMESTAMP_STR, LONGITUDE_STR, LATITUDE_STR, LOCAL_DEPTH_STR,
                MAGNITUDE_STR]
    FIND_DTC = [TIMESTAMP_STR, PROBABILITY_STR, PHASE_STR, STATION_STR]
    for sfx, (source, detect) in results:
      if sfx == PUN_EXT:
        SOURCE = event_merger_l(SOURCE, source, FIND_SRC)
      elif sfx == DAT_EXT:
        DETECT = event_merger_l(DETECT, detect, FIND_DTC)
      elif sfx == HPL_EXT:
        SOURCE = event_merger_l(SOURCE, source, FIND_SRC)
        DETECT = event_merger_l(DETECT, detect, FIND_DTC)
  else:
    SOURCE, DETECT = event_parser_(filename, start, end, stations)
    # try:
    # except Exception as e:
    #  print(f"WARNING: Unable to parse file: {filename}")
    #  print(e)
  SOURCE = SOURCE.astype({ID_STR: int}, errors='ignore')
  DETECT = DETECT.astype({ID_STR: int}, errors='ignore')
  return SOURCE, DETECT

import re
import pandas as pd
from pathlib import Path
from numpy import nan as NaN
from obspy import UTCDateTime
from numpy import isnan as isNaN
from datetime import timedelta as td
from obspy.core.event import read_events as obspy_read_events
from concurrent.futures import ThreadPoolExecutor

from constants import *
from errors import ERRORS

DEBUG = False

def event_merger_l(NEW : pd.DataFrame, OLD : pd.DataFrame, on : list) \
      -> pd.DataFrame:
  if NEW.empty: return OLD
  if OLD.empty: return NEW
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
        row[col] = OLD.loc[idx_r][col] if OLD.loc[idx_r][col] else row[col]
      NEW.loc[idx_l] = row
      idx_r += 1
  return NEW

# TODO: Parse HPC and QML files

# TODO: Implement polarity
RECORD_EXTRACTOR_DAT = re.compile(
  fr"^(?P<{STATION_STR}>[A-Z0-9\s]{{4}})"                         # Station
  fr"(?P<{P_TYPE_STR}>[aei1\s\?][Pp][cC\+0-4dD\-Up\s])"           # P Type
  fr"(?P<{P_WEIGHT_STR}>[0-4\s])"                                 # P Weight
  fr"[1-4\s]"                                                     # Unknown
  fr"(?P<{DATE_STR}>\d{{10}})\s"                                  # Date
  fr"(?P<{P_TIME_STR}>[\s\d]{{4}})"                               # P Time
  fr".{{8}}"                                                      # Unknown
  fr"(((?P<{S_TIME_STR}>[\s\d]{{4}})"                             # S Time
    fr"(?P<{S_TYPE_STR}>[eirsw\?13468\s][Ss][cC\+0-4dD\-Ue\?\s])" # S Type
    fr"(?P<{S_WEIGHT_STR}>[0-5\s]))|\s{{8}})"                     # S Weight
  fr"(.{{35}}"                                                    # Unknown
   fr"(?P<{EVENT_STR}>[\s\d]{{4}}))*")                            # Event
#print(RECORD_EXTRACTOR_DAT.pattern)
EVENT_EXTRACTOR_DAT = re.compile(r"^1.*$")                        # Event
EVENT_CONTRIVER_DAT = \
  "{STATION_STR}" + ALL_WILDCHAR_STR + PWAVE + ALL_WILDCHAR_STR + \
  "{P_WEIGHT_STR}" + ALL_WILDCHAR_STR + "{DATE_STR}" + SPACE_STR + \
  "{P_TIME_STR}" + ALL_WILDCHAR_STR * 8 + "{S_TIME_STR}" + ALL_WILDCHAR_STR + \
  SWAVE + ALL_WILDCHAR_STR + "{S_WEIGHT_STR}" + ALL_WILDCHAR_STR * 35 + \
  "{EVENT_STR}"
def event_parser_dat(filename : Path, start : UTCDateTime = None,
                     end : UTCDateTime = None, level : str = WARNING_STR,
                     stations : dict[str, set[str]] = None) \
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
  unble_msg : str = ERRORS[level][UNABLE_STR].format(pre="DAT, ",
    verb="parse", type="{type}", post="from line: {line}")
  # TODO: Attemp restoration before SHUTDOWN
  if not filename.exists(): raise FileNotFoundError(filename)
  DATA = list()
  with open(filename, 'r') as fr: lines = fr.readlines()
  for line in [l.strip() for l in lines]:
    if EVENT_EXTRACTOR_DAT.match(line): continue
    match = RECORD_EXTRACTOR_DAT.match(line)
    if match:
      result: dict[str] = match.groupdict()
      # Date
      try:
        if result[DATE_STR][-2:] == "60":
          # TODO: Warn about the 60 minutes
          result[DATE_STR] = \
            UTCDateTime.strptime(result[DATE_STR][:-2], DATETIME_FMT[:-4]) + \
            td(hours=1)
        else:
          result[DATE_STR] = UTCDateTime.strptime(result[DATE_STR],
                                                  DATETIME_FMT[:-2])
      except ValueError as e:
        print(unble_msg.format(type=DATE_STR, line=line))
        print(e)
        continue
      # We only consider the picks from the date range (if specified)
      if start is not None and result[DATE_STR] < start: continue
      if end is not None and result[DATE_STR] >= end + ONE_DAY: break
      # Station, We only consider the picks from the stations (if specified)
      result[STATION_STR] = result[STATION_STR].strip(SPACE_STR)
      if stations is not None and result[STATION_STR] not in \
        stations[result[DATE_STR].strftime(DATE_FMT)]: continue
      # P Time
      try:
        result[P_TIME_STR] = result[DATE_STR] + td(seconds=\
          float(result[P_TIME_STR][:2].replace(SPACE_STR, ZERO_STR) + \
                PERIOD_STR + \
                result[P_TIME_STR][2:].replace(SPACE_STR, ZERO_STR)))
      except ValueError as e:
        print(unble_msg.format(type=P_TIME_STR, line=line))
        print(e)
        continue
      # Event
      if result[EVENT_STR]:
        try:
          result[EVENT_STR] = int(result[EVENT_STR].replace(SPACE_STR,
                                                            ZERO_STR))
        except ValueError as e:
          result[EVENT_STR] = None
          print(unble_msg.format(type=EVENT_STR, line=line))
          print(e)
      DEFAULT_VALUE = 0
      # P Weight
      try:
        if result[P_WEIGHT_STR] == SPACE_STR:
          print(ERRORS[level][NOTABLE_STR].format(pre="DAT, ",
                                                  value=SPACE_STR,
                                                  key=P_WEIGHT_STR,
                                                  post=f"from line: {line}"))
          print(ERRORS[level][ASSIGN_STR].format(pre="DAT, ",
                                                 value=DEFAULT_VALUE,
                                                 key=P_WEIGHT_STR,
                                                 post=f"from line: {line}"))
          result[P_WEIGHT_STR] = DEFAULT_VALUE
        else:
          result[P_WEIGHT_STR] = int(result[P_WEIGHT_STR])
      except ValueError as e:
        print(unble_msg.format(type=P_WEIGHT_STR, line=line))
        print(e)
        continue
      DATA.append([result[EVENT_STR], result[P_TIME_STR], result[P_WEIGHT_STR],
                   PWAVE, None, result[STATION_STR]])
      # S Type
      if result[S_TYPE_STR]:
        # S Weight
        try:
          if result[S_WEIGHT_STR] == SPACE_STR:
            print(ERRORS[level][NOTABLE_STR].format(pre="DAT, ",
                                                    value=SPACE_STR,
                                                    key=S_WEIGHT_STR,
                                                    post=f"from line: {line}"))
            print(ERRORS[level][ASSIGN_STR].format(pre="DAT, ",
                                                   value=DEFAULT_VALUE,
                                                   key=S_WEIGHT_STR,
                                                   post=f"from line: {line}"))
            result[S_WEIGHT_STR] = DEFAULT_VALUE
          else:
            result[S_WEIGHT_STR] = int(result[S_WEIGHT_STR])
        except ValueError as e:
          print(unble_msg.format(type=S_WEIGHT_STR, line=line))
          print(e)
          continue
        # S Time
        try:
          result[S_TIME_STR] = result[DATE_STR] + td(seconds=float(\
            result[S_TIME_STR][:2].replace(SPACE_STR, ZERO_STR) + \
            PERIOD_STR + \
            result[S_TIME_STR][2:].replace(SPACE_STR, ZERO_STR)))
        except ValueError as e:
          print(unble_msg.format(type=S_TIME_STR, line=line))
          print(e)
          continue
        DATA.append([result[EVENT_STR], result[S_TIME_STR],
                     result[S_WEIGHT_STR], SWAVE, None, result[STATION_STR]])
      # TODO: Add debug method
      # if verbose:
      #   print(line)
      #   with open()
      #   print(EVENT_CONTRIVER_DAT.format(STATION_STR=result[STATION_STR].lfill(4, SPACE_STR),
      #                                    P_WEIGHT_STR=result[STATION_STR],
      #                                    DATE_STR=
      #                                    "{P_TIME_STR}"
      #                                    "{S_TIME_STR}"
      #                                    "{S_WEIGHT_STR}"
      #                                     "{EVENT_STR}"))
      continue
    if line == EMPTY_STR: continue
    print(unble_msg.format(type=EMPTY_STR, line=line))
  return pd.DataFrame(columns=HEADER_SRC), \
         pd.DataFrame(DATA, columns=HEADER_MANL)

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
def event_parser_pun(filename : Path, start : UTCDateTime = None,
                     end : UTCDateTime = None, level : str = WARNING_STR) \
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
  unble_msg : str = ERRORS[level][UNABLE_STR].format(pre="PUN, ",
    verb="parse", type="{type}", post="from line: {line}")
  if not filename.exists(): raise FileNotFoundError(filename)
  SOURCE = list()
  event: int = 0
  with open(filename, 'r') as fr: lines = fr.readlines()[1:]
  for line in [l.strip() for l in lines]:
    match = RECORD_EXTRACTOR_PUN.match(line)
    if match:
      result : dict[str] = match.groupdict()
      result[SECONDS_STR] = td(seconds=float(result[SECONDS_STR]))
      result[DATE_STR] = UTCDateTime.strptime(
        result[DATE_STR].replace(SPACE_STR, ZERO_STR), DATETIME_FMT[:-2]
      ) + result[SECONDS_STR]
      # We only consider the picks from the date range (if specified)
      if start is not None and result[DATE_STR] < start: continue
      if end is not None and result[DATE_STR] >= end + ONE_DAY: break
      result[LATITUDE_STR] = result[LATITUDE_STR].replace(SPACE_STR, ZERO_STR)
      result[LONGITUDE_STR] = result[LONGITUDE_STR].replace(SPACE_STR,
                                                            ZERO_STR)
      result[LOCAL_DEPTH_STR] = \
        float(result[LOCAL_DEPTH_STR].replace(SPACE_STR, ZERO_STR))
      result[MAGNITUDE_STR] = float(result[MAGNITUDE_STR].replace(SPACE_STR,
                                                                  ZERO_STR))
      result[NO_STR] = int(result[NO_STR].replace(SPACE_STR, ZERO_STR))
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
def event_parser_hpc(filename : Path, start : UTCDateTime = None,
                     end : UTCDateTime = None,
                     stations : dict[str, set[str]] = None) \
    -> tuple[pd.DataFrame, pd.DataFrame]:
  if not filename.exists(): raise FileNotFoundError(filename)
  DATA = list()
  event = 0
  with open(filename, 'r') as fr: lines = fr.readlines()
  for line in [l.strip() for l in lines]:
    match = RECORD_EXTRACTOR_HPC.match(line)
    if match:
      result : dict[str] = match.groupdict()
      result[STATION_STR] = result[STATION_STR].strip(SPACE_STR)
      if stations is not None and result[STATION_STR] not in stations: pass
      result[DATE_STR] = UTCDateTime.strptime(result[DATE_STR],
                                              DATETIME_FMT[:-2])
      if start is not None and result[DATE_STR] < start: continue
      if end is not None and result[DATE_STR] >= end + ONE_DAY: break
      result[P_TIME_STR] = result[DATE_STR] + td(seconds=float(
        result[P_TIME_STR][:2].replace(SPACE_STR, ZERO_STR) + PERIOD_STR + \
        result[P_TIME_STR][2:].replace(SPACE_STR, ZERO_STR)))
      result[S_TIME_STR] = result[DATE_STR] + td(seconds=float(
        result[S_TIME_STR].replace(SPACE_STR, ZERO_STR)))
      result[P_WEIGHT_STR] = int(result[P_WEIGHT_STR])
      result[S_WEIGHT_STR] = int(result[S_WEIGHT_STR])
      DATA.append([event, result[P_TIME_STR], result[P_WEIGHT_STR], PWAVE,
                   None, result[STATION_STR]])
      DATA.append([event, result[S_TIME_STR], result[S_WEIGHT_STR], SWAVE,
                   None, result[STATION_STR]])
      event += 1
      continue
    print("WARNING: (HPC) Unable to parse line:", line)
  return (pd.DataFrame(columns=HEADER_SRC),
          pd.DataFrame(DATA, columns=HEADER_MANL))

RECORD_EXTRACTOR_HPL = re.compile(
  fr"^(?P<{EVENT_STR}>[\d\s]{{6}})\s"                         # Event
  fr"(?P<{STATION_STR}>[A-Z0-9\s]{{4}})\s"                    # Station
  fr"(([\d\s]{{3}}\.\d)|\s{{5}})\s"                           # Unknown
  fr"([\d\s]{{3}})\s"                                         # Unknown
  fr"([\d\s]{{3}})\s"                                         # Unknown
  fr"(?P<{P_TYPE_STR}>[ei?\s]{PWAVE}[cC\+dD0\-\s])"           # P Type
  fr"(?P<{P_WEIGHT_STR}>[0-4])\s"                             # P Weight
  fr"(?P<{P_TIME_STR}>[\s\d]{{4}})\s"                         # P Time [hhmm]
  fr"(?P<{SECONDS_STR}>([\s\d]\d\.\d{{2}})|\s{{5}})"          # Seconds [ss.ss]
  fr"(([\s\d]{{2}}\d\.\d{{2}})|[\s\*]{{6}})\s"                # Unknown
  fr"(([\s\d]\d\.\d{{2}})|\s{{5}})\s"                         # Unknown
  fr"(([\s\d]\d\.\d{{2}})|\s{{5}})"                           # Unknown
  fr"(([\s\d\-]{{2}}\d\.\d{{2}})|[\s\*]{{6}})\s"              # Unknown
  fr"(([\s\-]\d\.\d{{2}})|[\s\*]{{5}})\s"                     # Unknown
  fr"([\d\s]{{3}})\s"                                         # Unknown
  fr"([\d\s]{{2}})\s"                                         # Unknown
  fr"(([\d\s]\d\.\d{{2}})|\s{{5}})\s"                         # Unknown
  fr"(\d|\s)\s"                                               # Unknown
  fr"(.{{4}})\s"                                              # Unknown
  fr"([ACEFGLORSTV\s]"                                        # Unknown
  fr"[ELTU\s]"                                                # Unknown
  fr"[DESU\s])"                                               # Unknown
  fr"([\d\s\*]{{4}})"                                         # Unknown
  fr"([\s\-](\d\.\d)|\s{{4}})[\*\s]\s"                        # Unknown
  fr"((?P<{S_TYPE_STR}>[ei\s]{SWAVE})\s"                      # S Type
  fr"(?P<{S_WEIGHT_STR}>[0-4])?\s"                            # S Weight
  fr"(?P<{S_TIME_STR}>[\s\d]\d\.\d{{2}}))?"                   # S Time
)
#print(RECORD_EXTRACTOR_HPL.pattern)
EVENT_EXTRACTOR_HPL = re.compile(
  fr"^(?P<{EVENT_STR}>[\d\s]{{6}})1"                          # Event
  fr"(?P<{DATE_STR}>\d{{6}}\s[\s\d]{{4}})\s"                  # Date [yymmdd hhmm]
  fr"((?P<{SECONDS_STR}>[\s\d]{{2}}\.\d{{2}})|\s{{5}})\s"     # Seconds [ss.ss]
  fr"((?P<{LATITUDE_STR}>[\s\d]{{2}}-[\s\d]{{2}}\.\d{{2}})|"  # Latitude
  fr"\s{{8}})\s{{2}}"
  fr"((?P<{LONGITUDE_STR}>[\s\d]{{2}}-[\s\d]{{2}}\.\d{{2}})|" # Longitude
  fr"\s{{8}})\s{{2}}"
  fr"((?P<{LOCAL_DEPTH_STR}>[\s\d]{{2}}\.\d{{2}})|"           # Depth
  fr"\s{{5}})\s{{2}}"
  fr"((?P<{MAGNITUDE_STR}>[\s\-]\d\.\d{{2}})|\s{{5}})\s"      # Magnitude
  fr"((?P<{NO_STR}>[\s\d]\d)|\s{{2}})"                        # NO
  fr"([\s\d]{{3}})\s"                                         # Unknown
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
#print(LOCATION_EXTRACTOR_HPL.pattern)
NOTES_EXTRACTOR_HPL = re.compile(fr"^\*\s+(?P<{NOTES_STR}>.*)")
def event_parser_hpl(filename : Path, start : UTCDateTime = None,
                     end : UTCDateTime = None, level : str = WARNING_STR,
                     stations : dict[str, set[str]] = None) \
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
  unble_msg : str = ERRORS[level][UNABLE_STR].format(pre="HPL, ",
    verb="parse", type="{type}", post="from line: {line}")
  notbl_msg : str = ERRORS[level][NOTABLE_STR].format(pre="HPL, ",
    value="{value}", key="{key}", post="from line: {line}")
  if not filename.exists(): raise FileNotFoundError(filename)
  SOURCE = list()
  DETECT = list()
  event_id: int = 0
  event_name: str = ""
  event_notes: str = ""
  event_detect: int = 0
  event_metadata = dict()
  event_spacetime = (UTCDateTime(0), 0, 0, 0)
  with open(filename, 'r') as fr: lines = fr.readlines()
  for line in [l.strip("\n") for l in lines]:
    if event_detect > 0:
      event_detect -= 1
      match = RECORD_EXTRACTOR_HPL.match(line)
      if match:
        result : dict[str] = match.groupdict()
        date = UTCDateTime(event_spacetime[0].date)
        result[STATION_STR] = result[STATION_STR].strip(SPACE_STR)
        if stations and result[STATION_STR] not in \
          stations[date.strftime(DATE_FMT)]: continue
        result[EVENT_STR] = int(result[EVENT_STR])
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
        result[P_TIME_STR] = UTCDateTime(date) + hrs + min
        DETECT.append([result[EVENT_STR],
                       result[P_TIME_STR] + result[SECONDS_STR],
                       result[P_WEIGHT_STR], PWAVE, None, result[STATION_STR]])
        if result[S_TYPE_STR]:
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
        result : dict[str] = match.groupdict()
        result[EVENT_STR] = int(result[EVENT_STR])
        event_id = result[EVENT_STR]
        result[SECONDS_STR] = td(seconds=float(result[SECONDS_STR])) \
                                if result[SECONDS_STR] else td(0)
        result[DATE_STR] = UTCDateTime.strptime(
          result[DATE_STR].replace(SPACE_STR, ZERO_STR), "%y%m%d0%H%M") + \
          result[SECONDS_STR]
        if start is not None and result[DATE_STR] < start:
          event_detect = -1
          continue
        if end is not None and result[DATE_STR] >= end + ONE_DAY: break
        result[LATITUDE_STR] = result[LATITUDE_STR].replace(SPACE_STR,
                                                            ZERO_STR)\
                                 if result[LATITUDE_STR] else None
        if result[LATITUDE_STR]: 
          splt = result[LATITUDE_STR].split(DASH_STR)
          result[LATITUDE_STR] = float(splt[0]) + float(splt[1]) / 60.
        result[LONGITUDE_STR] = result[LONGITUDE_STR].replace(SPACE_STR,
                                                              ZERO_STR) \
                                  if result[LONGITUDE_STR] else None
        if result[LONGITUDE_STR]:
          splt = result[LONGITUDE_STR].split(DASH_STR)
          result[LONGITUDE_STR] = float(splt[0]) + float(splt[1]) / 60.
        result[LOCAL_DEPTH_STR] = float(result[LOCAL_DEPTH_STR]) \
                                    if result[LOCAL_DEPTH_STR] else NaN
        event_spacetime = (result[DATE_STR], result[LATITUDE_STR],
                          result[LONGITUDE_STR], result[LOCAL_DEPTH_STR])
        result[MAGNITUDE_STR] = float(result[MAGNITUDE_STR]) \
                                  if result[MAGNITUDE_STR] else NaN
        if result[NO_STR]: result[NO_STR] = int(result[NO_STR])
        event_detect = int(result[NOTES_STR])
        event_metadata = {
          MAGNITUDE_STR : result[MAGNITUDE_STR],
          NO_STR        : result[NO_STR]
        }
        SOURCE.append([event_id, *event_spacetime, result[MAGNITUDE_STR],
                      result[NO_STR], *([None] * 7)])
        continue
      match = LOCATION_EXTRACTOR_HPL.match(line)
      if match:
        result : dict[str] = match.groupdict()
        event_name = result[LOC_NAME_STR]
        continue
      if event_detect == 0:
        match = NOTES_EXTRACTOR_HPL.match(line)
        if match:
          result : dict[str] = match.groupdict()
          event_notes = result[NOTES_STR]
          SOURCE[-1][-1] = event_notes
          continue
      if re.match(r"^\s*$", line): continue
    if event_detect >= 0: print(unble_msg.format(type=EMPTY_STR, line=line))
  SOURCE = pd.DataFrame(SOURCE, columns=HEADER_SRC)
  DETECT = pd.DataFrame(DETECT, columns=HEADER_MANL)
  return SOURCE, DETECT

# TODO: Implement ObsPy Catalog for OGS files
# TODO: Implement ObsPy Event for OGS files
# TODO: Implement ObsPy Origin for OGS files
def event_parser_qml(filename : Path, start : UTCDateTime = None,
                     end : UTCDateTime = None,
                     stations : dict[str, set[str]] = None) -> pd.DataFrame:
  if not filename.exists(): raise FileNotFoundError(filename)
  SOURCE : list[list] = list()
  DETECT : list[list] = list()
  for event in obspy_read_events(filename):
    if len(event.origins): e_origin = event.origins[0]
    else: continue
    e_time = e_origin.time
    if start is not None and e_time < start: continue
    if end is not None and e_time >= end + ONE_DAY: break
    DETECT.extend([[None, pick.time, pick.time_errors.uncertainty,
                    pick.phase_hint, pick.waveform_id.network_code,
                    pick.waveform_id.station_code] for pick in event.picks])
    SOURCE.append([None, e_time, e_origin.latitude, e_origin.longitude,
                   e_origin.depth, None, len(event.picks), *([None] * 6),
                   event.event_type])
  DETECT = pd.DataFrame(DETECT, columns=HEADER_MANL)
  DETECT = DETECT[DETECT[STATION_STR].isin(stations)] if stations else DETECT
  SOURCE = pd.DataFrame(SOURCE, columns=HEADER_SRC)
  return SOURCE, DETECT

STATION_EXTRACTOR_MOD = re.compile(
  fr"^(?P<{STATION_STR}>[A-Z0-9\s]{{4}})"                         # Station
  fr"(?P<{LONGITUDE_STR}>[\s\d]{{4}}\.\d{{2}}[NS])\s"             # Longitude
  fr"(?P<{LATITUDE_STR}>[\s\d]{{4}}\.\d{{2}}[EW])"                # Latitude
  fr"(?P<{LOCAL_DEPTH_STR}>[\s\d]{{4}}).*"                        # Depth
  fr"[\.\s](?P<{TIMESTAMP_STR}>[A-Z\d]+)?\s*$"                    # Unknown
)
def event_parser_mod(filename : Path,
                     stations : dict[str, set[str]] = None) -> dict[str, str]:
  if not filename.exists(): raise FileNotFoundError(filename)
  STATIONS = {station : station for station in stations} \
             if stations is not None else dict()
  DATA = list()
  with open(filename, 'r') as fr: lines = fr.readlines()
  for line in [l.strip() for l in lines]:
    match = STATION_EXTRACTOR_MOD.match(line)
    if match:
      result = match.groupdict()
      result[STATION_STR] = result[STATION_STR].strip(SPACE_STR)
      # TODO: Review MOD file
      if stations is not None and result[STATION_STR] not in STATIONS: pass
      if len(result[TIMESTAMP_STR]) <= 4:
        result[TIMESTAMP_STR] = result[STATION_STR]
      result[LONGITUDE_STR] = \
        result[LONGITUDE_STR][:2] + DASH_STR + \
        result[LONGITUDE_STR][2:-1].replace(SPACE_STR, ZERO_STR)
      result[LATITUDE_STR] = result[LATITUDE_STR][:2] + DASH_STR + \
                             result[LATITUDE_STR][2:-1].replace(SPACE_STR,
                                                                ZERO_STR)
      DATA.append([
        result[STATION_STR], result[LONGITUDE_STR], result[LATITUDE_STR],
        int(result[LOCAL_DEPTH_STR].replace(SPACE_STR, ZERO_STR)), None])
      STATIONS[result[STATION_STR]] = result[TIMESTAMP_STR]
      continue
    print("WARNING: (MOD) Unable to parse line:", line)
  DATA = pd.DataFrame(DATA, columns=HEADER_SNSR)
  DATA = DATA[DATA[STATION_STR].isin(STATIONS.keys())]
  for code, name in STATIONS.items():
    DATA.loc[DATA[STATION_STR] == code][TIMESTAMP_STR] = name
  return STATIONS

def event_parser_(filename : Path, start : UTCDateTime = None,
                  end : UTCDateTime = None,
                  stations : dict[str, set[str]] = None) -> pd.DataFrame:
  if not filename.exists(): raise FileNotFoundError(filename)
  sfx = filename.suffix
  if sfx == BLT_EXT: pass
  elif sfx == DAT_EXT: return event_parser_dat(filename, start, end,
                                               stations=stations)
  elif sfx == HPC_EXT: pass#return event_parser_hpc(filename, start, end,
                           #                    stations=stations)
  elif sfx == HPL_EXT: return event_parser_hpl(filename, start, end,
                                               stations=stations)
  elif sfx == MOD_EXT: pass#return event_parser_mod(filename, start, end,
                           #                    stations=stations)
  elif sfx == PRT_EXT: pass
  elif sfx == PUN_EXT: return event_parser_pun(filename, start, end)
  elif sfx == QML_EXT: return event_parser_qml(filename, start, end,
                                               stations=stations)
  print(ValueError(f"WARNING: Unknown file extension: {sfx}"))
  return pd.DataFrame(columns=HEADER_SRC), pd.DataFrame(columns=HEADER_MANL)

def event_parser(filename : Path, start : UTCDateTime = None,
                 end : UTCDateTime = None,
                 stations : dict[str, set[str]] = None) -> pd.DataFrame:
  if not filename.exists(): raise FileNotFoundError(filename)
  SOURCE = pd.DataFrame(columns=HEADER_SRC)
  DETECT = pd.DataFrame(columns=HEADER_MANL)
  if filename.is_dir():
    def process_file(file):
      return file.suffix, (event_parser(file, start, end, stations))
    with ThreadPoolExecutor() as executor:
      results = list(executor.map(process_file, filename.iterdir()))
    FIND_SRC = [TIMESTAMP_STR, LONGITUDE_STR, LATITUDE_STR, LOCAL_DEPTH_STR,
                MAGNITUDE_STR]
    FIND_DTC = [ID_STR, TIMESTAMP_STR, PROBABILITY_STR, PHASE_STR, STATION_STR]
    for sfx, (source, detect) in results:
      if sfx == PUN_EXT:
        SOURCE = event_merger_l(SOURCE, source, FIND_SRC)
      elif sfx == DAT_EXT: DETECT = event_merger_l(DETECT, detect, FIND_DTC)
      elif sfx == HPL_EXT:
        SOURCE = event_merger_l(SOURCE, source, FIND_SRC)
        DETECT = event_merger_l(DETECT, detect, FIND_DTC)
    if DETECT is not None and not DETECT.empty:
      SOURCE = SOURCE[SOURCE[ID_STR].isin(DETECT[ID_STR].unique())]
    #for file in filename.iterdir():
    #  if file.suffix == MOD_EXT:
    #    stations = event_parser_mod(file, stations)
    #    for code, name in stations.items():
    #      DETECT.loc[DETECT[STATION_STR] == code][STATION_STR] = name
  else:
    SOURCE, DETECT = event_parser_(filename, start, end, stations)
    #try:
    #except Exception as e:
    #  print(f"WARNING: Unable to parse file: {filename}")
    #  print(e)
  return SOURCE, DETECT
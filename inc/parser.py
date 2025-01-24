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

DEBUG = False

def event_merger_l(NEW : pd.DataFrame, OLD : pd.DataFrame, on : list) \
      -> pd.DataFrame:
  if NEW.empty: return OLD
  if OLD.empty: return NEW
  cols_n = NEW.columns
  off = [col for col in cols_n if col not in on]
  # if DEBUG:
  #   cols_o = OLD.columns
  #   assert all([col in cols_n and col in cols_o for col in on])
  #   assert all([col_a == col_b for col_a, col_b in zip(cols_n, cols_o)])
  idx_r: int = 0
  for idx_l, row in NEW.iterrows():
    if all([row[col] == OLD.loc[idx_r][col] for col in on]):
      for col in off:
        row[col] = OLD.loc[idx_r][col] if OLD.loc[idx_r][col] else row[col]
      NEW.loc[idx_l] = row
      idx_r += 1
  return NEW

# TODO: Parse HPC and QML files

# TODO: Implement polarity
RECORD_EXTRACTOR_DAT = \
  re.compile(fr"^(?P<{STATION_STR}>[A-Z0-9\s]{{4}})"                 # Station
             fr"(?P<{P_TYPE_STR}>[aei\s][Pp]" \
                              fr"[cC\+0-4dD\-Up\s])"                 # P Type
             fr"(?P<{P_WEIGHT_STR}>[0-4\s])"                         # P Weight
             fr"[1-4]"                                               # Unknown
             fr"(?P<{DATE_STR}>\d{{10}})\s"                          # Date
             fr"(?P<{P_TIME_STR}>[\s\d]{{4}})"                       # P Time
             fr".{{8}}"                                              # Unknown
             fr"(((?P<{S_TIME_STR}>[\s\d]{{4}})"                     # S Time
             fr"(?P<{S_TYPE_STR}>[esrwi?13468\s]" \
                              fr"[Ss][cC\+0-4dD\-Ue?\s])"            # S Type
               fr"(?P<{S_WEIGHT_STR}>[0-4\s]))|\s{{8}})"             # S Weight
             fr"(.{{35}}"                                            # Unknown
              fr"(?P<{EVENT_STR}>[\s\d]{{4}}))*")                    # Event
#print(RECORD_EXTRACTOR_DAT.pattern)
EVENT_EXTRACTOR_DAT = re.compile(r"^1.*$")                           # Event
EVENT_CONTRIVER_DAT = "{STATION_STR}" + ALL_WILDCHAR_STR + \
                      PWAVE + ALL_WILDCHAR_STR + \
                      "{P_WEIGHT_STR}" + ALL_WILDCHAR_STR + \
                      "{DATE_STR}" + SPACE_STR + \
                      "{P_TIME_STR}" + ALL_WILDCHAR_STR * 8 + \
                      "{S_TIME_STR}" + ALL_WILDCHAR_STR + \
                      SWAVE + ALL_WILDCHAR_STR + \
                      "{S_WEIGHT_STR}" + ALL_WILDCHAR_STR * 35 + \
                      "{EVENT_STR}"
#print(RECORD_EXTRACTOR_DAT.pattern)
def event_parser_dat(filename : Path, start : UTCDateTime = None,
                     end : UTCDateTime = None,
                     stations : set[str] = None) -> pd.DataFrame:
  warng_msg : str = "WARNING: (DAT) Unable to parse {value} from line: {line}"
  notbl_msg : str = "WARNING: (DAT) {title} has NOTABLE value ({value}) in " \
                                   "line: {line}"
  unkwn_msg : str = "WARNING: (DAT) {title} has UNKNOWN value ({value}) in " \
                                   "line: {line}"
  assgn_msg : str = "WARNING: (DAT) Assining default value ({value}) to " \
                                   "{title} in line: {line}"
  # TODO: Attemp restoration before SHUTDOWN
  error_msg : str = "FATAL: (DAT) "
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
          result[DATE_STR] = \
            UTCDateTime.strptime(result[DATE_STR][:-2], DATETIME_FMT[:-4]) + \
            td(hours=1)
        else:
          result[DATE_STR] = UTCDateTime.strptime(result[DATE_STR],
                                                  DATETIME_FMT[:-2])
      except ValueError as e:
        print(warng_msg.format(value=DATE_STR, line=line))
        print(e)
        continue
      # We only consider the picks from the date range (if specified)
      if start is not None and result[DATE_STR] < start: continue
      if end is not None and result[DATE_STR] >= end + ONE_DAY: continue
      # Station, We only consider the picks from the stations (if specified)
      result[STATION_STR] = result[STATION_STR].strip(SPACE_STR)
      if stations is not None and result[STATION_STR] not in stations: continue
      # P Time
      try:
        result[P_TIME_STR] = result[DATE_STR] + td(seconds=\
          float(result[P_TIME_STR][:2].replace(SPACE_STR, ZERO_STR) + \
                PERIOD_STR + \
                result[P_TIME_STR][2:].replace(SPACE_STR, ZERO_STR)))
      except ValueError as e:
        print(warng_msg.format(value=P_TIME_STR, line=line))
        print(e)
        continue
      # Event
      if result[EVENT_STR]:
        try:
          result[EVENT_STR] = int(result[EVENT_STR].replace(SPACE_STR,
                                                            ZERO_STR))
        except ValueError as e:
          result[EVENT_STR] = None
          print(warng_msg.format(value=EVENT_STR, line=line))
          print(e)
      DEFAULT_VALUE = 0
      # P Weight
      try:
        if result[P_WEIGHT_STR] == SPACE_STR:
          print(notbl_msg.format(P_WEIGHT_STR, SPACE_STR, line))
          print(assgn_msg.format(DEFAULT_VALUE, P_WEIGHT_STR, line))
          result[P_WEIGHT_STR] = DEFAULT_VALUE
        else:
          result[P_WEIGHT_STR] = int(result[P_WEIGHT_STR])
      except ValueError as e:
        print(warng_msg.format(value=P_WEIGHT_STR, line=line))
        print(e)
        continue
      DATA.append([result[EVENT_STR], result[P_TIME_STR], result[P_WEIGHT_STR],
                   PWAVE, None, result[STATION_STR]])
      # S Type
      if result[S_TYPE_STR]:
        # S Weight
        try:
          if result[S_WEIGHT_STR] == SPACE_STR:
            print(notbl_msg.format(S_WEIGHT_STR, SPACE_STR, line))
            print(assgn_msg.format(DEFAULT_VALUE, S_WEIGHT_STR, line))
            result[S_WEIGHT_STR] = DEFAULT_VALUE
          else:
            result[S_WEIGHT_STR] = int(result[S_WEIGHT_STR])
        except ValueError as e:
          print(warng_msg.format(value=S_WEIGHT_STR, line=line))
          print(e)
          continue
        # S Time
        try:
          result[S_TIME_STR] = result[DATE_STR] + td(seconds=float(\
            result[S_TIME_STR][:2].replace(SPACE_STR, ZERO_STR) + \
            PERIOD_STR + \
            result[S_TIME_STR][2:].replace(SPACE_STR, ZERO_STR)))
        except ValueError as e:
          print(warng_msg.format(value=S_TIME_STR, line=line))
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
    print(warng_msg.format(value=EMPTY_STR, line=line))
  return None, pd.DataFrame(DATA, columns=HEADER_MANL)

RECORD_EXTRACTOR_PUN = re.compile(
  fr"^1(?P<{DATE_STR}>\d{{6}}[\s\d]\d[\s\d]\d)\s"                 # Date
  fr"(?P<{SECONDS_STR}>[\s\d]\d\.\d{{2}})\s"                      # Seconds
  fr"(?P<{LATITUDE_STR}>[\s\d]\d-[\s\d]\d\.\d{{2}})\s{{2}}"       # Latitude
  fr"(?P<{LONGITUDE_STR}>[\s\d]\d-[\s\d]\d\.\d{{2}})\s{{2}}"      # Longitude
  fr"(?P<{LOCAL_DEPTH_STR}>[\s\d]\d\.\d{{2}})\s{{3}}"             # Depth
  fr"(?P<{MAGNITUDE_STR}>\d\.\d{{2}})\s"                          # Magnitude
  fr"(?P<{NO_STR}>[\s\d]\d)\s"                                    # NO
  fr"(?P<{GAP_STR}>[\s\d]{{3}})\s"                                # GAP
  fr"(?P<{DMIN_STR}>[\s\d]\d\.\d)\s"                              # DMIN
  fr"(?P<{RMS_STR}>\d\.\d{{2}})\s{{2}}"                           # RMS
  fr"(?P<{ERH_STR}>\d\.\d)\s{{2}}"                                # ERH
  fr"(?P<{ERZ_STR}>\d\.\d)\s"                                     # ERZ
  fr"(?P<{QM_STR}>[A-Z][0-9])")                                   # QM
def event_parser_pun(filename : Path, start : UTCDateTime = None,
                     end : UTCDateTime = None) -> pd.DataFrame:
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
      if end is not None and result[DATE_STR] >= end + ONE_DAY: continue
      result[LATITUDE_STR] = result[LATITUDE_STR].replace(SPACE_STR, ZERO_STR)
      result[LONGITUDE_STR] = result[LONGITUDE_STR].replace(SPACE_STR,
                                                            ZERO_STR)
      result[LOCAL_DEPTH_STR] = float(result[LOCAL_DEPTH_STR])
      result[MAGNITUDE_STR] = float(result[MAGNITUDE_STR])
      result[NO_STR] = int(result[NO_STR])
      result[GAP_STR] = int(result[GAP_STR])
      result[DMIN_STR] = float(result[DMIN_STR])
      result[RMS_STR] = float(result[RMS_STR])
      result[ERH_STR] = float(result[ERH_STR])
      result[ERZ_STR] = float(result[ERZ_STR])
      SOURCE.append([None, result[DATE_STR], result[LATITUDE_STR],
                     result[LONGITUDE_STR], result[LOCAL_DEPTH_STR],
                     result[MAGNITUDE_STR], result[NO_STR], result[GAP_STR],
                     result[DMIN_STR], result[RMS_STR], result[ERH_STR],
                     result[ERZ_STR], result[QM_STR], None])
      event += 1
      continue
    print("WARNING: (PUN) Unable to parse line:", line)
  return pd.DataFrame(SOURCE, columns=HEADER_SRC), None

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
                     stations : set[str] = None) -> pd.DataFrame:
  if not filename.exists(): raise FileNotFoundError(filename)
  DATA = list()
  event = 0

RECORD_EXTRACTOR_HPL = re.compile(
  fr"^(?P<{EVENT_STR}>\d+)\s"                                     # Event
  fr"(?P<{STATION_STR}>[A-Z0-9\s]{{4}})\s{{2}}"                   # Station
  fr".{{12}}\s"                                                   # Unknown
  fr"(?P<{P_TYPE_STR}>[ei]{PWAVE}[cC\+dD\-\s])"                   # P Type
  fr"(?P<{P_WEIGHT_STR}>[0-4])\s"                                 # P Weight
  fr"(?P<{P_TIME_STR}>[\s\d]{{4}})\s"                             # P Time
  fr"(?P<{SECONDS_STR}>[\s\d]\d\.\d{{2}})\s"                      # Seconds
  fr".{{62}}\s"                                                   # Unknown
  fr"((?P<{S_TYPE_STR}>[ei]{SWAVE})\s"                            # S Type
  fr"(?P<{S_WEIGHT_STR}>[0-4])?\s"                                # S Weight
  fr"(?P<{S_TIME_STR}>[\s\d]\d\.\d{{2}}))?"                       # S Time
)
EVENT_EXTRACTOR_HPL = re.compile(
  fr"^(?P<{EVENT_STR}>\d+)1"                                      # Event
  fr"(?P<{DATE_STR}>\d{{6}}\s[\s\d]\d[\s\d]\d)\s"                 # Date
  fr"(?P<{SECONDS_STR}>[\s\d]\d\.\d{{2}})?\s+"                    # Seconds
  fr"((?P<{LATITUDE_STR}>[\s\d]\d-[\s\d]\d\.\d{{2}})\s+"          # Latitude
  fr"(?P<{LONGITUDE_STR}>[\s\d]\d-[\s\d]\d\.\d{{2}})\s+"          # Longitude
  fr"(?P<{LOCAL_DEPTH_STR}>[\s\d]\d\.\d{{2}}))?\s+"               # Depth
  fr"(?P<{MAGNITUDE_STR}>\d\.\d{{2}})?\s+"                        # Magnitude
  fr"(?P<{NO_STR}>[\s\d]\d)?\s+"                                  # NO
)
LOCATION_EXTRACTOR_HPL = re.compile(
  fr"^\^(?P<{LOC_NAME_STR}>[A-Z\s\.']+\s\([A-Z]+\))"
)
NOTES_EXTRACTOR_HPL = re.compile(
  fr"^\*\s+(?P<{NOTES_STR}>.*)"
)
def event_parser_hpl(filename : Path, start : UTCDateTime = None,
                     end : UTCDateTime = None,
                     stations : set[str] = None) -> pd.DataFrame:
  if not filename.exists(): raise FileNotFoundError(filename)
  SOURCE = list()
  DETECT = list()
  event_id: int = 0
  event_name: str = ""
  event_notes: str = ""
  event_detect: str = False
  event_metadata = dict()
  event_spacetime = (UTCDateTime(0), 0, 0, 0)
  with open(filename, 'r') as fr: lines = fr.readlines()
  for line in [l.strip() for l in lines]:
    match = LOCATION_EXTRACTOR_HPL.match(line)
    if match:
      result : dict[str] = match.groupdict()
      event_name = result[LOC_NAME_STR]
      continue
    match = EVENT_EXTRACTOR_HPL.match(line)
    if match:
      event_detect = True
      result : dict[str] = match.groupdict()
      result[EVENT_STR] = int(result[EVENT_STR])
      event_id = result[EVENT_STR]
      result[SECONDS_STR] = td(seconds=float(result[SECONDS_STR])) \
                              if result[SECONDS_STR] else td(0)
      result[DATE_STR] = UTCDateTime.strptime(
        result[DATE_STR].replace(SPACE_STR, ZERO_STR), "%y%m%d0%H%M") + \
        result[SECONDS_STR]
      if start is not None and result[DATE_STR] < start:
        event_detect = False
        continue
      if end is not None and result[DATE_STR] >= end + ONE_DAY:
        event_detect = False
        break
      result[LATITUDE_STR] = result[LATITUDE_STR].replace(SPACE_STR, ZERO_STR) \
                               if result[LATITUDE_STR] else None
      result[LONGITUDE_STR] = \
        result[LONGITUDE_STR].replace(SPACE_STR, ZERO_STR) \
        if result[LONGITUDE_STR] else None
      result[LOCAL_DEPTH_STR] = float(result[LOCAL_DEPTH_STR]) \
                                  if result[LOCAL_DEPTH_STR] else NaN
      event_spacetime = (result[DATE_STR], result[LATITUDE_STR],
                         result[LONGITUDE_STR], result[LOCAL_DEPTH_STR])
      result[MAGNITUDE_STR] = float(result[MAGNITUDE_STR]) \
                                if result[MAGNITUDE_STR] else NaN
      result[NO_STR] = int(result[NO_STR]) if result[NO_STR] else NaN
      event_metadata = {
        MAGNITUDE_STR : result[MAGNITUDE_STR],
        NO_STR        : result[NO_STR]
      }
      SOURCE.append([event_id, *event_spacetime, result[MAGNITUDE_STR],
                     result[NO_STR], *([None] * 7)])
      continue
    match = RECORD_EXTRACTOR_HPL.match(line)
    if match and event_detect:
      result : dict[str] = match.groupdict()
      result[STATION_STR] = result[STATION_STR].strip(SPACE_STR)
      if stations is not None and result[STATION_STR] not in stations: continue
      result[EVENT_STR] = int(result[EVENT_STR])
      result[P_WEIGHT_STR] = int(result[P_WEIGHT_STR])
      result[SECONDS_STR] = td(seconds=float(result[SECONDS_STR]))
      result[P_TIME_STR] = UTCDateTime.strptime(
        event_spacetime[0].date.strftime(DATE_FMT) + \
        result[P_TIME_STR].replace(SPACE_STR, ZERO_STR), DATETIME_FMT[:-2])
      DETECT.append([result[EVENT_STR],
                     result[P_TIME_STR] + result[SECONDS_STR],
                     result[P_WEIGHT_STR], PWAVE, None, result[STATION_STR]])
      if result[S_TYPE_STR]:
        result[S_WEIGHT_STR] = int(result[S_WEIGHT_STR])
        result[S_TIME_STR] = td(seconds=float(result[S_TIME_STR]))
        DETECT.append([result[EVENT_STR],
                       result[P_TIME_STR] + result[S_TIME_STR],
                       result[S_WEIGHT_STR], SWAVE, None, result[STATION_STR]])
      continue
    match = NOTES_EXTRACTOR_HPL.match(line)
    if match:
      result : dict[str] = match.groupdict()
      event_notes = result[NOTES_STR]
      SOURCE[-1][-1] = event_notes
      continue
    if re.match(r"^\s*$", line): continue
    print("WARNING: (HPL) Unable to parse line:", line)
  SOURCE = pd.DataFrame(SOURCE, columns=HEADER_SRC)
  DETECT = pd.DataFrame(DETECT, columns=HEADER_MANL)
  return SOURCE, DETECT

# TODO: Implement ObsPy Catalog for OGS files
# TODO: Implement ObsPy Event for OGS files
# TODO: Implement ObsPy Origin for OGS files
def event_parser_qml(filename : Path, start : UTCDateTime = None,
                     end : UTCDateTime = None,
                     stations : set[str] = None) -> pd.DataFrame:
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
                     stations : set[str] = None) -> dict[str, str]:
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
      if stations is not None and result[STATION_STR] not in STATIONS:
        continue
      if len(result[TIMESTAMP_STR]) <= 4:
        result[TIMESTAMP_STR] = result[STATION_STR]
      result[LONGITUDE_STR] = \
        result[LONGITUDE_STR][:2] + DASH_STR + \
        result[LONGITUDE_STR][2:-1].replace(SPACE_STR, ZERO_STR)
      result[LATITUDE_STR] = result[LATITUDE_STR][:2] + DASH_STR + \
                             result[LATITUDE_STR][2:-1].replace(SPACE_STR, ZERO_STR)
      DATA.append([result[STATION_STR], result[LONGITUDE_STR],
                   result[LATITUDE_STR],
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
                  stations : set[str] = None) -> pd.DataFrame:
  if not filename.exists(): raise FileNotFoundError(filename)
  sfx = filename.suffix
  if sfx == DAT_EXT: return event_parser_dat(filename, start, end, stations)
  elif sfx == PUN_EXT: return event_parser_pun(filename, start, end)
  elif sfx == HPC_EXT: return event_parser_hpc(filename, start, end, stations)
  elif sfx == HPL_EXT: return event_parser_hpl(filename, start, end, stations)
  elif sfx == MOD_EXT: return event_parser_mod(filename, start, end, stations)
  elif sfx == QML_EXT: return event_parser_qml(filename, start, end, stations)
  print(ValueError(f"WARNING: Unknown file extension: {sfx}"))
  return None

def event_parser(filename : Path, start : UTCDateTime = None,
                 end : UTCDateTime = None,
                 stations : set[str] = None) -> pd.DataFrame:
  if not filename.exists(): raise FileNotFoundError(filename)
  if filename.is_dir():
    def process_file(file):
      sfx = file.suffix
      source = pd.DataFrame(columns=HEADER_SRC)
      detect = pd.DataFrame(columns=HEADER_MANL)
      if sfx == PUN_EXT:
        #source, detect = event_parser_pun(file, start, end)
        pass
      elif sfx == DAT_EXT:
        source, detect = event_parser_dat(file, start, end, stations)
      elif sfx == HPL_EXT:
        # source, detect = event_parser_hpl(file, start, end, stations)
        pass
      elif sfx == QML_EXT:
        source, detect = event_parser_qml(file, start, end, stations)
      else: print(ValueError(f"WARNING: Unknown file extension: {sfx}"))
      return sfx, (source, detect)
    with ThreadPoolExecutor() as executor:
      results = list(executor.map(process_file, filename.iterdir()))
    SOURCE = pd.DataFrame(columns=HEADER_SRC)
    DETECT = pd.DataFrame(columns=HEADER_MANL)
    FIND_SRC = [TIMESTAMP_STR, LONGITUDE_STR, LATITUDE_STR, LOCAL_DEPTH_STR,
                MAGNITUDE_STR, NO_STR]
    FIND_DTC = [ID_STR, TIMESTAMP_STR, PROBABILITY_STR, PHASE_STR, STATION_STR]
    for sfx, (source, detect) in results:
      if sfx == PUN_EXT: SOURCE = event_merger_l(SOURCE, source, FIND_SRC)
      elif sfx == DAT_EXT: DETECT = event_merger_l(DETECT, detect, FIND_DTC)
      elif sfx == HPL_EXT:
        SOURCE = event_merger_l(SOURCE, source, FIND_SRC)
        DETECT = event_merger_l(DETECT, detect, FIND_DTC)
    if DETECT is not None and not DETECT.empty:
      SOURCE = SOURCE[SOURCE[ID_STR].isin(DETECT[ID_STR].unique())]
    for file in filename.iterdir():
      if file.suffix == MOD_EXT:
        stations = event_parser_mod(file, stations)
        for code, name in stations.items():
          DETECT.loc[DETECT[STATION_STR] == code][STATION_STR] = name
  else: SOURCE, DETECT = event_parser_(filename, start, end, stations)
  return SOURCE, DETECT
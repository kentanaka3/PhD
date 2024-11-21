import re
import pandas as pd
from pathlib import Path
from obspy import UTCDateTime
from datetime import timedelta

from constants import *

# TODO: Implement polarity
RECORD_EXTRACTOR_DAT = \
  re.compile(fr"^(?P<{STATION_STR}>(\w{{4}}|\w{{3}}\s))"            # Station
             fr"(?P<{P_TYPE_STR}>[ei?]{PWAVE}[cd\s])"               # P Type
             fr"(?P<{P_WEIGHT_STR}>[0-4])"                          # P Weight
             fr"1(?P<{BEG_DATE_STR}>\d{{10}})"                      # Date
             fr"\s(?P<{P_TIME_STR}>\d{{4}})"                        # P Time
             fr"\s+((?P<{S_TIME_STR}>\d{{4}}|\d{{3}})"              # S Time
             fr"(?P<{S_TYPE_STR}>[ei?]{SWAVE}\s)"                   # S Type
             fr"(?P<{S_WEIGHT_STR}>[0-4]))*")                       # S Weight
EVENT_EXTRACTOR_DAT = re.compile(r"^1(\s+D)*\s*$")                  # Event
def event_parser_dat(filename : Path, start : UTCDateTime = None,
                     end : UTCDateTime = None, verbose : bool = False,
                     stations : set = None) -> pd.DataFrame:
  if not filename.exists(): raise FileNotFoundError(filename)
  DATA = list()
  event = 0
  with open(filename, 'r') as fr: lines = fr.readlines()
  for line in [l.strip() for l in lines]:
    if EVENT_EXTRACTOR_DAT.match(line):
      event += 1
      continue
    match = RECORD_EXTRACTOR_DAT.match(line)
    if match:
      result = match.groupdict()
      result[BEG_DATE_STR] = UTCDateTime.strptime(result[BEG_DATE_STR],
                                                  "%y%m%d%H%M")
      # We only consider the picks from the date range (if specified)
      if start is not None and result[BEG_DATE_STR] < start: continue
      if end is not None and result[BEG_DATE_STR] >= end + ONE_DAY: continue
      # We only consider the picks from the stations (if specified)
      result[STATION_STR] = result[STATION_STR].strip(SPACE_STR)
      if stations is not None and result[STATION_STR] not in stations: continue
      result[P_WEIGHT_STR] = int(result[P_WEIGHT_STR])
      result[P_TIME_STR] = \
        result[BEG_DATE_STR] + td(seconds=float(result[P_TIME_STR][:2] + \
                                                PERIOD_STR + \
                                                result[P_TIME_STR][2:]))
      DATA.append([event, result[STATION_STR], PWAVE, result[P_TIME_STR],
                  result[P_WEIGHT_STR]])
      if result[S_TIME_STR]:
        result[S_WEIGHT_STR] = int(result[S_WEIGHT_STR])
        result[S_TIME_STR] = \
          result[BEG_DATE_STR] + td(seconds=float(result[S_TIME_STR][:2] + \
                                                  PERIOD_STR + \
                                                  result[S_TIME_STR][2:]))
        DATA.append([event, result[STATION_STR], SWAVE, result[S_TIME_STR],
                    result[S_WEIGHT_STR]])
  HEADER = [EVENT_STR, STATION_STR, PHASE_STR, TIMESTAMP_STR, WEIGHT_STR]
  return pd.DataFrame(DATA, columns=HEADER)

# TODO: Parse HPC, HPL, PUN and QML files
RECORD_EXTRACTOR_PUN = re.compile(
  fr"^1(?P<{BEG_DATE_STR}>\d{{6}}[\s\d]\d[\s\d]\d)\s"             # Date
  fr"(?P<{SECONDS_STR}>[\s|\d]\d\.\d{{2}})\s"                     # Seconds
  fr"(?P<{LATITUDE_STR}>[\s|\d]\d-[\s|\d]\d\.\d{{2}})\s{{2}}"     # Latitude
  fr"(?P<{LONGITUDE_STR}>[\s|\d]\d-[\s|\d]\d\.\d{{2}})\s{{2}}"    # Longitude
  fr"(?P<{LOCAL_DEPTH_STR}>[\s|\d]\d\.\d{{2}})\s{{3}}"            # Depth
  fr"(?P<{MAGNITUDE_STR}>\d\.\d{{2}})\s"                          # Magnitude
  fr"(?P<{NO_STR}>[\s|\d]\d)\s"                                   # NO
  fr"(?P<{GAP_STR}>(\d{{3}}|\s\d{{2}}))\s"                        # GAP
  fr"(?P<{DMIN_STR}>[\s|\d]\d\.\d)\s"                             # DMIN
  fr"(?P<{RMS_STR}>\d\.\d{{2}})\s{{2}}"                           # RMS
  fr"(?P<{ERH_STR}>\d\.\d)\s{{2}}"                                # ERH
  fr"(?P<{ERZ_STR}>\d\.\d)\s"                                     # ERZ
  fr"(?P<{QM_STR}>[A-Z][0-9])")                                   # QM
def event_parser_pun(filename : Path, start : UTCDateTime = None,
                     end : UTCDateTime = None, verbose : bool = False,
                     stations : set = None) -> list:
  if not filename.exists(): raise FileNotFoundError(filename)
  DATA = list()
  with open(filename, 'r') as fr: lines = fr.readlines()[1:]
  for line in [l.strip() for l in lines]:
    match = RECORD_EXTRACTOR_PUN.match(line)
    if match:
      result = match.groupdict()
      result[SECONDS_STR] = timedelta(seconds=float(result[SECONDS_STR]))
      result[BEG_DATE_STR] = UTCDateTime.strptime(
        result[BEG_DATE_STR].replace(SPACE_STR, "0"), "%y%m%d%H%M") + \
        result[SECONDS_STR]
      # We only consider the picks from the date range (if specified)
      if start is not None and result[BEG_DATE_STR] < start: continue
      if end is not None and result[BEG_DATE_STR] >= end + ONE_DAY: continue
      result[LOCAL_DEPTH_STR] = float(result[LOCAL_DEPTH_STR])
      result[MAGNITUDE_STR] = float(result[MAGNITUDE_STR])
      result[NO_STR] = int(result[NO_STR])
      result[GAP_STR] = int(result[GAP_STR])
      result[DMIN_STR] = float(result[DMIN_STR])
      result[RMS_STR] = float(result[RMS_STR])
      result[ERH_STR] = float(result[ERH_STR])
      result[ERZ_STR] = float(result[ERZ_STR])
      DATA.append([result[BEG_DATE_STR], result[LATITUDE_STR],
                   result[LONGITUDE_STR], result[LOCAL_DEPTH_STR],
                   result[MAGNITUDE_STR], result[NO_STR], result[GAP_STR],
                   result[DMIN_STR], result[RMS_STR], result[ERH_STR],
                   result[ERZ_STR], result[QM_STR]])
  HEADER = [TIMESTAMP_STR, LATITUDE_STR, LONGITUDE_STR, LOCAL_DEPTH_STR,
            MAGNITUDE_STR, NO_STR, GAP_STR, DMIN_STR, RMS_STR, ERH_STR,
            ERZ_STR, QM_STR]
  return pd.DataFrame(DATA, columns=HEADER)

RECORD_EXTRACTOR_HPC = re.compile(
  fr"^(?P<{STATION_STR}>(\w{{4}}|\w{{3}}\s))"                     # Station
  fr"(?P<{P_TYPE_STR}>[ei?]{PWAVE}[cd\s])"                        # P Type
  fr"(?P<{P_WEIGHT_STR}>[0-4])"                                   # P Weight
  fr"1(?P<{BEG_DATE_STR}>\d{{10}})"                               # Date
  fr"\s(?P<{P_TIME_STR}>\d{{4}})"                                 # P Time
  fr"\s+((?P<{S_TIME_STR}>\d{{4}}|\d{{3}})"                       # S Time
  fr"(?P<{S_TYPE_STR}>[ei?]{SWAVE}\s)"                            # S Type
  fr"(?P<{S_WEIGHT_STR}>[0-4]))*")                                # S Weight
def event_parser_hpc(filename : Path, start : UTCDateTime = None,
                     end : UTCDateTime = None, verbose : bool = False,
                     stations : set = None) -> pd.DataFrame:
  if not filename.exists(): raise FileNotFoundError(filename)
  DATA = list()
  event = 0

RECORD_EXTRACTOR_HPL = re.compile(
  fr""
)
def event_parser_hpl(filename : Path, start : UTCDateTime = None,
                     end : UTCDateTime = None, verbose : bool = False,
                      stations : set = None) -> pd.DataFrame:
  if not filename.exists(): raise FileNotFoundError(filename)
  DATA = list()
  with open(filename, 'r') as fr: lines = fr.readlines()
  for line in [l.strip() for l in lines]:
    match = RECORD_EXTRACTOR_HPL.match(line)
    if match:
      result = match.groupdict()

def event_parser_qml(filename : Path, start : UTCDateTime = None,
                     end : UTCDateTime = None, verbose : bool = False,
                     stations : set = None) -> pd.DataFrame:
  if not filename.exists(): raise FileNotFoundError(filename)
  pass

def event_parser_(filename : Path, start : UTCDateTime = None,
                  end : UTCDateTime = None, verbose : bool = False,
                  stations : set = None) -> pd.DataFrame:
  if not filename.exists(): raise FileNotFoundError(filename)
  sfx = filename.suffix
  if sfx == DAT_EXT:
    return event_parser_dat(filename, start, end, verbose, stations)
  elif sfx == PUN_EXT:
    return event_parser_pun(filename, start, end, verbose, stations)
  elif sfx == HPC_EXT:
    return event_parser_hpc(filename, start, end, verbose, stations)
  elif sfx == HPL_EXT:
    return event_parser_hpl(filename, start, end, verbose, stations)
  elif sfx == QML_EXT:
    return event_parser_qml(filename, start, end, verbose, stations)
  else:
    raise ValueError(f"Unknown file extension: {sfx}")

def event_parser(filename : Path, start : UTCDateTime = None,
                 end : UTCDateTime = None, verbose : bool = False,
                 stations : set = None) -> pd.DataFrame:
  if not filename.exists(): raise FileNotFoundError(filename)
  if filename.is_dir():
    # TODO: Implement parallel processing
    # TODO: Handle concatenation of dataframes
    DATA = list()
    for file in filename.iterdir():
      DATA.append(event_parser_(file, start, end, verbose, stations))
    return pd.concat(DATA)
  else:
    return event_parser_(filename, start, end, verbose, stations)
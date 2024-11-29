import re
import pandas as pd
from pathlib import Path
from numpy import nan as NaN
from obspy import UTCDateTime
from numpy import isnan as isNaN
from datetime import timedelta as td

from constants import *

# TODO: Parse HPC and QML files

# TODO: Implement polarity
RECORD_EXTRACTOR_DAT = \
  re.compile(fr"^(?P<{STATION_STR}>[A-Z0-9\s]{{4}})"                # Station
             fr"(?P<{P_TYPE_STR}>[ei]{PWAVE}[cd\s])"                # P Type
             fr"(?P<{P_WEIGHT_STR}>[0-4])1"                         # P Weight
             fr"(?P<{BEG_DATE_STR}>\d{{10}})\s"                     # Date
             fr"(?P<{P_TIME_STR}>\d{{4}})\s+"                       # P Time
             fr"((?P<{S_TIME_STR}>\d{{4}}|\d{{3}})"                 # S Time
             fr"(?P<{S_TYPE_STR}>[ei]{SWAVE})\s"                    # S Type
             fr"(?P<{S_WEIGHT_STR}>[0-4]))?")                       # S Weight
EVENT_EXTRACTOR_DAT = re.compile(r"^1(\s+D)*\s*$")                  # Event
def event_parser_dat(filename : Path, start : UTCDateTime = None,
                     end : UTCDateTime = None, verbose : bool = False,
                     stations : set[str] = None) -> pd.DataFrame:
  if not filename.exists(): raise FileNotFoundError(filename)
  DATA = list()
  event: int = 0
  with open(filename, 'r') as fr: lines = fr.readlines()
  for line in [l.strip() for l in lines]:
    if EVENT_EXTRACTOR_DAT.match(line):
      event += 1
      continue
    match = RECORD_EXTRACTOR_DAT.match(line)
    if match:
      result: dict[str] = match.groupdict()
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
      if result[S_TYPE_STR]:
        result[S_WEIGHT_STR] = int(result[S_WEIGHT_STR])
        result[S_TIME_STR] = \
          result[BEG_DATE_STR] + td(seconds=float(result[S_TIME_STR][:2] + \
                                                  PERIOD_STR + \
                                                  result[S_TIME_STR][2:]))
        DATA.append([event, result[STATION_STR], SWAVE, result[S_TIME_STR],
                    result[S_WEIGHT_STR]])
      continue
    print("WARNING: (DAT) Unable to parse line:", line)
  HEADER = [EVENT_STR, STATION_STR, PHASE_STR, TIMESTAMP_STR, WEIGHT_STR]
  return pd.DataFrame(DATA, columns=HEADER)

RECORD_EXTRACTOR_PUN = re.compile(
  fr"^1(?P<{BEG_DATE_STR}>\d{{6}}[\s\d]\d[\s\d]\d)\s"             # Date
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
                     end : UTCDateTime = None, verbose : bool = False) \
    -> pd.DataFrame:
  if not filename.exists(): raise FileNotFoundError(filename)
  DATA = list()
  with open(filename, 'r') as fr: lines = fr.readlines()[1:]
  for line in [l.strip() for l in lines]:
    match = RECORD_EXTRACTOR_PUN.match(line)
    if match:
      result : dict[str] = match.groupdict()
      result[SECONDS_STR] = td(seconds=float(result[SECONDS_STR]))
      result[BEG_DATE_STR] = UTCDateTime.strptime(
        result[BEG_DATE_STR].replace(SPACE_STR, "0"), "%y%m%d%H%M") + \
        result[SECONDS_STR]
      # We only consider the picks from the date range (if specified)
      if start is not None and result[BEG_DATE_STR] < start: continue
      if end is not None and result[BEG_DATE_STR] >= end + ONE_DAY: continue
      result[LATITUDE_STR] = result[LATITUDE_STR].replace(SPACE_STR, "0")
      result[LONGITUDE_STR] = result[LONGITUDE_STR].replace(SPACE_STR, "0")
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
      continue
    print("WARNING: (PUN) Unable to parse line:", line)
  HEADER = [TIMESTAMP_STR, LATITUDE_STR, LONGITUDE_STR, LOCAL_DEPTH_STR,
            MAGNITUDE_STR, NO_STR, GAP_STR, DMIN_STR, RMS_STR, ERH_STR,
            ERZ_STR, QM_STR]
  return pd.DataFrame(DATA, columns=HEADER)

RECORD_EXTRACTOR_HPC = re.compile(
  fr"^(?P<{STATION_STR}>[A-Z0-9\s]{{4}})"                         # Station
  fr"(?P<{P_TYPE_STR}>[ei]{PWAVE}[cd\s])"                         # P Type
  fr"(?P<{P_WEIGHT_STR}>[0-4])1"                                  # P Weight
  fr"(?P<{BEG_DATE_STR}>\d{{10}})\s"                              # Date
  fr"(?P<{P_TIME_STR}>\d{{4}})\s+"                                # P Time
  fr"(?P<{S_TIME_STR}>\d{{4}}|\d{{3}})"                           # S Time
  fr"(?P<{S_TYPE_STR}>[ei]{SWAVE}\s)"                             # S Type
  fr"(?P<{S_WEIGHT_STR}>[0-4])")                                  # S Weight
def event_parser_hpc(filename : Path, start : UTCDateTime = None,
                     end : UTCDateTime = None, verbose : bool = False,
                     stations : set[str] = None) -> pd.DataFrame:
  if not filename.exists(): raise FileNotFoundError(filename)
  DATA = list()
  event = 0

RECORD_EXTRACTOR_HPL = re.compile(
  fr"^(?P<{EVENT_STR}>\d+)\s"                                     # Event
  fr"(?P<{STATION_STR}>[A-Z0-9\s]{{4}})\s+"                       # Station
  fr".+"                                                          # Unknown
  fr"(?P<{P_TYPE_STR}>[ei]{PWAVE}[cd\s])"                         # P Type
  fr"(?P<{P_WEIGHT_STR}>[0-4])\s"                                 # P Weight
  fr"(?P<{P_TIME_STR}>[\s\d]\d[\s\d]\d\s[\s\d]\d\.\d{{2}})\s"     # P Time
  fr".{{62}}\s"                                                   # Unknown
  fr"((?P<{S_TYPE_STR}>[ei]{SWAVE})\s"                            # S Type
  fr"(?P<{S_WEIGHT_STR}>[0-4])?\s"                                # S Weight
  fr"(?P<{S_TIME_STR}>[\s\d]\d\.\d{{2}}))?"                       # S Time
)
EVENT_EXTRACTOR_HPL = re.compile(
  fr"^(?P<{EVENT_STR}>\d+)1"                                      # Event
  fr"(?P<{BEG_DATE_STR}>\d{{6}}\s[\s\d]\d[\s\d]\d)\s"             # Date
  fr"(?P<{SECONDS_STR}>[\s\d]\d\.\d{{2}})?\s+"                    # Seconds
  fr"((?P<{LATITUDE_STR}>[\s\d]\d-[\s\d]\d\.\d{{2}})\s+"          # Latitude
  fr"(?P<{LONGITUDE_STR}>[\s\d]\d-[\s\d]\d\.\d{{2}})\s+"          # Longitude
  fr"(?P<{LOCAL_DEPTH_STR}>[\s\d]\d\.\d{{2}}))?\s+"               # Depth
  fr"(?P<{MAGNITUDE_STR}>\d\.\d{{2}})?\s+"                        # Magnitude
  fr"(?P<{NO_STR}>[\s\d]\d)?\s+"                                  # NO
)
LOCATION_EXTRACTOR_HPL = re.compile(
  fr"^\^(?P<{LOC_NAME_STR}>[A-Z\s]+\s\([A-Z]+\))"
)
NOTES_EXTRACTOR_HPL = re.compile(
  fr"^\*\s+(?P<{NOTES_STR}>.*)"
)
def event_parser_hpl(filename : Path, start : UTCDateTime = None,
                     end : UTCDateTime = None, verbose : bool = False,
                     stations : set[str] = None) -> pd.DataFrame:
  if not filename.exists(): raise FileNotFoundError(filename)
  DATA = list()
  event_id: int = 0
  event_name: str = ""
  event_notes: str = ""
  event_detect: str = False
  event_metadata = dict()
  event_spacetime = (UTCDateTime(0), 0, 0, 0)
  event_detection = list()
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
      event_detection = list()
      result : dict[str] = match.groupdict()
      result[EVENT_STR] = int(result[EVENT_STR])
      event_id = result[EVENT_STR]
      result[SECONDS_STR] = td(seconds=float(result[SECONDS_STR])) \
                              if result[SECONDS_STR] else td(0)
      result[BEG_DATE_STR] = UTCDateTime.strptime(
        result[BEG_DATE_STR].replace(SPACE_STR, "0"), "%y%m%d0%H%M") + \
        result[SECONDS_STR]
      if start is not None and result[BEG_DATE_STR] < start:
        event_detect = False
        continue
      if end is not None and result[BEG_DATE_STR] >= end + ONE_DAY:
        event_detect = False
        continue
      result[LATITUDE_STR] = result[LATITUDE_STR].replace(SPACE_STR, "0") \
                               if result[LATITUDE_STR] else NaN
      result[LONGITUDE_STR] = result[LONGITUDE_STR].replace(SPACE_STR, "0") \
                                if result[LONGITUDE_STR] else NaN
      result[LOCAL_DEPTH_STR] = float(result[LOCAL_DEPTH_STR]) \
                                  if result[LOCAL_DEPTH_STR] else NaN
      event_spacetime = (result[BEG_DATE_STR], result[LATITUDE_STR],
                         result[LONGITUDE_STR], result[LOCAL_DEPTH_STR])
      result[MAGNITUDE_STR] = float(result[MAGNITUDE_STR]) \
                                if result[MAGNITUDE_STR] else NaN
      result[NO_STR] = int(result[NO_STR]) if result[NO_STR] else NaN
      event_metadata = {
        MAGNITUDE_STR : result[MAGNITUDE_STR],
        NO_STR        : result[NO_STR]
      }
      continue
    match = RECORD_EXTRACTOR_HPL.match(line)
    if match and event_detect:
      result : dict[str] = match.groupdict()
      result[STATION_STR] = result[STATION_STR].strip(SPACE_STR)
      if stations is not None and result[STATION_STR] not in stations: continue
      result[EVENT_STR] = int(result[EVENT_STR])
      result[P_WEIGHT_STR] = int(result[P_WEIGHT_STR])
      if result[S_TYPE_STR]:
        result[S_WEIGHT_STR] = int(result[S_WEIGHT_STR])
        result[S_TIME_STR] = td(seconds=float(result[S_TIME_STR]))
      continue
    match = NOTES_EXTRACTOR_HPL.match(line)
    if match:
      result : dict[str] = match.groupdict()
      event_notes = result[NOTES_STR]
      continue
    if re.match(r"^\s*$", line): continue
    print("WARNING: (HPL) Unable to parse line:", line)
  HEADER = [ID_STR, LOC_NAME_STR, TIMESTAMP_STR, LATITUDE_STR, LONGITUDE_STR,
            LOCAL_DEPTH_STR, MAGNITUDE_STR, NO_STR]
  return pd.DataFrame(DATA)

def event_parser_qml(filename : Path, start : UTCDateTime = None,
                     end : UTCDateTime = None, verbose : bool = False,
                     stations : set[str] = None) -> pd.DataFrame:
  if not filename.exists(): raise FileNotFoundError(filename)
  pass

def event_parser_(filename : Path, start : UTCDateTime = None,
                  end : UTCDateTime = None, verbose : bool = False,
                  stations : set[str] = None) -> pd.DataFrame:
  if not filename.exists(): raise FileNotFoundError(filename)
  sfx = filename.suffix
  if sfx == DAT_EXT:
    return event_parser_dat(filename, start, end, verbose, stations)
  elif sfx == PUN_EXT:
    return event_parser_pun(filename, start, end, verbose)
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
                 stations : set[str] = None) -> pd.DataFrame:
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
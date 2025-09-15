import os
import re
import argparse
import numpy as np
import pandas as pd
import itertools as it
from pathlib import Path
from matplotlib.path import Path as mplPath

from datetime import datetime, timedelta as td
from matplotlib.cbook import flatten as flatten_list

import ogsconstants as OGS_C
import ogsplotter as OGS_P

DATA_PATH = Path(__file__).parent.parent.parent

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

def is_dir_path(string: str) -> Path:
  if os.path.isdir(string):
    return Path(os.path.abspath(string))
  else:
    raise NotADirectoryError(string)

def is_file_path(string: str) -> Path:
  if os.path.isfile(string):
    return Path(os.path.abspath(string))
  else:
    raise FileNotFoundError(string)

def is_date(string: str) -> datetime:
  return datetime.strptime(string, OGS_C.YYMMDD_FMT)

def is_julian(string: str) -> datetime:
  # TODO: Define and convert Julian date to Gregorian date
  raise NotImplementedError
  return datetime.strptime(string, OGS_C.YYMMDD_FMT)._set_julday(string)

def is_polygon(points: str) -> mplPath:
  return mplPath(points, closed=True)

class SortDatesAction(argparse.Action):
  def __call__(self, parser, namespace, values, option_string=None):
    setattr(namespace, self.dest, sorted(values)) # type: ignore

def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Parse OGS Manual Catalogs")
  parser.add_argument("-x", "--ext", default=OGS_C.ALL_WILDCHAR_STR, type=str,
                      nargs=OGS_C.ONE_MORECHAR_STR, metavar=OGS_C.EMPTY_STR,
                      help="File extension to process")
  parser.add_argument(
    '-d', "--directory", required=False, type=is_dir_path, default=None,
    nargs=OGS_C.ONE_MORECHAR_STR, metavar=OGS_C.EMPTY_STR,
    help="Directory path to the raw files")
  parser.add_argument(
    '-f', "--file", required=True, type=is_file_path, default=None,
    nargs=OGS_C.ONE_MORECHAR_STR, metavar=OGS_C.EMPTY_STR,
    help="File path to the raw files")
  parser.add_argument(
    '-v', "--verbose", action='store_true', default=False,
    help="Enable verbose output")
  date_group = parser.add_mutually_exclusive_group(required=False)
  date_group.add_argument(
    '-D', "--dates", required=False, metavar=OGS_C.DATE_STD, type=is_date,
    nargs=2, action=SortDatesAction,
    default=[datetime.strptime("240320", OGS_C.YYMMDD_FMT),
             datetime.strptime("240620", OGS_C.YYMMDD_FMT)],
    help="Specify the beginning and ending (inclusive) Gregorian date " \
         "(YYMMDD) range to work with.")
  date_group.add_argument(
    '-J', "--julian", required=False, metavar=OGS_C.DATE_STD,
    action=SortDatesAction, type=is_julian, default=None, nargs=2,
    help="Specify the beginning and ending (inclusive) Julian date (YYMMDD) " \
         "range to work with.")
  parser.add_argument(
    "-N", "--name", required=False, type=str,
    default=DATA_PATH / "catalogs" / "OGSCatalog",
    help="Name of the catalog")
  parser.add_argument(
    "-P", "--polygon", required=False, type=is_polygon,
    default=mplPath(OGS_C.OGS_POLY_REGION, closed=True),
    nargs=OGS_C.ONE_MORECHAR_STR, metavar=OGS_C.EMPTY_STR,
    help="Polygon WKT string to filter events")
  return parser.parse_args()

class DataFile:
  RECORD_EXTRACTOR_LIST : list = [] # TBD in subclasses
  EVENT_EXTRACTOR_LIST : list = [] # TBD in subclasses
  GROUP_PATTERN = re.compile(r"\(\?P<(\w+)>[\[\]\w\d\{\}\-\\\?\+]+\)(\w)*")
  def __init__(self, file_path: Path, start: datetime = datetime.max,
               end: datetime = datetime.min, verbose: bool = False,
               polygon : mplPath = mplPath(OGS_C.OGS_POLY_REGION, closed=True),
               name : Path = DATA_PATH / "catalogs" / "OGSCatalog"):
    self.file_path = file_path
    self.start = start
    self.end = end
    self.polygon : mplPath = polygon
    self.verbose = verbose
    self.name = name
    self.picks = pd.DataFrame(columns=[
      OGS_C.INDEX_STR, OGS_C.TIMESTAMP_STR, OGS_C.PHASE_STR, OGS_C.STATION_STR,
      OGS_C.ERT_STR, OGS_C.NOTES_STR, OGS_C.NETWORK_STR, OGS_C.GROUPS_STR])
    self.events = pd.DataFrame(columns=[
      OGS_C.INDEX_STR, OGS_C.TIMESTAMP_STR, OGS_C.LATITUDE_STR,
      OGS_C.LONGITUDE_STR, OGS_C.DEPTH_STR, OGS_C.NO_STR,
      OGS_C.GAP_STR, OGS_C.DMIN_STR, OGS_C.RMS_STR,
      OGS_C.ERH_STR, OGS_C.ERZ_STR, OGS_C.QM_STR, OGS_C.MAGNITUDE_L_STR,
      OGS_C.MAGNITUDE_D_STR, OGS_C.NOTES_STR,])
    self.RECORD_EXTRACTOR : re.Pattern = re.compile(OGS_C.EMPTY_STR.join(
      list(flatten_list(self.RECORD_EXTRACTOR_LIST)))) # TBD in subclasses
    self.EVENT_EXTRACTOR : re.Pattern = re.compile(OGS_C.EMPTY_STR.join(
      list(flatten_list(self.EVENT_EXTRACTOR_LIST)))) # TBD in subclasses
    self.read()
    self.log()

  def read(self):
    raise NotImplementedError
  DIR_FMT = {
    "year": "{:04}",
    "month": "{:02}",
    "day": "{:02}",
  }
  def log(self):
    name = self.name.stem
    sfx = str(self.file_path.suffix[1:]).upper()
    log = self.name / self.file_path.suffix
    DAYS = np.arange(self.start, self.end + OGS_C.ONE_DAY, OGS_C.ONE_DAY,
                     dtype='datetime64[D]').tolist()
    for day in DAYS:
      if self.file_path.suffix not in [OGS_C.PUN_EXT, OGS_C.TXT_EXT]:
        folder = log / "assignments" / \
                 self.DIR_FMT["year"].format(day.year) / \
                 self.DIR_FMT["month"].format(day.month)
        folder.mkdir(parents=True, exist_ok=True)
      if self.file_path.suffix not in [OGS_C.DAT_EXT]:
        folder = log / "events" / \
                self.DIR_FMT["year"].format(day.year) / \
                self.DIR_FMT["month"].format(day.month)
        folder.mkdir(parents=True, exist_ok=True)
    if not self.events.empty:
      self.events[OGS_C.GROUPS_STR] = self.events[OGS_C.TIMESTAMP_STR].apply(
        lambda x: x.date())
      date = datetime.min
      for date, df in self.events.groupby(OGS_C.GROUPS_STR):
        df.to_csv(log / "events" /
          self.DIR_FMT["year"].format(date.year) /
          self.DIR_FMT["month"].format(date.month) /
          f"{self.DIR_FMT["day"].format(date.day)}.csv", index=False)
      OGS_P.map_plotter(
        domain=OGS_C.OGS_STUDY_REGION,
        x=self.events[OGS_C.LONGITUDE_STR],
        y=self.events[OGS_C.LATITUDE_STR],
        label="OGS Catalog",
        legend=True,
        facecolors="none",
        edgecolors=OGS_C.OGS_BLUE,
        output=f"{sfx}{date.year}{name}Map.png")
      OGS_P.histogram_plotter(
        self.events[OGS_C.DEPTH_STR].dropna(),
        xlabel="Depth (km)",
        ylabel="Number of Events",
        title="OGS Catalog Depths",
        color=OGS_C.OGS_BLUE,
        output=f"{sfx}{date.year}{name}Depths.png",
        legend=True)
      OGS_P.histogram_plotter(
        self.events[OGS_C.ERH_STR].dropna(),
        xlabel="Horizontal Error (km)",
        ylabel="Number of Events",
        title="OGS Catalog Horizontal Errors",
        color=OGS_C.OGS_BLUE,
        output=f"{sfx}{date.year}{name}ERH.png",
        legend=True)
      OGS_P.histogram_plotter(
        self.events[OGS_C.ERZ_STR].dropna(),
        xlabel="Vertical Error (km)",
        ylabel="Number of Events",
        title="OGS Catalog Vertical Errors",
        color=OGS_C.OGS_BLUE,
        output=f"{sfx}{date.year}{name}ERZ.png",
        legend=True)
      OGS_P.day_plotter(
        self.events[OGS_C.GROUPS_STR],
        ylabel="Number of Events",
        title="OGS Catalog Events by Date",
        color=OGS_C.OGS_BLUE,
        output=f"{sfx}{date.year}{name}Date.png",
        grid=True)
      if self.file_path.suffix == OGS_C.TXT_EXT:
        OGS_P.histogram_plotter(
          self.events[OGS_C.MAGNITUDE_L_STR].dropna(),
          xlabel="Local Magnitude (Ml)",
          ylabel="Number of Events",
          title="OGS Catalog Magnitudes",
          color=OGS_C.OGS_BLUE,
          output=f"{sfx}{date.year}{name}Ml.png",
          legend=True)
    if not self.picks.empty:
      self.picks[OGS_C.GROUPS_STR] = self.picks[OGS_C.TIMESTAMP_STR].apply(
        lambda x: x.date())
      for date, df in self.picks.groupby(OGS_C.GROUPS_STR):
        df.to_csv(log / "assignments" /
          self.DIR_FMT["year"].format(date.year) /
          self.DIR_FMT["month"].format(date.month) /
          f"{self.DIR_FMT["day"].format(date.day)}.csv", index=False)
      OGS_P.histogram_plotter(
        self.picks[OGS_C.ERT_STR],
        xlabel="Estimated Reading Time (s)",
        ylabel="Number of Picks",
        title="OGS Catalog Estimated Reading Times",
        color=OGS_C.OGS_BLUE,
        output=f"{sfx}{date.year}{name}ERT.png")

  def debug(self, line, EXTRACTOR_LIST) -> str:
    RECORD_EXTRACTOR_DEBUG = list(reversed(list(it.accumulate(
      EXTRACTOR_LIST[:-1],
      lambda x, y: x + (y if isinstance(y, str) else
                        OGS_C.EMPTY_STR.join(list(flatten_list(y))))))))
    bug = self.GROUP_PATTERN.findall(EXTRACTOR_LIST[0])[0][0]
    for i, extractor in enumerate(RECORD_EXTRACTOR_DEBUG):
      match_extractor = re.match(extractor, line)
      if match_extractor:
        match_group = self.GROUP_PATTERN.findall(RECORD_EXTRACTOR_DEBUG[i - 1])
        match_compare = self.GROUP_PATTERN.findall(extractor)
        bug = match_group[-1][match_group[-1][1] != match_compare[-1][1]]
        break
    return bug

class DataCatalog(DataFile):
  class DataFilePUN(DataFile):
    RECORD_EXTRACTOR_LIST = [
      fr"^1(?P<{OGS_C.DATE_STR}>\d{{6}}[\s\d]\d[\s\d]\d)\s"         # Date
      fr"(?P<{OGS_C.SECONDS_STR}>[\s\d]\d\.\d{{2}})\s"              # Seconds
      # Latitude
      fr"(?P<{OGS_C.LATITUDE_STR}>[\s\d]\d-[\s\d]\d\.\d{{2}})\s{{2}}"
      # Longitude
      fr"(?P<{OGS_C.LONGITUDE_STR}>[\s\d]\d-[\s\d]\d\.\d{{2}})\s{{2}}"
      fr"(?P<{OGS_C.DEPTH_STR}>[\s\d]\d\.\d{{2}})\s{{2}}"           # Depth
      fr"(?P<{OGS_C.MAGNITUDE_D_STR}>[\-\s](\d\.\d{{2}}|\s{{4}}))\s"# Magnitude
      fr"(?P<{OGS_C.NO_STR}>[\s\d]\d)\s"                            # NO
      fr"(?P<{OGS_C.GAP_STR}>[\s\d]{{3}})"                          # GAP
      fr"(?P<{OGS_C.DMIN_STR}>[\s\d]{{2}}\d\.\d)"                   # DMIN
      fr"(?P<{OGS_C.RMS_STR}>[\s\d]\d\.\d{{2}})"                    # RMS
      fr"(?P<{OGS_C.ERH_STR}>([\s\d]{{2}}\d\.\d|\s{{5}}))"          # ERH
      fr"(?P<{OGS_C.ERZ_STR}>([\s\d]{{2}}\d\.\d|\s{{5}}))\s"        # ERZ
      fr"(?P<{OGS_C.QM_STR}>[A-Z][0-9])"                            # QM
    ]
    def __init__(self, file_path: Path, start: datetime = datetime.max,
                 end: datetime = datetime.min, verbose: bool = False,
                 polygon: mplPath = mplPath(
                   OGS_C.OGS_POLY_REGION, closed=True),
                 name: Path = DATA_PATH / "catalogs" / "OGSCatalog"):
      super().__init__(file_path, start, end, verbose, polygon, name)

    def read(self):
      assert self.file_path.suffix == OGS_C.PUN_EXT
      assert self.file_path.exists()
      SOURCE = list()
      event: int = 0
      with open(self.file_path, 'r') as fr: lines = fr.readlines()[1:]
      for line in [l.strip() for l in lines]:
        match = self.RECORD_EXTRACTOR.match(line)
        if match:
          result: dict = match.groupdict()
          result[OGS_C.SECONDS_STR] = \
            td(seconds=float(result[OGS_C.SECONDS_STR]))
          result[OGS_C.DATE_STR] = datetime.strptime(
              result[OGS_C.DATE_STR].replace(OGS_C.SPACE_STR, OGS_C.ZERO_STR),
              OGS_C.DATETIME_FMT[:-2]) + result[OGS_C.SECONDS_STR]
          # We only consider the picks from the date range (if specified)
          if self.start is not None and result[OGS_C.DATE_STR] < self.start:
            continue
          if (self.end is not None and
              result[OGS_C.DATE_STR] >= self.end + OGS_C.ONE_DAY): break
          result[OGS_C.LATITUDE_STR] = result[OGS_C.LATITUDE_STR].replace(
            OGS_C.SPACE_STR, OGS_C.ZERO_STR) \
              if result[OGS_C.LATITUDE_STR] else OGS_C.NONE_STR
          if result[OGS_C.LATITUDE_STR] != OGS_C.NONE_STR:
            splt = result[OGS_C.LATITUDE_STR].split(OGS_C.DASH_STR)
            result[OGS_C.LATITUDE_STR] = float(splt[0]) + float(splt[1]) / 60.
          result[OGS_C.LONGITUDE_STR] = result[OGS_C.LONGITUDE_STR].replace(
            OGS_C.SPACE_STR, OGS_C.ZERO_STR) \
              if result[OGS_C.LONGITUDE_STR] else OGS_C.NONE_STR
          if result[OGS_C.LONGITUDE_STR] != OGS_C.NONE_STR:
            splt = result[OGS_C.LONGITUDE_STR].split(OGS_C.DASH_STR)
            result[OGS_C.LONGITUDE_STR] = float(splt[0]) + float(splt[1]) / 60.
          point = (result[OGS_C.LONGITUDE_STR], result[OGS_C.LATITUDE_STR])
          if not self.polygon.contains_point(point): continue
          result[OGS_C.DEPTH_STR] = float(result[OGS_C.DEPTH_STR]) \
              if result[OGS_C.DEPTH_STR] else OGS_C.NONE_STR
          result[OGS_C.MAGNITUDE_D_STR] = \
            float(result[OGS_C.MAGNITUDE_D_STR].replace(
              OGS_C.SPACE_STR, OGS_C.ZERO_STR))
          result[OGS_C.NO_STR] = int(result[OGS_C.NO_STR].replace(
            OGS_C.SPACE_STR, OGS_C.ZERO_STR)) \
              if result[OGS_C.NO_STR] else OGS_C.NONE_STR
          result[OGS_C.GAP_STR] = int(result[OGS_C.GAP_STR].replace(
            OGS_C.SPACE_STR, OGS_C.ZERO_STR))
          result[OGS_C.DMIN_STR] = float(result[OGS_C.DMIN_STR].replace(
            OGS_C.SPACE_STR, OGS_C.ZERO_STR))
          result[OGS_C.RMS_STR] = float(result[OGS_C.RMS_STR].replace(
            OGS_C.SPACE_STR, OGS_C.ZERO_STR))
          result[OGS_C.ERH_STR] = float(result[OGS_C.ERH_STR].replace(
            OGS_C.SPACE_STR, OGS_C.ZERO_STR))
          result[OGS_C.ERZ_STR] = float(result[OGS_C.ERZ_STR].replace(
            OGS_C.SPACE_STR, OGS_C.ZERO_STR))
          SOURCE.append([
            event + result[OGS_C.DATE_STR].year * OGS_C.MAX_PICKS_YEAR,
            result[OGS_C.DATE_STR], result[OGS_C.LATITUDE_STR],
            result[OGS_C.LONGITUDE_STR], result[OGS_C.DEPTH_STR],
            result[OGS_C.MAGNITUDE_D_STR], result[OGS_C.NO_STR],
            result[OGS_C.GAP_STR], result[OGS_C.DMIN_STR],
            result[OGS_C.RMS_STR], result[OGS_C.ERH_STR],
            result[OGS_C.ERZ_STR], result[OGS_C.QM_STR], None,
            result[OGS_C.DATE_STR].strftime(OGS_C.DATE_FMT)
          ])
          event += 1
        self.debug(line, self.RECORD_EXTRACTOR_LIST)
      self.events = pd.DataFrame(SOURCE, columns=[
        OGS_C.INDEX_STR, OGS_C.TIMESTAMP_STR, OGS_C.LATITUDE_STR,
        OGS_C.LONGITUDE_STR, OGS_C.DEPTH_STR, OGS_C.MAGNITUDE_D_STR,
        OGS_C.NO_STR, OGS_C.GAP_STR, OGS_C.DMIN_STR, OGS_C.RMS_STR,
        OGS_C.ERH_STR, OGS_C.ERZ_STR, OGS_C.QM_STR, OGS_C.NOTES_STR,
        OGS_C.GROUPS_STR
      ]).astype({ OGS_C.INDEX_STR: int})

  class DataFileHPL(DataFile):
    RECORD_EXTRACTOR_LIST = [
      fr"^(?P<{OGS_C.INDEX_STR}>[\d\s]{{6}})\s",                    # Event
      fr"(?P<{OGS_C.STATION_STR}>[A-Z0-9\s]{{4}})\s",               # Station
      fr"([\d\s\.]{{5}})\s",                                        # Unknown
      fr"([\d\s]{{3}})\s",                                          # Unknown
      fr"([\d\s]{{3}})\s",                                          # Unknown
      fr"(?P<{OGS_C.P_ONSET_STR}>[ei?\s]){OGS_C.PWAVE}",            # P Onset
      # P Polarity
      fr"(?P<{OGS_C.P_POLARITY_STR}>[cC\+dD\-\s])",
      fr"(?P<{OGS_C.P_WEIGHT_STR}>[0-4])\s",                        # P Weight
      # P Time [hhmm]
      fr"(?P<{OGS_C.P_TIME_STR}>[\s\d]{{4}})\s",
      # Seconds [ss.ss]
      fr"(?P<{OGS_C.SECONDS_STR}>[\s\d\.]{{5}})\s",
      fr"(?P<A>[\s\d\-\.]{{5}})\s",                                 # Unknown
      fr"(?P<B>[\s\d\-\.]{{5}})\s",                                 # Unknown
      fr"(?P<C>[\s\d\-\.]{{5}})\s",                                 # Unknown
      fr"(?P<D>[\s\d\-\.]{{5}})\s",                                 # Unknown
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
      fr"(?P<J>[\s\d]{{4}})",                                       # Unknown
      fr"(?P<K>[\s\d\-\.\*]{{5}})\s",
      [
        fr"(((?P<{OGS_C.S_ONSET_STR}>[ei\s\?]){OGS_C.SWAVE}\s",     # S Onset
        fr"(?P<{OGS_C.S_WEIGHT_STR}>[0-5\s])\s",                    # S Weight
        fr"(?P<{OGS_C.S_TIME_STR}>[\s\d\.]{{5}})\s",                # S Time
        fr"(?P<P>[\s\d\-\.]{{5}})\s",                               # Unknown
        fr"(?P<Q>[\s\d\-\.]{{5}})\s{{3}}",                          # Unknown
        fr"(?P<R>[\s\d\.]{{3}})\s{{5}})|\s{{33}})\s"                # Unknown
      ],
      fr"(?P<S>[A-Z0-9\s]{{4}})\s{{4}}g[\sg]",                      # Station
    ]
    RECORD_EXTRACTOR = re.compile(OGS_C.EMPTY_STR.join(
      list(flatten_list(RECORD_EXTRACTOR_LIST))))
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
      fr"(?P<{OGS_C.NO_STR}>[\s\d]{{2}})\s",                        # NO
      fr"(?P<{OGS_C.DMIN_STR}>[\s\d]{{2}})\s",                      # DMIN
      fr"(?P<{OGS_C.GAP_STR}>[\s\d]{{3}})\s1\s",                    # GAP
      fr"(?P<{OGS_C.RMS_STR}>[\s\d\.]{{4}})\s",                     # RMS
      fr"(?P<{OGS_C.ERH_STR}>[\s\d\.]{{4}})\s",                     # ERH
      fr"(?P<{OGS_C.ERZ_STR}>[\s\d\.]{{4}})\s",                     # ERZ
      fr"(?P<{OGS_C.QM_STR}>[A-D\s])\s",                            # QM
      fr"(([A-D]/[A-D])|\s{{3}})\s",                                # Unknown
      fr"([\s\d\.]{{4}})\s",                                        # Unknown
      fr"([\s\d]{{2}})\s",                                          # Unknown
      fr"([\s\d]{{2}})",                                            # Unknown
      fr"([\-\s\d\.]{{5}})",                                        # Unknown
      fr"([\s\d\.]{{5}})\s",                                        # Unknown
      fr"([\s\d]{{2}})\s",                                          # Unknown
      fr"([\s\d\.]{{4}})\s",                                        # Unknown
      fr"([\s\d\.]{{4}})\s",                                        # Unknown
      fr"([\s\d]{{2}})\s",                                          # Unknown
      fr"([\s\d\-\.]{{4}})\s",                                      # Unknown
      fr"([\s\d\.]{{4}})",                                          # Unknown
      fr"([\s\d]{{2}})",                                            # Unknown
      fr"([\s\d\.]{{5}})\s",                                        # Unknown
      fr"([\s\d\.]{{4}})\s{{9}}",                                   # Unknown
      fr"(?P<{OGS_C.NOTES_STR}>[\s\d]\d)",                          # Notes
    ]
    LOCATION_EXTRACTOR_LIST = [
      fr"^\^(?P<{OGS_C.LOC_NAME_STR}>[A-Z\s\.']+(\s\([A-Z\-\s]+\))?)"
    ]
    LOCATION_EXTRACTOR = re.compile(OGS_C.EMPTY_STR.join(
      list(flatten_list(LOCATION_EXTRACTOR_LIST))))
    NOTES_EXTRACTOR_LIST = [
      fr"^\*\s+(?P<{OGS_C.NOTES_STR}>.*)"
    ]
    NOTES_EXTRACTOR = re.compile(OGS_C.EMPTY_STR.join(
      list(flatten_list(NOTES_EXTRACTOR_LIST))))
    def __init__(self, file_path: Path, start: datetime = datetime.max,
                 end: datetime = datetime.min, verbose: bool = False,
                 polygon: mplPath = mplPath(OGS_C.OGS_POLY_REGION,
                                            closed=True),
                 name: Path = DATA_PATH / "catalogs" / "OGSCatalog"):
      super().__init__(file_path, start, end, verbose, polygon, name)

    def read(self):
      assert self.file_path.suffix == OGS_C.HPL_EXT
      assert self.file_path.exists()
      SOURCE = list()
      DETECT = list()
      event_notes: str = ""
      event_detect: int = 0
      event_spacetime = (datetime.min, 0, 0, 0)
      with open(self.file_path, 'r') as fr: lines = fr.readlines()
      for line in [l.strip("\n") for l in lines]:
        if event_detect > 0:
          event_detect -= 1
          match = self.RECORD_EXTRACTOR.match(line)
          if match:
            result : dict = match.groupdict()
            if (result[OGS_C.EVENT_LOCALIZATION_STR] != "D" and
                  result[OGS_C.EVENT_TYPE_STR] != "L"):
              # print("WARNING: (HPL) Ignoring line:", line)
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
                print(e)
            result[OGS_C.P_WEIGHT_STR] = int(result[OGS_C.P_WEIGHT_STR])
            result[OGS_C.SECONDS_STR] = td(seconds=float(
              result[OGS_C.SECONDS_STR].replace(OGS_C.SPACE_STR,
                                                OGS_C.ZERO_STR)))
            result[OGS_C.P_TIME_STR] = result[OGS_C.P_TIME_STR].replace(
              OGS_C.SPACE_STR, OGS_C.ZERO_STR)
            date = event_spacetime[0]
            min = td(minutes=int(result[OGS_C.P_TIME_STR][2:]))
            hrs = td(hours=int(result[OGS_C.P_TIME_STR][:2]))
            result[OGS_C.P_TIME_STR] = datetime(
              date.year, date.month, date.day) + hrs + min
            if self.start is not None and result[OGS_C.P_TIME_STR] < self.start:
              event_detect = 0
              continue
            if (self.end is not None and
                result[OGS_C.P_TIME_STR] >= self.end + OGS_C.ONE_DAY): break
            DETECT.append([
              result[OGS_C.INDEX_STR],
              result[OGS_C.P_TIME_STR] + result[OGS_C.SECONDS_STR],
              OGS_C.H71_OFFSET[result[OGS_C.P_WEIGHT_STR]],
              OGS_C.PWAVE, None, result[OGS_C.STATION_STR], None,
              result[OGS_C.P_TIME_STR].strftime(OGS_C.DATE_FMT)])
            if result[OGS_C.S_TIME_STR]:
              result[OGS_C.S_WEIGHT_STR] = int(result[OGS_C.S_WEIGHT_STR])
              result[OGS_C.S_TIME_STR] = td(seconds=float(
                result[OGS_C.S_TIME_STR].replace(OGS_C.SPACE_STR,
                                                  OGS_C.ZERO_STR)))
              timestamp = result[OGS_C.P_TIME_STR] + result[OGS_C.S_TIME_STR]
              DETECT.append([result[OGS_C.INDEX_STR], timestamp,
                             OGS_C.H71_OFFSET[result[OGS_C.S_WEIGHT_STR]],
                             OGS_C.SWAVE, None, result[OGS_C.STATION_STR], 
                             None, timestamp.strftime(OGS_C.DATE_FMT)])
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
              continue
            if (self.end is not None and
                result[OGS_C.DATE_STR] >= self.end + OGS_C.ONE_DAY): break
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
            result[OGS_C.DEPTH_STR] = float(result[OGS_C.DEPTH_STR])\
                if result[OGS_C.DEPTH_STR] else OGS_C.NONE_STR
            event_spacetime = (
              datetime(result[OGS_C.DATE_STR].year,
                       result[OGS_C.DATE_STR].month,
                       result[OGS_C.DATE_STR].day,
                       result[OGS_C.DATE_STR].hour,
                       result[OGS_C.DATE_STR].minute) +
              result[OGS_C.SECONDS_STR],
              result[OGS_C.LONGITUDE_STR], result[OGS_C.LATITUDE_STR],
              result[OGS_C.DEPTH_STR])
            # # Number of Observations
            result[OGS_C.NO_STR] = int(result[OGS_C.NO_STR].replace(
              OGS_C.SPACE_STR, OGS_C.ZERO_STR)) \
                if result[OGS_C.NO_STR] else OGS_C.NONE_STR
            # # Gap
            result[OGS_C.GAP_STR] = int(result[OGS_C.GAP_STR].replace(
              OGS_C.SPACE_STR, OGS_C.ZERO_STR)) \
                if result[OGS_C.GAP_STR] else OGS_C.NONE_STR
            # # DMIN
            result[OGS_C.DMIN_STR] = float(result[OGS_C.DMIN_STR].replace(
              OGS_C.SPACE_STR, OGS_C.ZERO_STR)) \
                if result[OGS_C.DMIN_STR] else OGS_C.NONE_STR
            # # RMS
            result[OGS_C.RMS_STR] = float(result[OGS_C.RMS_STR].replace(
              OGS_C.SPACE_STR, OGS_C.ZERO_STR)) \
                if result[OGS_C.RMS_STR] else OGS_C.NONE_STR
            # # Error Horizontal
            result[OGS_C.ERH_STR] = float(result[OGS_C.ERH_STR].replace(
              OGS_C.SPACE_STR, OGS_C.ZERO_STR)) \
                if result[OGS_C.ERH_STR] else float("NaN")
            # # Error Vertical
            result[OGS_C.ERZ_STR] = float(result[OGS_C.ERZ_STR].replace(
              OGS_C.SPACE_STR, OGS_C.ZERO_STR)) \
                if result[OGS_C.ERZ_STR] else float("NaN")
            # # Quality Metric
            result[OGS_C.QM_STR] = result[OGS_C.QM_STR].strip(OGS_C.SPACE_STR)\
                if result[OGS_C.QM_STR] else OGS_C.NONE_STR
            event_detect = int(result[OGS_C.NOTES_STR])
            SOURCE.append([result[OGS_C.INDEX_STR], *event_spacetime,
                           result[OGS_C.MAGNITUDE_D_STR], result[OGS_C.NO_STR],
                           result[OGS_C.DMIN_STR], result[OGS_C.GAP_STR],
                           result[OGS_C.RMS_STR], result[OGS_C.ERH_STR],
                           result[OGS_C.ERZ_STR], result[OGS_C.QM_STR], None,
                           event_spacetime[0].strftime(OGS_C.DATE_FMT)])
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
        self.debug(line, self.EVENT_EXTRACTOR_LIST if event_detect == 0
                   else self.RECORD_EXTRACTOR_LIST)
      self.picks = pd.DataFrame(DETECT, columns=[
        OGS_C.INDEX_STR, OGS_C.TIMESTAMP_STR, OGS_C.ERT_STR, OGS_C.PHASE_STR,
        OGS_C.NOTES_STR, OGS_C.STATION_STR, OGS_C.NETWORK_STR,
        OGS_C.GROUPS_STR]).astype({ OGS_C.INDEX_STR: int})
      self.events = pd.DataFrame(SOURCE, columns=[
        OGS_C.INDEX_STR, OGS_C.TIMESTAMP_STR, OGS_C.LONGITUDE_STR,
        OGS_C.LATITUDE_STR, OGS_C.DEPTH_STR, OGS_C.MAGNITUDE_D_STR,
        OGS_C.NO_STR, OGS_C.DMIN_STR, OGS_C.GAP_STR, OGS_C.RMS_STR,
        OGS_C.ERH_STR, OGS_C.ERZ_STR, OGS_C.QM_STR, OGS_C.NOTES_STR,
        OGS_C.GROUPS_STR
      ])
      self.events[[OGS_C.LONGITUDE_STR, OGS_C.LATITUDE_STR]].apply(
        lambda x: mplPath(OGS_C.OGS_POLY_REGION, closed=True).contains_point(
          (x[OGS_C.LONGITUDE_STR], x[OGS_C.LATITUDE_STR])), axis=1)
      print(self.events)
      print(self.picks)


  class DataFileDAT(DataFile):
    RECORD_EXTRACTOR_LIST = [
      fr"^(?P<{OGS_C.STATION_STR}>[A-Z0-9\s]{{4}})",                # Station
      fr"(?P<{OGS_C.P_ONSET_STR}>[ei\s\?]){OGS_C.PWAVE}",           # P Onset
      fr"(?P<{OGS_C.P_POLARITY_STR}>[cC\+dD\-\s])",                 # P Polarity
      fr"(?P<{OGS_C.P_WEIGHT_STR}>[0-4\s])",                        # P Weight
      fr"1",                                                        # 1
      fr"(?P<{OGS_C.DATE_STR}>\d{{10}})\s",                         # Date
      fr"(?P<{OGS_C.P_TIME_STR}>[\s\d]{{4}})",                      # P Time
      fr".{{8}}",                                                   # Unknown
      [
        fr"(((?P<{OGS_C.S_TIME_STR}>[\s\d]{{4}})",                  # S Time
        fr"(?P<{OGS_C.S_ONSET_STR}>[ei\s\?]){OGS_C.SWAVE}",         # S Onset
        fr"(?P<{OGS_C.S_POLARITY_STR}>[cC\+dD\-\s])",               # S Polarity
        fr"(?P<{OGS_C.S_WEIGHT_STR}>[0-5\s]))|\s{{8}})"             # S Weight
      ],
      fr"\s{{22}}",                                                 # SPACE
      # Geo Zone
      fr"(?P<{OGS_C.GEO_ZONE_STR}>[{OGS_C.EMPTY_STR.join(
        OGS_C.OGS_GEO_ZONES.keys())}\s])",
      # Event Type
      fr"(?P<{OGS_C.EVENT_TYPE_STR}>[{OGS_C.EMPTY_STR.join(
        OGS_C.OGS_EVENT_TYPES.keys())}\s])",
      fr"(?P<{OGS_C.EVENT_LOCALIZATION_STR}>[D\s])",                # Event Loc
      fr"\s{{5}}",                                                  # SPACE
      fr"(?P<{OGS_C.DURATION_STR}>[\s\d]{{5}})",                    # Duration
      fr"(?P<{OGS_C.INDEX_STR}>[\s\d]{{4}})",                       # Event
      fr""
    ]
    EVENT_EXTRACTOR_LIST = [
      fr"(?P<{OGS_C.EVENT_TYPE_STR}>[{OGS_C.EMPTY_STR.join(
        OGS_C.OGS_EVENT_TYPES.keys())}\s])",
      fr"(?P<{OGS_C.EVENT_LOCALIZATION_STR}>[D\s])",                # Event Loc
      fr"\s{{5}}",                                                  # SPACE
      fr"(?P<{OGS_C.DURATION_STR}>[\s\d]{{5}})",                    # Duration
      fr"(?P<{OGS_C.INDEX_STR}>[\s\d]{{4}})",                       # Event
      fr""
    ]
    def __init__(self, file_path: Path, start: datetime = datetime.max,
                 end: datetime = datetime.min, verbose: bool = False,
                 polygon: mplPath = mplPath(
                   OGS_C.OGS_POLY_REGION, closed=True),
                 name: Path = DATA_PATH / "catalogs" / "OGSCatalog"):
      super().__init__(file_path, start, end, verbose, polygon, name)

    def read(self, level = OGS_C.WARNING_STR):
      assert self.file_path.suffix == OGS_C.DAT_EXT
      assert self.file_path.exists()
      # TODO: Attemp restoration before SHUTDOWN
      DETECT = list()
      with open(self.file_path, 'r') as fr: lines = fr.readlines()
      for line in [l.strip() for l in lines]:
        if self.EVENT_EXTRACTOR.match(line): continue
        match = self.RECORD_EXTRACTOR.match(line)
        if match:
          result: dict = match.groupdict()
          if (result[OGS_C.EVENT_LOCALIZATION_STR] != "D" and
              result[OGS_C.EVENT_TYPE_STR] != OGS_C.SPACE_STR and
              OGS_C.OGS_EVENT_TYPES[result[OGS_C.EVENT_TYPE_STR]] != \
                OGS_C.EVENT_LOCAL_EQ_STR):
            # print("WARNING: (DAT) Ignoring line:", line)
            continue
          # Date
          try:
            if int(result[OGS_C.DATE_STR][-2:]) >= 60:
              # print(notbl_msg.format(value="60", key=DATE_STR, line=line))
              result[OGS_C.DATE_STR] = \
                  datetime.strptime(result[OGS_C.DATE_STR][:-2],
                                    OGS_C.DATETIME_FMT[:-4]) + td(hours=1)
            else:
              result[OGS_C.DATE_STR] = datetime.strptime(
                result[OGS_C.DATE_STR], OGS_C.DATETIME_FMT[:-2])
          except ValueError as e:
            print(e)
            continue
          # We only consider the picks from the date range (if specified)
          if self.start is not None and result[OGS_C.DATE_STR] < self.start:
            continue
          if (self.end is not None and
              result[OGS_C.DATE_STR] >= self.end + OGS_C.ONE_DAY): break
          result[OGS_C.STATION_STR] = \
            result[OGS_C.STATION_STR].strip(OGS_C.SPACE_STR)
          date = result[OGS_C.DATE_STR].strftime(OGS_C.YYMMDD_FMT)
          # P Time
          try:
            result[OGS_C.P_TIME_STR] = result[OGS_C.DATE_STR] + \
              td(seconds=float(result[OGS_C.P_TIME_STR].replace(
                OGS_C.SPACE_STR, OGS_C.ZERO_STR)) / 100.)
          except ValueError as e:
            print(e)
            continue
          # Event
          if result[OGS_C.INDEX_STR]:
            try:
              result[OGS_C.INDEX_STR] = int(result[OGS_C.INDEX_STR].replace(
                OGS_C.SPACE_STR, OGS_C.ZERO_STR)) + \
                  result[OGS_C.DATE_STR].year * OGS_C.MAX_PICKS_YEAR
            except ValueError as e:
              result[OGS_C.INDEX_STR] = None
              print(e)
          DEFAULT_VALUE = 0
          # P Weight
          try:
            if result[OGS_C.P_WEIGHT_STR] == OGS_C.SPACE_STR:
              result[OGS_C.P_WEIGHT_STR] = DEFAULT_VALUE
            else:
              result[OGS_C.P_WEIGHT_STR] = int(result[OGS_C.P_WEIGHT_STR])
          except ValueError as e:
            print(e)
            continue
          DETECT.append([result[OGS_C.INDEX_STR], result[OGS_C.P_TIME_STR],
                         OGS_C.H71_OFFSET[int(result[OGS_C.P_WEIGHT_STR])],
                         OGS_C.PWAVE, None, result[OGS_C.STATION_STR], None,
                         result[OGS_C.P_TIME_STR].strftime(OGS_C.DATE_FMT)])
          # S Type
          if result[OGS_C.S_TIME_STR]:
            # S Weight
            try:
              if result[OGS_C.S_WEIGHT_STR] == OGS_C.SPACE_STR:
                result[OGS_C.S_WEIGHT_STR] = DEFAULT_VALUE
              else:
                result[OGS_C.S_WEIGHT_STR] = int(result[OGS_C.S_WEIGHT_STR])
            except ValueError as e:
              print(e)
              continue
            # S Time
            try:
              result[OGS_C.S_TIME_STR] = result[OGS_C.DATE_STR] + \
                td(seconds=float(result[OGS_C.S_TIME_STR].replace(
                  OGS_C.SPACE_STR, OGS_C.ZERO_STR)) / 100.)
            except ValueError as e:
              print(e)
              continue
            DETECT.append([result[OGS_C.INDEX_STR], result[OGS_C.S_TIME_STR],
                           OGS_C.H71_OFFSET[int(result[OGS_C.S_WEIGHT_STR])],
                           OGS_C.SWAVE, None, result[OGS_C.STATION_STR], None,
                           result[OGS_C.S_TIME_STR].strftime(OGS_C.DATE_FMT)])
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
        if line == OGS_C.EMPTY_STR: continue
        self.debug(line, self.RECORD_EXTRACTOR_LIST)
      self.picks = pd.DataFrame(DETECT, columns=[
        OGS_C.INDEX_STR, OGS_C.TIMESTAMP_STR, OGS_C.ERT_STR, OGS_C.PHASE_STR,
        OGS_C.NOTES_STR, OGS_C.STATION_STR, OGS_C.NETWORK_STR,
        OGS_C.GROUPS_STR]).astype({ OGS_C.INDEX_STR: int})
      self.picks[OGS_C.GROUPS_STR] = self.picks[OGS_C.TIMESTAMP_STR].apply(
        lambda x: x.date())


  class DataFileTXT(DataFile):
    def read(self):
      self.events = pd.read_csv(self.file_path, delimiter=";").rename(columns={
        "t_err": OGS_C.ERT_STR,
        "origin_time(UTC)": OGS_C.TIMESTAMP_STR,
        "lat": OGS_C.LATITUDE_STR,
        "lon": OGS_C.LONGITUDE_STR,
        "depth": OGS_C.DEPTH_STR,
        "gap": OGS_C.GAP_STR,
        "ml": OGS_C.MAGNITUDE_L_STR,
        "md": OGS_C.MAGNITUDE_D_STR,
        "h_err": OGS_C.ERH_STR,
        "v_err": OGS_C.ERZ_STR,
      })
      self.events[OGS_C.LONGITUDE_STR] = \
        self.events[OGS_C.LONGITUDE_STR].replace("-" * 7, "NaN").apply(float)
      self.events[OGS_C.LATITUDE_STR] = \
        self.events[OGS_C.LATITUDE_STR].replace("-" * 7, "NaN").apply(float)
      self.events = self.events[self.events[
        [OGS_C.LONGITUDE_STR, OGS_C.LATITUDE_STR]].apply(
          lambda x: self.polygon.contains_point(
            (x[OGS_C.LONGITUDE_STR], x[OGS_C.LATITUDE_STR])), axis=1)]
      self.events[OGS_C.ERT_STR] = \
        self.events[OGS_C.ERT_STR].replace("-" * 5, "NaN").apply(float)
      self.events[OGS_C.ERH_STR] = \
        self.events[OGS_C.ERH_STR].replace("-" * 5, "NaN").apply(float)
      self.events[OGS_C.ERZ_STR] = \
        self.events[OGS_C.ERZ_STR].replace("-" * 5, "NaN").apply(float)
      self.events[OGS_C.DEPTH_STR] = \
        self.events[OGS_C.DEPTH_STR].replace("-" * 5, "NaN").apply(float)
      self.events[OGS_C.MAGNITUDE_L_STR] = \
        self.events[OGS_C.MAGNITUDE_L_STR].replace("-" * 4, "NaN").apply(float)
      self.events[OGS_C.TIMESTAMP_STR] = \
        pd.to_datetime(self.events[OGS_C.TIMESTAMP_STR])
      self.events[OGS_C.INDEX_STR] = \
        self.events[OGS_C.INDEX_STR].apply(int) + \
          self.events[OGS_C.TIMESTAMP_STR].dt.year * OGS_C.MAX_PICKS_YEAR
      self.events.drop(columns=["event-id"], inplace=True)
      self.events = self.events.astype({ OGS_C.INDEX_STR: int})
      self.events = self.events[
        (self.events[OGS_C.TIMESTAMP_STR].between(
          self.start, self.end + OGS_C.ONE_DAY)) & 
        (self.events["event_type"] != "[suspected explosion]")]

  DATAFILE_TYPES = {
    OGS_C.HPL_EXT: DataFileHPL,
    OGS_C.DAT_EXT: DataFileDAT,
    OGS_C.TXT_EXT: DataFileTXT,
    OGS_C.PUN_EXT: DataFilePUN,
  }
  def __init__(self, args: argparse.Namespace) -> None:
    self.args = args
    self.files : list[DataFile] = list()
    super().__init__(args.name, args.dates[0], args.dates[1],
                     verbose=args.verbose, polygon=args.polygon,
                     name=DATA_PATH / "catalogs" / "OGSCatalog")

  def read(self) -> None:
    for fr in self.args.file:
      ext = Path(fr).suffix
      if ext in self.DATAFILE_TYPES:
        self.files.append(self.DATAFILE_TYPES[ext](
          Path(fr), self.args.dates[0], self.args.dates[1],
          verbose=self.args.verbose, polygon=self.args.polygon,
          name=self.args.name))

  def log(self) -> None:
    # Shut up (literally)
    pass


def main(args: argparse.Namespace) -> None:
  """
  Parse the arguments provided by the user.

  input:
    - args          (argparse.Namespace)

  output:
    - args          (argparse.Namespace)

  errors:
    - None

  notes:

  """
  OGS_Catalog = DataCatalog(args)

if __name__ == "__main__": main(parse_arguments())
import os
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.path import Path as mplPath

from datetime import datetime, timedelta as td

import ogsconstants as OGS_C
import ogsplotter as OGS_P
from ogshpl import DataFileHPL
from ogsdat import DataFileDAT
from ogspun import DataFilePUN
from ogstxt import DataFileTXT

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

class DataCatalog(OGS_C.OGSDataFile):
  DATAFILE_TYPES = {
    OGS_C.HPL_EXT: DataFileHPL,
    OGS_C.DAT_EXT: DataFileDAT,
    OGS_C.TXT_EXT: DataFileTXT,
    OGS_C.PUN_EXT: DataFilePUN,
  }
  def __init__(self, args: argparse.Namespace) -> None:
    self.args = args
    self.files : list[OGS_C.OGSDataFile] = list()
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
    self.EVENTS = pd.DataFrame(columns=[
      OGS_C.INDEX_STR, OGS_C.TIMESTAMP_STR, OGS_C.LATITUDE_STR,
      OGS_C.LONGITUDE_STR, OGS_C.DEPTH_STR, OGS_C.MAGNITUDE_D_STR,
      OGS_C.NO_STR, OGS_C.DMIN_STR, OGS_C.GAP_STR, OGS_C.RMS_STR,
      OGS_C.ERH_STR, OGS_C.ERZ_STR, OGS_C.QM_STR, OGS_C.NOTES_STR,
      OGS_C.GROUPS_STR])
    events = {}
    self.PICKS = pd.DataFrame(columns=[
      OGS_C.INDEX_STR, OGS_C.TIMESTAMP_STR, OGS_C.ERT_STR, OGS_C.PHASE_STR,
      OGS_C.NOTES_STR, OGS_C.STATION_STR, OGS_C.NETWORK_STR,
      OGS_C.GROUPS_STR])
    picks = {}
    # Events
    for f in self.files:
      if not f.events.empty:
        ext = f.filepath.suffix
        if ext == OGS_C.HPL_EXT:
          # MD, Depth
          f.events.drop(
            columns=[
            ],
            inplace=True)
          f.events[OGS_C.GROUPS_STR] = f.events[OGS_C.TIMESTAMP_STR].apply(
            lambda x: x.date())
          if self.EVENTS.empty:
            self.EVENTS = f.events
          else:
            self.EVENTS = pd.merge(self.EVENTS, f.events, how="outer", on=[
                                   OGS_C.TIMESTAMP_STR, OGS_C.GROUPS_STR],)
        elif ext == OGS_C.TXT_EXT:
          f.events.drop(
            columns=[OGS_C.MAGNITUDE_D_STR, OGS_C.DEPTH_STR, OGS_C.NOTES_STR],
            inplace=True)
          f.events[OGS_C.GROUPS_STR] = f.events[OGS_C.TIMESTAMP_STR].apply(
            lambda x: x.date())
          if self.EVENTS.empty:
            self.EVENTS = f.events
          else:
            self.EVENTS = pd.merge(self.EVENTS, f.events, how="outer", on=[
                OGS_C.TIMESTAMP_STR, OGS_C.GROUPS_STR, OGS_C.INDEX_STR,
                OGS_C.ERH_STR])
    self.EVENTS.to_csv("tmpEvents.csv", index=False)
    indexes = self.EVENTS[OGS_C.INDEX_STR].unique().tolist()
    for f in self.files:
      if not f.picks.empty:
        f.picks = f.picks[f.picks[OGS_C.INDEX_STR].isin(indexes)]
        f.picks[OGS_C.GROUPS_STR] = f.picks[OGS_C.TIMESTAMP_STR].apply(
            lambda x: x.date())
        f.picks.drop(
          columns=[OGS_C.NETWORK_STR, OGS_C.NOTES_STR],
          inplace=True)
        print(f.filepath)
        print(f.picks)
        if self.PICKS.empty:
          self.PICKS = f.picks
        else:
          self.PICKS = pd.merge(self.PICKS, f.picks, how="left", on=[
            OGS_C.INDEX_STR, OGS_C.TIMESTAMP_STR, OGS_C.STATION_STR,
            OGS_C.PHASE_STR])
    print(self.PICKS)
    self.PICKS.to_csv("tmpPicks.csv", index=False)


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
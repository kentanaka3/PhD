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

def is_polygon(points: str) -> mplPath:
  return mplPath(points, closed=True)

def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Parse OGS Manual Catalogs")
  parser.add_argument(
    "-x", "--ext", default=OGS_C.ALL_WILDCHAR_STR, type=str,
    nargs=OGS_C.ONE_MORECHAR_STR, metavar=OGS_C.EMPTY_STR,
    help="File extension to process")
  parser.add_argument(
    '-v', "--verbose", action='store_true', default=False,
    help="Enable verbose output")
  path_group = parser.add_mutually_exclusive_group(required=True)
  path_group.add_argument(
    '-d', "--directory", required=False, type=OGS_C.is_dir_path, default=None,
    help="Base directory for data files.")
  path_group.add_argument(
    '-f', "--file", required=False, type=OGS_C.is_file_path, default=None,
    nargs=OGS_C.ONE_MORECHAR_STR, metavar=OGS_C.EMPTY_STR,
    help="Base file for data files.")
  date_group = parser.add_mutually_exclusive_group(required=False)
  date_group.add_argument(
    '-D', "--dates", required=False, metavar=OGS_C.DATE_STD,
    type=OGS_C.is_date, nargs=2, action=OGS_C.SortDatesAction,
    default=[datetime.strptime("240320", OGS_C.YYMMDD_FMT),
             datetime.strptime("240620", OGS_C.YYMMDD_FMT)],
    help="Specify the beginning and ending (inclusive) Gregorian date " \
         "(YYMMDD) range to work with.")
  date_group.add_argument(
    '-J', "--julian", required=False, metavar=OGS_C.DATE_STD,
    action=OGS_C.SortDatesAction, type=OGS_C.is_julian, default=None, nargs=2,
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
    super().__init__(
      args.name, args.dates[0], args.dates[1], verbose=args.verbose,
      polygon=args.polygon, output=DATA_PATH / "catalogs" / "OGSCatalog")

  def read(self) -> None:
    if self.args.directory is None:
      for fr in self.args.file:
        ext = Path(fr).suffix
        if ext in self.DATAFILE_TYPES:
          self.files.append(self.DATAFILE_TYPES[ext](
            fr, self.args.dates[0], self.args.dates[1],
            verbose=self.args.verbose, polygon=self.args.polygon,
            output=self.args.name))
    else:
      for ext in self.args.ext:
        files = list(self.args.directory.rglob(f"*{ext}"))
        if len(files) == 0 and self.args.verbose:
          print(f"No *{ext} files found in {self.args.directory}")
        for fr in files:
          if fr.suffix in self.DATAFILE_TYPES:
            self.files.append(self.DATAFILE_TYPES[fr.suffix](
              fr, self.args.dates[0], self.args.dates[1],
              verbose=self.args.verbose, polygon=self.args.polygon,
              output=self.args.name))
    for f in self.files:
      f.read()
      f.log()

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
  OGS_Catalog.read()

if __name__ == "__main__": main(parse_arguments())
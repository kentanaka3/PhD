import argparse
import pandas as pd
from pathlib import Path
from matplotlib.path import Path as mplPath

from datetime import datetime

import ogsconstants as OGS_C
from ogshpl import DataFileHPL
from ogsdat import DataFileDAT
from ogspun import DataFilePUN
from ogstxt import DataFileTXT

DATA_PATH = Path(__file__).parent.parent.parent

def is_polygon(points: str) -> mplPath:
  return mplPath(points, closed=True)

def parse_arguments() -> argparse.Namespace:
  """
  Parse the arguments provided by the user.
  input:
    - None
    output:
    - args          (argparse.Namespace)
  errors:
    - None
  notes:
    - None
  """
  parser = argparse.ArgumentParser(description="Parse OGS Manual Catalogs")
  parser.add_argument(
    "-m", "--merge", action='store_true', default=False,
    help="Merge all data files into a single catalog")
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
    help="Polygon string to filter events")
  return parser.parse_args()

class DataCatalog(OGS_C.OGSDataFile):
  """
  Data catalog class for managing OGS data files. In order to add a new data
  file type, simply create a new class that inherits from OGSDataFile and add
  it to the DATAFILE_TYPES dictionary below.
  """
  DATAFILE_TYPES = {
    OGS_C.HPL_EXT: DataFileHPL, # (Highly recommended) Hypocenter information
    OGS_C.DAT_EXT: DataFileDAT, # (Recommended) Picks information
    OGS_C.TXT_EXT: DataFileTXT, # Local Magnitude information
    OGS_C.PUN_EXT: DataFilePUN, # Events
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

  def merge_events(self) -> None:
    print("Merging data files...")
    for f in self.files:
      if self.EVENTS.empty:
        f.get("EVENTS")
        self.EVENTS = pd.concat([self.EVENTS, f.EVENTS], ignore_index=True)
      elif f.filepath.suffix == OGS_C.TXT_EXT:
        """
        IDX_EVENTS_STR, TIME_STR, LATITUDE_STR, LONGITUDE_STR, DEPTH_STR,
        GAP_STR, ERZ_STR, ERH_STR, GROUPS_STR, NO_STR,
        NUMBER_P_PICKS_STR, NUMBER_S_PICKS_STR, NUMBER_P_AND_S_PICKS_STR,
        ML_STR, ML_MEDIAN_STR, ML_UNC_STR, ML_STATIONS_STR
        """
        print(f.EVENTS)
        self.EVENTS = pd.merge(
          self.EVENTS[[OGS_C.TIME_STR,
            OGS_C.LATITUDE_STR,
            OGS_C.LONGITUDE_STR,
            OGS_C.DEPTH_STR,
            OGS_C.GAP_STR,
            OGS_C.ERZ_STR,
            OGS_C.ERH_STR,
            OGS_C.GROUPS_STR,
            OGS_C.NO_STR,
            OGS_C.NUMBER_P_PICKS_STR,
            OGS_C.NUMBER_S_PICKS_STR,
            OGS_C.NUMBER_P_AND_S_PICKS_STR,
            OGS_C.ML_MEDIAN_STR,
            OGS_C.ML_UNC_STR,
            OGS_C.ML_STATIONS_STR
          ]],
          f.EVENTS[[OGS_C.TIME_STR,
            OGS_C.IDX_EVENTS_STR,
            OGS_C.MAGNITUDE_L_STR,
          ]],
          how="outer",
          on=[OGS_C.TIME_STR])
      elif f.filepath.suffix == OGS_C.PUN_EXT:
        self.EVENTS = pd.merge(
          self.EVENTS,
          f.EVENTS,
          how="outer",
          on=[OGS_C.TIME_STR, OGS_C.LATITUDE_STR,
              OGS_C.LONGITUDE_STR, OGS_C.DEPTH_STR])
    
    # Optimization: Vectorized aggregation instead of iterrows() loops
    # This replaces nested loops with efficient groupby operations
    
    # Count P and S picks per event using groupby + pivot
    if not self.PICKS.empty:
      phase_counts = self.PICKS.groupby(
        [OGS_C.IDX_PICKS_STR, OGS_C.PHASE_STR]
      ).size().unstack(fill_value=0)
      
      # Map phase counts to events
      for phase, column in [(OGS_C.PWAVE, OGS_C.NUMBER_P_PICKS_STR),
                            (OGS_C.SWAVE, OGS_C.NUMBER_S_PICKS_STR)]:
        if phase in phase_counts.columns:
          self.EVENTS[column] = self.EVENTS[OGS_C.IDX_EVENTS_STR].map(
            phase_counts[phase]
          ).fillna(0).astype(int)
        else:
          self.EVENTS[column] = 0
      
      # Count stations with both P and S picks per event
      station_phase_counts = self.PICKS.groupby(
        [OGS_C.IDX_PICKS_STR, OGS_C.STATION_STR]
      )[OGS_C.PHASE_STR].nunique()
      
      stations_with_both = station_phase_counts[station_phase_counts >= 2].groupby(
        level=0
      ).size()
      
      self.EVENTS[OGS_C.NUMBER_P_AND_S_PICKS_STR] = self.EVENTS[
        OGS_C.IDX_EVENTS_STR
      ].map(stations_with_both).fillna(0).astype(int)
    else:
      self.EVENTS[OGS_C.NUMBER_P_PICKS_STR] = 0
      self.EVENTS[OGS_C.NUMBER_S_PICKS_STR] = 0
      self.EVENTS[OGS_C.NUMBER_P_AND_S_PICKS_STR] = 0
    
    self.get("events")

  def merge_picks(self) -> None:
    for f in self.files:
      if self.PICKS.empty:
        f.get("PICKS")
        self.PICKS = pd.concat([self.PICKS, f.PICKS], ignore_index=True)
    self.get("picks")

  def merge(self) -> None:
    self.merge_picks()
    self.merge_events()
    print(self.filepath)
    self.filepath = Path(self.filepath.__str__() + ".all")
    self.log()

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
  if args.merge:
    OGS_Catalog.merge()

if __name__ == "__main__": main(parse_arguments())
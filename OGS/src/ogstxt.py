import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta as td
import pandas as pd

import ogsconstants as OGS_C

DATA_PATH = Path(__file__).parent.parent.parent

def parse_arguments():
  parser = argparse.ArgumentParser(description="Run OGS HPL quality checks")
  parser.add_argument("-f", "--file", type=Path, required=True,
                      help="Path to the input file")
  parser.add_argument(
    '-D', "--dates", required=False, metavar=OGS_C.DATE_STD,
    type=OGS_C.is_date, nargs=2, action=OGS_C.SortDatesAction,
    default=[datetime.strptime("240320", OGS_C.YYMMDD_FMT),
              datetime.strptime("240620", OGS_C.YYMMDD_FMT)],
    help="Specify the beginning and ending (inclusive) Gregorian date " \
          "(YYMMDD) range to work with.")
  return parser.parse_args()

class DataFileTXT(OGS_C.OGSDataFile):
    def read(self):
      self.events = pd.read_csv(self.filepath, delimiter=";").rename(columns={
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
      self.events[OGS_C.MAGNITUDE_D_STR] = \
        self.events[OGS_C.MAGNITUDE_D_STR].replace("-" * 4, "NaN").apply(float)
      self.events[OGS_C.TIMESTAMP_STR] = \
        pd.to_datetime(self.events[OGS_C.TIMESTAMP_STR])
      self.events[OGS_C.INDEX_STR] = \
        self.events[OGS_C.INDEX_STR].apply(int) + \
          self.events[OGS_C.TIMESTAMP_STR].dt.year * OGS_C.MAX_PICKS_YEAR
      self.events[OGS_C.GROUPS_STR] = \
        self.events[OGS_C.TIMESTAMP_STR].dt.date
      self.events[OGS_C.NOTES_STR] = None
      self.events.drop(columns=["event-id"], inplace=True)
      self.events = self.events.astype({ OGS_C.INDEX_STR: int})
      self.events = self.events[
        (self.events[OGS_C.TIMESTAMP_STR].between(
          self.start, self.end + OGS_C.ONE_DAY)) &
        (self.events["event_type"] != "[suspected explosion]")]

def main(args):
  datafile = DataFileTXT(args.file, args.dates[0], args.dates[1])
  datafile.read()
  datafile.log()

if __name__ == "__main__": main(parse_arguments())
import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

import ogsconstants as OGS_C

def parse_arguments():
  parser = argparse.ArgumentParser(description="Train OGS models")
  date_group = parser.add_mutually_exclusive_group(required=False)
  date_group.add_argument(
    '-D', "--dates", required=False, metavar=OGS_C.DATE_STD,
    type=OGS_C.is_date, nargs=2, action=OGS_C.SortDatesAction,
    default=[datetime.strptime("240320", OGS_C.YYMMDD_FMT),
             datetime.strptime("240620", OGS_C.YYMMDD_FMT)],
    help="Specify the beginning and ending (inclusive) Gregorian date " \
         "(YYMMDD) range to work with.")
  parser.add_argument("-P", "--picks", type=Path, required=True,
                      help="Path to the picks directory")
  parser.add_argument("-W", "--waveforms", type=Path, required=True,
                      help="Path to the waveforms directory")
  parser.add_argument("-d", "--debug", action="store_true",
                      help="Enable debug mode")
  return parser.parse_args()

class OGSTrainer:
  def __init__(self, args):
    self.picks: Path = args.picks
    start, end = args.dates
    self.waveforms = OGS_C.waveforms(args.waveforms, start, end)
    self.stations = {
      station.split(".")[1] : station for date in self.waveforms.values()
        for station in date.keys()
    }
    add = {}
    for key, val in self.stations.items():
      if len(key) > 4:
        add[key[:4]] = val
    self.stations.update(add)
    print(self.stations)
    self.debug = args.debug

  def train(self):
    # Training logic here
    if self.debug: print("Debug mode is enabled.")
    for filepath in self.picks.glob("*"):
      print(f"Filepath: {filepath}")
      df = pd.read_parquet(filepath)
      print(df.head())

def main(args):
  trainer = OGSTrainer(args)
  trainer.train()


if __name__ == "__main__": main(parse_arguments())
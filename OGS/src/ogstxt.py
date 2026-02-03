import argparse
from pathlib import Path
from unittest import result
from obspy import UTCDateTime
from datetime import datetime
import pandas as pd

import ogsconstants as OGS_C
from ogsdatafile import OGSDataFile

DATA_PATH = Path(__file__).parent.parent.parent

def parse_arguments():
  parser = argparse.ArgumentParser(description="Run OGS TXT quality checks")
  parser.add_argument(
    "-f", "--file", type=Path, required=True, nargs=OGS_C.ONE_MORECHAR_STR,
    help="Path to the input file")
  parser.add_argument(
    '-D', "--dates", required=False, metavar=OGS_C.DATE_STD,
    type=OGS_C.is_date, nargs=2, action=OGS_C.SortDatesAction,
    default=[datetime.strptime("20240320", OGS_C.YYMMDD_FMT),
             datetime.strptime("20240620", OGS_C.YYMMDD_FMT)],
    help="Specify the beginning and ending (inclusive) Gregorian date " \
          "(YYYYMMDD) range to work with.")
  parser.add_argument(
    '-v', "--verbose", action='store_true', default=False,
    help="Enable verbose output")
  return parser.parse_args()

class DataFileTXT(OGSDataFile):
  RECORD_EXTRACTOR_LIST = [
    fr"^(?P<{OGS_C.INDEX_STR}>\d{{5}})\s"                           # Index
    fr"\d{{4}}_\d{{5}}\s"                                           # Unused
    fr"(?P<{OGS_C.TIME_STR}>\d{{4}}-\d{{2}}-\d{{2}}T"               # Date
    fr"\d{{2}}:\d{{2}}:\d{{2}}\.\d{{3}})\s"                         # Time
    fr"(?P<{OGS_C.ERT_STR}>([\s\d]\d\.\d{{2}}|\-{{5}}))\s"          # ERT
    fr"(?P<{OGS_C.LATITUDE_STR}>(\d{{2}}\.\d{{4}}|\-{{7}}))\s"      # Latitude
    fr"(?P<{OGS_C.LONGITUDE_STR}>(\d{{2}}\.\d{{4}}|\-{{7}}))\s"     # Longitude
    fr"(?P<{OGS_C.ERH_STR}>([\s\d]{{2}}\d\.\d|\-{{5}}))\s"          # ERH
    fr"(?P<{OGS_C.DEPTH_STR}>([\s\d]{{2}}\d\.\d|\-{{5}}))\s"        # Depth
    fr"(?P<{OGS_C.ERZ_STR}>([\s\d]{{2}}\d\.\d|\-{{5}}))\s"          # ERZ
    fr"(?P<{OGS_C.GAP_STR}>([\s\d\-]{{3}}))\s"                      # GAP
    fr"(?P<{OGS_C.MAGNITUDE_L_STR}>([\-\s\d]\d\.\d|\-{{4}}))\s"     # ML
    fr"(?P<{OGS_C.MAGNITUDE_D_STR}>([\-\s\d]\d\.\d|\-{{4}}))\s"     # MD
    fr"(?P<{OGS_C.LOC_NAME_STR}>[\w\s\(\)]+)\s"                     # Place
    fr"(?P<{OGS_C.EVENT_TYPE_STR}>\[.*\])$"                         # Unused
  ]
  def read(self):
    assert self.input.exists(), \
      f"File {self.input} does not exist"
    assert self.input.suffix == OGS_C.TXT_EXT, \
      f"File extension must be {OGS_C.TXT_EXT}"
    SOURCE = list()
    with open(self.input, 'r') as fr: lines = fr.readlines()[1:]
    self.logger.info(f"Reading TXT file: {self.input}")
    for line in [l.strip() for l in lines]:
      match = self.RECORD_EXTRACTOR.match(line)
      if match:
        result: dict = match.groupdict()
        result[OGS_C.TIME_STR] = datetime.fromisoformat(
          result[OGS_C.TIME_STR])
        if self.start is not None and result[OGS_C.TIME_STR] < self.start:
          self.logger.debug(f"Skipping event before start date: {self.start}")
          self.logger.debug(line)
          continue
        if (self.end is not None and
            result[OGS_C.TIME_STR] > self.end + OGS_C.ONE_DAY):
          self.logger.debug("Stopping read at event after end date:"
                            f"{self.end}")
          self.logger.debug(line)
          break
        SOURCE.append([result[OGS_C.INDEX_STR],
                       result[OGS_C.TIME_STR],
                       result[OGS_C.ERT_STR],
                       result[OGS_C.LATITUDE_STR],
                       result[OGS_C.LONGITUDE_STR],
                       result[OGS_C.ERH_STR],
                       result[OGS_C.DEPTH_STR],
                       result[OGS_C.ERZ_STR],
                       result[OGS_C.GAP_STR],
                       result[OGS_C.MAGNITUDE_L_STR],
                       result[OGS_C.MAGNITUDE_D_STR],
                       result[OGS_C.LOC_NAME_STR],
                       result[OGS_C.EVENT_TYPE_STR]])
    self.EVENTS = pd.DataFrame(SOURCE, columns=[
      OGS_C.INDEX_STR,
      OGS_C.TIME_STR,
      OGS_C.ERT_STR,
      OGS_C.LATITUDE_STR,
      OGS_C.LONGITUDE_STR,
      OGS_C.ERH_STR,
      OGS_C.DEPTH_STR,
      OGS_C.ERZ_STR,
      OGS_C.GAP_STR,
      OGS_C.MAGNITUDE_L_STR,
      OGS_C.MAGNITUDE_D_STR,
      OGS_C.LOC_NAME_STR,
      OGS_C.EVENT_TYPE_STR,
    ])
    if self.EVENTS.empty:
      self.logger.warning(f"No valid TXT records found in {self.input}")
      return
    self.EVENTS[OGS_C.INDEX_STR] = \
      self.EVENTS[OGS_C.INDEX_STR].apply(int) + \
        self.EVENTS[OGS_C.TIME_STR].dt.year * OGS_C.MAX_PICKS_YEAR # type: ignore
    self.EVENTS[OGS_C.GROUPS_STR] = \
      self.EVENTS[OGS_C.TIME_STR].dt.date # type: ignore
    self.EVENTS[OGS_C.ERT_STR] = \
      self.EVENTS[OGS_C.ERT_STR].replace("-" * 5, "NaN").apply(float)
    self.EVENTS[OGS_C.LONGITUDE_STR] = \
      self.EVENTS[OGS_C.LONGITUDE_STR].replace("-" * 7, "NaN").apply(float)
    self.EVENTS[OGS_C.LATITUDE_STR] = \
      self.EVENTS[OGS_C.LATITUDE_STR].replace("-" * 7, "NaN").apply(float)
    self.EVENTS = self.EVENTS[self.EVENTS[
      [OGS_C.LONGITUDE_STR, OGS_C.LATITUDE_STR]].apply(
        lambda x: self.polygon.contains_point(
          (x[OGS_C.LONGITUDE_STR], x[OGS_C.LATITUDE_STR])), axis=1)]
    self.EVENTS[OGS_C.ERH_STR] = \
      self.EVENTS[OGS_C.ERH_STR].replace("-" * 5, "NaN").apply(float)
    self.EVENTS[OGS_C.DEPTH_STR] = \
      self.EVENTS[OGS_C.DEPTH_STR].replace("-" * 5, "NaN").apply(float)
    self.EVENTS[OGS_C.ERZ_STR] = \
      self.EVENTS[OGS_C.ERZ_STR].replace("-" * 5, "NaN").apply(float)
    self.EVENTS[OGS_C.GAP_STR] = \
      self.EVENTS[OGS_C.GAP_STR].replace("-" * 3, "NaN").apply(float)
    self.EVENTS[OGS_C.MAGNITUDE_L_STR] = \
      self.EVENTS[OGS_C.MAGNITUDE_L_STR].replace("-" * 4, "NaN").apply(float)
    self.EVENTS[OGS_C.MAGNITUDE_D_STR] = \
      self.EVENTS[OGS_C.MAGNITUDE_D_STR].replace("-" * 4, "NaN").apply(float)
    self.EVENTS[OGS_C.TIME_STR] = \
      pd.to_datetime(self.EVENTS[OGS_C.TIME_STR])
    self.EVENTS[OGS_C.NOTES_STR] = None
    self.EVENTS = self.EVENTS.astype({ OGS_C.INDEX_STR: int})
    self.EVENTS = self.EVENTS[(
      self.EVENTS[OGS_C.TIME_STR].between(
        self.start,
        self.end + OGS_C.ONE_DAY
      )
    ) & (
      self.EVENTS[OGS_C.EVENT_TYPE_STR] != "[suspected explosion]"
    )]
    for date, df in self.EVENTS.groupby(OGS_C.GROUPS_STR):
      self.events[UTCDateTime(date).date] = df

def main(args):
  for file in args.file:
    datafile = DataFileTXT(file, args.dates[0], args.dates[1],
                           verbose=args.verbose)
    datafile.read()
    datafile.log()

if __name__ == "__main__": main(parse_arguments())
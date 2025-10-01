import argparse
from pathlib import Path
from obspy import UTCDateTime
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

class DataFilePUN(OGS_C.OGSDataFile):
  RECORD_EXTRACTOR_LIST = [
    fr"^1(?P<{OGS_C.DATE_STR}>\d{{6}}[\s\d]\d[\s\d]\d)\s"           # Date
    fr"(?P<{OGS_C.SECONDS_STR}>[\s\d]\d\.\d{{2}})\s"                # Seconds
    # Latitude
    fr"(?P<{OGS_C.LATITUDE_STR}>[\s\d]\d-[\s\d]\d\.\d{{2}})\s{{2}}"
    # Longitude
    fr"(?P<{OGS_C.LONGITUDE_STR}>[\s\d]\d-[\s\d]\d\.\d{{2}})\s{{2}}"
    fr"(?P<{OGS_C.DEPTH_STR}>[\s\d]\d\.\d{{2}})\s{{2}}"             # Depth
    fr"(?P<{OGS_C.MAGNITUDE_D_STR}>[\-\s](\d\.\d{{2}}|\s{{4}}))\s"  # Magnitude
    fr"(?P<{OGS_C.NO_STR}>[\s\d]\d)\s"                              # NO
    fr"(?P<{OGS_C.GAP_STR}>[\s\d]{{3}})"                            # GAP
    fr"(?P<{OGS_C.DMIN_STR}>[\s\d]{{2}}\d\.\d)"                     # DMIN
    fr"(?P<{OGS_C.RMS_STR}>[\s\d]\d\.\d{{2}})"                      # RMS
    fr"(?P<{OGS_C.ERH_STR}>([\s\d]{{2}}\d\.\d|\s{{5}}))"            # ERH
    fr"(?P<{OGS_C.ERZ_STR}>([\s\d]{{2}}\d\.\d|\s{{5}}))\s"          # ERZ
    fr"(?P<{OGS_C.QM_STR}>[A-Z][0-9])"                              # QM
  ]
  def read(self):
    assert self.filepath.suffix == OGS_C.PUN_EXT
    assert self.filepath.exists()
    SOURCE = list()
    event: int = 0
    with open(self.filepath, 'r') as fr: lines = fr.readlines()[1:]
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
        result[OGS_C.DEPTH_STR] = float(result[OGS_C.DEPTH_STR]) \
            if result[OGS_C.DEPTH_STR] else OGS_C.NONE_STR
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
    self.EVENTS = pd.DataFrame(SOURCE, columns=[
      OGS_C.INDEX_STR, OGS_C.TIMESTAMP_STR, OGS_C.LATITUDE_STR,
      OGS_C.LONGITUDE_STR, OGS_C.DEPTH_STR, OGS_C.MAGNITUDE_D_STR,
      OGS_C.NO_STR, OGS_C.GAP_STR, OGS_C.DMIN_STR, OGS_C.RMS_STR,
      OGS_C.ERH_STR, OGS_C.ERZ_STR, OGS_C.QM_STR, OGS_C.NOTES_STR,
      OGS_C.GROUPS_STR
    ]).astype({ OGS_C.INDEX_STR: int})
    self.EVENTS[OGS_C.MAGNITUDE_D_STR] = \
      self.EVENTS[OGS_C.MAGNITUDE_D_STR].replace(" " * 5, "NaN").apply(float)
    self.EVENTS = self.EVENTS[self.EVENTS[
      [OGS_C.LONGITUDE_STR, OGS_C.LATITUDE_STR]].apply(
        lambda x: self.polygon.contains_point(
          (x[OGS_C.LONGITUDE_STR], x[OGS_C.LATITUDE_STR])), axis=1)]
    for date, df in self.EVENTS.groupby(OGS_C.GROUPS_STR):
      self.events[UTCDateTime(date).date] = df


def main(args):
  datafile = DataFilePUN(args.file, args.dates[0], args.dates[1])
  datafile.read()
  datafile.log()

if __name__ == "__main__": main(parse_arguments())
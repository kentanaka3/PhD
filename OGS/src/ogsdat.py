import re
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta as td

import ogsconstants as OGS_C

DATA_PATH = Path(__file__).parent.parent.parent

def is_date(string: str) -> datetime:
  return datetime.strptime(string, OGS_C.YYMMDD_FMT)

class SortDatesAction(argparse.Action):
  def __call__(self, parser, namespace, values, option_string=None):
    setattr(namespace, self.dest, sorted(values)) # type: ignore

def parse_arguments():
  parser = argparse.ArgumentParser(description="Run OGS HPL quality checks")
  parser.add_argument("-f", "--file", type=Path, required=True,
                      help="Path to the input file")
  parser.add_argument(
    '-D', "--dates", required=False, metavar=OGS_C.DATE_STD, type=is_date,
  nargs=2, action=SortDatesAction,
  default=[datetime.strptime("240320", OGS_C.YYMMDD_FMT),
           datetime.strptime("240620", OGS_C.YYMMDD_FMT)],
  help="Specify the beginning and ending (inclusive) Gregorian date " \
        "(YYMMDD) range to work with.")
  return parser.parse_args()

class DataFileDAT(OGS_C.OGSDataFile):
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
  def read(self):
    assert self.filepath.suffix == OGS_C.DAT_EXT
    assert self.filepath.exists()
    # TODO: Attemp restoration before SHUTDOWN
    DETECT = list()
    with open(self.filepath, 'r') as fr: lines = fr.readlines()
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
                        int(result[OGS_C.P_WEIGHT_STR]),
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
                          int(result[OGS_C.S_WEIGHT_STR]),
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
      if re.match(r"1\s+D$", line): continue
      if line == OGS_C.EMPTY_STR: continue
      self.debug(line, self.RECORD_EXTRACTOR_LIST)
    self.picks = pd.DataFrame(DETECT, columns=[
      OGS_C.INDEX_STR, OGS_C.TIMESTAMP_STR, OGS_C.ERT_STR, OGS_C.PHASE_STR,
      OGS_C.NOTES_STR, OGS_C.STATION_STR, OGS_C.NETWORK_STR,
      OGS_C.GROUPS_STR]).astype({ OGS_C.INDEX_STR: int})
    self.picks[OGS_C.GROUPS_STR] = self.picks[OGS_C.TIMESTAMP_STR].apply(
      lambda x: x.date())


def main(args):
  datafile = DataFileDAT(args.file, args.dates[0], args.dates[1])
  datafile.read()
  datafile.log()

if __name__ == "__main__": main(parse_arguments())
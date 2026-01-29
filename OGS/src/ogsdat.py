import re
import argparse
import pandas as pd
from pathlib import Path
from obspy import UTCDateTime
from datetime import datetime, timedelta as td

import ogsconstants as OGS_C
from ogsdatafile import OGSDataFile

DATA_PATH = Path(__file__).parent.parent.parent

def parse_arguments():
  parser = argparse.ArgumentParser(description="Run OGS HPL quality checks")
  parser.add_argument(
    "-f", "--file", type=Path, required=True, nargs=OGS_C.ONE_MORECHAR_STR,
    help="Path to the input file")
  parser.add_argument(
    '-D', "--dates", required=False, metavar=OGS_C.DATE_STD,
    type=OGS_C.is_date, nargs=2, action=OGS_C.SortDatesAction,
    default=[datetime.min, datetime.max - OGS_C.ONE_DAY],
    help="Specify the beginning and ending (inclusive) Gregorian date " \
         "(YYMMDD) range to work with.")
  parser.add_argument(
    '-v', "--verbose", default=False, action='store_true', required=False,
    help="Enable verbose output")
  return parser.parse_args()

class DataFileDAT(OGSDataFile):
  RECORD_EXTRACTOR_LIST = [
    fr"^(?P<{OGS_C.STATION_STR}>[A-Z0-9\s]{{4}})",                # Station
    fr"(?P<{OGS_C.P_ONSET_STR}>[ei\s\?]){OGS_C.PWAVE}",           # P Onset
    fr"(?P<{OGS_C.P_POLARITY_STR}>[cC\+dD\-\s])",                 # P Polarity
    fr"(?P<{OGS_C.P_WEIGHT_STR}>[0-4\s])",                        # P Weight
    fr"1",                                                        # 1
    fr"(?P<{OGS_C.DATE_STR}>\d{{10}})[\s0]",                         # Date
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
    assert self.input.exists(), \
      f"File {self.input} does not exist"
    assert self.input.suffix == OGS_C.DAT_EXT, \
      f"File extension must be {OGS_C.DAT_EXT}"
    # TODO: Attemp restoration before SHUTDOWN
    DETECT = list()
    with open(self.input, 'r') as fr: lines = fr.readlines()
    self.logger.info(f"Reading DAT file: {self.input}")
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
          self.logger.debug(f"Skipping pick before start date: {self.start}")
          self.logger.debug(line)
          continue
        if (self.end is not None and
            result[OGS_C.DATE_STR] >= self.end + OGS_C.ONE_DAY):
          self.logger.debug(f"Stopping read at pick after end date: {self.end}")
          self.logger.debug(line)
          break
        result[OGS_C.STATION_STR] = \
          result[OGS_C.STATION_STR].strip(OGS_C.SPACE_STR)
        date = result[OGS_C.DATE_STR].strftime(OGS_C.YYMMDD_FMT)
        # P Time
        try:
          result[OGS_C.P_TIME_STR] = result[OGS_C.DATE_STR] + \
            td(seconds=float(result[OGS_C.P_TIME_STR].replace(
              OGS_C.SPACE_STR, OGS_C.ZERO_STR)) / 100.)
        except ValueError as e:
          self.logger.error(e)
          continue
        # Event
        if result[OGS_C.INDEX_STR]:
          try:
            result[OGS_C.INDEX_STR] = int(result[OGS_C.INDEX_STR].replace(
              OGS_C.SPACE_STR, OGS_C.ZERO_STR)) + \
                result[OGS_C.DATE_STR].year * OGS_C.MAX_PICKS_YEAR
          except ValueError as e:
            result[OGS_C.INDEX_STR] = None
            self.logger.error(e)
        DEFAULT_VALUE = 0
        # P Weight
        try:
          if result[OGS_C.P_WEIGHT_STR] == OGS_C.SPACE_STR:
            result[OGS_C.P_WEIGHT_STR] = DEFAULT_VALUE
          else:
            result[OGS_C.P_WEIGHT_STR] = int(result[OGS_C.P_WEIGHT_STR])
        except ValueError as e:
          self.logger.error(e)
          continue
        DETECT.append([
          result[OGS_C.INDEX_STR],
          result[OGS_C.P_TIME_STR].strftime(OGS_C.DATE_FMT),
          result[OGS_C.P_TIME_STR],
          f".{result[OGS_C.STATION_STR]}.",
          OGS_C.PWAVE, int(result[OGS_C.P_WEIGHT_STR]),
          None, None, None, None, 1.0
        ])
        # S Type
        if result[OGS_C.S_TIME_STR]:
          # S Weight
          try:
            if result[OGS_C.S_WEIGHT_STR] == OGS_C.SPACE_STR:
              result[OGS_C.S_WEIGHT_STR] = DEFAULT_VALUE
            else:
              result[OGS_C.S_WEIGHT_STR] = int(result[OGS_C.S_WEIGHT_STR])
          except ValueError as e:
            self.logger.error(e)
            continue
          # S Time
          try:
            result[OGS_C.S_TIME_STR] = result[OGS_C.DATE_STR] + \
              td(seconds=float(result[OGS_C.S_TIME_STR].replace(
                OGS_C.SPACE_STR, OGS_C.ZERO_STR)) / 100.)
          except ValueError as e:
            self.logger.error(e)
            continue
          DETECT.append([
            result[OGS_C.INDEX_STR],
            result[OGS_C.S_TIME_STR].strftime(OGS_C.DATE_FMT),
            result[OGS_C.S_TIME_STR],
            f".{result[OGS_C.STATION_STR]}.",
            OGS_C.SWAVE, int(result[OGS_C.S_WEIGHT_STR]),
            None, None, None, None, 1.0
          ])
        continue
      if re.match(r"1\s*D?\s*.?$", line): continue
      if line == OGS_C.EMPTY_STR: continue
      self.logger.error(f"ERROR: (DAT) Could not parse line: {line}")
      self.debug(line, self.RECORD_EXTRACTOR_LIST)
    self.PICKS = pd.DataFrame(DETECT, columns=[
      OGS_C.IDX_PICKS_STR, OGS_C.GROUPS_STR, OGS_C.TIME_STR, OGS_C.STATION_STR,
      OGS_C.PHASE_STR, OGS_C.WEIGHT_STR, OGS_C.EPICENTRAL_DISTANCE_STR,
      OGS_C.DEPTH_STR, OGS_C.AMPLITUDE_STR, OGS_C.STATION_ML_STR,
      OGS_C.PROBABILITY_STR
    ]).astype({ OGS_C.IDX_PICKS_STR: int})
    self.PICKS[OGS_C.GROUPS_STR] = self.PICKS[OGS_C.TIME_STR].apply(
      lambda x: x.date())
    for date, df in self.PICKS.groupby(OGS_C.GROUPS_STR):
      self.picks[UTCDateTime(date).date] = df

def main(args):
  for file in args.file:
    datafile = DataFileDAT(file, args.dates[0], args.dates[1],
                           verbose=args.verbose)
    datafile.read()
    datafile.log()

if __name__ == "__main__": main(parse_arguments())
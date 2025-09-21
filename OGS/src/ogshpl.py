import re
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta as td
from matplotlib.path import Path as mplPath

import ogsconstants as OGS_C

from matplotlib.cbook import flatten as flatten_list

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

class DataFileHPL(OGS_C.OGSDataFile):
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
    fr"(([A-D]/[A-D])|\s{{3}})",                                  # Unknown
    fr"(?P<A>[\s\d\.]{{5}})\s",                                    # Unknown
    fr"(?P<B>[\s\d]{{2}})\s",                                      # Unknown
    fr"(?P<C>[\s\d]{{2}})",                                        # Unknown
    fr"(?P<D>[\-\s\d\.]{{5}})",                                    # Unknown
    fr"(?P<E>[\s\d\.]{{5}})\s",                                    # Unknown
    fr"(?P<F>[\s\d]{{2}})\s",                                      # Unknown
    fr"(?P<G>[\s\d\.]{{4}})\s",                                    # Unknown
    fr"(?P<H>[\s\d\.]{{4}})\s",                                    # Unknown
    fr"(?P<I>[\s\d]{{2}})\s",                                      # Unknown
    fr"(?P<J>[\s\d\-\.]{{4}})\s",                                  # Unknown
    fr"(?P<K>[\s\d\.]{{4}})",                                      # Unknown
    fr"(?P<L>[\s\d]{{2}})",                                        # Unknown
    fr"(?P<M>[\s\d\.]{{5}})\s",                                    # Unknown
    fr"(?P<N>[\s\d\.]{{4}})\s{{9}}",                               # Unknown
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
  def read(self):
    assert self.filepath.suffix == OGS_C.HPL_EXT
    assert self.filepath.exists()
    SOURCE = list()
    DETECT = list()
    event_notes: str = ""
    event_detect: int = 0
    event_spacetime = (datetime.min, 0, 0, 0)
    with open(self.filepath, 'r') as fr: lines = fr.readlines()
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
          result[OGS_C.P_TIME_STR] = datetime(
            date.year, date.month, date.day) + hrs + min
          if (self.start is not None and
              result[OGS_C.P_TIME_STR] < self.start):
            event_detect = -1 # Error
            continue
          if (self.end is not None and
              result[OGS_C.P_TIME_STR] >= self.end + OGS_C.ONE_DAY): break
          DETECT.append([
            result[OGS_C.INDEX_STR],
            result[OGS_C.P_TIME_STR] + result[OGS_C.SECONDS_STR],
            result[OGS_C.P_WEIGHT_STR],
            OGS_C.PWAVE,
            None,
            result[OGS_C.STATION_STR],
            None,
            result[OGS_C.P_TIME_STR].strftime(OGS_C.DATE_FMT)
          ])
          if result[OGS_C.S_TIME_STR]:
            result[OGS_C.S_WEIGHT_STR] = int(result[OGS_C.S_WEIGHT_STR])
            result[OGS_C.S_TIME_STR] = td(seconds=float(
              result[OGS_C.S_TIME_STR].replace(OGS_C.SPACE_STR,
                                                OGS_C.ZERO_STR)))
            DETECT.append([
              result[OGS_C.INDEX_STR],
              result[OGS_C.P_TIME_STR] + result[OGS_C.S_TIME_STR],
              result[OGS_C.S_WEIGHT_STR],
              OGS_C.SWAVE,
              None,
              result[OGS_C.STATION_STR],
              None,
              result[OGS_C.P_TIME_STR].strftime(OGS_C.DATE_FMT)
            ])
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
            event_detect = -1 # Error
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
          # # Quality Metric
          result[OGS_C.QM_STR] = result[OGS_C.QM_STR].strip(OGS_C.SPACE_STR) \
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
    self.events[OGS_C.ERH_STR] = \
      self.events[OGS_C.ERH_STR].replace(" " * 4, "NaN").apply(float)
    self.events[OGS_C.ERZ_STR] = \
      self.events[OGS_C.ERZ_STR].replace(" " * 4, "NaN").apply(float)
    self.events[OGS_C.MAGNITUDE_D_STR] = \
      self.events[OGS_C.MAGNITUDE_D_STR].replace(" " * 6, "NaN").apply(float)
    self.events = self.events[self.events[
      [OGS_C.LONGITUDE_STR, OGS_C.LATITUDE_STR]].apply(
        lambda x: self.polygon.contains_point(
          (x[OGS_C.LONGITUDE_STR], x[OGS_C.LATITUDE_STR])), axis=1)]

  def log(self) -> None:
    super().log()

def main(args):
  datafile = DataFileHPL(args.file, args.dates[0], args.dates[1])
  datafile.read()
  datafile.log()

if __name__ == "__main__": main(parse_arguments())
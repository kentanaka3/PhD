"""OGS data file abstractions and logging helpers."""

import re
import itertools as it
from pathlib import Path
from datetime import datetime
from obspy import UTCDateTime
from matplotlib.path import Path as mplPath
from matplotlib.cbook import flatten as flatten_list

import ogsconstants as OGS_C
from ogscatalog import OGSCatalog

class OGSDataFile(OGSCatalog):
  """Base class for OGS catalog-backed data files."""
  RECORD_EXTRACTOR_LIST : list = [] # TBD in subclasses
  EVENT_EXTRACTOR_LIST : list = [] # TBD in subclasses
  GROUP_PATTERN = re.compile(r"\(\?P<(\w+)>[\[\]\w\d\{\}\-\\\?\+]+\)(\w)*")
  def __init__(self, input: Path, start: datetime = datetime.max,
               end: datetime = datetime.min, verbose: bool = False,
               polygon : mplPath = mplPath(OGS_C.OGS_POLY_REGION, closed=True),
               output : Path = OGS_C.THIS_FILE.parent / "data" / "OGSCatalog"):
    """Initialize the data file wrapper and regex extractors."""
    super().__init__(input, start, end, verbose, polygon, output)
    self.RECORD_EXTRACTOR : re.Pattern = re.compile(OGS_C.EMPTY_STR.join(
      list(flatten_list(self.RECORD_EXTRACTOR_LIST)))) # TBD in subclasses
    self.EVENT_EXTRACTOR : re.Pattern = re.compile(OGS_C.EMPTY_STR.join(
      list(flatten_list(self.EVENT_EXTRACTOR_LIST)))) # TBD in subclasses
    self.name = self.input.suffix.lstrip(OGS_C.PERIOD_STR).upper()

  def read(self):
    """Read the input data into picks/events (subclasses must implement)."""
    raise NotImplementedError

  def log(self):
    """Persist parsed picks and events to the output directory."""
    log = self.output / self.input.suffix
    self.logger.info(f"Logging data to: {log}")
    # Picks
    for date, df in self.postload("picks").items():
      date = UTCDateTime(date).date
      dir_path = log / "assignments" / OGS_C.DASH_STR.join([
        f"{date.year}", f"{date.month:02}", f"{date.day:02}"])
      dir_path.parent.mkdir(parents=True, exist_ok=True)
      df.to_parquet(dir_path, index=False)
      self.logger.debug(f"Saved PICKS for {date} to {dir_path}")
    # Events
    for date, df in self.postload("events").items():
      dir_path = log / "events" / OGS_C.DASH_STR.join([
        f"{date.year}", f"{date.month:02}", f"{date.day:02}"])
      dir_path.parent.mkdir(parents=True, exist_ok=True)
      df.to_parquet(dir_path, index=False)
      self.logger.debug(f"Saved EVENTS for {date} to {dir_path}")

  def debug(self, line, EXTRACTOR_LIST):
    """Return the failing regex group for a given line and extractor list."""
    RECORD_EXTRACTOR_DEBUG = list(reversed(list(it.accumulate(
      EXTRACTOR_LIST[:-1],
      lambda x, y: x + (y if isinstance(y, str) else
                        OGS_C.EMPTY_STR.join(list(flatten_list(y))))))))
    bug = self.GROUP_PATTERN.findall(EXTRACTOR_LIST[0])
    for i, extractor in enumerate(RECORD_EXTRACTOR_DEBUG):
      match_extractor = re.match(extractor, line)
      if match_extractor:
        match_group = self.GROUP_PATTERN.findall(RECORD_EXTRACTOR_DEBUG[i - 1])
        match_compare = self.GROUP_PATTERN.findall(extractor)
        bug = match_group[-1][match_group[-1][1] != match_compare[-1][1]]
        print(f"{self.input.suffix} {bug} : {line}")
        break
    return bug
"""
=============================================================================
OGS Downloader - Multi-Backend FDSN Waveform Retrieval CLI
=============================================================================

OVERVIEW:
Command-line tool that downloads waveform data (and associated station
metadata) from FDSN-compatible data centers, supporting both Pyrocko and
ObsPy backends with the same arguments. Designed to populate the OGS waveform
archive used by the rest of the pipeline.

The module implements:

1. CLIENT ALIASING
    - ``PYROCKO_CLIENT_ALIASES``: maps OGS-named clients (INGV, GFZ, ETH,
      ORFEUS, etc.) onto Pyrocko's internal client identifiers.
    - ``TOKEN_CLIENTS``: data centers that require restricted-data tokens.

2. CHANNEL PRIORITIES
    - ``BAND_CODES``: high-rate band codes considered (HH, EH, HN, HG).
    - ``PYROCKO_CHANNEL_PRIORITIES`` and ``OBSPY_CHANNEL_PRIORITIES``: per-
      component preference lists; horizontal fallbacks include the 1/2
      orientations.

3. CLI HELPERS
    - ``positive_int``, ``parse_arguments``: argparse validators and the
      full parser definition with output, region, client and date arguments.
    - ``_split_filter_values``: parses include / exclude tokens used for the
      network / station / location / channel filters.
    - ``_pyrocko_site`` / ``_client_label`` / ``_datetime_to_pyrocko_time``:
      small format / id converters.

4. DAY-WINDOW PLANNING
    - ``day_start`` / ``day_window`` / ``day_directory`` /
      ``prepare_day_download`` / ``domain_kwargs``: split a date range into
      per-day download windows, resolve output directories under DATA_PATH,
      and build the rectangular geographic ``domain_kwargs`` dict expected
      by both backends.

5. DOWNLOADER CLASSES
    - ``BaseDownloader``: abstract base capturing CLI args, threading, and
      shared logging behavior.
    - ``PyrockoDownloader`` / ``ObsPyDownloader``: concrete backends that
      implement ``run()`` using their respective FDSN client libraries.
    - ``get_downloader``: factory that dispatches based on ``args.backend``.
    - ``data_downloader``: entry-point used by the CLI script.

USAGE:
    python ogsdownloader.py --backend pyrocko \
                            --clients INGV ETH \
                            --networks OX,NI \
                            --dates 240320 240620 \
                            --output /path/to/waveforms

DEPENDENCIES:
    - Pyrocko (squirrel + fdsn) and/or ObsPy (clients.fdsn): waveform IO
    - ThreadPoolExecutor: per-day parallelism within a single backend run
    - ogsconstants: client name strings, separators, wildcard tokens
    - ogsutils: shared logging and validation helpers

AUTHOR: AI2Seism Project
=============================================================================
"""

import argparse
import calendar
import io
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any

import ogsconstants as OGS_C
import ogsutils as OGS_U

# Project root (three levels above this file)
DATA_PATH = Path(__file__).parent.parent.parent

PYROCKO_CLIENT_ALIASES = {
  OGS_C.INGV_CLIENT_STR: "ingv",
  OGS_C.IRIS_CLIENT_STR: "iris",
  OGS_C.GFZ_CLIENT_STR: "geofon",
  OGS_C.ETH_CLIENT_STR: "ethz",
  OGS_C.ORFEUS_CLIENT_STR: "orfeus",
  OGS_C.GEOFON_CLIENT_STR: "geofon",
  OGS_C.RESIF_CLIENT_STR: "resif",
  OGS_C.LMU_CLIENT_STR: "lmu",
  OGS_C.USGS_CLIENT_STR: "usgs",
  OGS_C.EMSC_CLIENT_STR: "emsc",
  OGS_C.ODC_CLIENT_STR: "orfeus",
  OGS_C.GEONET_CLIENT_STR: "geonet",
  OGS_C.RASPISHAKE_CLIENT_STR: "raspishake",
}
TOKEN_CLIENTS = {
  OGS_C.INGV_CLIENT_STR,
  OGS_C.GFZ_CLIENT_STR,
}
BAND_CODES = ["HH", "EH", "HN", "HG"]
# PyRocko component groups: Z, then horizontals (N/1 and E/2 fall back to each other).
PYROCKO_CHANNEL_PRIORITIES = [
  [f"{band}{component}" for band in BAND_CODES]
  for component in ("Z", "N", "E")
]
PYROCKO_CHANNEL_PRIORITIES[1] += [f"{band}1" for band in BAND_CODES]
PYROCKO_CHANNEL_PRIORITIES[2] += [f"{band}2" for band in BAND_CODES]
PYROCKO_CHANNEL_QUERY = OGS_C.COMMA_STR.join(f"{band}?" for band in BAND_CODES)
OBSPY_CHANNEL_PRIORITIES = [f"{band}[ZNE]" for band in BAND_CODES]


def _datetime_to_pyrocko_time(value: datetime) -> float:
  return calendar.timegm(value.utctimetuple()) + value.microsecond / 1_000_000


def _pyrocko_site(client: str) -> str:
  if client.startswith(("http://", "https://")):
    return client.rstrip("/")
  return PYROCKO_CLIENT_ALIASES.get(client, client.lower())


def _client_label(client: str) -> str:
  label = "".join(char if char.isalnum() else "-" for char in client)
  return label.strip("-").lower() or "client"


def _split_filter_values(values: list[str]) -> tuple[list[str], set[str]]:
  includes = []
  excludes = set()
  for value in values:
    for raw_token in value.replace(OGS_C.COMMA_STR, " ").split():
      token = raw_token.strip()
      if not token:
        continue
      if token.startswith("-") and len(token) > 1:
        excludes.add(token[1:])
      else:
        includes.append(token)
  return includes or [OGS_C.ALL_WILDCHAR_STR], excludes


def positive_int(value: str) -> int:
  try:
    parsed_value = int(value)
  except ValueError as exc:
    raise argparse.ArgumentTypeError("must be a positive integer") from exc
  if parsed_value < 1:
    raise argparse.ArgumentTypeError("must be a positive integer")
  return parsed_value

def parse_arguments() -> argparse.Namespace:
  # Parse command-line arguments for waveform download.
  parser = argparse.ArgumentParser(
    description="Download waveform data from configured FDSN clients")
  # TODO: Handle security issues
  parser.add_argument(
    '-K', "--key", default=None, required=False, type=OGS_U.is_file_path,
    metavar=OGS_C.EMPTY_STR, help="Key to download the data from server.")
  parser.add_argument(
    '-N', "--network", default=[OGS_C.ALL_WILDCHAR_STR], type=str,
    nargs=OGS_C.ONE_MORECHAR_STR, metavar=OGS_C.EMPTY_STR, required=False,
    help="Specify a set of Networks to analyze. To allow downloading data for "
        f"any channel, set this option to \'{OGS_C.ALL_WILDCHAR_STR}\'. Use "
        f"the negative sign \'-\' to exclude specific networks (e.g. \'-OX\')."
  )
  parser.add_argument(
    '-S', "--station", default=[OGS_C.ALL_WILDCHAR_STR], type=str,
    nargs=OGS_C.ONE_MORECHAR_STR, metavar=OGS_C.EMPTY_STR, required=False,
    help="Specify a set of Stations to analyze. To allow downloading data for "
        f"any channel, set this option to \'{OGS_C.ALL_WILDCHAR_STR}\'. Use "
        f"the negative sign \'-\' to exclude specific stations (e.g. \'-SP, "
        f"-OL, -ED\')."
  )
  parser.add_argument(
    "-c", "--clip", required=False, type=str, metavar="HHMMSS",
    help="Specify the time of the center time")
  parser.add_argument(
    '-d', "--directory", required=False, type=OGS_U.is_dir_path,
    default=Path(DATA_PATH, OGS_C.WAVEFORMS_STR), metavar=OGS_C.EMPTY_STR,
    help="Directory path to the raw files")
  parser.add_argument(
    "--client", metavar=OGS_C.EMPTY_STR, default=OGS_C.OGS_CLIENTS_DEFAULT,
    required=False, type=str, nargs=OGS_C.ONE_MORECHAR_STR,
    help="Client to download the data")
  parser.add_argument(
    "--force", default=False, action='store_true', required=False,
    help="Force running all the pipeline")
  parser.add_argument(
    "--pyrocko", default=False, action='store_true',
    help="Enable PyRocko calls")
  parser.add_argument(
    "--review", default=None, type=OGS_U.is_dir_path, required=False,
    help="Review the downloaded data")
  parser.add_argument(
    "--timing", default=False, action='store_true', required=False,
    help="Enable timing")
  parser.add_argument(
    "--threads", default=1, type=positive_int, required=False,
    metavar=OGS_C.EMPTY_STR, help="Number of day-level download threads")
  date_group = parser.add_mutually_exclusive_group(required=False)
  date_group.add_argument(
    '-D', "--dates", required=False, metavar=OGS_C.DATE_STD,
    type=OGS_U.is_date, nargs=2, action=OGS_U.SortDatesAction,
    default=[datetime.strptime("20240320", OGS_C.YYYYMMDD_FMT),
             datetime.strptime("20240620", OGS_C.YYYYMMDD_FMT)],
    help="Specify the beginning and ending (inclusive) Gregorian date " \
         "(YYYYMMDD) range to work with.")
  date_group.add_argument(
    '-J', "--julian", required=False, metavar=OGS_C.DATE_STD,
    action=OGS_U.SortDatesAction, type=OGS_U.is_julian, default=None, nargs=2,
    help="Specify the beginning and ending (inclusive) Julian date (YYMMDD) " \
         "range to work with.")
  domain_group = parser.add_mutually_exclusive_group(required=False)
  domain_group.add_argument(
    "--rectdomain", type=float, nargs=4, default=OGS_C.OGS_STUDY_REGION,
    metavar=("lonW", "lonE", "latS", "latN"),
    help="Rectangular domain to download the data: [longitude West] "
         "[longitude East] [latitude South] [latitude North]")
  domain_group.add_argument(
    "--circdomain", nargs=4, type=float, # default=[12.808, 46.3583, 0., 0.3],
    metavar=("lon", "lat", "min_r", "max_r"),
    help="Circular domain to download the data: [center longitude] "
         "[center latitude] [minimum radius] [maximum radius]")
  verbal_group = parser.add_mutually_exclusive_group(required=False)
  verbal_group.add_argument(
    "--silent", default=False, action='store_true', help="Silent mode")
  # TODO: Add verbose LEVEL
  verbal_group.add_argument(
    "-v", "--verbose", default=False, action='store_true', help="Verbose mode")
  args = parser.parse_args()
  # TODO: Fix special cases
  # print(vars(args))
  return args

DIR_FMT = {
  "year": "{:04}",
  "month": "{:02}",
  "day": "{:02}",
}

def day_start(d_: date | datetime) -> datetime:
  if isinstance(d_, datetime):
    return d_
  return datetime(d_.year, d_.month, d_.day)

def day_window(args: argparse.Namespace, d_: date | datetime) -> tuple[datetime, datetime]:
  d_ = day_start(d_)
  if not args.clip:
    return d_, d_ + OGS_C.ONE_DAY
  clip_time = datetime.strptime(
    d_.strftime(OGS_C.DATE_FMT) + args.clip,
    OGS_C.DATE_FMT + OGS_C.TIME_FMT
  )
  return (
    clip_time - OGS_C.PICK_TRAIN_OFFSET,
    clip_time + OGS_C.PICK_TRAIN_OFFSET,
  )

def day_directory(args: argparse.Namespace, d_: datetime) -> Path:
  d_ = day_start(d_)
  return Path(
    args.directory /
    DIR_FMT["year"].format(d_.year) /
    DIR_FMT["month"].format(d_.month) /
    DIR_FMT["day"].format(d_.day)
  )

def prepare_day_download(
    args: argparse.Namespace, d_: date | datetime) -> tuple[str, datetime, datetime, Path]:
  d_ = day_start(d_)
  day_id = d_.strftime(OGS_C.YYYYMMDD_FMT)
  starttime, endtime = day_window(args, d_)
  day_path = day_directory(args, d_)
  day_path.mkdir(parents=True, exist_ok=True)
  return day_id, starttime, endtime, day_path

def domain_kwargs(args: argparse.Namespace) -> dict[str, float]:
  if args.rectdomain:
    return {
      "minlongitude": args.rectdomain[0],
      "maxlongitude": args.rectdomain[1],
      "minlatitude": args.rectdomain[2],
      "maxlatitude": args.rectdomain[3],
    }
  return {
    "longitude": args.circdomain[0],
    "latitude": args.circdomain[1],
    "minradius": args.circdomain[2],
    "maxradius": args.circdomain[3],
  }

class BaseDownloader(ABC):
  """
  Abstract base class for all waveform downloaders.
  """
  def __init__(self, args: argparse.Namespace):
    self.args = args
    self.logger = OGS_U.setup_logger(__name__, args.verbose, args.silent)
    self.start, self.end = args.dates
    self.days = [self.start + index * OGS_C.ONE_DAY for index in range((self.end - self.start).days + 1)]
    self.network_filter = OGS_C.COMMA_STR.join(args.network)
    self.station_includes, self.station_excludes = _split_filter_values(args.station)
    self.station_filter = OGS_C.COMMA_STR.join(self.station_includes)
    self.station_directory = Path(args.directory, OGS_C.STATION_STR)

  def day_start(self, d_: date | datetime) -> datetime:
    return day_start(d_)

  def day_window(self, d_: date | datetime) -> tuple[datetime, datetime]:
    return day_window(self.args, d_)

  def day_directory(self, d_: datetime) -> Path:
    return day_directory(self.args, d_)

  def prepare_day_download(self, d_: date | datetime) -> tuple[str, datetime, datetime, Path]:
    return prepare_day_download(self.args, d_)

  def domain_kwargs(self) -> dict[str, float]:
    return domain_kwargs(self.args)

  @abstractmethod
  def prepare(self) -> bool:
    """Prepare downloader (e.g., check dependencies, resolve filters)."""
    pass

  @abstractmethod
  def download_day(self, d_: date | datetime) -> None:
    """Download data for a specific day."""
    pass

  def run(self) -> None:
    if self.args.review:
      self.logger.info("Reviewing the downloaded data in directory: %s", self.args.review)
      return

    if not self.prepare():
      return

    if self.args.threads == 1:
      for d_ in self.days:
        self.download_day(d_)
    else:
      with ThreadPoolExecutor(max_workers=self.args.threads) as executor:
        futures = {executor.submit(self.download_day, d_): d_ for d_ in self.days}
        for future in as_completed(futures):
          d_ = futures[future]
          try:
            future.result()
          except Exception as e:
            self.logger.error("Unexpected error downloading date %s: %s",
                               d_.strftime(OGS_C.YYYYMMDD_FMT), e)

class PyrockoDownloader(BaseDownloader):
  """
  Pyrocko-based waveform downloader.
  """
  def __init__(self, args: argparse.Namespace):
    super().__init__(args)
    self.token = None

  def prepare(self) -> bool:
    try:
      import pyrocko as pr
      from pyrocko.client import fdsn as pyrocko_fdsn
      from obspy import read as read_stream
    except ModuleNotFoundError as exc:
      self.logger.error("PyRocko download support is unavailable: %s", exc)
      return False

    self.logger.info("PyRocko is available: %s", pr.__version__)
    self.station_directory.mkdir(parents=True, exist_ok=True)
    if self.args.key:
      try:
        self.token = Path(self.args.key).read_bytes().strip()
      except OSError as exc:
        self.logger.error("Error reading token file %s: %s", self.args.key, exc)
        return False
    return True

  def download_day(self, d_: date | datetime) -> None:
    from pyrocko.client import fdsn as pyrocko_fdsn
    from obspy import read as read_stream

    day_id, starttime, endtime, day_path = self.prepare_day_download(d_)
    start_pyrocko = _datetime_to_pyrocko_time(starttime)
    end_pyrocko = _datetime_to_pyrocko_time(endtime)

    station_kwargs = {
      "level": "channel",
      "format": "xml",
      "network": self.network_filter,
      "station": self.station_filter,
      "channel": PYROCKO_CHANNEL_QUERY,
      "starttime": start_pyrocko,
      "endtime": end_pyrocko,
      **self.domain_kwargs(),
    }
    total_traces = 0
    clients_with_data = 0

    for client in self.args.client:
      site = _pyrocko_site(client)
      client_label = _client_label(client)
      try:
        station_xml = pyrocko_fdsn.station(
          site=site, check=False, parsed=True, **station_kwargs)
      except pyrocko_fdsn.EmptyResult:
        self.logger.info("No PyRocko stations matched %s for date %s",
                         client, day_id)
        continue
      except Exception as exc:
        self.logger.error(
          "Error downloading station metadata from %s for date %s: %s",
          client, day_id, exc)
        continue

      selection = pyrocko_fdsn.make_data_selection(
        [station for station in station_xml.get_pyrocko_stations(
          timespan=(start_pyrocko, end_pyrocko),
          inconsistencies="warn")
         if station.station not in self.station_excludes],
        start_pyrocko,
        end_pyrocko,
        channel_prio=PYROCKO_CHANNEL_PRIORITIES)
      if not selection:
        self.logger.info("No PyRocko channels matched %s for date %s",
                         client, day_id)
        continue

      station_file = self.station_directory / (
        f"{client_label}__{day_id}"
        f"__stations{OGS_C.XML_EXT}")
      try:
        station_xml.dump_xml(filename=str(station_file), header=True)
      except Exception as exc:
        self.logger.error(
          "Error writing station metadata from %s for date %s: %s",
          client, day_id, exc)
        continue

      waveform_stream = None
      client_token = self.token if self.token is not None and client in \
        TOKEN_CLIENTS else None
      try:
        waveform_stream = pyrocko_fdsn.dataselect(
          site=site, check=False, token=client_token, selection=selection)
        waveform_bytes = waveform_stream.read()
      except pyrocko_fdsn.EmptyResult:
        self.logger.info("No PyRocko waveform data matched %s for date %s",
                         client, day_id)
        continue
      except Exception as exc:
        self.logger.error("Error downloading PyRocko data from %s for date %s: %s",
                          client, day_id, exc)
        continue
      finally:
        if waveform_stream is not None and hasattr(waveform_stream, "close"):
          waveform_stream.close()

      try:
        traces = read_stream(
          io.BytesIO(waveform_bytes), format=OGS_C.MSEED_STR.upper())
      except Exception as exc:
        self.logger.error("Error parsing PyRocko data from %s for date %s: %s",
                          client, day_id, exc)
        continue

      client_traces = 0
      for trace in traces:
        start_id = trace.stats.starttime.strftime("%Y%m%dT%H%M%SZ")
        end_id = trace.stats.endtime.strftime("%Y%m%dT%H%M%SZ")
        location = trace.stats.location or OGS_C.EMPTY_STR
        trace_file = day_path / (
          f"{trace.stats.network}.{trace.stats.station}.{location}."
          f"{trace.stats.channel}__{start_id}__{client_label}-{end_id}{OGS_C.MSEED_EXT}")
        trace.write(str(trace_file), format=OGS_C.MSEED_STR.upper())
        client_traces += 1

      if client_traces == 0:
        self.logger.info("PyRocko returned no writable traces from %s for date %s",
                         client, day_id)
        continue

      total_traces += client_traces
      clients_with_data += 1

    if total_traces == 0:
      self.logger.info("No PyRocko waveform data downloaded for date: %s",
                       day_id)
      return
    self.logger.info("Downloaded %s PyRocko traces for date %s from %s clients",
                     total_traces, day_id, clients_with_data)

class ObsPyDownloader(BaseDownloader):
  """
  ObsPy-based mass waveform downloader.
  """
  def __init__(self, args: argparse.Namespace):
    super().__init__(args)
    self.domain = None
    self.thread_state = threading.local()
    self.resolved_station_filter = None

  def prepare(self) -> bool:
    if self.args.rectdomain:
      from obspy.clients.fdsn.mass_downloader.domain import RectangularDomain
      self.domain = RectangularDomain(
        minlongitude=self.args.rectdomain[0], maxlongitude=self.args.rectdomain[1],
        minlatitude=self.args.rectdomain[2], maxlatitude=self.args.rectdomain[3])
    else:
      from obspy.clients.fdsn.mass_downloader.domain import CircularDomain
      self.domain = CircularDomain(
        latitude=self.args.circdomain[1], longitude=self.args.circdomain[0],
        minradius=self.args.circdomain[2], maxradius=self.args.circdomain[3])

    self.resolved_station_filter = self.resolve_station_filter()
    return bool(self.resolved_station_filter)

  def clients(self) -> dict[str, Any]:
    from obspy.clients.fdsn import Client
    cached_clients = getattr(self.thread_state, "clients", None)
    if cached_clients is not None:
      return cached_clients
    # ObsPy clients are cached per worker thread and never shared across them.
    cached_clients = dict()
    for client in self.args.client:
      try:
        cached_clients[client] = Client(client)
      except Exception as e:
        self.logger.error("Error creating client %s: %s", client, e)
        continue
      if self.args.key and client in TOKEN_CLIENTS:
        try:
          cached_clients[client].set_eida_token(self.args.key, validate=True)
        except Exception as e:
          self.logger.error("Error setting token for %s: %s", client, e)
    self.thread_state.clients = cached_clients
    return cached_clients

  def resolve_station_filter(self) -> str | None:
    if not self.station_excludes:
      return self.station_filter

    station_codes = []
    seen = set()
    station_query_kwargs = {
      "network": self.network_filter,
      "station": self.station_filter,
      "level": "station",
      "starttime": self.start,
      "endtime": self.end,
      **self.domain_kwargs(),
    }
    for client_name, client in self.clients().items():
      try:
        inventory = client.get_stations(**station_query_kwargs)
      except Exception as exc:
        self.logger.error("Error resolving stations from %s: %s", client_name, exc)
        continue
      if inventory is None:
        continue
      for network in inventory:
        for station in network:
          if station.code in self.station_excludes or station.code in seen:
            continue
          station_codes.append(station.code)
          seen.add(station.code)

    if not station_codes:
      self.logger.error("No stations remain after excluding: %s",
                        OGS_C.COMMA_STR.join(sorted(self.station_excludes)))
      return None
    self.logger.info("Resolved %s stations after excluding: %s",
                     len(station_codes),
                     OGS_C.COMMA_STR.join(sorted(self.station_excludes)))
    return OGS_C.COMMA_STR.join(station_codes)

  def download_day(self, d_: date | datetime) -> None:
    from obspy.clients.fdsn.mass_downloader import Restrictions, MassDownloader
    day_id, starttime, endtime, day_path = self.prepare_day_download(d_)
    self.logger.info("Downloading the data in the directory: %s", day_path)
    # Apply selection constraints to reduce data volume
    restrictions = Restrictions(
      starttime=starttime, endtime=endtime,
      network=self.network_filter,
      station=self.resolved_station_filter,
      channel_priorities=OBSPY_CHANNEL_PRIORITIES,
      reject_channels_with_gaps=False, minimum_length=0.0,
      minimum_interstation_distance_in_m=100,
      location_priorities=["", "00", "01", "02", "10"],
      chunklength_in_sec=86400
    )
    # Execute the mass download for the day
    mdl = MassDownloader(providers=self.clients().values())
    download_kwargs = dict()
    if self.args.threads > 1:
      download_kwargs["threads_per_client"] = 1
    try:
      mdl.download(self.domain, restrictions, mseed_storage=str(day_path),
                   stationxml_storage=str(self.station_directory),
                   **download_kwargs)
    except Exception as e:
      self.logger.error("Error downloading data for date %s: %s",
                        day_id, e)
    else:
      # Report completion per-day
      self.logger.info("Downloaded data for date: %s", day_id)

def get_downloader(args: argparse.Namespace) -> BaseDownloader:
  return PyrockoDownloader(args) if args.pyrocko else ObsPyDownloader(args)

def data_downloader(args: argparse.Namespace) -> None:
  """
  Download waveform data for the requested date range.
  """
  downloader = get_downloader(args)
  downloader.run()

if __name__ == "__main__": data_downloader(parse_arguments())
#!/usr/bin/env python

"""
===============================================================================
OGS Downloader - Multi-Backend FDSN Waveform Retrieval CLI
===============================================================================

OVERVIEW:
Command-line tool that downloads waveform data (and associated station
metadata) from FDSN-compatible data centers with ObsPy MassDownloader.
The --pyrocko option selects an unimplemented backend and raises
NotImplementedError. Downloads populate the daily OGS waveform archive.

USAGE:
  python -m OGS.src.ogsdownloader --client INGV ETH \
                                  --network OX NI \
                                  --station "* -SP -OL -ED" \
                                  --dates 20240320 20240620 \
                                  -W /path/to/waveforms

  MiniSEED paths:
    <waveforms>/YYYY/MM/DD/NET.STA.LOC.CHA__BEGDT__ENDDT.mseed
  Timestamps use YYYYMMDDTHHMMSSZ; LOC may be empty. The start time selects
  the day directory. StationXML storage is delegated to MassDownloader
  under the configured stations directory.

DEPENDENCIES:
  - ObsPy (clients.fdsn + mass_downloader): waveform and metadata retrieval
  - ThreadPoolExecutor: per-day parallelism within a single backend run
  - ogsconstants: client name strings, separators, wildcard tokens
  - ogsutils: shared logging and validation helpers

AUTHORS:
  - 健
  - Istituto Nazionale di Oceanografia e di Geofisica Sperimentale (OGS)
    Centro di Ricerche Sismologiche (CRS)
  - Università degli Studi di Trieste (UniTS)
    Dipartimento di Matematica, Informatica e Geoscienze (MIGe)
    Applied Data Science and Artificial Intelligence (ADSAI)
  - Terabit Network for Research and Academic Big Data in Italy (TeRABIT)
    Consorzio Interuniversitario del Nord-Est per il Calcolo Automatico (CINECA)

===============================================================================
"""

from abc import ABC, abstractmethod
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time, timezone
from fnmatch import fnmatchcase
from pathlib import Path
import tempfile
from typing import Any

from obspy import UTCDateTime
from obspy.clients.fdsn import Client as ObsPyFDSNClient
from obspy.clients.fdsn.mass_downloader import (
    CircularDomain,
    GlobalDomain,
    MassDownloader,
    RectangularDomain,
    Restrictions,
)

from . import ogsconstants as OGS_C, ogsutils as OGS_U

# Ordered channel-family patterns, each accepting Z/N/E components.
CHANNEL_PRIORITIES = ["HH[ZNE]", "EH[ZNE]", "HN[ZNE]", "HG[ZNE]"]
# Empty location codes are preferred over the common numbered alternatives.
LOCATION_PRIORITIES = ("", "00", "01", "02", "10")
TimeWindow = tuple[datetime, datetime]
Selection = tuple[str, str, str, str, *TimeWindow]


def daily_clipper(
    date: datetime, clip: time | None = None
) -> TimeWindow:
  if not clip:
    return date, date + OGS_C.ONE_DAY
  clip_time = datetime.strptime(
      date.strftime(OGS_C.YYYYMMDD_FMT) + clip.strftime(OGS_C.TIME_FMT),
      OGS_C.YYYYMMDD_FMT + OGS_C.TIME_FMT,
  )
  return (
      clip_time - OGS_C.PICK_TRAIN_OFFSET,
      clip_time + OGS_C.PICK_TRAIN_OFFSET,
  )


class BaseDownloader(ABC):
  """Shared lifecycle and FDSN request helpers for downloader backends.

  The base class normalizes the command-line namespace once, creates the
  requested time windows, parses network and station filters, probes FDSN
  services, and generates deterministic archive paths. Subclasses are
  responsible for turning those prepared values into backend-specific
  requests.
  """

  def __init__(self, args: Namespace) -> None:
    """Initialize common downloader state from parsed CLI arguments."""
    self.args = args
    self.logger = OGS_U.setup_logger(__name__, args.verbose, args.quiet)
    self.start, self.end = args.dates
    # A clip selects one event-centered interval; otherwise create one window
    # for every day in the inclusive requested range.
    self.ranges: list[TimeWindow] = (
        [daily_clipper(self.start, args.clip)] if args.clip
        else [(
            self.start + index * OGS_C.ONE_DAY,
            self.start + (index + 1) * OGS_C.ONE_DAY
        ) for index in range((self.end - self.start).days + 1)]
    )
    self.main_workers = min(4, args.threads)
    self.sub_workers = args.threads - self.main_workers
    self.networks, self.exclude_networks = self._filter(args.network)
    self.stations, self.exclude_stations = self._filter(args.station)
    self.logger.info(
        "Prepared %s for %s through %s in %d time window(s), with up to "
        "%d main worker(s) and %d sub-worker(s).",
        type(self).__name__, self.start, self.end, len(self.ranges),
        self.main_workers, self.sub_workers,
    )
    self.logger.debug(
        "Network selection include=%s exclude=%s; station selection "
        "include=%s exclude=%s.",
        self.networks, self.exclude_networks,
        self.stations, self.exclude_stations,
    )

  @property
  def client_kwargs(self) -> dict[str, Any]:
    """Return connection options shared by all ObsPy FDSN clients."""
    return {
        "timeout": getattr(self.args, "timeout", OGS_C.OGS_TIMEOUT),
        "eida_token": getattr(self.args, "key", None),
    }

  def waveform_path(self, selection: Selection) -> Path:
    """Build YYYY/MM/DD/NET.STA.LOC.CHA__BEGDT__ENDDT.mseed from the start."""
    network, station, location, channel, starttime, endtime = selection
    day_path = Path(
        OGS_U.day_directory(self.args.waveforms, starttime)
    )
    timestamp_fmt = f"{OGS_C.YYYYMMDD_FMT}T{OGS_C.TIME_FMT}Z"
    filename = OGS_C.PRC_FMT.format(
        NETWORK=network, STATION=station, LOCATION=location, CHANNEL=channel,
        BEGDT=starttime.strftime(timestamp_fmt),
        ENDDT=endtime.strftime(timestamp_fmt),
        EXT=OGS_C.MSEED_STR,
    )
    return day_path / filename

  def _waveform_storage(
      self, network: str, station: str, location: str, channel: str,
      starttime: UTCDateTime, endtime: UTCDateTime,
  ) -> str:
    """Adapt ObsPy's six-argument storage callback to waveform_path."""
    selection: Selection = (
        network,
        station,
        location,
        channel,
        starttime.datetime,
        endtime.datetime
    )
    return str(self.waveform_path(selection))

  @staticmethod
  def _filter(value: str | list[str] | None) -> tuple[list[str], list[str]]:
    items = [value] if isinstance(value, str) else value or []
    include, exclude = [], []
    for item in items:
      for s in item.split():
        (exclude if s.startswith("-") else include).append(
            s.lstrip("-")
        )
    return include or ["*"], exclude

  @property
  def domain_kwargs(self) -> dict[str, float]:
    """Translate the selected circular or rectangular domain to FDSN keys."""
    if self.args.circdomain:
      return {
          "longitude": self.args.circdomain[0],
          "latitude": self.args.circdomain[1],
          "minradius": self.args.circdomain[2],
          "maxradius": self.args.circdomain[3],
      }
    if self.args.rectdomain:
      return {
          "minlongitude": self.args.rectdomain[0],
          "maxlongitude": self.args.rectdomain[1],
          "minlatitude": self.args.rectdomain[2],
          "maxlatitude": self.args.rectdomain[3],
      }
    return {}

  def selection_kwargs(self, window: TimeWindow) -> dict[str, Any]:
    """Build common selection arguments for a time window."""
    start, end = window
    return {
        "starttime": UTCDateTime(start),
        "endtime": UTCDateTime(end),
        "network": OGS_C.COMMA_STR.join(self.networks),
        "station": OGS_C.COMMA_STR.join(self.stations),
    }

  def station_kwargs(
      self, window: TimeWindow, level: str = "channel"
  ) -> dict[str, Any]:
    """Build common station-service query arguments for a time window.

    Domain constraints come from the argument namespace (including parser
    defaults). The level defaults to "channel". Waveform restriction options
    are built separately by download_kwargs(); the domain is passed as an
    ObsPy domain object.
    """
    return {
        **self.selection_kwargs(window),
        "level": level,
        **self.domain_kwargs,
    }

  @property
  def _selection_kwargs(self) -> dict[str, Any]:
    """Return channel/location priorities and ObsPy restriction options."""
    return {
        "channel_priorities": CHANNEL_PRIORITIES,
        "chunklength_in_sec": 86400,
        "location_priorities": LOCATION_PRIORITIES,
        "minimum_length": 0.0,
        "minimum_interstation_distance_in_m": 100,
        "reject_channels_with_gaps": False,
    }

  def download_kwargs(self, window: TimeWindow) -> dict[str, Any]:
    """Build ObsPy restriction options for one time window."""
    return {
        **self.selection_kwargs(window),
        "exclude_networks": self.exclude_networks,
        "exclude_stations": self.exclude_stations,
        **self._selection_kwargs,
    }

  def _has_selected_channels(self, inventory: Any) -> bool:
    """
    Check whether station metadata contains channels selected for download.
    """
    for network in inventory:
      if any(
          fnmatchcase(network.code, pattern)
          for pattern in self.exclude_networks
      ):
        continue
      for station in network:
        if any(
            fnmatchcase(station.code, pattern)
            for pattern in self.exclude_stations
        ):
          continue
        for channel in station:
          if channel.location_code not in LOCATION_PRIORITIES:
            continue
          if any(
              fnmatchcase(channel.code, pattern)
              for pattern in CHANNEL_PRIORITIES
          ):
            return True
    return False

  @abstractmethod
  def download(self) -> None:
    """Execute backend waveform and metadata download across all configured windows."""
    pass


class ObsPyDownloader(BaseDownloader):
  def __init__(self, args: Namespace) -> None:
    super().__init__(args)
    self.clients: list[ObsPyFDSNClient] = []

  def _probe_clients(self) -> None:
    query_window: TimeWindow = (self.ranges[0][0], self.ranges[-1][1])
    for name in self.args.client:
      self.logger.info(
          "Initializing ObsPy FDSN client for provider '%s'.", name)
      try:
        client = ObsPyFDSNClient(name, **self.client_kwargs)
        inventory = client.get_stations(
            **self.station_kwargs(query_window, level="channel"),
        )
      except Exception as error:
        self.logger.error(
            "Failed to query station metadata from FDSN client '%s': %s",
            name, error,
        )
        continue
      if not self._has_selected_channels(inventory):
        self.logger.info(
            "FDSN client '%s' has no station channels matching the selection.",
            name,
        )
        continue
      self.clients.append(client)
    if not self.clients:
      self.logger.error("No FDSN clients have matching station channels.")
      raise ValueError(
          "At least one FDSN client must have matching station channels.")
    self.logger.info("Initialized %d ObsPy FDSN client(s).", len(self.clients))

  def _download_window(
      self,
      mass_downloader: MassDownloader,
      domain: CircularDomain | GlobalDomain | RectangularDomain,
      window: TimeWindow,
  ) -> None:
    self.logger.info(
        "Starting download for window %s to %s.", window[0], window[1]
    )
    mass_downloader.download(
        domain,
        Restrictions(**self.download_kwargs(window)),
        mseed_storage=self._waveform_storage,
        stationxml_storage=str(self.args.stations),
        threads_per_client=max(1, self.sub_workers),
    )
    self.logger.info(
        "Completed download for window %s to %s.", window[0], window[1]
    )

  def download(self) -> None:
    self._probe_clients()
    mass_downloader = MassDownloader(providers=self.clients)
    if self.args.circdomain:
      domain = CircularDomain(**self.domain_kwargs)
    elif self.args.rectdomain:
      domain = RectangularDomain(**self.domain_kwargs)
    else:
      domain = GlobalDomain()
    self.logger.info(
        "Using %s domain for waveform and station downloads: %s",
        type(domain).__name__, self.domain_kwargs or "global",
    )

    max_workers = max(1, min(self.main_workers, len(self.ranges)))
    self.logger.info(
        "Starting %d download window(s) with up to %d main worker(s) and "
        "%d sub-worker(s) per active window.",
        len(self.ranges), max_workers, max(1, self.sub_workers),
    )
    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      futures = {
          executor.submit(
              self._download_window, mass_downloader, domain, window
          ): window
          for window in self.ranges
      }
      for future in as_completed(futures):
        window = futures[future]
        try:
          future.result()
        except Exception as error:
          self.logger.error(
              "Download failed for date window %s to %s: %s",
              window[0], window[1], error,
          )
          raise
      self.logger.info("All %d download window(s) completed.", len(futures))


PYROCKO_SITE_MAP = {
    "OGS": OGS_C.OGS_CLIENT_STR,
    "INGV": "ingv",
    "ETH": "ethz",
    "BGR": "bgr",
    "IRIS": "iris",
    "GEONET": "geonet",
    "GFZ": "geofon",
    "IPGP": "ipgp",
    "ICGC": "icgc",
    "RESIF": "resif",
    "ORFEUS": "orfeus",
}


class PyrockoDownloader(BaseDownloader):
  """Waveform and metadata retrieval backend using Pyrocko FDSN."""
  from pyrocko import io as pyrocko_io
  from pyrocko.client import fdsn as pyrocko_fdsn
  from pyrocko.io import mseed as pyrocko_mseed, stationxml as pyrocko_sxml

  def _resolve_site(self, name: str) -> str:
    upper_name = name.strip().upper()
    if name.startswith(("http://", "https://")):
      return name
    return PYROCKO_SITE_MAP.get(upper_name, name.lower())

  def _query_stations_and_channels(
      self, site: str, window: TimeWindow
  ) -> tuple[pyrocko_sxml.FDSNStationXML | None, list[tuple[str, str, str, str]]]:
    kwargs: dict[str, Any] = {
        "site": site,
        "level": "channel",
        "parsed": True,
        "timeout": getattr(self.args, "timeout", OGS_C.OGS_TIMEOUT),
        **self.domain_kwargs,
    }
    if getattr(self.args, "key", None):
      kwargs["token"] = self.args.key
    tmin = window[0].replace(tzinfo=timezone.utc).timestamp()
    tmax = window[1].replace(tzinfo=timezone.utc).timestamp()
    try:
      sxml = pyrocko_fdsn.station(
          network=",".join(self.networks),
          station=",".join(self.stations),
          starttime=tmin,
          endtime=tmax,
          **kwargs,
      )
    except pyrocko_fdsn.EmptyResult:
      self.logger.info(
          "No station metadata for site %s matching the selection.", site)
      return None, []
    except Exception as error:
      self.logger.warning(
          "Pyrocko station query failed for site %s: %s", site, error)
      return None, []

    channels: list[tuple[str, str, str, str]] = []
    for net, sta, cha in sxml.iter_network_station_channels():
      if any(fnmatchcase(net.code, pat) for pat in self.exclude_networks):
        continue
      if any(fnmatchcase(sta.code, pat) for pat in self.exclude_stations):
        continue
      if cha.location_code not in LOCATION_PRIORITIES:
        continue
      if any(fnmatchcase(cha.code, pat) for pat in CHANNEL_PRIORITIES):
        channels.append((net.code, sta.code, cha.location_code, cha.code))
    return sxml, channels

  def _download_window(self, window: TimeWindow) -> None:
    self.logger.info(
        "Starting Pyrocko download for window %s to %s.", window[0], window[1])
    tmin = window[0].replace(tzinfo=timezone.utc).timestamp()
    tmax = window[1].replace(tzinfo=timezone.utc).timestamp()

    stations_dir = Path(self.args.stations)
    stations_dir.mkdir(parents=True, exist_ok=True)

    for client_name in self.args.client:
      site = self._resolve_site(client_name)
      sxml, eligible_channels = self._query_stations_and_channels(site, window)
      if not sxml or not eligible_channels:
        continue

      # Save StationXML metadata per station
      for net in sxml.network_list:
        for sta in net.station_list:
          xml_path = stations_dir / f"{net.code}.{sta.code}.xml"
          if self.args.force or not xml_path.exists():
            try:
              sub_sxml = pyrocko_sxml.FDSNStationXML(
                  source=sxml.source,
                  sender=sxml.sender,
                  network_list=[
                      pyrocko_sxml.Network(
                          code=net.code, start_date=net.start_date,
                          end_date=net.end_date, station_list=[sta]
                      )
                  ],
              )
              sub_sxml.dump_xml(filename=str(xml_path))
            except Exception as e:
              self.logger.debug(
                  "Failed saving StationXML for %s.%s: %s", net.code, sta.code, e)

      # Filter channels whose daily waveform files already exist unless --force is given
      needed_channels: list[tuple[str, str, str, str]] = []
      for net, sta, loc, cha in eligible_channels:
        dest = self.waveform_path((net, sta, loc, cha, window[0], window[1]))
        if self.args.force or not dest.exists():
          needed_channels.append((net, sta, loc, cha))

      if not needed_channels:
        self.logger.info(
            "All waveforms already exist for site %s in window %s to %s.",
            site, window[0], window[1]
        )
        continue

      selection = [
          (net, sta, loc, cha, tmin, tmax)
          for (net, sta, loc, cha) in needed_channels
      ]
      try:
        stream = pyrocko_fdsn.dataselect(
            site=site,
            selection=selection,
            timeout=getattr(self.args, "timeout", OGS_C.OGS_TIMEOUT),
            token=getattr(self.args, "key", None),
        )
        with tempfile.NamedTemporaryFile(suffix=".mseed") as tmp_file:
          tmp_file.write(stream.read())
          tmp_file.flush()
          for tr in pyrocko_mseed.iload(tmp_file.name):
            if tr.tmin >= tmax or tr.tmax <= tmin:
              continue
            # Slice strictly into the 24-hour day chunk [tmin, tmax]
            try:
              chopped_tr = tr.chop(tmin, tmax, inplace=False)
            except Exception:
              continue
            if len(chopped_tr.ydata) == 0:
              continue
            target_path = self.waveform_path((
                chopped_tr.network, chopped_tr.station,
                chopped_tr.location, chopped_tr.channel,
                window[0], window[1]
            ))
            if self.args.force or not target_path.exists():
              target_path.parent.mkdir(parents=True, exist_ok=True)
              pyrocko_io.save([chopped_tr], str(target_path), format="mseed")
      except pyrocko_fdsn.EmptyResult:
        self.logger.info(
            "No waveform data from site %s in window %s to %s.",
            site, window[0], window[1]
        )
      except Exception as error:
        self.logger.warning(
            "Waveform query error for client %s (%s): %s",
            client_name, site, error
        )

    self.logger.info(
        "Completed Pyrocko download for window %s to %s.", window[0], window[1])

  def download(self) -> None:
    max_workers = max(1, min(self.main_workers, len(self.ranges)))
    self.logger.info(
        "Starting %d Pyrocko download window(s) with up to %d worker(s).",
        len(self.ranges), max_workers
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
      futures = {
          executor.submit(self._download_window, window): window
          for window in self.ranges
      }
      for future in as_completed(futures):
        window = futures[future]
        try:
          future.result()
        except Exception as error:
          self.logger.error(
              "Pyrocko download failed for window %s to %s: %s",
              window[0], window[1], error
          )
          raise
    self.logger.info(
        "All %d Pyrocko download window(s) completed.", len(futures))


def data_downloader(args: Namespace) -> None:
  """Select a backend and download data described by a parsed CLI namespace.

  Args:
    args: Namespace returned by ``ogsutils.parse_downloader_args``. The
      ``pyrocko`` flag selects ``PyrockoDownloader``; otherwise the ObsPy
      implementation is selected.
  """
  downloader = (
      PyrockoDownloader(args) if args.pyrocko else ObsPyDownloader(args)
  )
  downloader.download()


if __name__ == "__main__":
  # Package-module entry point using the shared downloader argument parser.
  data_downloader(OGS_U.parse_downloader_args())

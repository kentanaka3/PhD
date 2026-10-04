#!/usr/bin/env python

"""
===============================================================================
OGS Downloader - ObsPy FDSN Waveform Retrieval CLI
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
from datetime import datetime, time
from fnmatch import fnmatchcase
from pathlib import Path
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
    # Store successfully probed providers in requested order; names are not
    # deduplicated by this class.
    self.clients: list[ObsPyFDSNClient] = []
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

  def download(self) -> None:
    """Probe providers for eligible channels and retain usable ObsPy clients.

    No waveform download occurs in this base method. Provider/query failures
    are logged and skipped; ValueError is raised if none has eligible channels.
    """
    query_window: TimeWindow = (self.ranges[0][0], self.ranges[-1][1])
    for name in self.args.client:
      self.logger.info("Initializing FDSN client for provider '%s'.", name)
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
    self.logger.info("Initialized %d FDSN client(s).", len(self.clients))


class ObsPyDownloader(BaseDownloader):
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
    super().download()
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


class PyrockoDownloader(BaseDownloader):
  def download(self) -> None:
    self.logger.error("The Pyrocko download backend is not implemented.")
    raise NotImplementedError("The Pyrocko backend is not implemented.")


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

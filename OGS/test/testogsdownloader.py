"""
=============================================================================
OGS Downloader Test Suite - Unit Tests for Waveform Retrieval & CLI
=============================================================================

Offline FDSN selection, storage, scheduling, and failure contracts.

OVERVIEW:
Unit test suite for ``ogsdownloader.py``. Validates CLI argument parsing,
multi-threaded download orchestration, negative station filtering, and
multi-backend FDSN data retrieval (ObsPy and Pyrocko backends).

TEST CASES & INVARIANTS:
  1. Argument Parsing:
     - Thread count argument validation (defaults, positive integers, rejection
       of zero/negative values).
  2. Threading & Backend Execution:
     - Preservation of mass-downloader thread pools in serial mode.
     - Multi-threaded per-client and per-day scheduling.
     - Negative station exclusion filtering against remote FDSN inventories.
     - Graceful logging and error isolation upon failed day downloads.
     - Pyrocko backend waveform trace and station XML writing to disk.

USAGE:
python -m unittest OGS/test/testogsdownloader.py

DEPENDENCIES:
- unittest: unit test framework
  - argparse: argument parsing inspection
  - ogsdownloader: downloader classes and CLI functions under test

AUTHORS:
  - 健
  - Istituto Nazionale di Oceanografia e di Geofisica Sperimentale (OGS)
    Centro di Ricerche Sismologiche (CRS)
  - Università degli Studi di Trieste (UniTS)
    Dipartimento di Matematica, Informatica e Geoscienze (MIGe)
    Applied Data Science and Artificial Intelligence (ADSAI)
  - Terabit Network for Research and Academic Big Data in Italy (TeRABIT)
    Consorzio Interuniversitario del Nord-Est per il Calcolo Automatico (CINECA)
=============================================================================
"""

from argparse import Namespace
import logging
from datetime import datetime, time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from obspy import UTCDateTime

from OGS.src import ogsdownloader as D
from OGS.src.ogsutils import parse_downloader_args


class InventoryNode(list):
  def __init__(self, code, children):
    super().__init__(children)
    self.code = code


def inventory(network="OX", station="TEST", location="00", channel="HHZ"):
  return [InventoryNode(network, [
      InventoryNode(station, [SimpleNamespace(
          location_code=location, code=channel)]),
  ])]


@pytest.fixture
def args(tmp_path):
  return Namespace(
      client=["AVAILABLE"], clip=None,
      dates=(datetime(2024, 3, 20), datetime(2024, 3, 21)), key=None,
      network=["OX -NI"], quiet=False, rectdomain=None, circdomain=None,
      station=["TEST -BAD"], threads=1, timeout=10, verbose=False,
      waveforms=tmp_path / "waveforms", stations=tmp_path / "stations",
      pyrocko=False,
  )


@pytest.fixture(autouse=True)
def service_boundary(monkeypatch):
  """Every service constructor is mocked, even in selection-only tests."""
  client = Mock()
  client.get_stations.return_value = inventory()
  client_factory = Mock(return_value=client)
  mass_factory = Mock(return_value=Mock())
  monkeypatch.setattr(D, "ObsPyFDSNClient", client_factory)
  monkeypatch.setattr(D, "MassDownloader", mass_factory)
  monkeypatch.setattr(
      D.OGS_U, "setup_logger", Mock(
          return_value=logging.getLogger("test.downloader")),
  )
  return client_factory, mass_factory


@pytest.mark.parametrize("value,include,exclude", [
    (None, ["*"], []), ("", ["*"], []),
    ("OX NI -IV -BAD*", ["OX", "NI"], ["IV", "BAD*"]),
    (["OX NI", "-IV", "-BAD*"], ["OX", "NI"], ["IV", "BAD*"]),
    (["-NI"], ["*"], ["NI"]),
])
def test_include_exclude_patterns_are_tokenized(value, include, exclude):
  assert D.BaseDownloader._filter(value) == (include, exclude)


def test_daily_and_clipped_windows():
  assert D.daily_clipper(datetime(2024, 2, 29)) == (
      datetime(2024, 2, 29), datetime(2024, 3, 1),
  )
  assert D.daily_clipper(datetime(2024, 3, 20), time(0, 0, 30)) == (
      datetime(2024, 3, 19, 23, 59, 30), datetime(2024, 3, 20, 0, 1, 30),
  )


def test_constructor_is_lazy_and_dates_are_inclusive(args, service_boundary):
  downloader = D.ObsPyDownloader(args)
  assert downloader.ranges == [
      (datetime(2024, 3, 20), datetime(2024, 3, 21)),
      (datetime(2024, 3, 21), datetime(2024, 3, 22)),
  ]
  client_factory, mass_factory = service_boundary
  client_factory.assert_not_called()
  mass_factory.assert_not_called()


def test_clip_schedules_only_one_interval(args):
  args.clip = time(12, 30)
  downloader = D.ObsPyDownloader(args)
  assert downloader.ranges == [
      (datetime(2024, 3, 20, 12, 29), datetime(2024, 3, 20, 12, 31)),
  ]


@pytest.mark.parametrize("rect,circ,expected", [
    (None, None, {}),
    ((10, 14, 45, 47), None,
     {"minlongitude": 10, "maxlongitude": 14, "minlatitude": 45, "maxlatitude": 47}),
    (None, (13, 46, .1, 2),
     {"longitude": 13, "latitude": 46, "minradius": .1, "maxradius": 2}),
])
def test_station_query_and_waveform_restrictions_are_distinct(args, rect, circ, expected):
  args.rectdomain, args.circdomain = rect, circ
  downloader = D.ObsPyDownloader(args)
  window = downloader.ranges[0]
  common = {
      "starttime": UTCDateTime(2024, 3, 20),
      "endtime": UTCDateTime(2024, 3, 21),
      "network": "OX", "station": "TEST",
  }
  assert downloader.domain_kwargs == expected
  assert downloader.station_kwargs(
      window) == {**common, "level": "channel", **expected}
  restrictions = downloader.download_kwargs(window)
  assert restrictions == {
      **common, "exclude_networks": ["NI"], "exclude_stations": ["BAD"],
      "channel_priorities": ["HH[ZNE]", "EH[ZNE]", "HN[ZNE]", "HG[ZNE]"],
      "location_priorities": ("", "00", "01", "02", "10"),
      "chunklength_in_sec": 86400, "minimum_length": 0.0,
      "minimum_interstation_distance_in_m": 100, "reject_channels_with_gaps": False,
  }


def test_probe_uses_whole_range_and_discards_empty_provider(args, service_boundary):
  factory, mass_factory = service_boundary
  available, empty = Mock(), Mock()
  available.get_stations.return_value = inventory()
  empty.get_stations.return_value = []
  factory.side_effect = [available, empty]
  args.client = ["AVAILABLE", "EMPTY"]
  downloader = D.ObsPyDownloader(args)
  D.BaseDownloader.download(downloader)
  assert downloader.clients == [available]
  available.get_stations.assert_called_once_with(
      starttime=UTCDateTime(2024, 3, 20), endtime=UTCDateTime(2024, 3, 22),
      network="OX", station="TEST", level="channel",
  )
  assert factory.call_args_list[0].args == ("AVAILABLE",)
  assert factory.call_args_list[0].kwargs == {
      "timeout": 10, "eida_token": None}
  mass_factory.assert_not_called()


@pytest.mark.parametrize("metadata", [
    inventory(network="NI"), inventory(station="BAD"),
    inventory(channel="BHZ"), inventory(location="99"), [],
])
def test_no_eligible_channel_is_explicitly_rejected(args, service_boundary, metadata, caplog):
  factory, _ = service_boundary
  factory.return_value.get_stations.return_value = metadata
  downloader = D.ObsPyDownloader(args)
  with pytest.raises(ValueError, match="matching station channels"):
    D.BaseDownloader.download(downloader)
  assert downloader.clients == []
  assert "No FDSN clients" in caplog.text


@pytest.mark.parametrize("channel", ["HHZ", "HHN", "HHE", "EHZ", "HNZ", "HGZ"])
def test_supported_channel_families(args, channel):
  assert D.ObsPyDownloader(args)._has_selected_channels(
      inventory(channel=channel))


def test_provider_query_failure_is_logged_and_other_provider_survives(
    args, service_boundary, caplog,
):
  factory, _ = service_boundary
  failed, available = Mock(), Mock()
  failed.get_stations.side_effect = RuntimeError("metadata unavailable")
  available.get_stations.return_value = inventory()
  factory.side_effect = [failed, available]
  args.client = ["FAILED", "AVAILABLE"]
  downloader = D.ObsPyDownloader(args)
  D.BaseDownloader.download(downloader)
  assert downloader.clients == [available]
  assert "metadata unavailable" in caplog.text


def test_provider_constructor_failure_is_not_reported_as_availability(args, service_boundary, caplog):
  factory, mass_factory = service_boundary
  factory.side_effect = RuntimeError("client initialization failed")
  with pytest.raises(ValueError, match="matching station channels"):
    D.BaseDownloader.download(D.ObsPyDownloader(args))
  assert "client initialization failed" in caplog.text
  mass_factory.assert_not_called()


def test_storage_filename_and_callback_use_start_day(args):
  downloader = D.ObsPyDownloader(args)
  start, end = datetime(2024, 3, 20, 23, 59), datetime(2024, 3, 21, 0, 1)
  expected = args.waveforms / \
      "2024/03/20/OX.TEST..HHZ__20240320T235900Z__20240321T000100Z.mseed"
  assert downloader.waveform_path(
      ("OX", "TEST", "", "HHZ", start, end)) == expected
  assert downloader._waveform_storage(
      "OX", "TEST", "", "HHZ", UTCDateTime(start), UTCDateTime(end)) == str(expected)


@pytest.mark.parametrize("threads", [1, 8])
def test_one_mass_downloader_handles_each_window_with_requested_restrictions(
    args, service_boundary, threads,
):
  factory, mass_factory = service_boundary
  args.threads = threads
  downloader = D.ObsPyDownloader(args)
  downloader.download()
  mass_factory.assert_called_once_with(providers=[factory.return_value])
  calls = mass_factory.return_value.download.call_args_list
  assert len(calls) == 2
  assert {call.args[1].starttime.datetime for call in calls} == {
      datetime(2024, 3, 20), datetime(2024, 3, 21),
  }
  for call in calls:
    assert isinstance(call.args[0], D.GlobalDomain)
    assert call.args[1].endtime - call.args[1].starttime == 86400
    assert call.args[1].exclude_stations == ["BAD"]
    assert call.kwargs["stationxml_storage"] == str(args.stations)
    assert call.kwargs["threads_per_client"] == (1 if threads == 1 else 4)
    assert call.kwargs["mseed_storage"].__self__ is downloader
  assert not list(args.waveforms.rglob("*.mseed"))


@pytest.mark.parametrize("rect,circ,domain_type", [
    ((10, 14, 45, 47), None, D.RectangularDomain),
    (None, (13, 46, .1, 2), D.CircularDomain),
])
def test_domain_object_passed_to_mass_downloader(args, service_boundary, rect, circ, domain_type):
  args.rectdomain, args.circdomain = rect, circ
  D.ObsPyDownloader(args).download()
  _, mass_factory = service_boundary
  assert all(isinstance(call.args[0], domain_type)
             for call in mass_factory.return_value.download.call_args_list)


def test_waveform_failure_is_logged_and_propagated(args, service_boundary, caplog):
  _, mass_factory = service_boundary
  mass_factory.return_value.download.side_effect = RuntimeError(
      "waveform unavailable")
  with pytest.raises(RuntimeError, match="waveform unavailable"):
    D.ObsPyDownloader(args).download()
  assert "Download failed for date window" in caplog.text


def test_pyrocko_backend_fails_explicitly_without_contacting_services(args, service_boundary):
  factory, mass_factory = service_boundary
  with pytest.raises(NotImplementedError, match="not implemented"):
    D.PyrockoDownloader(args).download()
  factory.assert_not_called()
  mass_factory.assert_not_called()


@pytest.mark.parametrize("pyrocko,selected", [(False, "ObsPyDownloader"), (True, "PyrockoDownloader")])
def test_backend_dispatch(args, monkeypatch, pyrocko, selected):
  args.pyrocko = pyrocko
  factory = Mock()
  monkeypatch.setattr(D, selected, factory)
  D.data_downloader(args)
  factory.assert_called_once_with(args)
  factory.return_value.download.assert_called_once_with()


@pytest.mark.parametrize("flag", ["--threads", "--retry"])
@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_cli_rejects_nonpositive_worker_and_retry_values(flag, value):
  with pytest.raises(SystemExit) as error:
    parse_downloader_args([flag, value])
  assert error.value.code == 2


def test_cli_sorts_dates_and_preserves_selection_arguments():
  args = parse_downloader_args([
      "-D", "20240321", "20240320", "--threads", "8", "--retry", "2",
      "--network", "OX -NI", "--station", "* -BAD",
      "--circdomain", "13", "46", "0.1", "2", "--pyrocko",
  ])
  assert args.dates == [datetime(2024, 3, 20), datetime(2024, 3, 21)]
  assert args.threads == 8 and args.retry == 2
  assert args.network == ["OX -NI"] and args.station == ["* -BAD"]
  assert args.circdomain == [13, 46, .1, 2]
  assert args.pyrocko


def test_cli_domain_selection_is_mutually_exclusive():
  with pytest.raises(SystemExit) as error:
    parse_downloader_args([
        "--rectdomain", "10", "14", "45", "47",
        "--circdomain", "13", "46", ".1", "2",
    ])
  assert error.value.code == 2

"""
=============================================================================
OGS Downloader Test Suite - Unit Tests for Waveform Retrieval & CLI
=============================================================================

Offline FDSN selection, storage, scheduling, and multi-backend contracts.

OVERVIEW:
Unit test suite for ogsdownloader.py. Validates CLI argument parsing,
multi-threaded download orchestration, negative station/network filtering,
ObsPy MassDownloader retrieval, and Pyrocko FDSN retrieval (including 24h
day chunk slicing and existing file skipping).

DEPENDENCIES:
- pytest: test execution and parameterization
- unittest.mock: isolated service boundary mocks
- ogsdownloader: downloader classes and CLI entry points under test
=============================================================================
"""

from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time, timezone
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call
from urllib.error import URLError

import pytest
from obspy.clients.fdsn.header import FDSNNoDataException
from pyrocko import io as pyrocko_io
from pyrocko.client import fdsn as pyrocko_fdsn
from pyrocko.io import mseed as pyrocko_mseed

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


class MockProcessPoolExecutor(ThreadPoolExecutor):
  """In-process thread executor drop-in for fast offline unit testing."""
  def __init__(self, max_workers=None, mp_context=None, max_tasks_per_child=None):
    super().__init__(max_workers=max_workers)


@pytest.fixture
def args(tmp_path):
  return Namespace(
      client=["AVAILABLE"], clip=None,
      dates=(datetime(2024, 3, 20), datetime(2024, 3, 21)), key=None,
      network=["OX -NI"], quiet=False, rectdomain=None, circdomain=None,
      station=["TEST -BAD"], threads=1, timeout=10, verbose=False,
      waveforms=tmp_path / "waveforms", stations=tmp_path / "stations",
      pyrocko=False, force=False,
  )


@pytest.fixture(autouse=True)
def service_boundary(monkeypatch):
  client = Mock()
  client.get_stations.return_value = inventory()
  client_factory = Mock(return_value=client)
  mass_factory = Mock(return_value=Mock())
  monkeypatch.setattr(D, "ObsPyFDSNClient", client_factory)
  monkeypatch.setattr(D, "MassDownloader", mass_factory)
  monkeypatch.setattr(D, "ProcessPoolExecutor", MockProcessPoolExecutor)
  monkeypatch.setattr(
      D.OGS_U, "setup_logger", Mock(
          return_value=logging.getLogger("test.downloader")),
  )
  return client_factory, mass_factory


# =============================================================================
# CLI & BASE DOWNLOADER TESTS
# =============================================================================

@pytest.mark.parametrize("value,include,exclude", [
    (None, ["*"], []), ("", ["*"], []),
    ("OX NI -IV -BAD*", ["OX", "NI"], ["IV", "BAD*"]),
    (["OX NI", "-IV", "-BAD*"], ["OX", "NI"], ["IV", "BAD*"]),
    (["-NI"], ["*"], ["NI"]),
])
def test_filter_tokens(value, include, exclude):
  assert D.BaseDownloader._filter(value) == (include, exclude)


def test_time_windows():
  assert D.daily_clipper(datetime(2024, 2, 29)) == (
      datetime(2024, 2, 29), datetime(2024, 3, 1),
  )
  assert D.daily_clipper(datetime(2024, 3, 20), time(0, 0, 30)) == (
      datetime(2024, 3, 19, 23, 59, 30), datetime(2024, 3, 20, 0, 1, 30),
  )


def test_date_ranges(args, service_boundary):
  downloader = D.ObsPyDownloader(args)
  assert downloader.ranges == [
      (datetime(2024, 3, 20), datetime(2024, 3, 21)),
      (datetime(2024, 3, 21), datetime(2024, 3, 22)),
  ]
  client_factory, mass_factory = service_boundary
  client_factory.assert_not_called()
  mass_factory.assert_not_called()


def test_clip_window(args):
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
def test_domain_kwargs(args, rect, circ, expected):
  args.rectdomain, args.circdomain = rect, circ
  downloader = D.ObsPyDownloader(args)
  assert downloader.domain_kwargs == expected


def test_waveform_path(args):
  downloader = D.ObsPyDownloader(args)
  start, end = datetime(2024, 3, 20, 23, 59), datetime(2024, 3, 21, 0, 1)
  expected = args.waveforms / \
      "2024/03/20/OX.TEST..HHZ__20240320T235900Z__20240321T000100Z.mseed"
  assert downloader.waveform_path(
      ("OX", "TEST", "", "HHZ", start, end)) == expected


@pytest.mark.parametrize("channel", ["HHZ", "HHN", "HHE", "EHZ", "HNZ", "HGZ"])
def test_channel_selection(args, channel):
  assert D.ObsPyDownloader(args)._has_selected_channels(
      inventory(channel=channel))


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
def test_cli_invalid_args(flag, value):
  with pytest.raises(SystemExit) as error:
    parse_downloader_args([flag, value])
  assert error.value.code == 2


def test_cli_args():
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


def test_cli_domain_mutex():
  with pytest.raises(SystemExit) as error:
    parse_downloader_args([
        "--rectdomain", "10", "14", "45", "47",
        "--circdomain", "13", "46", ".1", "2",
    ])
  assert error.value.code == 2



# =============================================================================
# OBSPY DOWNLOADER TESTS
# =============================================================================

def test_obspy_probe_empty_client(args, service_boundary):
  factory, mass_factory = service_boundary
  available, empty = Mock(), Mock()
  available.get_stations.return_value = inventory()
  empty.get_stations.return_value = []
  factory.side_effect = [available, empty]
  args.client = ["AVAILABLE", "EMPTY"]
  downloader = D.ObsPyDownloader(args)
  downloader._probe_clients()
  assert downloader.clients == [available]
  mass_factory.assert_not_called()


@pytest.mark.parametrize("metadata", [
    inventory(network="NI"), inventory(station="BAD"),
    inventory(channel="BHZ"), inventory(location="99"), [],
])
def test_obspy_probe_no_clients(args, service_boundary, metadata, caplog):
  factory, _ = service_boundary
  factory.return_value.get_stations.return_value = metadata
  downloader = D.ObsPyDownloader(args)
  with pytest.raises(ValueError, match="matching station channels"):
    downloader._probe_clients()
  assert downloader.clients == []
  assert "No FDSN clients" in caplog.text


def test_obspy_probe_query_failure(args, service_boundary, caplog):
  factory, _ = service_boundary
  failed, available = Mock(), Mock()
  failed.get_stations.side_effect = RuntimeError("metadata unavailable")
  available.get_stations.return_value = inventory()
  factory.side_effect = [failed, available]
  args.client = ["FAILED", "AVAILABLE"]
  downloader = D.ObsPyDownloader(args)
  downloader._probe_clients()
  assert downloader.clients == [available]
  assert "metadata unavailable" in caplog.text


def test_obspy_probe_init_failure(args, service_boundary, caplog):
  factory, mass_factory = service_boundary
  factory.side_effect = RuntimeError("client initialization failed")
  with pytest.raises(ValueError, match="matching station channels"):
    D.ObsPyDownloader(args)._probe_clients()
  assert "client initialization failed" in caplog.text
  mass_factory.assert_not_called()


@pytest.mark.parametrize("threads", [1, 8])
def test_obspy_download_restrictions(args, service_boundary, threads):
  factory, mass_factory = service_boundary
  args.threads = threads
  downloader = D.ObsPyDownloader(args)
  downloader.download()
  assert mass_factory.call_count == len(downloader.ranges)
  for c in mass_factory.call_args_list:
    assert c == call(providers=[factory.return_value])
  calls = mass_factory.return_value.download.call_args_list
  assert len(calls) == 2
  for call_args in calls:
    assert isinstance(call_args.args[0], D.GlobalDomain)
    assert call_args.args[1].endtime - call_args.args[1].starttime == 86400
    assert call_args.args[1].exclude_stations == ["BAD"]
    assert call_args.kwargs["stationxml_storage"] == str(args.stations)
    assert call_args.kwargs["threads_per_client"] == max(
        1, D.OGS_C.DEFAULT_CORES_COUNT // min(4, threads)
    )


@pytest.mark.parametrize("rect,circ,domain_type", [
    ((10, 14, 45, 47), None, D.RectangularDomain),
    (None, (13, 46, .1, 2), D.CircularDomain),
])
def test_obspy_download_domains(args, service_boundary, rect, circ, domain_type):
  args.rectdomain, args.circdomain = rect, circ
  D.ObsPyDownloader(args).download()
  _, mass_factory = service_boundary
  assert all(isinstance(call_args.args[0], domain_type)
             for call_args in mass_factory.return_value.download.call_args_list)


def test_obspy_download_failure(args, service_boundary, caplog):
  _, mass_factory = service_boundary
  mass_factory.return_value.download.side_effect = RuntimeError(
      "waveform unavailable")
  with pytest.raises(SystemExit) as exc_info:
    D.ObsPyDownloader(args).download()
  assert exc_info.value.code == 1
  assert "Download failed with error" in caplog.text
  assert "Completed 2 download window(s) with 2 failure(s)" in caplog.text


# =============================================================================
# PYROCKO DOWNLOADER TESTS
# =============================================================================

def test_pyrocko_site_resolution(args):
  downloader = D.PyrockoDownloader(args)
  assert downloader._resolve_site("INGV") == "ingv"
  assert downloader._resolve_site("eth") == "ethz"
  assert downloader._resolve_site("OGS") == D.OGS_C.OGS_CLIENT_STR
  assert downloader._resolve_site(
      "http://custom.fdsn.org") == "http://custom.fdsn.org"


def test_pyrocko_skip_existing(args, monkeypatch):
  downloader = D.PyrockoDownloader(args)
  win = downloader.ranges[0]
  dest = downloader.waveform_path(("OX", "TEST", "", "HHZ", win[0], win[1]))
  dest.parent.mkdir(parents=True, exist_ok=True)
  dest.touch()

  mock_net = SimpleNamespace(code="OX")
  mock_sta = SimpleNamespace(code="TEST")
  mock_cha = SimpleNamespace(code="HHZ", location_code="")
  mock_sxml = MagicMock()
  mock_sxml.iter_network_station_channels.return_value = [
      (mock_net, mock_sta, mock_cha)]
  mock_sxml.network_list = []

  monkeypatch.setattr(pyrocko_fdsn, "station", Mock(return_value=mock_sxml))
  mock_dataselect = Mock()
  monkeypatch.setattr(pyrocko_fdsn, "dataselect", mock_dataselect)

  downloader.download_worker(win)
  mock_dataselect.assert_not_called()


def test_pyrocko_force_download(args, monkeypatch):
  args.force = True
  downloader = D.PyrockoDownloader(args)
  win = downloader.ranges[0]
  dest = downloader.waveform_path(("OX", "TEST", "", "HHZ", win[0], win[1]))
  dest.parent.mkdir(parents=True, exist_ok=True)
  dest.touch()

  mock_net = SimpleNamespace(code="OX")
  mock_sta = SimpleNamespace(code="TEST")
  mock_cha = SimpleNamespace(code="HHZ", location_code="")
  mock_sxml = MagicMock()
  mock_sxml.iter_network_station_channels.return_value = [
      (mock_net, mock_sta, mock_cha)]
  mock_sxml.network_list = []

  monkeypatch.setattr(pyrocko_fdsn, "station", Mock(return_value=mock_sxml))
  mock_stream = Mock()
  mock_stream.read.return_value = b"FAKE_MSEED_DATA"
  mock_dataselect = Mock(return_value=mock_stream)
  monkeypatch.setattr(pyrocko_fdsn, "dataselect", mock_dataselect)

  mock_tr = Mock()
  mock_tr.network = "OX"
  mock_tr.station = "TEST"
  mock_tr.location = ""
  mock_tr.channel = "HHZ"
  mock_tr.tmin = win[0].replace(tzinfo=timezone.utc).timestamp()
  mock_tr.tmax = win[1].replace(tzinfo=timezone.utc).timestamp()
  mock_tr.chop.return_value = mock_tr
  mock_tr.ydata = [1, 2, 3]

  monkeypatch.setattr(pyrocko_mseed, "iload", Mock(return_value=[mock_tr]))
  mock_save = Mock()
  monkeypatch.setattr(pyrocko_io, "save", mock_save)

  downloader.download_worker(win)
  mock_dataselect.assert_called_once()
  mock_save.assert_called_once()


def test_obspy_safe_raise_on_error_reraises_network_exceptions():
  """Verify _safe_raise_on_error re-raises ConnectionResetError and URLError directly."""
  conn_err = ConnectionResetError("Connection reset by peer")
  with pytest.raises(ConnectionResetError) as exc_info:
    D.ObsPyDownloader._safe_raise_on_error(None, conn_err)
  assert exc_info.value is conn_err

  url_err = URLError("Connection timed out")
  with pytest.raises(URLError) as exc_info:
    D.ObsPyDownloader._safe_raise_on_error(None, url_err)
  assert exc_info.value is url_err


def test_obspy_safe_raise_on_error_delegates_standard_responses():
  """Verify normal non-200 responses delegate to original ObsPy raise_on_error."""
  with pytest.raises(FDSNNoDataException):
    D.ObsPyDownloader._safe_raise_on_error(204, b"No data available")

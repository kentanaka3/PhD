import argparse
import io
import json
import os
import sys
import tempfile
import threading
import types
import unittest
import unittest.mock
from datetime import datetime, timezone
from pathlib import Path

THIS_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(THIS_DIR + "/../src"))


def _install_obspy_import_stub():
  obspy_module = types.ModuleType("obspy")
  obspy_module.__path__ = []
  obspy_module.UTCDateTime = datetime
  sys.modules.setdefault("obspy", obspy_module)


try:
  import obspy
except ModuleNotFoundError:
  _install_obspy_import_stub()

import ogsconstants as OGS_C
from ogsdownloader import data_downloader, parse_arguments
import ogsdownloader


class DownloadRecorder:
  def __init__(self, failed_dates=None):
    self.failed_dates = set(failed_dates or [])
    self.failed_station_sites = set()
    self.failed_data_sites = set()
    self.lock = threading.Lock()
    self.clients = []
    self.domains = []
    self.restrictions = []
    self.mass_downloaders = []
    self.downloads = []
    self.station_queries = []
    self.parsed_streams = []
    self.trace_files = []
    self.pyrocko_station_calls = []
    self.pyrocko_station_queries = []
    self.pyrocko_selection_calls = []
    self.pyrocko_dataselect_calls = []
    self.station_xml_files = []
    self.closed_waveform_streams = []

  def append(self, name, value):
    with self.lock:
      getattr(self, name).append(value)


def _package(name):
  module = types.ModuleType(name)
  module.__path__ = []
  return module


def _fake_obspy_modules(recorder):
  obspy_module = _package("obspy")
  clients_module = _package("obspy.clients")
  fdsn_module = _package("obspy.clients.fdsn")
  mass_downloader_module = _package("obspy.clients.fdsn.mass_downloader")
  domain_module = _package("obspy.clients.fdsn.mass_downloader.domain")
  obspy_module.UTCDateTime = datetime

  class FakeTrace:
    def __init__(self, network, station, location, channel, starttime, endtime):
      self.stats = types.SimpleNamespace(
        network=network,
        station=station,
        location=location,
        channel=channel,
        starttime=starttime,
        endtime=endtime,
      )

    def write(self, filename, format=None):
      path = Path(filename)
      path.write_bytes(b"trace")
      recorder.append("trace_files", path)

  def read(stream, format=None):
    selection = json.loads(stream.read().decode())
    traces = []
    for network, station, location, channel, tmin, tmax in selection:
      traces.append(FakeTrace(
        network,
        station,
        location,
        channel,
        datetime.fromtimestamp(tmin, tz=timezone.utc).replace(tzinfo=None),
        datetime.fromtimestamp(tmax, tz=timezone.utc).replace(tzinfo=None),
      ))
    recorder.append("parsed_streams", {"format": format, "count": len(traces)})
    return traces

  class RectangularDomain:
    def __init__(self, **kwargs):
      self.kwargs = kwargs
      recorder.append("domains", self)

  class CircularDomain:
    def __init__(self, **kwargs):
      self.kwargs = kwargs
      recorder.append("domains", self)

  class Restrictions:
    def __init__(self, **kwargs):
      self.kwargs = kwargs
      recorder.append("restrictions", self)

  class Client:
    def __init__(self, name):
      self.name = name
      recorder.append("clients", self)

    def set_eida_token(self, key, validate=True):
      self.key = key
      self.validate = validate

    def get_stations(self, **kwargs):
      recorder.append("station_queries", {
        "client": self.name,
        "kwargs": kwargs,
      })
      return [[
        types.SimpleNamespace(code="AQU"),
        types.SimpleNamespace(code="MTRA"),
        types.SimpleNamespace(code="PZI"),
        types.SimpleNamespace(code="TEST"),
      ]]

  class MassDownloader:
    def __init__(self, providers):
      self.providers = list(providers)
      recorder.append("mass_downloaders", self)

    def download(self, domain, restrictions, **kwargs):
      recorder.append("downloads", {
        "domain": domain,
        "restrictions": restrictions,
        "kwargs": kwargs,
      })
      download_date = restrictions.kwargs["starttime"].strftime(
        OGS_C.YYYYMMDD_FMT)
      if download_date in recorder.failed_dates:
        raise RuntimeError("download failed")

  domain_module.RectangularDomain = RectangularDomain
  domain_module.CircularDomain = CircularDomain
  mass_downloader_module.Restrictions = Restrictions
  mass_downloader_module.MassDownloader = MassDownloader
  fdsn_module.Client = Client
  fdsn_module.mass_downloader = mass_downloader_module
  clients_module.fdsn = fdsn_module
  mass_downloader_module.domain = domain_module
  obspy_module.clients = clients_module
  obspy_module.read = read

  return {
    "obspy": obspy_module,
    "obspy.clients": clients_module,
    "obspy.clients.fdsn": fdsn_module,
    "obspy.clients.fdsn.mass_downloader": mass_downloader_module,
    "obspy.clients.fdsn.mass_downloader.domain": domain_module,
  }


def _fake_pyrocko_modules(recorder):
  pyrocko_module = _package("pyrocko")
  client_module = _package("pyrocko.client")
  fdsn_module = types.ModuleType("pyrocko.client.fdsn")
  pyrocko_module.__version__ = "test"

  class EmptyResult(Exception):
    pass

  class FakeChannel:
    def __init__(self, name):
      self.name = name

  class FakeStation:
    def __init__(self, network, station, location, channels):
      self.network = network
      self.station = station
      self.location = location
      self._channels = [FakeChannel(channel) for channel in channels]

    def get_channels(self):
      return list(self._channels)

  class FakeStationXML:
    def __init__(self, site):
      self.site = site

    def get_pyrocko_stations(self, **kwargs):
      recorder.append("pyrocko_station_queries", {
        "site": self.site,
        "kwargs": kwargs,
      })
      network = ("".join(char for char in self.site if char.isalnum())[:2] or "NW").upper()
      return [FakeStation(network, f"{network}01", "", ["HHZ", "HHN", "HHE"])]

    def dump_xml(self, filename=None, header=False):
      path = Path(filename)
      path.write_text(f"<stations site='{self.site}' header='{header}' />")
      recorder.append("station_xml_files", path)

  class FakeWaveformStream:
    def __init__(self, selection):
      self.selection = selection

    def read(self):
      return json.dumps(self.selection).encode()

    def close(self):
      recorder.append("closed_waveform_streams", True)

  def station(site="geofon", **kwargs):
    recorder.append("pyrocko_station_calls", {"site": site, "kwargs": kwargs})
    if site in recorder.failed_station_sites:
      raise EmptyResult()
    return FakeStationXML(site)

  def make_data_selection(stations, tmin, tmax, channel_prio=None):
    recorder.append("pyrocko_selection_calls", {
      "tmin": tmin,
      "tmax": tmax,
      "channel_prio": channel_prio,
    })
    selection = []
    for station in stations:
      for channel in station.get_channels():
        selection.append(
          (station.network, station.station, station.location, channel.name, tmin, tmax)
        )
    return selection

  def dataselect(site="geofon", token=None, selection=None, **kwargs):
    recorder.append("pyrocko_dataselect_calls", {
      "site": site,
      "token": token,
      "selection": selection,
      "kwargs": kwargs,
    })
    if site in recorder.failed_data_sites:
      raise EmptyResult()
    return FakeWaveformStream(selection)

  fdsn_module.EmptyResult = EmptyResult
  fdsn_module.station = station
  fdsn_module.make_data_selection = make_data_selection
  fdsn_module.dataselect = dataselect
  client_module.fdsn = fdsn_module
  pyrocko_module.client = client_module

  return {
    "pyrocko": pyrocko_module,
    "pyrocko.client": client_module,
    "pyrocko.client.fdsn": fdsn_module,
  }


class TestOGSDownloaderArguments(unittest.TestCase):
  def test_threads_argument_defaults_and_accepts_positive_values(self):
    with unittest.mock.patch.object(sys, "argv", ["ogsdownloader.py"]):
      self.assertEqual(parse_arguments().threads, 1)

    with unittest.mock.patch.object(
        sys, "argv", ["ogsdownloader.py", "--threads", "3"]):
      self.assertEqual(parse_arguments().threads, 3)

  def test_threads_argument_rejects_non_positive_values(self):
    with unittest.mock.patch.object(
        sys, "argv", ["ogsdownloader.py", "--threads", "0"]), \
        unittest.mock.patch("sys.stderr", new=io.StringIO()):
      with self.assertRaises(SystemExit) as error_context:
        parse_arguments()

    self.assertEqual(error_context.exception.code, 2)


class TestOGSDownloaderThreading(unittest.TestCase):
  def _args(self, directory, start="20240101", end="20240101", threads=1,
            pyrocko=False, client=None, key=None, network=None, station=None):
    return argparse.Namespace(
      verbose=False,
      silent=True,
      review=None,
      pyrocko=pyrocko,
      rectdomain=[9.5, 15.0, 44.3, 47.5],
      circdomain=None,
      dates=[
        datetime.strptime(start, OGS_C.YYYYMMDD_FMT),
        datetime.strptime(end, OGS_C.YYYYMMDD_FMT),
      ],
      directory=directory,
      client=client or ["TEST"],
      key=key,
      network=network or ["*"],
      station=station or ["*"],
      clip=None,
      threads=threads,
    )

  def _run_downloader(self, args, failed_dates=None):
    recorder = DownloadRecorder(failed_dates=failed_dates)
    logger = unittest.mock.MagicMock()
    modules = _fake_obspy_modules(recorder)

    with unittest.mock.patch.dict(sys.modules, modules), \
        unittest.mock.patch.object(
          ogsdownloader.OGS_U, "setup_logger", return_value=logger):
      data_downloader(args)

    return recorder, logger

  def _run_pyrocko_downloader(self, args, failed_station_sites=None,
                              failed_data_sites=None):
    recorder = DownloadRecorder()
    recorder.failed_station_sites = set(failed_station_sites or [])
    recorder.failed_data_sites = set(failed_data_sites or [])
    logger = unittest.mock.MagicMock()
    modules = _fake_obspy_modules(recorder)
    modules.update(_fake_pyrocko_modules(recorder))

    with unittest.mock.patch.dict(sys.modules, modules), \
        unittest.mock.patch.object(
          ogsdownloader.OGS_U, "setup_logger", return_value=logger):
      data_downloader(args)

    return recorder, logger

  def _download_dates(self, recorder):
    return {
      download["restrictions"].kwargs["starttime"].strftime(
        OGS_C.YYYYMMDD_FMT)
      for download in recorder.downloads
    }

  def test_serial_download_preserves_mass_downloader_thread_default(self):
    with tempfile.TemporaryDirectory() as directory:
      args = self._args(Path(directory), threads=1)

      recorder, logger = self._run_downloader(args)

    self.assertEqual(len(recorder.downloads), 1)
    self.assertNotIn("threads_per_client", recorder.downloads[0]["kwargs"])
    self.assertEqual(self._download_dates(recorder), {"20240101"})
    logger.error.assert_not_called()

  def test_threaded_download_uses_one_obspy_thread_per_client_for_each_day(self):
    with tempfile.TemporaryDirectory() as directory:
      args = self._args(Path(directory), end="20240102", threads=2)

      recorder, logger = self._run_downloader(args)

    self.assertEqual(len(recorder.downloads), 2)
    self.assertEqual(self._download_dates(recorder), {"20240101", "20240102"})
    for download in recorder.downloads:
      self.assertEqual(download["kwargs"]["threads_per_client"], 1)
    logger.error.assert_not_called()

  def test_obspy_negative_station_filter_resolves_inventory_station_string(self):
    with tempfile.TemporaryDirectory() as directory:
      args = self._args(
        Path(directory), client=[OGS_C.INGV_CLIENT_STR], network=["IV"],
        station=["*", "-AQU", "-MTRA", "-TEST"])

      recorder, logger = self._run_downloader(args)

    self.assertEqual(len(recorder.station_queries), 1)
    self.assertEqual(recorder.station_queries[0]["kwargs"]["network"], "IV")
    self.assertEqual(recorder.station_queries[0]["kwargs"]["station"], "*")
    self.assertEqual(len(recorder.downloads), 1)
    restrictions = recorder.downloads[0]["restrictions"].kwargs
    self.assertEqual(restrictions["network"], "IV")
    self.assertEqual(restrictions["station"], "PZI")
    logger.error.assert_not_called()

  def test_failed_download_logs_error_without_success_for_failed_day(self):
    with tempfile.TemporaryDirectory() as directory:
      args = self._args(Path(directory), end="20240102", threads=1)

      recorder, logger = self._run_downloader(
        args, failed_dates={"20240102"})

    self.assertEqual(len(recorder.downloads), 2)
    self.assertEqual(self._download_dates(recorder), {"20240101", "20240102"})

    error_dates = [call.args[1] for call in logger.error.call_args_list]
    self.assertIn("20240102", error_dates)

    success_dates = [
      call.args[1]
      for call in logger.info.call_args_list
      if call.args and call.args[0] == "Downloaded data for date: %s"
    ]
    self.assertIn("20240101", success_dates)
    self.assertNotIn("20240102", success_dates)

  def test_pyrocko_download_writes_station_xml_and_trace_files(self):
    with tempfile.TemporaryDirectory() as directory:
      args = self._args(
        Path(directory), end="20240102", threads=2, pyrocko=True,
        client=[OGS_C.IRIS_CLIENT_STR])

      recorder, logger = self._run_pyrocko_downloader(args)

      waveform_files = sorted(Path(directory).glob("*/*/*/*.mseed"))
      station_files = sorted((Path(directory) / OGS_C.STATION_STR).glob("*.xml"))

    self.assertEqual(len(recorder.pyrocko_station_calls), 2)
    self.assertEqual(len(recorder.pyrocko_dataselect_calls), 2)
    self.assertEqual(len(waveform_files), 6)
    self.assertEqual(len(station_files), 2)
    self.assertTrue(all("__iris-" in waveform.name for waveform in waveform_files))
    self.assertEqual(
      recorder.pyrocko_selection_calls[0]["channel_prio"],
      ogsdownloader.PYROCKO_CHANNEL_PRIORITIES,
    )
    logger.error.assert_not_called()

  def test_pyrocko_reads_token_file_for_eida_clients_only(self):
    with tempfile.TemporaryDirectory() as directory:
      token_path = Path(directory) / "token.txt"
      token_path.write_text("TESTTOKEN\n")
      args = self._args(
        Path(directory), pyrocko=True,
        client=[OGS_C.INGV_CLIENT_STR, OGS_C.IRIS_CLIENT_STR],
        key=token_path)

      recorder, logger = self._run_pyrocko_downloader(args)

    tokens_by_site = {
      call["site"]: call["token"] for call in recorder.pyrocko_dataselect_calls
    }
    self.assertEqual(tokens_by_site["ingv"], b"TESTTOKEN")
    self.assertIsNone(tokens_by_site["iris"])
    logger.error.assert_not_called()


if __name__ == "__main__": unittest.main()
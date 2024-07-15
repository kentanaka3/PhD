#!/bin/python
import os
from pathlib import Path
PRJ_PATH = Path(os.path.dirname(__file__)).parent
SRC_PATH = os.path.join(PRJ_PATH, "src")
DATA_PATH = os.path.join(PRJ_PATH, "data", "test")
TEST_PATH = os.path.join(DATA_PATH, "waveforms")
import sys
# Add to path
if SRC_PATH not in sys.path: sys.path.append(SRC_PATH)
import unittest
import shutil
import json
from AdriaArray import *

EXPECTED_STR = "expected"

def timedeltafmt(string):
  numbers = [float(n) for n in string.split(":")]
  return timedelta(hours=numbers[0], minutes=numbers[1], seconds=numbers[2])

class TestArgparse(unittest.TestCase):
  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "-v"])
  def test_non_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])
    self.assertEqual(args.directory,
                     os.path.join(PRJ_PATH, "data", "waveforms"))
    self.assertEqual(args.groups, [BEG_DATE_STR, NETWORK_STR, STATION_STR])
    self.assertEqual(args.julian, False)
    self.assertEqual(args.models, [PHASENET_STR, EQTRANSFORMER_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.train, False)
    self.assertEqual(args.verbose, True)
    self.assertEqual(args.weights, [INSTANCE_STR, ORIGINAL_STR, STEAD_STR,
                                    SCEDC_STR])

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "-M", PHASENET_STR, "-v"])
  def test_model_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])
    self.assertEqual(args.directory,
                     os.path.join(PRJ_PATH, "data", "waveforms"))
    self.assertEqual(args.groups, [BEG_DATE_STR, NETWORK_STR, STATION_STR])
    self.assertEqual(args.julian, False)
    self.assertEqual(args.models, [PHASENET_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.train, False)
    self.assertEqual(args.verbose, True)
    self.assertEqual(args.weights, [INSTANCE_STR, ORIGINAL_STR, STEAD_STR,
                                    SCEDC_STR])

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "-M", PHASENET_STR, "-v"])
  def test_models_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])
    self.assertEqual(args.directory,
                     os.path.join(PRJ_PATH, "data", "waveforms"))
    self.assertEqual(args.groups, [BEG_DATE_STR, NETWORK_STR, STATION_STR])
    self.assertEqual(args.julian, False)
    self.assertEqual(args.models, [PHASENET_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.train, False)
    self.assertEqual(args.verbose, True)
    self.assertEqual(args.weights, [INSTANCE_STR, ORIGINAL_STR, STEAD_STR,
                                    SCEDC_STR])

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "-W", INSTANCE_STR, "-v"])
  def test_weight_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])
    self.assertEqual(args.directory,
                     os.path.join(PRJ_PATH, "data", "waveforms"))
    self.assertEqual(args.groups, [BEG_DATE_STR, NETWORK_STR, STATION_STR])
    self.assertEqual(args.julian, False)
    self.assertEqual(args.models, [PHASENET_STR, EQTRANSFORMER_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.train, False)
    self.assertEqual(args.verbose, True)
    self.assertEqual(args.weights, [INSTANCE_STR])

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-W", INSTANCE_STR, ORIGINAL_STR,
                        "-v"])
  def test_weights_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])
    self.assertEqual(args.directory,
                     os.path.join(PRJ_PATH, "data", "waveforms"))
    self.assertEqual(args.groups, [BEG_DATE_STR, NETWORK_STR, STATION_STR])
    self.assertEqual(args.julian, False)
    self.assertEqual(args.models, [PHASENET_STR, EQTRANSFORMER_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.train, False)
    self.assertEqual(args.verbose, True)
    self.assertEqual(args.weights, [INSTANCE_STR, ORIGINAL_STR])

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "-G", BEG_DATE_STR, "-v"])
  def test_group_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])
    self.assertEqual(args.directory,
                     os.path.join(PRJ_PATH, "data", "waveforms"))
    self.assertEqual(args.groups, [BEG_DATE_STR])
    self.assertEqual(args.julian, False)
    self.assertEqual(args.models, [PHASENET_STR, EQTRANSFORMER_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.train, False)
    self.assertEqual(args.verbose, True)
    self.assertEqual(args.weights, [INSTANCE_STR, ORIGINAL_STR, STEAD_STR,
                                    SCEDC_STR])

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-G", BEG_DATE_STR, NETWORK_STR,
                        "-v"])
  def test_groups_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])
    self.assertEqual(args.directory,
                     os.path.join(PRJ_PATH, "data", "waveforms"))
    self.assertEqual(args.groups, [BEG_DATE_STR, NETWORK_STR])
    self.assertEqual(args.julian, False)
    self.assertEqual(args.models, [PHASENET_STR, EQTRANSFORMER_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.train, False)
    self.assertEqual(args.verbose, True)
    self.assertEqual(args.weights, [INSTANCE_STR, ORIGINAL_STR, STEAD_STR,
                                    SCEDC_STR])

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-D", "230601", "230731", "-v"])
  def test_range_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])
    self.assertEqual(args.directory,
                     os.path.join(PRJ_PATH, "data", "waveforms"))
    self.assertEqual(args.groups, [BEG_DATE_STR, NETWORK_STR, STATION_STR])
    self.assertEqual(args.julian, False)
    self.assertEqual(args.models, [PHASENET_STR, EQTRANSFORMER_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.train, False)
    self.assertEqual(args.verbose, True)
    self.assertEqual(args.weights, [INSTANCE_STR, ORIGINAL_STR, STEAD_STR,
                                    SCEDC_STR])

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "-v"])
  def test_verbose_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])
    self.assertEqual(args.directory,
                     os.path.join(PRJ_PATH, "data", "waveforms"))
    self.assertEqual(args.groups, [BEG_DATE_STR, NETWORK_STR, STATION_STR])
    self.assertEqual(args.julian, False)
    self.assertEqual(args.models, [PHASENET_STR, EQTRANSFORMER_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.train, False)
    self.assertEqual(args.verbose, True)
    self.assertEqual(args.weights, [INSTANCE_STR, ORIGINAL_STR, STEAD_STR,
                                    SCEDC_STR])

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "-T", "-v"])
  def test_train_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])
    self.assertEqual(args.directory,
                     os.path.join(PRJ_PATH, "data", "waveforms"))
    self.assertEqual(args.groups, [BEG_DATE_STR, NETWORK_STR, STATION_STR])
    self.assertEqual(args.julian, False)
    self.assertEqual(args.models, [PHASENET_STR, EQTRANSFORMER_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.train, True)
    self.assertEqual(args.verbose, True)
    self.assertEqual(args.weights, [INSTANCE_STR, ORIGINAL_STR, STEAD_STR,
                                    SCEDC_STR])

class TestWaveformTable(unittest.TestCase):
  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "-v"])
  def test_non_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [12]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", '-N', "IV", "-v"])
  def test_network_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [12]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", '-N', "SI", "ST", "-v"])
  def test_networks_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [12]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", '-S', "LUSI", "-v"])
  def test_station_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [12]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-S', "LUSI", "PANI", "-v"])
  def test_stations_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [12]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", '-C', "EHZ", "-v"])
  def test_channel_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [4]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", '-C', "HHZ", "HHN", "-v"])
  def test_channels_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [8]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-v", '-N', "SI", "ST", '-S', "MAGA",
                        "LUSI"])
  def test_networks_stations_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [12]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-N', "SI", "ST", '-C', "HHN", "HHZ",
                        "-v"])
  def test_networks_channels_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [8]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-S', "MAGA", "LUSI", '-C', "HHN",
                        "HHZ", "-v"])
  def test_stations_channels_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [8]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-S', "MAGA", "LUSI", '-C', "HHN",
                        "HHZ", '-D', "230605", "230606", "-v"])
  def test_stations_channels_dates_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [8]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

class TestReadTraces(unittest.TestCase):
  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "-v"])
  def test_non_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "-G", BEG_DATE_STR, "-v"])
  def test_group_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-v", "-G", BEG_DATE_STR, NETWORK_STR,
                        STATION_STR])
  def test_groups_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)

class TestModel(unittest.TestCase):
  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-v", "-d", TEST_PATH,
                        "-G", BEG_DATE_STR, NETWORK_STR, STATION_STR, "-M",
                        PHASENET_STR, EQTRANSFORMER_STR])
  def test_classification(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    for x, y in list(itertools.product(args.models, args.weights)):
      model = get_model(x, y)
      for categories, trace_files in WAVEFORMS_DATA:
        output = classify_stream(categories, trace_files, model, x, y, args)
        pass

class TestPickParser(unittest.TestCase):
  def test_parse_pick(self):
    global DATA_PATH
    MNL_DATA_PATH = os.path.join(DATA_PATH, "manual")
    filename = os.path.join(MNL_DATA_PATH, "manual.dat")
    events = event_parser(filename)
    # with open(os.path.join(MNL_DATA_PATH, EXPECTED_STR + JSON_EXT), 'w') as fp:
    #   json.dump(events, fp, default=str)
    with open(os.path.join(MNL_DATA_PATH, EXPECTED_STR + JSON_EXT), 'r') as fr:
      expected = json.load(fr)
    for key, event in events.items():
      for s, station in enumerate(event):
        expected[str(key)][s][BEG_DATE_STR] = \
          UTCDateTime(expected[str(key)][s][BEG_DATE_STR])
        if expected[str(key)][s][P_TIME_STR] is not None:
          expected[str(key)][s][P_TIME_STR] = \
            timedeltafmt(expected[str(key)][s][P_TIME_STR])
        if expected[str(key)][s][S_TIME_STR] is not None:
          expected[str(key)][s][S_TIME_STR] = \
            timedeltafmt(expected[str(key)][s][S_TIME_STR])
    for key, event in events.items():
      for s, station in enumerate(event):
        for k, v in station.items():
          self.assertEqual(v, expected[str(key)][s][k])

if __name__ == "__main__":
  unittest.main()
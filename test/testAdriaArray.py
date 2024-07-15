#!/bin/python
import os
from pathlib import Path
PRJ_PATH = Path(os.path.dirname(__file__)).parent
SRC_PATH = os.path.join(PRJ_PATH, "src")
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
  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-v", "-d", RAW_DATA_PATH])
  def test_non_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])
    self.assertEqual(args.groups, [BEG_DATE_STR, NETWORK_STR, STATION_STR])
    self.assertEqual(args.julian, False)
    self.assertEqual(args.models, [PHASENET_STR, EQTRANSFORMER_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.train, False)
    self.assertEqual(args.verbose, True)
    self.assertEqual(args.weights, [INSTANCE_STR, ORIGINAL_STR, STEAD_STR,
                                    SCEDC_STR])

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-M", PHASENET_STR, "-v", "-d",
                        RAW_DATA_PATH])
  def test_model_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])
    self.assertEqual(args.groups, [BEG_DATE_STR, NETWORK_STR, STATION_STR])
    self.assertEqual(args.julian, False)
    self.assertEqual(args.models, [PHASENET_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.train, False)
    self.assertEqual(args.verbose, True)
    self.assertEqual(args.weights, [INSTANCE_STR, ORIGINAL_STR, STEAD_STR,
                                    SCEDC_STR])

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "-M", PHASENET_STR, "-v",
                                    "-d", RAW_DATA_PATH])
  def test_models_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])
    self.assertEqual(args.groups, [BEG_DATE_STR, NETWORK_STR, STATION_STR])
    self.assertEqual(args.julian, False)
    self.assertEqual(args.models, [PHASENET_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.train, False)
    self.assertEqual(args.verbose, True)
    self.assertEqual(args.weights, [INSTANCE_STR, ORIGINAL_STR, STEAD_STR,
                                    SCEDC_STR])

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-W", INSTANCE_STR, "-v", "-d",
                        RAW_DATA_PATH])
  def test_weight_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])
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
                        "-d", RAW_DATA_PATH, "-v"])
  def test_weights_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])
    self.assertEqual(args.groups, [BEG_DATE_STR, NETWORK_STR, STATION_STR])
    self.assertEqual(args.julian, False)
    self.assertEqual(args.models, [PHASENET_STR, EQTRANSFORMER_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.train, False)
    self.assertEqual(args.verbose, True)
    self.assertEqual(args.weights, [INSTANCE_STR, ORIGINAL_STR])

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-G", BEG_DATE_STR, "-v", "-d",
                        RAW_DATA_PATH])
  def test_group_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])
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
                       ["AdriaArray.py", "-G", BEG_DATE_STR, NETWORK_STR, "-d",
                        RAW_DATA_PATH, "-v"])
  def test_groups_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])
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
                       ["AdriaArray.py", "-D", "230601", "230731", "-v", "-d",
                        RAW_DATA_PATH])
  def test_range_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])
    self.assertEqual(args.groups, [BEG_DATE_STR, NETWORK_STR, STATION_STR])
    self.assertEqual(args.julian, False)
    self.assertEqual(args.models, [PHASENET_STR, EQTRANSFORMER_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.train, False)
    self.assertEqual(args.verbose, True)
    self.assertEqual(args.weights, [INSTANCE_STR, ORIGINAL_STR, STEAD_STR,
                                    SCEDC_STR])

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-v", "-d", RAW_DATA_PATH])
  def test_verbose_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])
    self.assertEqual(args.groups, [BEG_DATE_STR, NETWORK_STR, STATION_STR])
    self.assertEqual(args.julian, False)
    self.assertEqual(args.models, [PHASENET_STR, EQTRANSFORMER_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.train, False)
    self.assertEqual(args.verbose, True)
    self.assertEqual(args.weights, [INSTANCE_STR, ORIGINAL_STR, STEAD_STR,
                                    SCEDC_STR])

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-T", "-v", "-d", RAW_DATA_PATH])
  def test_train_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])
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
  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-v", "-d", RAW_DATA_PATH])
  def test_non_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [0]*len(WAVEFORMS_DATA)
    SIZE[0] = 12
    SIZE[1] = 12
    SIZE[2] = 12
    SIZE[3] = 12
    SIZE[4] = 12
    SIZE[5] = 12
    SIZE[6] = 12
    SIZE[7] = 12
    SIZE[8] = 12
    SIZE[9] = 12
    SIZE[10] = 12
    SIZE[11] = 12
    SIZE[12] = 12
    SIZE[13] = 12
    SIZE[14] = 12
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-N', "IV", "-d", RAW_DATA_PATH,
                        "-v"])
  def test_network_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [0]*len(WAVEFORMS_DATA)
    SIZE[0] = 12
    SIZE[1] = 12
    SIZE[2] = 12
    SIZE[3] = 12
    SIZE[4] = 12
    SIZE[5] = 12
    SIZE[6] = 12
    SIZE[7] = 12
    SIZE[8] = 12
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-N', "SI", "ST", "-v", "-d",
                        RAW_DATA_PATH])
  def test_networks_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [0]*len(WAVEFORMS_DATA)
    SIZE[0] = 12
    SIZE[1] = 12
    SIZE[2] = 12
    SIZE[3] = 12
    SIZE[4] = 12
    SIZE[5] = 12
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-S', "LUSI", "-v", "-d",
                        RAW_DATA_PATH])
  def test_station_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [0]*len(WAVEFORMS_DATA)
    SIZE[0] = 12
    SIZE[1] = 12
    SIZE[2] = 12
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-S', "LUSI", "PANI", "-v", "-d",
                        RAW_DATA_PATH])
  def test_stations_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [0]*len(WAVEFORMS_DATA)
    SIZE[0] = 12
    SIZE[1] = 12
    SIZE[2] = 12
    SIZE[3] = 12
    SIZE[4] = 12
    SIZE[5] = 12
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-C', "EHZ", "-v", "-d",
                        RAW_DATA_PATH])
  def test_channel_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [0]*len(WAVEFORMS_DATA)
    SIZE[0] = 4
    SIZE[1] = 4
    SIZE[2] = 4
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-C', "HHZ", "HHN", "-v", "-d",
                        RAW_DATA_PATH])
  def test_channels_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [0]*len(WAVEFORMS_DATA)
    SIZE[0] = 8
    SIZE[1] = 8
    SIZE[2] = 8
    SIZE[3] = 8
    SIZE[4] = 8
    SIZE[5] = 8
    SIZE[6] = 8
    SIZE[7] = 8
    SIZE[8] = 8
    SIZE[9] = 8
    SIZE[10] = 8
    SIZE[11] = 8
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-v", '-N', "SI", "ST", '-S', "MAGA",
                        "LUSI", "-d", RAW_DATA_PATH])
  def test_networks_stations_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [0]*len(WAVEFORMS_DATA)
    SIZE[0] = 12
    SIZE[1] = 12
    SIZE[2] = 12
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-N', "SI", "ST", '-C', "HHN", "HHZ",
                        "-v", "-d", RAW_DATA_PATH])
  def test_networks_channels_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [0]*len(WAVEFORMS_DATA)
    SIZE[0] = 8
    SIZE[1] = 8
    SIZE[2] = 8
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-S', "MAGA", "LUSI", '-C', "HHN",
                        "HHZ", "-v", "-d", RAW_DATA_PATH])
  def test_stations_channels_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [0]*len(WAVEFORMS_DATA)
    SIZE[0] = 8
    SIZE[1] = 8
    SIZE[2] = 8
    SIZE[3] = 8
    SIZE[4] = 8
    SIZE[5] = 8
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-S', "MAGA", "LUSI", '-C', "HHN",
                        "HHZ", '-D', "230605", "230606", "-v", "-d",
                        RAW_DATA_PATH])
  def test_stations_channels_dates_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [0]*len(WAVEFORMS_DATA)
    SIZE[0] = 8
    SIZE[1] = 8
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

class TestReadTraces(unittest.TestCase):
  def tearDown(self) -> None:
    shutil.rmtree(os.path.join(DATA_PATH, PRC_STR))
    pass

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-v", "-d", RAW_DATA_PATH])
  def test_non_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    for _, trace_files in WAVEFORMS_DATA:
      print(trace_files)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-G", BEG_DATE_STR, "-v", "-d",
                        RAW_DATA_PATH])
  def test_group_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    for _, trace_files in WAVEFORMS_DATA:
      print(trace_files)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-v", "-G", BEG_DATE_STR, NETWORK_STR,
                        STATION_STR, "-d", RAW_DATA_PATH])
  def test_groups_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    for _, trace_files in WAVEFORMS_DATA:
      print(trace_files)

class TestModel(unittest.TestCase):
  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-v", "-G", BEG_DATE_STR, NETWORK_STR,
                        STATION_STR, "-M", PHASENET_STR, EQTRANSFORMER_STR,
                        "-d", RAW_DATA_PATH])
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
  DATA_PATH = os.path.join(DATA_PATH, "test")
  RAW_DATA_PATH = os.path.join(DATA_PATH, "waveforms")
  unittest.main()
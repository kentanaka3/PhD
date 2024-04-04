#!/bin/python
import unittest
import shutil
from unittest.mock import patch
from src.AdriaArray import *

TEST_PATH = os.path.join(DATA_PATH, "test")
RAW_TEST_PATH = os.path.join(TEST_PATH, "waveforms")
PRC_TEST_PATH = os.path.join(TEST_PATH, "processed")
ANT_TEST_PATH = os.path.join(TEST_PATH, "annotated")
CLF_TEST_PATH = os.path.join(TEST_PATH, "classified")

class TestArgparse(unittest.TestCase):
  @patch("sys.argv", ["AdriaArray.py"])
  def test_non_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, ["20230601", "20230731"])
    self.assertEqual(args.groups, [BEG_DATE_STR])
    self.assertEqual(args.models, [PHASENET_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.verbose, False)
    self.assertEqual(args.weights, [INSTANCE_STR])

  @patch("sys.argv", ["AdriaArray.py", "-M", PHASENET_STR])
  def test_model_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, ["20230601", "20230731"])
    self.assertEqual(args.groups, [BEG_DATE_STR])
    self.assertEqual(args.models, [PHASENET_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.verbose, False)
    self.assertEqual(args.weights, [INSTANCE_STR])

  @patch("sys.argv", ["AdriaArray.py", "-M", PHASENET_STR, EQTRANSFORMER_STR])
  def test_models_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, ["20230601", "20230731"])
    self.assertEqual(args.groups, [BEG_DATE_STR])
    self.assertEqual(args.models, [PHASENET_STR, EQTRANSFORMER_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.verbose, False)
    self.assertEqual(args.weights, [INSTANCE_STR])

  @patch("sys.argv", ["AdriaArray.py", "-W", INSTANCE_STR])
  def test_weight_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, ["20230601", "20230731"])
    self.assertEqual(args.groups, [BEG_DATE_STR])
    self.assertEqual(args.models, [PHASENET_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.verbose, False)
    self.assertEqual(args.weights, [INSTANCE_STR])

  @patch("sys.argv", ["AdriaArray.py", "-W", INSTANCE_STR, ORIGINAL_STR])
  def test_weights_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, ["20230601", "20230731"])
    self.assertEqual(args.groups, [BEG_DATE_STR])
    self.assertEqual(args.models, [PHASENET_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.verbose, False)
    self.assertEqual(args.weights, [INSTANCE_STR, ORIGINAL_STR])

  @patch("sys.argv", ["AdriaArray.py", "-G", BEG_DATE_STR])
  def test_group_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, ["20230601", "20230731"])
    self.assertEqual(args.groups, [BEG_DATE_STR])
    self.assertEqual(args.models, [PHASENET_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.verbose, False)
    self.assertEqual(args.weights, [INSTANCE_STR])

  @patch("sys.argv", ["AdriaArray.py", "-G", BEG_DATE_STR, NETWORK_STR,
                      STATION_STR])
  def test_groups_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, ["20230601", "20230731"])
    self.assertEqual(args.groups, [BEG_DATE_STR, NETWORK_STR, STATION_STR])
    self.assertEqual(args.models, [PHASENET_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.verbose, False)
    self.assertEqual(args.weights, [INSTANCE_STR])

  @patch("sys.argv", ["AdriaArray.py", "-D", "20230601", "20230731"])
  def test_range_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, ["20230601", "20230731"])
    self.assertEqual(args.groups, [BEG_DATE_STR])
    self.assertEqual(args.models, [PHASENET_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.verbose, False)
    self.assertEqual(args.weights, [INSTANCE_STR])

  @patch("sys.argv", ["AdriaArray.py", "-v"])
  def test_verbose_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.dates, ["20230601", "20230731"])
    self.assertEqual(args.groups, [BEG_DATE_STR])
    self.assertEqual(args.models, [PHASENET_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.verbose, True)
    self.assertEqual(args.weights, [INSTANCE_STR])

class TestWaveformTable(unittest.TestCase):
  @patch("sys.argv", ["AdriaArray.py"])
  def test_non_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, RAW_TEST_PATH)
    self.assertEqual(WAVEFORMS_DATA.size, len(HEADER) * 45)

  @patch("sys.argv", ["AdriaArray.py", '-N', "IV"])
  def test_network_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, RAW_TEST_PATH)
    self.assertEqual(WAVEFORMS_DATA.size, len(HEADER) * 27)

  @patch("sys.argv", ["AdriaArray.py", '-N', "SI", "ST"])
  def test_networks_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, RAW_TEST_PATH)
    self.assertEqual(WAVEFORMS_DATA.size, len(HEADER) * 18)

  @patch("sys.argv", ["AdriaArray.py", '-S', "LUSI"])
  def test_station_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, RAW_TEST_PATH)
    self.assertEqual(WAVEFORMS_DATA.size, len(HEADER) * 9)

  @patch("sys.argv", ["AdriaArray.py", '-S', "LUSI", "PANI"])
  def test_stations_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, RAW_TEST_PATH)
    self.assertEqual(WAVEFORMS_DATA.size, len(HEADER) * 18)

  @patch("sys.argv", ["AdriaArray.py", '-C', "EHZ"])
  def test_channel_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, RAW_TEST_PATH)
    self.assertEqual(WAVEFORMS_DATA.size, len(HEADER) * 3)

  @patch("sys.argv", ["AdriaArray.py", '-C', "HHZ", "HHN"])
  def test_channels_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, RAW_TEST_PATH)
    self.assertEqual(WAVEFORMS_DATA.size, len(HEADER) * 24)

  @patch("sys.argv", ["AdriaArray.py", '-N', "SI", "ST", '-S', "MAGA", "LUSI"])
  def test_networks_stations_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, RAW_TEST_PATH)
    self.assertEqual(WAVEFORMS_DATA.size, len(HEADER) * 9)

  @patch("sys.argv", ["AdriaArray.py", '-N', "SI", "ST", '-C', "HHN", "HHZ"])
  def test_networks_channels_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, RAW_TEST_PATH)
    self.assertEqual(WAVEFORMS_DATA.size, len(HEADER) * 6)

  @patch("sys.argv", ["AdriaArray.py", '-S', "MAGA", "LUSI", '-C', "HHN",
                      "HHZ"])
  def test_stations_channels_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, RAW_TEST_PATH)
    self.assertEqual(WAVEFORMS_DATA.size, len(HEADER) * 12)

  @patch("sys.argv", ["AdriaArray.py", '-S', "MAGA", "LUSI", '-C', "HHN",
                      "HHZ", '-D', "20230605", "20230606"])
  def test_stations_channels_dates_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, RAW_TEST_PATH)
    self.assertEqual(WAVEFORMS_DATA.size, len(HEADER) * 4)

class TestReadTraces(unittest.TestCase):
  def tearDown(self) -> None:
    shutil.rmtree(PRC_TEST_PATH)

  @patch("sys.argv", ["AdriaArray.py"])
  def test_non_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, RAW_TEST_PATH)
    for clean in [True, False]:
      for group in WAVEFORMS_DATA.groupby(args.groups):
        stream, output = read_traces(group[1], PRC_TEST_PATH)
        self.assertEqual(clean, output)
        if clean: clean_stream(stream, PRC_TEST_PATH)

  @patch("sys.argv", ["AdriaArray.py", "-G", BEG_DATE_STR])
  def test_group_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, RAW_TEST_PATH)
    for clean in [True, False]:
      for group in WAVEFORMS_DATA.groupby(args.groups):
        stream, output = read_traces(group[1], PRC_TEST_PATH)
        self.assertEqual(clean, output)
        if clean: clean_stream(stream, PRC_TEST_PATH)

  @patch("sys.argv", ["AdriaArray.py", "-G", BEG_DATE_STR, NETWORK_STR,
                      STATION_STR])
  def test_groups_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, RAW_TEST_PATH)
    for clean in [True, False]:
      for group in WAVEFORMS_DATA.groupby(args.groups):
        stream, output = read_traces(group[1], PRC_TEST_PATH)
        self.assertEqual(clean, output)
        if clean: clean_stream(stream, PRC_TEST_PATH)

class TestAnnotation(unittest.TestCase):
  def tearDown(self) -> None:
    shutil.rmtree(PRC_TEST_PATH)

  @patch("sys.argv", ["AdriaArray.py", "-G", BEG_DATE_STR, NETWORK_STR,
                      STATION_STR, "-M", PHASENET_STR, EQTRANSFORMER_STR])
  def test_annotation(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, RAW_TEST_PATH)
    for x, y in list(itertools.product(args.models, args.weights)):
      model = MODEL_WEIGHTS_DICT[x][CLASS_STR].from_pretrained(y)
      for group in WAVEFORMS_DATA.groupby(args.groups):
        stream, _ = read_traces(group[1], PRC_TEST_PATH)
        clean_stream(stream, PRC_TEST_PATH)
        output = model.classify(stream, batch_size=256, P_threshold=0.2,
                                S_threshold=0.1, parallelism=8).picks
        expected = PickList()
        CLF_FILE = os.path.join(CLF_TEST_PATH, "_".join([*group[0], x, y]) + \
                                               PICKLE_EXT)
        with open(CLF_FILE, 'rb') as fr:
          while True:
            try:
              expected += pickle.load(fr)
            except EOFError:
              break
        for a, b in zip(output, expected):
          self.assertEqual(a.trace_id, b.trace_id)
          self.assertEqual(a.peak_time, b.peak_time)
          self.assertEqual(a.peak_value, b.peak_value)
          self.assertEqual(a.phase, b.phase)

if __name__ == "__main__":
  unittest.main()
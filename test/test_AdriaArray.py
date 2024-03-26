#!/bin/python
import unittest
from unittest.mock import patch
from src.AdriaArray import *

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
    WAVEFORMS_DATA = waveform_table(args, os.path.join(DATA_PATH, "test"))
    self.assertEqual(WAVEFORMS_DATA.size, len(HEADER) * 75)

  @patch("sys.argv", ["AdriaArray.py", '-N', "IV"])
  def test_network_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, os.path.join(DATA_PATH, "test"))
    self.assertEqual(WAVEFORMS_DATA.size, len(HEADER) * 45)

  @patch("sys.argv", ["AdriaArray.py", '-N', "SI", "ST"])
  def test_networks_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, os.path.join(DATA_PATH, "test"))
    self.assertEqual(WAVEFORMS_DATA.size, len(HEADER) * 30)

  @patch("sys.argv", ["AdriaArray.py", '-S', "LUSI"])
  def test_station_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, os.path.join(DATA_PATH, "test"))
    self.assertEqual(WAVEFORMS_DATA.size, len(HEADER) * 15)

  @patch("sys.argv", ["AdriaArray.py", '-S', "LUSI", "PANI"])
  def test_stations_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, os.path.join(DATA_PATH, "test"))
    self.assertEqual(WAVEFORMS_DATA.size, len(HEADER) * 30)

  @patch("sys.argv", ["AdriaArray.py", '-C', "EHZ"])
  def test_channel_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, os.path.join(DATA_PATH, "test"))
    self.assertEqual(WAVEFORMS_DATA.size, len(HEADER) * 5)

  @patch("sys.argv", ["AdriaArray.py", '-C', "HHZ", "HHN"])
  def test_channels_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, os.path.join(DATA_PATH, "test"))
    self.assertEqual(WAVEFORMS_DATA.size, len(HEADER) * 40)

  @patch("sys.argv", ["AdriaArray.py", '-N', "SI", "ST", '-S', "MAGA", "LUSI"])
  def test_networks_stations_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, os.path.join(DATA_PATH, "test"))
    self.assertEqual(WAVEFORMS_DATA.size, len(HEADER) * 15)

  @patch("sys.argv", ["AdriaArray.py", '-N', "SI", "ST", '-C', "HHN", "HHZ"])
  def test_networks_channels_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, os.path.join(DATA_PATH, "test"))
    self.assertEqual(WAVEFORMS_DATA.size, len(HEADER) * 10)

  @patch("sys.argv", ["AdriaArray.py", '-S', "MAGA", "LUSI", '-C', "HHN",
                      "HHZ"])
  def test_stations_channels_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, os.path.join(DATA_PATH, "test"))
    self.assertEqual(WAVEFORMS_DATA.size, len(HEADER) * 20)

  @patch("sys.argv", ["AdriaArray.py", '-S', "MAGA", "LUSI", '-C', "HHN",
                      "HHZ", '-D', "20230605", "20230606"])
  def test_stations_channels_dates_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args, os.path.join(DATA_PATH, "test"))
    print(WAVEFORMS_DATA)
    self.assertEqual(WAVEFORMS_DATA.size, len(HEADER) * 4)

if __name__ == "__main__":
  unittest.main()
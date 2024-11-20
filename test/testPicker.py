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
from picker import *
import initializer as ini

DATA_PATH = Path(PRJ_PATH, "data", "test")
TEST_PATH = Path(DATA_PATH, "waveforms")

class TestWaveformTable(unittest.TestCase):
  """
  - test_non_args
  - test_network_args
  - test_networks_args
  - test_not_network_args
  - test_not_networks_args
  - test_station_args
  - test_stations_args
  - test_not_station_args
  - test_not_stations_args
  - test_channel_args
  - test_channels_args
  - test_not_channel_args
  - test_not_channels_args
  - test_networks_stations_args
  - test_networks_channels_args
  - test_stations_channels_args
  - test_download_network_station_channels_args
  """
  @unittest.mock.patch("sys.argv", ["picker.py", "-v", "-d", str(TEST_PATH)])
  def test_non_args(self):
    args = ini.parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    EXPECTED = {
      MODEL_STR: [PHASENET_STR, EQTRANSFORMER_STR],
      WEIGHT_STR: [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR],
      NETWORK_STR: None,
      STATION_STR: None,
      CHANNEL_STR: None,
      BEG_DATE_STR: [UTCDateTime(year=2023, month=6, day=1).__str__(),
                     UTCDateTime(year=2023, month=8, day=1).__str__()],
      GROUPS_STR: [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      DIRECTORY_STR: str(TEST_PATH),
      PWAVE: PWAVE_THRESHOLD,
      SWAVE: SWAVE_THRESHOLD,
      JULIAN_STR: False,
      DENOISER_STR: False,
      DOMAIN_STR: [44.5, 47, 10, 14],
      CLIENT_STR: [OGS_CLIENT_STR],
    }
    self.assertEqual(primary_arguments(args), EXPECTED)
    self.assertEqual(read_arguments(args), EXPECTED)
    # TODO: Implement waveform table test

  @unittest.mock.patch("sys.argv", ["picker.py", '-N', "OX", "-d",
                                    str(TEST_PATH), "-v"])
  def test_network_args(self):
    args = ini.parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    EXPECTED = {
      MODEL_STR: [PHASENET_STR, EQTRANSFORMER_STR],
      WEIGHT_STR: [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR],
      NETWORK_STR: ["OX"],
      STATION_STR: None,
      CHANNEL_STR: None,
      BEG_DATE_STR: [UTCDateTime(year=2023, month=6, day=1).__str__(),
                     UTCDateTime(year=2023, month=8, day=1).__str__()],
      GROUPS_STR: [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      DIRECTORY_STR: str(TEST_PATH),
      PWAVE: PWAVE_THRESHOLD,
      SWAVE: SWAVE_THRESHOLD,
      JULIAN_STR: False,
      DENOISER_STR: False,
      DOMAIN_STR: [44.5, 47, 10, 14],
      CLIENT_STR: [OGS_CLIENT_STR],
    }
    self.assertEqual(primary_arguments(args), EXPECTED)
    self.assertEqual(read_arguments(args), EXPECTED)
    # TODO: Implement waveform table test

  @unittest.mock.patch("sys.argv", ["picker.py", '-N', "OX", "ST", "-v", "-d",
                                    str(TEST_PATH)])
  def test_networks_args(self):
    args = ini.parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    EXPECTED = {
      MODEL_STR: [PHASENET_STR, EQTRANSFORMER_STR],
      WEIGHT_STR: [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR],
      NETWORK_STR: ["OX", "ST"],
      STATION_STR: None,
      CHANNEL_STR: None,
      BEG_DATE_STR: [UTCDateTime(year=2023, month=6, day=1).__str__(),
                     UTCDateTime(year=2023, month=8, day=1).__str__()],
      GROUPS_STR: [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      DIRECTORY_STR: str(TEST_PATH),
      PWAVE: PWAVE_THRESHOLD,
      SWAVE: SWAVE_THRESHOLD,
      JULIAN_STR: False,
      DENOISER_STR: False,
      DOMAIN_STR: [44.5, 47, 10, 14],
      CLIENT_STR: [OGS_CLIENT_STR],
    }
    self.assertEqual(primary_arguments(args), EXPECTED)
    self.assertEqual(read_arguments(args), EXPECTED)
    # TODO: Implement waveform table test

  @unittest.mock.patch("sys.argv", ["picker.py", '-N', "OE", "-v", "-d",
                                    str(TEST_PATH), "-D", "230601", "230605"])
  def test_not_network_args(self):
    args = ini.parse_arguments()
    self.assertRaises(FileNotFoundError, waveform_table, args)
    EXPECTED = {
      MODEL_STR: [PHASENET_STR, EQTRANSFORMER_STR],
      WEIGHT_STR: [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR],
      NETWORK_STR: ["OE"],
      STATION_STR: None,
      CHANNEL_STR: None,
      BEG_DATE_STR: [UTCDateTime(year=2023, month=6, day=1).__str__(),
                      UTCDateTime(year=2023, month=6, day=5).__str__()],
      GROUPS_STR: [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      DIRECTORY_STR: str(TEST_PATH),
      PWAVE: PWAVE_THRESHOLD,
      SWAVE: SWAVE_THRESHOLD,
      JULIAN_STR: False,
      DENOISER_STR: False,
      DOMAIN_STR: [44.5, 47, 10, 14],
      CLIENT_STR: [OGS_CLIENT_STR],
    }
    self.assertEqual(primary_arguments(args), EXPECTED)
    self.assertEqual(read_arguments(args), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", '-N', "OE", "ZO", "-v", "-d",
                                    str(TEST_PATH), "-D", "230601", "230605"])
  def test_not_networks_args(self):
    args = ini.parse_arguments()
    self.assertRaises(FileNotFoundError, waveform_table, args)
    EXPECTED = {
      MODEL_STR: [PHASENET_STR, EQTRANSFORMER_STR],
      WEIGHT_STR: [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR],
      NETWORK_STR: ["OE", "ZO"],
      STATION_STR: None,
      CHANNEL_STR: None,
      BEG_DATE_STR: [UTCDateTime(year=2023, month=6, day=1).__str__(),
                     UTCDateTime(year=2023, month=6, day=5).__str__()],
      GROUPS_STR: [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      DIRECTORY_STR: str(TEST_PATH),
      PWAVE: PWAVE_THRESHOLD,
      SWAVE: SWAVE_THRESHOLD,
      JULIAN_STR: False,
      DENOISER_STR: False,
      DOMAIN_STR: [44.5, 47, 10, 14],
      CLIENT_STR: [OGS_CLIENT_STR],
    }
    self.assertEqual(primary_arguments(args), EXPECTED)
    self.assertEqual(read_arguments(args), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", '-S', "LUSI", "-v", "-d",
                                    str(TEST_PATH)])
  def test_station_args(self):
    args = ini.parse_arguments()
    self.assertRaises(FileNotFoundError, waveform_table, args)
    EXPECTED = {
      MODEL_STR: [PHASENET_STR, EQTRANSFORMER_STR],
      WEIGHT_STR: [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR],
      NETWORK_STR: None,
      STATION_STR: ["LUSI"],
      CHANNEL_STR: None,
      BEG_DATE_STR: [UTCDateTime(year=2023, month=6, day=1).__str__(),
                     UTCDateTime(year=2023, month=8, day=1).__str__()],
      GROUPS_STR: [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      DIRECTORY_STR: str(TEST_PATH),
      PWAVE: PWAVE_THRESHOLD,
      SWAVE: SWAVE_THRESHOLD,
      JULIAN_STR: False,
      DENOISER_STR: False,
      DOMAIN_STR: [44.5, 47, 10, 14],
      CLIENT_STR: [OGS_CLIENT_STR],
    }
    self.assertEqual(primary_arguments(args), EXPECTED)
    self.assertEqual(read_arguments(args), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", '-S', "LUSI", "PANI", "-v",
                                    "-d", str(TEST_PATH)])
  def test_stations_args(self):
    args = ini.parse_arguments()
    self.assertRaises(FileNotFoundError, waveform_table, args)
    EXPECTED = {
      MODEL_STR: [PHASENET_STR, EQTRANSFORMER_STR],
      WEIGHT_STR: [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR],
      NETWORK_STR: None,
      STATION_STR: ["LUSI", "PANI"],
      CHANNEL_STR: None,
      BEG_DATE_STR: [UTCDateTime(year=2023, month=6, day=1).__str__(),
                     UTCDateTime(year=2023, month=8, day=1).__str__()],
      GROUPS_STR: [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      DIRECTORY_STR: str(TEST_PATH),
      PWAVE: PWAVE_THRESHOLD,
      SWAVE: SWAVE_THRESHOLD,
      JULIAN_STR: False,
      DENOISER_STR: False,
      DOMAIN_STR: [44.5, 47, 10, 14],
      CLIENT_STR: [OGS_CLIENT_STR],
    }
    self.assertEqual(primary_arguments(args), EXPECTED)
    self.assertEqual(read_arguments(args), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", '-C', "EHZ", "-v", "-d",
                                    str(TEST_PATH)])
  def test_channel_args(self):
    args = ini.parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    EXPECTED = {
      MODEL_STR: [PHASENET_STR, EQTRANSFORMER_STR],
      WEIGHT_STR: [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR],
      NETWORK_STR: None,
      STATION_STR: None,
      CHANNEL_STR: ["EHZ"],
      BEG_DATE_STR: [UTCDateTime(year=2023, month=6, day=1).__str__(),
                     UTCDateTime(year=2023, month=8, day=1).__str__()],
      GROUPS_STR: [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      DIRECTORY_STR: str(TEST_PATH),
      PWAVE: PWAVE_THRESHOLD,
      SWAVE: SWAVE_THRESHOLD,
      JULIAN_STR: False,
      DENOISER_STR: False,
      DOMAIN_STR: [44.5, 47, 10, 14],
      CLIENT_STR: [OGS_CLIENT_STR],
    }
    self.assertEqual(primary_arguments(args), EXPECTED)
    self.assertEqual(read_arguments(args), EXPECTED)
    # TODO: Implement waveform table test

  @unittest.mock.patch("sys.argv", ["picker.py", '-C', "HHZ", "HHN", "-v",
                                    "-d", str(TEST_PATH)])
  def test_channels_args(self):
    args = ini.parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    EXPECTED = {
      MODEL_STR: [PHASENET_STR, EQTRANSFORMER_STR],
      WEIGHT_STR: [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR],
      NETWORK_STR: None,
      STATION_STR: None,
      CHANNEL_STR: ["HHZ", "HHN"],
      BEG_DATE_STR: [UTCDateTime(year=2023, month=6, day=1).__str__(),
                     UTCDateTime(year=2023, month=8, day=1).__str__()],
      GROUPS_STR: [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      DIRECTORY_STR: str(TEST_PATH),
      PWAVE: PWAVE_THRESHOLD,
      SWAVE: SWAVE_THRESHOLD,
      JULIAN_STR: False,
      DENOISER_STR: False,
      DOMAIN_STR: [44.5, 47, 10, 14],
      CLIENT_STR: [OGS_CLIENT_STR],
    }
    self.assertEqual(primary_arguments(args), EXPECTED)
    self.assertEqual(read_arguments(args), EXPECTED)
    # TODO: Implement waveform table test

  @unittest.mock.patch("sys.argv", ["picker.py", '-N', "SI", "ST", "-v", '-S',
                                    "MAGA", "LUSI", "-d", str(TEST_PATH)])
  def test_networks_stations_args(self):
    args = ini.parse_arguments()
    self.assertRaises(FileNotFoundError, waveform_table, args)
    EXPECTED = {
      MODEL_STR: [PHASENET_STR, EQTRANSFORMER_STR],
      WEIGHT_STR: [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR],
      NETWORK_STR: ["SI", "ST"],
      STATION_STR: ["MAGA", "LUSI"],
      CHANNEL_STR: None,
      BEG_DATE_STR: [UTCDateTime(year=2023, month=6, day=1).__str__(),
                     UTCDateTime(year=2023, month=8, day=1).__str__()],
      GROUPS_STR: [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      DIRECTORY_STR: str(TEST_PATH),
      PWAVE: PWAVE_THRESHOLD,
      SWAVE: SWAVE_THRESHOLD,
      JULIAN_STR: False,
      DENOISER_STR: False,
      DOMAIN_STR: [44.5, 47, 10, 14],
      CLIENT_STR: [OGS_CLIENT_STR],
    }
    self.assertEqual(primary_arguments(args), EXPECTED)
    self.assertEqual(read_arguments(args), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", '-N', "SI", "ST", "-v", '-C',
                                    "HHN", "HHZ", "-d", str(TEST_PATH)])
  def test_networks_channels_args(self):
    args = ini.parse_arguments()
    self.assertRaises(FileNotFoundError, waveform_table, args)
    EXPECTED = {
      MODEL_STR: [PHASENET_STR, EQTRANSFORMER_STR],
      WEIGHT_STR: [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR],
      NETWORK_STR: ["SI", "ST"],
      STATION_STR: None,
      CHANNEL_STR: ["HHN", "HHZ"],
      BEG_DATE_STR: [UTCDateTime(year=2023, month=6, day=1).__str__(),
                     UTCDateTime(year=2023, month=8, day=1).__str__()],
      GROUPS_STR: [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      DIRECTORY_STR: str(TEST_PATH),
      PWAVE: PWAVE_THRESHOLD,
      SWAVE: SWAVE_THRESHOLD,
      JULIAN_STR: False,
      DENOISER_STR: False,
      DOMAIN_STR: [44.5, 47, 10, 14],
      CLIENT_STR: [OGS_CLIENT_STR],
    }
    self.assertEqual(primary_arguments(args), EXPECTED)
    self.assertEqual(read_arguments(args), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", '-S', "MAGA", "LUSI", "-v",
                                    '-C', "HHN", "HHZ", "-d", str(TEST_PATH)])
  def test_stations_channels_args(self):
    args = ini.parse_arguments()
    self.assertRaises(FileNotFoundError, waveform_table, args)
    EXPECTED = {
      MODEL_STR: [PHASENET_STR, EQTRANSFORMER_STR],
      WEIGHT_STR: [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR],
      NETWORK_STR: None,
      STATION_STR: ["MAGA", "LUSI"],
      CHANNEL_STR: ["HHN", "HHZ"],
      BEG_DATE_STR: [UTCDateTime(year=2023, month=6, day=1).__str__(),
                     UTCDateTime(year=2023, month=8, day=1).__str__()],
      GROUPS_STR: [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      DIRECTORY_STR: str(TEST_PATH),
      PWAVE: PWAVE_THRESHOLD,
      SWAVE: SWAVE_THRESHOLD,
      JULIAN_STR: False,
      DENOISER_STR: False,
      DOMAIN_STR: [44.5, 47, 10, 14],
      CLIENT_STR: [OGS_CLIENT_STR],
    }
    self.assertEqual(primary_arguments(args), EXPECTED)
    self.assertEqual(read_arguments(args), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", '-S', "MAGA", "LUSI", "-v",
                                    '-C', "HHN", "HHZ", '-D', "230605",
                                    "230606", "-d", str(TEST_PATH)])
  def test_stations_channels_dates_args(self):
    args = ini.parse_arguments()
    self.assertRaises(FileNotFoundError, waveform_table, args)
    EXPECTED = {
      MODEL_STR: [PHASENET_STR, EQTRANSFORMER_STR],
      WEIGHT_STR: [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR],
      NETWORK_STR: None,
      STATION_STR: ["MAGA", "LUSI"],
      CHANNEL_STR: ["HHN", "HHZ"],
      BEG_DATE_STR: [UTCDateTime(year=2023, month=6, day=5).__str__(),
                     UTCDateTime(year=2023, month=6, day=6).__str__()],
      GROUPS_STR: [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      DIRECTORY_STR: str(TEST_PATH),
      PWAVE: PWAVE_THRESHOLD,
      SWAVE: SWAVE_THRESHOLD,
      JULIAN_STR: False,
      DENOISER_STR: False,
      DOMAIN_STR: [44.5, 47, 10, 14],
      CLIENT_STR: [OGS_CLIENT_STR],
    }
    self.assertEqual(primary_arguments(args), EXPECTED)
    self.assertEqual(read_arguments(args), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", '-N', "OE", "-S", "ABTA",
                                    "-C", "*", "-v", "-d", str(TEST_PATH),
                                    "-D", "240601", "240605", "--download"])
  def test_download_network_station_channels_args(self):
    args = ini.parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    EXPECTED = {
      MODEL_STR: [PHASENET_STR, EQTRANSFORMER_STR],
      WEIGHT_STR: [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR],
      NETWORK_STR: ["OE"],
      STATION_STR: ["ABTA"],
      CHANNEL_STR: ["*"],
      BEG_DATE_STR: [UTCDateTime(year=2024, month=6, day=1).__str__(),
                      UTCDateTime(year=2024, month=6, day=5).__str__()],
      GROUPS_STR: [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      DIRECTORY_STR: str(TEST_PATH),
      PWAVE: PWAVE_THRESHOLD,
      SWAVE: SWAVE_THRESHOLD,
      JULIAN_STR: False,
      DENOISER_STR: False,
      DOMAIN_STR: [44.5, 47, 10, 14],
      CLIENT_STR: [OGS_CLIENT_STR],
    }
    self.assertEqual(primary_arguments(args), EXPECTED)
    self.assertEqual(read_arguments(args), EXPECTED)

class TestReadTraces(unittest.TestCase):
  def tearDown(self) -> None:
    Path(DATA_PATH, WAVEFORMS_STR + CSV_EXT).unlink()
    Path(DATA_PATH, ARGUMENTS_STR + JSON_EXT).unlink()

  @unittest.mock.patch("sys.argv", ["picker.py", "-G", BEG_DATE_STR, "-v",
                                    "-d", str(TEST_PATH)])
  def test_group_args(self):
    args = ini.parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    for _, trace_files in WAVEFORMS_DATA.groupby(args.groups):
      stream = read_traces(trace_files, args)
      for tr in stream:
        self.assertGreaterEqual(tr.stats.starttime,
                                UTCDateTime(tr.stats.starttime.date))
        self.assertLessEqual(tr.stats.endtime,
                             UTCDateTime(tr.stats.starttime.date) + ONE_DAY)

  @unittest.mock.patch("sys.argv", ["picker.py", "-v", "-d", str(TEST_PATH)])
  def test_groups_args(self):
    args = ini.parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    for _, trace_files in WAVEFORMS_DATA.groupby(args.groups):
      stream = read_traces(trace_files, args)
      for tr in stream:
        self.assertGreaterEqual(tr.stats.starttime,
                                UTCDateTime(tr.stats.starttime.date))
        self.assertLessEqual(tr.stats.endtime,
                             UTCDateTime(tr.stats.starttime.date) + ONE_DAY)

class TestModel(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    clf_dir = Path(DATA_PATH, CLF_STR)
    if clf_dir.exists():
      for f in clf_dir.iterdir():
        if f.is_file():
          f.unlink()

  """
  def test_model_selector(self):
    self.assertEqual(get_model(DEEPDENOISER_STR, ORIGINAL_STR),
                     sbm.DeepDenoiser.from_pretrained(ORIGINAL_STR))
    self.assertEqual(get_model(DEEPDENOISER_STR, INSTANCE_STR),
                     sbm.DeepDenoiser.from_pretrained(INSTANCE_STR))
    self.assertEqual(get_model(DEEPDENOISER_STR, SCEDC_STR),
                     sbm.DeepDenoiser.from_pretrained(SCEDC_STR))
    self.assertEqual(get_model(DEEPDENOISER_STR, STEAD_STR),
                     sbm.DeepDenoiser.from_pretrained(STEAD_STR))
    self.assertEqual(get_model(EQTRANSFORMER_STR, ORIGINAL_STR),
                     sbm.EQTransformer.from_pretrained(ORIGINAL_STR))
    self.assertEqual(get_model(EQTRANSFORMER_STR, INSTANCE_STR),
                     sbm.EQTransformer.from_pretrained(INSTANCE_STR))
    self.assertEqual(get_model(EQTRANSFORMER_STR, SCEDC_STR),
                     sbm.EQTransformer.from_pretrained(SCEDC_STR))
    self.assertEqual(get_model(EQTRANSFORMER_STR, STEAD_STR),
                     sbm.EQTransformer.from_pretrained(STEAD_STR))
    self.assertEqual(get_model(PHASENET_STR, ORIGINAL_STR),
                     sbm.PhaseNet.from_pretrained(ORIGINAL_STR))
    self.assertEqual(get_model(PHASENET_STR, INSTANCE_STR),
                     sbm.PhaseNet.from_pretrained(INSTANCE_STR))
    self.assertEqual(get_model(PHASENET_STR, SCEDC_STR),
                     sbm.PhaseNet.from_pretrained(SCEDC_STR))
    self.assertEqual(get_model(PHASENET_STR, STEAD_STR),
                     sbm.PhaseNet.from_pretrained(STEAD_STR))
    # TODO: Implement and test new trained AdriaArray weights
    # self.assertEqual(get_model(DEEPDENOISER_STR, ADRIAARRAY_STR),
    #                  sbm.DeepDenoiser.from_pretrained(ADRIAARRAY_STR))
    self.assertEqual(get_model(PHASENET_STR, ADRIAARRAY_STR),
                     sbm.PhaseNet.from_pretrained(ADRIAARRAY_STR))
    self.assertEqual(get_model(EQTRANSFORMER_STR, ADRIAARRAY_STR),
                     sbm.EQTransformer.from_pretrained(ADRIAARRAY_STR))
  """
  @unittest.mock.patch("sys.argv", ["picker.py", "-v", "-d", str(TEST_PATH)])
  def test_classification(self):
    args = ini.parse_arguments()
    MODELS, WAVEFORMS_DATA = set_up(args)
    for categories, trace_files in WAVEFORMS_DATA.groupby(args.groups):
      classify_stream(categories, trace_files, MODELS, args)

  @unittest.mock.patch("sys.argv", ["picker.py", "-v", "-d", str(TEST_PATH),
                                    "--denoiser"])
  def test_denoised_classification(self):
    args = ini.parse_arguments()
    MODELS, WAVEFORMS_DATA = set_up(args)
    for categories, trace_files in WAVEFORMS_DATA.groupby(args.groups):
      classify_stream(categories, trace_files, MODELS, args)

if __name__ == "__main__": unittest.main()
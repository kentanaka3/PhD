#!/bin/python
import os
from pathlib import Path
import unittest.mock
PRJ_PATH = Path(os.path.dirname(__file__)).parent
INC_PATH = os.path.join(PRJ_PATH, "inc")
import sys
# Add to path
if INC_PATH not in sys.path: sys.path.append(INC_PATH)
import unittest
from obspy.core.utcdatetime import UTCDateTime

from constants import *
import initializer as ini

DATA_PATH = Path(PRJ_PATH, "data", "test")
TEST_PATH = Path(DATA_PATH, "waveforms")

class TestArgparse(unittest.TestCase):
  def setUp(self):
    with open(Path(DATA_PATH, "file.key"), 'w') as fp:
      pass

  def tearDown(self):
    os.remove(Path(DATA_PATH, "file.key"))

  @unittest.mock.patch("sys.argv", ["picker.py"])
  def test_default(self):
    args = ini.parse_arguments()
    self.assertEqual(args.batch, 4096)
    self.assertEqual(args.channel, None)
    self.assertEqual(args.client, [OGS_CLIENT_STR])
    self.assertEqual(args.circdomain, [46.3583, 12.808, 0., 0.3])
    self.assertEqual(args.config, None)
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])
    self.assertEqual(args.denoiser, False)
    self.assertEqual(args.directory, Path(PRJ_PATH, "data", WAVEFORMS_STR))
    self.assertEqual(args.rectdomain, None)
    self.assertEqual(args.download, False)
    self.assertEqual(args.file, None)
    self.assertEqual(args.force, False)
    self.assertEqual(args.groups, [BEG_DATE_STR, NETWORK_STR, STATION_STR])
    self.assertEqual(args.julian, None)
    self.assertEqual(args.key, None)
    self.assertEqual(args.models, [PHASENET_STR, EQTRANSFORMER_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.pwave, PWAVE_THRESHOLD)
    self.assertEqual(args.pyrocko, False)
    self.assertEqual(args.rectdomain, None)
    self.assertEqual(args.station, None)
    self.assertEqual(args.swave, SWAVE_THRESHOLD)
    self.assertEqual(args.timing, False)
    self.assertEqual(args.train, False)
    self.assertEqual(args.verbose, False)
    self.assertEqual(args.weights, [INSTANCE_STR, ORIGINAL_STR, STEAD_STR,
                                    SCEDC_STR])

  @unittest.mock.patch("sys.argv", ["picker.py", "-C",  "EHZ"])
  def test_channel_args(self):
    args = ini.parse_arguments()
    self.assertEqual(args.channel, ["EHZ"])

  @unittest.mock.patch("sys.argv",
                       ["picker.py", "--client", INGV_CLIENT_STR])
  def test_client_args(self):
    args = ini.parse_arguments()
    self.assertEqual(args.client, [INGV_CLIENT_STR])

  @unittest.mock.patch("sys.argv", ["picker.py", "-D", "230602", "230603"])
  def test_dates_args(self):
    args = ini.parse_arguments()
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=2),
                                  UTCDateTime(year=2023, month=6, day=3)])

  @unittest.mock.patch("sys.argv", ["picker.py", "-D", "230630", "230629"])
  def test_dates_order_args(self):
    args = ini.parse_arguments()
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=29),
                                  UTCDateTime(year=2023, month=6, day=30)])

  @unittest.mock.patch("sys.argv", ["picker.py", "-D", "230629", "230632"])
  def test_dates_f_value_args(self):
    with self.assertRaises(SystemExit) as cm:
      ini.parse_arguments()
    self.assertEqual(cm.exception.code, 2)

  @unittest.mock.patch("sys.argv", ["picker.py", "-G", BEG_DATE_STR])
  def test_group_args(self):
    args = ini.parse_arguments()
    self.assertEqual(args.groups, [BEG_DATE_STR])

  @unittest.mock.patch("sys.argv",
                       ["picker.py", "-G", BEG_DATE_STR, NETWORK_STR])
  def test_groups_args(self):
    args = ini.parse_arguments()
    self.assertEqual(args.groups, [BEG_DATE_STR, NETWORK_STR])

  @unittest.mock.patch("sys.argv",
                       ["picker.py", "-K",
                        Path(DATA_PATH, "file.key").__str__()])
  def test_key_args(self):
    args = ini.parse_arguments()
    self.assertEqual(args.key, Path(DATA_PATH, "file.key"))

  @unittest.mock.patch("sys.argv", ["picker.py", "-M", PHASENET_STR])
  def test_model_args(self):
    args = ini.parse_arguments()
    self.assertEqual(args.models, [PHASENET_STR])

  @unittest.mock.patch("sys.argv", ["picker.py", "-M", PHASENET_STR,
                                    EQTRANSFORMER_STR])
  def test_models_args(self):
    args = ini.parse_arguments()
    self.assertEqual(args.models, [PHASENET_STR, EQTRANSFORMER_STR])

  @unittest.mock.patch("sys.argv", ["picker.py", "-D", "230601", "230731"])
  def test_range_args(self):
    args = ini.parse_arguments()
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])

  @unittest.mock.patch("sys.argv", ["picker.py", "-T"])
  def test_train_args(self):
    args = ini.parse_arguments()
    self.assertEqual(args.train, True)

  @unittest.mock.patch("sys.argv", ["picker.py", "-v"])
  def test_verbose_args(self):
    args = ini.parse_arguments()
    self.assertEqual(args.verbose, True)

  @unittest.mock.patch("sys.argv", ["picker.py", "--denoiser"])
  def test_denoiser_args(self):
    args = ini.parse_arguments()
    self.assertEqual(args.denoiser, True)

  @unittest.mock.patch("sys.argv", ["picker.py", "--timing"])
  def test_timing_args(self):
    args = ini.parse_arguments()
    self.assertEqual(args.timing, True)

  @unittest.mock.patch("sys.argv", ["picker.py", "-W", INSTANCE_STR])
  def test_weight_args(self):
    args = ini.parse_arguments()
    self.assertEqual(args.weights, [INSTANCE_STR])

  @unittest.mock.patch("sys.argv", ["picker.py", "-W", INSTANCE_STR,
                                    ORIGINAL_STR])
  def test_weights_args(self):
    args = ini.parse_arguments()
    self.assertEqual(args.weights, [INSTANCE_STR, ORIGINAL_STR])

class TestArguments(unittest.TestCase):
  @unittest.mock.patch("sys.argv", ["picker.py", "-v"])
  def test_default(self):
    args = ini.parse_arguments()
    EXPECTED = {
      CHANNEL_STR   : None,
      DOMAIN_STR    : [46.3583, 12.808, 0., 0.3],
      BEG_DATE_STR  : [UTCDateTime(year=2023, month=6, day=1).__str__(),
                       UTCDateTime(year=2023, month=7, day=31).__str__()],
      DENOISER_STR  : False,
      DIRECTORY_STR : Path("data", WAVEFORMS_STR).__str__(),
      GROUPS_STR    : [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      MODEL_STR     : [PHASENET_STR, EQTRANSFORMER_STR],
      NETWORK_STR   : None,
      PWAVE         : PWAVE_THRESHOLD,
      STATION_STR   : None,
      SWAVE         : SWAVE_THRESHOLD,
      WEIGHT_STR    : [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR]
    }
    self.assertEqual(ini.dump_args(args, False), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", "-C", "EHZ", "-v"])
  def test_channel(self):
    args = ini.parse_arguments()
    EXPECTED = {
      CHANNEL_STR   : ["EHZ"],
      DOMAIN_STR    : [46.3583, 12.808, 0., 0.3],
      BEG_DATE_STR  : [UTCDateTime(year=2023, month=6, day=1).__str__(),
                       UTCDateTime(year=2023, month=7, day=31).__str__()],
      DENOISER_STR  : False,
      DIRECTORY_STR : Path("data", WAVEFORMS_STR).__str__(),
      GROUPS_STR    : [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      MODEL_STR     : [PHASENET_STR, EQTRANSFORMER_STR],
      NETWORK_STR   : None,
      PWAVE         : PWAVE_THRESHOLD,
      STATION_STR   : None,
      SWAVE         : SWAVE_THRESHOLD,
      WEIGHT_STR    : [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR]
    }
    self.assertEqual(ini.dump_args(args, False), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", "-C", "EHZ", "EHN", "-v"])
  def test_channels(self):
    args = ini.parse_arguments()
    EXPECTED = {
      CHANNEL_STR   : ["EHZ", "EHN"],
      DOMAIN_STR    : [46.3583, 12.808, 0., 0.3],
      BEG_DATE_STR  : [UTCDateTime(year=2023, month=6, day=1).__str__(),
                       UTCDateTime(year=2023, month=7, day=31).__str__()],
      DENOISER_STR  : False,
      DIRECTORY_STR : Path("data", WAVEFORMS_STR).__str__(),
      GROUPS_STR    : [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      MODEL_STR     : [PHASENET_STR, EQTRANSFORMER_STR],
      NETWORK_STR   : None,
      PWAVE         : PWAVE_THRESHOLD,
      STATION_STR   : None,
      SWAVE         : SWAVE_THRESHOLD,
      WEIGHT_STR    : [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR]
    }
    self.assertEqual(ini.dump_args(args, False), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", "-N", "OX", "-v"])
  def test_network(self):
    args = ini.parse_arguments()
    EXPECTED = {
      CHANNEL_STR   : None,
      DOMAIN_STR    : [46.3583, 12.808, 0., 0.3],
      BEG_DATE_STR  : [UTCDateTime(year=2023, month=6, day=1).__str__(),
                       UTCDateTime(year=2023, month=7, day=31).__str__()],
      DENOISER_STR  : False,
      DIRECTORY_STR : Path("data", WAVEFORMS_STR).__str__(),
      GROUPS_STR    : [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      MODEL_STR     : [PHASENET_STR, EQTRANSFORMER_STR],
      NETWORK_STR   : ["OX"],
      PWAVE         : PWAVE_THRESHOLD,
      STATION_STR   : None,
      SWAVE         : SWAVE_THRESHOLD,
      WEIGHT_STR    : [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR]
    }
    self.assertEqual(ini.dump_args(args, False), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", "-N", "OX", "ST", "-v"])
  def test_networks(self):
    args = ini.parse_arguments()
    EXPECTED = {
      CHANNEL_STR   : None,
      DOMAIN_STR    : [46.3583, 12.808, 0., 0.3],
      BEG_DATE_STR  : [UTCDateTime(year=2023, month=6, day=1).__str__(),
                       UTCDateTime(year=2023, month=7, day=31).__str__()],
      DENOISER_STR  : False,
      DIRECTORY_STR : Path("data", WAVEFORMS_STR).__str__(),
      GROUPS_STR    : [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      MODEL_STR     : [PHASENET_STR, EQTRANSFORMER_STR],
      NETWORK_STR   : ["OX", "ST"],
      PWAVE         : PWAVE_THRESHOLD,
      STATION_STR   : None,
      SWAVE         : SWAVE_THRESHOLD,
      WEIGHT_STR    : [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR]
    }
    self.assertEqual(ini.dump_args(args, False), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", "-S", "BAD", "-v"])
  def test_station(self):
    args = ini.parse_arguments()
    EXPECTED = {
      CHANNEL_STR   : None,
      DOMAIN_STR    : [46.3583, 12.808, 0., 0.3],
      BEG_DATE_STR  : [UTCDateTime(year=2023, month=6, day=1).__str__(),
                       UTCDateTime(year=2023, month=7, day=31).__str__()],
      DENOISER_STR  : False,
      DIRECTORY_STR : Path("data", WAVEFORMS_STR).__str__(),
      GROUPS_STR    : [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      MODEL_STR     : [PHASENET_STR, EQTRANSFORMER_STR],
      NETWORK_STR   : None,
      PWAVE         : PWAVE_THRESHOLD,
      STATION_STR   : ["BAD"],
      SWAVE         : SWAVE_THRESHOLD,
      WEIGHT_STR    : [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR]
    }
    self.assertEqual(ini.dump_args(args, False), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", "-S", "BAD", "VARA", "-v"])
  def test_stations(self):
    args = ini.parse_arguments()
    EXPECTED = {
      CHANNEL_STR   : None,
      DOMAIN_STR    : [46.3583, 12.808, 0., 0.3],
      BEG_DATE_STR  : [UTCDateTime(year=2023, month=6, day=1).__str__(),
                       UTCDateTime(year=2023, month=7, day=31).__str__()],
      DENOISER_STR  : False,
      DIRECTORY_STR : Path("data", WAVEFORMS_STR).__str__(),
      GROUPS_STR    : [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      MODEL_STR     : [PHASENET_STR, EQTRANSFORMER_STR],
      NETWORK_STR   : None,
      PWAVE         : PWAVE_THRESHOLD,
      STATION_STR   : ["BAD", "VARA"],
      SWAVE         : SWAVE_THRESHOLD,
      WEIGHT_STR    : [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR]
    }
    self.assertEqual(ini.dump_args(args, False), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", "-D", "230601", "230604",
                                    "-v"])
  def test_dates(self):
    args = ini.parse_arguments()
    EXPECTED = {
      CHANNEL_STR   : None,
      DOMAIN_STR    : [46.3583, 12.808, 0., 0.3],
      BEG_DATE_STR  : [UTCDateTime(year=2023, month=6, day=1).__str__(),
                       UTCDateTime(year=2023, month=6, day=4).__str__()],
      DENOISER_STR  : False,
      DIRECTORY_STR : Path("data", WAVEFORMS_STR).__str__(),
      GROUPS_STR    : [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      MODEL_STR     : [PHASENET_STR, EQTRANSFORMER_STR],
      NETWORK_STR   : None,
      PWAVE         : PWAVE_THRESHOLD,
      STATION_STR   : None,
      SWAVE         : SWAVE_THRESHOLD,
      WEIGHT_STR    : [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR]
    }
    self.assertEqual(ini.dump_args(args, False), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", "-d", DATA_PATH.__str__(),
                                    "-v"])
  def test_directory(self):
    args = ini.parse_arguments()
    EXPECTED = {
      CHANNEL_STR   : None,
      DOMAIN_STR    : [46.3583, 12.808, 0., 0.3],
      BEG_DATE_STR  : [UTCDateTime(year=2023, month=6, day=1).__str__(),
                       UTCDateTime(year=2023, month=7, day=31).__str__()],
      DENOISER_STR  : False,
      DIRECTORY_STR : Path("data", "test").__str__(),
      GROUPS_STR    : [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      MODEL_STR     : [PHASENET_STR, EQTRANSFORMER_STR],
      NETWORK_STR   : None,
      PWAVE         : PWAVE_THRESHOLD,
      STATION_STR   : None,
      SWAVE         : SWAVE_THRESHOLD,
      WEIGHT_STR    : [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR]
    }
    self.assertEqual(ini.dump_args(args, False), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", "--denoiser", "-v"])
  def test_denoiser(self):
    args = ini.parse_arguments()
    EXPECTED = {
      CHANNEL_STR   : None,
      DOMAIN_STR    : [46.3583, 12.808, 0., 0.3],
      BEG_DATE_STR  : [UTCDateTime(year=2023, month=6, day=1).__str__(),
                       UTCDateTime(year=2023, month=7, day=31).__str__()],
      DENOISER_STR  : True,
      DIRECTORY_STR : Path("data", WAVEFORMS_STR).__str__(),
      GROUPS_STR    : [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      MODEL_STR     : [PHASENET_STR, EQTRANSFORMER_STR],
      NETWORK_STR   : None,
      PWAVE         : PWAVE_THRESHOLD,
      STATION_STR   : None,
      SWAVE         : SWAVE_THRESHOLD,
      WEIGHT_STR    : [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR]
    }
    self.assertEqual(ini.dump_args(args, False), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", "--rectdomain", "46.3583",
                                    "12.808", "0.1", "0.2", "-v"])
  def test_rectdomain(self):
    args = ini.parse_arguments()
    EXPECTED = {
      CHANNEL_STR   : None,
      DOMAIN_STR    : [46.3583, 12.808, 0.1, 0.2],
      BEG_DATE_STR  : [UTCDateTime(year=2023, month=6, day=1).__str__(),
                       UTCDateTime(year=2023, month=7, day=31).__str__()],
      DENOISER_STR  : False,
      DIRECTORY_STR : Path("data", WAVEFORMS_STR).__str__(),
      GROUPS_STR    : [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      MODEL_STR     : [PHASENET_STR, EQTRANSFORMER_STR],
      NETWORK_STR   : None,
      PWAVE         : PWAVE_THRESHOLD,
      STATION_STR   : None,
      SWAVE         : SWAVE_THRESHOLD,
      WEIGHT_STR    : [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR]
    }
    self.assertEqual(ini.dump_args(args, False), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", "--circdomain", "46.3583",
                                    "12.808", "0.1", "0.2", "-v"])
  def test_circdomain(self):
    args = ini.parse_arguments()
    EXPECTED = {
      CHANNEL_STR   : None,
      DOMAIN_STR    : [46.3583, 12.808, 0.1, 0.2],
      BEG_DATE_STR  : [UTCDateTime(year=2023, month=6, day=1).__str__(),
                       UTCDateTime(year=2023, month=7, day=31).__str__()],
      DENOISER_STR  : False,
      DIRECTORY_STR : Path("data", WAVEFORMS_STR).__str__(),
      GROUPS_STR    : [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      MODEL_STR     : [PHASENET_STR, EQTRANSFORMER_STR],
      NETWORK_STR   : None,
      PWAVE         : PWAVE_THRESHOLD,
      STATION_STR   : None,
      SWAVE         : SWAVE_THRESHOLD,
      WEIGHT_STR    : [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR]
    }
    self.assertEqual(ini.dump_args(args, False), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", "-G", BEG_DATE_STR, "-v"])
  def test_group(self):
    args = ini.parse_arguments()
    EXPECTED = {
      CHANNEL_STR   : None,
      DOMAIN_STR    : [46.3583, 12.808, 0., 0.3],
      BEG_DATE_STR  : [UTCDateTime(year=2023, month=6, day=1).__str__(),
                       UTCDateTime(year=2023, month=7, day=31).__str__()],
      DENOISER_STR  : False,
      DIRECTORY_STR : Path("data", WAVEFORMS_STR).__str__(),
      GROUPS_STR    : [BEG_DATE_STR],
      MODEL_STR     : [PHASENET_STR, EQTRANSFORMER_STR],
      NETWORK_STR   : None,
      PWAVE         : PWAVE_THRESHOLD,
      STATION_STR   : None,
      SWAVE         : SWAVE_THRESHOLD,
      WEIGHT_STR    : [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR]
    }
    self.assertEqual(ini.dump_args(args, False), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", "-G", BEG_DATE_STR,
                                    NETWORK_STR, "-v"])
  def test_groups(self):
    args = ini.parse_arguments()
    EXPECTED = {
      CHANNEL_STR   : None,
      DOMAIN_STR    : [46.3583, 12.808, 0., 0.3],
      BEG_DATE_STR  : [UTCDateTime(year=2023, month=6, day=1).__str__(),
                       UTCDateTime(year=2023, month=7, day=31).__str__()],
      DENOISER_STR  : False,
      DIRECTORY_STR : Path("data", WAVEFORMS_STR).__str__(),
      GROUPS_STR    : [BEG_DATE_STR, NETWORK_STR],
      MODEL_STR     : [PHASENET_STR, EQTRANSFORMER_STR],
      NETWORK_STR   : None,
      PWAVE         : PWAVE_THRESHOLD,
      STATION_STR   : None,
      SWAVE         : SWAVE_THRESHOLD,
      WEIGHT_STR    : [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR]
    }
    self.assertEqual(ini.dump_args(args, False), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", "-M", PHASENET_STR, "-v"])
  def test_model(self):
    args = ini.parse_arguments()
    EXPECTED = {
      CHANNEL_STR   : None,
      DOMAIN_STR    : [46.3583, 12.808, 0., 0.3],
      BEG_DATE_STR  : [UTCDateTime(year=2023, month=6, day=1).__str__(),
                       UTCDateTime(year=2023, month=7, day=31).__str__()],
      DENOISER_STR  : False,
      DIRECTORY_STR : Path("data", WAVEFORMS_STR).__str__(),
      GROUPS_STR    : [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      MODEL_STR     : [PHASENET_STR],
      NETWORK_STR   : None,
      PWAVE         : PWAVE_THRESHOLD,
      STATION_STR   : None,
      SWAVE         : SWAVE_THRESHOLD,
      WEIGHT_STR    : [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR]
    }
    self.assertEqual(ini.dump_args(args, False), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", "-M", PHASENET_STR,
                                    EQTRANSFORMER_STR, "-v"])
  def test_models(self):
    args = ini.parse_arguments()
    EXPECTED = {
      CHANNEL_STR   : None,
      DOMAIN_STR    : [46.3583, 12.808, 0., 0.3],
      BEG_DATE_STR  : [UTCDateTime(year=2023, month=6, day=1).__str__(),
                       UTCDateTime(year=2023, month=7, day=31).__str__()],
      DENOISER_STR  : False,
      DIRECTORY_STR : Path("data", WAVEFORMS_STR).__str__(),
      GROUPS_STR    : [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      MODEL_STR     : [PHASENET_STR, EQTRANSFORMER_STR],
      NETWORK_STR   : None,
      PWAVE         : PWAVE_THRESHOLD,
      STATION_STR   : None,
      SWAVE         : SWAVE_THRESHOLD,
      WEIGHT_STR    : [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR]
    }
    self.assertEqual(ini.dump_args(args, False), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", "--pwave", "0.5", "-v"])
  def test_pwave(self):
    args = ini.parse_arguments()
    EXPECTED = {
      CHANNEL_STR   : None,
      DOMAIN_STR    : [46.3583, 12.808, 0., 0.3],
      BEG_DATE_STR  : [UTCDateTime(year=2023, month=6, day=1).__str__(),
                       UTCDateTime(year=2023, month=7, day=31).__str__()],
      DENOISER_STR  : False,
      DIRECTORY_STR : Path("data", WAVEFORMS_STR).__str__(),
      GROUPS_STR    : [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      MODEL_STR     : [PHASENET_STR, EQTRANSFORMER_STR],
      NETWORK_STR   : None,
      PWAVE         : 0.5,
      STATION_STR   : None,
      SWAVE         : SWAVE_THRESHOLD,
      WEIGHT_STR    : [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR]
    }
    self.assertEqual(ini.dump_args(args, False), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", "--swave", "0.5", "-v"])
  def test_swave(self):
    args = ini.parse_arguments()
    EXPECTED = {
      CHANNEL_STR   : None,
      DOMAIN_STR    : [46.3583, 12.808, 0., 0.3],
      BEG_DATE_STR  : [UTCDateTime(year=2023, month=6, day=1).__str__(),
                       UTCDateTime(year=2023, month=7, day=31).__str__()],
      DENOISER_STR  : False,
      DIRECTORY_STR : Path("data", WAVEFORMS_STR).__str__(),
      GROUPS_STR    : [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      MODEL_STR     : [PHASENET_STR, EQTRANSFORMER_STR],
      NETWORK_STR   : None,
      PWAVE         : PWAVE_THRESHOLD,
      STATION_STR   : None,
      SWAVE         : 0.5,
      WEIGHT_STR    : [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR]
    }
    self.assertEqual(ini.dump_args(args, False), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", "--weights", INSTANCE_STR,
                                    "-v"])
  def test_weight(self):
    args = ini.parse_arguments()
    EXPECTED = {
      CHANNEL_STR   : None,
      DOMAIN_STR    : [46.3583, 12.808, 0., 0.3],
      BEG_DATE_STR  : [UTCDateTime(year=2023, month=6, day=1).__str__(),
                       UTCDateTime(year=2023, month=7, day=31).__str__()],
      DENOISER_STR  : False,
      DIRECTORY_STR : Path("data", WAVEFORMS_STR).__str__(),
      GROUPS_STR    : [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      MODEL_STR     : [PHASENET_STR, EQTRANSFORMER_STR],
      NETWORK_STR   : None,
      PWAVE         : PWAVE_THRESHOLD,
      STATION_STR   : None,
      SWAVE         : SWAVE_THRESHOLD,
      WEIGHT_STR    : [INSTANCE_STR]
    }
    self.assertEqual(ini.dump_args(args, False), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", "--weights", INSTANCE_STR,
                                    ORIGINAL_STR, "-v"])
  def test_weights(self):
    args = ini.parse_arguments()
    EXPECTED = {
      CHANNEL_STR   : None,
      DOMAIN_STR    : [46.3583, 12.808, 0., 0.3],
      BEG_DATE_STR  : [UTCDateTime(year=2023, month=6, day=1).__str__(),
                       UTCDateTime(year=2023, month=7, day=31).__str__()],
      DENOISER_STR  : False,
      DIRECTORY_STR : Path("data", WAVEFORMS_STR).__str__(),
      GROUPS_STR    : [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      MODEL_STR     : [PHASENET_STR, EQTRANSFORMER_STR],
      NETWORK_STR   : None,
      PWAVE         : PWAVE_THRESHOLD,
      STATION_STR   : None,
      SWAVE         : SWAVE_THRESHOLD,
      WEIGHT_STR    : [INSTANCE_STR, ORIGINAL_STR]
    }
    self.assertEqual(ini.dump_args(args, False), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", "-v", "--config",
                                    Path(DATA_PATH, "config.json").__str__()])
  def test_config(self):
    args = ini.parse_arguments()
    EXPECTED = {
      CHANNEL_STR   : [ALL_WILDCHAR_STR],
      DOMAIN_STR    : [46.3583, 12.808, 0., 0.3],
      BEG_DATE_STR  : [UTCDateTime(year=2023, month=6, day=1).__str__(),
                       UTCDateTime(year=2023, month=7, day=31).__str__()],
      DENOISER_STR  : False,
      DIRECTORY_STR : Path("data", WAVEFORMS_STR).__str__(),
      GROUPS_STR    : [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      MODEL_STR     : [PHASENET_STR, EQTRANSFORMER_STR],
      NETWORK_STR   : [ALL_WILDCHAR_STR],
      PWAVE         : PWAVE_THRESHOLD,
      STATION_STR   : None,
      SWAVE         : SWAVE_THRESHOLD,
      WEIGHT_STR    : [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR],
    }
    self.assertEqual(ini.dump_args(args, False), EXPECTED)

  @unittest.mock.patch("sys.argv", ["picker.py", "-N", "OX", "ST", "--config",
                                    str(Path(DATA_PATH, "config.json")), "-v"])
  def test_config_update(self):
    args = ini.parse_arguments()
    EXPECTED = {
      CHANNEL_STR   : [ALL_WILDCHAR_STR],
      DOMAIN_STR    : [46.3583, 12.808, 0., 0.3],
      BEG_DATE_STR  : [UTCDateTime(year=2023, month=6, day=1).__str__(),
                       UTCDateTime(year=2023, month=7, day=31).__str__()],
      DENOISER_STR  : False,
      DIRECTORY_STR : Path("data", WAVEFORMS_STR).__str__(),
      GROUPS_STR    : [BEG_DATE_STR, NETWORK_STR, STATION_STR],
      MODEL_STR     : [PHASENET_STR, EQTRANSFORMER_STR],
      NETWORK_STR   : ["OX", "ST"],
      PWAVE         : PWAVE_THRESHOLD,
      STATION_STR   : None,
      SWAVE         : SWAVE_THRESHOLD,
      WEIGHT_STR    : [INSTANCE_STR, ORIGINAL_STR, STEAD_STR, SCEDC_STR],
    }
    self.assertEqual(ini.dump_args(args, False), EXPECTED)

class TestWaveforms(unittest.TestCase):
  @unittest.mock.patch("sys.argv", ["picker.py", "-D", "230601", "230604",
                                    "-d", str(TEST_PATH), "--force", "-v"])
  def tearDownClass():
    args = ini.parse_arguments()
    _ = ini.dump_args(args, True)
    _ = ini.waveform_table(args)

  @unittest.mock.patch("sys.argv", ["picker.py", "-d", TEST_PATH.__str__(),
                                    "-v"])
  def test_default(self):
    args = ini.parse_arguments()
    WAVEFORMS_DATA = ini.waveform_table(args).values.tolist()
    EXPECTED = [["MN", "TRI", "HHE", "230601"],
                ["MN", "TRI", "HHN", "230601"],
                ["MN", "TRI", "HHZ", "230601"],
                ["OX", "BAD", "HHE", "230601"],
                ["OX", "BAD", "HHN", "230601"],
                ["OX", "BAD", "HHZ", "230601"],
                ["OX", "CAE", "HHE", "230601"],
                ["OX", "CAE", "HHN", "230601"],
                ["OX", "CAE", "HHZ", "230601"],
                ["ST", "VARA", "EHE", "230601"],
                ["ST", "VARA", "EHN", "230601"],
                ["ST", "VARA", "EHZ", "230601"],
                ["MN", "TRI", "HHE", "230602"],
                ["MN", "TRI", "HHN", "230602"],
                ["MN", "TRI", "HHZ", "230602"],
                ["OX", "BAD", "HHE", "230602"],
                ["OX", "BAD", "HHN", "230602"],
                ["OX", "BAD", "HHZ", "230602"],
                ["OX", "CAE", "HHE", "230602"],
                ["OX", "CAE", "HHN", "230602"],
                ["OX", "CAE", "HHZ", "230602"],
                ["MN", "TRI", "HHE", "230603"],
                ["MN", "TRI", "HHN", "230603"],
                ["MN", "TRI", "HHZ", "230603"],
                ["OX", "BAD", "HHE", "230603"],
                ["OX", "BAD", "HHN", "230603"],
                ["OX", "BAD", "HHZ", "230603"],
                ["OX", "CAE", "HHE", "230603"],
                ["OX", "CAE", "HHN", "230603"],
                ["OX", "CAE", "HHZ", "230603"],
                ["ST", "VARA", "EHE", "230603"],
                ["ST", "VARA", "EHN", "230603"],
                ["ST", "VARA", "EHZ", "230603"],
                ["MN", "TRI", "HHE", "230604"],
                ["MN", "TRI", "HHN", "230604"],
                ["MN", "TRI", "HHZ", "230604"],
                ["OX", "BAD", "HHE", "230604"],
                ["OX", "BAD", "HHN", "230604"],
                ["OX", "BAD", "HHZ", "230604"],
                ["OX", "CAE", "HHE", "230604"],
                ["OX", "CAE", "HHN", "230604"],
                ["OX", "CAE", "HHZ", "230604"],
                ["ST", "VARA", "EHE", "230604"],
                ["ST", "VARA", "EHN", "230604"],
                ["ST", "VARA", "EHZ", "230604"]]
    self.assertListEqual(EXPECTED, WAVEFORMS_DATA)

  @unittest.mock.patch("sys.argv", ["picker.py", "-d", TEST_PATH.__str__(),
                                    "-N", "OX", "-v"])
  def test_network(self):
    args = ini.parse_arguments()
    WAVEFORMS_DATA = ini.waveform_table(args).values.tolist()
    EXPECTED = [["OX", "BAD", "HHE", "230601"],
                ["OX", "BAD", "HHN", "230601"],
                ["OX", "BAD", "HHZ", "230601"],
                ["OX", "CAE", "HHE", "230601"],
                ["OX", "CAE", "HHN", "230601"],
                ["OX", "CAE", "HHZ", "230601"],
                ["OX", "BAD", "HHE", "230602"],
                ["OX", "BAD", "HHN", "230602"],
                ["OX", "BAD", "HHZ", "230602"],
                ["OX", "CAE", "HHE", "230602"],
                ["OX", "CAE", "HHN", "230602"],
                ["OX", "CAE", "HHZ", "230602"],
                ["OX", "BAD", "HHE", "230603"],
                ["OX", "BAD", "HHN", "230603"],
                ["OX", "BAD", "HHZ", "230603"],
                ["OX", "CAE", "HHE", "230603"],
                ["OX", "CAE", "HHN", "230603"],
                ["OX", "CAE", "HHZ", "230603"],
                ["OX", "BAD", "HHE", "230604"],
                ["OX", "BAD", "HHN", "230604"],
                ["OX", "BAD", "HHZ", "230604"],
                ["OX", "CAE", "HHE", "230604"],
                ["OX", "CAE", "HHN", "230604"],
                ["OX", "CAE", "HHZ", "230604"]]
    self.assertListEqual(EXPECTED, WAVEFORMS_DATA)

  @unittest.mock.patch("sys.argv", ["picker.py", "-d", TEST_PATH.__str__(),
                                    "-N", "OX", "ST", "-v"])
  def test_networks(self):
    args = ini.parse_arguments()
    WAVEFORMS_DATA = ini.waveform_table(args).values.tolist()
    EXPECTED = [["OX", "BAD", "HHE", "230601"],
                ["OX", "BAD", "HHN", "230601"],
                ["OX", "BAD", "HHZ", "230601"],
                ["OX", "CAE", "HHE", "230601"],
                ["OX", "CAE", "HHN", "230601"],
                ["OX", "CAE", "HHZ", "230601"],
                ["ST", "VARA", "EHE", "230601"],
                ["ST", "VARA", "EHN", "230601"],
                ["ST", "VARA", "EHZ", "230601"],
                ["OX", "BAD", "HHE", "230602"],
                ["OX", "BAD", "HHN", "230602"],
                ["OX", "BAD", "HHZ", "230602"],
                ["OX", "CAE", "HHE", "230602"],
                ["OX", "CAE", "HHN", "230602"],
                ["OX", "CAE", "HHZ", "230602"],
                ["OX", "BAD", "HHE", "230603"],
                ["OX", "BAD", "HHN", "230603"],
                ["OX", "BAD", "HHZ", "230603"],
                ["OX", "CAE", "HHE", "230603"],
                ["OX", "CAE", "HHN", "230603"],
                ["OX", "CAE", "HHZ", "230603"],
                ["ST", "VARA", "EHE", "230603"],
                ["ST", "VARA", "EHN", "230603"],
                ["ST", "VARA", "EHZ", "230603"],
                ["OX", "BAD", "HHE", "230604"],
                ["OX", "BAD", "HHN", "230604"],
                ["OX", "BAD", "HHZ", "230604"],
                ["OX", "CAE", "HHE", "230604"],
                ["OX", "CAE", "HHN", "230604"],
                ["OX", "CAE", "HHZ", "230604"],
                ["ST", "VARA", "EHE", "230604"],
                ["ST", "VARA", "EHN", "230604"],
                ["ST", "VARA", "EHZ", "230604"]]
    self.assertListEqual(EXPECTED, WAVEFORMS_DATA)

  @unittest.mock.patch("sys.argv", ["picker.py", "-d", TEST_PATH.__str__(),
                                    "-S", "VARA", "-v"])
  def test_station(self):
    args = ini.parse_arguments()
    WAVEFORMS_DATA = ini.waveform_table(args).values.tolist()
    EXPECTED = [["ST", "VARA", "EHE", "230601"],
                ["ST", "VARA", "EHN", "230601"],
                ["ST", "VARA", "EHZ", "230601"],
                ["ST", "VARA", "EHE", "230603"],
                ["ST", "VARA", "EHN", "230603"],
                ["ST", "VARA", "EHZ", "230603"],
                ["ST", "VARA", "EHE", "230604"],
                ["ST", "VARA", "EHN", "230604"],
                ["ST", "VARA", "EHZ", "230604"]]
    self.assertListEqual(EXPECTED, WAVEFORMS_DATA)

  @unittest.mock.patch("sys.argv", ["picker.py", "-d", TEST_PATH.__str__(),
                                    "-S", "VARA", "BAD", "-v"])
  def test_stations(self):
    args = ini.parse_arguments()
    WAVEFORMS_DATA = ini.waveform_table(args).values.tolist()
    EXPECTED = [["OX", "BAD", "HHE", "230601"],
                ["OX", "BAD", "HHN", "230601"],
                ["OX", "BAD", "HHZ", "230601"],
                ["ST", "VARA", "EHE", "230601"],
                ["ST", "VARA", "EHN", "230601"],
                ["ST", "VARA", "EHZ", "230601"],
                ["OX", "BAD", "HHE", "230602"],
                ["OX", "BAD", "HHN", "230602"],
                ["OX", "BAD", "HHZ", "230602"],
                ["OX", "BAD", "HHE", "230603"],
                ["OX", "BAD", "HHN", "230603"],
                ["OX", "BAD", "HHZ", "230603"],
                ["ST", "VARA", "EHE", "230603"],
                ["ST", "VARA", "EHN", "230603"],
                ["ST", "VARA", "EHZ", "230603"],
                ["OX", "BAD", "HHE", "230604"],
                ["OX", "BAD", "HHN", "230604"],
                ["OX", "BAD", "HHZ", "230604"],
                ["ST", "VARA", "EHE", "230604"],
                ["ST", "VARA", "EHN", "230604"],
                ["ST", "VARA", "EHZ", "230604"]]
    self.assertListEqual(EXPECTED, WAVEFORMS_DATA)

  @unittest.mock.patch("sys.argv", ["picker.py", "-d", TEST_PATH.__str__(),
                                    "-N", "EG", "-v"])
  def test_no_network(self):
    args = ini.parse_arguments()
    self.assertRaises(FileNotFoundError, ini.waveform_table, args)

  @unittest.mock.patch("sys.argv", ["picker.py", "-d", TEST_PATH.__str__(),
                                    "-N", "EG", "EA", "-v"])
  def test_no_networks(self):
    args = ini.parse_arguments()
    self.assertRaises(FileNotFoundError, ini.waveform_table, args)

  @unittest.mock.patch("sys.argv", ["picker.py", "-d", TEST_PATH.__str__(),
                                    "-S", "MAD", "-v"])
  def test_no_station(self):
    args = ini.parse_arguments()
    self.assertRaises(FileNotFoundError, ini.waveform_table, args)

  @unittest.mock.patch("sys.argv", ["picker.py", "-d", TEST_PATH.__str__(),
                                    "-S", "MAD", "CAB", "-v"])
  def test_no_stations(self):
    args = ini.parse_arguments()
    self.assertRaises(FileNotFoundError, ini.waveform_table, args)

  @unittest.mock.patch("sys.argv", ["picker.py", "-d", TEST_PATH.__str__(),
                                    "-N", "OX", "EG", "-v"])
  def test_semi_network(self):
    args = ini.parse_arguments()
    EXPECTED = [["OX", "BAD", "HHE", "230601"],
                ["OX", "BAD", "HHN", "230601"],
                ["OX", "BAD", "HHZ", "230601"],
                ["OX", "CAE", "HHE", "230601"],
                ["OX", "CAE", "HHN", "230601"],
                ["OX", "CAE", "HHZ", "230601"],
                ["OX", "BAD", "HHE", "230602"],
                ["OX", "BAD", "HHN", "230602"],
                ["OX", "BAD", "HHZ", "230602"],
                ["OX", "CAE", "HHE", "230602"],
                ["OX", "CAE", "HHN", "230602"],
                ["OX", "CAE", "HHZ", "230602"],
                ["OX", "BAD", "HHE", "230603"],
                ["OX", "BAD", "HHN", "230603"],
                ["OX", "BAD", "HHZ", "230603"],
                ["OX", "CAE", "HHE", "230603"],
                ["OX", "CAE", "HHN", "230603"],
                ["OX", "CAE", "HHZ", "230603"],
                ["OX", "BAD", "HHE", "230604"],
                ["OX", "BAD", "HHN", "230604"],
                ["OX", "BAD", "HHZ", "230604"],
                ["OX", "CAE", "HHE", "230604"],
                ["OX", "CAE", "HHN", "230604"],
                ["OX", "CAE", "HHZ", "230604"]]
    self.assertListEqual(EXPECTED, ini.waveform_table(args).values.tolist())

class TestClassifications(unittest.TestCase):
  @unittest.mock.patch("sys.argv", ["picker.py", "-d", TEST_PATH.__str__(),
                                    "-v"])
  def test_default(self):
    args = ini.parse_arguments()
    CLASSIFICATIONS_DATA = ini.data_header(args, CLF_STR).values.tolist()
    EXPECTED = [[EQTRANSFORMER_STR, INSTANCE_STR, "230601", "MN", "TRI"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230601", "OX", "BAD"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230601", "OX", "CAE"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230602", "MN", "TRI"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230602", "OX", "BAD"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230602", "OX", "CAE"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230603", "MN", "TRI"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230603", "OX", "BAD"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230603", "OX", "CAE"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230604", "MN", "TRI"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230604", "OX", "BAD"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230604", "OX", "CAE"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230604", "ST", "VARA"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230601", "MN", "TRI"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230601", "OX", "BAD"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230601", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230602", "MN", "TRI"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230602", "OX", "BAD"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230602", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230603", "MN", "TRI"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230603", "OX", "BAD"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230603", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230604", "MN", "TRI"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230604", "OX", "BAD"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230604", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230604", "ST", "VARA"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230601", "MN", "TRI"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230601", "OX", "BAD"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230601", "OX", "CAE"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230602", "MN", "TRI"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230602", "OX", "BAD"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230602", "OX", "CAE"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230603", "MN", "TRI"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230603", "OX", "BAD"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230603", "OX", "CAE"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230604", "MN", "TRI"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230604", "OX", "BAD"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230604", "OX", "CAE"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230604", "ST", "VARA"],
                [EQTRANSFORMER_STR, STEAD_STR, "230601", "MN", "TRI"],
                [EQTRANSFORMER_STR, STEAD_STR, "230601", "OX", "BAD"],
                [EQTRANSFORMER_STR, STEAD_STR, "230601", "OX", "CAE"],
                [EQTRANSFORMER_STR, STEAD_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, STEAD_STR, "230602", "MN", "TRI"],
                [EQTRANSFORMER_STR, STEAD_STR, "230602", "OX", "BAD"],
                [EQTRANSFORMER_STR, STEAD_STR, "230602", "OX", "CAE"],
                [EQTRANSFORMER_STR, STEAD_STR, "230603", "MN", "TRI"],
                [EQTRANSFORMER_STR, STEAD_STR, "230603", "OX", "BAD"],
                [EQTRANSFORMER_STR, STEAD_STR, "230603", "OX", "CAE"],
                [EQTRANSFORMER_STR, STEAD_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, STEAD_STR, "230604", "MN", "TRI"],
                [EQTRANSFORMER_STR, STEAD_STR, "230604", "OX", "BAD"],
                [EQTRANSFORMER_STR, STEAD_STR, "230604", "OX", "CAE"],
                [EQTRANSFORMER_STR, STEAD_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, INSTANCE_STR, "230601", "MN", "TRI"],
                [PHASENET_STR, INSTANCE_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, INSTANCE_STR, "230602", "MN", "TRI"],
                [PHASENET_STR, INSTANCE_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230603", "MN", "TRI"],
                [PHASENET_STR, INSTANCE_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, INSTANCE_STR, "230604", "MN", "TRI"],
                [PHASENET_STR, INSTANCE_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "MN", "TRI"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230602", "MN", "TRI"],
                [PHASENET_STR, ORIGINAL_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "MN", "TRI"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "MN", "TRI"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, SCEDC_STR, "230601", "MN", "TRI"],
                [PHASENET_STR, SCEDC_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, SCEDC_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, SCEDC_STR, "230602", "MN", "TRI"],
                [PHASENET_STR, SCEDC_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, SCEDC_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230603", "MN", "TRI"],
                [PHASENET_STR, SCEDC_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, SCEDC_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, SCEDC_STR, "230604", "MN", "TRI"],
                [PHASENET_STR, SCEDC_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, SCEDC_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, STEAD_STR, "230601", "MN", "TRI"],
                [PHASENET_STR, STEAD_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, STEAD_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, STEAD_STR, "230602", "MN", "TRI"],
                [PHASENET_STR, STEAD_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, STEAD_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230603", "MN", "TRI"],
                [PHASENET_STR, STEAD_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, STEAD_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, STEAD_STR, "230604", "MN", "TRI"],
                [PHASENET_STR, STEAD_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, STEAD_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230604", "ST", "VARA"]]
    self.assertListEqual(EXPECTED, CLASSIFICATIONS_DATA)

  @unittest.mock.patch("sys.argv", ["picker.py", "-d", TEST_PATH.__str__(),
                                    "-N", "OX", "-v"])
  def test_network(self):
    args = ini.parse_arguments()
    CLASSIFICATIONS_DATA = ini.data_header(args, CLF_STR).values.tolist()
    EXPECTED = [[EQTRANSFORMER_STR, INSTANCE_STR, "230601", "OX", "BAD"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230601", "OX", "CAE"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230602", "OX", "BAD"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230602", "OX", "CAE"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230603", "OX", "BAD"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230603", "OX", "CAE"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230604", "OX", "BAD"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230604", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230601", "OX", "BAD"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230601", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230602", "OX", "BAD"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230602", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230603", "OX", "BAD"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230603", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230604", "OX", "BAD"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230604", "OX", "CAE"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230601", "OX", "BAD"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230601", "OX", "CAE"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230602", "OX", "BAD"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230602", "OX", "CAE"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230603", "OX", "BAD"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230603", "OX", "CAE"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230604", "OX", "BAD"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230604", "OX", "CAE"],
                [EQTRANSFORMER_STR, STEAD_STR, "230601", "OX", "BAD"],
                [EQTRANSFORMER_STR, STEAD_STR, "230601", "OX", "CAE"],
                [EQTRANSFORMER_STR, STEAD_STR, "230602", "OX", "BAD"],
                [EQTRANSFORMER_STR, STEAD_STR, "230602", "OX", "CAE"],
                [EQTRANSFORMER_STR, STEAD_STR, "230603", "OX", "BAD"],
                [EQTRANSFORMER_STR, STEAD_STR, "230603", "OX", "CAE"],
                [EQTRANSFORMER_STR, STEAD_STR, "230604", "OX", "BAD"],
                [EQTRANSFORMER_STR, STEAD_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, SCEDC_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, SCEDC_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, SCEDC_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, SCEDC_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, STEAD_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, STEAD_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, STEAD_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, STEAD_STR, "230604", "OX", "CAE"]]
    self.assertListEqual(EXPECTED, CLASSIFICATIONS_DATA)

  @unittest.mock.patch("sys.argv", ["picker.py", "-d", TEST_PATH.__str__(),
                                    "-N", "OX", "ST", "-v"])
  def test_networks(self):
    args = ini.parse_arguments()
    CLASSIFICATIONS_DATA = ini.data_header(args, CLF_STR).values.tolist()
    EXPECTED = [[EQTRANSFORMER_STR, INSTANCE_STR, "230601", "OX", "BAD"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230601", "OX", "CAE"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230602", "OX", "BAD"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230602", "OX", "CAE"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230603", "OX", "BAD"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230603", "OX", "CAE"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230604", "OX", "BAD"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230604", "OX", "CAE"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230604", "ST", "VARA"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230601", "OX", "BAD"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230601", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230602", "OX", "BAD"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230602", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230603", "OX", "BAD"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230603", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230604", "OX", "BAD"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230604", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230604", "ST", "VARA"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230601", "OX", "BAD"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230601", "OX", "CAE"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230602", "OX", "BAD"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230602", "OX", "CAE"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230603", "OX", "BAD"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230603", "OX", "CAE"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230604", "OX", "BAD"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230604", "OX", "CAE"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230604", "ST", "VARA"],
                [EQTRANSFORMER_STR, STEAD_STR, "230601", "OX", "BAD"],
                [EQTRANSFORMER_STR, STEAD_STR, "230601", "OX", "CAE"],
                [EQTRANSFORMER_STR, STEAD_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, STEAD_STR, "230602", "OX", "BAD"],
                [EQTRANSFORMER_STR, STEAD_STR, "230602", "OX", "CAE"],
                [EQTRANSFORMER_STR, STEAD_STR, "230603", "OX", "BAD"],
                [EQTRANSFORMER_STR, STEAD_STR, "230603", "OX", "CAE"],
                [EQTRANSFORMER_STR, STEAD_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, STEAD_STR, "230604", "OX", "BAD"],
                [EQTRANSFORMER_STR, STEAD_STR, "230604", "OX", "CAE"],
                [EQTRANSFORMER_STR, STEAD_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, INSTANCE_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, INSTANCE_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, INSTANCE_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, SCEDC_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, SCEDC_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, SCEDC_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, SCEDC_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, SCEDC_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, SCEDC_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, SCEDC_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, STEAD_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, STEAD_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, STEAD_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, STEAD_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, STEAD_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, STEAD_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, STEAD_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230604", "ST", "VARA"]]
    self.assertListEqual(EXPECTED, CLASSIFICATIONS_DATA)

  @unittest.mock.patch("sys.argv", ["picker.py", "-d", TEST_PATH.__str__(),
                                    "-S", "VARA", "-v"])
  def test_station(self):
    args = ini.parse_arguments()
    CLASSIFICATIONS_DATA = ini.data_header(args, CLF_STR).values.tolist()
    EXPECTED = [[EQTRANSFORMER_STR, INSTANCE_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230604", "ST", "VARA"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230604", "ST", "VARA"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230604", "ST", "VARA"],
                [EQTRANSFORMER_STR, STEAD_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, STEAD_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, STEAD_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, INSTANCE_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, INSTANCE_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, INSTANCE_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, SCEDC_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, SCEDC_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, SCEDC_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, STEAD_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, STEAD_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, STEAD_STR, "230604", "ST", "VARA"]]
    self.assertListEqual(EXPECTED, CLASSIFICATIONS_DATA)

  @unittest.mock.patch("sys.argv", ["picker.py", "-d", TEST_PATH.__str__(),
                                    "-S", "VARA", "CAE", "-v"])
  def test_stations(self):
    args = ini.parse_arguments()
    CLASSIFICATIONS_DATA = ini.data_header(args, CLF_STR).values.tolist()
    EXPECTED = [[EQTRANSFORMER_STR, INSTANCE_STR, "230601", "OX", "CAE"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230602", "OX", "CAE"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230603", "OX", "CAE"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230604", "OX", "CAE"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230604", "ST", "VARA"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230601", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230602", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230603", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230604", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230604", "ST", "VARA"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230601", "OX", "CAE"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230602", "OX", "CAE"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230603", "OX", "CAE"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230604", "OX", "CAE"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230604", "ST", "VARA"],
                [EQTRANSFORMER_STR, STEAD_STR, "230601", "OX", "CAE"],
                [EQTRANSFORMER_STR, STEAD_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, STEAD_STR, "230602", "OX", "CAE"],
                [EQTRANSFORMER_STR, STEAD_STR, "230603", "OX", "CAE"],
                [EQTRANSFORMER_STR, STEAD_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, STEAD_STR, "230604", "OX", "CAE"],
                [EQTRANSFORMER_STR, STEAD_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, INSTANCE_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, INSTANCE_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, INSTANCE_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, SCEDC_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, SCEDC_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, SCEDC_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, STEAD_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, STEAD_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, STEAD_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230604", "ST", "VARA"]]
    self.assertListEqual(EXPECTED, CLASSIFICATIONS_DATA)

  @unittest.mock.patch("sys.argv", ["picker.py", "-d", TEST_PATH.__str__(),
                                    "-S", "VARA", "EG", "-v"])
  def test_semi_station(self):
    args = ini.parse_arguments()
    CLASSIFICATIONS_DATA = ini.data_header(args, CLF_STR).values.tolist()
    EXPECTED = [[EQTRANSFORMER_STR, INSTANCE_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230604", "ST", "VARA"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230604", "ST", "VARA"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, SCEDC_STR, "230604", "ST", "VARA"],
                [EQTRANSFORMER_STR, STEAD_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, STEAD_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, STEAD_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, INSTANCE_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, INSTANCE_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, INSTANCE_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, SCEDC_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, SCEDC_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, SCEDC_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, STEAD_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, STEAD_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, STEAD_STR, "230604", "ST", "VARA"]]
    self.assertListEqual(EXPECTED, CLASSIFICATIONS_DATA)

  @unittest.mock.patch("sys.argv", ["picker.py", "-d", TEST_PATH.__str__(),
                                    "-M", PHASENET_STR, "-v"])
  def test_model(self):
    args = ini.parse_arguments()
    CLASSIFICATIONS_DATA = ini.data_header(args, CLF_STR).values.tolist()
    EXPECTED = [[PHASENET_STR, INSTANCE_STR, "230601", "MN", "TRI"],
                [PHASENET_STR, INSTANCE_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, INSTANCE_STR, "230602", "MN", "TRI"],
                [PHASENET_STR, INSTANCE_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230603", "MN", "TRI"],
                [PHASENET_STR, INSTANCE_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, INSTANCE_STR, "230604", "MN", "TRI"],
                [PHASENET_STR, INSTANCE_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "MN", "TRI"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230602", "MN", "TRI"],
                [PHASENET_STR, ORIGINAL_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "MN", "TRI"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "MN", "TRI"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, SCEDC_STR, "230601", "MN", "TRI"],
                [PHASENET_STR, SCEDC_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, SCEDC_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, SCEDC_STR, "230602", "MN", "TRI"],
                [PHASENET_STR, SCEDC_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, SCEDC_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230603", "MN", "TRI"],
                [PHASENET_STR, SCEDC_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, SCEDC_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, SCEDC_STR, "230604", "MN", "TRI"],
                [PHASENET_STR, SCEDC_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, SCEDC_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, STEAD_STR, "230601", "MN", "TRI"],
                [PHASENET_STR, STEAD_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, STEAD_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, STEAD_STR, "230602", "MN", "TRI"],
                [PHASENET_STR, STEAD_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, STEAD_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230603", "MN", "TRI"],
                [PHASENET_STR, STEAD_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, STEAD_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, STEAD_STR, "230604", "MN", "TRI"],
                [PHASENET_STR, STEAD_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, STEAD_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230604", "ST", "VARA"]]
    self.assertListEqual(EXPECTED, CLASSIFICATIONS_DATA)

  @unittest.mock.patch("sys.argv", ["picker.py", "-d", TEST_PATH.__str__(),
                                    "-M", PHASENET_STR, DEEPDENOISER_STR,
                                    "-v"])
  def test_semi_model(self):
    args = ini.parse_arguments()
    CLASSIFICATIONS_DATA = ini.data_header(args, CLF_STR).values.tolist()
    EXPECTED = [[PHASENET_STR, INSTANCE_STR, "230601", "MN", "TRI"],
                [PHASENET_STR, INSTANCE_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, INSTANCE_STR, "230602", "MN", "TRI"],
                [PHASENET_STR, INSTANCE_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230603", "MN", "TRI"],
                [PHASENET_STR, INSTANCE_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, INSTANCE_STR, "230604", "MN", "TRI"],
                [PHASENET_STR, INSTANCE_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "MN", "TRI"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230602", "MN", "TRI"],
                [PHASENET_STR, ORIGINAL_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "MN", "TRI"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "MN", "TRI"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, SCEDC_STR, "230601", "MN", "TRI"],
                [PHASENET_STR, SCEDC_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, SCEDC_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, SCEDC_STR, "230602", "MN", "TRI"],
                [PHASENET_STR, SCEDC_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, SCEDC_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230603", "MN", "TRI"],
                [PHASENET_STR, SCEDC_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, SCEDC_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, SCEDC_STR, "230604", "MN", "TRI"],
                [PHASENET_STR, SCEDC_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, SCEDC_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, SCEDC_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, STEAD_STR, "230601", "MN", "TRI"],
                [PHASENET_STR, STEAD_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, STEAD_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, STEAD_STR, "230602", "MN", "TRI"],
                [PHASENET_STR, STEAD_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, STEAD_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230603", "MN", "TRI"],
                [PHASENET_STR, STEAD_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, STEAD_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, STEAD_STR, "230604", "MN", "TRI"],
                [PHASENET_STR, STEAD_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, STEAD_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, STEAD_STR, "230604", "ST", "VARA"]]
    self.assertListEqual(EXPECTED, CLASSIFICATIONS_DATA)

  @unittest.mock.patch("sys.argv", ["picker.py", "-d", TEST_PATH.__str__(),
                                    "-W", ORIGINAL_STR, "-v"])
  def test_weight(self):
    args = ini.parse_arguments()
    CLASSIFICATIONS_DATA = ini.data_header(args, CLF_STR).values.tolist()
    EXPECTED = [[EQTRANSFORMER_STR, ORIGINAL_STR, "230601", "MN", "TRI"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230601", "OX", "BAD"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230601", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230602", "MN", "TRI"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230602", "OX", "BAD"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230602", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230603", "MN", "TRI"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230603", "OX", "BAD"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230603", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230604", "MN", "TRI"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230604", "OX", "BAD"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230604", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "MN", "TRI"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230602", "MN", "TRI"],
                [PHASENET_STR, ORIGINAL_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "MN", "TRI"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "MN", "TRI"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "ST", "VARA"]]
    self.assertListEqual(EXPECTED, CLASSIFICATIONS_DATA)

  @unittest.mock.patch("sys.argv", ["picker.py", "-d", TEST_PATH.__str__(),
                                    "-W", ORIGINAL_STR, INSTANCE_STR, "-v"])
  def test_weights(self):
    args = ini.parse_arguments()
    CLASSIFICATIONS_DATA = ini.data_header(args, CLF_STR).values.tolist()
    EXPECTED = [[EQTRANSFORMER_STR, INSTANCE_STR, "230601", "MN", "TRI"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230601", "OX", "BAD"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230601", "OX", "CAE"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230602", "MN", "TRI"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230602", "OX", "BAD"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230602", "OX", "CAE"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230603", "MN", "TRI"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230603", "OX", "BAD"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230603", "OX", "CAE"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230604", "MN", "TRI"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230604", "OX", "BAD"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230604", "OX", "CAE"],
                [EQTRANSFORMER_STR, INSTANCE_STR, "230604", "ST", "VARA"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230601", "MN", "TRI"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230601", "OX", "BAD"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230601", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230601", "ST", "VARA"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230602", "MN", "TRI"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230602", "OX", "BAD"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230602", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230603", "MN", "TRI"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230603", "OX", "BAD"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230603", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230603", "ST", "VARA"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230604", "MN", "TRI"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230604", "OX", "BAD"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230604", "OX", "CAE"],
                [EQTRANSFORMER_STR, ORIGINAL_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, INSTANCE_STR, "230601", "MN", "TRI"],
                [PHASENET_STR, INSTANCE_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, INSTANCE_STR, "230602", "MN", "TRI"],
                [PHASENET_STR, INSTANCE_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230603", "MN", "TRI"],
                [PHASENET_STR, INSTANCE_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, INSTANCE_STR, "230604", "MN", "TRI"],
                [PHASENET_STR, INSTANCE_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, INSTANCE_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, INSTANCE_STR, "230604", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "MN", "TRI"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230601", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230602", "MN", "TRI"],
                [PHASENET_STR, ORIGINAL_STR, "230602", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230602", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "MN", "TRI"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230603", "ST", "VARA"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "MN", "TRI"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "OX", "BAD"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "OX", "CAE"],
                [PHASENET_STR, ORIGINAL_STR, "230604", "ST", "VARA"]]
    self.assertListEqual(EXPECTED, CLASSIFICATIONS_DATA)

if __name__ == "__main__": unittest.main()
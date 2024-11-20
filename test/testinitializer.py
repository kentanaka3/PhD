#!/bin/python
import os
from pathlib import Path
PRJ_PATH = Path(os.path.dirname(__file__)).parent
DATA_PATH = os.path.join(PRJ_PATH, "data")
INC_PATH = os.path.join(PRJ_PATH, "inc")
import sys
# Add to path
if INC_PATH not in sys.path: sys.path.append(INC_PATH)
import unittest
from obspy.core.utcdatetime import UTCDateTime

from constants import *
import initializer as ini

class TestArgparse(unittest.TestCase):
  def setUp(self):
    with open(Path(DATA_PATH, "file.key"), 'w') as fp:
      pass

  def tearDown(self):
    os.remove(Path(DATA_PATH, "file.key"))

  @unittest.mock.patch("sys.argv", ["picker.py"])
  def test_non_args(self):
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

if __name__ == "__main__": unittest.main()
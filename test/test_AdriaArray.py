#!/bin/python
import unittest
from unittest.mock import patch
from src.AdriaArray import *

class TestArgparse(unittest.TestCase):
  @patch("sys.argv", ["AdriaArray.py"])
  def test_non_args(self):
    args = parse_arguments()
    self.assertEqual(args.dates, ["20230601", "20230731"])
    self.assertEqual(args.groups, [BEG_DATE_STR])
    self.assertEqual(args.models, [PHASENET_STR])
    self.assertEqual(args.verbose, False)
    self.assertEqual(args.weights, [INSTANCE_STR])

  @patch("sys.argv", ["AdriaArray.py", "-M", PHASENET_STR])
  def test_model_args(self):
    args = parse_arguments()
    self.assertEqual(args.dates, ["20230601", "20230731"])
    self.assertEqual(args.groups, [BEG_DATE_STR])
    self.assertEqual(args.models, [PHASENET_STR])
    self.assertEqual(args.verbose, False)
    self.assertEqual(args.weights, [INSTANCE_STR])

  @patch("sys.argv", ["AdriaArray.py", "-M", PHASENET_STR, EQTRANSFORMER_STR])
  def test_models_args(self):
    args = parse_arguments()
    self.assertEqual(args.dates, ["20230601", "20230731"])
    self.assertEqual(args.groups, [BEG_DATE_STR])
    self.assertEqual(args.models, [PHASENET_STR, EQTRANSFORMER_STR])
    self.assertEqual(args.verbose, False)
    self.assertEqual(args.weights, [INSTANCE_STR])

  @patch("sys.argv", ["AdriaArray.py", "-W", INSTANCE_STR])
  def test_weight_args(self):
    args = parse_arguments()
    self.assertEqual(args.dates, ["20230601", "20230731"])
    self.assertEqual(args.models, [PHASENET_STR])
    self.assertEqual(args.groups, [BEG_DATE_STR])
    self.assertEqual(args.verbose, False)
    self.assertEqual(args.weights, [INSTANCE_STR])

  @patch("sys.argv", ["AdriaArray.py", "-W", INSTANCE_STR, ORIGINAL_STR])
  def test_weights_args(self):
    args = parse_arguments()
    self.assertEqual(args.dates, ["20230601", "20230731"])
    self.assertEqual(args.groups, [BEG_DATE_STR])
    self.assertEqual(args.models, [PHASENET_STR])
    self.assertEqual(args.verbose, False)
    self.assertEqual(args.weights, [INSTANCE_STR, ORIGINAL_STR])

  @patch("sys.argv", ["AdriaArray.py", "-G", BEG_DATE_STR])
  def test_group_args(self):
    args = parse_arguments()
    self.assertEqual(args.dates, ["20230601", "20230731"])
    self.assertEqual(args.groups, [BEG_DATE_STR])
    self.assertEqual(args.models, [PHASENET_STR])
    self.assertEqual(args.verbose, False)
    self.assertEqual(args.weights, [INSTANCE_STR])

  @patch("sys.argv", ["AdriaArray.py", "-G", BEG_DATE_STR, NETWORK_STR,
                      STATION_STR])
  def test_groups_args(self):
    args = parse_arguments()
    self.assertEqual(args.dates, ["20230601", "20230731"])
    self.assertEqual(args.groups, [BEG_DATE_STR, NETWORK_STR, STATION_STR])
    self.assertEqual(args.models, [PHASENET_STR])
    self.assertEqual(args.verbose, False)
    self.assertEqual(args.weights, [INSTANCE_STR])

  @patch("sys.argv", ["AdriaArray.py", "-D", "20230601", "20230731"])
  def test_range_args(self):
    args = parse_arguments()
    self.assertEqual(args.dates, ["20230601", "20230731"])
    self.assertEqual(args.groups, [BEG_DATE_STR])
    self.assertEqual(args.models, [PHASENET_STR])
    self.assertEqual(args.verbose, False)
    self.assertEqual(args.weights, [INSTANCE_STR])

  @patch("sys.argv", ["AdriaArray.py", "-v"])
  def test_verbose_args(self):
    args = parse_arguments()
    self.assertEqual(args.dates, ["20230601", "20230731"])
    self.assertEqual(args.groups, [BEG_DATE_STR])
    self.assertEqual(args.models, [PHASENET_STR])
    self.assertEqual(args.verbose, True)
    self.assertEqual(args.weights, [INSTANCE_STR])

class TestWaveformTable(unittest.TestCase):
  @patch("sys.argv", ["AdriaArray.py"])
  def test_non_args(self):
    args = parse_arguments()
    RAW_DATA_PATH = os.path.join(DATA_PATH, "test")
    WAVEFORMS_DATA = waveform_table(args)

if __name__ == "__main__":
  unittest.main()
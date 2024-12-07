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

class TestReadTraces(unittest.TestCase):
  def tearDown(self) -> None:
    Path(DATA_PATH, WAVEFORMS_STR + CSV_EXT).unlink()
    Path(DATA_PATH, ARGUMENTS_STR + JSON_EXT).unlink()

  @unittest.mock.patch("sys.argv", ["picker.py", "-G", BEG_DATE_STR, "-v",
                                    "-d", str(TEST_PATH)])
  def test_group_args(self):
    args = ini.parse_arguments()
    WAVEFORMS_DATA = ini.waveform_table(args)
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
    WAVEFORMS_DATA = ini.waveform_table(args)
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
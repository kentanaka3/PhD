#!/bin/python
import os
from pathlib import Path
import unittest.mock
PRJ_PATH = Path(os.path.dirname(__file__)).parent
SRC_PATH = os.path.join(PRJ_PATH, "src")
import sys
# Add to path
if SRC_PATH not in sys.path: sys.path.append(SRC_PATH)
import unittest
import shutil
import json
from AdriaArray import *

BASE_PATH = Path(PRJ_PATH, "data")
DATA_PATH = Path(BASE_PATH, "test")
TEST_PATH = Path(DATA_PATH, "waveforms")

ANNOTATION_TD_OFFSET = td(seconds=2.5)

class TestArgparse(unittest.TestCase):
  def setUp(self):
    with open(Path(DATA_PATH, "file.key"), 'w') as fp:
      pass

  def tearDown(self):
    os.remove(Path(DATA_PATH, "file.key"))

  @unittest.mock.patch("sys.argv", ["AdriaArray.py"])
  def test_non_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, None)
    self.assertEqual(args.client, [INGV_STR])
    self.assertEqual(args.batch, 4096)
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=8, day=1)])
    self.assertEqual(args.directory, Path(BASE_PATH, "waveforms"))
    self.assertEqual(args.domain, [44.5, 47, 10, 14])
    self.assertEqual(args.download, False)
    self.assertEqual(args.groups, [BEG_DATE_STR, NETWORK_STR, STATION_STR])
    self.assertEqual(args.julian, False)
    self.assertEqual(args.key, None)
    self.assertEqual(args.models, [PHASENET_STR, EQTRANSFORMER_STR])
    self.assertEqual(args.network, None)
    self.assertEqual(args.pwave, 0.2)
    self.assertEqual(args.pyrocko, False)
    self.assertEqual(args.station, None)
    self.assertEqual(args.swave, 0.1)
    self.assertEqual(args.train, False)
    self.assertEqual(args.verbose, False)
    self.assertEqual(args.weights, [INSTANCE_STR, ORIGINAL_STR, STEAD_STR,
                                    SCEDC_STR])

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "-C",  "EHZ"])
  def test_channel_args(self):
    args = parse_arguments()
    self.assertEqual(args.channel, ["EHZ"])

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "--client", IRIS_STR])
  def test_client_args(self):
    args = parse_arguments()
    self.assertEqual(args.client, [IRIS_STR])

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "-D", "230602", "230603"])
  def test_dates_args(self):
    args = parse_arguments()
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=2),
                                  UTCDateTime(year=2023, month=6, day=3)])

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "-D", "230630", "230629"])
  def test_dates_order_args(self):
    args = parse_arguments()
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=29),
                                  UTCDateTime(year=2023, month=6, day=30)])

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "-D", "230629", "230632"])
  def test_dates_f_value_args(self):
    with self.assertRaises(SystemExit) as cm:
      parse_arguments()
    self.assertEqual(cm.exception.code, 2)

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "-G", BEG_DATE_STR])
  def test_group_args(self):
    args = parse_arguments()
    self.assertEqual(args.groups, [BEG_DATE_STR])

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-G", BEG_DATE_STR, NETWORK_STR])
  def test_groups_args(self):
    args = parse_arguments()
    self.assertEqual(args.groups, [BEG_DATE_STR, NETWORK_STR])

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-K",
                        Path(DATA_PATH, "file.key").__str__()])
  def test_key_args(self):
    args = parse_arguments()
    self.assertEqual(args.key, Path(DATA_PATH, "file.key"))

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "-M", PHASENET_STR])
  def test_model_args(self):
    args = parse_arguments()
    self.assertEqual(args.models, [PHASENET_STR])

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "-M", PHASENET_STR,
                                    EQTRANSFORMER_STR])
  def test_models_args(self):
    args = parse_arguments()
    self.assertEqual(args.models, [PHASENET_STR, EQTRANSFORMER_STR])

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "-D", "230601", "230731"])
  def test_range_args(self):
    args = parse_arguments()
    self.assertEqual(args.dates, [UTCDateTime(year=2023, month=6, day=1),
                                  UTCDateTime(year=2023, month=7, day=31)])

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "-T"])
  def test_train_args(self):
    args = parse_arguments()
    self.assertEqual(args.train, True)

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "-v"])
  def test_verbose_args(self):
    args = parse_arguments()
    self.assertEqual(args.verbose, True)

  @unittest.mock.patch("sys.argv", ["AdriaArray.py", "-W", INSTANCE_STR])
  def test_weight_args(self):
    args = parse_arguments()
    self.assertEqual(args.weights, [INSTANCE_STR])

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-W", INSTANCE_STR, ORIGINAL_STR])
  def test_weights_args(self):
    args = parse_arguments()
    self.assertEqual(args.weights, [INSTANCE_STR, ORIGINAL_STR])

class TestWaveformTable(unittest.TestCase):
  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-v", "-d", TEST_PATH.__str__()])
  def test_non_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [12]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-N', "OX", "-d", TEST_PATH.__str__(),
                        "-v"])
  def test_network_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [12]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-N', "OX", "ST", "-v", "-d",
                        TEST_PATH.__str__()])
  def test_networks_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [12]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-S', "LUSI", "-v", "-d",
                        TEST_PATH.__str__()])
  def test_station_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [12]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-S', "LUSI", "PANI", "-v", "-d",
                        TEST_PATH.__str__()])
  def test_stations_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [12]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-C', "EHZ", "-v", "-d",
                        TEST_PATH.__str__()])
  def test_channel_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [4]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-C', "HHZ", "HHN", "-v", "-d",
                        TEST_PATH.__str__()])
  def test_channels_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [8]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-N', "SI", "ST", '-S', "MAGA",
                        "LUSI", "-d", TEST_PATH.__str__(), "-v"])
  def test_networks_stations_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [12]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-N', "SI", "ST", '-C', "HHN", "HHZ",
                        "-d", TEST_PATH.__str__(), "-v"])
  def test_networks_channels_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [8]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-S', "MAGA", "LUSI", '-C', "HHN",
                        "HHZ", "-d", TEST_PATH.__str__(), "-v"])
  def test_stations_channels_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [8]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-S', "MAGA", "LUSI", '-C', "HHN",
                        "HHZ", '-D', "230605", "230606", "-v", "-d",
                        TEST_PATH.__str__()])
  def test_stations_channels_dates_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [8]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", '-S', "MAGA", "LUSI", '-C', "HHN",
                        "HHZ", '-D', "230605", "230606", "-v", "-d",
                        TEST_PATH.__str__()])
  def test_download_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    SIZE = [8]*len(WAVEFORMS_DATA)
    for (_, trace_files), size in zip(WAVEFORMS_DATA, SIZE):
      self.assertEqual(trace_files.size, size)

class TestReadTraces(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    prc_dir = Path(DATA_PATH, PRC_STR)
    if prc_dir.exists():
      for f in prc_dir.iterdir():
        if f.is_file():
          f.unlink()

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-G", BEG_DATE_STR, "-v", "-d",
                        TEST_PATH.__str__()])
  def test_group_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    for _, dataset_name in list(itertools.product(args.models, args.weights)):
      for _, trace_files in WAVEFORMS_DATA:
        stream = read_traces(trace_files, dataset_name, args)
        for tr in stream:
          self.assertGreaterEqual(tr.stats.starttime,
                                  UTCDateTime(tr.stats.starttime.date))
          self.assertLessEqual(tr.stats.endtime,
                               UTCDateTime(tr.stats.starttime.date) + ONE_DAY)

  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-v", "-d", TEST_PATH.__str__()])
  def test_groups_args(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    for _, dataset_name in list(itertools.product(args.models, args.weights)):
      for _, trace_files in WAVEFORMS_DATA:
        stream = read_traces(trace_files, dataset_name, args)
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
  @unittest.mock.patch("sys.argv",
                       ["AdriaArray.py", "-v", "-d", TEST_PATH.__str__(),
                        "-G", BEG_DATE_STR, NETWORK_STR, STATION_STR, "-M",
                        PHASENET_STR, EQTRANSFORMER_STR])
  def test_classification(self):
    args = parse_arguments()
    WAVEFORMS_DATA = waveform_table(args)
    for model_name, dataset_name in list(itertools.product(args.models,
                                                           args.weights)):
      MODEL = get_model(model_name, dataset_name)
      if MODEL is None: continue
      for categories, trace_files in WAVEFORMS_DATA:
        output = classify_stream(categories, trace_files, model_name,
                                 dataset_name, MODEL, args)
        print(output)

if __name__ == "__main__":
  unittest.main()
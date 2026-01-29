import os
import sys
import pandas as pd
import unittest
from pathlib import Path
THIS_DIR = os.path.dirname(__file__)
sys.path.append(THIS_DIR + "/../src")

from datetime import datetime

import ogsconstants as OGS_C

class TestOGSConstants(unittest.TestCase):
  def test_inventory(self):
    stations = Path(os.path.abspath(THIS_DIR + "/../data/station"))
    inventory = OGS_C.inventory(stations)
    expected = pd.read_parquet(
      Path(os.path.abspath(THIS_DIR + "/../data/OGSCatalog/station.parquet")))
    pd.testing.assert_frame_equal(
      inventory.reset_index(drop=True),
      expected.reset_index(drop=True))

  def test_waveforms(self):
    start = datetime.strptime("240101", OGS_C.YYMMDD_FMT)
    end = datetime.strptime("241231", OGS_C.YYMMDD_FMT)
    waveforms_path = Path(os.path.abspath(THIS_DIR + "/../data/waveforms"))
    waveforms = OGS_C.waveforms(waveforms_path, start, end)
    expected = pd.read_parquet(
      Path(os.path.abspath(THIS_DIR + "/../data/OGSCatalog/waveforms.parquet")))
    pd.testing.assert_frame_equal(
      waveforms.reset_index(drop=True),
      expected.reset_index(drop=True))

if __name__ == "__main__":
  unittest.main()
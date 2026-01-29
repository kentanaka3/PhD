import os
import sys
import pandas as pd
import unittest.mock
from pathlib import Path
THIS_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(THIS_DIR + "/../src"))

from datetime import datetime

import ogsconstants as OGS_C
from ogspun import DataFilePUN, parse_arguments

DATA_DIR = Path(os.path.abspath(THIS_DIR + "/../data"))
DATA_FILE = "onlyEQ-2024.pun"
class TestOGSPUN(unittest.TestCase):
  @unittest.mock.patch("sys.argv", [
    "ogspun.py", "-D", "20240320", "20240620",
    "-f", str(DATA_DIR / "manual" / "onlyEQ-2024.pun"),
    "-v"
  ])
  def test_args(self):
    args = parse_arguments()
    self.assertEqual(args.file, [Path(DATA_DIR / "manual" / "onlyEQ-2024.pun")])
    self.assertEqual(args.dates[0],
                     datetime.strptime("20240320", OGS_C.YYYYMMDD_FMT))
    self.assertEqual(args.dates[1],
                     datetime.strptime("20240620", OGS_C.YYYYMMDD_FMT))
    self.assertTrue(args.verbose)

  def test_read(self):
    print()
    datafile = DataFilePUN(
      DATA_DIR / "manual" / DATA_FILE,
      start=datetime.strptime("20240101", OGS_C.YYYYMMDD_FMT),
      end=datetime.strptime("20241231", OGS_C.YYYYMMDD_FMT),
      verbose=True)
    datafile.read()
    filepath = DATA_DIR / "OGSCatalog" / (DATA_FILE + ".parquet")
    # datafile.EVENTS.to_parquet(filepath)
    expected = pd.read_parquet(filepath)
    pd.testing.assert_frame_equal(
      datafile.EVENTS.reset_index(drop=True),
      expected.reset_index(drop=True)
    )

if __name__ == "__main__":
  unittest.main()
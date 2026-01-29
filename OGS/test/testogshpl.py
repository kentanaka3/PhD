import os
import sys
import pandas as pd
import unittest.mock
from pathlib import Path
THIS_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(THIS_DIR + "/../src"))

from datetime import datetime

import ogsconstants as OGS_C
from ogshpl import DataFileHPL, parse_arguments

DATA_DIR = Path(os.path.abspath(THIS_DIR + "/../data"))
DATA_FILES = [
  "onlyEQ-2024.hpl",
  "onlyEQ_NLL1D-2024.hpl"
]
class TestOGSHPL(unittest.TestCase):
  @unittest.mock.patch("sys.argv", [
    "ogshpl.py", "-D", "20240320", "20240620",
    "-f", str(DATA_DIR / "manual" / "onlyEQ-2024.hpl"),
    "-v"
  ])
  def test_args(self):
    args = parse_arguments()
    self.assertEqual(args.file, [Path(DATA_DIR / "manual" / "onlyEQ-2024.hpl")])
    self.assertEqual(args.dates[0],
                     datetime.strptime("20240320", OGS_C.YYYYMMDD_FMT))
    self.assertEqual(args.dates[1],
                     datetime.strptime("20240620", OGS_C.YYYYMMDD_FMT))
    self.assertTrue(args.verbose)

  def test_read(self):
    print()
    for file in DATA_FILES:
      datafile = DataFileHPL(
        DATA_DIR / "manual" / file,
        start=datetime.strptime("20240101", OGS_C.YYYYMMDD_FMT),
        end=datetime.strptime("20241231", OGS_C.YYYYMMDD_FMT),
        verbose=True)
      datafile.read()
      filepath = DATA_DIR / "OGSCatalog" / (file + ".parquet")
      # datafile.EVENTS.to_parquet(filepath)
      expected = pd.read_parquet(filepath)
      pd.testing.assert_frame_equal(
        datafile.EVENTS.reset_index(drop=True),
        expected.reset_index(drop=True))

if __name__ == "__main__":
  unittest.main()
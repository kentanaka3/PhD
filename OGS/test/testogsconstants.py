import os
import sys
import pandas as pd
import unittest
from pathlib import Path
THIS_DIR = os.path.dirname(__file__)
sys.path.append(THIS_DIR + "/../src")

from datetime import datetime

import ogsconstants as OGS_C

if __name__ == "__main__":
  unittest.main()
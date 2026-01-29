import os
import sys
import pandas as pd
import unittest.mock
from pathlib import Path
THIS_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(THIS_DIR + "/../src"))

from datetime import datetime

import ogsconstants as OGS_C
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from pathlib import Path
# Set the project folder
PRJ_PATH = Path(os.path.dirname(__file__)).parent
INC_PATH = os.path.join(PRJ_PATH, "inc")
IMG_PATH = os.path.join(PRJ_PATH, "img")
DATA_PATH = os.path.join(PRJ_PATH, "data")
import sys
# Add to path
if INC_PATH not in sys.path: sys.path.append(INC_PATH)
import numpy as np
import pandas as pd

import initializer as ini
import analyzer as aly
import parser as prs
from constants import *
import matplotlib.pyplot as plt

args = ini.parse_arguments()
aly.plot_cluster(ini.classified_loader(args), ini.associated_loader(args),
                 args)
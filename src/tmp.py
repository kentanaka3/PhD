import numpy as np
import matplotlib.pyplot as plt
from constants import *
import parser as prs
import analyzer as aly
import initializer as ini
import pandas as pd
import sys
from pathlib import Path
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Set the project folder
PRJ_PATH = Path(os.path.dirname(__file__)).parent
INC_PATH = os.path.join(PRJ_PATH, "inc")
IMG_PATH = os.path.join(PRJ_PATH, "img")
DATA_PATH = os.path.join(PRJ_PATH, "data")
# Add to path
if INC_PATH not in sys.path:
    sys.path.append(INC_PATH)


args = ini.parse_arguments()
SOURCE, DETECT = ini.true_loader(args)

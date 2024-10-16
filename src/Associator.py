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
import obspy
import argparse

from constants import *
import Picker as Pkr

def main(args : argparse.Namespace) -> None:
  STATIONS = obspy.read_inventory(Path(DATA_PATH, STATION_STR))
  return

if __name__ == "__main__": main(Pkr.parse_arguments())
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gamma.utils import association

from constants import *
import Picker as Pkr

def main(args : argparse.Namespace) -> None:
  global DATA_PATH
  DATA_PATH = Path(args.directory).parent
  MODELS, WAVEFORMS_DATA = Pkr.set_up(args)
  stations = args.stations if (args.station is not None and
                               args.station != ALL_WILDCHAR_STR) else \
             (WAVEFORMS_DATA[NETWORK_STR] + PERIOD_STR + \
              WAVEFORMS_DATA[STATION_STR]).unique()
  INVENTORY = obspy.Inventory()
  for station in stations:
    INVENTORY.extend(obspy.read_inventory(Path(DATA_PATH, STATION_STR,
                                               station + XML_EXT)))
  if args.verbose:
    print(INVENTORY)
    #INVENTORY.plot(projection="local")
  THRESHOLDS = [round(t, 2) for t in np.linspace(0.2, 0.9, 8)]
  PRED = Pkr.load_data(args)
  for (model, weight), dataframe in PRED.groupby([MODEL_STR, WEIGHT_STR]):
    if args.verbose: print(f"Model: {model}, Weight: {weight}")
    for threshold in THRESHOLDS:
      PREDICTIONS = association(dataframe, threshold, INVENTORY)
  return

if __name__ == "__main__": main(Pkr.parse_arguments())
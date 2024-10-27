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
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gamma.utils import association

from constants import *
import Picker as Pkr

def station_graph(inventory : obspy.Inventory) -> None:
  x = [station.longitude for network in inventory for station in network]
  y = [station.latitude for network in inventory for station in network]
  station = [station.code for network in inventory for station in network]
  P = np.c_[x, y] * 1e-3
  TRI = sp.spatial.Delaunay(P)
  # TODO: Either replicate the inventory plot method or adapt it to plot
  #       the Delaunay triangulation
  fig = inventory.plot(projection="local", show=False, method="cartopy")
  ax = fig.axes[0]
  plt.triplot(P[:,0], P[:,1], TRI.simplices.copy(), color='r',
              linestyle='-', lw=2)
  for i, txt in enumerate(station):
    ax.annotate(txt, (P[i, 0], P[i, 1]), color='k', fontweight='bold')
  plt.tight_layout()
  IMG_FILE = Path(IMG_PATH, STATION_STR + PNG_EXT)
  plt.savefig(IMG_FILE)
  return

def associate_events(PRED : pd.DataFrame, STATION : pd.DataFrame,
                     model_name : str, dataset_name : str, threshold : float) -> None:
  if PRED.empty: return
  PRED[ID_STR] = PRED[NETWORK_STR] + PERIOD_STR + PRED[STATION_STR]
  PRED.rename(columns={PHASE_STR : TYPE_STR}, inplace=True)
  print(PRED)
  catalog, assignment = association(PRED, STATION, config=ASSOCIATION_CONFIG,
                                    method=ASSOCIATION_CONFIG[METHOD_STR])
  print(catalog)
  print(assignment)
  return

def main(args : argparse.Namespace) -> None:
  global DATA_PATH
  DATA_PATH = Path(args.directory).parent
  WAVEFORMS_DATA = Pkr.waveform_table(args)
  stations = (WAVEFORMS_DATA[NETWORK_STR] + PERIOD_STR + \
              WAVEFORMS_DATA[STATION_STR]).unique()
  INVENTORY = obspy.Inventory()
  for station in stations:
    INVENTORY.extend(obspy.read_inventory(Path(DATA_PATH, STATION_STR,
                                               station + XML_EXT)))
  HEADER = [ID_STR, X_COORD_STR, Y_COORD_STR, Z_COORD_STR]
  STATION = {(PERIOD_STR.join([network.code, station.code]),
              station.longitude, station.latitude, station.elevation)
             for network in INVENTORY for station in network}
  STATION = pd.DataFrame(STATION, columns=HEADER)
  print(STATION)
  # if args.verbose: station_graph(INVENTORY)
  THRESHOLDS = [round(t, 2) for t in np.linspace(0.2, 0.9, 8)]
  PRED = Pkr.load_data(args)
  for threshold in THRESHOLDS:
    for (model, dataset), dataframe in PRED.groupby([MODEL_STR, WEIGHT_STR]):
      dataframe = dataframe[dataframe[PROBABILITY_STR] >= threshold]\
                    .reset_index(drop=True)
      associate_events(dataframe, STATION, model, dataset, threshold)
  return

if __name__ == "__main__": main(Pkr.parse_arguments())
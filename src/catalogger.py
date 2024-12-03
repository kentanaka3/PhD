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
import json
import obspy
import argparse
import scipy as sp
import numpy as np
import pandas as pd
from pyproj import Proj
import matplotlib.pyplot as plt
from gamma.utils import association, estimate_eps

from constants import *
import initializer as ini

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

class AssociateConfig:
  def __init__(self, INVENTORY : obspy.Inventory, center : tuple = None,
                     x_lim : tuple = None, y_lim : tuple = None,
                     z_lim : tuple = None, t_lim : tuple = None,
                     vel : tuple = None, ncpu : int = 32,
                     method : str = BAYES_GAUSS_MIX_MODEL_STR,
                     use_amplitude : bool = False, file = None) -> None:
    self.dims = [X_COORD_STR, Y_COORD_STR, Z_COORD_STR]
    self.station = pd.DataFrame({(
      PERIOD_STR.join([network.code, station.code]), station.longitude,
      station.latitude, station.elevation * (-1e-3))
      for network in INVENTORY for station in network},
      columns=[ID_STR] + self.dims)
    if file is not None:
      # TODO: Implement the file reading method
      pass
    self.x_min = self.station[X_COORD_STR].min()
    self.x_mid = self.station[X_COORD_STR].median()
    self.x_max = self.station[X_COORD_STR].max()

    self.y_min = self.station[Y_COORD_STR].min()
    self.y_mid = self.station[Y_COORD_STR].median()
    self.y_max = self.station[Y_COORD_STR].max()
    print(self.x_min, self.x_mid, self.x_max)
    print(self.y_min, self.y_mid, self.y_max)
    self.center = (self.x_mid, self.y_mid)
    if center: self.center = center
    self.proj = Proj(f"+proj=sterea +lon_0={self.center[0]} "
                     f"+lat_0={self.center[1]} +units=km")
    # from deg to km
    self.station[[X_COORD_STR, Y_COORD_STR]] = \
      self.station.apply(lambda s:
        pd.Series(self.proj(latitude=s[Y_COORD_STR],
                            longitude=s[X_COORD_STR])), axis=1)

    self.x_lim = (2 * self.x_min - self.x_mid, 2 * self.x_max - self.x_mid)
    self.x_lim = self.proj(latitude=[self.center[1]] * 2,
                           longitude=self.x_lim)[0]
    if x_lim: self.x_lim = x_lim

    self.y_lim = (2 * self.y_min - self.y_mid, 2 * self.y_max - self.y_mid)
    self.y_lim = self.proj(latitude=self.y_lim,
                           longitude=[self.center[0]] * 2)[1]
    if y_lim: self.y_lim = y_lim

    self.z_lim = (0, 30)
    if z_lim: self.z_lim = z_lim
    self.t_lim = (None, None)
    if t_lim: self.t_lim = t_lim

    self.bfgs_bounds = (
      (self.x_lim[0] - 1, self.x_lim[1] + 1),
      (self.y_lim[0] - 1, self.y_lim[1] + 1),
      (self.z_lim[0], self.z_lim[1] + 1),
      (self.t_lim[0], self.t_lim[1])
    )

    self.method = method
    self.oversample_factor = 5 if method == BAYES_GAUSS_MIX_MODEL_STR else 1
    self.use_amplitude = use_amplitude

    self.vel = {
      PWAVE: 5.85,
      SWAVE: 5.85 / 1.78
    }
    if vel: self.vel = {key : value for key, value in zip([PWAVE, SWAVE], vel)}

    # DBSCAN parameters
    self.use_dbscan = False
    self.dbscan_eps = None
    self.dbscan_min_samples = 0
    self.dbscan(3)

    self.min_picks_per_eq = 5
    self.min_p_picks_per_eq = 3
    self.min_s_picks_per_eq = 2

    self.max_sigma11 = 1.0
    self.max_sigma22 = 1.0
    self.max_sigma12 = 1.0

    self.ncpu = ncpu
    return

  def __repr__(self) -> str:
    return {
      "dims": self.dims,
      "use_dbscan": self.use_dbscan,
      "use_amplitude": self.use_amplitude,
      X_COORD_STR: self.x_lim,
      Y_COORD_STR: self.y_lim,
      Z_COORD_STR: self.z_lim,
      "method": self.method,
      "oversample_factor": self.oversample_factor,
      "vel": {
        "p" : self.vel[PWAVE],
        "s" : self.vel[SWAVE]
      },
      "dbscan_eps": self.dbscan_eps,
      "dbscan_min_samples": self.dbscan_min_samples,
      "min_picks_per_eq": self.min_picks_per_eq,
      "min_p_picks_per_eq": self.min_p_picks_per_eq,
      "min_s_picks_per_eq": self.min_s_picks_per_eq,
      "max_sigma11": self.max_sigma11,
      "bfgs_bounds": self.bfgs_bounds
    }

  def __str__(self) -> str:
    return f"AssociateConfig({self.__repr__()})"

  def dbscan(self, min_samples = 3) -> None:
    self.use_dbscan = True
    self.dbscan_min_samples = min_samples
    self.dbscan_eps = estimate_eps(self.station, self.vel[PWAVE])
    return

def associate_events(PRED : pd.DataFrame, config : AssociateConfig,
                     model_name : str, dataset_name : str,
                     threshold : float) -> None:
  if PRED.empty: return
  PRED[ID_STR] = PRED[NETWORK_STR] + PERIOD_STR + PRED[STATION_STR]
  PRED.rename(columns={PHASE_STR : TYPE_STR}, inplace=True)
  PRED[TIMESTAMP_STR] = PRED[TIMESTAMP_STR].apply(lambda x: x.datetime)
  catalog, assignment = association(PRED, config.station,
                                    config=config.__repr__(),
                                    method=config.method)
  print(catalog)
  print(assignment)
  return

def main(args : argparse.Namespace) -> None:
  global DATA_PATH
  DATA_PATH = Path(args.directory).parent
  WAVEFORMS_DATA = ini.waveform_table(args)
  stations = (WAVEFORMS_DATA[NETWORK_STR] + PERIOD_STR + \
              WAVEFORMS_DATA[STATION_STR]).unique()
  INVENTORY = obspy.Inventory()
  for station in stations:
    station_file = Path(DATA_PATH, STATION_STR, station + XML_EXT)
    if not station_file.exists():
      print(f"WARNING: Station file {station_file} does not exist.")
      continue
    INVENTORY.extend(obspy.read_inventory(station_file))
  CONFIG = AssociateConfig(INVENTORY, file=args.file)
  print(CONFIG)
  # if args.verbose: station_graph(INVENTORY)
  THRESHOLDS = [round(t, 2) for t in np.linspace(0.2, 0.9, 8)]
  PRED = ini.classified_header(args)
  for threshold in THRESHOLDS:
    for (model, dataset), dataframe in PRED.groupby([MODEL_STR, WEIGHT_STR]):
      dataframe = dataframe[dataframe[PROBABILITY_STR] >= threshold]\
                    .reset_index(drop=True)
      associate_events(dataframe, CONFIG, model, dataset, threshold)
  return

if __name__ == "__main__": main(ini.parse_arguments())
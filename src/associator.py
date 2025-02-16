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
import copy
import obspy
import argparse
import scipy as sp
import numpy as np
import pandas as pd
from pyproj import Proj
import matplotlib.pyplot as plt
from copy import deepcopy as dcpy
from datetime import datetime as dt
from datetime import timedelta as td
PROGRAM_NAME = "GaMMA"
from gamma.utils import association, estimate_eps

from constants import *
import initializer as ini

MPI_RANK = 0
MPI_SIZE = 1
MPI_COMM = None

DATES = None

class AssociateConfig:
  def __init__(self, INVENTORY : obspy.Inventory, center : tuple = None,
                     x_lim : tuple = None, y_lim : tuple = None,
                     z_lim : tuple = None, t_lim : tuple = None,
                     vel : tuple = None, ncpu : int = 32,
                     method : str = BAYES_GAUSS_MIX_MODEL_STR,
                     use_amplitude : bool = False, file : Path = None):
    self.dims = [X_COORD_STR, Y_COORD_STR, Z_COORD_STR]
    self.station = pd.DataFrame({(
      PERIOD_STR.join([network.code, station.code]), station.longitude,
      station.latitude, station.elevation * (-1e-3))
      for network in INVENTORY for station in network},
      columns=[ID_STR] + self.dims)
    if file is not None:
      if len(file) > 1: raise NotImplementedError("Multiple station files.")
      # TODO: Implement the file reading method
      pass
    self.x_min = self.station[X_COORD_STR].min()
    self.x_mid = self.station[X_COORD_STR].median()
    self.x_max = self.station[X_COORD_STR].max()

    self.y_min = self.station[Y_COORD_STR].min()
    self.y_mid = self.station[Y_COORD_STR].median()
    self.y_max = self.station[Y_COORD_STR].max()
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

    self.max_sigma11 = 1.0 # TODO: Explore 2.0
    self.max_sigma22 = 1.0
    self.max_sigma12 = 1.0

    self.ncpu = ncpu

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

def dates2associate(args : argparse.Namespace) -> tuple:
  global MPI_RANK, MPI_SIZE, MPI_COMM
  start, end = args.dates
  if MPI_SIZE == 1: return start, end
  end += ONE_DAY
  DAYS = td(seconds=end - start).days
  N = td(days=(DAYS // MPI_SIZE))
  R = td(days=(DAYS % MPI_SIZE))
  if MPI_RANK < R.days:
    N += ONE_DAY
    R = td(0)
  start += N * MPI_RANK + R
  end = start + N - ONE_DAY
  if args.verbose: print(f"Rank {MPI_RANK}: {start} - {end}")
  return start, end

def associate_events(PRED : pd.DataFrame, config : AssociateConfig,
                     args : argparse.Namespace) -> None:
  global DATES
  if args.verbose: print("Associating events...")
  PRED[ID_STR] = PRED[NETWORK_STR] + PERIOD_STR + PRED[STATION_STR]
  PRED.rename(columns={PHASE_STR : TYPE_STR}, inplace=True)
  PRED[TIMESTAMP_STR] = PRED[TIMESTAMP_STR].apply(lambda x: x.datetime)
  start, end = dates2associate(args)
  if DATES is None:
    DATES = [start.datetime]
    while DATES[-1] <= end.datetime: DATES.append(DATES[-1] + ONE_DAY)
  SOURCE = pd.DataFrame(columns=HEADER_ASCT)
  DETECT = pd.DataFrame(columns=HEADER_PRED)
  for (model, weight), PRE in PRED.groupby([MODEL_STR, WEIGHT_STR]):
    id : int = 0
    for start, end in zip(DATES[:-1], DATES[1:]):
      # For each day in the dataset we will associate the events
      PR = PRE[PRE[TIMESTAMP_STR].between(start, end, inclusive='left')]
      if args.force: PR = PR[
        ((PR[TYPE_STR] == PWAVE) & (PR[PROBABILITY_STR] >= args.pwave)) |
        ((PR[TYPE_STR] == SWAVE) & (PR[PROBABILITY_STR] >= args.swave))]
      if PR.empty: continue
      catalog, assignment = association(PR, config.station,
                                        config=config.__repr__(),
                                        method=config.method)
      if not (len(catalog) or len(assignment)): continue
      SRC = list()
      PKS = pd.DataFrame(columns=HEADER_PRED)
      for event in catalog:
        PICKS = pd.DataFrame(
          [PR.loc[row] for row, idx, _ in assignment
                        if idx == event["event_index"]]).reset_index(drop=True)
        PICKS[MODEL_STR] = model
        PICKS[WEIGHT_STR] = weight
        th = float("{:0.1}".format(PICKS[PROBABILITY_STR].min()))
        PICKS[THRESHOLD_STR] = th
        PICKS[ID_STR] = id
        PICKS.rename(columns={TYPE_STR : PHASE_STR}, inplace=True)
        PICKS.sort_values(TIMESTAMP_STR, inplace=True)
        PKS = pd.concat([PKS, PICKS], ignore_index=True) if not PKS.empty \
                                                       else PICKS
        SRC.append([model, weight, th, id, dt.fromisoformat(event['time']),
                    *config.proj(event[X_COORD_STR], event[Y_COORD_STR],
                                 inverse=True), event[Z_COORD_STR],
                    event[MAGNITUDE_STR], len(PICKS.index), *([None] * 6),
                    PROGRAM_NAME])
        id += 1
      SRC = pd.DataFrame(SRC, columns=HEADER_ASCT).sort_values(TIMESTAMP_STR)
      SRC = SRC.reset_index(drop=True)
      L = len(SOURCE)
      rpl = {int(a): int(b) for a, b in zip(dcpy(SRC[ID_STR].to_list()),
                                            range(L, L + len(SRC)))}.get
      SRC[ID_STR] = SRC[ID_STR].apply(rpl)
      PKS[ID_STR] = PKS[ID_STR].apply(rpl)
      SOURCE = pd.concat([SOURCE, SRC], ignore_index=True) \
                 if not SOURCE.empty else SRC
      DETECT = pd.concat([DETECT, PKS], ignore_index=True) \
                 if not DETECT.empty else PKS
  FILEPATH = Path(DATA_PATH, ("D" if args.denoiser else EMPTY_STR) +
                  AST_STR + CSV_EXT)
  SOURCE.sort_values(SORT_HIERARCHY_PRED, inplace=True)
  SOURCE.to_csv(FILEPATH, index=False)
  FILEPATH = Path(DATA_PATH, ("D" if args.denoiser else EMPTY_STR) +
                  ASCT_STR + CSV_EXT)
  DETECT.sort_values(SORT_HIERARCHY_PRED, inplace=True)
  DETECT.to_csv(FILEPATH, index=False)
  for (model, weight, network, station), dtfrm in \
    DETECT.groupby([MODEL_STR, WEIGHT_STR, NETWORK_STR, STATION_STR]):
    for start, end in zip(DATES[:-1], DATES[1:]):
      df = dtfrm[dtfrm[TIMESTAMP_STR].between(start, end, inclusive='left')]
      if df.empty: continue
      start = start.strftime("%y%m%d")
      FILEPATH = Path(DATA_PATH, AST_STR, start, network, station,
                      ("D_" if args.denoiser else EMPTY_STR) +
                      UNDERSCORE_STR.join([start, network, station, model,
                                           weight]) + CSV_EXT)
      FILEPATH.parent.mkdir(parents=True, exist_ok=True)
      df.to_csv(FILEPATH, index=False)
  return SOURCE, DETECT

def set_up(args : argparse.Namespace) -> AssociateConfig:
  """
    Set up the environment for the associator pipeline based on the available
    computational resources.

  input:
    - args    (argparse.Namespace) : command line arguments

  output:
    - CONFIG  (AssociateConfig) : configuration object for the associator

  errors:
    - FileNotFoundError : if the station file does not exist

  notes:

  """
  global DATA_PATH
  DATA_PATH = args.directory.parent
  # global MPI_SIZE, MPI_RANK, MPI_COMM
  # MPI_COMM = MPI.COMM_WORLD
  # MPI_SIZE = MPI_COMM.Get_size()
  # MPI_RANK = MPI_COMM.Get_rank()
  # CONFIG = None
  # if MPI_RANK == 0:
  # TODO: Implement the dask client method
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
  if args.verbose:
    INVENTORY.plot(projection="local", show=False, method="cartopy", size=5,
                   water_fill_color="lightblue", color_per_network=True,
                   outfile=Path(IMG_PATH, STATION_STR + PNG_EXT))
  CONFIG = AssociateConfig(INVENTORY, file=args.file)
  # CONFIG = MPI_COMM.bcast(CONFIG, root=0)
  return CONFIG

def main(args : argparse.Namespace) -> None:
  global DATA_PATH
  DATA_PATH = args.directory.parent
  CONFIG = set_up(args)
  PRED = ini.classified_loader(args)
  # TODO: Implement the pyocto method
  if args.pyocto: pass
  DATA, PRED = associate_events(copy.deepcopy(PRED), CONFIG, args)
  return

if __name__ == "__main__": main(ini.parse_arguments())
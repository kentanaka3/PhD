from gamma.utils import association, estimate_eps
from concurrent.futures import ThreadPoolExecutor
from obspy.core.utcdatetime import UTCDateTime
from datetime import timedelta as td
from datetime import datetime as dt
from copy import deepcopy as dcpy
from pyproj import Proj
import pandas as pd
import argparse
import obspy
import sys
from pathlib import Path
import os
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Set the project folder
PRJ_PATH = Path(os.path.dirname(__file__)).parent
INC_PATH = os.path.join(PRJ_PATH, "inc")
IMG_PATH = os.path.join(PRJ_PATH, "img")
DATA_PATH = os.path.join(PRJ_PATH, "data")
# Add to path
if INC_PATH in sys.path:
  import initializer as ini
  from resources.constants import *
else:
  sys.path.append(INC_PATH)
  import initializer as ini
  from resources.constants import *

MPI_RANK = 0
MPI_SIZE = 1
MPI_COMM = None

DATES = None

id_str = "id"
prob_str = "prob"
timestamp_str = "timestamp"


class AssociateConfig:
  def __init__(self, INVENTORY: obspy.Inventory, file: Path,
               center: tuple[float, float] = (12, 46),
               x_lim: tuple[float, float] = (-300, 400),
               y_lim: tuple[float, float] = (-350, 250),
               z_lim: tuple[float, float] = (0, 40),
               t_lim: tuple[float, float] = (None, None),
               vel: tuple = None, ncpu: int = 32,
               method: str = BAYES_GAUSS_MIX_MODEL_STR,
               use_amplitude: bool = False):
    self.dims = [X_COORD_STR, Y_COORD_STR, Z_COORD_STR]
    self.station = pd.DataFrame({(
        PERIOD_STR.join([network.code, station.code]), station.longitude,
        station.latitude, station.elevation * (-1e-3))
        for network in INVENTORY for station in network},
        columns=[id_str] + self.dims)
    if file is not None:
      if len(file) > 1:
        raise NotImplementedError("Multiple station files.")
      # TODO: Implement the file reading method
      pass
    # TODO: Hard code center for projection
    self.x_lim = x_lim
    self.y_lim = y_lim
    self.x_mid, self.y_mid = center
    self.z_lim = z_lim
    self.t_lim = t_lim
    self.proj = Proj(OGS_PROJECTION.format(lon=self.x_mid, lat=self.y_mid))
    # from deg to km
    self.station[[X_COORD_STR, Y_COORD_STR]] = \
        self.station.apply(lambda s: pd.Series(self.proj(
            latitude=s[Y_COORD_STR], longitude=s[X_COORD_STR])), axis=1)

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
    if vel:
      self.vel = {key: value for key, value in zip([PWAVE, SWAVE], vel)}

    # DBSCAN parameters
    self.use_dbscan = False
    self.dbscan_eps = None
    self.dbscan_min_samples = 0
    self.dbscan(3)

    self.min_picks_per_eq = 5
    self.min_p_picks_per_eq = 3
    self.min_s_picks_per_eq = 2

    self.max_sigma11 = 1.0  # TODO: Explore 2.0
    self.max_sigma22 = 1.0
    self.max_sigma12 = 1.0

    self.ncpu = ncpu

  def __repr__(self):
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
        "p": self.vel[PWAVE],
        "s": self.vel[SWAVE]
      },
      "dbscan_eps": self.dbscan_eps,
      "dbscan_min_samples": self.dbscan_min_samples,
      "min_picks_per_eq": self.min_picks_per_eq,
      "min_p_picks_per_eq": self.min_p_picks_per_eq,
      "min_s_picks_per_eq": self.min_s_picks_per_eq,
      "covariance_prior_pre": [5.0, 5.0],
      "max_sigma11": self.max_sigma11,
      "bfgs_bounds": self.bfgs_bounds
    }

  def __str__(self) -> str:
    return f"AssociateConfig({self.__repr__()})"

  def dbscan(self, min_samples=3) -> None:
    self.use_dbscan = True
    self.dbscan_min_samples = min_samples
    self.dbscan_eps = estimate_eps(self.station, self.vel[PWAVE])
    return


def dates2associate(args: argparse.Namespace) -> tuple:
  global MPI_RANK, MPI_SIZE, MPI_COMM
  start, end = args.dates
  if MPI_SIZE == 1:
    return start, end
  end += ONE_DAY
  DAYS = td(seconds=end - start).days
  N = td(days=(DAYS // MPI_SIZE))
  R = td(days=(DAYS % MPI_SIZE))
  if MPI_RANK < R.days:
    N += ONE_DAY
    R = td(0)
  start += N * MPI_RANK + R
  end = start + N - ONE_DAY
  if args.verbose:
    print(f"Rank {MPI_RANK}: {start} - {end}")
  return start, end


def associate_events(PRED: pd.DataFrame, config: AssociateConfig,
                     args: argparse.Namespace) -> None:
  global DATES
  if args.verbose:
    print("Associating events")
  PRED[id_str] = PRED[NETWORK_STR] + PERIOD_STR + PRED[STATION_STR]
  PRED[TIME_STR] = PRED[TIME_STR].apply(lambda x: x.datetime)
  PRED.rename(columns={
      PHASE_STR: TYPE_STR,
      TIME_STR: timestamp_str,
      PROBABILITY_STR: prob_str}, inplace=True)
  SOURCE = pd.DataFrame(columns=HEADER_ASCT)
  DETECT = pd.DataFrame(columns=HEADER_PRED)
  ids = {model: {weight: 0 for weight in args.weights}
         for model in args.models}
  for (model, weight, date), PRE in PRED.groupby([
          MODEL_STR, WEIGHT_STR, ID_STR]):
    if not ((model in args.models) and (weight in args.weights)) or PRE.empty:
      continue
    print(f"Processing {date}")
    SOURCE_FOLDER = Path(DATA_PATH, AST_STR, "events", *date.split(DASH_STR))
    DETECT_FOLDER = Path(DATA_PATH, AST_STR, "assignments",
                         *date.split(DASH_STR))
    SOURCE_FOLDER.mkdir(parents=True, exist_ok=True)
    DETECT_FOLDER.mkdir(parents=True, exist_ok=True)
    FILEPATH = Path(SOURCE_FOLDER, "D_" if args.denoiser else EMPTY_STR +
                    UNDERSCORE_STR.join([model, weight]))
    if not args.force and FILEPATH.exists():
      SOURCE = pd.concat([SOURCE, pd.read_csv(FILEPATH)], ignore_index=True)\
          if not SOURCE.empty else pd.read_csv(FILEPATH)
      for network_path in DETECT_FOLDER.iterdir():
        if not network_path.is_dir():
          continue
        for station_path in network_path.iterdir():
          if not station_path.is_dir():
            continue
          fr = Path(station_path, "D_" if args.denoiser else EMPTY_STR +
                    UNDERSCORE_STR.join([
                        date, network_path.name, station_path.name, model,
                        weight]) + CSV_EXT)
          if fr.exists():
            print(f"Loading {fr}")
            DETECT = pd.concat([DETECT, pd.read_csv(fr)], ignore_index=True)\
                if not DETECT.empty else pd.read_csv(fr)
      continue
    # For each day in the dataset we will associate the events
    if args.force:
      PRE = PRE[
          ((PRE[TYPE_STR] == PWAVE) & (PRE[prob_str] >= args.pwave)) |
          ((PRE[TYPE_STR] == SWAVE) & (PRE[prob_str] >= args.swave))]
    if PRE.empty:
      continue
    if args.verbose:
      print(f"Saving {FILEPATH}")
    FILEPATH.touch()
    catalog, assignment = association(PRE, config.station,
                                      config=config.__repr__(),
                                      method=config.method)
    if not (len(catalog) or len(assignment)):
      FILEPATH.unlink()  # Remove empty file
      continue
    SRC = list()
    PKS = pd.DataFrame(columns=HEADER_PRED)
    for event in catalog:
      PICKS = pd.DataFrame([
          PRE.loc[row] for row, idx, _ in assignment
          if idx == event["event_index"]]).reset_index(drop=True)
      PICKS[MODEL_STR] = model
      PICKS[WEIGHT_STR] = weight
      th = float("{:0.1}".format(PICKS[prob_str].mean()))
      PICKS[THRESHOLD_STR] = th
      PICKS["index"] = ids[model][weight]
      PICKS.rename(columns={TYPE_STR: PHASE_STR}, inplace=True)
      PICKS.sort_values(timestamp_str, inplace=True)
      PKS = pd.concat([PKS, PICKS], ignore_index=True) if not PKS.empty \
          else PICKS
      SRC.append([model, weight, th, ids[model][weight],
                  dt.fromisoformat(event[TIME_STR]), *reversed(config.proj(
                      event[X_COORD_STR], event[Y_COORD_STR], inverse=True)),
                  event[Z_COORD_STR], event[MAGNITUDE_STR], len(PICKS.index),
                  *([None] * 6), GMMA_STR])
      ids[model][weight] += 1
    SRC = pd.DataFrame(SRC, columns=HEADER_ASCT).sort_values(
        TIME_STR).reset_index(drop=True)
    L = len(SOURCE)
    rpl = {int(a): int(b) for a, b in zip(dcpy(SRC["index"].to_list()),
                                          range(L, L + len(SRC)))}.get
    SRC["index"] = SRC["index"].apply(rpl)
    PKS["index"] = PKS["index"].apply(rpl)
    SRC.to_parquet(FILEPATH, index=False)
    SOURCE = pd.concat([SOURCE, SRC], ignore_index=True) \
        if not SOURCE.empty else SRC
    DETECT = pd.concat([DETECT, PKS], ignore_index=True) \
        if not DETECT.empty else PKS
    if PKS.empty:
      continue

    def fp(x) -> None:
      keys, dtfrm = x
      FILEPATH = Path(DETECT_FOLDER, *keys,
                      ("D_" if args.denoiser else EMPTY_STR) +
                      UNDERSCORE_STR.join([*keys, model, weight]) + CSV_EXT)
      FILEPATH.parent.mkdir(parents=True, exist_ok=True)
      dtfrm.to_csv(FILEPATH, index=False)
    with ThreadPoolExecutor() as executor:
      executor.map(fp, PKS.groupby([NETWORK_STR, STATION_STR]))
  FILEPATH = Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) +
                  SOURCE_STR + CSV_EXT)
  SOURCE.sort_values(SORT_HIERARCHY_PRED, inplace=True)
  SOURCE.to_csv(FILEPATH, index=False)
  FILEPATH = Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) +
                  DETECT_STR + CSV_EXT)
  DETECT.rename(columns={timestamp_str: TIME_STR}, inplace=True)
  DETECT.sort_values(SORT_HIERARCHY_PRED, inplace=True)
  DETECT.to_csv(FILEPATH, index=False)
  return SOURCE, DETECT


def set_up(args: argparse.Namespace) -> AssociateConfig:
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
  stations = (WAVEFORMS_DATA[NETWORK_STR] + PERIOD_STR +
              WAVEFORMS_DATA[STATION_STR]).unique()
  INVENTORY = obspy.Inventory()
  for station in stations:
    station_file = Path(DATA_PATH, STATION_STR, station + XML_EXT)
    if not station_file.exists():
      print(f"WARNING: Station file {station_file} does not exist.")
      continue
    INVENTORY.extend(obspy.read_inventory(station_file))
  CONFIG = AssociateConfig(INVENTORY, file=args.file)
  # CONFIG = MPI_COMM.bcast(CONFIG, root=0)
  return CONFIG


def main(args: argparse.Namespace) -> None:
  global DATA_PATH
  DATA_PATH = args.directory.parent
  CONFIG = set_up(args)
  PRED: pd.DataFrame = pd.DataFrame(columns=HEADER_PRED)
  FILEPATH = Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) +
                  CLSSFD_STR + CSV_EXT)
  if (not args.force and FILEPATH.exists() and
          ini.read_args(args, False) == ini.dump_args(args, True)):
    if args.verbose:
      print(f"Loading {FILEPATH}")
    PRED = ini.data_loader(FILEPATH)
    PRED[TIME_STR] = PRED[TIME_STR].apply(lambda x: UTCDateTime(x))
  else:
    PRED = ini.classified_loader(args)
  PRED = PRED[
      ((PRED[PHASE_STR] == PWAVE) & (PRED[PROBABILITY_STR] >= args.pwave)) |
      ((PRED[PHASE_STR] == SWAVE) & (PRED[PROBABILITY_STR] >= args.swave))]
  if args.verbose:
    PRED.to_csv(FILEPATH, index=False)
  if PRED.empty:
    print("No events to associate.")
    return
  # TODO: Implement the pyocto method
  if args.pyocto:
    pass
  SOURCE, DETECT = associate_events(PRED, CONFIG, args)
  return


if __name__ == "__main__":
  main(ini.parse_arguments())

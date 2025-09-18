import os
import argparse
import numpy as np
import obspy as op
import pandas as pd
from pathlib import Path
from obspy import UTCDateTime
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.path import Path as mplPath
from sklearn.metrics import ConfusionMatrixDisplay as ConfMtxDisp
import ogsconstants as OGS_C
import ogsplotter as OGS_P
import ogsbpgma as OGS_B

RES_DIR = Path(__file__).parent.parent.parent / "data" / "compare"
RES_DIR.mkdir(parents=True, exist_ok=True)
IMG_PATH = Path(__file__).parent.parent / "img"

def is_date(string: str) -> datetime:
  return datetime.strptime(string, OGS_C.YYMMDD_FMT)

class SortDatesAction(argparse.Action):
  def __call__(self, parser, namespace, values, option_string=None):
    setattr(namespace, self.dest, sorted(values)) # type: ignore

def parse_arguments():
  parser = argparse.ArgumentParser(description="Process some data files.")
  parser.add_argument('-B', "--base", type=Path, required=True,
                      help="Base directory for data files.")
  parser.add_argument(
    '-D', "--dates", required=False, metavar=OGS_C.DATE_STD, type=is_date,
  nargs=2, action=SortDatesAction,
  default=[datetime.strptime("240320", OGS_C.YYMMDD_FMT),
           datetime.strptime("240620", OGS_C.YYMMDD_FMT)],
  help="Specify the beginning and ending (inclusive) Gregorian date " \
        "(YYMMDD) range to work with.")
  parser.add_argument('-S', "--station", type=Path, required=False,
                      help="Path to the station directory.")
  parser.add_argument('-T', "--target", type=Path, required=True,
                      help="Target file for processed data.")
  parser.add_argument('-W', "--waveform", type=Path, required=False,
                      help="Path to the waveform directory.")
  parser.add_argument('-v', "--verbose", action='store_true',
                      help="Increase output verbosity.")
  parser.add_argument('-H', "--use-hungarian", action='store_true',
                      help="Use Hungarian algorithm for matching (requires SciPy).")
  parser.add_argument('--hungarian-min-weight', type=float, default=0.0,
                      help="Minimum edge weight to keep when using Hungarian matching.")
  return parser.parse_args()

class Catalog:
  MODULES = None # to be defined in subclasses
  def __init__(self, filepath: Path, args: argparse.Namespace):
    self.filepath : Path = filepath
    self.args = args
    self.RESULTS_PATH = list()

  def preload(self): pass # to be optionally defined in subclasses

  def load_(self, filepath: Path): pass # to be defined in subclasses

  def load(self): pass # to be optionally defined in subclasses

class Base(Catalog):
  """Base Catalog Class. Handles OGS: TXT, DAT, HPL, PUN formats.
  Loads data from CSV files in events and assignments folders. Filters data by 
  date range and geographical region. Saves filtered data to RESULTS_PATH. 
  Data is stored in self.groups dictionary. Keys are 'events' and 
  'assignments', Values are pandas DataFrames. Events DataFrame has columns: 
  timestamp, latitude, longitude, depth, magnitude, ML, ERH, ERZ, GROUPS
  Assignments DataFrame has columns: timestamp, station, phase, ERT, GROUPS
  GROUPS column is date extracted from timestamp. Filters events to only those 
  within OGS_POLY_REGION. Handles different file formats with subclasses.
  Each subclass implements load_ method to read specific format.
  MODULES dictionary maps file extensions to subclasses.
  The load_ method identifies file format and instantiates appropriate subclass.
  If no specific format found, searches subdirectories for known formats.
  """
  class DataSet:
    """Base DataSet Class. Handles loading and storing data from CSV files.
    Loads data from events and assignments folders. Filters data by date range
    and geographical region. Data is stored in self.groups dictionary. Keys are
    'events' and 'assignments', Values are pandas DataFrames. Events DataFrame 
    has columns: timestamp, latitude, longitude, depth, magnitude, ML, ERH, 
    ERZ, GROUPS. Assignments DataFrame has columns: timestamp, station, phase,
    ERT, GROUPS. GROUPS column is date extracted from timestamp. Filters events
    to only those within OGS_POLY_REGION. Handles different file formats with
    subclasses. Each subclass implements load_ method to read specific format.
    MODULES dictionary maps file extensions to subclasses. The load_ method
    identifies file format and instantiates appropriate subclass. If no specific
    format found, searches subdirectories for known formats.
    """
    def __init__(self, filepath: Path, args: argparse.Namespace):
      self.groups = {}
      self.args = args
      self.load_(filepath)

    def load_(self, filepath: Path):
      for csv_file in filepath.glob("events/*/*/*.csv"):
        date = datetime.strptime(OGS_C.DASH_STR.join(str(
          csv_file).split("/")[-3:])[:-4], OGS_C.DATE_FMT)
        if date < self.args.dates[0] or date > self.args.dates[1]: continue
        self.groups.setdefault("events", {})[date] = pd.read_csv(csv_file)
        if OGS_C.MAGNITUDE_L_STR not in self.groups["events"][date].columns:
          self.groups["events"][date][OGS_C.MAGNITUDE_L_STR] = float('NaN')
        self.groups["events"][date] = self.groups["events"][date][
        self.groups["events"][date][[OGS_C.LONGITUDE_STR,
                                     OGS_C.LATITUDE_STR]].apply(
          lambda x: mplPath(OGS_C.OGS_POLY_REGION, closed=True).contains_point(
            (x[OGS_C.LONGITUDE_STR], x[OGS_C.LATITUDE_STR])), axis=1)]
      for csv_file in filepath.glob("assignments/*/*/*.csv"):
        date = datetime.strptime(OGS_C.DASH_STR.join(str(
          csv_file).split("/")[-3:])[:-4], OGS_C.DATE_FMT)
        if date < self.args.dates[0] or date > self.args.dates[1]: continue
        self.groups.setdefault("assignments", {})[date] = pd.read_csv(csv_file)

  class OGS_TXT(DataSet):
    def __init__(self, filepath: Path, args: argparse.Namespace):
      super().__init__(filepath, args)

  class OGS_DAT(DataSet):
    def __init__(self, filepath: Path, args: argparse.Namespace):
      super().__init__(filepath, args)

  class OGS_HPL(DataSet):
    def __init__(self, filepath: Path, args: argparse.Namespace):
      super().__init__(filepath, args)

  class OGS_PUN(DataSet):
    def __init__(self, filepath: Path, args: argparse.Namespace):
      super().__init__(filepath, args)

  MODULES = {
    ".txt": OGS_TXT,
    ".dat": OGS_DAT,
    ".hpl": OGS_HPL,
    ".pun": OGS_PUN,
  }
  def __init__(self, filepath: Path, args: argparse.Namespace):
    super().__init__(filepath, args)
    self.load_(filepath)

  def load_(self, filepath: Path):
    folders = []
    path = str(filepath)
    while path != "/":
      path, folder = os.path.split(path)
      folders.append(folder)
    myFlag = False
    for folder in folders:
      if folder in self.MODULES.keys():
        self.RESULTS_PATH.append((folder,
                                  self.MODULES[folder](filepath, self.args)))
        myFlag = True
    if myFlag: return
    for subfolder in filepath.glob("*"):
      if subfolder.is_dir() and subfolder.name in self.MODULES.keys():
        self.RESULTS_PATH.append((subfolder.name, self.MODULES[subfolder.name](
          subfolder, self.args)))

class Target(Catalog):
  class DataSet:
    def __init__(self, filepath: Path, args: argparse.Namespace):
      self.filepath = filepath
      self.groups = {}
      self.args = args
      self.load_(filepath)
      self.save(self.groups.get("assignments"), dep="assignments")
      self.save(self.groups.get("events"), dep="events")

    def save(self, data: pd.DataFrame, dep: str = ""):
      if data is None or data.empty: return
      for group_name, group_data in data.groupby(OGS_C.GROUPS_STR):
        if type(group_name) is str:
          date = datetime.strptime(group_name, "%Y-%m-%d").date()
        else:
          date = group_name
        year = str(date.year)
        month = str(date.month)
        day = str(date.day)
        savepath = RES_DIR / self.__class__.__name__ / dep / \
                   year / f"{month:02}" / f"{day}.csv"
        savepath.parent.mkdir(parents=True, exist_ok=True)
        group_data.to_csv(savepath, index=False)
      if dep == "events":
        OGS_P.map_plotter(
          OGS_C.OGS_STUDY_REGION,
          x=data[OGS_C.LONGITUDE_STR],
          y=data[OGS_C.LATITUDE_STR],
          label=f"{self.__class__.__name__} Catalog",
          output=IMG_PATH / f"{self.__class__.__name__}Map.png",
          legend=True,
        )
        # Histogram of Depths
        OGS_P.histogram_plotter(
          data[OGS_C.DEPTH_STR].dropna(),
          xlabel="Depth (km)",
          ylabel="Number of Events",
          title=f"{self.__class__.__name__} Catalog Depths",
          color=OGS_C.OGS_BLUE,
          output=IMG_PATH / f"{self.__class__.__name__}Depths.png",
          legend=True
        )
        OGS_P.day_plotter(
          data[OGS_C.GROUPS_STR],
          ylabel="Number of Events",
          title=f"Events per Day",
          output=IMG_PATH / f"{self.__class__.__name__}CumulativeEvents.png",
          legend=True
        )
      else:
        OGS_P.day_plotter(
          data[OGS_C.GROUPS_STR],
          ylabel="Number of Picks",
          title=f"Picks per Day",
          output=IMG_PATH / f"{self.__class__.__name__}CumulativePicks.png",
          legend=True
        )

    def load_(self, pathname: Path):
      for parquet_file in pathname.glob("assignments/*"):
        df = pd.read_parquet(parquet_file)
        if df.empty: continue
        self.groups.setdefault("assignments", {})[parquet_file.name] = df
      if len(self.groups.get("assignments", [])) > 0:
        self.groups["assignments"] = pd.concat(
          self.groups["assignments"].values(), axis=0)
        self.groups["assignments"].rename(columns={
          "group": OGS_C.GROUPS_STR,
          "timestamp": OGS_C.TIMESTAMP_STR
        }, inplace=True)
        self.groups["assignments"][OGS_C.NETWORK_STR] = \
          self.groups["assignments"][OGS_C.STATION_STR].str.split(".").str[0]
        self.groups["assignments"][OGS_C.STATION_STR] = \
          self.groups["assignments"][OGS_C.STATION_STR].str.split(".").str[1]
        self.groups["assignments"][OGS_C.GROUPS_STR] = pd.to_datetime(
          self.groups["assignments"][OGS_C.TIMESTAMP_STR]).dt.date
      else:
        self.groups["assignments"] = pd.DataFrame()
      for parquet_file in pathname.glob("events/*"):
        df = pd.read_parquet(parquet_file)
        if df.empty: continue
        self.groups.setdefault("events", []).append(df)
      if len(self.groups.get("events", [])) > 0:
        self.groups["events"] = pd.concat(self.groups["events"], axis=0)
        self.groups["events"][OGS_C.GROUPS_STR] = pd.to_datetime(
          self.groups["events"][OGS_C.TIMESTAMP_STR]).dt.date
      else:
        self.groups["events"] = pd.DataFrame()

  class SeisBenchPicker(DataSet):
    def __init__(self, pathname: Path, args: argparse.Namespace):
      path, folder = os.path.split(pathname)
      super().__init__(pathname if folder == self.__class__.__name__
                                else Path(path), args=args)

    def load_(self, pathname: Path):
      for parquet_file in pathname.glob("picks/*"):
        df = pd.read_parquet(parquet_file)
        if df.empty: continue
        self.groups.setdefault("assignments", {})[parquet_file.name] = df
      if len(self.groups.get("assignments", [])) > 0:
        self.groups["assignments"] = pd.concat(
          self.groups["assignments"].values(), axis=0)
        self.groups["assignments"].rename(columns={
          "group": OGS_C.GROUPS_STR,
          "timestamp": OGS_C.TIMESTAMP_STR
        }, inplace=True)
        self.groups["assignments"][OGS_C.NETWORK_STR] = \
          self.groups["assignments"][OGS_C.STATION_STR].str.split(".").str[0]
        self.groups["assignments"][OGS_C.STATION_STR] = \
          self.groups["assignments"][OGS_C.STATION_STR].str.split(".").str[1]
        self.groups["assignments"][OGS_C.GROUPS_STR] = pd.to_datetime(
          self.groups["assignments"][OGS_C.TIMESTAMP_STR]).dt.date

  class GammaAssociator(DataSet):
    def __init__(self, pathname: Path, args: argparse.Namespace):
      path, folder = os.path.split(pathname)
      super().__init__(pathname if folder == self.__class__.__name__
                                else Path(path), args=args)
      if "assignments" in self.groups.keys():
        self.groups["assignments"][OGS_C.NETWORK_STR] = \
          self.groups["assignments"][OGS_C.STATION_STR].str.split(".").str[0]
        self.groups["assignments"][OGS_C.STATION_STR] = \
          self.groups["assignments"][OGS_C.STATION_STR].str.split(".").str[1]
        self.groups["assignments"][OGS_C.GROUPS_STR] = pd.to_datetime(
          self.groups["assignments"][OGS_C.TIMESTAMP_STR]).dt.date
      if "events" in self.groups.keys():
        self.groups["events"][OGS_C.MAGNITUDE_L_STR] = float('NaN')
        self.groups["events"][OGS_C.ERZ_STR] = float('NaN')
        self.groups["events"][OGS_C.ERH_STR] = float('NaN')
        self.groups["events"] = self.groups["events"][
        self.groups["events"][[OGS_C.LONGITUDE_STR, OGS_C.LATITUDE_STR]].apply(
          lambda x: mplPath(OGS_C.OGS_POLY_REGION, closed=True).contains_point(
            (x[OGS_C.LONGITUDE_STR], x[OGS_C.LATITUDE_STR])), axis=1)]

  class PyOctoAssociator(DataSet):
    def __init__(self, pathname: Path, args: argparse.Namespace):
      path, folder = os.path.split(pathname)
      super().__init__(pathname if folder == self.__class__.__name__
                                else Path(path), args=args)
      if "events" in self.groups.keys():
        self.groups["events"][OGS_C.MAGNITUDE_L_STR] = float('NaN')
        self.groups["events"][OGS_C.ERZ_STR] = float('NaN')
        self.groups["events"][OGS_C.ERH_STR] = float('NaN')
        self.groups["events"] = self.groups["events"][
        self.groups["events"][[OGS_C.LONGITUDE_STR, OGS_C.LATITUDE_STR]].apply(
          lambda x: mplPath(OGS_C.OGS_POLY_REGION, closed=True).contains_point(
            (x[OGS_C.LONGITUDE_STR], x[OGS_C.LATITUDE_STR])), axis=1)]

  class NonLinLoc(DataSet):
    def __init__(self, pathname: Path, args: argparse.Namespace):
      path, folder = os.path.split(pathname)
      super().__init__(pathname if folder == self.__class__.__name__
                                else Path(path), args=args)
      if "assignments" in self.groups.keys():
        self.groups["assignments"][OGS_C.GROUPS_STR] = pd.to_datetime(
          self.groups["assignments"][OGS_C.TIMESTAMP_STR]).dt.date
        self.groups["assignments"].drop(columns=[
          "group__duplicate__drop__"], inplace=True)
      if "events" in self.groups.keys():
        self.groups["events"][OGS_C.MAGNITUDE_L_STR] = float('NaN')
        self.groups["events"][OGS_C.ERZ_STR] = float('NaN')
        self.groups["events"][OGS_C.ERH_STR] = float('NaN')
        self.groups["events"][OGS_C.GROUPS_STR] = pd.to_datetime(
          self.groups["events"][OGS_C.TIMESTAMP_STR]).dt.date
        self.groups["events"] = self.groups["events"][
        self.groups["events"][[OGS_C.LONGITUDE_STR, OGS_C.LATITUDE_STR]].apply(
          lambda x: mplPath(OGS_C.OGS_POLY_REGION, closed=True).contains_point(
            (x[OGS_C.LONGITUDE_STR], x[OGS_C.LATITUDE_STR])), axis=1)]

  class OGSMagnitude(DataSet):
    def __init__(self, pathname: Path, args: argparse.Namespace):
      path, folder = os.path.split(pathname)
      super().__init__(pathname if folder == self.__class__.__name__
                                else Path(path), args=args)
      if "assignments" in self.groups.keys():
        self.groups["assignments"][OGS_C.GROUPS_STR] = pd.to_datetime(
          self.groups["assignments"][OGS_C.TIMESTAMP_STR]).dt.date
      if "events" in self.groups.keys():
        self.groups["events"][OGS_C.MAGNITUDE_L_STR] = float('NaN')
        self.groups["events"][OGS_C.ERZ_STR] = float('NaN')
        self.groups["events"][OGS_C.ERH_STR] = float('NaN')
        self.groups["events"][OGS_C.GROUPS_STR] = pd.to_datetime(
          self.groups["events"][OGS_C.TIMESTAMP_STR]).dt.date
        self.groups["events"] = self.groups["events"][
        self.groups["events"][[OGS_C.LONGITUDE_STR, OGS_C.LATITUDE_STR]].apply(
          lambda x: mplPath(OGS_C.OGS_POLY_REGION, closed=True).contains_point(
            (x[OGS_C.LONGITUDE_STR], x[OGS_C.LATITUDE_STR])), axis=1)]

  class events(DataSet):
    def __init__(self, csv: Path, args: argparse.Namespace):
      super().__init__(csv, args=args)

    def load_(self, pathname: Path):
      self.groups = {"events": pd.read_csv(str(pathname))}
      self.groups["events"].rename(columns={
        "vertical_uncertainty": OGS_C.ERZ_STR,
        "max_horizontal_uncertainty": OGS_C.ERH_STR,
        "ML": OGS_C.MAGNITUDE_L_STR,
      }, inplace=True)
      self.groups["events"][OGS_C.GROUPS_STR] = pd.to_datetime(
        self.groups["events"][OGS_C.TIMESTAMP_STR]).dt.date
      self.groups["events"] = self.groups["events"][
        self.groups["events"][[OGS_C.LONGITUDE_STR, OGS_C.LATITUDE_STR]].apply(
          lambda x: mplPath(OGS_C.OGS_POLY_REGION, closed=True).contains_point(
            (x[OGS_C.LONGITUDE_STR], x[OGS_C.LATITUDE_STR])), axis=1)]


  class assignments(DataSet):
    def __init__(self, csv: Path, args: argparse.Namespace):
      super().__init__(csv, args=args)

    def load_(self, pathname: Path):
      self.groups = {"assignments": pd.read_csv(str(pathname))}
      self.groups["assignments"][OGS_C.GROUPS_STR] = pd.to_datetime(
        self.groups["assignments"][OGS_C.TIMESTAMP_STR]).dt.date
      self.groups["assignments"][OGS_C.NETWORK_STR] = \
        self.groups["assignments"][OGS_C.STATION_STR].str.split(".").str[0]
      self.groups["assignments"][OGS_C.STATION_STR] = \
        self.groups["assignments"][OGS_C.STATION_STR].str.split(".").str[1]
      self.groups["assignments"].drop(columns=[
        "group", "group__duplicate__drop__"], inplace=True)

  MODULES = {
    "SeisBenchPicker": SeisBenchPicker,
    "GammaAssociator": GammaAssociator,
    "PyOctoAssociator": PyOctoAssociator,
    "NonLinLoc": NonLinLoc,
    "OGSMagnitude": OGSMagnitude,
    "events.csv": events,
    "assignments.csv": assignments,
  }
  def __init__(self, filepath: Path, args: argparse.Namespace):
    super().__init__(filepath, args)
    folders = []
    path = str(filepath)
    while path != "/":
      path, folder = os.path.split(path)
      folders.append(folder)
    myFlag = False
    for folder in folders:
      if folder in self.MODULES.keys():
        self.RESULTS_PATH.append((folder,
                                  self.MODULES[folder](filepath, args)))
        myFlag = True
    if myFlag: return
    for subfolder in filepath.glob("*"):
      if subfolder.name in self.MODULES.keys():
        self.RESULTS_PATH.append((subfolder.name, self.MODULES[subfolder.name](
          subfolder, args)))

class Comparison:
  def __init__(self, base_path: Path, target_path: Path,
               args: argparse.Namespace):
    self.base_path : Path = base_path
    self.target_path : Path = target_path
    self.start : datetime = args.dates[0]
    self.end : datetime = args.dates[1]
    self.args = args
    self.verbose : bool = args.verbose
    self.BaseCatalog = self.load_(self.base_path, base=True)
    self.TargetCatalog = self.load_(self.target_path, base=False)

  def inventory(self, stations: Path) -> op.Inventory:
    INVENTORY = op.Inventory()
    for st in stations.glob("*.xml"):
      try:
        S = op.read_inventory(str(st))
      except Exception as e:
        print(f"WARNING: Unable to read {st}")
        print(e)
        continue
      INVENTORY.extend(S)
    INVENTORY = {
      sta.code: (sta.longitude, sta.latitude, sta.elevation, net.code)
      for net in INVENTORY.networks for sta in net.stations
    }
    inv = pd.DataFrame.from_dict(
      INVENTORY, orient='index',
      columns=[OGS_C.LONGITUDE_STR, OGS_C.LATITUDE_STR, OGS_C.DEPTH_STR,
               OGS_C.NETWORK_STR])
    mystations = OGS_P.map_plotter(OGS_C.OGS_STUDY_REGION, legend=True, 
                                   marker='^', output="OGS_Stations.png")
    cmap = plt.get_cmap("turbo")
    colors = cmap(np.linspace(0, 1, inv[OGS_C.NETWORK_STR].nunique()))
    for i, (net, sta) in enumerate(inv.groupby(OGS_C.NETWORK_STR)):
      mystations.add_plot(sta[OGS_C.LONGITUDE_STR], sta[OGS_C.LATITUDE_STR],
                          label=net, color=None, facecolors='none',
                          edgecolors=colors[i], legend=True)
    mystations.savefig()
    plt.close()
    return INVENTORY

  def load_(self, elem: Path, base: bool) -> Catalog:
    return Base(elem, self.args) if base else Target(elem, self.args)

  MDL_EQUIV = {
    OGS_C.DAT_EXT: [1, 0],
    OGS_C.HPL_EXT: [1, 1],
    OGS_C.TXT_EXT: [0, 1],
    OGS_C.PUN_EXT: [0, 1],
    "SeisBenchPicker": [1, 0],
    "GammaAssociator": [1, 1],
    "PyOctoAssociator": [1, 1],
    "NonLinLoc": [1, 1],
    "OGSMagnitude": [1, 1],
    "events.csv": [0, 1],
    "assignments.csv": [1, 0],
  }

  def compare(self):
    self.TP = dict()
    self.FN = dict()
    self.FP = dict()
    for m, base in self.BaseCatalog.RESULTS_PATH:
      if base.groups == {}:
        print(f"WARNING: No data found in {m}")
        print(base.groups)
        continue
      for n, target in self.TargetCatalog.RESULTS_PATH:
        try:
          a = target.groups.keys()
        except AttributeError:
          print(f"WARNING: No data found in {n}")
          print(target.groups)
          continue
        res = np.asarray(self.MDL_EQUIV[m]) + np.asarray(self.MDL_EQUIV[n])
        for i, val in enumerate(res):
          if val != 2: continue
          if i == 0:
            a = [OGS_C.PWAVE, OGS_C.SWAVE, OGS_C.NONE_STR]
            header = [OGS_C.BASE_STR, OGS_C.TARGET_STR, OGS_C.UNKNOWN_STR,
                      OGS_C.INDEX_STR, OGS_C.TIMESTAMP_STR, OGS_C.PHASE_STR,
                      OGS_C.STATION_STR, OGS_C.ERT_STR]
            method = OGS_C.CLSSFD_STR
            key = "assignments"
          else:
            a = [OGS_C.EVENT_STR, OGS_C.NONE_STR]
            header = [OGS_C.BASE_STR, OGS_C.TARGET_STR, OGS_C.UNKNOWN_STR,
                      OGS_C.INDEX_STR, OGS_C.TIMESTAMP_STR, OGS_C.LATITUDE_STR,
                      OGS_C.LONGITUDE_STR, OGS_C.DEPTH_STR,
                      OGS_C.MAGNITUDE_L_STR, OGS_C.ERH_STR, OGS_C.ERZ_STR,
                      OGS_C.NOTES_STR]
            method = OGS_C.SOURCE_STR
            key = "events"
          CFN_MTX = pd.DataFrame(0, index=a, columns=a)
          cfn_mtx = pd.DataFrame(0, index=a, columns=a)
          TP, FN, FP = set(), [], set()
          for date, TARGET in target.groups[key].groupby(OGS_C.GROUPS_STR):
            date = datetime.strptime(str(date), OGS_C.DATE_FMT)
            if date not in base.groups[key].keys() or TARGET.empty: continue
            BASE = base.groups[key][date]
            BASE[OGS_C.TIMESTAMP_STR] = BASE[OGS_C.TIMESTAMP_STR].apply(
            lambda x: UTCDateTime(x))
            TARGET[OGS_C.TIMESTAMP_STR] = pd.to_datetime(
              TARGET[OGS_C.TIMESTAMP_STR]).apply(lambda x: UTCDateTime(x))
            TARGET[OGS_C.NOTES_STR] = target.__class__.__name__

            cfn_mtx, tp, fn, fp = OGS_B.conf_mtx(
              BASE, TARGET, m, n,
              method=method,
              use_hungarian=self.args.use_hungarian,
              min_weight=self.args.hungarian_min_weight,
            )
            CFN_MTX += cfn_mtx
            TP.update(tp)
            FN.extend(fn)
            FP.update(fp)
          print("Confusion Matrix:\n", CFN_MTX)
          print(f"True Positives: {len(TP)}")
          print(f"False Negatives: {len(FN)}")
          print(f"False Positives: {len(FP)}")
          print(f"Recall: {len(TP) / (len(TP) + len(FN))}")
          TP = pd.DataFrame(TP, columns=header)
          TP[OGS_C.TIMESTAMP_STR] = TP[OGS_C.TIMESTAMP_STR].apply(
            lambda x: (UTCDateTime(x[0]), UTCDateTime(x[1])))
          TP[OGS_C.INDEX_STR] = TP[OGS_C.INDEX_STR].apply(
            lambda x: (int(x[0]), int(x[1])))
          self.TP.setdefault(key, dict())[f"{n}{m}"] = TP
          FN = pd.DataFrame(FN, columns=header)
          FN[OGS_C.TIMESTAMP_STR] = FN[OGS_C.TIMESTAMP_STR].apply(UTCDateTime)
          self.FN.setdefault(key, dict())[f"{n}{m}"] = FN
          FP = pd.DataFrame(FP, columns=header)
          self.FP.setdefault(key, dict())[f"{n}{m}"] = FP
          disp = ConfMtxDisp(CFN_MTX.values, display_labels=CFN_MTX.columns)
          disp.plot(values_format='d')
          for labels in disp.text_.ravel():
            labels.set(color=OGS_C.MEX_PINK, fontsize=12, fontweight="bold")
          disp.im_.set(cmap="Blues", norm="log")
          if i == 0:
            trpppk = CFN_MTX.loc[OGS_C.PWAVE, OGS_C.PWAVE]
            trsspk = CFN_MTX.loc[OGS_C.SWAVE, OGS_C.SWAVE]
            fnppk = CFN_MTX.loc[OGS_C.PWAVE, OGS_C.NONE_STR]
            fnsspk = CFN_MTX.loc[OGS_C.SWAVE, OGS_C.NONE_STR]
            plt.title(f"Recall P: {trpppk / (fnppk + trpppk):.4f}, "
                      f"Recall S: {trsspk / (fnsspk + trsspk):.4f}, "
                      f"Recall: {(trpppk + trsspk) / \
                                 (fnppk + fnsspk + trpppk + trsspk):.4f}")
          else:
            fnev = CFN_MTX.loc[OGS_C.EVENT_STR, OGS_C.NONE_STR]
            trev = CFN_MTX.loc[OGS_C.EVENT_STR, OGS_C.EVENT_STR]
            plt.title(f"Recall: {trev / (fnev + trev):.4f}")
          plt.savefig(Path(IMG_PATH, f"{n}{m}ConfusionMatrix.png"))
          plt.close()
          if i == 0:
            fn_dict = {phase: df for phase, df in FN.groupby(OGS_C.PHASE_STR)}
            for phase, df in TP.groupby(OGS_C.PHASE_STR):
              print(f"Recall {phase}: "
                    f"{len(df) / (len(df) + len(fn_dict[phase]))}")
            # Time Difference Histogram
            OGS_P.histogram_plotter(
              TP[OGS_C.TIMESTAMP_STR].apply(lambda x: x[0] - x[1]),
              xlabel="Time Difference (s)",
              title="Event Time Difference",
              method=OGS_C.SOURCE_STR,
              output=f"{n}{m}TimeDiff.png",
              legend=True)
            plt.close()
            # ERT Distribution
            OGS_P.histogram_plotter(
              TP[OGS_C.ERT_STR].apply(lambda x: x[1] + (1. / x[0])),
              label="ERT (OGS)", output=f"{n}{m}ERT.png")
            plt.close()
            # Probability Distribution
            prob = OGS_P.histogram_plotter(
              TP[OGS_C.ERT_STR].apply(lambda x: x[1]),
              label="Probability (SBC)", legend=True)
            for c, (f, df) in zip([OGS_C.ALN_GREEN, OGS_C.MEX_PINK],
                                 TP.groupby(OGS_C.PHASE_STR)):
              prob.add_plot(
                df[OGS_C.ERT_STR].apply(lambda x: x[1]), color=c, label=f,
                legend=True, alpha=0.5, facecolor=c, edgecolor=OGS_C.OGS_BLUE,
                output=f"{n}{m}Probability.png")
            plt.close()
          else:
            # True Positive Map
            myplot = OGS_P.map_plotter(
              domain=OGS_C.OGS_STUDY_REGION,
              x=TP[OGS_C.LONGITUDE_STR].apply(lambda x: x[0]),
              y=TP[OGS_C.LATITUDE_STR].apply(lambda x: x[0]),
              facecolors="none", edgecolors=OGS_C.OGS_BLUE, legend=True,
              label="True Positive (OGS)")
            myplot.add_plot(
              TP[OGS_C.LONGITUDE_STR].apply(lambda x: x[1]),
              TP[OGS_C.LATITUDE_STR].apply(lambda x: x[1]), color=None,
                label="True Positive (SBC)", legend=True, facecolors="none",
                edgecolors=OGS_C.MEX_PINK, output=f"{n}{m}True.png")
            plt.close()
            # False Negative and False Positive Map
            myplot = OGS_P.map_plotter(
              domain=OGS_C.OGS_STUDY_REGION,
              x=FN[OGS_C.LONGITUDE_STR], y=FN[OGS_C.LATITUDE_STR],
              label="False Negative (OGS)", legend=True,)
            myplot.add_plot(
              FP[OGS_C.LONGITUDE_STR], FP[OGS_C.LATITUDE_STR], color=None,
                label="False Positive (SBC)", legend=True, facecolors="none",
                edgecolors=OGS_C.MEX_PINK, output=f"{n}{m}False.png")
            plt.close()
            # Time Difference Histogram
            OGS_P.histogram_plotter(
              TP[OGS_C.TIMESTAMP_STR].apply(lambda x: x[0] - x[1]),
              xlabel="Time Difference (s) [OGS - SBC]",
              title="Event Time Difference",
              method=OGS_C.SOURCE_STR,
              output=f"{n}{m}EventsTimeDiff.png",
              legend=True)
            plt.close()
            # Magnitude Difference Histogram
            OGS_P.histogram_plotter(
              TP[OGS_C.MAGNITUDE_L_STR].apply(lambda x: x[0] - x[1]),
              xlabel="Magnitude Difference (ML) [OGS - SBC]",
              title="Event Magnitude Difference",
              method=OGS_C.SOURCE_STR,
              output=f"{n}{m}MagnitudeDiff.png",
              legend=True)
            plt.close()
            # True Magnitude Distribution
            x = TP[OGS_C.MAGNITUDE_L_STR].apply(lambda x: x[0])
            y = TP[OGS_C.MAGNITUDE_L_STR].apply(lambda x: x[1])
            mx, mn = max(x.max(), y.max()), min(x.min(), y.min())
            mag = OGS_P.scatter_plotter(
              x, y, xlabel="OGS Magnitude", facecolors='none', legend=True,
              edgecolors=OGS_C.OGS_BLUE, ylabel="SBC Magnitude",
              title="Magnitude Prediction", aspect='equal')
            mag.add_plot([mn, mx], [mn, mx], color=OGS_C.MEX_PINK,
                         aspect='equal', legend=True,
                         output=f"{n}{m}TPDistMagnitude.png")
            plt.close()
            # Depth Difference Histogram
            OGS_P.histogram_plotter(
              TP[OGS_C.DEPTH_STR].apply(lambda x: x[0] - x[1]),
              xlabel="Depth Difference (km) [OGS - SBC]",
              title="Event Depth Difference",
              method=OGS_C.SOURCE_STR,
              output=f"{n}{m}DepthDiff.png",
              legend=True)
            plt.close()
            # Event Location Scatter Plot
            OGS_P.histogram_plotter(OGS_P.v_lat_long_to_distance(
              TP[OGS_C.LONGITUDE_STR].apply(lambda x: x[0]),
              TP[OGS_C.LATITUDE_STR].apply(lambda x: x[0]),
              TP[OGS_C.DEPTH_STR].apply(lambda x: 0),
              TP[OGS_C.LONGITUDE_STR].apply(lambda x: x[1]),
              TP[OGS_C.LATITUDE_STR].apply(lambda x: x[1]),
              TP[OGS_C.DEPTH_STR].apply(lambda x: x[1])),
              xlabel="Hypocentral Distance",
              title="Event Hypocentral Distance Difference",
              output=f"{n}{m}HypoDistDiff.png",
              legend=True)
            plt.close()
      for _, BASE in base.groups.items():
        BASE = pd.concat(BASE.values(), axis=0)
        if BASE.empty: continue
        for n, target in self.TargetCatalog.RESULTS_PATH:
          try:
            a = target.groups.keys()
          except AttributeError:
            print(f"WARNING: No data found in {n}")
            print(target.groups)
            continue
          res = np.asarray(self.MDL_EQUIV[m]) + np.asarray(self.MDL_EQUIV[n])
          for i, val in enumerate(res):
            if val != 2: continue
            if i == 0:
              a = [OGS_C.PWAVE, OGS_C.SWAVE, OGS_C.NONE_STR]
              method = OGS_C.CLSSFD_STR
              key = "assignments"
            else:
              a = [OGS_C.EVENT_STR, OGS_C.NONE_STR]
              method = OGS_C.SOURCE_STR
              key = "events"
    return

  def summary(self):
    return

def main(args : argparse.Namespace):
  comp = Comparison(args.base, args.target, args)
  comp.compare()
  comp.summary()
  return

if __name__ == "__main__": main(parse_arguments())
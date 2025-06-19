import seisbench.util as sbu
import seisbench.data as sbd
import parser as prs
from errors import ERRORS
from constants import *
from concurrent.futures import ThreadPoolExecutor
from obspy.core.utcdatetime import UTCDateTime
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
import calendar as cal
import pandas as pd
import obspy as op
import numpy as np
import argparse
import pickle
import json
from pathlib import Path
import os
# Set the project folder
PRJ_PATH = Path(os.path.dirname(__file__)).parent
IMG_PATH = Path(PRJ_PATH, "img")
DATA_PATH = Path(PRJ_PATH, "data")


# Seisbench


def dataset_builder(args: argparse.Namespace, SOURCE: pd.DataFrame = None,
                    DETECT: pd.DataFrame = None,
                    WAVEFORMS: pd.DataFrame = None):
  assert len(args.weights) == 1
  WEIGHT = args.weights[0]
  global DATA_PATH
  DATA_PATH = args.directory.parent
  if WAVEFORMS is None or args.force:
    WAVEFORMS = waveform_table(args)
  if SOURCE is None or DETECT is None or args.force:
    SOURCE, DETECT = true_loader(args, WAVEFORMS=WAVEFORMS)
  DATASET_PATH = Path(DATA_PATH, MODELS_STR, WEIGHT)
  DATASET_PATH.mkdir(parents=True, exist_ok=True)
  METADATA_PATH = Path(DATASET_PATH, METADATA_STR + CSV_EXT)
  WAVEFORM_PATH = Path(DATASET_PATH, WAVEFORMS_STR + HDF5_EXT)
  with sbd.WaveformDataWriter(METADATA_PATH, WAVEFORM_PATH) as WFW:
    print(f"Creating dataset for {WEIGHT}")
    WFW.data_format = {
        "dimension_order": "CW",
        "component_order": "ZNE",
        "measurement": "velocity",
        "unit": "counts",
        "instrument_response": "not restituted",
    }
    for _, SRC in SOURCE.iterrows():
      idx = SRC[ID_STR]
      date = SRC[TIME_STR].date
      latitude = SRC[LATITUDE_STR]
      longitude = SRC[LONGITUDE_STR]
      depth = SRC[LOCAL_DEPTH_STR]
      magnitude = SRC[MAGNITUDE_STR]
      event_params = {
          "source_id": idx,
          "source_origin_time": SRC[TIME_STR],
          "source_latitude_deg": latitude,
          "source_longitude_deg": longitude,
          "source_depth_km": depth,
          "source_magnitude": magnitude,
          "split": "train"
      }
      waveforms_d = WAVEFORMS[WAVEFORMS[DATE_STR] == date.strftime(DATE_FMT)]
      if waveforms_d.empty:
        continue
      for station, DTC in DETECT[DETECT[ID_STR] == idx].groupby(STATION_STR):
        waveforms = waveforms_d[waveforms_d[STATION_STR] == station]
        if waveforms.empty:
          continue
        id = waveforms[NETWORK_STR].unique()[0] + PERIOD_STR + station
        STATION_PATH = Path(DATA_PATH, STATION_STR, id + XML_EXT)
        STATION = op.read_inventory(STATION_PATH)[0][0]
        if STATION is None:
          continue
        traces = list(waveforms.index)
        start = DTC[TIME_STR].min() - PICK_OFFSET_TRAIN
        end = DTC[TIME_STR].max() + PICK_OFFSET_TRAIN
        stream = op.Stream()
        for trace in traces:
          if not Path(trace).exists():
            continue
          # TODO: Warning msg
          stream += op.read(trace, starttime=start, endtime=end,
                            nearest_sample=True)
          stream.resample(SAMPLING_RATE)
        # TODO: If filtered, consider that for TEST must be filtered as well
        # stream.detrend(type="linear").filter(type="highpass", freq=1., corners=4,
        #                                     zerophase=True)
        # TODO: Warning msg
        if len(stream) == 0:
          continue
        actual_t_start, data, _ = sbu.stream_to_array(
            stream, component_order=WFW.data_format["component_order"])
        trace_params = {
            "station_network_code": stream[-1].stats.network,
            "station_code": stream[-1].stats.station,
            "trace_channel": stream[-1].stats.channel,
            "station_location_code": stream[-1].stats.location,
            "station_latitude_deg": STATION.latitude,
            "station_longitude_deg": STATION.longitude,
            "station_elevation_m": STATION.elevation,
            "trace_sampling_rate_hz": SAMPLING_RATE,
            "trace_start_time": str(actual_t_start)
        }
        for phase, pick in DTC.groupby(PHASE_STR):
          sample = int((pick[TIME_STR].iloc[0] -
                       actual_t_start) * SAMPLING_RATE)
          trace_params[f"trace_{phase}_status"] = "manual"
          trace_params[f"trace_{phase}_arrival_sample"] = int(sample)
          trace_params[f"trace_{phase}_quality"] = float(
              pick[PROBABILITY_STR].iloc[0])
        WFW.add_trace({**event_params, **trace_params}, data)
        if args.verbose:
          print("Adding trace", trace_params)


def data_loader(filepath: Path):
  if not filepath.exists():
    raise FileNotFoundError(filepath)
  data = None
  sfx = filepath.suffix
  if sfx == JSON_EXT:
    with open(filepath, 'r') as f:
      data = json.load(f)
  elif sfx == CSV_EXT:
    data = pd.read_csv(filepath)
  elif sfx == HDF5_EXT:
    data = pd.read_hdf(filepath)
  elif sfx == MSEED_EXT:
    try:
      data = op.read(filepath)
    except:
      print(filepath)
      filepath.unlink()
  elif sfx == PICKLE_EXT:
    try:
      with open(filepath, 'rb') as f:
        data = pickle.load(f)
    except:
      print(filepath)
      filepath.unlink()
  elif sfx == DAT_EXT:
    return prs.event_parser_dat(filepath)
  elif sfx == PUN_EXT:
    return prs.event_parser_pun(filepath)
  elif sfx == HPC_EXT:
    return prs.event_parser_hpc(filepath)
  elif sfx == HPL_EXT:
    return prs.event_parser_hpl(filepath)
  elif sfx == MOD_EXT:
    return prs.event_parser_mod(filepath)
  elif sfx == QML_EXT:
    return prs.event_parser_qml(filepath)
  else:
    print(NotImplementedError)
  return data


def is_date(string: str) -> UTCDateTime:
  return UTCDateTime.strptime(string, YYMMDD_FMT)


def is_julian(string: str) -> UTCDateTime:
  # TODO: Define and convert Julian date to Gregorian date
  raise NotImplementedError
  return UTCDateTime.strptime(string, YYMMDD_FMT)._set_julday(string)


def is_file_path(string: str) -> Path:
  if os.path.isfile(string):
    return Path(os.path.abspath(string))
  else:
    raise FileNotFoundError(string)


def is_dir_path(string: str) -> Path:
  if os.path.isdir(string):
    return Path(os.path.abspath(string))
  else:
    raise NotADirectoryError(string)


def is_path(string: str) -> Path:
  if os.path.isfile(string) or os.path.isdir(string):
    return Path(os.path.abspath(string))
  else:
    raise FileNotFoundError(string)


class SortDatesAction(argparse.Action):
  def __call__(self, parser, namespace, values, option_string=None):
    setattr(namespace, self.dest, sorted(values))


class LoadFileAction(argparse.Action):
  def __call__(self, parser, namespace, values, option_string=None):
    with values as f:
      setattr(namespace, self.dest, json.load(f))


def parse_arguments():
  parser = argparse.ArgumentParser(description="Process AdriaArray Dataset")
  parser.add_argument('-C', "--channel", default=None, nargs=ALL_WILDCHAR_STR,
                      metavar=EMPTY_STR, required=False, type=str,
                      help="Specify a set of Channels to analyze. To allow "
                           "downloading data for any channel, set this option "
                           f"to \'{ALL_WILDCHAR_STR}\'.")
  # TODO: Handle security issues
  parser.add_argument('-F', "--file", default=list(), required=False,
                      type=is_path, nargs=ALL_WILDCHAR_STR, metavar=EMPTY_STR,
                      help="Supporting file path")
  parser.add_argument('-K', "--key", default=None, required=False,
                      type=is_file_path, metavar=EMPTY_STR,
                      help="Key to download the data from server.")
  parser.add_argument('-M', "--models", choices=MODEL_WEIGHTS_DICT.keys(),
                      default=[PHASENET_STR, EQTRANSFORMER_STR], type=str,
                      nargs=ONE_MORECHAR_STR, metavar=EMPTY_STR,
                      required=False,
                      help="Specify a set of Machine Learning based models")
  parser.add_argument('-N', "--network", default=[ALL_WILDCHAR_STR], type=str,
                      nargs=ONE_MORECHAR_STR, metavar=EMPTY_STR,
                      required=False,
                      help="Specify a set of Networks to analyze. To allow "
                           "downloading data for any channel, set this option "
                           f"to \'{ALL_WILDCHAR_STR}\'.")
  parser.add_argument('-S', "--station", default=[ALL_WILDCHAR_STR], type=str,
                      nargs=ONE_MORECHAR_STR, metavar=EMPTY_STR,
                      required=False,
                      help="Specify a set of Stations to analyze. To allow "
                           "downloading data for any channel, set this option "
                           f"to \'{ALL_WILDCHAR_STR}\'.")
  parser.add_argument('-T', "--train", default=False, action='store_true',
                      required=False, help="Train the model")
  parser.add_argument('-W', "--weights", required=False, metavar=EMPTY_STR,
                      default=[INSTANCE_STR, ORIGINAL_STR, STEAD_STR,
                               SCEDC_STR], nargs=ONE_MORECHAR_STR, type=str,
                      help="Specify a set of pretrained weights for the "
                           "selected Machine Learning based model. "
                           "WARNING: Weights which are not available for the "
                           "selected models will be skipped.")
  parser.add_argument('-b', "--batch", default=4096, type=int, required=False,
                      help="Batch size for the Machine Learning model")
  parser.add_argument('-c', "--config", default=None, type=is_file_path,
                      required=False, metavar=EMPTY_STR,
                      help="JSON configuration file path to load the "
                           "arguments. WARNING: The arguments specified in "
                           "the command line will overwrite the arguments in "
                           "the file.")
  parser.add_argument('-d', "--directory", required=False, type=is_dir_path,
                      default=Path(DATA_PATH, WAVEFORMS_STR),
                      help="Directory path to the raw files")
  parser.add_argument('-o', "--option", default=ALL_WILDCHAR_STR,
                      required=False, type=str,
                      help="Specify a set of options")
  parser.add_argument('-p', "--pwave", default=PWAVE_THRESHOLD, type=float,
                      required=False, help=f"{PWAVE} wave threshold.")
  parser.add_argument('-s', "--swave", default=SWAVE_THRESHOLD, type=float,
                      required=False, help=f"{SWAVE} wave threshold.")
  parser.add_argument("--client", default=[OGS_CLIENT_STR, INGV_CLIENT_STR,
                                           GFZ_CLIENT_STR, IRIS_CLIENT_STR,
                                           ETH_CLIENT_STR, ORFEUS_CLIENT_STR,
                                           COLLALTO_CLIENT_STR],
                      required=False, type=str, nargs=ONE_MORECHAR_STR,
                      help="Client to download the data")
  parser.add_argument("--denoiser", default=False, action='store_true',
                      required=False,
                      help="Enable Deep Denoiser model to filter the noise "
                           "previous to run the Machine Learning base model")
  parser.add_argument("--download", default=False, action='store_true',
                      required=False, help="Download the data")
  parser.add_argument("--interactive", default=False, action='store_true',
                      required=False, help="Interactive mode")
  parser.add_argument("--force", default=False, action='store_true',
                      required=False, help="Force running all the pipeline")
  parser.add_argument("--pyrocko", default=False, action='store_true',
                      help="Enable PyRocko calls")
  parser.add_argument("--pyocto", default=False, action='store_true',
                      help="Enable PyOcto calls")
  parser.add_argument("--timing", default=False, action='store_true',
                      required=False, help="Enable timing")
  date_group = parser.add_mutually_exclusive_group(required=False)
  date_group.add_argument('-D', "--dates", required=False, metavar="YYMMDD",
                          type=is_date, nargs=2, action=SortDatesAction,
                          default=[UTCDateTime.strptime("230601", YYMMDD_FMT),
                                   UTCDateTime.strptime("231231", YYMMDD_FMT)],
                          help="Specify the beginning and ending (inclusive) "
                               "Gregorian date (YYMMDD) range to work with.")
  date_group.add_argument('-J', "--julian", required=False, metavar="YYMMDD",
                          action=SortDatesAction, type=is_julian, default=None,
                          nargs=2,
                          help="Specify the beginning and ending (inclusive) "
                               "Julian date (YYMMDD) range to work with.")
  domain_group = parser.add_mutually_exclusive_group(required=False)
  domain_group.add_argument("--rectdomain", type=float, nargs=4,
                            metavar=("lonW", "lonE", "latS", "latN"),
                            default=OGS_STUDY_REGION,
                            help="Rectangular domain to download the data: "
                                 "[longitude West] [longitude East] "
                                 "[latitude South] [latitude North]")
  domain_group.add_argument("--circdomain", nargs=4, type=float,
                            # default=[46.3583, 12.808, 0., 0.3],
                            metavar=("lat", "lon", "min_r", "max_r"),
                            help="Circular domain to download the data: "
                                 "[center latitude] [center longitude] "
                                 "[minimum radius] [maximum radius]")
  verbal_group = parser.add_mutually_exclusive_group(required=False)
  verbal_group.add_argument("--silent", default=False, action='store_true',
                            help="Silent mode")
  # TODO: Add verbose LEVEL
  verbal_group.add_argument("-v", "--verbose", default=False,
                            action='store_true', help="Verbose mode")
  args = parser.parse_args()
  if args.config:
    config = data_loader(args.config)
    for key, value in config.items():
      if key in vars(args) and vars(args)[key] is None:
        setattr(args, key, value)
  # TODO: Fix special cases
  # print(vars(args))
  return args


def station_set(STATIONS: dict[str, set[str]]) -> set[str]:
  """
  Return a set of stations from the given dictionary.

  input:
    - STATIONS     (dict[str, set[str]])

  output:
    - set[str]

  errors:
    - None

  notes:

  """
  stations = set()
  for st in STATIONS.values():
    stations.update(st)
  return stations


def dump_args(args: argparse.Namespace,
              overwrite: bool = False) -> dict[str, str]:
  """
  Return the primary arguments used to execute the program. If the overwrite
  flag is set to True, then the primary arguments will be saved to a JSON file
  in the data directory.

  input:
    - args          (argparse.Namespace)
    - overwrite     (bool)

  output:
    - dict

  errors:
    - None

  notes:

  """
  global DATA_PATH
  DATA_PATH = args.directory.parent
  ARGUMENTS_FILE = Path(DATA_PATH, ARGUMENTS_STR + JSON_EXT)
  arg_dict = {
      MODEL_STR: args.models,
      WEIGHT_STR: args.weights,
      NETWORK_STR: args.network,
      STATION_STR: args.station,
      CHANNEL_STR: args.channel,
      DATE_STR: [a.__str__() for a in args.dates] if args.dates else
                [a.__str__() for a in args.julian],
      DIRECTORY_STR: args.directory.relative_to(PRJ_PATH).__str__(),
      DENOISER_STR: args.denoiser,
      DOMAIN_STR: args.rectdomain if args.rectdomain else args.circdomain
  }
  if overwrite:
    with open(ARGUMENTS_FILE, 'w') as fw:
      json.dump(arg_dict, fw, indent=2)
  return arg_dict


def read_args(args: argparse.Namespace,
              overwrite: bool = False) -> dict[str, str]:
  """
  Read the primary arguments used to execute the program from a JSON file in
  the data directory.

  input:
    - args          (argparse.Namespace)
    - overwrite     (bool)

  output:
    - dict

  errors:
    - FileNotFoundError

  notes:

  """
  global DATA_PATH
  DATA_PATH = args.directory.parent
  ARGUMENTS_FILE = Path(DATA_PATH, ARGUMENTS_STR + JSON_EXT)
  arg_dict = data_loader(ARGUMENTS_FILE) if (not overwrite and
                                             ARGUMENTS_FILE.exists()) else \
      dump_args(args, overwrite)
  return arg_dict


def data_header(args: argparse.Namespace,
                folder: str = CLF_STR) -> pd.DataFrame:
  """
  Construct a table of files based on the specified arguments.

  input:
    - args        (argparse.Namespace)
    - folder      (Path)

  output:
    - pandas.DataFrame

  errors:
    - FileNotFoundError

  notes:

  """
  global DATA_PATH
  DATA_PATH = args.directory.parent
  PATH = Path(DATA_PATH, folder)
  if args.verbose:
    print("Constructing the Table of", folder)
  if not PATH.exists():
    raise FileNotFoundError(PATH)
  RESULTS = list()
  start, end = args.dates
  for year in PATH.iterdir():
    if not year.is_dir():
      continue
    for month in year.iterdir():
      if not month.is_dir():
        continue
      for day in month.iterdir():
        if not day.is_dir():
          continue
        c_date = UTCDateTime(year=int(year.name), month=int(month.name),
                             day=int(day.name))
        if c_date < start or c_date >= end + ONE_DAY:
          continue
        for network_path in day.iterdir():
          if (args.network != [ALL_WILDCHAR_STR] and
                  network_path.stem not in args.network):
            continue
          if not network_path.is_dir():
            continue
          for station_path in network_path.iterdir():
            if (args.station != [ALL_WILDCHAR_STR] and
                    station_path.stem not in args.station):
              continue
            if not station_path.is_dir():
              continue
            for file_path in station_path.iterdir():
              # Handle (daily, network, station, model, weight) files
              filename = file_path.stem
              if args.denoiser == filename.startswith("D"):
                vars = filename.split(UNDERSCORE_STR)[int(args.denoiser):]
                RESULTS.append([str(file_path), *vars])
  # TODO: How unfortunate to not have thought handling a filesystem structure
  #       aligned with the Table structure
  HEADER = [FILENAME_STR, TIME_STR, NETWORK_STR, STATION_STR, MODEL_STR,
            WEIGHT_STR]
  RESULTS = pd.DataFrame(RESULTS, columns=HEADER)
  RESULTS = RESULTS[HEADER_FSYS]
  if args.network != [ALL_WILDCHAR_STR]:
    RESULTS = RESULTS[RESULTS[NETWORK_STR].isin(args.network)]
  if args.station != [ALL_WILDCHAR_STR]:
    RESULTS = RESULTS[RESULTS[STATION_STR].isin(args.station)]
  RESULTS = RESULTS[RESULTS[MODEL_STR].isin(args.models)]
  RESULTS = RESULTS[RESULTS[WEIGHT_STR].isin(args.weights)]
  RESULTS.sort_values(HEADER_FSYS[1:], inplace=True)
  RESULTS.set_index(FILENAME_STR, inplace=True)
  return RESULTS


def _loader(args: argparse.Namespace, folder: str) -> pd.DataFrame:
  if folder == CLF_STR:
    return classified_loader(args)
  elif folder == AST_STR:
    return associated_loader(args)
  else:
    return waveform_table(args)


def true_loader(args: argparse.Namespace, WAVEFORMS: pd.DataFrame = None,
                INVENTORY: op.Inventory = None,
                STATIONS: dict[str, set[str]] = None) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
  assert len(args.file) == 1
  global DATA_PATH
  DATA_PATH = args.directory.parent
  if STATIONS is None or INVENTORY is None:
    INVENTORY, STATIONS = station_loader(args, WAVEFORMS)
  stations = station_set(STATIONS)
  start, end = [x.strftime(DATE_FMT) for x in args.dates]
  SRC_FILE = Path(DATA_PATH, UNDERSCORE_STR.join([
      TRUE_STR, SOURCE_STR, start, end]) + CSV_EXT)
  DTC_FILE = Path(DATA_PATH, UNDERSCORE_STR.join([
      TRUE_STR, DETECT_STR, start, end]) + CSV_EXT)
  if SRC_FILE.exists() and not args.force:
    SOURCE = data_loader(SRC_FILE)
    SOURCE[TIME_STR] = SOURCE[TIME_STR].apply(
        lambda x: UTCDateTime(x))
    DETECT = data_loader(DTC_FILE)
    DETECT[TIME_STR] = DETECT[TIME_STR].apply(
        lambda x: UTCDateTime(x))
  else:
    SOURCE, DETECT = prs.event_parser(args.file[0], *args.dates, STATIONS)
    SOURCE = SOURCE[SOURCE[LATITUDE_STR] != NONE_STR].reset_index(drop=True)
    SOURCE = SOURCE.astype({LATITUDE_STR: float, LONGITUDE_STR: float,
                            LOCAL_DEPTH_STR: float, })
    DETECT = DETECT[DETECT[ID_STR].isin(SOURCE[ID_STR])].reset_index(drop=True)
    if args.verbose:
      SOURCE.to_csv(SRC_FILE, index=False)
      DETECT.to_csv(DTC_FILE, index=False)
  print("Picks Detections")
  # Table
  MTX = pd.DataFrame(0, index=DETECT[PROBABILITY_STR].unique(),
                     columns=DETECT[PHASE_STR].unique(), dtype=int)
  for id, val in DETECT[[PROBABILITY_STR, PHASE_STR]].value_counts().items():
    MTX.loc[id[0], id[1]] = val
  MTX["Total"] = MTX.sum(axis=1)
  MTX.sort_index(inplace=True)
  print(MTX.to_string(), end="\n\n")
  print(MTX.sum(axis=0), end="\n\n")
  PandS = 0
  for (id, st), df in DETECT.groupby([ID_STR, STATION_STR]):
    if len(df.groupby(PHASE_STR)) == 2:
      PandS += 1
    elif len(df.groupby(PHASE_STR)) != len(df.index):
      print(f"WARNING: {id} {st}")
      print(df, end="\n\n")
  print(f"True: {PandS} P and S", end="\n\n")
  # Another Table
  print("P & S Operator Weights")
  ABSENT = -1
  categories = sorted(DETECT[PROBABILITY_STR].unique().tolist()) + [ABSENT]
  PS_MTX = pd.DataFrame(0, index=categories, columns=categories, dtype=int)
  for (id, st), df in DETECT.groupby([ID_STR, STATION_STR]):
    x = df[[PHASE_STR, PROBABILITY_STR]]
    PS_MTX.loc[ABSENT if PWAVE not in set(x[PHASE_STR].to_list())
               else x[x[PHASE_STR] == PWAVE][PROBABILITY_STR],
               ABSENT if df[PHASE_STR].nunique() == 1
               else x[x[PHASE_STR] == SWAVE][PROBABILITY_STR]] += 1
  print(PS_MTX.to_string(), end="\n\n")
  if args.verbose:
    # SOURCE
    def plot_spatial_dist():
      FIG = plt.figure(figsize=(20, 10), layout="tight")
      GS = gridspec.GridSpec(1, 2, figure=FIG, wspace=0, width_ratios=[1, 7])
      plt.rcParams.update({'font.size': 12})
      if args.rectdomain:
        pm = 0.5
        xy = [args.rectdomain[0], args.rectdomain[2]]
        w = args.rectdomain[1] - args.rectdomain[0]
        h = args.rectdomain[3] - args.rectdomain[2]
        extent = [args.rectdomain[0] - pm, args.rectdomain[1] + pm,
                  args.rectdomain[2] - pm, args.rectdomain[3] + pm]
      if args.circdomain:
        raise NotImplementedError
        extent = [args.circdomain[1] - args.circdomain[3],
                  args.circdomain[1] + args.circdomain[3],
                  args.circdomain[0] - args.circdomain[3],
                  args.circdomain[0] + args.circdomain[3]]
      proj = ccrs.PlateCarree()
      stAx = FIG.add_subplot(GS[1], projection=proj,)
      stAx.add_patch(mpatches.Polygon(
          OGS_POLY_REGION, closed=True, linewidth=1, color='red', fill=False,
          label="OGS Catalog"))
      stAx.add_patch(mpatches.Rectangle(xy, w, h, linewidth=1, color='blue',
                                        fill=False, label=OGS_STUDY_STR))
      norm = plt.Normalize(vmin=0, vmax=SOURCE[LOCAL_DEPTH_STR].max())
      mask = SOURCE[MAGNITUDE_STR] >= OGS_MAX_MAGNITUDE
      EQ = SOURCE[~mask]
      stAx.scatter(EQ[LONGITUDE_STR], EQ[LATITUDE_STR], facecolors="none",
                   edgecolors=mpl.cm.plasma(norm(EQ[LOCAL_DEPTH_STR])),
                   transform=proj, alpha=1, s=10*(3.5**EQ[MAGNITUDE_STR]),
                   label=fr"$<$ {OGS_MAX_MAGNITUDE}")
      EQ = SOURCE[mask]
      stAx.scatter(EQ[LONGITUDE_STR], EQ[LATITUDE_STR], c=EQ[LOCAL_DEPTH_STR],
                   marker='*', edgecolors='black', norm=norm, transform=proj,
                   cmap="plasma", alpha=0.5, s=100*(1.5**EQ[MAGNITUDE_STR]),
                   label=fr"$\geq$ {OGS_MAX_MAGNITUDE} $M_{{EQ}}$")
      stAx.legend(fontsize=16)
      stAx.add_feature(cfeature.OCEAN, facecolor=("lightblue"))
      stAx.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor=MEX_PINK)
      stAx.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
      stAx.set_extent(extent, crs=proj)
      stAx.set_aspect('equal', adjustable='box')
      FIG.colorbar(stAx.collections[1], ax=stAx, shrink=0.8, aspect=50,
                   orientation='vertical', label="Depth (km)")
      gl = stAx.gridlines()
      gl.left_labels = True
      gl.top_labels = True
      # Region
      rgAx = FIG.add_axes((.67, .02, 0.15, 0.27), projection=proj)
      rgAx.add_patch(mpatches.Rectangle(xy, w, h, linewidth=1, color='blue',
                                        fill=False))
      rgAx.add_feature(cfeature.OCEAN, facecolor=("lightblue"))
      rgAx.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor=MEX_PINK)
      rgAx.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
      rgAx.set_extent([6, 19, 36, 48], crs=proj)
      rgAx.set_aspect('equal', adjustable='box')
      ita = rgAx.annotate("Italy", xy=(0.5, 0.55), xycoords='axes fraction',
                          ha='center', va='center', fontsize=20,
                          color=MEX_PINK)
      ita.set(rotation=-30)
      # Depth
      dpAx = FIG.add_axes((-.025, .07, 0.13, 0.48))
      dpAx.set_title("Depth Distribution")
      SOURCE[LOCAL_DEPTH_STR].hist(bins=NUM_BINS, ax=dpAx,
                                   orientation='horizontal')
      dpAx.set(xlabel="Number of Events", ylabel="Depth (km)")
      # Magnitude
      mgAx = FIG.add_axes((-.025, .65, 0.13, 0.3))
      mgAx.set_title("Magnitude Distribution")
      SOURCE[MAGNITUDE_STR].hist(bins=NUM_BINS, ax=mgAx)
      mgAx.set(xlabel="Magnitude", ylabel="Number of Events")
      """
      for i in range(len(MTX.index)):
        for j in range(len(MTX.columns)):
          tbAx.text(j, i, MTX.loc[i, j],
                    ha="center", va="center", color="w")
      """
      IMG_FILE = Path(IMG_PATH, UNDERSCORE_STR.join([
          TRUE_STR, SOURCE_STR, start, end]) + PNG_EXT)
      plt.tight_layout()
      plt.savefig(IMG_FILE, bbox_inches='tight')
      plt.close()
    plot_spatial_dist()

    def plot_magnitude_dist():
      _, ax = plt.subplots(figsize=(10, 5), layout='tight')
      plt.rcParams.update({'font.size': 12})
      SOURCE[MAGNITUDE_STR].hist(bins=NUM_BINS, color='grey', alpha=0.5, ax=ax)
      ax.set_title("Magnitude Distribution")
      ax.set(xlabel="Magnitude", ylabel="Number of Events")
      IMG_FILE = Path(IMG_PATH, (UNDERSCORE_STR.join([
          TRUE_STR, MAGNITUDE_STR, start, end]) + PNG_EXT))
      plt.tight_layout()
      plt.savefig(IMG_FILE, bbox_inches='tight')
      plt.close()
    plot_magnitude_dist()

    def plot_depth_dist():
      _, ax = plt.subplots(figsize=(10, 5), layout='tight')
      plt.rcParams.update({'font.size': 12})
      SOURCE[LOCAL_DEPTH_STR].hist(bins=NUM_BINS, color='grey', alpha=0.5)
      ax.set_title("Depth Distribution")
      ax.set(xlabel="Depth (km)", ylabel="Number of Events")
      IMG_FILE = Path(IMG_PATH, (UNDERSCORE_STR.join([
          TRUE_STR, LOCAL_DEPTH_STR, start, end]) + PNG_EXT))
      plt.tight_layout()
      plt.savefig(IMG_FILE, bbox_inches='tight')
      plt.close()
    plot_depth_dist()

    def plot_stations():
      FIG = plt.figure(figsize=(20, 10), layout="tight")
      GS = gridspec.GridSpec(1, 2, figure=FIG, wspace=0, width_ratios=[1, 7])
      plt.rcParams.update({'font.size': 12})
      st = set(DETECT[STATION_STR].unique().tolist())
      print(f"Catalog Stations: {len(st)}/{len(stations)}")
      start, end = args.dates
      dates = np.arange(start.datetime, end.datetime + ONE_DAY,
                        ONE_DAY).astype("datetime64[D]")
      st = pd.DataFrame([[n.code, s.code, s.latitude, s.longitude, 0]
                         for n in INVENTORY for s in n],
                        columns=[NETWORK_STR, STATION_STR, LATITUDE_STR,
                                 LONGITUDE_STR, "Total"]).drop_duplicates()
      for d, s in STATIONS.items():
        st.loc[st[STATION_STR].isin(list(s)), "Total"] += 1
      if args.rectdomain:
        pm = 0.5
        xy = [args.rectdomain[0], args.rectdomain[2]]
        w = args.rectdomain[1] - args.rectdomain[0]
        h = args.rectdomain[3] - args.rectdomain[2]
        extent = [args.rectdomain[0] - pm, args.rectdomain[1] + pm,
                  args.rectdomain[2] - pm, args.rectdomain[3] + pm]
      if args.circdomain:
        raise NotImplementedError
        extent = [args.circdomain[1] - args.circdomain[3],
                  args.circdomain[1] + args.circdomain[3],
                  args.circdomain[0] - args.circdomain[3],
                  args.circdomain[0] + args.circdomain[3]]
      proj = ccrs.PlateCarree()
      stAx = FIG.add_subplot(GS[1], projection=proj,)
      stAx.add_patch(mpatches.Polygon(
          OGS_POLY_REGION, closed=True, linewidth=1, color='red', fill=False,
          label="OGS Catalog"))
      stAx.add_patch(mpatches.Rectangle(xy, w, h, linewidth=1, color='blue',
                                        fill=False, label=OGS_STUDY_STR))
      stAx.scatter(st[LONGITUDE_STR], st[LATITUDE_STR], s=50, marker='^',
                   transform=proj, label=STATION_STR, cmap="cool_r",
                   c=st["Total"], edgecolors="black", linewidths=1,
                   alpha=0.5)
      x = {id: DETECT.loc[DETECT[ID_STR] == id, STATION_STR].nunique()
           for id, _ in SOURCE.groupby(ID_STR)}
      norm = plt.Normalize(vmin=0, vmax=max(x.values()))
      stAx.scatter(
          SOURCE[LONGITUDE_STR], SOURCE[LATITUDE_STR], alpha=1, transform=proj,
          edgecolors=[mpl.cm.binary(norm(x[st])) for st in SOURCE[ID_STR]],
          s=10*(3.5**SOURCE[MAGNITUDE_STR]), facecolors="none",
          label=fr"$<$ {OGS_MAX_MAGNITUDE}")
      stAx.add_feature(cfeature.OCEAN, facecolor=("lightblue"))
      stAx.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor=MEX_PINK)
      stAx.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
      stAx.set_extent(extent, crs=proj)
      stAx.set_aspect('equal', adjustable='box')
      FIG.colorbar(stAx.collections[0], ax=stAx, shrink=0.8, aspect=50,
                   orientation='vertical', label="Number of Days")
      gl = stAx.gridlines()
      gl.left_labels = True
      gl.top_labels = True
      # Region
      rgAx = FIG.add_axes((.64, .02, 0.2, 0.3), projection=proj)
      rgAx.add_patch(mpatches.Rectangle(xy, w, h, linewidth=1, color='blue',
                                        fill=False))
      rgAx.add_feature(cfeature.OCEAN, facecolor=("lightblue"))
      rgAx.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor=MEX_PINK)
      rgAx.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
      rgAx.set_extent([6, 19, 36, 48], crs=proj)
      rgAx.set_aspect('equal', adjustable='box')
      ita = rgAx.annotate("Italy", xy=(0.5, 0.55), xycoords='axes fraction',
                          ha='center', va='center', fontsize=20,
                          color=MEX_PINK)
      ita.set(rotation=-30)
      # Number of Active Days Distribution
      ndAx = FIG.add_axes((-.025, .06, 0.12, .55))
      st["Total"].hist(bins=NUM_BINS, ax=ndAx, orientation='horizontal')
      ndAx.set(xlabel="Number of Stations", ylabel="Number of Active days",
               xscale='log')
      """
      # Operator Weights Distribution
      owAx = FIG.add_axes((-.025, .65, 0.12, 0.3))
      disp = ConfMtxDisp(PS_MTX.values, display_labels=PS_MTX.columns,)
      disp.plot(ax=owAx, colorbar=False)
      disp.im_.set(cmap="Blues", norm="log")
      owAx.set(xlabel=f"{SWAVE} Weight", ylabel=f"{PWAVE} Weight")
      for labels in disp.text_.ravel():
        labels.set(color=MEX_PINK, fontweight="bold")
      """
      # Number of Stations Distribution
      nsAx = FIG.add_axes((-.025, .68, 0.12, 0.3))
      nsAx.hist(x.values(), bins=NUM_BINS)
      nsAx.set(xlabel="Number of Stations", ylabel="Number of Events",
               yscale='log')
      IMG_FILE = Path(IMG_PATH, (UNDERSCORE_STR.join([
          TRUE_STR, STATION_STR, start.strftime(DATE_FMT),
          end.strftime(DATE_FMT)]) + PNG_EXT))
      plt.tight_layout()
      plt.savefig(IMG_FILE, bbox_inches='tight')
      plt.close()
    plot_stations()
  return SOURCE, DETECT


def classified_loader(args: argparse.Namespace) -> pd.DataFrame:
  """
  input  :
    - args          (argparse.Namespace)

  output :
    - pd.DataFrame

  errors :
    - FileNotFoundError
    - AttributeError

  notes  :
    | MODEL | WEIGHT | TIMESTAMP | NETWORK | STATION | PHASE | PROBABILITY |
    ------------------------------------------------------------------------

    The data is loaded from the directory given in the arguments. The data is
    then sorted by the timestamp and returned as a pandas DataFrame.
  """
  global DATA_PATH
  DATA_PATH = Path(args.directory).parent
  CLF_PATH = Path(DATA_PATH, CLF_STR)
  if not CLF_PATH.exists():
    raise FileNotFoundError(CLF_PATH)
  DATA = list()
  for (model, weight, date), dataframe in \
          data_header(args, CLF_STR).groupby([
              MODEL_STR, WEIGHT_STR, TIME_STR]):
    if args.force and args.verbose:
      HIST = list()
    for filepath, (_, _, _, network, station) in dataframe.iterrows():
      PICK = [[model, weight, "{:.1f}".format(p.peak_value),
               str(p.peak_time.strftime(DATE_FMT)), str(p.peak_time),
               p.peak_value, p.phase, network, station]
              for p in data_loader(Path(filepath))]
      DATA += PICK
      PICK = pd.DataFrame(PICK, columns=HEADER_PRED)
      if args.force and args.verbose:
        w = reversed([len(PICK[PICK[THRESHOLD_STR] == th].index)
                      for th in THRESHOLDS])
        HIST.append([PERIOD_STR.join([network, station]), *w])
    if args.force and args.verbose:
      HIST = pd.DataFrame(HIST, columns=[ID_STR, *THRESHOLDS])\
               .set_index(ID_STR).sort_values(THRESHOLDS, ascending=False)
      IMG_FILE = \
          Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) +
               UNDERSCORE_STR.join([CLSSFD_STR, model, weight, date]) +
               PNG_EXT)
      HIST.plot(kind='bar', stacked=True, figsize=(20, 7), layout='tight')
      for leg in plt.legend().get_texts():
        leg.set_text(rf"$\geq$ {leg.get_text()}")
      plt.title(SPACE_STR.join([model, weight, date]))
      plt.ylabel("Number of Picks")
      plt.yscale('log')
      plt.ylim(0.9)
      plt.tight_layout()
      plt.savefig(IMG_FILE, bbox_inches='tight')
      plt.close()
  DATA = pd.DataFrame(DATA, columns=HEADER_PRED)\
           .sort_values(SORT_HIERARCHY_PRED).reset_index(drop=True)
  DATA[TIME_STR] = DATA[TIME_STR].apply(lambda x: UTCDateTime(x))
  return DATA


def associated_loader(args: argparse.Namespace) -> pd.DataFrame:
  """
  input  :
    - args          (argparse.Namespace)

  output :
    - pd.DataFrame

  errors :
    - FileNotFoundError
    - AttributeError

  notes  :
    | MODEL | WEIGHT | TIMESTAMP | NETWORK | STATION | PHASE | PROBABILITY |
    ------------------------------------------------------------------------

    The data is loaded from the directory given in the arguments. The data is
    then sorted by the timestamp and returned as a pandas DataFrame.
  """
  global DATA_PATH
  DATA_PATH = Path(args.directory).parent
  AST_PATH = Path(DATA_PATH, AST_STR)
  if not AST_PATH.exists():
    raise FileNotFoundError(AST_PATH)
  DATA = pd.DataFrame(columns=HEADER_PRED)
  for (model, weight, date), dataframe in \
          data_header(args, AST_STR).groupby([MODEL_STR, WEIGHT_STR,
                                              TIME_STR]):
    if args.verbose:
      HIST = list()
    for filepath, (_, _, _, network, station) in dataframe.iterrows():
      PICKS = pd.DataFrame(data_loader(Path(filepath)), columns=HEADER_PRED)
      DATA = pd.concat([DATA, PICKS],
                       ignore_index=True) if not DATA.empty else PICKS
      if args.verbose:
        w = reversed([len(PICKS[PICKS[THRESHOLDS] == th].index)
                      for th in THRESHOLDS])
        HIST.append([PERIOD_STR.join([network, station]), *w])
    if args.force and args.verbose:
      HIST = pd.DataFrame(HIST, columns=[ID_STR, *reversed(THRESHOLDS)])\
               .set_index(ID_STR).sort_values(THRESHOLDS, ascending=False)
      IMG_FILE = \
          Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) +
               UNDERSCORE_STR.join([DETECT_STR, model, weight, date]) +
               PNG_EXT)
      HIST.plot(kind='bar', stacked=True, figsize=(20, 7), layout='tight')
      for leg in plt.legend().get_texts():
        leg.set_text(rf"$\geq$ {leg.get_text()}")
      plt.ylabel("Number of Events")
      plt.title(SPACE_STR.join([model, weight, date]))
      plt.tight_layout()
      plt.savefig(IMG_FILE, bbox_inches='tight')
      plt.close()
  DATA = DATA.sort_values(SORT_HIERARCHY_PRED).reset_index(drop=True)
  DATA[TIME_STR] = DATA[TIME_STR].apply(lambda x: UTCDateTime(x))
  return DATA


def waveform_table(args: argparse.Namespace) -> pd.DataFrame:
  """
  Construct a table of files based on the specified arguments. A JSON file with
  the arguments will be saved in the data directory to act as a checksum and
  keep track of the arguments used to construct the table of files.

  input:
    - args        (argparse.Namespace)

  output:
    - pandas.DataFrame

  errors:
    - FileNotFoundError

  notes:
    If the starttime hour of the trace file is 23, then we assume the trace
    file records to be the next day
  """
  global DATA_PATH
  DATA_PATH = args.directory.parent
  WAVEFORMS_FILE = Path(DATA_PATH, WAVEFORMS_STR + CSV_EXT)
  start, end = args.dates
  if not args.force and WAVEFORMS_FILE.exists() and \
          (read_args(args, False) == dump_args(args, True)):
    # If the table of files already exists, we read the file and return the
    # table of files.
    if args.verbose:
      print("Reading the Table of Files")
    WAVEFORMS_DATA = data_loader(WAVEFORMS_FILE)
    WAVEFORMS_DATA[DATE_STR] = WAVEFORMS_DATA[DATE_STR].apply(
        lambda x: UTCDateTime.strptime(str(x), DATE_FMT))
    WAVEFORMS_DATA = \
        WAVEFORMS_DATA[(WAVEFORMS_DATA[DATE_STR] >= start) &
                       (WAVEFORMS_DATA[DATE_STR] < end + ONE_DAY)]
    WAVEFORMS_DATA[DATE_STR] = WAVEFORMS_DATA[DATE_STR].apply(
        lambda x: x.strftime(DATE_FMT))
  else:
    # Construct the table of files based on the specified arguments
    WAVEFORMS_DATA = list()
    if args.verbose:
      print("Constructing the Table of Files")
    results = list()
    for year_path in Path(DATA_PATH, WAVEFORMS_STR).iterdir():
      year = UTCDateTime(year=int(year_path.name), month=12, day=31)
      if year < start:
        continue
      for month_path in year_path.iterdir():
        month = UTCDateTime(year=year.year, month=int(month_path.name),
                            day=cal.monthrange(year.year,
                                               int(month_path.name))[1])
        if month < start:
          continue
        for day_path in month_path.iterdir():
          day = UTCDateTime(year=year.year, month=month.month,
                            day=int(day_path.name))
          if day < start or day >= end + ONE_DAY:
            continue
          for waveform_path in day_path.iterdir():
            if not waveform_path.is_file():
              continue
            # Read the waveform file and extract the necessary information
            try:
              stream = op.read(waveform_path)
            except Exception as e:
              print(f"WARNING: Unable to read {waveform_path}")
              print(e)
              continue
            # Extract the necessary information from the waveform file
            result = [str(waveform_path), stream[0].stats.network,
                      stream[0].stats.station, stream[0].stats.channel,
                      day.strftime(DATE_FMT)]
            results.append(result)
    # If the table of files was constructed, we save it to a CSV file
    WAVEFORMS_DATA = [result for result in results if result is not None]
    HEADER = [FILENAME_STR, NETWORK_STR, STATION_STR, CHANNEL_STR, DATE_STR]
    WAVEFORMS_DATA = pd.DataFrame(WAVEFORMS_DATA, columns=HEADER)
  # Filter the table of files based on the specified arguments
  for argument, filter in [(args.network, NETWORK_STR),
                           (args.station, STATION_STR),
                           (args.channel, CHANNEL_STR)]:
    if argument and argument != [ALL_WILDCHAR_STR]:
      WAVEFORMS_DATA = \
          WAVEFORMS_DATA[WAVEFORMS_DATA[filter].isin(argument)]
  # If no files were found in the specified directory, return an error message
  # and exit the program.
  if WAVEFORMS_DATA.empty and not args.silent:
    print(f"""FATAL: No files which meet the following criteria:
         --network {args.network}
         --station {args.station}
         --channel {args.channel}
         --dates   {SPACE_STR.join([d.__str__() for d in args.dates])}
       were found in the specified directory: {args.directory}""")
    if args.key is None:
      print("HINT: If you want to download the data from the server, please "
            "specify the download option \"--download\" and (if needed) "
            "provide a key file with the argument \"--key <key>\" for the "
            "optionally specified client with the argument \"--client "
            "<client>\"")
    raise FileNotFoundError
  WAVEFORMS_DATA.sort_values([DATE_STR, FILENAME_STR], inplace=True)
  WAVEFORMS_DATA.set_index(FILENAME_STR, inplace=True)
  WAVEFORMS_DATA.to_csv(WAVEFORMS_FILE)
  return WAVEFORMS_DATA


def station_loader(args: argparse.Namespace, WAVEFORMS=None) \
        -> tuple[op.Inventory, dict[str, set[str]]]:
  if WAVEFORMS is None:
    WAVEFORMS = waveform_table(args)
  start, end = args.dates
  dates = [str(d) for d in np.arange(start.datetime, end.datetime + ONE_DAY,
                                     ONE_DAY).astype("datetime64[D]").tolist()]
  min_s, max_s = np.inf, 0
  stations = set()
  STATIONS = dict()
  for d, ST in WAVEFORMS.groupby(DATE_STR):
    if args.station[0] == ALL_WILDCHAR_STR:
      ST = ST[ST[STATION_STR] in args.station]
    if ST.empty:
      continue
    STATIONS[d] = set(ST[STATION_STR].unique())
    ST = (ST[NETWORK_STR] + PERIOD_STR + ST[STATION_STR]).unique()
    min_s = min(min_s, len(ST))
    max_s = max(max_s, len(ST))
    stations.update(ST)
  print(f"Min Stations: {min_s}, Max Stations: {max_s}")
  print(f"Total Stations: {len(stations)}")
  INVENTORY = op.Inventory()
  for st in stations:
    STATION_PATH = Path(DATA_PATH, STATION_STR, st + XML_EXT)
    try:
      S = op.read_inventory(STATION_PATH)
    except Exception as e:
      print(f"WARNING: Unable to read {STATION_PATH}")
      print(e)
      continue
    s = S[0][0]
    if s.longitude < args.rectdomain[0] or s.longitude > args.rectdomain[1] or\
       s.latitude < args.rectdomain[2] or s.latitude > args.rectdomain[3]:
      print(f"WARNING: Station {st} is outside the specified domain")
      continue
    INVENTORY.extend(S)
  if args.verbose:
    # Plot the stations
    def plot_stations():
      POSITIONS = list()
      for net in INVENTORY:
        for st in net:
          if PERIOD_STR.join([net.code, st.code]) in stations:
            POSITIONS.append([net.code, st.code, st.latitude, st.longitude])
      POSITIONS = pd.DataFrame(POSITIONS, columns=[
          NETWORK_STR, STATION_STR, LATITUDE_STR, LONGITUDE_STR]) \
          .drop_duplicates()
      FIG = plt.figure(figsize=(20, 10), layout='tight')
      GS = gridspec.GridSpec(1, 2, figure=FIG, wspace=0, width_ratios=[1, 7])
      plt.rcParams.update({'font.size': 12})
      if args.rectdomain:
        pm = 0.5
        xy = [args.rectdomain[0], args.rectdomain[2]]
        w = args.rectdomain[1] - args.rectdomain[0]
        h = args.rectdomain[3] - args.rectdomain[2]
        extent = [args.rectdomain[0] - pm, args.rectdomain[1] + pm,
                  args.rectdomain[2] - pm, args.rectdomain[3] + pm]
      if args.circdomain:
        raise NotImplementedError
        extent = [args.circdomain[1] - args.circdomain[3],
                  args.circdomain[1] + args.circdomain[3],
                  args.circdomain[0] - args.circdomain[3],
                  args.circdomain[0] + args.circdomain[3]]
      proj = ccrs.PlateCarree()
      stAx = FIG.add_subplot(GS[1], projection=proj,)
      stAx.add_patch(mpatches.Polygon(
          OGS_POLY_REGION, closed=True, linewidth=1, color='red', fill=False,
          label="OGS Catalog"))
      stAx.add_patch(mpatches.Rectangle(xy, w, h, linewidth=1, color='blue',
                                        fill=False, label=OGS_STUDY_STR))
      stAx.add_feature(cfeature.OCEAN, facecolor=("lightblue"))
      stAx.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor=MEX_PINK)
      stAx.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
      stAx.set_extent(extent, crs=proj)
      stAx.set_aspect('equal', adjustable='box')
      gl = stAx.gridlines()
      gl.left_labels = True
      gl.top_labels = True
      cmap = plt.get_cmap("turbo")
      colors = cmap(np.linspace(0, 1, POSITIONS[NETWORK_STR].nunique()))
      for i, (net, df) in enumerate(POSITIONS.groupby(NETWORK_STR)):
        stAx.scatter(df[LONGITUDE_STR], df[LATITUDE_STR], s=50, marker='^',
                     transform=proj, label=net, facecolors='none',
                     edgecolors=colors[i], linewidths=2.5)
      stAx.legend(loc='lower left', fontsize=16)
      # Region
      rgAx = FIG.add_axes((.72, .02, 0.2, 0.3), projection=proj)
      rgAx.add_patch(mpatches.Rectangle(xy, w, h, linewidth=1, color='blue',
                                        fill=False))
      rgAx.add_feature(cfeature.OCEAN, facecolor=("lightblue"))
      rgAx.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor=MEX_PINK)
      rgAx.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
      rgAx.set_extent([6, 19, 36, 48], crs=proj)
      rgAx.set_aspect('equal', adjustable='box')
      ita = rgAx.annotate("Italy", xy=(0.5, 0.55), xycoords='axes fraction',
                          ha='center', va='center', fontsize=20,
                          color=MEX_PINK)
      ita.set(rotation=-30)
      IMG_FILE = Path(IMG_PATH, UNDERSCORE_STR.join([
          STATION_STR, "distribution"]) + PNG_EXT)
      plt.tight_layout()
      plt.savefig(IMG_FILE, bbox_inches='tight')
      plt.close()
    plot_stations()
    # Number of station in time

    def plot_time_stations():
      _, ax = plt.subplots(figsize=(10, 5), layout='tight')
      plt.rcParams.update({'font.size': 12})
      ax.plot(dates, [len(STATIONS[str(d)]) for d in dates])
      ax.set_title("Active number of stations per day")
      ax.set(xlabel="Date", ylabel="Number of stations",
             ylim=(0, (max_s + 9) // 10 * 10))
      ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
      for label in ax.get_xticklabels():
        label.set(rotation=30, horizontalalignment='right')
      ax.grid()
      IMG_FILE = Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) +
                      UNDERSCORE_STR.join([
                          STATION_STR, start.strftime(DATE_FMT),
                          end.strftime(DATE_FMT)]) + PNG_EXT)
      plt.tight_layout()
      plt.savefig(IMG_FILE, bbox_inches='tight')
      plt.close()
    plot_time_stations()
    # Plot the availability of stations

    def plot_availability():
      IMG_FILE = Path(IMG_PATH, UNDERSCORE_STR.join([
          STATION_STR, "availability"]) + PNG_EXT)
      _, ax = plt.subplots(figsize=(20, 20), layout='tight')
      ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
      ax.set_title("Available Stations")
      ax.set(xlabel=None, ylabel="Station")
      plt.rcParams.update({'font.size': 10})
      tmp = list()
      for st in stations:
        s = st.split(PERIOD_STR)
        tmp.append([*s] + [int(s[1] in STATIONS[str(d)]) for d in dates])
      tmp = pd.DataFrame(
          tmp, columns=[NETWORK_STR, STATION_STR] + dates).sort_values(
          [NETWORK_STR, STATION_STR])
      for i, (net, df) in enumerate(tmp.groupby(NETWORK_STR)):
        tmp.loc[tmp[NETWORK_STR] == net, dates] = df.loc[:, dates].apply(
            lambda x: x * (i + 1), axis=1)
      tmp.drop(columns=NETWORK_STR, inplace=True)
      tmp.set_index(STATION_STR, inplace=True)
      tmp.columns = dates
      ax.imshow(tmp, cmap='turbo', aspect='auto')
      # Assigning labels of y-axis
      xlabels = ax.get_xticklabels()
      print(xlabels)
      # according to dataframe
      plt.yticks(range(len(tmp.index)), tmp.index)
      for label in ax.get_xticklabels():
        label.set(rotation=30, horizontalalignment='right')
      plt.tight_layout()
      plt.savefig(IMG_FILE, bbox_inches='tight')
      plt.close()
    plot_availability()
  return INVENTORY, STATIONS

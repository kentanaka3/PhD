import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from pathlib import Path
# Set the project folder
PRJ_PATH = Path(os.path.dirname(__file__)).parent
IMG_PATH = Path(PRJ_PATH, "img")
DATA_PATH = Path(PRJ_PATH, "data")
import json
import pickle
import argparse
import numpy as np
import obspy as op
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy.core.utcdatetime import UTCDateTime
from concurrent.futures import ThreadPoolExecutor

from constants import *
from errors import ERRORS
import parser as prs

def data_loader(filepath : Path) -> any:
  if not filepath.exists(): raise FileNotFoundError
  data = None
  sfx = filepath.suffix
  if sfx == JSON_EXT:
    with open(filepath, 'r') as f: data = json.load(f)
  elif sfx == CSV_EXT: data = pd.read_csv(filepath)
  elif sfx == HDF5_EXT: data = pd.read_hdf(filepath)
  elif sfx == MSEED_EXT:
    try:
      data = op.read(filepath)
    except:
      print(filepath)
      filepath.unlink()
  elif sfx == PICKLE_EXT:
    try:
      with open(filepath, 'rb') as f: data = pickle.load(f)
    except:
      print(filepath)
      filepath.unlink()
  elif sfx == DAT_EXT: return prs.event_parser_dat(filepath)
  elif sfx == PUN_EXT: return prs.event_parser_pun(filepath)
  elif sfx == HPC_EXT: return prs.event_parser_hpc(filepath)
  elif sfx == HPL_EXT: return prs.event_parser_hpl(filepath)
  elif sfx == MOD_EXT: return prs.event_parser_mod(filepath)
  elif sfx == QML_EXT: return prs.event_parser_qml(filepath)
  else: print(NotImplementedError)
  return data

def is_date(string : str) -> UTCDateTime:
  return UTCDateTime.strptime(string, DATE_FMT)

def is_julian(string : str) -> UTCDateTime:
  # TODO: Define and convert Julian date to Gregorian date
  raise NotImplementedError
  return UTCDateTime.strptime(string, DATE_FMT)._set_julday(string)

def is_file_path(string : str) -> Path:
  if os.path.isfile(string): return Path(os.path.abspath(string))
  else: raise FileNotFoundError(string)

def is_dir_path(string : str) -> Path:
  if os.path.isdir(string): return Path(os.path.abspath(string))
  else: raise NotADirectoryError(string)

def is_path(string : str) -> Path:
  if os.path.isfile(string) or os.path.isdir(string):
    return Path(os.path.abspath(string))
  else: raise FileNotFoundError(string)

class SortDatesAction(argparse.Action):
  def __call__(self, parser, namespace, values, option_string=None):
    setattr(namespace, self.dest, sorted(values))

class LoadFileAction(argparse.Action):
  def __call__(self, parser, namespace, values, option_string=None):
    with values as f: setattr(namespace, self.dest, json.load(f))

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
  parser.add_argument('-G', "--groups", nargs=ONE_MORECHAR_STR, required=False,
                      metavar=EMPTY_STR,
                      default=[DATE_STR, NETWORK_STR, STATION_STR],
                      help="Analize the data based on a specified list")
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
  parser.add_argument('-S', "--station", default=[ALL_WILDCHAR_STR],type=str,
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
                                           ETH_CLIENT_STR, ORFEUS_CLIENT_STR],
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
                          default=[UTCDateTime.strptime("230601", DATE_FMT),
                                   UTCDateTime.strptime("231231", DATE_FMT)],
                          help="Specify the beginning and ending (inclusive) "
                               "Gregorian date (YYMMDD) range to work with.")
  date_group.add_argument('-J', "--julian", required=False, metavar="YYMMDD",
                          action=SortDatesAction, type=is_julian, default=None,
                          nargs=2,
                          help="Specify the beginning and ending (inclusive) "
                               "Julian date (YYMMDD) range to work with.")
  domain_group = parser.add_mutually_exclusive_group(required=False)
  domain_group.add_argument("--rectdomain", default=None, type=float, nargs=4,
                            metavar=("min_lat", "max_lat", "min_lon",
                                     "max_lon"),
                            # default=[44.5, 47, 10, 14.5],
                            help="Rectangular domain to download the data: "
                                 "[minimum latitude] [maximum latitude] "
                                 "[minimum longitude] [maximum longitude]")
  domain_group.add_argument("--circdomain", nargs=4, type=float,
                            default=[46.3583, 12.808, 0., 0.3],
                            metavar=("lat", "lon", "min_rad", "max_rad"),
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
  #print(vars(args))
  return args

def dump_args(args : argparse.Namespace,
              overwrite : bool = False) -> dict[str, str]:
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
  arg_dict : dict[str, str] = {
    MODEL_STR     : args.models,
    WEIGHT_STR    : args.weights,
    NETWORK_STR   : args.network,
    STATION_STR   : args.station,
    CHANNEL_STR   : args.channel,
    DATE_STR      : [a.__str__() for a in args.dates] if args.dates else
                    [a.__str__() for a in args.julian],
    GROUPS_STR    : args.groups,
    DIRECTORY_STR : args.directory.relative_to(PRJ_PATH).__str__(),
    DENOISER_STR  : args.denoiser,
    DOMAIN_STR    : args.rectdomain if args.rectdomain else args.circdomain
  }
  if overwrite:
    with open(ARGUMENTS_FILE, 'w') as fw: json.dump(arg_dict, fw, indent=2)
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

def data_header(args : argparse.Namespace,
                folder : str = CLF_STR) -> pd.DataFrame:
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
  if args.verbose: print("Constructing the Table of", folder)
  if not PATH.exists(): raise FileNotFoundError
  RESULTS = list()
  start, end = args.dates
  for date_path in PATH.iterdir():
    if DATE_STR in args.groups and date_path.is_dir():
      c_date = UTCDateTime.strptime(date_path.name, DATE_FMT)
      if c_date < start or c_date >= end + ONE_DAY: continue
      for network_path in date_path.iterdir():
        if (args.network != [ALL_WILDCHAR_STR] and
            network_path.stem not in args.network): continue
        if (NETWORK_STR in args.groups or STATION_STR in args.groups) and \
           network_path.is_dir():
          for station_path in network_path.iterdir():
            if (args.station != [ALL_WILDCHAR_STR] and
                station_path.stem not in args.station): continue
            if STATION_STR in args.groups and station_path.is_dir():
              for file_path in station_path.iterdir():
                # Handle (daily, network, station, model, weight) files
                filename = file_path.stem
                if args.denoiser == filename.startswith("D"):
                  vars = filename.split(UNDERSCORE_STR)[int(args.denoiser):]
                  RESULTS.append([str(file_path), *vars])
            else:
              # TODO: Handle (daily, network, model, weight) files
              filename = station_path.stem
              if args.denoiser == filename.startswith("D"):
                vars = filename.split(UNDERSCORE_STR)[int(args.denoiser):]
                continue
                RESULTS.append([str(station_path), *vars])
        else:
          # TODO: Handle (daily, model, weight) files
          filename = network_path.stem
          if args.denoiser == filename.startswith("D"):
            vars = filename.split(UNDERSCORE_STR)[int(args.denoiser):]
            continue
            RESULTS.append([str(network_path), *vars])
    else:
      # TODO: Handle (model, weight) files
      raise NotImplementedError
  # TODO: How unfortunate to not have thought handling a filesystem structure
  #       aligned with the Table structure
  HEADER = [FILENAME_STR, TIMESTAMP_STR, NETWORK_STR, STATION_STR, MODEL_STR,
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

def _loader(args : argparse.Namespace, folder : str) -> pd.DataFrame:
  if folder == CLF_STR:
    return classified_loader(args)
  elif folder == AST_STR:
    return associated_loader(args)
  else:
    return waveform_table(args)

def true_loader(args : argparse.Namespace, WAVEFORMS : pd.DataFrame = None,
                stations : dict[str, set[str]] = None) \
      -> tuple[pd.DataFrame, pd.DataFrame]:
  assert len(args.file) == 1
  global DATA_PATH
  DATA_PATH = args.directory.parent
  if stations is None: stations = station_loader(args, WAVEFORMS)
  SOURCE, DETECT = prs.event_parser(args.file[0], *args.dates, stations)
  if args.verbose:
    SOURCE.to_csv(Path(DATA_PATH,
                       UNDERSCORE_STR.join([TRUE_STR, SOURCE_STR]) + CSV_EXT),
                       index=False)
    DETECT.to_csv(Path(DATA_PATH,
                       UNDERSCORE_STR.join([TRUE_STR, DETECT_STR]) + CSV_EXT),
                       index=False)
  SOURCE = SOURCE[SOURCE[NOTES_STR].isnull() &
                  SOURCE[LATITUDE_STR].notna()].reset_index(drop=True)
  DETECT = DETECT[DETECT[ID_STR].isin(SOURCE[ID_STR])].reset_index(drop=True)
  print("Picks Detections")
  for phase in [PWAVE, SWAVE]:
    print(f"True ({phase}): {len(DETECT[DETECT[PHASE_STR] == phase].index)}")
  print(f"True: {len(DETECT.index)}")
  print("Events Sources")
  print(f"True: {len(SOURCE.index)}")
  if args.verbose:
    pass
  return SOURCE, DETECT

def classified_loader(args : argparse.Namespace) -> pd.DataFrame:
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
  DATA_PATH  = Path(args.directory).parent
  CLF_PATH = Path(DATA_PATH, CLF_STR)
  if not CLF_PATH.exists(): raise FileNotFoundError
  DATA = []
  for (model, weight, date), dataframe in \
    data_header(args, CLF_STR).groupby([MODEL_STR, WEIGHT_STR, TIMESTAMP_STR]):
    if args.force and args.verbose: HIST = list()
    for filepath, (_, _, _, network, station) in dataframe.iterrows():
      PICK = [[model, weight, "{:.1f}".format(p.peak_value), np.nan,
               str(p.peak_time), p.peak_value, p.phase, network, station]
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
        Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
             UNDERSCORE_STR.join([CLSSFD_STR, model, weight, date]) + PNG_EXT)
      HIST.plot(kind='bar', stacked=True, figsize=(20, 7))
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
  DATA[TIMESTAMP_STR] = DATA[TIMESTAMP_STR].apply(lambda x : UTCDateTime(x))
  return DATA

def associated_loader(args : argparse.Namespace) -> pd.DataFrame:
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
  DATA_PATH  = Path(args.directory).parent
  AST_PATH = Path(DATA_PATH, AST_STR)
  if not AST_PATH.exists(): raise FileNotFoundError
  DATA = pd.DataFrame(columns=HEADER_PRED)
  for (model, weight, date), dataframe in \
    data_header(args, AST_STR).groupby([MODEL_STR, WEIGHT_STR, TIMESTAMP_STR]):
    if args.verbose: HIST = list()
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
        Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
             UNDERSCORE_STR.join([DETECT_STR, model, weight, date]) + PNG_EXT)
      HIST.plot(kind='bar', stacked=True, figsize=(20, 7))
      for leg in plt.legend().get_texts():
        leg.set_text(rf"$\geq$ {leg.get_text()}")
      plt.ylabel("Number of Events")
      plt.title(SPACE_STR.join([model, weight, date]))
      plt.tight_layout()
      plt.savefig(IMG_FILE, bbox_inches='tight')
      plt.close()
  DATA = DATA.sort_values(SORT_HIERARCHY_PRED).reset_index(drop=True)
  DATA[TIMESTAMP_STR] = DATA[TIMESTAMP_STR].apply(lambda x : UTCDateTime(x))
  return DATA

def waveform_table(args : argparse.Namespace) -> pd.DataFrame:
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
    if args.verbose: print("Reading the Table of Files")
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
    if args.verbose: print("Constructing the Table of Files")

    def process_file(trc_file : Path) -> list[str]:
      try:
        trc = op.read(trc_file, headonly=True)[0].stats
      except:
        print(f"WARNING: Unable to read {trc_file}")
        return None
      trc_start = UTCDateTime(trc.starttime.date)
      if trc.starttime.hour == 23: trc_start += ONE_DAY
      if start <= trc_start < end + ONE_DAY:
        return [trc_file.__str__(), trc.network, trc.station, trc.channel,
                trc_start.strftime(DATE_FMT)]
      return None
    with ThreadPoolExecutor() as executor:
      results = list(executor.map(process_file, args.directory.iterdir()))

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
  if args.force: WAVEFORMS_DATA.to_csv(WAVEFORMS_FILE)
  return WAVEFORMS_DATA

def station_loader(args : argparse.Namespace, WAVEFORMS : pd.DataFrame = None)\
      -> dict[str, set[str]]:
  if WAVEFORMS is None: WAVEFORMS = waveform_table(args)
  start, end = args.dates
  DATES = [start.datetime]
  while DATES[-1] < end.datetime: DATES.append(DATES[-1] + ONE_DAY)
  DATES = [d.strftime(DATE_FMT) for d in DATES]
  min_s, max_s = np.inf, 0
  stations = set()
  STATIONS = dict()
  for d in DATES:
    ST = WAVEFORMS.loc[WAVEFORMS[DATE_STR] == d, [NETWORK_STR, STATION_STR]]
    if ST.empty: continue
    STATIONS[d] = set(ST[STATION_STR].unique())
    ST = (ST[NETWORK_STR] + PERIOD_STR + ST[STATION_STR]).unique()
    min_s = min(min_s, len(ST))
    max_s = max(max_s, len(ST))
    stations.update(ST)
    if args.verbose: print(f"Stations {d}: {len(ST)}")
  print(f"Min Stations: {min_s}, Max Stations: {max_s}")
  print(f"Total Stations: {len(stations)}")
  if args.verbose:
    dates = [np.datetime64(UTCDateTime.strptime(d, DATE_FMT)) for d in DATES]
    # Plot the stations
    INVENTORY = op.Inventory()
    for st in stations:
      STATION_PATH = Path(DATA_PATH, STATION_STR, st + XML_EXT)
      if not STATION_PATH.exists():
        print(f"WARNING: Station file {STATION_PATH} does not exist.")
        continue
      INVENTORY.extend(op.read_inventory(STATION_PATH))
    INVENTORY.plot(projection="local", show=False, method="cartopy", size=25,
                   water_fill_color="lightblue", color_per_network=True,
                   resolution="h", label=False,
                   outfile=Path(IMG_PATH, STATION_STR + PNG_EXT))
    # Plot the Number of station in time
    _, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates[:-1], [len(STATIONS[d]) for d in DATES[:-1]])
    ax.set_title("Active number of stations per day")
    # TODO: Add a base 10 multiplier for max_s
    ax.set(xlabel="Date", ylabel="Number of stations", ylim=(0, max_s))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for label in ax.get_xticklabels():
      label.set(rotation=30, horizontalalignment='right')
    ax.grid()
    IMG_FILE = Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) +
                    UNDERSCORE_STR.join([STATION_STR, start.strftime(DATE_FMT),
                                         end.strftime(DATE_FMT)]) + PNG_EXT)
    plt.savefig(IMG_FILE, bbox_inches='tight')
    plt.close()
  return STATIONS
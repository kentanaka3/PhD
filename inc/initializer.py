import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from pathlib import Path
# Set the project folder
PRJ_PATH = Path(os.path.dirname(__file__)).parent
IMG_PATH = Path(PRJ_PATH, "img")
DATA_PATH = Path(PRJ_PATH, "data")
import json
import obspy
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy.core.utcdatetime import UTCDateTime

from constants import *

def data_loader(filepath : Path) -> any:
  if not filepath.exists(): raise FileNotFoundError
  if filepath.suffix == JSON_EXT:
    with open(filepath, 'r') as f: data = json.load(f)
  elif filepath.suffix == CSV_EXT:
    data = pd.read_csv(filepath)
  elif filepath.suffix == HDF5_EXT:
    data = pd.read_hdf(filepath)
  elif filepath.suffix == PICKLE_EXT:
    with open(filepath, 'rb') as f: data = pickle.load(f)
  else:
    raise NotImplementedError
  return data

def is_date(string : str) -> UTCDateTime:
  return UTCDateTime.strptime(string, DATE_FMT)

def is_julian(string : str) -> UTCDateTime:
  # TODO: Define and convert Julian date to Gregorian date
  raise NotImplementedError
  return UTCDateTime.strptime(string, DATE_FMT)._set_julday(string)

def is_file_path(string : str) -> Path:
  if os.path.isfile(string): return Path(string)
  else: raise FileNotFoundError(string)

def is_dir_path(string : str) -> Path:
  if os.path.isdir(string): return Path(string)
  else: raise NotADirectoryError(string)

def is_path(string : str) -> Path:
  if os.path.isfile(string) or os.path.isdir(string): return Path(string)
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
  parser.add_argument('-F', "--file", default=None, required=False,
                      type=is_path, metavar=EMPTY_STR,
                      help="Supporting file path")
  parser.add_argument('-G', "--groups", nargs='+', required=False,
                      metavar=EMPTY_STR,
                      default=[BEG_DATE_STR, NETWORK_STR, STATION_STR],
                      help="Analize the data based on a specified list")
  parser.add_argument('-K', "--key", default=None, required=False,
                      type=is_file_path, metavar=EMPTY_STR,
                      help="Key to download the data from server.")
  parser.add_argument('-M', "--models", choices=MODEL_WEIGHTS_DICT.keys(),
                      required=False, metavar=EMPTY_STR, type=str, nargs='+',
                      default=[PHASENET_STR, EQTRANSFORMER_STR],
                      help="Specify a set of Machine Learning based models")
  parser.add_argument('-N', "--network", default=None, nargs=ALL_WILDCHAR_STR,
                      metavar=EMPTY_STR, required=False, type=str,
                      help="Specify a set of Networks to analyze. To allow "
                           "downloading data for any channel, set this option "
                           f"to \'{ALL_WILDCHAR_STR}\'.")
  parser.add_argument('-S', "--station", default=None, nargs=ALL_WILDCHAR_STR,
                      metavar=EMPTY_STR, required=False, type=str,
                      help="Specify a set of Stations to analyze. To allow "
                           "downloading data for any channel, set this option "
                           f"to \'{ALL_WILDCHAR_STR}\'.")
  parser.add_argument('-T', "--train", default=False, action='store_true',
                      required=False, help="Train the model")
  parser.add_argument('-W', "--weights", required=False, metavar=EMPTY_STR,
                      default=[INSTANCE_STR, ORIGINAL_STR, STEAD_STR,
                               SCEDC_STR], nargs='+', type=str,
                      help="Specify a set of pretrained weights for the "
                           "selected Machine Learning based model. "
                           "WARNING: Weights which are not available for the "
                           "selected models will not be considered")
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
  parser.add_argument('-p', "--pwave", default=PWAVE_THRESHOLD, type=float,
                      required=False, help=f"{PWAVE} wave threshold.")
  parser.add_argument('-s', "--swave", default=SWAVE_THRESHOLD, type=float,
                      required=False, help=f"{SWAVE} wave threshold.")
  parser.add_argument("--client", default=[OGS_CLIENT_STR], required=False,
                      type=str, nargs='+', help="Client to download the data")
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
  parser.add_argument("--timing", default=False, action='store_true',
                      required=False, help="Enable timing")
  date_group = parser.add_mutually_exclusive_group(required=False)
  date_group.add_argument('-D', "--dates", required=False, metavar="YYMMDD",
                          type=is_date, nargs=2, action=SortDatesAction,
                          default=[UTCDateTime.strptime("230601", DATE_FMT),
                                   UTCDateTime.strptime("230731", DATE_FMT)],
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
                                 "[latitude] [longitude] [minimum radius] "
                                 "[maximum radius]")
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
  DATA_PATH = Path(args.directory).parent
  ARGUMENTS_FILE = Path(DATA_PATH, ARGUMENTS_STR + JSON_EXT)
  arg_dict : dict[str, str] = {
    MODEL_STR     : args.models,
    WEIGHT_STR    : args.weights,
    NETWORK_STR   : args.network,
    STATION_STR   : args.station,
    CHANNEL_STR   : args.channel,
    BEG_DATE_STR  : [a.__str__() for a in args.dates] if args.dates else
                    [a.__str__() for a in args.julian],
    GROUPS_STR    : args.groups,
    DIRECTORY_STR : args.directory.relative_to(PRJ_PATH).__str__(),
    PWAVE         : args.pwave,
    SWAVE         : args.swave,
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
  DATA_PATH = Path(args.directory).parent
  ARGUMENTS_FILE = Path(DATA_PATH, ARGUMENTS_STR + JSON_EXT)
  if not overwrite and ARGUMENTS_FILE.exists():
    arg_dict = data_loader(ARGUMENTS_FILE)
  else:
    arg_dict = dump_args(args, overwrite)
  return arg_dict

def classified_header(args : argparse.Namespace) -> pd.DataFrame:
  """
  Construct a table of files based on the specified arguments.

  input:
    - args        (argparse.Namespace)

  output:
    - pandas.DataFrame

  errors:
    - FileNotFoundError

  notes:

  """
  global DATA_PATH
  DATA_PATH = Path(args.directory).parent
  CLF_PATH = Path(DATA_PATH, CLF_STR)
  if args.verbose: print("Constructing the Table of Classifications")
  if not CLF_PATH.exists(): raise FileNotFoundError
  CLASSIFICATIONS = list()
  start, end = args.dates
  for date_path in CLF_PATH.iterdir():
    if date_path.is_dir():
      c_date = UTCDateTime.strptime(date_path.name, DATE_FMT)
      if c_date < start or c_date >= end + ONE_DAY: continue
      for network_path in date_path.iterdir():
        if network_path.is_dir():
          for station_path in network_path.iterdir():
            if station_path.is_dir():
              for file_path in station_path.iterdir():
                filename = file_path.stem
                if args.denoiser and filename.startswith("D"):
                  vars = filename.split(UNDERSCORE_STR)[1:]
                  CLASSIFICATIONS.append([str(file_path), *vars])
                elif not args.denoiser and not filename.startswith("D"):
                  vars = filename.split(UNDERSCORE_STR)
                  CLASSIFICATIONS.append([str(file_path), *vars])
            else:
              # TODO: Handle daily, network, model, weight files
              continue
        else:
          # TODO: Handle daily, model, weight files
          continue
    else: raise NotImplementedError
  HEADER = [FILENAME_STR, BEG_DATE_STR, NETWORK_STR, STATION_STR, MODEL_STR,
            WEIGHT_STR]
  CLASSIFICATIONS = pd.DataFrame(CLASSIFICATIONS, columns=HEADER)
  R_HEADER = [FILENAME_STR, MODEL_STR, WEIGHT_STR, BEG_DATE_STR, NETWORK_STR,
              STATION_STR]
  CLASSIFICATIONS = CLASSIFICATIONS[R_HEADER]
  if args.network and args.network != [ALL_WILDCHAR_STR]:
    CLASSIFICATIONS = CLASSIFICATIONS[
      CLASSIFICATIONS[NETWORK_STR].isin(args.network)]
  if args.station and args.station != [ALL_WILDCHAR_STR]:
    CLASSIFICATIONS = CLASSIFICATIONS[
      CLASSIFICATIONS[STATION_STR].isin(args.station)]
  CLASSIFICATIONS = CLASSIFICATIONS[
    CLASSIFICATIONS[MODEL_STR].isin(args.models)]
  CLASSIFICATIONS = CLASSIFICATIONS[
    CLASSIFICATIONS[WEIGHT_STR].isin(args.weights)]
  CLASSIFICATIONS.sort_values(R_HEADER[1:], inplace=True)
  CLASSIFICATIONS.set_index(FILENAME_STR, inplace=True)
  return CLASSIFICATIONS

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
  HEADER = [MODEL_STR, WEIGHT_STR, TIMESTAMP_STR, NETWORK_STR, STATION_STR,
            PHASE_STR, PROBABILITY_STR]
  z = [round(t, 2) for t in np.linspace(0.2, 1.0, 9)]
  for (model, weight, date), dataframe in \
    classified_header(args).groupby([MODEL_STR, WEIGHT_STR, BEG_DATE_STR]):
    if args.verbose: HIST = list()
    for filepath, (_, _, _, network, station) in dataframe.iterrows():
      PICKS = [[model, weight, p.peak_time.__str__(), network, station,
                p.phase, p.peak_value] for p in data_loader(Path(filepath))]
      DATA += PICKS
      PICKS = pd.DataFrame(PICKS, columns=HEADER)
      if args.verbose:
        w = reversed([len(PICKS[(PICKS[PROBABILITY_STR] >= a) &
                                (PICKS[PROBABILITY_STR] < b)].index)
                      for a, b in zip(z[:-1], z[1:])])
        HIST.append([PERIOD_STR.join([network, station]), *w])
    if args.verbose:
      HIST = pd.DataFrame(HIST, columns=[ID_STR, *reversed(z[:-1])])\
               .set_index(ID_STR).sort_values(z[:-1], ascending=False)
      IMG_FILE = \
        Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
             UNDERSCORE_STR.join(["HIST", model, weight, date]) + PNG_EXT)
      HIST.plot(kind='bar', stacked=True, figsize=(20, 7))
      for leg in plt.legend().get_texts():
        leg.set_text(rf"$\geq$ {leg.get_text()}")
      plt.title(SPACE_STR.join([model, weight, date]))
      plt.tight_layout()
      plt.savefig(IMG_FILE)
      plt.close()
  DATA = pd.DataFrame(DATA, columns=HEADER).sort_values(SORT_HIERARCHY_PRED)\
           .reset_index(drop=True)
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
  DATA_PATH = Path(args.directory).parent
  WAVEFORMS_FILE = Path(DATA_PATH, WAVEFORMS_STR + CSV_EXT)
  start, end = args.dates
  if not args.force and WAVEFORMS_FILE.exists() and \
    (read_args(args, False) == dump_args(args, True)):
    # If the table of files already exists, we read the file and return the
    # table of files.
    if args.verbose: print("Reading the Table of Files")
    WAVEFORMS_DATA = data_loader(WAVEFORMS_FILE)
    WAVEFORMS_DATA[BEG_DATE_STR] = WAVEFORMS_DATA[BEG_DATE_STR].apply(
      lambda x: UTCDateTime.strptime(str(x), DATE_FMT))
    WAVEFORMS_DATA = \
      WAVEFORMS_DATA[(WAVEFORMS_DATA[BEG_DATE_STR] >= start) &
                     (WAVEFORMS_DATA[BEG_DATE_STR] < end + ONE_DAY)]
    WAVEFORMS_DATA[BEG_DATE_STR] = WAVEFORMS_DATA[BEG_DATE_STR].apply(
      lambda x: UTCDateTime.strftime(x, DATE_FMT))
  else:
    # Construct the table of files based on the specified arguments
    WAVEFORMS_DATA = list()
    if args.verbose: print("Constructing the Table of Files")
    for trc_file in args.directory.iterdir():
      try:
        trc = obspy.read(trc_file, headonly=True)[0].stats
      except:
        print(f"WARNING: Unable to read {trc_file}")
        continue
      trc_start = UTCDateTime(trc.starttime.date)
      if trc.starttime.hour == 23: trc_start += ONE_DAY
      if start <= trc_start and trc_start <= end:
        WAVEFORMS_DATA.append([trc_file.__str__(), trc.network, trc.station,
                               trc.channel,
                               UTCDateTime.strftime(trc_start, DATE_FMT)])
    HEADER = [FILENAME_STR, NETWORK_STR, STATION_STR, CHANNEL_STR,
              BEG_DATE_STR]
    WAVEFORMS_DATA = pd.DataFrame(WAVEFORMS_DATA, columns=HEADER)
  # Filter the table of files based on the specified arguments
  if args.network and args.network != [ALL_WILDCHAR_STR]:
    WAVEFORMS_DATA = \
      WAVEFORMS_DATA[WAVEFORMS_DATA[NETWORK_STR].isin(args.network)]
  if args.station and args.station != [ALL_WILDCHAR_STR]:
    WAVEFORMS_DATA = \
      WAVEFORMS_DATA[WAVEFORMS_DATA[STATION_STR].isin(args.station)]
  if args.channel and args.channel != [ALL_WILDCHAR_STR]:
    WAVEFORMS_DATA = \
      WAVEFORMS_DATA[WAVEFORMS_DATA[CHANNEL_STR].isin(args.channel)]
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
            "provide a key file with the argument \"--key\" <key> for the "
            "specified client with the argument \"--client\" <client>")
    raise FileNotFoundError
  WAVEFORMS_DATA.sort_values([BEG_DATE_STR, FILENAME_STR], inplace=True)
  WAVEFORMS_DATA.set_index(FILENAME_STR, inplace=True)
  WAVEFORMS_DATA.to_csv(WAVEFORMS_FILE)
  return WAVEFORMS_DATA
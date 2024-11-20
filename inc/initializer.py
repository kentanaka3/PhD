import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from pathlib import Path
# Set the project folder
PRJ_PATH = Path(os.path.dirname(__file__)).parent
DATA_PATH = Path(PRJ_PATH, "data")
import json
import argparse
from obspy.core.utcdatetime import UTCDateTime

from constants import *

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
  parser.add_argument('-c', "--config", default=None, type=open,
                      action=LoadFileAction,
                      required=False, metavar=EMPTY_STR,
                      help="Configuration file path")
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
                            # default=[44.5, 45, 10, 14.5],
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
  return parser.parse_args()
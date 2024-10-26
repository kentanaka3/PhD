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
from constants import *
import json
import torch
import pickle
import argparse
import itertools
import numpy as np
import numba as nb
import pandas as pd
from mpi4py import MPI
import matplotlib.pyplot as plt

# ObsPy
import obspy
from obspy.core.utcdatetime import UTCDateTime

# SeisBench
import seisbench.util as sbu
import seisbench.data as sbd
import seisbench.generate as sbg

# TODO: Study GaMMA associator with folder
# TODO: Colab PyOcto associator to be tested with GaMMA
# TODO: Get Vel Model
# TODO: Discuss constants.NORM = "peak"

def is_date(string : str) -> UTCDateTime:
  return UTCDateTime.strptime(string, DATE_FMT)

def is_file_path(string : str) -> Path:
  if os.path.isfile(string): return Path(string)
  else: raise NotADirectoryError(string)

def is_dir_path(string : str) -> Path:
  if os.path.isdir(string): return Path(string)
  else: raise NotADirectoryError(string)

class SortDatesAction(argparse.Action):
  def __call__(self, parser, namespace, values, option_string=None):
    setattr(namespace, self.dest, sorted(values))

def parse_arguments():
  parser = argparse.ArgumentParser(description="Process AdriaArray Dataset")
  parser.add_argument('-C', "--channel", default=None, nargs=ALL_WILDCHAR_STR,
                      metavar=EMPTY_STR, required=False, type=str,
                      help="Specify a set of Channels to analyze. To allow "
                           "downloading data for any channel, set this option "
                           f"to \'{ALL_WILDCHAR_STR}\'.")
  parser.add_argument('-D', "--dates", nargs=2, required=False, type=is_date,
                      metavar="DATE", action=SortDatesAction,
                      default=[UTCDateTime.strptime("230601", DATE_FMT),
                               UTCDateTime.strptime("230801", DATE_FMT)],
                      help="Specify the beggining and ending (inclusive) date "
                           "(YYMMDD) range to work with.")
  parser.add_argument('-G', "--groups", nargs='+', required=False,
                      metavar=EMPTY_STR,
                      default=[BEG_DATE_STR, NETWORK_STR, STATION_STR],
                      help="Analize the data based on a specified list")
  parser.add_argument('-J', "--julian", default=False, action="store_true",
                      help="Transform the selected dates into Julian date.")
  # TODO: Implement data retrieval
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
                      metavar=EMPTY_STR,
                      help="Batch size for the Machine Learning model")
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
  domain_group = parser.add_mutually_exclusive_group(required=False)
  domain_group.add_argument("--rectdomain", default=None, type=float, nargs=4,
                            metavar=('min_lat', 'max_lat', 'min_lon',
                                     'max_lon'),
                            help="Rectangular domain to download the data: "
                                 "[minimum latitude] [maximum latitude] "
                                 "[minimum longitude] [maximum longitude]")
  domain_group.add_argument("--circdomain", nargs=4, type=float,
                            default=[46.3583, 12.808, 0., 0.3],
                            metavar=('lat', 'lon', 'min_rad', 'max_rad'),
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

def read_data(path : str):
  # Load the data
  with open(Path(path), 'rb') as f: data = pickle.load(f)
  return data

def load_data(args : argparse.Namespace) -> pd.DataFrame:
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
  start, end = args.dates
  z = [round(t, 2) for t in np.linspace(0.2, 1.0, 9)]
  for model in args.models:
    for weight in args.weights:
      for date_path in CLF_PATH.iterdir():
        if args.verbose: HIST = []
        date = date_path.name
        date_obj = UTCDateTime.strptime(date, DATE_FMT)
        if date_obj < start or date_obj >= end + ONE_DAY: continue
        for network_path in date_path.iterdir():
          network = network_path.name
          for station_path in network_path.iterdir():
            station = station_path.name
            f = Path(station_path, ("D_" if args.denoiser else EMPTY_STR) + \
                     UNDERSCORE_STR.join([date, network, station, model,
                                          weight]) + PICKLE_EXT)
            PICKS = [[model, weight, p.peak_time, network, station, p.phase,
                      p.peak_value] for p in read_data(f)]
            DATA += PICKS
            PICKS = pd.DataFrame(PICKS, columns=HEADER)
            if args.verbose:
              w = reversed([len(PICKS[(PICKS[PROBABILITY_STR] >= a) &
                                      (PICKS[PROBABILITY_STR] < b)].index)
                            for a, b in zip(z[:-1], z[1:])])
              HIST.append([station_path.relative_to(date_path).__str__(), *w])
        if args.verbose:
          HIST = pd.DataFrame(HIST, columns=[FILE_STR, *reversed(z[:-1])])\
                  .set_index(FILE_STR).sort_values(z[:-1], ascending=False)
          IMG_FILE = \
            Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                UNDERSCORE_STR.join(["HIST", model, weight, date]) + PNG_EXT)
          HIST.plot(kind='bar', stacked=True, figsize=(20, 7))
          plt.title(SPACE_STR.join([model, weight, date]))
          plt.tight_layout()
          plt.savefig(IMG_FILE)
          plt.close()
  return pd.DataFrame(DATA, columns=HEADER).sort_values(TIMESTAMP_STR)\
                                           .reset_index(drop=True)

def data_downloader(args : argparse.Namespace) -> None:
  """
  Download the data from the server based on the specified arguments. If the
  data is already present in the directory, the data will be replaced by the
  new data.

  input:
    - args          (argparse.Namespace)

  output:
    - None

  errors:
    - None

  notes:

  """
  global DATA_PATH
  DATA_PATH = Path(args.directory).parent
  if args.verbose:
    print("Downloading the Data to the directory:", args.directory)
  if args.pyrocko:
    # We enable the option to use the PyRocko module to download the data as it
    # is more efficient than the ObsPy module by multithreading the download.
    import pyrocko as pr

  else:
    from obspy.clients.fdsn import Client
    CLIENTS = [Client(client) for client in args.client]
    if args.rectdomain:
      from obspy.clients.fdsn.mass_downloader.domain import RectangularDomain
      domain = RectangularDomain(minlatitude=args.rectdomain[0],
                                 maxlatitude=args.rectdomain[1],
                                 minlongitude=args.rectdomain[2],
                                 maxlongitude=args.rectdomain[3])
    else:
      from obspy.clients.fdsn.mass_downloader.domain import CircularDomain
      domain = CircularDomain(latitude=args.circdomain[0],
                              longitude=args.circdomain[1],
                              minradius=args.circdomain[2],
                              maxradius=args.circdomain[3])
    from obspy.clients.fdsn.mass_downloader import Restrictions, MassDownloader
    start, end = args.dates
    restrictions = Restrictions(starttime=start, endtime=end + ONE_DAY,
                                network=COMMA_STR.join(args.network),
                                station=COMMA_STR.join(args.station),
                                channel_priorities=["HH[ZNE]", "EH[ZNE]",
                                                    "HN[ZNE]", "HG[ZNE]"],
                                reject_channels_with_gaps=False,
                                minimum_length=0.0,
                                minimum_interstation_distance_in_m=100.0,
                                location_priorities=["", "00", "10"],
                                chunklength_in_sec=86400)
    if args.key:
      # NOTE: It is assumed a single token file is applicable for all clients
      for cl in CLIENTS: cl.set_eida_token(args.key, validate=True)
    mdl = MassDownloader(providers=CLIENTS)
    mdl.download(domain, restrictions, mseed_storage=args.directory.__str__(),
                 stationxml_storage=Path(DATA_PATH, STATION_STR).__str__())

def waveform_table(args : argparse.Namespace) -> pd.DataFrame:
  """
  Construct a table of files based on the specified arguments. If a key file is
  provided, the data will be downloaded from the server and the table of files
  will be constructed based on the specified arguments. The table of files will
  be saved in the data directory. A JSON file with the arguments will be saved
  in the data directory to act as a checksum and keep track of the arguments
  used to construct the table of files.

  args.key == None -> No data download (local data)
    args.network == None -> All locally available networks
    args.station == None -> All locally available stations
    args.channel == None -> All locally available channels

    args.network == '*' -> All locally available networks
    args.station == '*' -> All locally available stations
    args.channel == '*' -> All locally available channels

    args.network != [None, '*'] -> Specific locally available networks
    args.station != [None, '*'] -> Specific locally available stations
    args.channel != [None, '*'] -> Specific locally available channels

  args.key != None -> Data download (remote data)
    args.network == None -> Error
    args.station == None -> Error
    args.channel == None -> Error

    args.network == '*' -> All remotely available networks
    args.station == '*' -> All remotely available stations
    args.channel == '*' -> All remotely available channels

    args.network != [None, '*'] -> Specific remotely available networks
    args.station != [None, '*'] -> Specific remotely available stations
    args.channel != [None, '*'] -> Specific remotely available channels

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
  HEADER = [FILENAME_STR, NETWORK_STR, STATION_STR, CHANNEL_STR, BEG_DATE_STR]
  WAVEFORMS_DATA = list()
  WAVEFORMS_FILE = Path(DATA_PATH, WAVEFORMS_STR + CSV_EXT)
  ARGUMENTS_FILE = Path(DATA_PATH, ARGUMENTS_STR + JSON_EXT)
  if args.force or not ARGUMENTS_FILE.exists() or \
     read_arguments(args) != primary_arguments(args):
    # If the arguments file does not exist or the arguments are different from
    # the ones in the JSON file, we save the arguments to the JSON file and
    # construct the table of files based on the specified arguments.
    read_arguments(args, overwrite=True)
  else:
    # If the arguments are the same as the ones in the JSON file, we load the
    # table of files from the CSV file.
    if WAVEFORMS_FILE.exists():
      if args.verbose:
        print("Found and loading previously constructed table of files:",
              WAVEFORMS_FILE)
      DATAFRAME = pd.read_csv(WAVEFORMS_FILE)
      DATAFRAME[FILENAME_STR] = DATAFRAME[FILENAME_STR].apply(Path)
      DATAFRAME[BEG_DATE_STR] = DATAFRAME[BEG_DATE_STR].apply(str)
      DATAFRAME.set_index(FILENAME_STR, inplace=True)
      return DATAFRAME
  if args.download or args.key is not None:
    # We download the data from the server and construct the table of files 
    # based on the specified arguments.
    data_downloader(args)
  # Construct the table of files based on the specified arguments
  if args.verbose: print("Constructing the Table of Files")
  for trc_file in args.directory.iterdir():
    fr = Path(args.directory, trc_file)
    if fr.is_file():
      try:
        trc = obspy.read(fr, headonly=True)[0].stats
      except:
        continue
      start = UTCDateTime(trc.starttime.date)
      if trc.starttime.hour == 23: start += ONE_DAY
      # We start by assuming the trace file meets all the criterias:
      # (network, station, channel, date range). If the trace file does meet
      # we save the metadata to the list of files called the waveform table.
      outcome = True
      # If the user has specified the network different from the wildcard, then
      # we check if the network of the trace file is in the list of networks,
      # otherwise we assume the user wants to analyze all downloaded networks.
      # The same logic applies to the station and channel.
      if args.network and args.network != [ALL_WILDCHAR_STR]:
        outcome = outcome and any([n == trc.network for n in args.network])
      if args.station and args.station != [ALL_WILDCHAR_STR] and outcome:
        outcome = outcome and any([n == trc.station for n in args.station])
      if args.channel and args.channel != [ALL_WILDCHAR_STR] and outcome:
        outcome = outcome and any([n == trc.channel for n in args.channel])
      outcome = outcome and (args.dates[0] <= start and start < args.dates[1])
      if outcome:
        WAVEFORMS_DATA.append([fr, trc.network, trc.station, trc.channel,
                               UTCDateTime.strftime(start, DATE_FMT)])
  if not WAVEFORMS_DATA and not args.silent:
    # If no files were found in the specified directory, return an error
    # message and exit the program.
    print(f"""FATAL: No files which meet the following criteria:
         --network {args.network}
         --station {args.station}
         --channel {args.channel}
         --dates   {SPACE_STR.join([d.__str__() for d in args.dates])}
       were found in the specified directory: {args.directory}""")
    if args.key is None:
      print("HINT: If you want to download the data from the server, please "
            "specify the download option \"--download\" or provide a key file "
            "with the argument \"--key\" <key> for the specified client with "
            "the argument \"--client\" <client>")
    raise FileNotFoundError
  WAVEFORMS_DATA = \
    pd.DataFrame(WAVEFORMS_DATA, columns=HEADER).set_index(FILENAME_STR)
  WAVEFORMS_DATA.sort_values(by=[BEG_DATE_STR, FILENAME_STR], inplace=True)
  WAVEFORMS_DATA.to_csv(WAVEFORMS_FILE)
  return WAVEFORMS_DATA

@nb.njit(nogil=True)
def filter_data_(data : np.array) -> bool:
  for d in data:
    if np.isnan(d) or np.isinf(d): return True
  return False

@nb.jit()
def filter_data(data : np.array) -> bool:
  # if np.isnan(trc.data).any() or np.isinf(trc.data).any(): return True
  return filter_data_(data)

def clean_stream(stream : obspy.Stream, FMT_DICT : dict,
                 args : argparse.Namespace) -> obspy.Stream:
  """
  Clean the stream by resampling, merging, removing NaN and Inf values, and
  trim the stream to a single day. If the denoiser option is enabled, the
  stream will be denoised by the Deep Denoiser model.

  input:
    - stream        (obspy.Stream)
    - FMT_DICT      (dict)
    - args          (argparse.Namespace)

  output:
    - obspy.Stream

  errors:
    - None

  notes:
    TODO: Review the inplace operation of the Stream
  """
  global DATA_PATH
  DATA_PATH = Path(args.directory).parent
  if args.verbose: print("Cleaning the Stream")
  # Sample has to be 100 Hz
  stream = stream.resample(SAMPLING_RATE)
  stream.merge(method=1, fill_value='interpolate')
  for trc in stream:
    # Remove Stream.Trace if it contains NaN or Inf
    if filter_data(trc.data): stream.remove(trc)
  start = UTCDateTime.strptime(FMT_DICT[BEG_DATE_STR], DATE_FMT)
  stream = stream.trim(starttime=start, endtime=start + ONE_DAY, pad=True,
                       fill_value=0, nearest_sample=False)
  if args.denoiser:
    if args.verbose: print("Denoising the Stream")
    global DENOISER
    stream = DENOISER.annotate(stream)
  if args.verbose:
    IMG_FILE = Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                    PRC_FMT.format(NETWORK=FMT_DICT[NETWORK_STR],
                                   STATION=FMT_DICT[STATION_STR],
                                   CHANNEL=FMT_DICT[CHANNEL_STR],
                                   BEGDT=FMT_DICT[BEG_DATE_STR], EXT=EPS_STR))
    stream.plot(outfile=IMG_FILE, size=(1000, 600), format=EPS_STR, dpi=300)
  return stream

def primary_arguments(args : argparse.Namespace) -> dict:
  """
  Return the primary arguments from the specified file.

  input:
    - args          (argparse.Namespace)

  output:
    - dict

  errors:
    - None

  notes:

  """
  return {
    MODEL_STR     : args.models,
    WEIGHT_STR    : args.weights,
    NETWORK_STR   : args.network,
    STATION_STR   : args.station,
    CHANNEL_STR   : args.channel,
    BEG_DATE_STR  : [a.__str__() for a in args.dates],
    GROUPS_STR    : args.groups,
    DIRECTORY_STR : args.directory.__str__(),
    PWAVE         : args.pwave,
    SWAVE         : args.swave,
    JULIAN_STR    : args.julian,
    DENOISER_STR  : args.denoiser,
    DOMAIN_STR    : args.rectdomain if args.rectdomain else args.circdomain,
    CLIENT_STR    : args.client
  }

def read_arguments(args : argparse.Namespace, overwrite = False) -> dict:
  """
  Read the primary arguments from the arguments file and return the primary
  arguments dictionary.

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
  if overwrite:
    # Save the arguments to a JSON file
    with open(ARGUMENTS_FILE, 'w') as fw:
      json.dump(primary_arguments(args), fw, indent=2)
  if not ARGUMENTS_FILE.exists() and not args.silent:
    print("FATAL: Arguments file not found:", ARGUMENTS_FILE)
    raise FileNotFoundError
  # Read the arguments from the JSON file
  with open(ARGUMENTS_FILE, 'r') as fr:
    return json.load(fr)

def read_traces(trace_files, args : argparse.Namespace) -> obspy.Stream:
  """
  Read the traces from the specified files and return a clean Stream.

  input:
    - trace_files   ()
    - args          (argparse.Namespace)

  output:
    - obspy.Stream

  errors:
    - None

  notes:

  """
  global DATA_PATH
  DATA_PATH = Path(args.directory).parent
  stream = obspy.Stream()
  FMT_DICT = {category : EMPTY_STR for category in [NETWORK_STR, STATION_STR,
                                                    CHANNEL_STR, BEG_DATE_STR]}
  for category in args.groups:
    FMT_DICT[category] = trace_files[category].unique()[0]
  for _, row in trace_files.iterrows():
    if args.verbose: print("Attempting to read from raw file:", row.name)
    if not row.name.exists() and not args.silent:
      # TODO: Download the file
      print("CRITICAL: File not found:", row.name)
      continue
    else:
      stream += obspy.read(row.name)
  # Clean the stream
  return clean_stream(stream, FMT_DICT, args)

def interactive_plot(stream : obspy.Stream, picks : sbu.PickList,
                     model_name : str, dataset_name) -> None:
  """
  Plot the Stream with the picks on the Stream.

  input:
    - stream        (obspy.Stream)
    - picks         (seisbench.util.PickList)
    - model_name    (str)
    - dataset_name  (str)

  output:

  errors:
    - None

  notes:

  """
  events = [(np.datetime64(pick.peak_time), pick.peak_value,
             ('b' if pick.phase == PWAVE else 'r')) for pick in picks]
  fig = stream.plot(handle=True, method='full', size=(3000, 1000))
  fig.suptitle(SPACE_STR.join([fig.get_suptitle(), model_name, dataset_name]),
               fontsize=24)
  for ax in fig.get_axes():
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
      item.set_fontsize(18)
    for p, a, c in events: ax.axvline(p, linestyle='--', color=c, alpha=a)
  fig.tight_layout()
  plt.show()

def classify_stream(categories : tuple, trace_files, MODELS : dict,
                    args : argparse.Namespace) -> None:
  """
  Classify the stream. If 'force' is set to True, the classification will be
  performed regardless of the existence of the file.

  input:
    - categories    (tuple)
    - trace_files   ()
    - MODELS        (dict)
    - args          (argparse.Namespace)

  output:

  errors:
    - None

  notes:

  """
  global DATA_PATH
  DATA_PATH = Path(args.directory).parent
  categories = [str(c) for c in categories]
  CLF_PATH = Path(DATA_PATH, CLF_STR, *categories)
  CLF_PATH.mkdir(parents=True, exist_ok=True)
  clf_files = \
    [(Path(CLF_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
           UNDERSCORE_STR.join([*categories, model_name, dataset_name]) + \
           PICKLE_EXT), model_name, dataset_name)
     for model_name, dataset_name in MODELS.keys()]
  if args.force:
    clf_found = []
  else:
    clf_found = [clf for clf in clf_files if clf[0].is_file()]
    clf_files = [clf for clf in clf_files if not clf[0].is_file()]
  if clf_files:
    stream = read_traces(trace_files, args)
    if args.verbose: print("Classifying the Stream")
    for CLF_FILE, model_name, dataset_name in clf_files:
      MODEL = MODELS[(model_name, dataset_name)]
      if MODEL is None: continue
      output = MODEL.classify(stream, batch_size=args.batch,
                              P_threshold=args.pwave,
                              S_threshold=args.swave).picks
      THRESHOLD_DICT = {PWAVE : args.pwave, SWAVE : args.swave}
      output = sbu.PickList([pick for pick in output if pick.peak_value > \
                             THRESHOLD_DICT[pick.phase]])
      with open(CLF_FILE, 'wb') as fp: pickle.dump(output, fp)
      if args.verbose:
        print(f"Classification results for model: {model_name}, with "
              f"preloaded weight: {dataset_name}, categorized by {categories}")
        print(output)
      if args.interactive:
        # TODO: Plot without blocking the execution of the pipeline
        interactive_plot(stream, output, model_name, dataset_name)
  for CLF_FILE, model_name, dataset_name in clf_found:
    with open(CLF_FILE, 'rb') as fp: output = pickle.load(fp)
    if args.verbose:
      print(f"Classification results for model: {model_name}, with "
            f"preloaded weight: {dataset_name}, categorized by {categories}")
      print(output)
    if args.interactive:
      stream = read_traces(trace_files, args)
      interactive_plot(stream, output, model_name, dataset_name)

def get_model(model_name : str, dataset_name : str, silent = False) \
      -> sbm.base.SeisBenchModel:
  """
  Given a 'model_name' trained on the 'dataset_name', return the associated
  testing model. If the model is not found, return None.

  input:
    - model_name    (str)
    - dataset_name  (str)
    - silent        (bool)

  output:
    - seisbench.models.base.SeisBenchModel

  errors:
    - None

  notes:

  """
  global GPU_RANK
  try:
    model = MODEL_WEIGHTS_DICT[model_name].from_pretrained(dataset_name)
  except:
    if not silent: print(f"WARNING: Pretrained weights '{dataset_name}' not "
                         f"found for model '{model_name}'")
    return None
  # Enable GPU calls if available
  if GPU_RANK >= 0: model.cuda()
  if not silent: print(model_name, model.weights_docstring)
  return model

def set_up(args : argparse.Namespace) -> dict:
  """
  Set up the environment for the pipeline based on the available computational
  resources.

  input:
    - args          (argparse.Namespace)

  output:
    - dict

  errors:
    - None

  notes:

  """
  global GPU_SIZE, GPU_RANK
  GPU_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 0
  global MPI_SIZE, MPI_RANK, MPI_COMM
  MPI_COMM = MPI.COMM_WORLD
  MPI_SIZE = MPI_COMM.Get_size()
  MPI_RANK = MPI_COMM.Get_rank()
  if MPI_RANK < GPU_SIZE: GPU_RANK = MPI_RANK % GPU_SIZE
  if args.verbose: print(f"Setting MPI {MPI_RANK} to " + \
                         (f"GPU {GPU_RANK}" if GPU_RANK >= 0 else "CPU"))
  torch.cuda.set_device(GPU_RANK)
  MODELS = None
  WAVEFORMS_DATA = None
  if MPI_RANK == 0:
    if args.verbose:
      print("MPI size:", MPI_SIZE)
      print("GPU size:", GPU_SIZE)
    MODELS = [(m, w) for m, w in itertools.product(args.models, args.weights)
              if get_model(m, w, True) is not None]
    WAVEFORMS_DATA = waveform_table(args)
  MODELS = MPI_COMM.bcast(MODELS, root=0)
  WAVEFORMS_DATA = MPI_COMM.bcast(WAVEFORMS_DATA, root=0)
  # Split the MODELS among the MPI processes
  num_models = len(MODELS)
  models_idx = num_models // MPI_SIZE
  rest_idx = num_models % MPI_SIZE

  # Determine the start and end indices for each process
  start_idx = MPI_RANK * models_idx + min(MPI_RANK, rest_idx)
  end_idx = start_idx + models_idx + (1 if MPI_RANK < rest_idx else 0)

  # Assign the models to the current process
  MODELS = MODELS[start_idx:end_idx]
  if args.verbose: print(f"Process {MPI_RANK} handles models {MODELS}")
  return {(model_name, dataset_name) :
            get_model(model_name, dataset_name, args.silent)
          for model_name, dataset_name in MODELS}, WAVEFORMS_DATA

def main(args : argparse.Namespace):
  MODELS, WAVEFORMS_DATA = set_up(args)
  if args.denoiser:
    global DENOISER
    DENOISER = get_model(DEEPDENOISER_STR, ORIGINAL_STR, args.silent)
  if args.train: # Train
    if args.verbose: print("Training the Model")
    # Generate a Dataset
    # Train the model
    # Save the model
  else: # Test
    if args.verbose: print("Testing the Model")
    if args.timing:
      TIMING = np.zeros(len(WAVEFORMS_DATA.groupby(args.groups)))
      i = 0
    for categories, trace_files in WAVEFORMS_DATA.groupby(args.groups):
      if args.timing: start_time = MPI.Wtime()
      # Classify the Stream
      classify_stream(categories, trace_files, MODELS, args)
      if args.timing:
        TIMING[i] = MPI.Wtime() - start_time
        i += 1
      torch.cuda.empty_cache()
    if args.timing:
      global MPI_COMM, MPI_RANK, MPI_SIZE
      TOTALS = np.zeros_like(TIMING)
      MPI_COMM.Reduce([TIMING, MPI.DOUBLE], [TOTALS, MPI.DOUBLE], op=MPI.SUM,
                      root=0)
      TOTALS = TOTALS / MPI_SIZE
      if MPI_RANK == 0:
        print(f"  Total time: {sum(TOTALS):.2f} s")
        print(f"Average time: {np.mean(TOTALS):.2f} s")
        print(f"    Variance: {np.var(TOTALS):.2f} s")
        print(f"Maximum time: {np.max(TOTALS):.2f} s")
        print(f"Minimum time: {np.min(TOTALS):.2f} s")
        print(f" Median time: {np.median(TOTALS):.2f} s")
  return

if __name__ == "__main__": main(parse_arguments())
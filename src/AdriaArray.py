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
                      help="Specify the date (YYMMDD) range to work with. If "
                           "files are not present")
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
  # TODO: Add verbose LEVEL
  parser.add_argument('-v', "--verbose", default=False, action='store_true')
  parser.add_argument("--client", default=[INGV_STR], type=str, required=False,
                      nargs='+', help="Client to download the data")
  parser.add_argument("--denoiser", default=False, action='store_true',
                      required=False,
                      help="Enable Deep Denoiser model to filter the noise "
                           "previous to run the Machine Learning base model")
  parser.add_argument("--domain", default=[44.5, 47, 10, 14], required=False,
                      metavar=EMPTY_STR, nargs=4, type=float,
                      help="Domain to download the data")
  parser.add_argument("--pyrocko", default=False, action='store_true',
                      help="Enable PyRocko calls")
  return parser.parse_args()

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
  if args.pyrocko:
    # We enable the option to use the PyRocko module to download the data as it
    # is more efficient than the ObsPy module by multithreading the download.
    import pyrocko as pr

  else:
    from obspy.clients.fdsn import Client
    from obspy.clients.fdsn.mass_downloader import \
      RectangularDomain, Restrictions, MassDownloader
    domain = RectangularDomain(minlatitude=args.domain[0],
                               maxlatitude=args.domain[1],
                               minlongitude=args.domain[2],
                               maxlongitude=args.domain[3])
    restrictions = Restrictions(starttime=args.dates[0], endtime=args.dates[1],
                                channel_priorities=args.channel,
                                network=args.network, station=args.station)
    for client in args.client:
      cl = Client(client)
      # NOTE: It is assumed that a single token file is applicable for all
      #       clients
      cl.set_eida_token(args.key, validate=True)
      mdl = MassDownloader(providers=[cl])
      mdl.download(domain, restrictions, mseed_storage=args.directory)

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
  if not ARGUMENTS_FILE.exists() or \
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
      return pd.read_csv(WAVEFORMS_FILE, index_col=FILENAME_STR)
  if args.verbose:
    print("Constructing the Table of Files")
  if args.key is not None:
    # If a key is provided, we download the data from the server and construct
    # the table of files based on the specified arguments.
    data_downloader(args)
  # Construct the table of files based on the specified arguments
  for trc_file in args.directory.iterdir():
    fr = Path(args.directory, trc_file)
    if fr.is_file():
      try:
        trc = obspy.read(fr, headonly=True)[0].stats
      except:
        continue
      start = UTCDateTime(trc.starttime.date)
      if trc.starttime.hour == 23: start += ONE_DAY
      end = start + ONE_DAY
      # We start by assuming the trace file meets all the criterias:
      # (network, station, channel, date range). If the trace file does meet
      # we save the metadata to the list of files.
      outcome = True
      # If the user has specified the network different from the wildcard, then
      # we check if the network of the trace file is in the list of networks,
      # otherwise we assume the user wants to analyze all downloaded networks.
      if args.network and args.network != ALL_WILDCHAR_STR:
        outcome = outcome and any([n == trc.network for n in args.network])
      if args.station and args.station != ALL_WILDCHAR_STR and outcome:
        outcome = outcome and any([n == trc.station for n in args.station])
      if args.channel and args.channel != ALL_WILDCHAR_STR and outcome:
        outcome = outcome and any([n == trc.channel for n in args.channel])
      outcome = outcome and (args.dates[0] <= start and end <= args.dates[1])
      if outcome:
        WAVEFORMS_DATA.append([fr, trc.network, trc.station, trc.channel,
                               UTCDateTime.strftime(start, DATE_FMT)])
  if WAVEFORMS_DATA == []:
    # If no files were found in the specified directory, return an error
    # message and exit the program.
    print(
      f"""FATAL: No files which meet the following criteria:
         --network {args.network}
         --station {args.station}
         --channel {args.channel}
         --dates   {SPACE_STR.join([d.__str__() for d in args.dates])}
       were found in the specified directory: {args.directory}""")
    if args.key is None:
      print("HINT: If you want to download the data from the server, please "
            "specify a key file with the argument \"--key\" <key>.")
    raise FileNotFoundError
  WAVEFORMS_DATA = \
    pd.DataFrame(WAVEFORMS_DATA, columns=HEADER).set_index(FILENAME_STR)
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

def clean_stream(stream : obspy.Stream, dataset_name : str, FMT_DICT : dict,
                 args : argparse.Namespace) -> obspy.Stream:
  """
  Clean the stream by resampling, merging, removing NaN and Inf values, and
  trim the stream to a single day. If the denoiser option is enabled, the
  stream will be denoised by the Deep Denoiser model.

  input:
    - stream        (obspy.Stream)
    - FMT_DICT      (dict)
    - args          (argparse.Namespace)
    - dataset_name  (str)

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
    denoiser = get_model(DEEPDENOISER_STR, dataset_name)
    if denoiser is not None: stream = denoiser.annotate(stream)
  if args.verbose:
    IMG_FILE = Path(IMG_PATH, PRC_FMT.format(NETWORK=FMT_DICT[NETWORK_STR],
                                             STATION=FMT_DICT[STATION_STR],
                                             CHANNEL=FMT_DICT[CHANNEL_STR],
                                             BEGDT=FMT_DICT[BEG_DATE_STR],
                                             EXT=EPS_STR))
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
    DOMAIN_STR    : args.domain,
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
  if not ARGUMENTS_FILE.exists():
    print("FATAL: Arguments file not found:", ARGUMENTS_FILE)
    raise FileNotFoundError
  # Read the arguments from the JSON file
  with open(ARGUMENTS_FILE, 'r') as fr:
    return json.load(fr)

def read_traces(trace_files, dataset_name : str, args : argparse.Namespace) \
  -> obspy.Stream:
  """
  Read the traces from the specified files and return a Stream. If the file has
  been previously processed, the Stream will be read from the processed file.

  input:
    - trace_files   ()
    - args          (argparse.Namespace)
    - dataset_name  (str)

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
  PRC_PATH = Path(DATA_PATH, PRC_STR)
  PRC_PATH.mkdir(parents=False, exist_ok=True)
  STRM_FILE = Path(PRC_PATH, PRC_FMT.format(NETWORK=FMT_DICT[NETWORK_STR],
                                            STATION=FMT_DICT[STATION_STR],
                                            CHANNEL=FMT_DICT[CHANNEL_STR],
                                            BEGDT=FMT_DICT[BEG_DATE_STR],
                                            EXT=MSEED_STR))
  if STRM_FILE.exists():
    if args.verbose:
      print("Found and reading previously processed file:", STRM_FILE)
    stream = obspy.read(STRM_FILE)
  else:
    for _, row in trace_files.iterrows():
      if args.verbose:
        print("Attempting to read from raw file:", row.name)
      if not row.name.exists():
        # TODO: Download the file
        print("CRITICAL: File not found:", row.name)
        continue
      else:
        stream += obspy.read(row.name)
    # Clean the stream
    stream = clean_stream(stream, dataset_name, FMT_DICT, args)
    stream.write(STRM_FILE, format=MSEED_STR)
  return stream

def classify_stream(categories : tuple, trace_files, model_name : str,
                    dataset_name : str, MODEL : sbm.base.SeisBenchModel,
                    args : argparse.Namespace, force = False) -> sbu.PickList:
  """
  Classify the stream based on the specified model and dataset. If 'force' is
  set to True, the classification will be performed regardless of the existence
  of the file.

  input:
    - categories    (tuple)
    - trace_files   ()
    - model_name    (str)
    - dataset_name  (str)
    - MODEL         (seisbench.models.base.SeisBenchModel)
    - args          (argparse.Namespace)
    - force         (bool)

  output:
    - seisbench.util.PickList

  errors:
    - None

  notes:

  """
  global DATA_PATH
  DATA_PATH = Path(args.directory).parent
  CLF_PATH = Path(DATA_PATH, CLF_STR, *categories)
  CLF_PATH.mkdir(parents=True, exist_ok=True)
  CLF_FILE = Path(CLF_PATH, UNDERSCORE_STR.join([*categories, model_name,
                                                 dataset_name]) + PICKLE_EXT)
  if not force and CLF_FILE.is_file():
    if args.verbose:
      print("Found and loading previously classified results:", CLF_FILE)
    with open(CLF_FILE, 'rb') as fr:
      output = pickle.load(fr)
  else:
    # Read or download all the involved data (waveforms / traces) and the
    # collection of traces is called a "stream"
    stream = read_traces(trace_files, dataset_name, args)
    if args.verbose: print("Classifying the Stream")
    output = MODEL.classify(stream, batch_size=args.batch,
                            P_threshold=args.pwave,
                            S_threshold=args.swave).picks
    with open(CLF_FILE, 'wb') as fp: pickle.dump(output, fp)
  if args.verbose:
    print(f"Classification results for model: {model_name}, with preloaded "
          f"weight: {dataset_name}, categorized by {categories}")
    print(output)
  return output

def get_model(model_name : str, dataset_name : str) -> sbm.base.SeisBenchModel:
  """
  Given a 'model_name' trained on the 'dataset_name', return the associated
  testing model. If the model is not found, return None.

  input:
    - model_name    (str)
    - dataset_name  (str)

  output:
    - seisbench.models.base.SeisBenchModel

  errors:
    - None

  notes:

  """
  try:
    model = MODEL_WEIGHTS_DICT[model_name].from_pretrained(dataset_name)
  except:
    print(f"WARNING: Pretrained weights {dataset_name} not found for model "
          f"{model_name}")
    return None
  # Enable GPU calls if available
  if torch.cuda.is_available(): model.cuda()
  print(model_name, model.weights_docstring)
  return model

def main(args : argparse.Namespace):
  WAVEFORMS_DATA = waveform_table(args)
  if args.train: # Train
    if args.verbose: print("Training the Model")
    # Generate a Dataset
    # Train the model
    # Save the model
  else: # Test
    if args.verbose: print("Testing the Model")
    for model_name, dataset_name in list(itertools.product(args.models,
                                                           args.weights)):
      MODEL = get_model(model_name, dataset_name)
      if MODEL is None: continue
      for categories, trace_files in WAVEFORMS_DATA.groupby(args.groups):
        # Classify the Stream
        output = classify_stream(categories, trace_files, model_name,
                                 dataset_name, MODEL, args)
  return

if __name__ == "__main__": main(parse_arguments())
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

# TODO: Fix the following WARNINGS
# Parts of the input stream consist of fragments shorter than the number of input samples. Output might be empty.
# Detected multiple records for the same time and component that did not agree. All mismatching traces will be ignored.
# Parts of the input stream consist of fragments shorter than the number of input samples or misaligned traces. Output might be empty.
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
                      help="Batch size for the annotation process")
  parser.add_argument('-d', "--directory", required=False, type=is_dir_path,
                      default=Path(DATA_PATH, "waveforms"),
                      help="Directory path to the raw files")
  parser.add_argument('-p', "--pwave", default=PWAVE_THRESHOLD, type=float,
                      required=False, help="P wave threshold.")
  parser.add_argument('-s', "--swave", default=SWAVE_THRESHOLD, type=float,
                      required=False, help="S wave threshold.")
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
  parser.add_argument("--download", default=False, action='store_true',
                      help="Download the data from the server.")
  parser.add_argument("--pyrocko", default=False, action='store_true',
                      help="Enable PyRocko calls")
  return parser.parse_args()

def data_downloader(args : argparse.Namespace) -> list:
  """
  Download the data from the server
  input:
    - args          (argparse.Namespace)

  output:
    - list

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
  return []

def waveform_table(args : argparse.Namespace):
  """
  Construct a table of files based on the specified arguments. If the download
  option is enabled, the data will be downloaded directly from the server and
  replace all existing files in the directory.
  TODO: Consider generating a catalog of the downloaded data for future use.
  input:
    - args        (argparse.Namespace)

  output:
    - pandas.DataFrame

  errors:
    - None

  notes:
    If the starttime of the trace is 23:00 hrs, then we assume the date to be
    recorded is the next day
  """
  if args.verbose: print("Constructing the Table of Files")
  WAVEFORMS_DATA = list()
  if args.download: WAVEFORMS_DATA = data_downloader(args)
  else:
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
        outcome = True
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
    if WAVEFORMS_DATA == [] and args.network is not None and \
       args.station is not None and args.channel is not None:
      # If no files are found in the specified directory, download the data
      # only if the user has specified the network, station, channel and date
      # range, as these are the minimum requirements to download the data.
      data_downloader(args)
  return pd.DataFrame(WAVEFORMS_DATA, columns=HEADER).set_index(FILENAME_STR)\
                                                     .groupby(args.groups)

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
  for trc in stream:
    # Remove Stream.Trace if it contains NaN or Inf
    if filter_data(trc.data): stream.remove(trc)
  # Sample has to be 100 Hz
  stream = stream.resample(SAMPLING_RATE)
  start = UTCDateTime.strptime(FMT_DICT[BEG_DATE_STR], DATE_FMT)
  stream = stream.trim(starttime=start, endtime=start + ONE_DAY)
  if args.denoiser:
    if args.verbose: print("Denoising the Stream")
    denoiser = get_model(DEEPDENOISER_STR, dataset_name)
    if denoiser is not None: stream = denoiser.annotate(stream)
  if args.verbose:
    stream.plot(outfile=Path(IMG_PATH,
                             PRC_FMT.format(NETWORK=FMT_DICT[NETWORK_STR],
                                            STATION=FMT_DICT[STATION_STR],
                                            CHANNEL=FMT_DICT[CHANNEL_STR],
                                            BEGDT=FMT_DICT[BEG_DATE_STR],
                                            EXT=EPS_EXT)),
                size=(1000, 600), format=EPS_EXT, dpi=300)
  return stream

def read_traces(trace_files, dataset_name : str, args : argparse.Namespace) ->\
    obspy.Stream:
  """
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
  DATA_PATH = args.directory.parent
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
                                            EXT=MSEED_EXT))
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

def annotate_stream(categories : tuple, trace_files, model_name : str,
                    dataset_name : str, MODEL : sbm.base.SeisBenchModel,
                    args : argparse.Namespace) -> sbu.PickList:
  """
  input:
    - categories    (tuple)
    - trace_files   ()
    - model_name    (str)
    - dataset_name  (str)
    - MODEL         (seisbench.models.base.SeisBenchModel)
    - args          (argparse.Namespace)

  output:
    - seisbench.util.PickList

  errors:
    - None

  notes:

  """
  global DATA_PATH
  DATA_PATH = Path(args.directory).parent
  ANT_PATH = Path(DATA_PATH, ANT_STR, *categories)
  ANT_PATH.mkdir(parents=True, exist_ok=True)
  ANT_FILE = Path(ANT_PATH, "_".join([*categories, model_name, dataset_name]) \
                  + "." + PICKLE_EXT)
  if ANT_FILE.is_file():
    if args.verbose:
      print("Found and loading previously annotated results:", ANT_FILE)
    output = sbu.PickList()
    with open(ANT_FILE, 'rb') as fr:
      while True:
        try:
          output += pickle.load(fr)
        except EOFError:
          break
  else:
    # Read or download all the involved data (waveforms / traces) and the
    # collection of traces is called a "stream"
    stream = read_traces(trace_files, dataset_name, args)
    if args.verbose: print("Annotating the Stream")
    output = MODEL.annotate(stream, batch_size=args.batch,
                            P_threshold=args.pwave,
                            S_threshold=args.swave)
    with open(ANT_FILE, 'wb') as fp: pickle.dump(output, fp)
  if args.verbose:
    print(f"Annotation results for model: {model_name}, with preloaded "
          f"weight: {dataset_name}, categorized by {categories}")
    print(output)
  return output

def get_model(model_name : str, dataset_name : str) -> sbm.base.SeisBenchModel:
  """
  Given a model_name trained on the dataset_name, return the associated testing
  model.

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
      for categories, trace_files in WAVEFORMS_DATA:
        # Annotation
        output = annotate_stream(categories, trace_files, model_name,
                                 dataset_name, MODEL, args)
        # Annotation
  return

if __name__ == "__main__": main(parse_arguments())
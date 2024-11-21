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

from constants import *
import initializer as ini

# TODO: Study GaMMA associator with folder
# TODO: Colab PyOcto associator to be tested with GaMMA
# TODO: Get Vel Model
# TODO: Discuss constants.NORM = "peak"

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
  stream.resample(SAMPLING_RATE)
  stream.merge(method=1, fill_value='interpolate')
  # TODO: Consider using the Stream.detrend() method
  for trc in stream:
    # Remove Stream.Trace if it contains NaN or Inf
    if filter_data(trc.data): stream.remove(trc)
  start = UTCDateTime.strptime(FMT_DICT[BEG_DATE_STR], DATE_FMT)
  stream.trim(starttime=start, endtime=start + ONE_DAY, pad=True, fill_value=0,
              nearest_sample=False)
  if args.denoiser:
    if args.verbose: print("Denoising the Stream")
    global DENOISER
    stream = DENOISER.annotate(stream, copy=False)
  # TODO: Implement interactive plot
  if args.interactive: pass
  return stream

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
  fig = stream.plot(handle=True, method='full', size=(3000, 1000),
                    equal_scale=False)
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
    WAVEFORMS_DATA = ini.waveform_table(args)
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

def main(args : argparse.Namespace) -> None:
  if args.download:
    data_downloader(args)
    return
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

if __name__ == "__main__": main(ini.parse_arguments())
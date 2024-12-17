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
import torch
import pickle
import argparse
import itertools
import numpy as np
import numba as nb
import pandas as pd
from mpi4py import MPI
from threading import Thread
import matplotlib.pyplot as plt

# ObsPy
import obspy
from obspy.core.utcdatetime import UTCDateTime

# SeisBench
PROGRAM_NAME = "SeisBench"
import seisbench.util as sbu
import seisbench.models as sbm

from constants import *
import downloader as dwn
import initializer as ini

# TODO: Colab PyOcto associator to be tested with GaMMA
# TODO: Discuss constants.NORM = "peak"
# TODO: Create a Directory structure generator for the output

MPI_RANK = 0
MPI_SIZE = 1
MPI_COMM = None
GPU_RANK = -1
GPU_SIZE = 0

WORKFOLDER = DIRSTRUCT(Path(DATA_PATH, CLF_STR))

DENOISER = None

@nb.njit(nogil=True)
def filter_data_(data : np.array) -> bool:
  for d in data:
    if np.isnan(d) or np.isinf(d): return True
  return False

@nb.jit()
def filter_data(data : np.array) -> bool:
  # if np.isnan(trc.data).any() or np.isinf(trc.data).any(): return True
  return filter_data_(data)

def clean_stream(stream : obspy.Stream, FMT_DICT : dict[str, str],
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

def read_traces(trace_files : pd.DataFrame, args : argparse.Namespace) \
    -> obspy.Stream:
  """
  Read the traces from the specified files and return a clean Stream.

  input:
    - trace_files   (pd.DataFrame)
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
  FMT_DICT : dict[str, str] = {category : EMPTY_STR
                               for category in [NETWORK_STR, STATION_STR,
                                                CHANNEL_STR, BEG_DATE_STR]}
  for category in args.groups:
    FMT_DICT[category] = trace_files[category].unique()[0]
  for _, row in trace_files.iterrows():
    if args.verbose: print("Attempting to read from raw file:", row.name)
    if not Path(row.name).exists() and not args.silent:
      # TODO: Download the file
      print("CRITICAL: File not found:", row.name)
      continue
    else:
      stream += obspy.read(row.name)
  # Clean the stream
  return clean_stream(stream, FMT_DICT, args)

def interactive_plot(stream : obspy.Stream, picks : sbu.PickList,
                     model_name : str, dataset_name : str) -> None:
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

def classify_stream(clf_files : tuple[list], model : sbm.base.SeisBenchModel,
                    key : tuple[str], args : argparse.Namespace) -> None:
  """
  Classify the stream. If 'force' is set to True, the classification will be
  performed regardless of the existence of the file.

  input:
    - clf_files     (tuple)
    - model         (seisbench.models.base.SeisBenchModel)
    - key           (tuple)
    - args          (argparse.Namespace)

  output:

  errors:
    - None

  notes:

  """
  global DATA_PATH
  DATA_PATH = Path(args.directory).parent
  for categories, trace_files in clf_files:
    output = model.classify(read_traces(trace_files, args),
                            batch_size=args.batch, P_threshold=args.pwave,
                            S_threshold=args.swave).picks
    CLF_FILE = Path(DATA_PATH, CLF_STR, *categories, 
                    ("D_" if args.denoiser else EMPTY_STR) + \
                    UNDERSCORE_STR.join([*categories, *key]) + \
                    PICKLE_EXT)
    CLF_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CLF_FILE, 'wb') as fp: pickle.dump(output, fp)

def get_model(model_name : str, dataset_name : str, silent : bool = False) \
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

def set_up(args : argparse.Namespace) \
    -> list[dict[tuple[str], sbm.base.SeisBenchModel], pd.DataFrame]:
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
    dwn.data_downloader(args)
    return
  MODELS, WAVEFORMS_DATA = set_up(args)
  if args.denoiser:
    global DENOISER
    DENOISER = sbm.DeepDenoiser(sampling_rate=SAMPLING_RATE)
  if args.train: # Train
    if args.verbose: print("Training the Model")
    # Generate a Dataset
    # Train the model
    # Save the model
  else: # Test
    if args.verbose: print("Testing the Model")
    if args.timing:
      # TODO: Fix timing analysis
      TIMING = np.zeros(len(WAVEFORMS_DATA.groupby(args.groups)))
      i = 0
    for key, model in MODELS.items():
      key : list[str] = list(key)
      if model is None: continue
      if args.verbose:
        print("Testing model: {}, with preloaded weight: {}".format(*key))
      clf_files = []
      clf_found = []
      for categories, trace_files in WAVEFORMS_DATA.groupby(args.groups):
        categories = [str(c) for c in categories]
        CLF_FILE = Path(DATA_PATH, CLF_STR, *categories, 
                        ("D_" if args.denoiser else EMPTY_STR) + \
                        UNDERSCORE_STR.join([*categories, *key]) + PICKLE_EXT)
        if not args.force and CLF_FILE.exists():
          clf_found.append((categories, trace_files))
        clf_files.append((categories, trace_files))
      if clf_files:
        # P1
        if args.timing: start_time = MPI.Wtime()
        classify_stream(clf_files, model, key, args)
        if args.timing:
          TIMING[i] += MPI.Wtime() - start_time
        i += 1
      """
      # TODO: Spawn two threads and synchronize them
      def classify_and_plot():
        classify_stream(clf_files, model, key, args)
        for categories, trace_files in clf_files:
          if args.verbose:
        print("Classification results for model: {}, with preloaded weight: "
          "{}, categorized by {}".format(*key, categories))
        CLF_FILE = Path(DATA_PATH, CLF_STR, *categories,
                ("D_" if args.denoiser else EMPTY_STR) + \
                UNDERSCORE_STR.join([*categories, *key]) + \
                PICKLE_EXT)
        with open(CLF_FILE, 'rb') as fp: output = pickle.load(fp)
        print(output)
          if args.interactive:
        stream = read_traces(trace_files, args)
        interactive_plot(stream, output, *key)

      thread1 = Thread(target=classify_and_plot)
      thread2 = Thread(target=classify_and_plot)

      thread1.start()
      thread2.start()

      thread1.join()
      thread2.join()
      """
      # Clear the GPU memory (nowait)
      torch.cuda.empty_cache()
      # P2
      if args.verbose:
        for categories, trace_files in clf_found:
          print("Classification results for model: {}, with preloaded weight: "
                "{}, categorized by {}".format(*key, categories))
          CLF_FILE = Path(DATA_PATH, CLF_STR, *categories, 
                          ("D_" if args.denoiser else EMPTY_STR) + \
                          UNDERSCORE_STR.join([*categories, *key]) + \
                          PICKLE_EXT)
          with open(CLF_FILE, 'rb') as fp: output = pickle.load(fp)
          print(output)
          if args.interactive:
            stream = read_traces(trace_files, args)
            interactive_plot(stream, output, *key)
      # synchronize
      if args.verbose:
        for categories, trace_files in clf_files:
          print("Classification results for model: {}, with preloaded weight: "
                "{}, categorized by {}".format(*key, categories))
          CLF_FILE = Path(DATA_PATH, CLF_STR, *categories, 
                          ("D_" if args.denoiser else EMPTY_STR) + \
                          UNDERSCORE_STR.join([*categories, *key]) + \
                          PICKLE_EXT)
          with open(CLF_FILE, 'rb') as fp: output = pickle.load(fp)
          print(output)
          if args.interactive:
            stream = read_traces(trace_files, args)
            interactive_plot(stream, output, *key)
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
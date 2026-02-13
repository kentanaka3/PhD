import seisbench.models as sbm
import seisbench.util as sbu
from obspy.core.utcdatetime import UTCDateTime
import obspy
import matplotlib.pyplot as plt
from mpi4py import MPI
import pandas as pd
import numba as nb
import numpy as np
import itertools
import argparse
import pickle
import torch
import sys
from pathlib import Path
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Set the project folder
PRJ_PATH = Path(os.path.dirname(__file__)).parent
INC_PATH = os.path.join(PRJ_PATH, "inc")
IMG_PATH = os.path.join(PRJ_PATH, "img")
DATA_PATH = os.path.join(PRJ_PATH, "data")

# Add to path
if INC_PATH not in sys.path:
  sys.path.append(INC_PATH)
  from resources.constants import *
  import initializer as ini
else:
  from resources.constants import *
  import inc.initializer as ini

# ObsPy

# SeisBench
PROGRAM_NAME = "SeisBench"


# TODO: Colab PyOcto associator to be tested with GaMMA
# TODO: Discuss constants.NORM = "peak"
# TODO: Create a Directory structure generator for the output

MPI_RANK = 0
MPI_SIZE = 1
MPI_COMM = None
GPU_RANK = -1
GPU_SIZE = 0

DENOISER = None

DATES = None


@nb.njit(nogil=True)
def filter_data_(data: np.array) -> bool:
  """
  Check if the array contains NaN or Inf values. This function uses Numba
  for JIT-compilation and is run without the Global Interpreter Lock (GIL)
  to allow thread-level parallelism.

  input: data (np.array)
  output: bool - True if any element is NaN or Inf
  optimization: Numba's @njit with nogil enables parallel filtering in threads.
  """
  for d in data:
    if np.isnan(d) or np.isinf(d):
      return True
  return False


@nb.jit()
def filter_data(data: np.array) -> bool:
  # if np.isnan(trc.data).any() or np.isinf(trc.data).any(): return True
  """
  Wrapper for filter_data_ using JIT to further optimize execution.
  """
  return filter_data_(data)


def clean_stream(
    stream: obspy.Stream, FMT_DICT: dict[str, str], args: argparse.Namespace
) -> obspy.Stream:
  """
  Process seismic waveform streams by:
  1. Resampling to a standard rate
  2. Merging incomplete data
  3. Removing NaN or Inf traces
  4. Trimming to one day duration
  5. Denoising with optional model

  Parallelization:
  - JIT-compiled data filtering allows efficient execution.
  - Can be integrated with multithreaded stream processing per trace.

  input: stream (obspy.Stream), FMT_DICT (dict), args (argparse.Namespace)
  output: cleaned and optionally denoised obspy.Stream
  """
  # TODO: Model parameter for HIGHPASS filtering MODEL.FILTER_ARGS (present in
  #        SeisBench) Recommended for thesis
  global DATA_PATH
  DATA_PATH = args.directory.parent
  if args.verbose:
    print("Cleaning the Stream")
  # Sample has to be 100 Hz
  stream.resample(SAMPLING_RATE)
  stream.merge(method=1, fill_value="interpolate")
  # TODO: Consider using the Stream.detrend() method
  for trc in stream:
    # Remove Stream.Trace if it contains NaN or Inf
    if filter_data(trc.data):
      stream.remove(trc)
  start = UTCDateTime.strptime(FMT_DICT[DATE_STR], DATE_FMT)
  # TODO: Padding might add artifacts that may give bad results.
  stream.trim(starttime=start, endtime=start + ONE_DAY, pad=True, fill_value=0,
              nearest_sample=False)
  if args.denoiser:
    if args.verbose:
      print("Denoising the Stream")
    # TODO: Denoiser model doesnt seem to work properly
    stream = sbm.DeepDenoiser(sampling_rate=SAMPLING_RATE).annotate(
        stream, copy=False
    )
  # TODO: Implement interactive plot
  if args.interactive:
    pass
  return stream


def read_traces(trace_files: pd.DataFrame, args: argparse.Namespace) -> obspy.Stream:
  """
  Read waveform traces listed in a DataFrame, then clean the stream.

  input: trace_files (DataFrame with filenames), args (argparse.Namespace)
  output: obspy.Stream cleaned
  optimization: Sequential I/O but parallelizable by grouping and reading
  distinct trace sets on separate threads or MPI ranks.
  """
  global DATA_PATH
  DATA_PATH = args.directory.parent
  stream = obspy.Stream()
  FMT_DICT: dict[str, str] = {category: EMPTY_STR for category in [
      NETWORK_STR, STATION_STR, CHANNEL_STR, DATE_STR]
  }
  for category in [DATE_STR, NETWORK_STR, STATION_STR]:
    FMT_DICT[category] = trace_files[category].unique()[0]
  for _, row in trace_files.iterrows():
    tmp = ini.data_loader(Path(row.name))
    if tmp is None:
      print(f"WARNING: No data found for {row.name}")
      continue
    stream += tmp
  # Clean the stream
  return clean_stream(stream, FMT_DICT, args)


def interactive_plot(
    stream: obspy.Stream, picks: sbu.PickList, model_name: str, dataset_name: str
) -> None:
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
             ("b" if pick.phase == PWAVE else "r")) for pick in picks]
  fig = stream.plot(handle=True, method="full", size=(3000, 1000),
                    equal_scale=False)
  fig.suptitle(SPACE_STR.join([fig.get_suptitle(), model_name, dataset_name]),
               fontsize=24)
  for ax in fig.get_axes():
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                 + ax.get_xticklabels() + ax.get_yticklabels()):
      item.set_fontsize(18)
    for p, a, c in events:
      ax.axvline(p, linestyle="--", color=c, alpha=a)
  fig.tight_layout()
  plt.show()


def classify_stream(clf_files: list[tuple[tuple[str], pd.DataFrame]],
                    model: sbm.base.SeisBenchModel, key: tuple[str],
                    args: argparse.Namespace) -> None:
  """
  Classify waveform streams using a given model and store the picks.

  input: clf_files (grouped file tuples), model (SeisBench model),
          key (model + dataset), args (argparse)
  output: None (saves output to file)
  optimization:
  - Can be parallelized with MPI where each rank handles a different set
    of clf_files.
  - GPU acceleration enabled if available.
  """
  global DATA_PATH
  DATA_PATH = args.directory.parent
  for (date, network, station), trace_files in clf_files:
    fname = UNDERSCORE_STR.join([date, network, station, *key])
    CLF_FILE = Path(
        DATA_PATH, CLF_STR, *date.split(DASH_STR), network, station,
        ("D_" if args.denoiser else EMPTY_STR) + fname + PICKLE_EXT)
    CLF_FILE.parent.mkdir(parents=True, exist_ok=True)
    CLF_FILE.touch(exist_ok=True)
    output = model.classify(read_traces(trace_files, args),
                            batch_size=args.batch, P_threshold=args.pwave,
                            S_threshold=args.swave).picks
    with open(CLF_FILE, "wb") as fp:
      pickle.dump(output, fp)
    print(CLF_FILE)


def get_model(model_name: str, dataset_name: str,
              silent: bool = False) -> sbm.base.SeisBenchModel:
  """
  Retrieve a pretrained model.

  input: model_name (str), dataset_name (str), silent (bool)
  output: loaded model or None
  optimization: model.cuda() enables GPU computation for faster inference.
  """
  global GPU_RANK
  try:
    model = MODEL_WEIGHTS_DICT[model_name].from_pretrained(dataset_name)
  except:
    if not silent:
      print(f"WARNING: Pretrained weights '{dataset_name}' not "
            f"found for model '{model_name}'")
    return None
  # Enable GPU calls if available
  if torch.cuda.is_available() and GPU_RANK >= 0:
    model.cuda()
  elif torch.backends.mps.is_available() and GPU_RANK >= 0:
    model.to(torch.device("mps"))
  if not silent:
    print(model_name, model.weights_docstring)
  return model


def set_up(args: argparse.Namespace) \
        -> list[dict[tuple[str], sbm.base.SeisBenchModel], pd.DataFrame]:
  """
  Initialize MPI, GPU, and model assignments for parallel execution.

  input: args (argparse.Namespace)
  output: dictionary of models assigned to current MPI rank and waveform table
  optimization:
  - Uses MPI to divide models evenly across processes.
  - GPU usage coordinated per MPI rank (cuda or MPS backend).
  - Data distributed only once and broadcast.
  """
  global GPU_SIZE, GPU_RANK
  GPU_SIZE = 0
  global MPI_SIZE, MPI_RANK, MPI_COMM
  MPI_COMM = MPI.COMM_WORLD
  MPI_SIZE = MPI_COMM.Get_size()
  MPI_RANK = MPI_COMM.Get_rank()
  if torch.cuda.is_available():
    GPU_SIZE = torch.cuda.device_count()
    if MPI_RANK < GPU_SIZE:
      GPU_RANK = MPI_RANK % GPU_SIZE
    if args.verbose:
      print(f"Setting MPI {MPI_RANK} to " +
            (f"GPU {GPU_RANK}" if GPU_RANK >= 0 else "CPU"))
    torch.cuda.set_device(GPU_RANK)
  elif torch.backends.mps.is_available():
    GPU_SIZE = 1
    GPU_RANK = 0
    if args.verbose:
      print(f"Setting MPI {MPI_RANK} to GPU {GPU_RANK}")
    torch.device("mps")
  else:
    if args.verbose:
      print(f"Setting MPI {MPI_RANK} to CPU")
    GPU_RANK = -1
  MODELS = None
  WAVEFORMS = None
  if MPI_RANK == 0 and args.verbose:
    print("MPI size:", MPI_SIZE)
    print("GPU size:", GPU_SIZE)
  MODELS = [(m, w) for m, w in itertools.product(args.models, args.weights)
            if (m in MODEL_WEIGHTS_DICT if args.train else
                get_model(m, w, True)) is not None]
  WAVEFORMS = ini.waveform_table(args)
  MODELS = MPI_COMM.bcast(MODELS, root=0)
  WAVEFORMS = MPI_COMM.bcast(WAVEFORMS, root=0)
  # Split the MODELS among the MPI processes
  num_models = len(MODELS)
  models_idx = num_models // MPI_SIZE
  rest_idx = num_models % MPI_SIZE

  # Determine the start and end indices for each process
  start_idx = MPI_RANK * models_idx + min(MPI_RANK, rest_idx)
  end_idx = start_idx + models_idx + (1 if MPI_RANK < rest_idx else 0)

  # Assign the models to the current process
  MODELS = MODELS[start_idx:end_idx]
  if args.verbose:
    print(f"Process {MPI_RANK} handles models {MODELS}")
  return {
      (model_name, dataset_name): MODEL_WEIGHTS_DICT[model_name] if args.train
      else get_model(model_name, dataset_name, args.silent)
      for model_name, dataset_name in MODELS}, WAVEFORMS


def main(args: argparse.Namespace) -> None:
  """
  Orchestrates the full seismic phase picking pipeline.

  input: args (argparse.Namespace)
  output: None

  strategy:
  - If --download is set, data is downloaded and the program exits.
  - If --train is enabled, dataset preprocessing and training routines are executed.
  - Otherwise, models are loaded and applied to classify waveform data.

  optimization:
  - MPI ranks handle separate models (data parallelism)
  - GPU acceleration for model inference when available
  - Optional multi-threading for concurrent classification + visualization
  - Timed operations and statistics collection across MPI ranks
  """
  global DATA_PATH
  DATA_PATH = args.directory.parent
  if args.download:
    import downloader as dwn
    dwn.data_downloader(args)
    return
  MODELS, WAVEFORMS = set_up(args)
  if args.timing:
    TIMING = np.zeros(len(MODELS))
  if args.train:  # Train
    if args.verbose:
      print("Training the Model")
    assert len(args.weights) == 1
    args.weights = args.weights[0]
    import seisbench as sb
    import seisbench.data as sbd
    import seisbench.generate as sbg
    from torch.utils.data import DataLoader
    for i, (name, model) in enumerate(MODELS.items()):
      if not args.file:
        raise FileNotFoundError("Dataset not found")
      if len(args.file) > 1:
        raise NotImplementedError("Multiple files")
      if args.file[0]:
        global DATES
        start, end = args.dates
        if DATES is None:
          DATES = [start.datetime]
          while DATES[-1] < end.datetime:
            DATES.append(DATES[-1] + ONE_DAY)
        STATIONS = dict()
        WAVEFORMS[DATE_STR] = WAVEFORMS[DATE_STR].apply(
            lambda x: UTCDateTime.strptime(x, DATE_FMT)
        )
        for s, e in zip(DATES[:-1], DATES[1:]):
          S = WAVEFORMS.loc[
              WAVEFORMS[DATE_STR].between(s, e, inclusive="left"),
              STATION_STR].unique()
          if S.empty:
            continue
          STATIONS[s.strftime(DATE_FMT)] = set(S)
        SOURCE, DETECT = ini.true_loader(args)
      else:
        raise NotImplementedError
        print(
            "WARNING: obspy.clients.fdsn.header.FDSNNoDataException: "
            "No data available for request. HTTP Status code: 204"
        )
        from obspy.clients.fdsn import Client
        from obspy.core.event import Catalog

        CATALOG = Catalog()
        for client in args.client:
          if args.rectdomain:
            CATALOG += Client(client).get_events(
                *args.dates,
                minlongitude=args.rectdomain[0],
                maxlongitude=args.rectdomain[1],
                minlatitude=args.rectdomain[2],
                maxlatitude=args.rectdomain[3],
                includearrivals=True,
            )
          elif args.circdomain:
            CATALOG += Client(client).get_events(
                *args.dates,
                latitude=args.circdomain[0],
                longitude=args.circdomain[1],
                minradius=args.circdomain[2],
                maxradius=args.circdomain[3],
                includearrivals=True,
            )
        # TODO: Extract the SOURCE and DETECT from the CATALOG
      SOURCE = SOURCE[
          SOURCE[NOTES_STR].isnull() & SOURCE[LATITUDE_STR].notna()
      ].reset_index(drop=True)
      DETECT = DETECT[DETECT[ID_STR].isin(
          SOURCE[ID_STR])].reset_index(drop=True)
      name: list[str] = list(name)
      DATASET_PATH = Path(sb.cache_root, DATASETS_STR, name[-1])
      print(f"Creating {args.weights} dataset path:", DATASET_PATH)
      DATASET_PATH.mkdir(parents=True, exist_ok=True)
      METADATA_PATH = Path(DATASET_PATH, METADATA_STR + CSV_EXT)
      exit()
      if METADATA_PATH.exists():
        METADATA = pd.read_csv(METADATA_PATH)
        if args.verbose:
          print("Metadata:", METADATA)
      else:
        METADATA = pd.DataFrame(
            columns=[DATE_STR, NETWORK_STR, STATION_STR, CHANNEL_STR]
        )
      DATASET = sbd.WaveformDataset(DATASET_PATH, sampling_rate=SAMPLING_RATE)
      if args.verbose:
        print("Dataset Metadata:")
        print(DATASET.metadata)
      if args.verbose:
        print("Training model: {}, with weight name: {}".format(*name))
      # Generate a Dataset
      # Train the model
      # Save the model
  else:  # Test
    if args.verbose:
      print("Testing the Model")
    for i, (name, model) in enumerate(MODELS.items()):
      name: list[str] = list(name)
      if args.verbose:
        print("Testing model: {}, with preloaded weight: {}".format(*name))
      clf_files: list[tuple[tuple[str], pd.DataFrame]] = list()
      clf_found: list[tuple[tuple[str], pd.DataFrame]] = list()
      RERUN = list()
      for (date, network, station), trace_files in \
              WAVEFORMS.groupby([DATE_STR, NETWORK_STR, STATION_STR]):
        fname = UNDERSCORE_STR.join([date, network, station, *name])
        CLF_FILE = Path(
            DATA_PATH, CLF_STR, *date.split(DASH_STR), network, station,
            ("D_" if args.denoiser else EMPTY_STR) + fname + PICKLE_EXT)
        if not args.force and CLF_FILE.exists():
          clf_found.append(((date, network, station), trace_files))
        else:
          clf_files.append(((date, network, station), trace_files))
      # P1
      if clf_files:
        if args.timing:
          start_time = MPI.Wtime()
        classify_stream(clf_files, model, name, args)
        if args.timing:
          TIMING[i] += MPI.Wtime() - start_time
      """
      # TODO: Spawn two threads and synchronize them
      def classify_and_plot():
        classify_stream(clf_files, model, name, args)
        for categories, trace_files in clf_files:
          if args.verbose:
        print("Classification results for model: {}, with preloaded weight: "
          "{}, categorized by {}".format(*name, categories))
        CLF_FILE = Path(DATA_PATH, CLF_STR, *categories,
                ("D_" if args.denoiser else EMPTY_STR) + \
                UNDERSCORE_STR.join([*categories, *name]) + \
                PICKLE_EXT)
        with open(CLF_FILE, 'rb') as fp: output = pickle.load(fp)
        print(output)
          if args.interactive:
        stream = read_traces(trace_files, args)
        interactive_plot(stream, output, *name)

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

      # Optional: threaded classification + interactive plotting (disabled)
      # Concept: parallel plotting and classification in threads

      # Phase 2: Show cached results
      if args.verbose:
        for (date, network, station), trace_files in clf_found:
          print("Classification results for model: {}, with preloaded weight: "
                "{}".format(*name))
          fname = UNDERSCORE_STR.join([date, network, station, *name])
          CLF_FILE = Path(
              DATA_PATH, CLF_STR, *date.split(DASH_STR), network, station,
              ("D_" if args.denoiser else EMPTY_STR) + fname + PICKLE_EXT)
          try:
            with open(CLF_FILE, "rb") as fp:
              output = pickle.load(fp)
          except Exception as e:
            print("WARNING: ", e)
            RERUN.append(((date, network, station), trace_files))
            CLF_FILE.unlink()
            continue
          print(output)
          if args.interactive:
            interactive_plot(read_traces(trace_files, args), output, *name)
      # Phase 3: Retry failed traces
      if RERUN:
        classify_stream(RERUN, model, name, args)
      # Final interactive output and sync
      if args.verbose:
        for (date, network, station), trace_files in clf_files + RERUN:
          print("Classification results for model: {}, with preloaded weight: "
                "{}".format(*name))
          CLF_FILE = Path(
              DATA_PATH, CLF_STR, *date.split(DASH_STR), network, station,
              ("D_" if args.denoiser else EMPTY_STR)
              + UNDERSCORE_STR.join([date, network, station, *name])
              + PICKLE_EXT)
          try:
            with open(CLF_FILE, "rb") as fp:
              output = pickle.load(fp)
          except Exception as e:
            print("WARNING: ", e)
            os.remove(CLF_FILE)
            continue
          print(output)
          if args.interactive:
            interactive_plot(read_traces(trace_files, args), output, *name)
    if args.timing:
      global MPI_COMM, MPI_RANK, MPI_SIZE
      TOTALS = np.zeros_like(TIMING)
      if MPI_COMM is None:
        TOTALS = TIMING
      else:
        MPI_COMM.Reduce(
            [TIMING, MPI.DOUBLE], [TOTALS, MPI.DOUBLE], op=MPI.SUM, root=0
        )
      TOTALS = TOTALS / MPI_SIZE
      if MPI_RANK == 0:
        print(f"  Total time: {sum(TOTALS):.2f} s")
        print(f"Average time: {np.mean(TOTALS):.2f} s")
        print(f"    Variance: {np.var(TOTALS):.2f} s")
        print(f"Maximum time: {np.max(TOTALS):.2f} s")
        print(f"Minimum time: {np.min(TOTALS):.2f} s")
        print(f" Median time: {np.median(TOTALS):.2f} s")
  return


if __name__ == "__main__":
  main(ini.parse_arguments())

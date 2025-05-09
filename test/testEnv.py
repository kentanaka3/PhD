def main():
  print("[Hello World!]")
  # Check if the necessary packages are available
  # MPI
  MPI_COMM = None
  MPI_RANK = 0
  MPI_SIZE = 1
  try:
    from mpi4py import MPI
    MPI_COMM = MPI.COMM_WORLD
    MPI_RANK = MPI_COMM.Get_rank()
    MPI_SIZE = MPI_COMM.Get_size()
    print("[MPI] is available with rank:", MPI_RANK, "and size:", MPI_SIZE)
  except ImportError as e:
    print("[MPI] is NOT available", e)
  # PyTorch
  GPU_DEVICE = "cpu"
  GPU_SIZE = 0
  GPU_RANK = -1
  try:
    import torch
    if torch.cuda.is_available():
      GPU_DEVICE = "cuda"
      GPU_SIZE = torch.cuda.device_count()
      print("[PyTorch] Number of CUDA devices: ", GPU_SIZE)
      if MPI_RANK < GPU_SIZE:
        GPU_RANK = MPI_RANK % GPU_SIZE
        torch.cuda.set_device(GPU_RANK)
        print((GPU_RANK, torch.cuda.get_device_properties(GPU_RANK)))
      else:
        for i in range(GPU_SIZE):
          print((i, torch.cuda.get_device_properties(i)))
    elif torch.backends.mps.is_available():
      GPU_DEVICE = "mps"
      GPU_SIZE = 1
      GPU_RANK = 0
      print("[PyTorch] Number of MPS devices: ", GPU_SIZE)
  except ImportError as e:
    print("[PyTorch] is NOT available", e)
  torch.device(GPU_DEVICE)
  try:
    import obspy
    print("[ObsPy] is available")
  except ImportError as e:
    print("[ObsPy] is NOT available", e)
  try:
    import seisbench
    print("[Seisbench] is available")
  except:
    print("[Seisbench] is NOT available")
  try:
    import numba
    print(f"[Numba] is available with: {numba.get_num_threads()} threads")
  except:
    print("[Numba] is NOT available")
  try:
    import gamma
    print("[GaMMA] is available")
  except:
    print("[GaMMA] is NOT available")
  return


if __name__ == "__main__":
  main()

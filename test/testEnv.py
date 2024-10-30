def main():
  print("Hello World!")
  try:
    from mpi4py import MPI
    MPI_COMM = MPI.COMM_WORLD
    MPI_RANK = MPI_COMM.Get_rank()
    MPI_SIZE = MPI_COMM.Get_size()
    print("MPI is available with rank:", MPI_RANK, "and size:", MPI_SIZE)
    try:
      import torch
      GPU_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 0
      print("Number of CUDA devices: ", GPU_SIZE)
      if MPI_RANK < GPU_SIZE:
        GPU_RANK = MPI_RANK % GPU_SIZE
        torch.cuda.set_device(GPU_RANK)
        print((GPU_RANK, torch.cuda.get_device_properties(GPU_RANK)))
      else:
        for i in range(GPU_SIZE):
          print((i, torch.cuda.get_device_properties(i)))
    except ImportError:
      print("PyTorch is NOT available")
  except:
    print("MPI is NOT available")
    try:
      import torch
      GPU_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 0
      print("Number of CUDA devices: ", GPU_SIZE)
      for i in range(GPU_SIZE): print((i, torch.cuda.get_device_properties(i)))
    except ImportError:
      print("PyTorch is NOT available")
  try:
    import obspy
    print("ObsPy is available")
  except ImportError:
    print("ObsPy is NOT available")
  try:
    import seisbench
    print("Seisbench is available")
  except:
    print("Seisbench is NOT available")
  try:
    import numba
    print(f"Numba is available with: {numba.get_num_threads()} threads")
  except:
    print("Numba is NOT available")
  return

if __name__ == "__main__": main()


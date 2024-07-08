def main():
  print("Hello World!")
  try:
    import torch
    print("CUDA availability: ", torch.cuda.is_available())
    for i in range(torch.cuda.device_count()):
      print((i, torch.cuda.get_device_properties(i)))
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
    print("Numba is available with:", numba.config.NUMBA_DEFAULT_NUM_THREADS,
          "threads")
  except:
    print("Numba is NOT available")
  return

if __name__ == "__main__": main()

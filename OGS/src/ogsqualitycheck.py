import os
import argparse
import obspy as op
from pathlib import Path

import ogsconstants as OGS_C

DATA_PATH = Path(__file__).parent.parent.parent

def is_dir_path(string: str) -> Path:
  if os.path.isdir(string):
    return Path(os.path.abspath(string))
  else:
    raise NotADirectoryError(string)

def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Quality check OGS waveforms")
  parser.add_argument('-d', "--directory", required=False, type=is_dir_path,
                      default=Path(DATA_PATH, OGS_C.WAVEFORMS_STR),
                      help="Directory path to the raw files")
  return parser.parse_args()


def data_check(args: argparse.Namespace) -> None:
  for waveform in args.directory.glob("*/*/*/*.mseed"):
    st = op.read(waveform)
    if len(st[0]) < 1000:
      os.remove(waveform)
      print(f"Removed {waveform} with {len(st[0])} samples")

if __name__ == "__main__": data_check(parse_arguments())
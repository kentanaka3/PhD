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

class StationChannel:
  def __init__(self, network: str, station: str, location: str, channel: str) -> None:
    self.network = network
    self.station = station
    self.location = location
    self.channel = channel

  def __repr__(self) -> str:
    return f"{self.network}.{self.station}.{self.location}.{self.channel}"

def data_check(args: argparse.Namespace) -> None:
  STATIONS_RM : list[StationChannel] = []
  for waveform in args.directory.glob("*/*/*/*.mseed"):
    st = op.read(waveform)
    if len(st[0]) < 1000:
      os.remove(waveform)
      print(f"Removed {waveform} with {len(st[0])} samples")
      STATIONS_RM.append(
        StationChannel(st[0].stats.network, st[0].stats.station,
                       st[0].stats.location, st[0].stats.channel))
    if st[0].stats.sampling_rate != 100:
      os.remove(waveform)
      print(f"Removed {waveform} with {st[0].stats.sampling_rate} Hz sampling rate")
      st[0].resample(100).write(waveform, format="MSEED")

if __name__ == "__main__": data_check(parse_arguments())
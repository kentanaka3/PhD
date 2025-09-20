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

STATIONS_RM : list[StationChannel] = []
def removeFile(filePath: Path, st) -> None:
  os.remove(filePath)
  if st is not None:
    STATIONS_RM.append(
      StationChannel(st.network, st.station, st.location, st.channel))

def data_check(args: argparse.Namespace) -> None:
  for waveform in args.directory.glob("*/*/*/*.mseed"):
    try:
      st = op.read(waveform)
    except Exception as e:
      removeFile(waveform, None)
      print(f"Removed corrupted file {waveform} due to {e}")
      continue
    if len(st[0]) > 1000: continue
    if st[0].stats.sampling_rate == 100: continue
    wvfrmStream = op.read(waveform)
    removeFile(waveform, st=st[0].stats)
    wvfrmStream = wvfrmStream.resample(100)
    wvfrmStream.write(waveform)
    print(f"Resampled {waveform} from {st[0].stats.sampling_rate} Hz sampling rate")

if __name__ == "__main__": data_check(parse_arguments())
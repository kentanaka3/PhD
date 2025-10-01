import os
import argparse
import obspy as op
from pathlib import Path
from obspy import UTCDateTime

import ogsconstants as OGS_C

DATA_PATH = Path(__file__).parent.parent.parent

def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Quality check OGS waveforms")
  parser.add_argument(
    '-d', "--directory", required=False,type=OGS_C.is_dir_path,
    default=Path(DATA_PATH, OGS_C.WAVEFORMS_STR),
    help="Directory path to the raw files")
  return parser.parse_args()

def removeFile(filePath: Path, st) -> None:
  os.remove(filePath)

def data_check(args: argparse.Namespace) -> None:
  for waveform in args.directory.glob("**/*.mseed"):
    print(f"Checking {waveform}")
    try:
      wvfrmStream = op.read(waveform)
    except Exception as e:
      removeFile(waveform, None)
      print(f"Removed corrupted file {waveform} due to {e}")
      continue
    rate = wvfrmStream[0].stats.sampling_rate
    if rate == 100: continue
    removeFile(waveform, st=wvfrmStream[0].stats)
    wvfrmStream.resample(100)
    wvfrmStream.merge(method=1, fill_value=0)
    start = UTCDateTime(wvfrmStream[0].stats.starttime)
    if start.hour == 23: start = start.date + OGS_C.ONE_DAY
    else: start = start.date
    start = UTCDateTime.strptime(str(start), OGS_C.DATE_FMT)
    wvfrmStream.trim(starttime=start, endtime=start + OGS_C.ONE_DAY, pad=True,
                     fill_value=0, nearest_sample=False)
    wvfrmStream.write(waveform, format="MSEED")
    print(f"Resampled {waveform} from {rate} Hz sampling rate")

if __name__ == "__main__": data_check(parse_arguments())
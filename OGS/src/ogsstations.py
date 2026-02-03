
import argparse
from pathlib import Path
from datetime import datetime, timedelta as td

import ogsconstants as OGS_C
import ogsplotter as OGS_P

def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="OGS Stations Module",
  )
  parser.add_argument(
    "-s", "--stations", type=Path, required=True,
    help="Directory containing station files",
  )
  parser.add_argument(
    "-w", "--waveform", type=Path, required=True,
    help="Path to the waveform files",
  )
  parser.add_argument(
    "-D", "--dates", required=False, metavar=OGS_C.DATE_STD,
    type=OGS_C.is_date, nargs=2, action=OGS_C.SortDatesAction,
    default=[datetime.strptime("20240320", OGS_C.YYYYMMDD_FMT),
             datetime.strptime("20240620", OGS_C.YYYYMMDD_FMT)],
    help="Specify the beginning and ending (inclusive) Gregorian date " \
          "(YYYYMMDD) range to work with."
  )
  return parser.parse_args()

def main(args: argparse.Namespace) -> None:
  INVENTORY = OGS_C.inventory(args.directory)
  waveform = OGS_C.waveforms(args.waveform, args.dates[0], args.dates[1])

if __name__ == "__main__": main(parse_arguments())
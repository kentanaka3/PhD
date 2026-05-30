"""
=============================================================================
OGS Station Waveform Inventory Helper - CLI Entry Point
=============================================================================

OVERVIEW:
Thin command-line wrapper around :func:`ogsutils.waveforms` that discovers
and summarizes the per-station waveform inventory available under a source
directory over a given date window. The script is intentionally minimal so it
can be invoked from shell pipelines or SLURM job scripts.

USAGE:
    python ogsstation.py <src_root> <waveform_dir> <stations_dir> \
                         <YYYYMMDD_start> <YYYYMMDD_end> [<output_dir>]

Where:
    - ``src_root``       is prepended to ``sys.path`` so ``ogsutils`` resolves
    - ``waveform_dir``   directory tree containing daily waveform files
    - ``stations_dir``   station metadata directory
    - ``YYYYMMDD_*``     inclusive Gregorian date range to scan
    - ``output_dir``     optional output directory (default: current dir)

DEPENDENCIES:
    - ogsutils.waveforms: actual scanning / inventory logic

AUTHOR: AI2Seism Project
=============================================================================
"""

import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, sys.argv[1])
from ogsutils import waveforms

def main():
  output = Path(sys.argv[6]) if len(sys.argv) > 6 else Path(".")
  output.mkdir(parents=True, exist_ok=True)
  waveforms(
    Path(sys.argv[2]), Path(sys.argv[3]),
    datetime.strptime(sys.argv[4], '%Y%m%d'),
    datetime.strptime(sys.argv[5], '%Y%m%d'),
    output=output
  )

if __name__ == "__main__":  main()

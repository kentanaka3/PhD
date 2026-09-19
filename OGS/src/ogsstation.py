"""
=============================================================================
OGS Station Waveform Inventory Helper - CLI Entry Point
=============================================================================

OVERVIEW:
Command-line wrapper around :func:`ogsutils.waveforms` that discovers
and summarizes the per-station waveform inventory available under a source
directory over a given date window. Supports parallel multi-threaded scanning.

USAGE:
    python ogsstation.py <src_root>
                         -S <stations> \
                         -W <waveforms> \
                         [-D <YYYYMMDD> <YYYYMMDD>] \
                         [-o <output>] [--threads <N>]

Where:
    - ``src_root``         is prepended to ``sys.path`` so ``ogsutils`` resolves
    - ``-S / --stations``  station metadata directory
    - ``-W / --waveforms`` directory tree containing daily waveform files
    - ``-D / -J``          date range (Gregorian YYYYMMDD or Julian YYYYJJJ)
    - ``-o / --output``    optional output directory (default: current dir)
    - ``--threads``        optional worker thread count (default: SLURM/CPU count)

DEPENDENCIES:
    - ogsutils.waveforms: actual scanning / inventory logic

AUTHORS:
  - 健
  - Istituto Nazionale di Oceanografia e di Geofisica Sperimentale (OGS)
    Centro di Ricerche Sismologiche (CRS)
  - Università degli Studi di Trieste (UniTS)
    Dipartimento di Matematica, Informatica e Geoscienze (MIGe)
    Applied Data Science and Artificial Intelligence (ADSAI)
  - Terabit Network for Research and Academic Big Data in Italy (TeRABIT)
    Consorzio Interuniversitario del Nord-Est per il Calcolo Automatico (CINECA)
=============================================================================
"""

import ogsutils as OGS_U


def main():
  args = OGS_U.parse_station_args()

  args.output.mkdir(parents=True, exist_ok=True)
  OGS_U.waveforms(
      waveforms=args.waveforms,
      stations=args.stations,
      start=args.dates[0],
      end=args.dates[1],
      output=args.output,
  )


if __name__ == "__main__":
  main()

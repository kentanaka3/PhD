"""
===============================================================================
OGS Station Waveform Inventory Helper - CLI Entry Point
===============================================================================

OVERVIEW:
Command-line wrapper around :func:`ogsutils.waveforms` that discovers and
summarizes daily MiniSEED files under the waveform directory over an inclusive
date window. Scanning concurrency comes from shared constants, not a CLI
thread argument.

USAGE:
    python -m OGS.src.ogsstation <src_root> \
        -S <stations> \
        -W <waveforms> \
        [-D <YYYYMMDD> <YYYYMMDD>] \
        [-o <output>] [--threads <N>]

Where:
    - ``src_root``         required legacy parser positional argument; unused
                           by this entry point, which does not alter sys.path
    - ``-S / --stations``  station metadata directory
    - ``-W / --waveforms`` directory tree containing daily waveform files
    - ``-D / -J``          date range (Gregorian YYYYMMDD or Julian YYYYJJJ)
    - ``-o / --output``    optional output directory (default: current dir)

OUTPUT:
    Creates the output directory, then delegates CSVs (OGSWaveforms.csv,
    OGSInventory.csv), the station map (OGSStations.png), and, when data are
    available, the availability plot (OGSAvailability.png) to
    ogsutils.waveforms.
    Expected files are <waveforms>/YYYY/MM/DD/
    NET.STA.LOC.CHA__YYYYMMDDTHHMMSSZ__YYYYMMDDTHHMMSSZ.mseed.

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

===============================================================================
"""

from . import ogsutils as OGS_U


def main():
  args = OGS_U.parse_station_args()

  args.output.mkdir(parents=True, exist_ok=True)
  OGS_U.waveforms(
      waveforms=args.waveforms,
      stations=args.stations,
      start=args.dates[0],
      end=args.dates[1],
      output=args.output,
      threads=args.threads,
  )


if __name__ == "__main__":
  main()

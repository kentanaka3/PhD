import os
import argparse
import logging
import numpy as np
import obspy as op
from pathlib import Path
from datetime import datetime

import ogsconstants as OGS_C

# Project root (three levels above this file)
DATA_PATH = Path(__file__).parent.parent.parent

def is_path(string: str) -> Path:
  # Validate that a path exists and return it as an absolute Path
  if os.path.isfile(string) or os.path.isdir(string):
    return Path(os.path.abspath(string))
  else: raise FileNotFoundError(string)

def parse_arguments() -> argparse.Namespace:
  # Parse command-line arguments for dataset download
  parser = argparse.ArgumentParser(description="Process AdriaArray Dataset")
  # TODO: Handle security issues
  parser.add_argument(
    '-K', "--key", default=None, required=False, type=OGS_C.is_file_path,
    metavar=OGS_C.EMPTY_STR, help="Key to download the data from server.")
  parser.add_argument(
    '-N', "--network", default=[OGS_C.ALL_WILDCHAR_STR], type=str,
    nargs=OGS_C.ONE_MORECHAR_STR, metavar=OGS_C.EMPTY_STR, required=False,
    help="Specify a set of Networks to analyze. To allow downloading data for "
        f"any channel, set this option to \'{OGS_C.ALL_WILDCHAR_STR}\'.")
  parser.add_argument(
    '-S', "--station", default=[OGS_C.ALL_WILDCHAR_STR], type=str,
    nargs=OGS_C.ONE_MORECHAR_STR, metavar=OGS_C.EMPTY_STR, required=False,
    help="Specify a set of Stations to analyze. To allow downloading data for "
        f"any channel, set this option to \'{OGS_C.ALL_WILDCHAR_STR}\'.")
  parser.add_argument(
    "-c", "--clip", required=False, type=str, metavar="HHMMSS",
    help="Specify the time of the center time")
  parser.add_argument(
    '-d', "--directory", required=False, type=OGS_C.is_dir_path,
    default=Path(DATA_PATH, OGS_C.WAVEFORMS_STR), metavar=OGS_C.EMPTY_STR,
    help="Directory path to the raw files")
  parser.add_argument(
    "--client", metavar=OGS_C.EMPTY_STR, default=OGS_C.OGS_CLIENTS_DEFAULT,
    required=False, type=str, nargs=OGS_C.ONE_MORECHAR_STR,
    help="Client to download the data")
  parser.add_argument(
    "--force", default=False, action='store_true', required=False,
    help="Force running all the pipeline")
  parser.add_argument(
    "--pyrocko", default=False, action='store_true',
    help="Enable PyRocko calls")
  parser.add_argument(
    "--timing", default=False, action='store_true', required=False,
    help="Enable timing")
  date_group = parser.add_mutually_exclusive_group(required=False)
  date_group.add_argument(
    '-D', "--dates", required=False, metavar=OGS_C.DATE_STD,
    type=OGS_C.is_date, nargs=2, action=OGS_C.SortDatesAction,
    default=[datetime.strptime("20240320", OGS_C.YYYYMMDD_FMT),
             datetime.strptime("20240620", OGS_C.YYYYMMDD_FMT)],
    help="Specify the beginning and ending (inclusive) Gregorian date " \
         "(YYYYMMDD) range to work with.")
  date_group.add_argument(
    '-J', "--julian", required=False, metavar=OGS_C.DATE_STD,
    action=OGS_C.SortDatesAction, type=OGS_C.is_julian, default=None, nargs=2,
    help="Specify the beginning and ending (inclusive) Julian date (YYMMDD) " \
         "range to work with.")
  domain_group = parser.add_mutually_exclusive_group(required=False)
  domain_group.add_argument(
    "--rectdomain", type=float, nargs=4, default=OGS_C.OGS_STUDY_REGION,
    metavar=("lonW", "lonE", "latS", "latN"),
    help="Rectangular domain to download the data: [longitude West] "
         "[longitude East] [latitude South] [latitude North]")
  domain_group.add_argument(
    "--circdomain", nargs=4, type=float, # default=[12.808, 46.3583, 0., 0.3],
    metavar=("lon", "lat", "min_r", "max_r"),
    help="Circular domain to download the data: [center longitude] "
         "[center latitude] [minimum radius] [maximum radius]")
  verbal_group = parser.add_mutually_exclusive_group(required=False)
  verbal_group.add_argument(
    "--silent", default=False, action='store_true', help="Silent mode")
  # TODO: Add verbose LEVEL
  verbal_group.add_argument(
    "-v", "--verbose", default=False, action='store_true', help="Verbose mode")
  args = parser.parse_args()
  # TODO: Fix special cases
  # print(vars(args))
  return args

def _setup_logger(verbose: bool, silent: bool) -> logging.Logger:
  """Configure and return a module logger."""
  logger = logging.getLogger(__name__)
  # Avoid duplicate handlers when the module is imported multiple times
  if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
      fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
  # Choose verbosity level from CLI flags
  if silent:
    logger.setLevel(logging.WARNING)
  else:
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
  # Keep logs local to this module
  logger.propagate = False
  return logger

def data_downloader(args: argparse.Namespace) -> None:
  """
  Download the data from the server based on the specified arguments. If the
  data is already present in the directory, the data will be replaced by the
  new data.

  input:
    - args          (argparse.Namespace)

  output:
    - None

  errors:
    - None

  notes:

  """
  # Initialize logger based on CLI flags
  logger = _setup_logger(args.verbose, args.silent)

  if args.pyrocko:
    # We enable the option to use the PyRocko module to download the data as it
    # is more efficient than the ObsPy module by multithreading the download.
    import pyrocko as pr

  else:
    # Build the download domain (rectangular or circular)
    if args.rectdomain:
      from obspy.clients.fdsn.mass_downloader.domain import RectangularDomain
      domain = RectangularDomain(
        minlongitude=args.rectdomain[0], maxlongitude=args.rectdomain[1],
        minlatitude=args.rectdomain[2], maxlatitude=args.rectdomain[3])
    else:
      from obspy.clients.fdsn.mass_downloader.domain import CircularDomain
      domain = CircularDomain(
        latitude=args.circdomain[0], longitude=args.circdomain[1],
        minradius=args.circdomain[2], maxradius=args.circdomain[3])
    from obspy.clients.fdsn.mass_downloader import Restrictions, MassDownloader
    # Expand the requested date range into a list of day start times
    start, end = args.dates
    DAYS = np.arange(start, end + OGS_C.ONE_DAY, OGS_C.ONE_DAY,
                     dtype='datetime64[D]').tolist()
    # Directory naming convention for year/month/day
    DIR_FMT = {
      "year": "{:04}",
      "month": "{:02}",
      "day": "{:02}",
    }
    # Download day by day to keep file sizes manageable
    for d_ in DAYS:
      # Compute the time window centered at the optional clip time
      starttime = datetime.strptime(
        d_.strftime(OGS_C.DATE_FMT) + args.clip,
        OGS_C.DATE_FMT + OGS_C.TIME_FMT
      ) - OGS_C.PICK_TRAIN_OFFSET if args.clip else d_
      endtime = datetime.strptime(
        d_.strftime(OGS_C.DATE_FMT) + args.clip,
        OGS_C.DATE_FMT + OGS_C.TIME_FMT
      ) + OGS_C.PICK_TRAIN_OFFSET if args.clip else d_ + OGS_C.ONE_DAY
      # Build the output directory for the current day
      D_FILE = Path(
        args.directory /
        DIR_FMT["year"].format(d_.year) /
        DIR_FMT["month"].format(d_.month) /
        DIR_FMT["day"].format(d_.day)
      )
      D_FILE.mkdir(parents=True, exist_ok=True)
      logger.info("Downloading the data in the directory: %s", D_FILE)
      # Apply selection constraints to reduce data volume
      restrictions = Restrictions(
        starttime=starttime, endtime=endtime,
        network=OGS_C.COMMA_STR.join(args.network),
        station=OGS_C.COMMA_STR.join(args.station),
        channel_priorities=["HH[ZNE]", "EH[ZNE]", "HN[ZNE]", "HG[ZNE]"],
        reject_channels_with_gaps=False, minimum_length=0.0,
        minimum_interstation_distance_in_m=100,
        location_priorities=["", "00", "01", "02", "10"],
        chunklength_in_sec=86400
      )
      from obspy.clients.fdsn import Client
      # Instantiate all requested FDSN clients
      CLIENTS: dict[str, Client] = dict()
      for client in args.client:
        try: CLIENTS[client] = Client(client)
        except Exception as e:
          logger.error("Error creating client %s: %s", client, e)
          continue
        if args.key and client in [
          OGS_C.INGV_CLIENT_STR,
          OGS_C.GFZ_CLIENT_STR
        ]:
          # NOTE: It is assumed a single token file is applicable for all
          #       clients
          try: CLIENTS[client].set_eida_token(args.key, validate=True)
          except Exception as e:
            logger.error("Error setting token for %s: %s", client, e)
      # Execute the mass download for the day
      mdl = MassDownloader(providers=CLIENTS.values())
      try:
        mdl.download(domain, restrictions, mseed_storage=D_FILE.__str__(),
                     stationxml_storage=Path(args.directory.parent,
                                             OGS_C.STATION_STR).__str__())
      except Exception as e:
        logger.error("Error downloading data: %s", e)
      # Report completion per-day
      logger.info("Downloaded data for date: %s",
                  d_.strftime(OGS_C.YYYYMMDD_FMT))

if __name__ == "__main__": data_downloader(parse_arguments())
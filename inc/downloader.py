import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from pathlib import Path
# Set the project folder
PRJ_PATH = Path(os.path.dirname(__file__)).parent
DATA_PATH = Path(PRJ_PATH, "data")

import argparse

from constants import *
import initializer as ini

def data_downloader(args : argparse.Namespace) -> None:
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
  global DATA_PATH
  DATA_PATH = args.directory.parent
  if args.verbose:
    print("Downloading the Data to the directory:", args.directory)
  if args.pyrocko:
    # We enable the option to use the PyRocko module to download the data as it
    # is more efficient than the ObsPy module by multithreading the download.
    import pyrocko as pr

  else:
    if args.rectdomain:
      from obspy.clients.fdsn.mass_downloader.domain import RectangularDomain
      domain = RectangularDomain(minlatitude=args.rectdomain[0],
                                 maxlatitude=args.rectdomain[1],
                                 minlongitude=args.rectdomain[2],
                                 maxlongitude=args.rectdomain[3])
    else:
      from obspy.clients.fdsn.mass_downloader.domain import CircularDomain
      domain = CircularDomain(latitude=args.circdomain[0],
                              longitude=args.circdomain[1],
                              minradius=args.circdomain[2],
                              maxradius=args.circdomain[3])
    from obspy.clients.fdsn.mass_downloader import Restrictions, MassDownloader
    start, end = args.dates
    restrictions = Restrictions(starttime=start, endtime=end + ONE_DAY,
                                network=COMMA_STR.join(args.network),
                                station=COMMA_STR.join(args.station),
                                channel_priorities=["HH[ZNE]", "EH[ZNE]",
                                                    "HN[ZNE]", "HG[ZNE]"],
                                reject_channels_with_gaps=False,
                                minimum_length=0.0,
                                minimum_interstation_distance_in_m=100.0,
                                location_priorities=["", "00", "10"],
                                chunklength_in_sec=86400)
    from obspy.clients.fdsn import Client
    CLIENTS = {client : Client(client) for client in args.client}
    if args.key:
      # NOTE: It is assumed a single token file is applicable for all clients
      for cl, CL in CLIENTS:
        if cl in [INGV_CLIENT_STR, GFZ_CLIENT_STR]:
          try:
            CL.set_eida_token(args.key, validate=True)
          except Exception as e:
            print(f"Error setting token for {cl}: {e}")
    mdl = MassDownloader(providers=CLIENTS.values())
    mdl.download(domain, restrictions, mseed_storage=args.directory.__str__(),
                 stationxml_storage=Path(DATA_PATH, STATION_STR).__str__())

def main(args : argparse.Namespace) -> None:
  """
  Main function to download the data from the server based on the specified
  arguments.

  input:
    - args          (argparse.Namespace)

  output:
    - None

  errors:
    - None

  notes:

  """
  data_downloader(args)
  return

if __name__ == "__main__": main(ini.parse_arguments())
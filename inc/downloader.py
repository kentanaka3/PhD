import initializer as ini
from constants import *
import argparse
from pathlib import Path
from obspy.core.utcdatetime import UTCDateTime
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Set the project folder
PRJ_PATH = Path(os.path.dirname(__file__)).parent
DATA_PATH = Path(PRJ_PATH, "data")


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
    t = np.arange(start.datetime, end.datetime, td(days=31)).tolist()
    t.append(end.datetime)
    for s_, e_ in zip(t[:-1], t[1:]):
      restrictions = Restrictions(starttime=s_, endtime=e_ + ONE_DAY,
                                  network=COMMA_STR.join(args.network),
                                  station=COMMA_STR.join(args.station),
                                  channel_priorities=["HH[ZNE]", "EH[ZNE]",
                                                      "HN[ZNE]", "HG[ZNE]"],
                                  reject_channels_with_gaps=False,
                                  minimum_length=0.0,
                                  minimum_interstation_distance_in_m=100.0,
                                  location_priorities=[
                                      "", "00", "01", "02", "10"],
                                  chunklength_in_sec=86400)
      from obspy.clients.fdsn import Client
      CLIENTS = dict()
      for client in args.client:
        try:
          CLIENTS[client] = Client(client)
        except Exception as e:
          print(f"Error creating client {client}: {e}")
          continue
        if args.key and client in [INGV_CLIENT_STR, GFZ_CLIENT_STR]:
          # NOTE: It is assumed a single token file is applicable for all
          #       clients
          try:
            Client(client).set_eida_token(args.key, validate=True)
          except Exception as e:
            print(f"Error setting token for {client}: {e}")
      mdl = MassDownloader(providers=CLIENTS.values())
      try:
        mdl.download(domain, restrictions,
                     mseed_storage=args.directory.__str__(),
                     stationxml_storage=Path(DATA_PATH, STATION_STR).__str__())
      except Exception as e:
        print(f"Error downloading data: {e}")


def main(args: argparse.Namespace) -> None:
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


if __name__ == "__main__":
  main(ini.parse_arguments())

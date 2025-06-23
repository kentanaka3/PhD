import numpy as np
import pandas as pd
from pathlib import Path
from ml_catalog.modules import LocalMagnitude


class OGSMagnitude(LocalMagnitude):
  """
  This magnitude scale use has been calibrated for ...
  """

  def __init__(self, station_corrections: pd.DataFrame):
    self.station_corrections = station_corrections
    super().__init__(hypocentral_range=(10, 700))

  def get_log_amp_0(
      self,
      dist_epi_km: np.ndarray,
      depth_km: np.ndarray,
      stations: pd.Series,
  ) -> np.ndarray:
    # Hypocentral distance in km
    r = np.sqrt(dist_epi_km**2 + depth_km**2)
    c0 = -18.0471
    c1 = 1.105
    c2 = 147.111
    c3 = 4.015e-5
    c4 = 1.33885
    c5 = pd.DataFrame([stations, [0] * len(stations)])
    for station, correction in self.station_corrections.iterrows():
      if station in stations:
        c5.loc[1, station] = correction['correction']
    return c0 + c1 * np.log10(r) + c2 * np.log10(r * c3 + c4) + c5

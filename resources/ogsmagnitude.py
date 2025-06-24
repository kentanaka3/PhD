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
    c5 = pd.DataFrame([[station, 0.] for station in stations],
                      columns=["station", "c5"])
    for _, row in self.station_corrections.iterrows():
      if row["station"] in [s.split(".")[1] for s in stations.values]:
        c5.loc[c5["station"] == row["station"], "c5"] = float(row["c5"])
    c5 = np.nanmean(c5["c5"])
    return c0 + c1 * np.log10(r) + c2 * np.log10(r * c3 + c4) + c5

  def _calc_station_magnitude(self, assignments: pd.DataFrame) -> None:
    assignments["station_ML"] = np.log10(
        assignments["amplitude"]
    ) + self.get_log_amp_0(
        assignments["epicentral_distance"],
        assignments["depth"],
        assignments["station"],
    )
    assignments.loc[
        ~self._is_distance_valid(
            assignments["epicentral_distance"], assignments["depth"]
        ),
        "station_ML",
    ] = np.nan
    assignments.loc[
        assignments["phase"] != self.phase,
        "station_ML",
    ] = np.nan
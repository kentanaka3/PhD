import numpy as np
import pandas as pd
from ml_catalog.modules import LocalMagnitude


class OGSLocalMagnitude(LocalMagnitude):
  """
  This magnitude scale use has been calibrated for ...
  """

  def __init__(self,
               station_corrections: pd.DataFrame,
               ignore_stations: pd.DataFrame = pd.DataFrame(),
               components: str = "NE") -> None:
    self.components = components
    self.station_corrections = station_corrections
    self.ignore_stations = ignore_stations
    super().__init__(hypocentral_range=(3, 150))

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
    c5 = stations.to_frame()
    c5["c5"] = 0.0
    c5["station"] = c5["station"].str.split(".").str[1]
    for _, row in self.station_corrections.iterrows():
      if row["station"] in c5["station"].values:
        c5.loc[c5["station"] == row["station"], "c5"] = float(row["c5"])
    return c0 + c1 * np.log10(r) + c2 * np.log10(r * c3 + c4) + c5["c5"]

  def _calc_station_amplitude(self, assignments: pd.DataFrame) -> None:
    """
    Calculate the amplitude for each station and event_idx in the assignments
    DataFrame. The amplitude is calculated as the geometric mean of the
    amplitudes of the P and S picks, if available.
    """
    SNR_THRESHOLD = 1.3
    # Remove amplitude column to avoid confusion
    assignments.drop(columns="amplitude", inplace=True)
    # Step 1
    mask_ = assignments["phase"] == self.phase
    # We merge all P picks with S picks based on event_idx and station.
    # This should return (merged) a single row for each event detected from a
    # station containing the SNR of P and the maximum amplitude registered of S
    # pick if found, once again in the same row.
    # NOTE: This assumes that there will be always 1 P pick and optionally 1 S
    #       pick for each event_idx and station.
    # NOTE: If there is no S pick, the amplitude will be NaN for that station
    #       for that event_idx.
    merged = pd.merge(assignments[mask_], assignments[~mask_], how="inner",
                      on=["event_idx", "station"], suffixes=[
                        f"_{self.phase}",
                        f"_{"S" if self.phase == "P" else "P"}"])
    # Step 2
    # Compute the amplitude of the S pick for each component if the SNR of the
    # P pick is above the threshold
    for component in self.components:
      mask_ = merged[f"snr_{component}_P"] >= SNR_THRESHOLD
      merged.loc[mask_, f"amplitude_{component}"] = merged.loc[
        mask_, f"amplitude_{component}_S"]
      if merged[~mask_].empty:
        continue
      merged.loc[~mask_, f"amplitude_{component}"] = np.nan
    # Compute the geometric mean of the valid amplitudes
    merged["amplitude"] = np.exp(np.nanmean(np.log(
      merged[[f"amplitude_{component}" for component in self.components]]),
      axis=1))
    # Step 3
    # We have determined the amplitude for each event_idx and station, now we
    # merge it back to the assignments DataFrame. This will add the amplitude
    # column to the assignments DataFrame.
    assignments["amplitude"] = pd.merge(
      assignments, merged[["event_idx", "station", "amplitude"]], how="left",
      on=["event_idx", "station"])["amplitude"]

  def _calc_station_magnitude(self, assignments: pd.DataFrame) -> None:
    self._calc_station_amplitude(assignments)
    super()._calc_station_magnitude(assignments)

  def _calc_event_magnitudes(self, events: pd.DataFrame,
                             assignments: pd.DataFrame) -> pd.DataFrame:
    magnitudes = []
    for (event_idx, group), event_df in assignments.groupby(["event_idx", "group"]):
      event_df = event_df[event_df["phase"] == self.phase]

      # Remove listed stations
      if not self.ignore_stations.empty:
        event_df = event_df[
          event_df["station"].isin(self.ignore_stations["station"])
        ]

      station_magnitudes = event_df["station_ML"].values
      valid = ~np.isnan(station_magnitudes)
      if np.sum(valid) >= 3:
        med = np.nanmedian(station_magnitudes)
        # Calculate the absolute deviation from the median
        abs_dev = np.abs(station_magnitudes - med)
        # Compute the median absolute deviation
        mad = np.nanmedian(abs_dev)
        if mad > 0:
          # Remove stations with absolute deviation NO greater than 5 times the
          # median absolute deviation
          station_magnitudes = station_magnitudes[abs_dev <= 5 * mad | ~valid]

      n_stations = np.sum(~np.isnan(station_magnitudes))
      magnitudes.append(
        {
          "idx": event_idx,
          "group": group,
          "ML": np.nanmean(station_magnitudes),
          "ML_median": np.nanmedian(station_magnitudes),
          "ML_unc": np.nanstd(station_magnitudes) / np.sqrt(n_stations - 1),
          "ML_stations": n_stations,
        }
      )
    magnitudes = pd.DataFrame(magnitudes)

    return pd.merge(events, magnitudes, on=["idx", "group"])
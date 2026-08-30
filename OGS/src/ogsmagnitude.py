"""
=============================================================================
OGS Local Magnitude Module - Calibrated M_L for the OGS Network
=============================================================================

OVERVIEW:
Provides :class:`OGSLocalMagnitude`, an OGS-specific subclass of
``ml_catalog.modules.LocalMagnitude`` that implements the local magnitude
scale calibrated for the OGS / Swiss-Alpine seismicity. The class:

1. Loads per-station amplitude corrections from a ``pandas`` table.
2. Optionally restricts station contributions to a network whitelist.
3. Optionally ignores stations known to produce unreliable amplitudes.
4. Computes the reference log-amplitude attenuation curve
   ``log10(A_0) = c0 + c1*log10(r) + c2*log10(r*c3 + c4) + c5_station``
   where ``r`` is the hypocentral distance in kilometers.
5. Simulates a Wood-Anderson response on horizontal components, screens picks
   by SNR, and aggregates station magnitudes into the event-level M_L using
   median-absolute-deviation outlier rejection (5x MAD cutoff).

CALIBRATION CONSTANTS:
    c0 = -18.0471, c1 = 1.105, c2 = 147.111, c3 = 4.015e-5, c4 = 1.33885

USAGE:
    from ogsmagnitude import OGSLocalMagnitude

    ml = OGSLocalMagnitude(
        station_corrections=pd.read_csv("station_corrections.csv"),
        ignore_stations=pd.DataFrame({"station": ["BAD1", "BAD2"]}),
        networkfocus=["OX", "NI"],
        components="NE",
    )
    builder.add_module(ml)

DEPENDENCIES:
    - numpy / pandas: vectorized math and tabular handling
    - ml_catalog.modules.LocalMagnitude: base class providing the assignment
      orchestration and event-level event_magnitudes pipeline

AUTHOR: AI2Seism Project
=============================================================================
"""

import numpy as np
import pandas as pd
from ml_catalog.modules import LocalMagnitude


class OGSLocalMagnitude(LocalMagnitude):
  """
  OGS-specific implementation of the ML Catalog LocalMagnitude.

  OGS has performed extensive calibration of the local magnitude scale for
  Switzerland and surrounding regions. This implementation includes station
  corrections and the option to ignore specific stations that are known to
  produce unreliable amplitude measurements.
  """

  def __init__(self,
               station_corrections: pd.DataFrame,
               ignore_stations: pd.DataFrame = pd.DataFrame(),
               networkfocus: list[str] = [],
               components: str = "NE",
               attenuation_params: dict | None = None) -> None:
    self.components = components
    self.station_corrections = station_corrections
    self.ignore_stations = ignore_stations
    self.networkfocus = networkfocus
    self.attenuation_params = {
        "c0": -18.0471,
        "c1": 1.105,
        "c2": 147.111,
        "c3": 4.015e-5,
        "c4": 1.33885,
    }
    if attenuation_params is not None:
      self.attenuation_params.update(attenuation_params)
    super().__init__(hypocentral_range=(3, 150))

  def get_log_amp_0(
      self,
      dist_epi_km: np.ndarray,
      depth_km: np.ndarray,
      stations: pd.Series,
  ) -> np.ndarray:
    # Hypocentral distance in km (guarded against r=0 singularity)
    r = np.maximum(np.sqrt(dist_epi_km**2 + depth_km**2), 0.01)
    c0 = self.attenuation_params["c0"]
    c1 = self.attenuation_params["c1"]
    c2 = self.attenuation_params["c2"]
    c3 = self.attenuation_params["c3"]
    c4 = self.attenuation_params["c4"]
    c5 = stations.to_frame()
    # Legacy code for station corrections (commented out)
    # c5["c5"] = 0.0
    # c5["station"] = c5["station"].str.split(".").str[1]
    # for _, row in self.station_corrections.iterrows():
    #   if row["station"] in c5["station"].values:
    #     c5.loc[c5["station"] == row["station"], "c5"] = float(row["c5"])
    # return c0 + c1 * np.log10(r) + c2 * np.log10(r * c3 + c4) + c5["c5"]
    c5["station"] = c5["station"].astype(str).str.split(".").str[1]

    if (
        self.station_corrections is not None
        and not self.station_corrections.empty
        and "station" in self.station_corrections.columns
        and "c5" in self.station_corrections.columns
    ):
      corr_map = dict(
          zip(
              self.station_corrections["station"].astype(str),
              self.station_corrections["c5"].astype(float),
          )
      )
      c5["c5"] = c5["station"].map(corr_map).fillna(0.0).astype(float)
    else:
      c5["c5"] = 0.0

    return c0 + c1 * np.log10(r) + c2 * np.log10(r * c3 + c4) + c5["c5"].values

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
    merged = pd.merge(
        assignments[mask_], assignments[~mask_], how="inner",
        on=["event_idx", "station"], suffixes=[
            f"_{self.phase}", f"_{'S' if self.phase == 'P' else 'P'}"
        ]
    )
    # Step 2
    # Compute the amplitude of the S pick for each component if the SNR of the
    # P pick is above the threshold
    for component in self.components:
      mask_ = merged[f"snr_{component}_P"] >= SNR_THRESHOLD
      merged.loc[mask_, f"amplitude_{component}"] = merged.loc[
          mask_, f"amplitude_{component}_S"
      ]
      if merged[~mask_].empty:
        continue
      merged.loc[~mask_, f"amplitude_{component}"] = np.nan
    # Compute the geometric mean of the valid amplitudes
    merged["amplitude"] = np.exp(np.nanmean(
        np.log(
            merged[[f"amplitude_{component}" for component in self.components]]
        ), axis=1)
    )
    # Step 3
    # We have determined the amplitude for each event_idx and station, now we
    # merge it back to the assignments DataFrame. This will add the amplitude
    # column to the assignments DataFrame.
    assignments["amplitude"] = pd.merge(
        assignments, merged[["event_idx", "station", "amplitude"]], how="left",
        on=["event_idx", "station"]
    )["amplitude"]

  def _calc_station_magnitude(self, assignments: pd.DataFrame) -> None:
    self._calc_station_amplitude(assignments)
    super()._calc_station_magnitude(assignments)

  def _calc_event_magnitudes(self, events: pd.DataFrame,
                             assignments: pd.DataFrame) -> pd.DataFrame:
    magnitudes = []
    for (event_idx, group), event_df in assignments.groupby(
        ["event_idx", "group"]
    ):
      event_df = event_df[event_df["phase"] == self.phase]

      # Use specific list of networks if provided
      if self.networkfocus:
        event_df["network"] = event_df["station"].str.split(".").str[0]
        event_df = event_df[event_df["network"].isin(self.networkfocus)]
        event_df.drop(columns="network", inplace=True)

      # Remove listed stations
      if not self.ignore_stations.empty:
        event_df = event_df[
            ~event_df["station"].isin(self.ignore_stations["station"])
        ]

      # Remove stations with absolute deviation NO greater than 5 times the
      # median absolute deviation
      station_magnitudes = event_df["station_ML"].values
      valid = ~np.isnan(station_magnitudes)
      if np.sum(valid) >= 3:
        med = np.nanmedian(station_magnitudes)
        # Calculate the absolute deviation from the median
        abs_dev = np.abs(station_magnitudes - med)
        # Compute the median absolute deviation
        mad = np.nanmedian(abs_dev)
        if mad > 0:
          station_magnitudes = station_magnitudes[abs_dev <= 5 * mad]

      n_stations = np.sum(~np.isnan(station_magnitudes))
      magnitudes.append(
          {
              "idx": event_idx,
              "group": group,
              "ML": np.nanmean(station_magnitudes),
              "ML_median": np.nanmedian(station_magnitudes),
              "ML_unc": (
                  np.nanstd(station_magnitudes) / np.sqrt(n_stations - 1)
                  if n_stations > 1 else np.nan
              ),
              "ML_stations": n_stations,
          }
      )
    return pd.merge(events, pd.DataFrame(magnitudes), on=["idx", "group"])

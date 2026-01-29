"""Sequence clustering pipeline for OGS seismic catalogs.

This script loads catalog windows, prepares features, runs clustering
algorithms, computes metrics, saves per-cluster CSVs, and generates
map/cross-section plots for each configured angle.
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import ogsconstants as OGS_C
import ogsclustering as OGS_CL
from ogscatalog import OGSCatalog

from typing import Tuple, Optional, Callable, Any, Dict

def parse_arguments() -> argparse.Namespace:
  """Parse CLI arguments for the sequence clustering tool."""
  parser = argparse.ArgumentParser(
    description="OGS Sequence Clustering Tool")
  parser.add_argument(
    "-i", "--input", required=True, type=OGS_C.is_file_path,
    help="Input file containing seismic event data"
  )
  parser.add_argument(
    "-v", "--verbose", action='store_true', default=False,
    help="Enable verbose output"
  )
  return parser.parse_args()

class OGSSequence(OGS_CL.OGSClusteringZoo):
  """
  Orchestrates the full OGS sequence clustering procedure.

  Parameters
  ----------
  metadata : dict
    Configuration metadata for the clustering run.
  verbose : bool, optional
    Enable verbose logging output, by default False.
  Attributes
  ----------
  best_params : dict[int, dict[str, dict[str, Any]]]
    Best parameters for each range, algorithm, and metric.
  logger : logging.Logger
    Module-level logger for logging messages.
  Methods
  -------
  run(X=None, figsize=(16, 12), feature_x=0, feature_y=1, y_true=None,
      **common_kwargs)
    Execute the clustering pipeline and generate plots.
  _init_figure(figsize=(16, 12),
               **kwargs) -> dict[str, Tuple[Figure, np.ndarray]]
    Initialize the figure and axes for plotting.
  _load_catalog(range_idx: int, range_: list) -> OGSCatalog
    Load a catalog window based on the specified range.
  _compute_centers(myCatalog: OGSCatalog) -> Tuple[float, float, float, float]
    Compute the cartesian and geographic centers of the catalog events.
  _prepare_events(myCatalog: OGSCatalog, R: float = 6371.0) -> None
    Prepare event features including cartesian coordinates and inter-event
    times.
  _save_clusters(myCatalog: OGSCatalog, algo_name: str,
                 metric_name: Optional[str] = None) -> None
    Save clustered events to CSV files.
  plot_map_view(ax: Axes, df: pd.DataFrame, lon_col: str, lat_col: str,
                mag_col: str, range_idx: int, center: Tuple[float, float],
                angle_rad: float) -> Axes
    Plot clustered events on a map view.
  plot_cross_section(ax: Axes, df: pd.DataFrame, depth_col: str, mag_col: str,
                     range_idx: int, center: Tuple[float, float],
                     angle_rad: float) -> Axes
    Plot clustered events on a cross-section view.
  _plot_results(ax: np.ndarray, range_idx: int, angle_deg: float,
                myCatalog: OGSCatalog) -> None
    Plot map and cross-section results for a catalog window.
  _finalize_figure(fig: Any, ax: np.ndarray, **kwargs) -> None
    Finalize and save the figure after plotting.
  """

  def __init__(self, metadata: dict, verbose: bool = False):
    """Initialize the sequence orchestrator with metadata and verbosity."""
    super().__init__(metadata=metadata, verbose=verbose)
    self.best_params: dict[int, dict[str, dict[str, Any]]] = {}
    self.logger = self._setup_logger()

  def _setup_logger(self) -> logging.Logger:
    """Configure and return a module-level logger."""
    logger = logging.getLogger(self.__class__.__name__)
    if not logger.handlers:
      handler = logging.StreamHandler()
      formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
      )
      handler.setFormatter(formatter)
      logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
    logger.propagate = False
    return logger

  @property
  def metadata_ranges(self) -> list:
    """Configured catalog time windows as list of [start, end] strings."""
    return self._metadata.get("ranges", [])

  @property
  def metadata_angles(self) -> list:
    """Configured azimuth angles (degrees) for cross-sections."""
    return self._metadata.get("angles_deg", [])

  @property
  def metadata_map_bounds(self) -> Tuple[float, float, float, float]:
    """Map bounds as (lon_min, lon_max, lat_min, lat_max)."""
    return tuple(self._metadata.get("map_deg", [13.09, 13.46, 42.44, 42.61]))

  @property
  def metadata_cross_bounds(self) -> Tuple[float, float, float, float]:
    """Cross-section plot bounds as (x_min, x_max, y_min, y_max)."""
    return tuple(self._metadata.get("cross_km", [-10.0, 10.0, 15.0, 0.0]))

  @property
  def metadata_map_size(self) -> Tuple[float, float]:
    """Map window size in km as (width_km, height_km)."""
    return tuple(self._metadata.get("map_km", [30.0, 50.0]))

  def run(self,
          X: Optional[np.ndarray] = None,
          figsize: Tuple[int, int] = (16, 12),
          feature_x: int = 0,
          feature_y: int = 1,
          y_true: Optional[np.ndarray] = None,
          **common_kwargs) -> None:
    """Execute clustering, save cluster CSVs, and generate plots.

    Parameters not used directly are accepted for API compatibility with
    the parent clustering zoo.
    """
    algorithms = self._algorithms
    if not algorithms:
      self.logger.warning("No algorithms provided in metadata; nothing to process.")
      return
    ranges = self.metadata_ranges
    n = len(ranges)  # columns: ranges
    if n == 0:
      self.logger.warning("No ranges provided in metadata; nothing to process.")
      return

    # --------------------- Process Ranges ---------------------
    for range_idx, range_ in enumerate(ranges):
      myCatalog = self._load_catalog(range_idx, range_)
      self._prepare_events(myCatalog)
      # Feature set: planar coordinates, depth, and inter-event time
      data = StandardScaler().fit_transform(myCatalog.EVENTS[[
        "X_KM", "Y_KM", OGS_C.DEPTH_STR, "INTEREVENT"
      ]])
      self.best_params[range_idx] = {}
      for algo_name, _ in algorithms.items():
        self.best_params[range_idx][algo_name] = {}
        for metric_name, _ in self._metrics.items():
          params = self._optimize_for_metric(
            algo_name,
            data,
            metric_name
          )
          self.best_params[range_idx][algo_name][metric_name] = params
          myCatalog.EVENTS["cluster"] = params.get("labels")
          # Persist per-cluster CSVs for the current range
          self._save_clusters(myCatalog, algo_name, metric_name)

    # --------------------- Plotting ---------------------
    for angle_deg in self.metadata_angles:
      for metric_name, _ in self._metrics.items():
        for algo_idx, (algo_name, _) in enumerate(algorithms.items()):
          fig, ax = self._init_figure()[""]
          for range_idx, range_ in enumerate(ranges):
            myCatalog = self._load_catalog(range_idx, range_)
            self._prepare_events(myCatalog)
            params = self.best_params[range_idx][algo_name][metric_name]
            myCatalog.EVENTS["cluster"] = params.get("labels")
            # Draw map and cross-section for each catalog window
            self._plot_results(ax, range_idx, angle_deg, myCatalog)
          self._finalize_figure(fig, ax, n_cols=n, angle_deg=angle_deg,
                                algo_name=algo_name, metric_name=metric_name)

  def _init_figure(self,
                   figsize: Tuple[int, int] = (16, 12),
                   **kwargs) -> dict[str, Tuple[Figure, np.ndarray]]:
    """Create the figure/axes grid and apply common labels and bounds."""
    n_cols = len(self.metadata_ranges)
    lon_min, lon_max, lat_min, lat_max = self.metadata_map_bounds
    x_min, x_max, y_min, y_max = self.metadata_cross_bounds
    fig, ax = plt.subplots(2, n_cols, figsize=(5 * n_cols, 4 * 2))
    ax[0, 0].set_ylabel("Latitude (°)")
    ax[1, 0].set_ylabel("Depth (km)")
    for col in range(n_cols):
      ax[0, col].set(xlabel="Longitude (°)", xlim=(lon_min, lon_max),
                     ylim=(lat_min, lat_max))
      ax[0, col].tick_params(axis='y', labelleft=(col == 0))
      ax[0, col].text(0.05, 0.95, chr(97 + 2 * col) + ")",
                      transform=ax[0, col].transAxes, fontsize=16,
                      verticalalignment='top', bbox=dict(facecolor='white',
                                                         alpha=0.8))
      ax[1, col].invert_yaxis()
      ax[1, col].set(xlabel="Projection (km)", xlim=(x_min, x_max),
                     ylim=(y_min, y_max))
      ax[1, col].tick_params(axis='y', labelleft=(col == 0))
      ax[1, col].text(0.05, 0.95, chr(98 + 2 * col) + ")",
                      transform=ax[1, col].transAxes, fontsize=16,
                      verticalalignment='top', bbox=dict(facecolor='white',
                                                         alpha=0.8))
      for annotation in self._metadata.get("annotations", []):
        lon, lat, text = annotation
        ax[0, col].text(lon, lat, text, fontsize=8, color="black")
    return {"": (fig, ax)}

  def _load_catalog(self, range_idx: int, range_: list) -> OGSCatalog:
    """Load a catalog window and report summary statistics."""
    self.logger.info("Window #%s: %s to %s", range_idx + 1, range_[0], range_[1])
    myCatalog = OGSCatalog(
      input=Path(self._metadata["directory"]),
      start=datetime.strptime(range_[0], OGS_C.DATE_FMT),
      end=datetime.strptime(range_[1], OGS_C.DATE_FMT),
      verbose=self.verbose,
      output=Path(OGS_C.UNDERSCORE_STR.join(range_)),
      name=f"Catalog_Range_{range_idx + 1}"
    )
    myCatalog.load("EVENTS")
    self.logger.info("Number of events = %s", len(myCatalog.EVENTS))
    self.logger.info("Max magnitude    = %s", myCatalog.EVENTS[OGS_C.ML_STR].max())
    return myCatalog

  @staticmethod
  def _compute_centers(
        myCatalog: OGSCatalog
      ) -> Tuple[float, float, float, float]:
    """Compute cartesian and geographic centers for a catalog."""
    center_x, center_y = myCatalog.EVENTS[["X_KM", "Y_KM"]].mean().to_numpy()
    center_lon, center_lat = myCatalog.EVENTS[
      [OGS_C.LONGITUDE_STR, OGS_C.LATITUDE_STR]
    ].mean().to_numpy()
    return center_x, center_y, center_lon, center_lat

  def _prepare_events(self,
                      myCatalog: OGSCatalog,
                      R: float = 6371.0) -> None:
    """Prepare event features: timestamps, cartesian coords, inter-event time."""
    # Ensure timestamp column exists for sorting and inter-event computation
    myCatalog.EVENTS[OGS_C.TIME_STR] = pd.to_datetime(
      myCatalog.EVENTS[OGS_C.TIME_STR]
    )
    myCatalog.EVENTS[OGS_C.TIMESTAMP_STR] = myCatalog.EVENTS[
      OGS_C.TIME_STR
    ].astype("int64") // 10**9
    # Convert lat/lon to a simple local Cartesian approximation (km)
    lat = myCatalog.EVENTS[OGS_C.LATITUDE_STR].to_numpy()
    lon = myCatalog.EVENTS[OGS_C.LONGITUDE_STR].to_numpy()
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    myCatalog.EVENTS["X_KM"] = R * lon_rad * np.cos(lat_rad)
    myCatalog.EVENTS["Y_KM"] = R * lat_rad
    # Sort by time and compute inter-event time in seconds
    myCatalog.EVENTS.sort_values(by=OGS_C.TIMESTAMP_STR, inplace=True)
    myCatalog.EVENTS.reset_index(drop=True, inplace=True)
    myCatalog.EVENTS["INTEREVENT"] = np.diff(
      myCatalog.EVENTS[OGS_C.TIMESTAMP_STR], prepend=0
    )
    myCatalog.EVENTS.loc[0, "INTEREVENT"] = 0

  @staticmethod
  def _save_clusters(myCatalog: OGSCatalog,
                     algo_name: str,
                     metric_name: Optional[str] = None) -> None:
    """Save cluster members to CSV files under the Clusters directory."""
    dir_path = Path("Clusters") / algo_name
    if metric_name: dir_path = dir_path / metric_name
    dir_path = dir_path / OGS_C.UNDERSCORE_STR.join([myCatalog.output.name])
    os.makedirs(dir_path, exist_ok=True)
    for cluster_id, cluster_data in myCatalog.EVENTS.groupby("cluster"):
      cluster_data.to_csv(dir_path / f"{cluster_id}.csv", index=False)

  def plot_map_view(self,
                    ax: Axes,
                    df: pd.DataFrame,
                    lon_col: str,
                    lat_col: str,
                    mag_col: str,
                    range_idx: int,
                    center: Tuple[float, float],
                    angle_rad: float) -> Axes:
    """
    Plot clustered events on the map view with labels and projection line.
    """
    df = df[df["cluster"] != -1].copy()
    enc, uniq, cmap, norm = OGS_CL.labels_to_colormap(df["cluster"].to_numpy())
    sc = ax.scatter(
      df[lon_col].to_numpy(),
      df[lat_col].to_numpy(),
      s=df[mag_col].to_numpy(),
      c=enc,
      cmap=cmap,
      norm=norm,
      linewidths=0.0
    )
    center_lon, center_lat = center
    cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(len(uniq)))
    cbar.ax.set_yticklabels([str(lab) for lab in uniq])

    big = df[mag_col].to_numpy() > 3.5
    ax.scatter(
      df.loc[big, lon_col].to_numpy(),
      df.loc[big, lat_col].to_numpy(),
      color="red", marker="*", s=100, label="Magnitude > 3.5"
    )

    sin_angle = np.sin(angle_rad)
    cos_angle = np.cos(angle_rad)
    line_lon = [center_lon - 0.2 * sin_angle, center_lon + 0.2 * sin_angle]
    line_lat = [center_lat - 0.2 * cos_angle, center_lat + 0.2 * cos_angle]
    ax.plot(line_lon, line_lat, color="k", linestyle="--", lw=0.5)

    for cl_id, cl_data in df.groupby("cluster"):
      if cl_id == -1: continue
      ax.text(cl_data[lon_col].mean(),
              cl_data[lat_col].mean(),
              chr(97 + 2 * range_idx).upper() + str(cl_id),
              fontsize=8, fontweight="bold")
    return ax

  def plot_cross_section(self,
                         ax: Axes,
                         df: pd.DataFrame,
                         depth_col: str,
                         mag_col: str,
                         range_idx: int,
                         center: Tuple[float, float],
                         angle_rad: float) -> Axes:
    """Plot clustered events on the cross-section view with labels."""
    df = df[df["cluster"] != -1].copy()
    center_x, center_y = center
    sin_angle = np.sin(angle_rad)
    cos_angle = np.cos(angle_rad)
    x_km = df["X_KM"].to_numpy() - center_x
    y_km = df["Y_KM"].to_numpy() - center_y
    df["PROJECTION_KM"] = x_km * sin_angle + y_km * cos_angle
    mask = (
      (df["PROJECTION_KM"] >= -0.5 * self.metadata_map_size[1]) &
      (df["PROJECTION_KM"] <= 0.5 * self.metadata_map_size[1]) &
      (df["X_KM"] >= center_x - 0.5 * self.metadata_map_size[0]) &
      (df["X_KM"] <= center_x + 0.5 * self.metadata_map_size[0])
    )
    df = df[mask]

    enc, uniq, cmap, norm = OGS_CL.labels_to_colormap(df["cluster"].to_numpy())
    sc = ax.scatter(
      df["PROJECTION_KM"].to_numpy(),
      df[depth_col].to_numpy(),
      s=df[mag_col].to_numpy(),
      c=enc,
      cmap=cmap,
      norm=norm,
      linewidths=0.0
    )
    cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(len(uniq)))
    cbar.ax.set_yticklabels([str(lab) for lab in uniq])

    big = df[mag_col].to_numpy() > 3.5
    ax.scatter(
      df.loc[big, "PROJECTION_KM"].to_numpy(),
      df.loc[big, depth_col].to_numpy(),
      color="red", marker="*", s=100, label="Magnitude > 3.5"
    )

    for cl_id, cl_data in df.groupby("cluster"):
      if cl_id == -1: continue
      ax.text(
        cl_data["PROJECTION_KM"].mean(),
        cl_data[depth_col].mean(),
        chr(98 + 2 * range_idx).upper() + str(cl_id),
        fontsize=8, fontweight="bold")
    return ax

  def _plot_results(self,
                    ax: np.ndarray,
                    range_idx: int,
                    angle_deg: float,
                    myCatalog: OGSCatalog) -> None:
    """Plot map and cross-section for a single catalog window."""
    center_x, center_y, center_lon, center_lat = \
      self._compute_centers(myCatalog)
    angle_rad = np.radians(angle_deg)

    self.plot_map_view(
      ax[0, range_idx],
      myCatalog.EVENTS,
      lon_col=OGS_C.LONGITUDE_STR,
      lat_col=OGS_C.LATITUDE_STR,
      mag_col=OGS_C.ML_STR,
      range_idx=range_idx,
      center=(center_lon, center_lat),
      angle_rad=angle_rad
    )

    self.plot_cross_section(
      ax[1, range_idx],
      myCatalog.EVENTS,
      depth_col=OGS_C.DEPTH_STR,
      mag_col=OGS_C.ML_STR,
      range_idx=range_idx,
      center=(center_x, center_y),
      angle_rad=angle_rad
    )

  def _finalize_figure(self,
                       fig: Any,
                       ax: np.ndarray,
                       **kwargs) -> None:
    """Finalize layout, save the figure, and close the plot."""
    n_cols = kwargs["n_cols"]
    angle_deg = kwargs["angle_deg"]
    algo_name = kwargs["algo_name"]
    metric_name = kwargs.get("metric_name")

    ax[0, n_cols - 1].legend(fontsize=16, loc="lower left")
    plt.tight_layout()
    output_filepath = Path("Clusters") / \
      f"{algo_name}_{metric_name}_{angle_deg:.1f}.png"
    plt.savefig(output_filepath)
    self.logger.info("Saved plot to %s", output_filepath)
    plt.close()


def main(args: argparse.Namespace) -> None:
  """CLI entry point for the sequence clustering tool."""
  with open(args.input, 'r') as infile: metadata = json.load(infile)
  run = OGSSequence(metadata=metadata, verbose=args.verbose)
  run.run()

if __name__ == "__main__": main(parse_arguments())
"""
=============================================================================
OGS Sequence Clustering Pipeline - Seismic Event Cluster Analysis
=============================================================================

OVERVIEW:
This module implements an automated seismic sequence clustering pipeline for
analyzing spatiotemporal patterns in earthquake catalogs. It identifies and
visualizes clusters of seismic events using machine learning algorithms,
enabling the detection of earthquake sequences, swarms, and aftershock patterns.

KEY FEATURES:
  - Multi-algorithm clustering: Supports multiple clustering algorithms
    (DBSCAN, HDBSCAN, OPTICS, etc.) with hyperparameter optimization
  - Multi-metric evaluation: Optimizes clustering using various metrics
    (silhouette score, Davies-Bouldin index, Calinski-Harabasz, etc.)
  - Temporal windowing: Processes catalog in configurable time windows
  - Dual visualization: Generates map views and cross-section plots
  - Cartesian projection: Converts lat/lon to local km coordinates
  - Inter-event time features: Uses temporal spacing as clustering feature

PIPELINE STAGES:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 1. LOAD CATALOG     Load seismic events for each time window            │
  │ 2. PREPARE FEATURES Convert to Cartesian, compute inter-event times     │
  │ 3. STANDARDIZE      Scale features using StandardScaler                 │
  │ 4. OPTIMIZE         Find best parameters per algorithm/metric           │
  │ 5. CLUSTER          Assign events to clusters                           │
  │ 6. SAVE             Export per-cluster CSV files                        │
  │ 7. VISUALIZE        Generate map and cross-section plots                │
  └─────────────────────────────────────────────────────────────────────────┘

FEATURE SET:
  The clustering uses 4 features (all standardized):
    - X_KM: East-West position in kilometers (from longitude)
    - Y_KM: North-South position in kilometers (from latitude)
    - DEPTH: Hypocenter depth in kilometers
    - INTEREVENT: Time since previous event in seconds

METADATA CONFIGURATION (JSON):
  {
    "directory": "/path/to/catalog",
    "ranges": [
      ["2022-01-01", "2022-06-30"],
      ["2022-07-01", "2022-12-31"]
    ],
    "angles_deg": [0, 45, 90],
    "map_deg": [lon_min, lon_max, lat_min, lat_max],
    "cross_km": [x_min, x_max, depth_max, depth_min],
    "map_km": [width_km, height_km],
    "annotations": [[lon, lat, "Label"], ...],
    "algorithms": {...},
    "metrics": {...}
  }

USAGE:
  Command line:
    python ogssequence.py -i config.json -v

  Programmatic:
    from ogssequence import OGSSequence
    run = OGSSequence(metadata=config_dict, verbose=True)
    run.run()

OUTPUT:
  - Clusters/{algorithm}/{metric}/{range}/cluster_id.csv  (per-cluster CSVs)
  - Clusters/{algorithm}_{metric}_{angle}.png            (visualization plots)

VISUALIZATION:
  Each plot contains:
    - Top row: Map view (longitude vs latitude) with cluster colors
    - Bottom row: Cross-section (projection vs depth) along specified azimuth
    - Events colored by cluster ID
    - High-magnitude events (>3.5) marked with red stars
    - Projection line showing cross-section orientation

DEPENDENCIES:
  - numpy: Numerical operations and array handling
  - pandas: DataFrame operations for event data
  - scikit-learn: StandardScaler for feature normalization
  - matplotlib: Plotting and visualization
  - ogsclustering: Custom clustering algorithms and utilities
  - ogscatalog: Catalog loading and management

=============================================================================
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

# Standard library: Operating system interface for directory creation
import os

# Standard library: JSON parsing for metadata configuration files
import json

# Standard library: Command-line argument parsing
import argparse

# Standard library: Logging framework for status messages
import logging

# NumPy: Numerical computing for array operations and trigonometry
import numpy as np

# Pandas: DataFrame operations for event catalog manipulation
import pandas as pd

# Standard library: Filesystem path handling
from pathlib import Path

# Standard library: Date/time parsing and manipulation
from datetime import datetime

# Scikit-learn: Feature standardization (zero mean, unit variance)
from sklearn.preprocessing import StandardScaler

# Matplotlib: Plotting library for visualization
import matplotlib.pyplot as plt

# Matplotlib: Type hints for axes objects
from matplotlib.axes import Axes

# Matplotlib: Type hints for figure objects
from matplotlib.figure import Figure

# Local module: OGS-specific constants (column names, date formats)
import ogsconstants as OGS_C

# Local module: Clustering algorithms and utilities (parent class)
import ogsclustering as OGS_CL

# Local module: Catalog loading and management
from ogscatalog import OGSCatalog

# Type hints for improved code documentation
from typing import Tuple, Optional, Callable, Any, Dict


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_arguments() -> argparse.Namespace:
  """
  Parse command-line arguments for the sequence clustering tool.

  Returns:
      argparse.Namespace with:
        - input: Path to JSON metadata configuration file
        - verbose: Boolean flag for debug output
  """
  parser = argparse.ArgumentParser(
    description="OGS Sequence Clustering Tool")

  # -i/--input: Path to JSON configuration file (required)
  parser.add_argument(
    "-i", "--input", required=True, type=OGS_C.is_file_path,
    help="Input file containing seismic event data"
  )

  # -v/--verbose: Enable detailed logging output
  parser.add_argument(
    "-v", "--verbose", action='store_true', default=False,
    help="Enable verbose output"
  )

  return parser.parse_args()


# =============================================================================
# OGSSequence Class - Sequence Clustering Orchestrator
# =============================================================================

class OGSSequence(OGS_CL.OGSClusteringZoo):
  """
  Orchestrates the full OGS sequence clustering procedure.

  This class extends OGSClusteringZoo to provide a complete pipeline for
  seismic sequence analysis, including catalog loading, feature preparation,
  clustering optimization, result persistence, and visualization.

  Parameters
  ----------
  metadata : dict
    Configuration metadata for the clustering run, including:
    - directory: Path to catalog data
    - ranges: List of [start, end] date pairs for time windows
    - angles_deg: Azimuth angles for cross-section views
    - map_deg: Geographic bounds for map plots
    - cross_km: Bounds for cross-section plots
    - algorithms: Clustering algorithm configurations
    - metrics: Evaluation metric configurations

  verbose : bool, optional
    Enable verbose logging output, by default False.

  Attributes
  ----------
  best_params : dict[int, dict[str, dict[str, Any]]]
    Nested dictionary storing best parameters for each:
    - range_idx: Time window index
    - algorithm name: Clustering algorithm
    - metric name: Evaluation metric

  logger : logging.Logger
    Module-level logger for status and debug messages.

  Methods
  -------
  run(X=None, figsize=(16, 12), feature_x=0, feature_y=1, y_true=None,
      **common_kwargs)
    Execute the clustering pipeline and generate plots.

  _init_figure(figsize=(16, 12), **kwargs) -> dict[str, Tuple[Figure, np.ndarray]]
    Initialize the figure and axes grid for plotting.

  _load_catalog(range_idx: int, range_: list) -> OGSCatalog
    Load a catalog window based on the specified date range.

  _compute_centers(myCatalog: OGSCatalog) -> Tuple[float, float, float, float]
    Compute the Cartesian and geographic centers of catalog events.

  _prepare_events(myCatalog: OGSCatalog, R: float = 6371.0) -> None
    Prepare event features including Cartesian coordinates and inter-event times.

  _save_clusters(myCatalog: OGSCatalog, algo_name: str,
                 metric_name: Optional[str] = None) -> None
    Save clustered events to CSV files organized by cluster ID.

  plot_map_view(ax: Axes, df: pd.DataFrame, lon_col: str, lat_col: str,
                mag_col: str, range_idx: int, center: Tuple[float, float],
                angle_rad: float) -> Axes
    Plot clustered events on a geographic map view.

  plot_cross_section(ax: Axes, df: pd.DataFrame, depth_col: str, mag_col: str,
                     range_idx: int, center: Tuple[float, float],
                     angle_rad: float) -> Axes
    Plot clustered events on a depth cross-section view.

  _plot_results(ax: np.ndarray, range_idx: int, angle_deg: float,
                myCatalog: OGSCatalog) -> None
    Plot map and cross-section results for a catalog window.

  _finalize_figure(fig: Any, ax: np.ndarray, **kwargs) -> None
    Finalize layout, save the figure, and close the plot.
  """

  # -------------------------------------------------------------------------
  # CONSTRUCTOR
  # -------------------------------------------------------------------------

  def __init__(self, metadata: dict, verbose: bool = False):
    """
    Initialize the sequence orchestrator with metadata and verbosity settings.

    Args:
        metadata: Configuration dictionary from JSON file
        verbose: Enable debug-level logging output
    """
    # Initialize parent clustering zoo with metadata and verbosity
    super().__init__(metadata=metadata, verbose=verbose)

    # Dictionary to store best clustering parameters per range/algo/metric
    # Structure: {range_idx: {algo_name: {metric_name: params_dict}}}
    self.best_params: dict[int, dict[str, dict[str, Any]]] = {}

    # Configure module-level logger
    self.logger = self._setup_logger()

  # -------------------------------------------------------------------------
  # HELPER: Logger Setup
  # -------------------------------------------------------------------------

  def _setup_logger(self) -> logging.Logger:
    """
    Configure and return a module-level logger with formatted output.

    Creates a StreamHandler with timestamp, class name, level, and message.
    Sets level to DEBUG if verbose mode enabled, otherwise INFO.

    Returns:
        logging.Logger: Configured logger instance
    """
    # Get or create logger with class name
    logger = logging.getLogger(self.__class__.__name__)

    # Only add handler if none exist (avoid duplicate handlers)
    if not logger.handlers:
      # Create console handler for stdout output
      handler = logging.StreamHandler()

      # Define log format: timestamp | class | level | message
      formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
      )
      handler.setFormatter(formatter)
      logger.addHandler(handler)

    # Set logging level based on verbosity flag
    logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

    # Prevent propagation to root logger (avoid duplicate messages)
    logger.propagate = False

    return logger

  # -------------------------------------------------------------------------
  # PROPERTIES: Metadata Accessors
  # -------------------------------------------------------------------------

  @property
  def metadata_ranges(self) -> list:
    """
    Configured catalog time windows as list of [start, end] date strings.

    Returns:
        list: List of [start_date, end_date] pairs, e.g.,
              [["2022-01-01", "2022-06-30"], ["2022-07-01", "2022-12-31"]]
    """
    return self._metadata.get("ranges", [])

  @property
  def metadata_angles(self) -> list:
    """
    Configured azimuth angles (degrees) for cross-section projections.

    Returns:
        list: List of angles in degrees, e.g., [0, 45, 90]
    """
    return self._metadata.get("angles_deg", [])

  @property
  def metadata_map_bounds(self) -> Tuple[float, float, float, float]:
    """
    Geographic map bounds for plot axes.

    Returns:
        Tuple: (lon_min, lon_max, lat_min, lat_max) in degrees
    """
    return tuple(self._metadata.get("map_deg", [13.09, 13.46, 42.44, 42.61]))

  @property
  def metadata_cross_bounds(self) -> Tuple[float, float, float, float]:
    """
    Cross-section plot bounds for depth vs projection axes.

    Returns:
        Tuple: (x_min, x_max, depth_max, depth_min) in kilometers
               Note: depth axis is inverted (positive downward)
    """
    return tuple(self._metadata.get("cross_km", [-10.0, 10.0, 15.0, 0.0]))

  @property
  def metadata_map_size(self) -> Tuple[float, float]:
    """
    Map window size for cross-section filtering.

    Returns:
        Tuple: (width_km, height_km) defining the region to include
    """
    return tuple(self._metadata.get("map_km", [30.0, 50.0]))

  # -------------------------------------------------------------------------
  # MAIN METHOD: run() - Execute Full Pipeline
  # -------------------------------------------------------------------------

  def run(self,
          X: Optional[np.ndarray] = None,
          figsize: Tuple[int, int] = (16, 12),
          feature_x: int = 0,
          feature_y: int = 1,
          y_true: Optional[np.ndarray] = None,
          **common_kwargs) -> None:
    """
    Execute the full clustering pipeline: optimize, save, and visualize.

    Workflow:
      1. For each time range: load catalog, prepare features, cluster
      2. For each algorithm/metric combination: optimize and save clusters
      3. For each angle: generate map and cross-section plots

    Parameters not used directly (X, figsize, feature_x, feature_y, y_true)
    are accepted for API compatibility with the parent clustering zoo.

    Args:
        X: Ignored (features computed internally from catalog)
        figsize: Ignored (computed from number of ranges)
        feature_x: Ignored (not used in this implementation)
        feature_y: Ignored (not used in this implementation)
        y_true: Ignored (unsupervised clustering)
        **common_kwargs: Additional keyword arguments (passed through)
    """
    # Get configured algorithms from parent class
    algorithms = self._algorithms

    # Validate that algorithms are configured
    if not algorithms:
      self.logger.warning("No algorithms provided in metadata; "
                          "nothing to process.")
      return

    # Get time ranges from metadata
    ranges = self.metadata_ranges
    n = len(ranges)  # Number of columns in plot (one per range)

    # Validate that ranges are configured
    if n == 0:
      self.logger.warning("No ranges provided in metadata; "
                          "nothing to process.")
      return

    # =========================================================================
    # PHASE 1: CLUSTERING AND OPTIMIZATION
    # =========================================================================
    # Process each time range independently

    # Cache catalogs to avoid redundant disk I/O and re-computation in Phase 2
    cached_catalogs: list = [None] * n

    for range_idx, range_ in enumerate(ranges):
      # Load catalog for this time window
      myCatalog = self._load_catalog(range_idx, range_)

      # Skip empty catalogs
      if myCatalog.EVENTS.empty:
        self.logger.warning("Window #%s has no events; skipping.", range_idx + 1)
        continue

      # Prepare features: Cartesian coordinates, inter-event times
      self._prepare_events(myCatalog)

      # Cache the prepared catalog for Phase 2
      cached_catalogs[range_idx] = myCatalog

      # Extract and standardize feature matrix
      # Features: X_KM (E-W position), Y_KM (N-S position), DEPTH, INTEREVENT
      data = StandardScaler().fit_transform(myCatalog.EVENTS[[
        "X_KM", "Y_KM", OGS_C.DEPTH_STR, "INTEREVENT"
      ]])

      # Initialize parameter storage for this range
      self.best_params[range_idx] = {}

      # Iterate over all algorithms
      for algo_name, _ in algorithms.items():
        self.best_params[range_idx][algo_name] = {}

        # Iterate over all evaluation metrics
        for metric_name, _ in self._metrics.items():
          # Optimize hyperparameters for this algorithm/metric combination
          # Returns dict with 'labels' and optimized parameters
          params = self._optimize_for_metric(
            algo_name,
            data,
            metric_name
          )

          # Store optimized parameters
          self.best_params[range_idx][algo_name][metric_name] = params

          # Assign cluster labels to events (skip if optimization failed)
          labels = params.get("labels")
          if labels is None:
            self.logger.warning(
              "No valid clustering found for %s/%s in window #%s; skipping.",
              algo_name, metric_name, range_idx + 1)
            continue
          myCatalog.EVENTS["cluster"] = labels

          # Persist per-cluster CSV files
          self._save_clusters(myCatalog, algo_name, metric_name)

    # =========================================================================
    # PHASE 2: VISUALIZATION
    # =========================================================================
    # Generate plots for each angle, metric, and algorithm combination

    for angle_deg in self.metadata_angles:
      for metric_name, _ in self._metrics.items():
        for algo_idx, (algo_name, _) in enumerate(algorithms.items()):
          # Create figure with subplot grid: 2 rows (map, cross) × n columns (ranges)
          fig, ax = self._init_figure()[""]

          # Plot each time range in its column
          for range_idx, range_ in enumerate(ranges):
            # Reuse cached catalog from Phase 1
            myCatalog = cached_catalogs[range_idx]

            # Skip ranges that had no events
            if myCatalog is None or range_idx not in self.best_params:
              continue

            # Retrieve stored cluster labels
            params = self.best_params[range_idx][algo_name][metric_name]
            labels = params.get("labels")
            if labels is None:
              continue
            myCatalog.EVENTS["cluster"] = labels

            # Draw map view (top row) and cross-section (bottom row)
            self._plot_results(ax, range_idx, angle_deg, myCatalog)

          # Save figure and close
          self._finalize_figure(fig, ax, n_cols=n, angle_deg=angle_deg,
                                algo_name=algo_name, metric_name=metric_name)

  # -------------------------------------------------------------------------
  # HELPER: Initialize Figure
  # -------------------------------------------------------------------------

  def _init_figure(self,
                   figsize: Tuple[int, int] = (16, 12),
                   **kwargs) -> dict[str, Tuple[Figure, np.ndarray]]:
    """
    Create the figure/axes grid and apply common labels and bounds.

    Creates a 2×n_cols grid where:
      - Row 0: Map views (lon vs lat) for each time range
      - Row 1: Cross-sections (projection vs depth) for each range

    Args:
        figsize: Figure size (overridden by computed size)
        **kwargs: Additional arguments (unused)

    Returns:
        dict: {"": (fig, ax)} where ax is 2D array of Axes objects
    """
    # Get number of columns (one per time range)
    n_cols = len(self.metadata_ranges)

    # Extract plot bounds from metadata
    lon_min, lon_max, lat_min, lat_max = self.metadata_map_bounds
    x_min, x_max, y_min, y_max = self.metadata_cross_bounds

    # Create figure with 2 rows × n_cols columns
    # Size scales with number of columns: 5 inches per column
    fig, ax = plt.subplots(2, n_cols, figsize=(5 * n_cols, 4 * 2))

    # Set y-axis labels for leftmost column only
    ax[0, 0].set_ylabel("Latitude (°)")
    ax[1, 0].set_ylabel("Depth (km)")

    # Configure each column (time range)
    for col in range(n_cols):
      # ----- Top row: Map view configuration -----
      ax[0, col].set(xlabel="Longitude (°)", xlim=(lon_min, lon_max),
                     ylim=(lat_min, lat_max))
      # Only show y-tick labels on leftmost column
      ax[0, col].tick_params(axis='y', labelleft=(col == 0))
      # Add subplot label (a, c, e, ...)
      ax[0, col].text(0.05, 0.95, chr(97 + 2 * col) + ")",
                      transform=ax[0, col].transAxes, fontsize=16,
                      verticalalignment='top', bbox=dict(facecolor='white',
                                                         alpha=0.8))

      # ----- Bottom row: Cross-section configuration -----
      ax[1, col].invert_yaxis()  # Depth increases downward
      ax[1, col].set(xlabel="Projection (km)", xlim=(x_min, x_max),
                     ylim=(y_min, y_max))
      ax[1, col].tick_params(axis='y', labelleft=(col == 0))
      # Add subplot label (b, d, f, ...)
      ax[1, col].text(0.05, 0.95, chr(98 + 2 * col) + ")",
                      transform=ax[1, col].transAxes, fontsize=16,
                      verticalalignment='top', bbox=dict(facecolor='white',
                                                         alpha=0.8))

      # ----- Add custom annotations from metadata -----
      for annotation in self._metadata.get("annotations", []):
        lon, lat, text = annotation
        ax[0, col].text(lon, lat, text, fontsize=8, color="black")

    return {"": (fig, ax)}

  # -------------------------------------------------------------------------
  # HELPER: Load Catalog
  # -------------------------------------------------------------------------

  def _load_catalog(self, range_idx: int, range_: list) -> OGSCatalog:
    """
    Load a catalog window and report summary statistics.

    Args:
        range_idx: Index of the time range (for logging)
        range_: [start_date, end_date] strings in DATE_FMT format

    Returns:
        OGSCatalog: Loaded catalog with EVENTS DataFrame populated
    """
    # Log the time window being processed
    self.logger.info("Window #%s: %s to %s", range_idx + 1, range_[0], range_[1])

    # Build polygon from map_deg bounds if available
    from matplotlib.path import Path as mplPath
    lon_min, lon_max, lat_min, lat_max = self.metadata_map_bounds
    polygon = mplPath([
      (lon_min, lat_min), (lon_min, lat_max),
      (lon_max, lat_max), (lon_max, lat_min)
    ], closed=True)

    # Create catalog instance with time bounds
    myCatalog = OGSCatalog(
      input=Path(self._metadata["directory"]),
      start=datetime.strptime(range_[0], OGS_C.DATE_FMT),
      end=datetime.strptime(range_[1], OGS_C.DATE_FMT),
      verbose=self.verbose,
      polygon=polygon,
      output=Path(OGS_C.UNDERSCORE_STR.join(range_)),
      name=f"Catalog_Range_{range_idx + 1}"
    )

    # Load events from parquet files
    myCatalog.get("EVENTS")

    # Log summary statistics
    self.logger.info("Number of events = %s", len(myCatalog.EVENTS))
    self.logger.info("Max magnitude    = %s", myCatalog.EVENTS[OGS_C.ML_STR].max())

    return myCatalog

  # -------------------------------------------------------------------------
  # HELPER: Compute Centers
  # -------------------------------------------------------------------------

  @staticmethod
  def _compute_centers(
        myCatalog: OGSCatalog
      ) -> Tuple[float, float, float, float]:
    """
    Compute Cartesian and geographic centers for a catalog.

    Args:
        myCatalog: Catalog with X_KM, Y_KM, lon, lat columns

    Returns:
        Tuple: (center_x_km, center_y_km, center_lon, center_lat)
    """
    # Compute mean Cartesian coordinates (km)
    center_x, center_y = myCatalog.EVENTS[["X_KM", "Y_KM"]].mean().to_numpy()

    # Compute mean geographic coordinates (degrees)
    center_lon, center_lat = myCatalog.EVENTS[
      [OGS_C.LONGITUDE_STR, OGS_C.LATITUDE_STR]
    ].mean().to_numpy()

    return center_x, center_y, center_lon, center_lat

  # -------------------------------------------------------------------------
  # HELPER: Prepare Events
  # -------------------------------------------------------------------------

  def _prepare_events(self,
                      myCatalog: OGSCatalog,
                      R: float = 6371.0) -> None:
    """
    Prepare event features: timestamps, Cartesian coords, inter-event time.

    Transforms geographic coordinates to local Cartesian approximation
    and computes inter-event times for temporal clustering.

    Args:
        myCatalog: Catalog to prepare (modified in place)
        R: Earth radius in km (default: 6371.0)

    Computed columns:
        - TIMESTAMP: Unix timestamp (seconds since epoch)
        - X_KM: East-West position in km (from longitude)
        - Y_KM: North-South position in km (from latitude)
        - INTEREVENT: Seconds since previous event
    """
    # Ensure timestamp column exists and is datetime type
    myCatalog.EVENTS[OGS_C.TIME_STR] = pd.to_datetime(
      myCatalog.EVENTS[OGS_C.TIME_STR]
    )

    # Convert to Unix timestamp (seconds since 1970-01-01)
    # Divide by 10^9 to convert nanoseconds to seconds
    myCatalog.EVENTS[OGS_C.TIMESTAMP_STR] = myCatalog.EVENTS[
      OGS_C.TIME_STR
    ].astype("int64") // 10**9

    # ----- Convert lat/lon to local Cartesian approximation (km) -----
    # This is a simple equirectangular projection suitable for small regions
    lat = myCatalog.EVENTS[OGS_C.LATITUDE_STR].to_numpy()
    lon = myCatalog.EVENTS[OGS_C.LONGITUDE_STR].to_numpy()

    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # X = R * lon * cos(lat) : East-West distance in km
    myCatalog.EVENTS["X_KM"] = R * lon_rad * np.cos(lat_rad)

    # Y = R * lat : North-South distance in km
    myCatalog.EVENTS["Y_KM"] = R * lat_rad

    # ----- Compute inter-event times -----
    # Sort events chronologically
    myCatalog.EVENTS.sort_values(by=OGS_C.TIMESTAMP_STR, inplace=True)
    myCatalog.EVENTS.reset_index(drop=True, inplace=True)

    # Compute time difference between consecutive events
    # prepend=0 adds a zero at the start for the diff operation
    myCatalog.EVENTS["INTEREVENT"] = np.diff(
      myCatalog.EVENTS[OGS_C.TIMESTAMP_STR], prepend=0
    )

    # Set first event's inter-event time to 0 (no previous event)
    myCatalog.EVENTS.loc[0, "INTEREVENT"] = 0

  # -------------------------------------------------------------------------
  # HELPER: Save Clusters
  # -------------------------------------------------------------------------

  @staticmethod
  def _save_clusters(myCatalog: OGSCatalog,
                     algo_name: str,
                     metric_name: Optional[str] = None) -> None:
    """
    Save cluster members to CSV files under the Clusters directory.

    Creates a hierarchical directory structure:
      Clusters/{algorithm}/{metric}/{range}/{cluster_id}.csv

    Args:
        myCatalog: Catalog with 'cluster' column assigned
        algo_name: Name of the clustering algorithm
        metric_name: Name of the optimization metric (optional)
    """
    # Build output directory path
    dir_path = Path("Clusters") / algo_name

    # Add metric subdirectory if specified
    if metric_name: dir_path = dir_path / metric_name

    # Add range identifier subdirectory
    dir_path = dir_path / OGS_C.UNDERSCORE_STR.join([myCatalog.output.name])

    # Create directory structure
    os.makedirs(dir_path, exist_ok=True)

    # Save each cluster to a separate CSV file
    for cluster_id, cluster_data in myCatalog.EVENTS.groupby("cluster"):
      cluster_data.to_csv(dir_path / f"{cluster_id}.csv", index=False)

  # -------------------------------------------------------------------------
  # PLOTTING: Map View
  # -------------------------------------------------------------------------

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

    Features:
      - Scatter plot colored by cluster ID
      - Colorbar with cluster labels
      - High-magnitude events (>3.5) marked with red stars
      - Dashed line showing cross-section orientation
      - Cluster centroid labels

    Args:
        ax: Matplotlib Axes object to plot on
        df: DataFrame with event data
        lon_col: Column name for longitude
        lat_col: Column name for latitude
        mag_col: Column name for magnitude (used for marker size)
        range_idx: Time range index (for labeling)
        center: (center_lon, center_lat) for projection line
        angle_rad: Cross-section azimuth in radians

    Returns:
        Axes: The modified axes object
    """
    # Filter out noise points (cluster = -1)
    df = df[df["cluster"] != -1].copy()

    # Convert cluster labels to colormap encoding
    enc, uniq, cmap, norm = OGS_CL.labels_to_colormap(df["cluster"].to_numpy())

    # Create scatter plot with cluster coloring
    sc = ax.scatter(
      df[lon_col].to_numpy(),
      df[lat_col].to_numpy(),
      s=df[mag_col].to_numpy(),  # Size proportional to magnitude
      c=enc,                      # Color by cluster encoding
      cmap=cmap,
      norm=norm,
      linewidths=0.0              # No edge lines
    )

    # Extract center coordinates
    center_lon, center_lat = center

    # Add colorbar with cluster ID labels
    cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(len(uniq)))
    cbar.ax.set_yticklabels([str(lab) for lab in uniq])

    # Highlight high-magnitude events (M > 3.5) with red stars
    big = df[mag_col].to_numpy() > 3.5
    ax.scatter(
      df.loc[big, lon_col].to_numpy(),
      df.loc[big, lat_col].to_numpy(),
      color="red", marker="*", s=100, label="Magnitude > 3.5"
    )

    # Draw cross-section projection line through center
    sin_angle = np.sin(angle_rad)
    cos_angle = np.cos(angle_rad)
    line_lon = [center_lon - 0.2 * sin_angle, center_lon + 0.2 * sin_angle]
    line_lat = [center_lat - 0.2 * cos_angle, center_lat + 0.2 * cos_angle]
    ax.plot(line_lon, line_lat, color="k", linestyle="--", lw=0.5)

    # Add cluster ID labels at cluster centroids
    for cl_id, cl_data in df.groupby("cluster"):
      if cl_id == -1: continue  # Skip noise
      ax.text(cl_data[lon_col].mean(),
              cl_data[lat_col].mean(),
              chr(97 + 2 * range_idx).upper() + str(cl_id),  # e.g., "A0", "A1"
              fontsize=8, fontweight="bold")

    return ax

  # -------------------------------------------------------------------------
  # PLOTTING: Cross-Section
  # -------------------------------------------------------------------------

  def plot_cross_section(self,
                         ax: Axes,
                         df: pd.DataFrame,
                         depth_col: str,
                         mag_col: str,
                         range_idx: int,
                         center: Tuple[float, float],
                         angle_rad: float) -> Axes:
    """
    Plot clustered events on the cross-section view with labels.

    Projects 3D event locations onto a vertical plane defined by the
    azimuth angle, showing depth vs along-strike position.

    Features:
      - Events projected onto vertical cross-section plane
      - Scatter plot colored by cluster ID
      - High-magnitude events marked with red stars
      - Cluster centroid labels

    Args:
        ax: Matplotlib Axes object to plot on
        df: DataFrame with event data (must have X_KM, Y_KM columns)
        depth_col: Column name for depth
        mag_col: Column name for magnitude
        range_idx: Time range index (for labeling)
        center: (center_x_km, center_y_km) for projection origin
        angle_rad: Cross-section azimuth in radians

    Returns:
        Axes: The modified axes object
    """
    # Filter out noise points (cluster = -1)
    df = df[df["cluster"] != -1].copy()

    # Extract center coordinates in km
    center_x, center_y = center

    # Compute projection direction components
    sin_angle = np.sin(angle_rad)
    cos_angle = np.cos(angle_rad)

    # Translate to center-relative coordinates
    x_km = df["X_KM"].to_numpy() - center_x
    y_km = df["Y_KM"].to_numpy() - center_y

    # Project onto cross-section line: distance along strike
    df["PROJECTION_KM"] = x_km * sin_angle + y_km * cos_angle

    # Filter to events within the map window
    mask = (
      (df["PROJECTION_KM"] >= -0.5 * self.metadata_map_size[1]) &
      (df["PROJECTION_KM"] <= 0.5 * self.metadata_map_size[1]) &
      (df["X_KM"] >= center_x - 0.5 * self.metadata_map_size[0]) &
      (df["X_KM"] <= center_x + 0.5 * self.metadata_map_size[0])
    )
    df = df[mask]

    # Convert cluster labels to colormap encoding
    enc, uniq, cmap, norm = OGS_CL.labels_to_colormap(df["cluster"].to_numpy())

    # Create scatter plot with cluster coloring
    sc = ax.scatter(
      df["PROJECTION_KM"].to_numpy(),
      df[depth_col].to_numpy(),
      s=df[mag_col].to_numpy(),
      c=enc,
      cmap=cmap,
      norm=norm,
      linewidths=0.0
    )

    # Add colorbar with cluster ID labels
    cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(len(uniq)))
    cbar.ax.set_yticklabels([str(lab) for lab in uniq])

    # Highlight high-magnitude events with red stars
    big = df[mag_col].to_numpy() > OGS_C.OGS_MAX_MAGNITUDE
    ax.scatter(
      df.loc[big, "PROJECTION_KM"].to_numpy(),
      df.loc[big, depth_col].to_numpy(),
      color="red", marker="*", s=100,
      label=f"Magnitude > {OGS_C.OGS_MAX_MAGNITUDE}",
    )

    # Add cluster ID labels at cluster centroids
    for cl_id, cl_data in df.groupby("cluster"):
      if cl_id == -1: continue  # Skip noise
      ax.text(
        cl_data["PROJECTION_KM"].mean(),
        cl_data[depth_col].mean(),
        chr(98 + 2 * range_idx).upper() + str(cl_id),  # e.g., "B0", "B1"
        fontsize=8, fontweight="bold",
      )

    return ax

  # -------------------------------------------------------------------------
  # HELPER: Plot Results
  # -------------------------------------------------------------------------

  def _plot_results(self,
                    ax: np.ndarray,
                    range_idx: int,
                    angle_deg: float,
                    myCatalog: OGSCatalog) -> None:
    """
    Plot map and cross-section for a single catalog window.

    Coordinates the map view and cross-section plotting for one time
    range, using the appropriate axes from the subplot grid.

    Args:
        ax: 2D array of Axes (2 rows × n_cols)
        range_idx: Column index for this time range
        angle_deg: Cross-section azimuth in degrees
        myCatalog: Catalog with cluster labels assigned
    """
    # Compute centers in both coordinate systems
    center_x, center_y, center_lon, center_lat = \
      self._compute_centers(myCatalog)

    # Convert angle to radians
    angle_rad = np.radians(angle_deg)

    # Draw map view in top row
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

    # Draw cross-section in bottom row
    self.plot_cross_section(
      ax[1, range_idx],
      myCatalog.EVENTS,
      depth_col=OGS_C.DEPTH_STR,
      mag_col=OGS_C.ML_STR,
      range_idx=range_idx,
      center=(center_x, center_y),
      angle_rad=angle_rad
    )

  # -------------------------------------------------------------------------
  # HELPER: Finalize Figure
  # -------------------------------------------------------------------------

  def _finalize_figure(self,
                       fig: Any,
                       ax: np.ndarray,
                       **kwargs) -> None:
    """
    Finalize layout, save the figure, and close the plot.

    Args:
        fig: Matplotlib Figure object
        ax: Array of Axes objects
        **kwargs: Must include:
          - n_cols: Number of columns (for legend placement)
          - angle_deg: Azimuth angle (for filename)
          - algo_name: Algorithm name (for filename)
          - metric_name: Metric name (for filename)
    """
    # Extract required kwargs
    n_cols = kwargs["n_cols"]
    angle_deg = kwargs["angle_deg"]
    algo_name = kwargs["algo_name"]
    metric_name = kwargs.get("metric_name")

    # Add legend to the rightmost map subplot
    ax[0, n_cols - 1].legend(fontsize=16, loc="lower left")

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Construct output filename
    output_filepath = Path("Clusters") / \
      f"{algo_name}_{metric_name}_{angle_deg:.1f}.png"

    # Ensure output directory exists
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save figure to file
    plt.savefig(output_filepath)
    self.logger.info("Saved plot to %s", output_filepath)

    # Close figure to free memory
    plt.close()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args: argparse.Namespace) -> None:
  """
  CLI entry point for the sequence clustering tool.

  Loads JSON metadata configuration and executes the full clustering pipeline.

  Args:
      args: Parsed command-line arguments with 'input' and 'verbose' fields
  """
  # Load metadata configuration from JSON file
  with open(args.input, 'r') as infile: metadata = json.load(infile)

  # Create sequence clustering orchestrator
  run = OGSSequence(metadata=metadata, verbose=args.verbose)

  # Execute the full pipeline
  run.run()


# Script entry point: parse arguments and run main
if __name__ == "__main__": main(parse_arguments())
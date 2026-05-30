"""
=============================================================================
OGS Utilities Module - Shared Helpers for Catalog Comparison Workflows
=============================================================================

OVERVIEW:
This module is the catch-all toolbox used by the rest of the ``ogs*`` package.
Everything here is either pure (no in-package side effects) or talks to the
filesystem / logging system on behalf of higher-level modules. It deliberately
groups together the small primitives that would otherwise be re-implemented in
several places.

CONTENTS BY SECTION:

1. LOGGING
   - ``ColorFormatter``: ANSI-colored ``logging.Formatter`` with per-level
     symbol prefixes (``>>>``, ``/!\\``, ``[X]``, ``...``, ``!!!``).
   - ``setup_logger``: One-call configuration that wires the formatter onto
     a per-name logger with verbose / silent toggles.

2. DISTANCE & SIMILARITY FUNCTIONS
   - Pick-level: ``dist_prob``, ``dist_phase``, ``diff_time``, ``dist_time``,
     ``dist_pick``.
   - Event-level: ``diff_space``, ``dist_space``, ``dist_event``.
   - These compose into the cost functions consumed by the BGMA bipartite
     graph matchers further below.

3. POLYGON CONTAINMENT
   - ``contains_point`` / ``contains_points``: pure-numpy ray-casting
     implementations used as a lightweight alternative to
     ``matplotlib.path.Path.contains_points`` when only the geometry is
     needed.

4. ARGUMENT PARSING UTILITIES
   - ``is_date`` / ``is_julian`` / ``is_file_path`` / ``is_dir_path``:
     argparse-compatible validators that raise ``ArgumentTypeError`` on
     failure.
   - ``decimeter``, ``labels_to_colormap``: small numeric/plot helpers.

5. STATION INVENTORY MANAGEMENT
   - ``inventory``: reads station metadata from disk into a normalized
     pandas DataFrame used by catalog plotters.

6. WAVEFORM FILE DISCOVERY
   - ``waveforms``: indexes miniSEED / SAC files on disk and returns a
     date- and station-keyed lookup table.

7. ARGPARSE CUSTOM ACTIONS
   - ``SortDatesAction``: argparse action that parses ``[start, end]`` date
     ranges and sorts them so call-sites can rely on ascending order.

8. BIPARTITE GRAPH MATCHING (BGMA backbone)
   - ``OGSBPGraph``: abstract base for one-to-one matching using
     ``networkx.bipartite``.
   - ``OGSBPGraphPicks`` / ``OGSBPGraphEvents``: concrete subclasses that
     wire the appropriate cost functions from section 2 into the matcher.

USAGE:
  from ogsutils import setup_logger, dist_event, OGSBPGraphPicks

  log = setup_logger(__name__, verbose=True)
  matcher = OGSBPGraphPicks(...)
  matches, missed, proposed = matcher.solve()

DEPENDENCIES:
  - numpy, pandas      : array + DataFrame primitives
  - networkx           : bipartite matching backend
  - obspy.UTCDateTime  : robust datetime parsing for CLI inputs
  - ogsconstants       : shared column-name / unit constants

AUTHOR: AI2Seism Project
=============================================================================
"""

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import os                                   # Operating system interface
import sys                                  # System-specific parameters
import logging                              # Logging facility
import argparse                             # Command-line argument parsing
from pathlib import Path                    # Object-oriented filesystem paths
from datetime import datetime, timedelta as td  # Date and time manipulation
from typing import Any, Optional, Sequence, Tuple, cast  # Type hinting

try:
  from . import ogsconstants as OGS_C
except ImportError:
  import ogsconstants as OGS_C

# =============================================================================
# THIRD-PARTY LIBRARY IMPORTS
# =============================================================================
import numpy as np                         # Numerical computing
import pandas as pd                        # Data manipulation and analysis
import networkx as nx                      # Graph algorithms
                                           # (bipartite matching)
from obspy import UTCDateTime              # Seismology-specific datetime

# =============================================================================
# LOGGING
# =============================================================================


class ColorFormatter(logging.Formatter):
  """Logging formatter with ANSI colors and step-tracing symbols.

  Produces output like:
    >>> 2026-02-13 19:30:08 | OGSSequence          | INFO     | Window #1 ...
    /!\\ 2026-02-13 19:30:08 | ogscatalog.OGSCatalog | WARNING  | Loading ...
    ... 2026-02-13 19:30:08 | ogscatalog.OGSCatalog | DEBUG    | Skipping ...
    [X] 2026-02-13 19:30:08 | ogsdat.OGSdat        | ERROR    | Could not ...
  """

  COLORS = {
    logging.DEBUG:    "\033[36m",    # Cyan
    logging.INFO:     "\033[32m",    # Green
    logging.WARNING:  "\033[33m",    # Yellow
    logging.ERROR:    "\033[31m",    # Red
    logging.CRITICAL: "\033[1;31m",  # Bold Red
  }
  SYMBOLS = {
    logging.DEBUG:    "...",   # trace detail
    logging.INFO:     ">>>",   # step progress
    logging.WARNING:  "/!\\",  # caution
    logging.ERROR:    "[X]",   # failure
    logging.CRITICAL: "!!!",   # critical failure
  }
  RESET = "\033[0m"
  BASE_FMT = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"

  def format(self, record: logging.LogRecord) -> str:
    color = self.COLORS.get(record.levelno, "")
    symbol = self.SYMBOLS.get(record.levelno, "   ")
    formatted = super().format(record)
    return f"{color}{symbol} {formatted}{self.RESET}"


def setup_logger(
  name: str,
  verbose: bool = False,
  silent: bool = False,
) -> logging.Logger:
  """Create and configure a logger with colored, step-tracing output.

  Parameters
  ----------
  name : str
    Logger name (typically ``__name__`` or a class-qualified name).
  verbose : bool
    If True, set log level to DEBUG.
  silent : bool
    If True, set log level to WARNING (overrides *verbose*).

  Returns
  -------
  logging.Logger
    Configured logger instance.
  """
  logger = logging.getLogger(name)
  if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    formatter = ColorFormatter(
      fmt=ColorFormatter.BASE_FMT,
      datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
  if silent:
    logger.setLevel(logging.WARNING)
  else:
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
  logger.propagate = False
  return logger

# =============================================================================
# DISTANCE AND SIMILARITY FUNCTIONS
# =============================================================================
# Functions for computing distances and similarity scores between picks/events


def dist_prob(B: pd.Series, T: pd.Series) -> float:
  """
  Calculate probability ratio between target and base picks.

  Used as a component in the weighted pick matching score.
  Higher target probability relative to base yields higher score.

  Args:
    B: Base pick (ground truth) as pandas Series with PROBABILITY_STR.
    T: Target pick (prediction) as pandas Series with PROBABILITY_STR.

  Returns:
    Ratio of target probability to base probability.
  """
  return T[OGS_C.PROBABILITY_STR] / B[OGS_C.PROBABILITY_STR]


def dist_phase(B: pd.Series, T: pd.Series) -> float:
  """
  Check if phase types match between base and target picks.

  Args:
    B: Base pick as pandas Series with PHASE_STR.
    T: Target pick as pandas Series with PHASE_STR.

  Returns:
    1.0 if phases match (both P or both S), 0.0 otherwise.
  """
  return int(T[OGS_C.PHASE_STR] == B[OGS_C.PHASE_STR])


def diff_time(B: pd.Series, T: pd.Series) -> float:
  """
  Calculate absolute time difference between two picks/events.

  Args:
    B: Base record as pandas Series with TIME_STR (UTCDateTime).
    T: Target record as pandas Series with TIME_STR (UTCDateTime).

  Returns:
    Absolute time difference in seconds.
  """
  return abs(T[OGS_C.TIME_STR] - B[OGS_C.TIME_STR])


def dist_time(B: pd.Series, T: pd.Series,
              offset: td = OGS_C.PICK_TIME_OFFSET) -> float:
  """
  Calculate normalized time similarity score.

  Converts time difference to a similarity score between 0 and 1,
  where 1 means perfect match and 0 means at the tolerance limit.

  Args:
    B: Base record as pandas Series with TIME_STR.
    T: Target record as pandas Series with TIME_STR.
    offset: Maximum time tolerance (default: PICK_TIME_OFFSET).

  Returns:
    Similarity score: 1 - (time_diff / tolerance).
  """
  return 1. - (diff_time(B, T) / offset.total_seconds())


def diff_space(
    B: pd.Series,
    T: pd.Series,
    ndim: int = 2,
    p: float = 2.
  ) -> float:
  from obspy.geodetics import gps2dist_azimuth
  """
  Calculate spatial distance between two locations using geodetic formulas.

  Uses ObsPy's gps2dist_azimuth for accurate great-circle distance.
  Optionally includes depth difference for 3D distance calculation.

  Args:
      B: Base location as pandas Series with LATITUDE_STR, LONGITUDE_STR,
          and optionally DEPTH_STR.
      T: Target location as pandas Series with same columns.
      ndim: Number of dimensions (2 for epicentral, 3 for hypocentral).
      p: Power for distance metric (2 = Euclidean).

  Returns:
      Distance in kilometers, rounded to 4 decimal places.
  """
  # Calculate horizontal distance using geodetic formula (returns meters)
  horizontal_dist_km = gps2dist_azimuth(
      B[OGS_C.LATITUDE_STR], B[OGS_C.LONGITUDE_STR],
      T[OGS_C.LATITUDE_STR], T[OGS_C.LONGITUDE_STR])[0] / 1000.

  # Add vertical component if 3D distance requested (depth in m)
  vertical_component = ((B[OGS_C.DEPTH_STR] - T[OGS_C.DEPTH_STR]) / 1000.) ** p \
    if ndim == 3 else 0.

  # Compute Lp norm distance
  return float(format(np.sqrt(horizontal_dist_km ** p + vertical_component), ".4f"))


def dist_space(B: pd.Series, T: pd.Series,
              offset: float = OGS_C.EVENT_DIST_OFFSET) -> float:
  """
  Calculate normalized spatial similarity score.

  Converts spatial distance to a similarity score between 0 and 1.

  Args:
      B: Base location as pandas Series.
      T: Target location as pandas Series.
      offset: Maximum distance tolerance in km (default: EVENT_DIST_OFFSET).

  Returns:
      Similarity score: 1 - (distance / tolerance).
  """
  return 1. - diff_space(B, T) / offset


def contains_point(
    point: tuple[float, float],
    polygon: Sequence[tuple[float, float]] | np.ndarray,
    include_boundary: bool = True,
    eps: float = OGS_C.EPSILON,
  ) -> bool:
  """
  Test whether a 2D point lies inside a polygon.

  Uses the ray-casting algorithm (odd-even rule) and supports optional
  boundary inclusion via an explicit point-on-segment check.

  Args:
    point: Query point as (x, y), typically (longitude, latitude).
    polygon: Polygon vertices as (x, y) pairs.
    include_boundary: If True, points on polygon edges/vertices are inside.
    eps: Numerical tolerance used in boundary checks.

  Returns:
    True if the point is inside the polygon, False otherwise.

  Raises:
    ValueError: If polygon is not shaped like an (N, 2) vertex array.
  """
  vertices = np.asarray(polygon, dtype=float)
  if vertices.ndim != 2 or vertices.shape[1] != 2:
    raise ValueError("polygon must be an (N, 2) array-like of (x, y) vertices")
  if len(vertices) < 3:
    return False

  x, y = point

  # Make boundary behavior explicit and deterministic.
  if include_boundary:
    for i in range(len(vertices)):
      x1, y1 = vertices[i]
      x2, y2 = vertices[(i + 1) % len(vertices)]

      min_x, max_x = min(x1, x2) - eps, max(x1, x2) + eps
      min_y, max_y = min(y1, y2) - eps, max(y1, y2) + eps

      # Cross product is ~0 when the point is collinear with the segment.
      cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
      if abs(cross) <= eps and min_x <= x <= max_x and min_y <= y <= max_y:
        return True

  inside = False
  j = len(vertices) - 1
  for i in range(len(vertices)):
    xi, yi = vertices[i]
    xj, yj = vertices[j]

    # Edge intersects the horizontal ray to the right of (x, y).
    intersects = ((yi > y) != (yj > y))
    if intersects:
      x_intersection = (xj - xi) * (y - yi) / (yj - yi) + xi
      if x < x_intersection:
        inside = not inside
    j = i

  return inside


def contains_points(
      polygon: np.ndarray,
      points: np.ndarray
    ) -> np.ndarray:
  """Vectorized ray-casting point-in-polygon test.

  Determines which points lie inside a polygon using the ray-casting
  algorithm. For each point, a horizontal ray is cast to the right
  and the number of polygon edge crossings is counted. An odd number
  of crossings means the point is inside.

  Parameters
  ----------
  polygon : np.ndarray
    Polygon vertices as an (N, 2) array of (x, y) coordinates.
    The polygon is automatically closed (last vertex connects to first).
  points : np.ndarray
    Query points as an (M, 2) array of (x, y) coordinates.

  Returns
  -------
  np.ndarray
    Boolean array of shape (M,) where True indicates the point is
    inside the polygon.

  Notes
  -----
  Uses fully vectorized NumPy operations (no Python loops over points),
  making it efficient for large point sets. Points exactly on an edge
  may be classified as either inside or outside.

  Examples
  --------
  >>> poly = [(0, 0), (1, 0), (1, 1), (0, 1)]
  >>> pts = [(0.5, 0.5), (2.0, 2.0)]
  >>> contains_points(poly, np.array(pts))
  array([ True, False])
  """
  polygon = np.asarray(polygon)
  n_edges = len(polygon)
  # Polygon edge start and end vertices: (M, 2) each
  v1 = polygon
  v2 = np.roll(polygon, -1, axis=0)

  # Extract coordinates: (M,) arrays for edges, (N,) arrays for points
  x1, y1 = v1[:, 0], v1[:, 1]  # edge start
  x2, y2 = v2[:, 0], v2[:, 1]  # edge end
  px, py = points[:, 0], points[:, 1]  # query points

  # Broadcast to (M, N): edge i × point j
  # Whether point j's y-coordinate is between edge i's y-endpoints
  # One endpoint must be strictly above, the other at or below
  y1_mn = y1[:, None]  # (M, 1)
  y2_mn = y2[:, None]  # (M, 1)
  py_mn = py[None, :]  # (1, N)

  cond_a = (y1_mn <= py_mn) & (y2_mn > py_mn)   # upward crossing
  cond_b = (y1_mn > py_mn) & (y2_mn <= py_mn)    # downward crossing
  crosses = cond_a | cond_b  # (M, N)

  # Compute x-coordinate where the ray y=py intersects edge i
  # x_intersect = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
  dy = y2_mn - y1_mn  # (M, 1)
  # Avoid division by zero (horizontal edges never cross a horizontal ray)
  dy_safe = np.where(dy == 0, 1.0, dy)
  t = (py_mn - y1_mn) / dy_safe  # (M, N)
  x_intersect = x1[:, None] + t * (x2 - x1)[:, None]  # (M, N)

  # Point is to the left of the intersection (ray goes rightward)
  right_of_point = x_intersect > px[None, :]  # (M, N)

  # Count crossings: edge crosses the ray if it spans py AND intersects
  # to the right of the point
  inside = np.sum(crosses & right_of_point, axis=0) % 2 == 1  # (N,)

  return inside


def dist_pick(B: pd.Series, T: pd.Series,
              time_offset_sec: td = OGS_C.PICK_TIME_OFFSET) -> float:
  """
  Calculate weighted similarity score for pick matching.

  Combines time similarity (97%), phase match (2%), and probability
  ratio (1%) into a single matching score for bipartite graph edges.

  Args:
      B: Base pick (ground truth) as pandas Series.
      T: Target pick (prediction) as pandas Series.
      time_offset_sec: Time tolerance for matching.

  Returns:
      Weighted similarity score between 0 and 1.
  """
  return (
    97. * dist_time(T, B, time_offset_sec) +  # Time dominates (97%)
    2. * dist_phase(T, B) +                    # Phase type (2%)
    1. * dist_prob(T, B)                       # Probability ratio (1%)
  ) / 100.



def dist_event(T: pd.Series, P: pd.Series,
               time_offset_sec: td = OGS_C.EVENT_TIME_OFFSET,
               space_offset_km: float = OGS_C.EVENT_DIST_OFFSET) -> float:
  """
  Calculate weighted similarity score for event matching.

  Combines time similarity (99%) and spatial similarity (1%) for
  matching detected events to catalog events.

  Args:
    T: Target event as pandas Series.
    P: Predicted/reference event as pandas Series.
    time_offset_sec: Time tolerance for matching.
    space_offset_km: Spatial tolerance in km.

  Returns:
    Weighted similarity score between 0 and 1.
  """
  return (99. * dist_time(T, P, time_offset_sec) +   # Time dominates (99%)
          1. * dist_space(T, P, space_offset_km)) / 100.  # Space (1%)


# =============================================================================
# ARGUMENT PARSING UTILITY FUNCTIONS
# =============================================================================
# Functions for validating and converting command-line arguments


def is_date(string: str) -> datetime:
  """
  Parse a date string in YYYYMMDD format.

  Used as argparse type converter for date arguments.

  Args:
    string: Date string in YYYYMMDD format (e.g., "20220115").

  Returns:
    datetime object representing the parsed date.

  Raises:
    ValueError: If string doesn't match expected format.
  """
  return datetime.strptime(string, OGS_C.YYYYMMDD_FMT)


def is_julian(string: str) -> datetime:
  """
  Parse a Julian day number to datetime (NOT IMPLEMENTED).

  TODO: Define and convert Julian date to Gregorian date.

  Args:
    string: Julian day string.

  Returns:
    datetime object.

  Raises:
    NotImplementedError: This function is not yet implemented.
  """
  # TODO: Define and convert Julian date to Gregorian date
  raise NotImplementedError("Julian date parsing is not yet implemented.")


def is_file_path(string: str) -> Path:
  """
  Validate and convert a string to an absolute file path.

  Used as argparse type converter for file arguments.

  Args:
    string: Path string to validate.

  Returns:
    Absolute Path object if file exists.

  Raises:
    FileNotFoundError: If the file does not exist.
  """
  if os.path.isfile(string):
    return Path(os.path.abspath(string))
  else:
    raise FileNotFoundError(string)


def is_dir_path(string: str) -> Path:
  """
  Validate and convert a string to an absolute directory path.

  Used as argparse type converter for directory arguments.

  Args:
    string: Path string to validate.

  Returns:
    Absolute Path object if directory exists.

  Raises:
    NotADirectoryError: If the directory does not exist.
  """
  if os.path.isdir(string):
    return Path(os.path.abspath(string))
  else:
    raise NotADirectoryError(string)


def decimeter(value, scale='normal') -> int:
  """
  Round a value up to a "nice" number for axis limits.

  Computes the next aesthetically pleasing round number above the input,
  useful for setting plot axis limits.

  Args:
    value: Numeric value to round up.
    scale: Rounding mode:
        - 'normal': Round to next multiple of leading digit + 1
        - 'log': Round to next power of 10
        - other: Round to next multiple of 10

  Returns:
    Rounded integer value.

  Example:
    >>> decimeter(47)  # Returns 50
    >>> decimeter(123, 'log')  # Returns 1000
  """
  # Find the order of magnitude (number of digits - 1)
  base = np.floor(np.log10(abs(value)))

  if scale == 'normal':
    # Round up to next "nice" number (e.g., 47 -> 50, 123 -> 200)
    return ((value // 10 ** base) + 1) * 10 ** base
  elif scale == 'log':
    # Round up to next power of 10
    return int(10 ** (base + 1))

  # Default: round up to next multiple of 10
  return np.ceil(value / 10) * 10


def labels_to_colormap(
      labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Any, Any]:
  """
  Map arbitrary cluster labels to sequential indices for colormapping.

  Handles cases where labels include noise points (label=-1) or
  non-sequential cluster IDs. Creates a discrete colormap with
  one color per unique label.

  Parameters
  ----------
  labels : np.ndarray
    Cluster labels array, may include -1 for noise points.

  Returns
  -------
  tuple
    (encoded_labels, unique_labels, colormap, norm)
    - encoded_labels: Labels mapped to 0..K-1
    - unique_labels: Original unique label values
    - colormap: Matplotlib colormap resampled to K colors
    - norm: BoundaryNorm for discrete color mapping

  Example
  -------
  >>> labels = np.array([0, 1, 1, -1, 2, 0])
  >>> encoded, unique, cmap, norm = labels_to_colormap(labels)
  >>> # encoded: [1, 2, 2, 0, 3, 1] (with -1 mapped to 0)
  """
  from matplotlib.colors import BoundaryNorm  # Discrete colormap normalization
  from matplotlib import cm                   # Colormap registry

  # Find all unique labels (may include -1 for noise)
  unique = np.unique(labels)

  # Create mapping from original labels to sequential indices
  label_to_idx = {lab: i for i, lab in enumerate(unique)}

  # Apply mapping to all labels
  encoded = np.vectorize(label_to_idx.get, otypes=[int])(labels)

  # Create discrete colormap with exactly len(unique) colors
  cmap = cast(Any, cm.get_cmap("Paired")).resampled(len(unique))

  # Create boundary norm for discrete color assignment
  # Boundaries at -0.5, 0.5, 1.5, ... ensure each integer maps to one color
  norm = BoundaryNorm(np.arange(-0.5, len(unique) + 0.5), cmap.N)

  return encoded, unique, cmap, norm

# =============================================================================
# STATION INVENTORY MANAGEMENT
# =============================================================================


def inventory(
    stations: Path,
    output: Optional[Path] = None
  ) -> pd.DataFrame:
  """
  Load and process station metadata from StationXML files.

  Reads all .xml files from the specified directory, extracts station
  coordinates, and assigns colors for plotting.

  Args:
    stations: Path to directory containing StationXML files.

  Returns:
    pd.DataFrame: DataFrame containing station metadata with columns:
    LONGITUDE_STR, LATITUDE_STR, DEPTH_STR, NETWORK_STR, STATION_STR,
    NETCOLOR_STR, STACOLOR_STR

  Side Effects:
    - Logs warnings for unreadable station files
  """
  logger = setup_logger(__name__)
  # Import ObsPy utilities (lazy import to avoid circular dependencies)
  from obspy import Inventory, read_inventory

  # Initialize empty ObsPy Inventory container
  myInventory = Inventory()

  # Read all StationXML files in the directory
  for station in stations.glob("*.xml"):
    try:
      S = read_inventory(str(station))
    except Exception as e:
      logger.warning(f"Unable to read {station}")
      logger.warning(str(e))
      continue
    myInventory.extend(S)

  elements: list[list] = []
  for net in sorted(myInventory.networks, key=lambda x: x.code):
    for sta in net.stations:
      elements.append([
        f"{net.code}.{sta.code}.",  # Unique station ID
        sta.longitude,
        sta.latitude,
        sta.elevation,
        net.code,
        sta.code,
      ])
  INVENTORY = pd.DataFrame(
    elements,
    columns=[
      OGS_C.INDEX_STR, OGS_C.LONGITUDE_STR, OGS_C.LATITUDE_STR,
      OGS_C.DEPTH_STR, OGS_C.NETWORK_STR, OGS_C.STATION_STR
    ],
  ).sort_values(by=[OGS_C.INDEX_STR]).reset_index(drop=True)

  # Use labels_to_colormap for consistent network and station coloring
  from sklearn.preprocessing import LabelEncoder
  net_encoder = LabelEncoder()
  sta_encoder = LabelEncoder()

  network_series = INVENTORY[OGS_C.NETWORK_STR]
  station_series = INVENTORY[OGS_C.STATION_STR]
  if not isinstance(network_series, pd.Series):
    raise TypeError("Expected network column to resolve to a Series")
  if not isinstance(station_series, pd.Series):
    raise TypeError("Expected station column to resolve to a Series")

  net_labels = np.asarray(net_encoder.fit_transform(network_series.to_numpy()))
  sta_labels = np.asarray(sta_encoder.fit_transform(station_series.to_numpy()))

  _, _, net_cmap, net_norm = labels_to_colormap(net_labels)
  _, _, sta_cmap, sta_norm = labels_to_colormap(sta_labels)

  INVENTORY[OGS_C.NETCOLOR_STR] = [net_cmap(net_norm(l)) for l in net_labels]
  INVENTORY[OGS_C.STACOLOR_STR] = [sta_cmap(sta_norm(l)) for l in sta_labels]

  if output is not None:
    INVENTORY.to_csv(output / "OGSInventory.csv", index=False)
  return INVENTORY


# =============================================================================
# WAVEFORM FILE DISCOVERY
# =============================================================================


def waveforms(
    waveforms: Path,
    stations: Path,
    start: datetime,
    end: datetime,
    output: Path = Path("."),
    vlines: list[tuple[datetime, str, str]] = []
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """
  Scan directory for waveform files within a specified date range.

  Recursively searches for MiniSEED files, organizes them by date and
  station, and generates a data availability plot.

  Args:
    waveforms: Path to the waveforms directory to scan.
    stations: Path to directory containing StationXML files.
    start: Start date (inclusive) of the date range.
    end: End date (inclusive) of the date range.
    output: Path to directory where availability plot will be saved.
    vlines: List of tuples containing datetime objects, labels, and colors
            to mark with vertical lines on the plot.

  Returns:
    pd.DataFrame: DataFrame containing waveform file information with columns:
    NETWORK_STR, STATION_STR, LOC_NAME_STR, CHANNEL_STR, DATE_STR, FILENAME_STR
    Each row represents a waveform file.

  Side Effects:
    Generates "OGSAvailability.png" showing station count over time.

  Note:
    Expects waveform filenames in format:
    NET.STA.LOC.CHA__YYYYMMDDTHHMMSS__...mseed
  """
  # Import plotting utilities (lazy import)
  import ogsplotter as OGS_P
  from matplotlib import pyplot as plt

  elements = []
  # Scan all MiniSEED files recursively
  for wf in waveforms.glob("**/*.mseed"):
    if wf.name.startswith("."): continue  # Skip hidden files
    # Parse filename: NET.STA.LOC.CHA__YYYYMMDDTHHMMSS__suffix.mseed
    stid, dateinitid, _ = wf.stem.split(
      OGS_C.UNDERSCORE_STR + OGS_C.UNDERSCORE_STR
    )

    # Parse date from filename
    dateinitid = UTCDateTime(dateinitid).date
    if dateinitid < start.date() or dateinitid > end.date():
      continue  # Skip files outside date range
    elements.append([*stid.split(OGS_C.PERIOD_STR), dateinitid, wf])

  WAVEFORMS = pd.DataFrame(
    elements,
    columns=[
      OGS_C.NETWORK_STR, OGS_C.STATION_STR, OGS_C.LOC_NAME_STR,
      OGS_C.CHANNEL_STR, OGS_C.DATE_STR, OGS_C.FILENAME_STR
    ]
  )
  logger = setup_logger(__name__)
  WAVEFORMS.to_csv(output / "OGSWaveforms.csv", index=False)
  logger.info(f"Saved file to {output / 'OGSWaveforms.csv'}")
  INVENTORY = inventory(stations)
  INVENTORY = INVENTORY.merge(
    WAVEFORMS[[OGS_C.NETWORK_STR, OGS_C.STATION_STR]],
    how="inner",
    on=[OGS_C.NETWORK_STR, OGS_C.STATION_STR]
  ).drop_duplicates()
  INVENTORY.to_csv(output / "OGSInventory.csv", index=False)
  logger.info(f"Saved file to {output / 'OGSInventory.csv'}")
  mystations = OGS_P.map_plotter(
    OGS_C.OGS_STUDY_REGION,
    legend=True,
    marker="^",
  )
  for net, df in INVENTORY.groupby(OGS_C.NETWORK_STR):
    mystations.add_plot(
      df[OGS_C.LONGITUDE_STR], df[OGS_C.LATITUDE_STR], label=net,
      color=None, facecolors="none", edgecolors=df[OGS_C.NETCOLOR_STR],
      legend=True,
    )
  mystations.savefig(output / "OGSStations.png")
  plt.close()

  NET_COLORS = INVENTORY[
    [OGS_C.NETWORK_STR, OGS_C.NETCOLOR_STR]
  ].drop_duplicates().set_index(OGS_C.NETWORK_STR)[OGS_C.NETCOLOR_STR].to_dict()
  start_day = start.date()
  end_day = end.date()
  DAYS = [
    start_day + td(days=offset)
    for offset in range((end_day - start_day).days + 1)
  ]
  counts = {
    day: {net: 0 for net in WAVEFORMS[OGS_C.NETWORK_STR].unique()}
    for day in DAYS
  }
  for group_key, group in WAVEFORMS.groupby(
    [OGS_C.DATE_STR, OGS_C.NETWORK_STR]
  ):
    date, net = cast(tuple[Any, Any], group_key)
    counts[date][net] = len(group[OGS_C.STATION_STR].unique())
  df = pd.DataFrame(counts).sort_index().T
  if not df.empty and df.values.size > 0:
    x, y = [UTCDateTime(xx).date for xx in df.index], df.values.T
    OGS_P.stack_plotter(
      x, y, labels=df.columns.tolist(),
      colors=[NET_COLORS.get(net, "gray") for net in df.columns],
      xlabel="Date", ylabel="Station Count",
      output=output / "OGSAvailability.png",
      vlines=vlines,
      legend=True
    )
    plt.close()
  else:
    logger.warning("No waveform data available for availability plot.")
  return WAVEFORMS, INVENTORY


# =============================================================================
# ARGPARSE CUSTOM ACTIONS
# =============================================================================


class SortDatesAction(argparse.Action):
  """
  Custom argparse action to sort date arguments chronologically.

  When multiple dates are provided as command-line arguments, this action
  ensures they are stored in sorted order.

  Example:
      parser.add_argument('-D', nargs=2, action=SortDatesAction)
      # Args "-D 20220115 20220101" will be stored as [20220101, 20220115]
  """

  def __call__(
    self,
    parser: argparse.ArgumentParser,
    namespace: argparse.Namespace,
    values: Any,
    option_string: Optional[str] = None,
  ) -> None:
    """Sort and store the values."""
    sorted_values = sorted(cast(Sequence[str], values))
    namespace.__dict__[self.dest] = sorted_values


# =============================================================================
# BIPARTITE GRAPH MATCHING CLASSES
# =============================================================================
# Classes for optimal assignment between ground truth and predicted data
# using maximum weight bipartite matching via NetworkX


class OGSBPGraph():
  """
  Base class for bipartite graph matching between two datasets.

  Provides the framework for constructing bipartite graphs where nodes
  represent data records and edges represent potential matches with
  associated similarity weights.

  Attributes:
    Base: DataFrame containing reference/ground truth records.
    Target: DataFrame containing records to match against Base.
    G: NetworkX Graph representing the bipartite structure.
    E: Set of matched edge pairs (base_idx, target_idx + len(base)).

  Architecture:
    Base nodes: indices 0 to len(Base)-1
    Target nodes: indices len(Base) to len(Base)+len(Target)-1
    Edges: Connect Base[i] to Target[j] if they are potential matches

  Note:
    This is an abstract base class. Subclasses must implement makeMatch().
  """

  def __init__(self, Base: pd.DataFrame, Target: pd.DataFrame,
               verbose: bool = False):
    """
    Initialize bipartite graph with Base and Target datasets.

    Args:
        Base: Reference dataset (ground truth picks or events).
        Target: Dataset to match against Base (predictions).
    """
    # Reset indices to ensure consistent node numbering
    self.Base = Base.reset_index(drop=True)
    self.Target = Target.reset_index(drop=True)

    # Initialize empty graph and edge set
    self.G = nx.Graph()
    self.E: set[tuple[int, int]] = set()

    self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}",
                               verbose=verbose, silent=False)

    # Build graph and compute matching if both datasets are non-empty
    if not self.Base.empty and not self.Target.empty:
      self.makeMatch()

  def makeMatch(self) -> None:
    """
    Construct the bipartite graph and compute maximum weight matching.

    Must be implemented by subclasses to define edge construction logic.

    Raises:
        NotImplementedError: If called on base class.
    """
    raise NotImplementedError

  def matched_pairs_array(self) -> np.ndarray:
    """Return matched pairs as an oriented ``int64`` array.

    The returned array has shape ``(n_matches, 2)`` and preserves the
    existing node interpretation used by ``self.E``: column 0 is always a
    Base index and column 1 is always a Target index offset by ``len(Base)``.
    ``self.E`` is left untouched for backward compatibility.
    """
    n_matches = len(self.E)
    if n_matches == 0:
      return np.empty((0, 2), dtype=np.int64)

    base_count = len(self.Base)
    pairs = np.empty((n_matches, 2), dtype=np.int64)
    for idx, (left, right) in enumerate(self.E):
      if left < base_count <= right:
        pairs[idx, 0] = left
        pairs[idx, 1] = right
      elif right < base_count <= left:
        pairs[idx, 0] = right
        pairs[idx, 1] = left
      else:
        raise ValueError(
          f"Unexpected matching edge ({left}, {right}) for base size {base_count}."
        )
    return pairs


class OGSBPGraphPicks(OGSBPGraph):
  """
  Bipartite graph for optimal pick assignment between datasets.

  Implements maximum weight bipartite matching to find the optimal
  one-to-one correspondence between manual (Base) and predicted (Target)
  phase picks. Uses NetworkX's max_weight_matching algorithm.

  The matching considers:
  - Time proximity: Picks must be within PICK_TIME_OFFSET
  - Station matching: Only same-station picks can match
  - Phase type: P-P and S-S matches preferred
  - Probability: Higher confidence picks weighted more

  Attributes:
    Inherited from OGSBPGraph.

  Example:
    >>> matcher = OGSBPGraphPicks(manual_picks_df, predicted_picks_df)
    >>> matched_pairs = matcher.E  # Set of (base_idx, target_idx+I) tuples

  Note:
    - Base DataFrame should have: TIME_STR, STATION_STR, PHASE_STR
    - Target DataFrame should have: TIME_STR, STATION_STR, PHASE_STR,
      PROBABILITY_STR
    - Uses station-based pre-filtering for O(n) improvement
  """

  def __init__(self, Base: pd.DataFrame, Target: pd.DataFrame,
               verbose: bool = True):
    """
    Initialize pick matcher with optional probability column creation.

    Args:
      Base: Manual picks DataFrame (ground truth).
      Target: Predicted picks DataFrame from ML model.
    """

    # Ensure PROBABILITY_STR column exists, defaulting to 1.0 if absent
    # (manual picks often don't have probability values)
    if OGS_C.PROBABILITY_STR not in Base.columns:
      Base[OGS_C.PROBABILITY_STR] = 1.0

    # Vectorized UTCDateTime conversion using list comprehension: faster than
    # apply(lambda) for large datasets
    if OGS_C.TIME_STR in Base.columns:
      Base[OGS_C.TIME_STR] = [UTCDateTime(x) for x in Base[OGS_C.TIME_STR]]
    if OGS_C.TIME_STR in Target.columns:
      Target[OGS_C.TIME_STR] = [UTCDateTime(x) for x in Target[OGS_C.TIME_STR]]

    # Call parent constructor (triggers makeMatch)
    super().__init__(Base, Target, verbose=verbose)

  def makeMatch(self) -> None:
    """
    Build bipartite graph and compute maximum weight matching for picks.

    Algorithm:
    1. Group target picks by station for O(1) lookup
    2. For each base pick, find target picks at same station
    3. Add edge if time difference <= PICK_TIME_OFFSET
    4. Edge weight = dist_pick() similarity score
    5. Compute max weight matching (not max cardinality)

    Result stored in self.E as set of matched index pairs.
    """
    I = len(self.Base)  # Offset for target node indices
    J = len(self.Target)
    self.G = nx.Graph()

    # Node indices: Base picks = 0 to I-1, Target picks = I to I+J-1
    # [0, 1, 2, ..., I-1], [I, I+1, I+2, ..., I+J-1]
    # Matching Example:
    # [3, I+5] means Base index 3 is matched to Target index 5 (adjusted by I)
    # [4, I+7] means Base index 4 is matched to Target index 7 (adjusted by I)
    self.Base[OGS_C.STATION_STR] = self.Base[OGS_C.STATION_STR].astype(str)
    self.Target[OGS_C.STATION_STR] = self.Target[OGS_C.STATION_STR].astype(str)

    offset_seconds = OGS_C.PICK_TIME_OFFSET.total_seconds()
    target_times = np.fromiter(
      (
        cast(UTCDateTime, time).timestamp
        for time in self.Target[OGS_C.TIME_STR]
      ),
      dtype=float,
      count=J,
    )

    # Pre-index target picks by station and sorted time so each BASE row only
    # scores candidates that can actually satisfy PICK_TIME_OFFSET.
    target_by_station: dict[str, tuple[np.ndarray[Any, Any],
                                       np.ndarray[Any, Any]]] = {}
    for station, positions in self.Target.groupby(
      OGS_C.STATION_STR, sort=False
    ).indices.items():
      station_positions = np.asarray(positions, dtype=np.int64)
      order = np.argsort(target_times[station_positions], kind="mergesort")
      sorted_positions = station_positions[order]
      target_by_station[station] = (
        sorted_positions,
        target_times[sorted_positions],
      )

    # Build edges between matching picks
    for idxBase, rowBase in self.Base.iterrows():
      station = rowBase[OGS_C.STATION_STR]

      # Only iterate over targets at the same station
      if station not in target_by_station:
        continue

      target_positions, station_times = target_by_station[station]
      base_time = cast(UTCDateTime, rowBase[OGS_C.TIME_STR]).timestamp
      start = int(np.searchsorted(
        station_times, base_time - offset_seconds, side="left"
      ))
      stop = int(np.searchsorted(
        station_times, base_time + offset_seconds, side="right"
      ))

      for target_pos in target_positions[start:stop]:
        rowTarget = self.Target.iloc[int(target_pos)]
        self.G.add_edge(
          idxBase, int(target_pos) + I,  # Target offset by I
          weight=dist_pick(rowBase, rowTarget)
        )

    # Compute maximum weight matching (optimal assignment)
    self.E = nx.max_weight_matching(self.G, maxcardinality=False,
                                    weight='weight')


class OGSBPGraphEvents(OGSBPGraph):
  """
  Bipartite graph for optimal event assignment between datasets.

  Implements maximum weight bipartite matching to find the optimal
  one-to-one correspondence between manual (Base) and detected (Target)
  seismic events. Uses both temporal and spatial constraints.

  The matching considers:
  - Time proximity: Events must be within EVENT_TIME_OFFSET (2 sec)
  - Spatial proximity: Events must be within EVENT_DIST_OFFSET (8 km)
  - Weight: 99% time similarity + 1% spatial similarity

  Attributes:
    Inherited from OGSBPGraph.

  Example:
    >>> matcher = OGSBPGraphEvents(catalog_events_df, detected_events_df)
    >>> matched_pairs = matcher.E

  Note:
    - Requires: TIME_STR, LATITUDE_STR, LONGITUDE_STR columns
    - Optional: DEPTH_STR for 3D distance calculation
    - Uses time-based pre-filtering for efficiency
  """

  def __init__(self, Base: pd.DataFrame, Target: pd.DataFrame,
               verbose: bool = True):
    """
    Initialize event matcher with time column normalization.

    Handles different time column names from various sources
    (e.g., "event_time" from some associators, "time" from others).

    Args:
      Base: Catalog events DataFrame (ground truth).
      Target: Detected events DataFrame from associator.
    """
    # Handle "event_time" column name variant
    if "event_time" in Target.columns:
      Target[OGS_C.TIME_STR] = UTCDateTime(Target["event_time"])

    # Vectorized UTCDateTime conversion
    if OGS_C.TIME_STR in Base.columns:
      Base[OGS_C.TIME_STR] = [UTCDateTime(x) for x in Base[OGS_C.TIME_STR]]
    if "time" in Target.columns:
      Target[OGS_C.TIME_STR] = [UTCDateTime(x) for x in Target["time"]]

    # Call parent constructor (triggers makeMatch)
    super().__init__(Base, Target, verbose=verbose)

  def makeMatch(self):
    """
    Build bipartite graph and compute maximum weight matching for events.

    Algorithm:
    1. Vectorize time values for efficient filtering
    2. For each base event, pre-filter targets by time window
    3. Check spatial distance for time-proximate candidates
    4. Add edge if both constraints met, weight = dist_event()
    5. Compute max weight matching

    Pre-filtering by time significantly reduces the O(n*m) comparison space,
    especially for sparse event catalogs.
    """
    I = len(self.Base)  # Offset for target node indices

    # Vectorized time values for efficient filtering
    base_times = self.Base[OGS_C.TIME_STR].values
    target_times = self.Target[OGS_C.TIME_STR].values

    # Build edges between matching events
    for idxBase, rowBase in self.Base.iterrows():
      # No time pre-filtering for simplicity
      target_candidates = self.Target.copy()
      # Pre-filter targets by time window (reduces candidates significantly)
      time_mask = np.abs(
        target_times - rowBase[OGS_C.TIME_STR]
      ) <= OGS_C.EVENT_TIME_OFFSET.total_seconds()
      target_candidates = target_candidates[time_mask]

      for idxTarget, rowTarget in target_candidates.iterrows():
        # Only check spatial distance if time constraint is met
        if diff_space(rowBase, rowTarget) <= OGS_C.EVENT_DIST_OFFSET:
          # Add edge with similarity weight
          self.G.add_edge(
            idxBase, int(idxTarget) + I,
            weight=dist_event(rowBase, rowTarget)
          )

    # Compute maximum weight matching
    self.E = nx.max_weight_matching(self.G, maxcardinality=False,
                                    weight='weight')
"""
===============================================================================
OGS Utilities Module - Shared Helpers for Catalog Comparison Workflows
===============================================================================

OVERVIEW:
This module is the catch-all toolbox used by the rest of the ``ogs*`` package.
It groups numerical helpers, CLI parsers, logging, metadata discovery, and
matching primitives. Some helpers write files or configure logging, and matcher
constructors normalize caller-supplied DataFrames in place before copying them.

CONTENTS BY SECTION:

1. LOGGING
   - ``ColorFormatter``: ANSI-colored ``logging.Formatter`` with per-level
     symbol prefixes (``>>>``, ``/!\\``, ``[X]``, ``...``, ``!!!``).
   - ``setup_logger``: One-call configuration that wires the formatter onto
     a per-name logger with verbose / quiet toggles.

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
     converters raising ``ValueError``, ``FileNotFoundError``, or
     ``NotADirectoryError`` on failure. ``positive_int`` raises argparse's
     ``ArgumentTypeError``.
   - ``decimeter``, ``labels_to_colormap``: small numeric/plot helpers.
   - ``add_*_arguments`` / ``parse_*_args``: shared CLI argument definitions
     and entrypoint parsers.

5. STATION INVENTORY MANAGEMENT
   - ``inventory``: reads station metadata from disk into a normalized
     pandas DataFrame used by catalog plotters.

6. WAVEFORM FILE DISCOVERY
   - ``waveforms``: scans daily MiniSEED files and returns waveform metadata
     and the matching station inventory, with CSV and plot artifacts.

7. ARGPARSE CUSTOM ACTIONS
   - ``SortDatesAction``: sorts values already converted by argparse's
     ``type`` converter.

8. BIPARTITE GRAPH MATCHING (BGMA backbone)
   - ``OGSBPGraph``: subclass-oriented base for one-to-one matching using
     ``networkx.max_weight_matching``.
   - ``OGSBPGraphPicks`` / ``OGSBPGraphEvents``: concrete subclasses that
     wire the appropriate cost functions from section 2 into the matcher.

USAGE:
  from OGS.src.ogsutils import setup_logger, dist_event, OGSBPGraphPicks

  log = setup_logger(__name__, verbose=True)
  matcher = OGSBPGraphPicks(base_picks, target_picks)
  pairs = matcher.matched_pairs_array()  # Target nodes offset by len(base_picks)

DEPENDENCIES:
  - numpy, pandas : array + DataFrame primitives
  - networkx      : bipartite matching backend
  - obspy         : UTCDateTime normalization and StationXML/geodesy helpers
  - matplotlib    : colormaps and waveform-discovery plots
  - sklearn       : station/network label encoding
  - ogsplotter    : lazily imported waveform-discovery figure builders
  - ogsconstants  : shared column-name / unit constants

AUTHORS:
  - 健
  - Istituto Nazionale di Oceanografia e di Geofisica Sperimentale (OGS)
    Centro di Ricerche Sismologiche (CRS)
  - Università degli Studi di Trieste (UniTS)
    Dipartimento di Matematica, Informatica e Geoscienze (MIGe)
    Applied Data Science and Artificial Intelligence (ADSAI)
  - Terabit Network for Research and Academic Big Data in Italy (TeRABIT)
    Consorzio Interuniversitario del Nord-Est per il Calcolo Automatico (CINECA)

===============================================================================
"""

# =============================================================================
# STANDARD AND THIRD-PARTY IMPORTS
# =============================================================================
import argparse                         # Command-line argument parsing
import logging                          # Logging facility
import networkx as nx                   # Graph algorithms (bipartite matching)
import numpy as np                      # Numerical computing
import os                               # Operating system interface
import pandas as pd                     # Data manipulation and analysis
import sys                              # System-specific parameters
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor  # Multi-threaded scanning
# Date and time manipulation
from datetime import date, datetime, time, timedelta as td
from obspy import UTCDateTime           # Seismology-specific datetime
from pathlib import Path                # Object-oriented filesystem paths
from typing import Any, Optional, Sequence, Tuple, cast  # Type hinting

from . import ogsconstants as OGS_C

# Fallback WORK directory under OGS; WORK_PATH overrides the data root.
DATA_PATH = Path(__file__).resolve().parents[3] / "WORK"
DEFAULT_WAVE_PATH = Path(
    os.environ.get("WORK_PATH", DATA_PATH), OGS_C.WAVEFORM_STR
)
DEFAULT_STATION_PATH = Path(
    os.environ.get("WORK_PATH", DATA_PATH), OGS_C.STATION_STR
)


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
    quiet: bool = False,
) -> logging.Logger:
  """Create and configure a logger with colored, step-tracing output.

  Parameters
  ----------
  name : str
    Logger name (typically ``__name__`` or a class-qualified name).
  verbose : bool
    If True, set log level to DEBUG.
  quiet : bool
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
  if quiet:
    logger.setLevel(logging.WARNING)
  else:
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
  logger.propagate = False
  return logger

# =============================================================================
# DISTANCE AND SIMILARITY FUNCTIONS
# =============================================================================
# Functions for computing distances and similarity scores between picks/events


def dist_prob(B: pd.Series, T: pd.Series, eps: float = 1e-6) -> float:
  """
  Calculate probability ratio between target and base picks.

  Used as a component in the weighted pick matching score.
  Higher target probability relative to base yields higher score.

  Args:
    B: Base pick as pandas Series; missing/null PROBABILITY_STR defaults to 1
    T: Target pick as pandas Series; missing/null PROBABILITY_STR defaults to 1
    eps: Small epsilon to prevent division by zero.

  Returns:
    Ratio of target probability to base probability, bounded in [0.0, 1.0].
  """
  # Legacy code for probability ratio calculation (commented out)
  # return T[OGS_C.PROBABILITY_STR] / B[OGS_C.PROBABILITY_STR]
  prob_t = float(T[OGS_C.PROBABILITY_STR]) if (
      OGS_C.PROBABILITY_STR in T and pd.notna(T[OGS_C.PROBABILITY_STR])
  ) else 1.0
  prob_t = max(prob_t, 0.0)
  prob_b = float(B[OGS_C.PROBABILITY_STR]) if (
      OGS_C.PROBABILITY_STR in B and pd.notna(B[OGS_C.PROBABILITY_STR])
  ) else 1.0
  prob_b = max(prob_b, eps)
  return float(min(max(prob_t / prob_b, 0.0), 1.0))


def dist_phase(B: pd.Series, T: pd.Series) -> float:
  """
  Check if phase types match between base and target picks.

  Args:
    B: Base pick as pandas Series with PHASE_STR.
    T: Target pick as pandas Series with PHASE_STR.

  Returns:
    Integer 1 if phase values are equal, 0 otherwise.
  """
  return int(T[OGS_C.PHASE_STR] == B[OGS_C.PHASE_STR])


def diff_time(B: pd.Series, T: pd.Series) -> float:
  """
  Calculate the absolute difference of two time-column values.

  Args:
    B: Base record as pandas Series with subtractable TIME_STR values.
    T: Target record as pandas Series with compatible TIME_STR values.

  Returns:
    Absolute subtraction result: seconds for UTCDateTime values, a timedelta
    for datetime/Timestamp values. No unit conversion is performed here.
  """
  return abs(T[OGS_C.TIME_STR] - B[OGS_C.TIME_STR])


def dist_time(B: pd.Series, T: pd.Series,
              offset: td = OGS_C.PICK_TIME_OFFSET) -> float:
  """
  Calculate normalized time similarity score.

  Converts time difference to a similarity score between 0 and 1,
  where 1 means perfect match and 0 means at the tolerance limit.

  Args:
    B: Base record with TIME_STR values whose subtraction yields seconds.
    T: Target record with compatible TIME_STR values (e.g. UTCDateTime).
    offset: Nonzero normalization tolerance (default: PICK_TIME_OFFSET).

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

  Uses ObsPy's gps2dist_azimuth for horizontal geodetic distance.
  Optionally includes depth difference for 3D distance calculation.

  Args:
      B: Base location as pandas Series with LATITUDE_STR, LONGITUDE_STR, and
         DEPTH_STR in meters when ndim=3.
      T: Target location as pandas Series with same columns.
      ndim: 3 includes vertical separation; other values are horizontal only.
      p: Exponent applied to horizontal and signed vertical components. The
         final root is always a square root; p=2 is Euclidean.

  Returns:
      Distance in kilometers, rounded to 4 decimal places.
  """
  # Calculate horizontal distance using geodetic formula (returns meters)
  horizontal_dist_km = gps2dist_azimuth(
      B[OGS_C.LATITUDE_STR], B[OGS_C.LONGITUDE_STR],
      T[OGS_C.LATITUDE_STR], T[OGS_C.LONGITUDE_STR])[0] / 1000.

  # Add vertical component if 3D distance requested (depth in m)
  vertical_component = (
      (B[OGS_C.DEPTH_STR] - T[OGS_C.DEPTH_STR]) / 1000.
  ) ** p if ndim == 3 else 0.

  # Compute Lp norm distance
  return float(format(np.sqrt(horizontal_dist_km ** p + vertical_component), ".4f"))


def dist_space(
    B: pd.Series,
    T: pd.Series,
    offset: float = OGS_C.EVENT_DIST_OFFSET
) -> float:
  """
  Calculate normalized spatial similarity score.

  Returns 1 at zero distance and 0 at the tolerance limit.

  Args:
      B: Base location as pandas Series.
      T: Target location as pandas Series.
      offset: Nonzero distance normalization in km
              (default: EVENT_DIST_OFFSET).

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
    include_boundary: If True, points on edges/vertices are explicitly
      included. If False, boundary classification is left to the ray-casting
      rule.
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


def contains_points(polygon: np.ndarray, points: np.ndarray) -> np.ndarray:
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
  # Polygon edge start and end vertices: (N, 2) each
  v1 = polygon
  v2 = np.roll(polygon, -1, axis=0)

  # Extract coordinates: (N,) arrays for edges, (M,) arrays for points
  x1, y1 = v1[:, 0], v1[:, 1]  # edge start
  x2, y2 = v2[:, 0], v2[:, 1]  # edge end
  px, py = points[:, 0], points[:, 1]  # query points

  # Broadcast to (N, M): edge i × point j
  # Whether point j's y-coordinate is between edge i's y-endpoints
  # One endpoint must be strictly above, the other at or below
  y1_mn = y1[:, None]  # (N, 1)
  y2_mn = y2[:, None]  # (N, 1)
  py_mn = py[None, :]  # (1, M)

  cond_a = (y1_mn <= py_mn) & (y2_mn > py_mn)   # upward crossing
  cond_b = (y1_mn > py_mn) & (y2_mn <= py_mn)   # downward crossing
  crosses = cond_a | cond_b  # (N, M)

  # Compute x-coordinate where the ray y=py intersects edge i
  # x_intersect = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
  dy = y2_mn - y1_mn  # (N, 1)
  # Avoid division by zero (horizontal edges never cross a horizontal ray)
  dy_safe = np.where(dy == 0, 1.0, dy)
  t = (py_mn - y1_mn) / dy_safe  # (N, M)
  x_intersect = x1[:, None] + t * (x2 - x1)[:, None]  # (N, M)

  # Point is to the left of the intersection (ray goes rightward)
  right_of_point = x_intersect > px[None, :]  # (N, M)

  # Count crossings: edge crosses the ray if it spans py AND intersects
  # to the right of the point
  inside = np.sum(crosses & right_of_point, axis=0) % 2 == 1  # (M,)

  return inside


def dist_pick(
    B: pd.Series, T: pd.Series, time_offset_sec: td = OGS_C.PICK_TIME_OFFSET
) -> float:
  """
  Calculate weighted similarity score for pick matching.

  Combines time similarity (97%), phase match (2%), and probability
  ratio (1%) into a single matching score for bipartite graph edges.

  Args:
      B: Base pick (ground truth) as pandas Series.
      T: Target pick (prediction) as pandas Series.
      time_offset_sec: Nonzero timedelta used to normalize time difference.

  Returns:
      Weighted similarity score. Times must subtract to numeric seconds; phase
      and probability inputs follow dist_phase/prob.
  """
  return (
      97. * dist_time(B, T, time_offset_sec)  # Time dominates (97%)
      + 2. * dist_phase(B, T)                 # Phase type (2%)
      + 1. * dist_prob(B, T)                  # Probability ratio (1%)
  ) / 100.


def dist_event(T: pd.Series, P: pd.Series,
               time_offset_sec: td = OGS_C.EVENT_TIME_OFFSET,
               space_offset_km: float = OGS_C.EVENT_DIST_OFFSET) -> float:
  """
  Calculate weighted similarity score for event matching.

  Combines time similarity (99%) and spatial similarity (1%) for
  matching detected events to catalog events.

  Args:
    T: First event as pandas Series (the graph passes its Base row here).
    P: Second event as pandas Series (the graph passes its Target row here).
    time_offset_sec: Nonzero timedelta used to normalize time difference.
    space_offset_km: Nonzero spatial normalization in km.

  Returns:
    Weighted similarity score. Times must subtract to numeric seconds; spatial
    similarity uses horizontal distance only.
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
  Parse a Julian day number to datetime.

  Args:
    string: Julian date string in YYYYJJJ format.

  Returns:
    datetime object.

  Raises:
    ValueError: If string doesn't match expected format.
  """
  return datetime.strptime(string, "%Y%j")


def is_time(string: str) -> time:
  """
  Parse a time string in HHMMSS format.

  Used as argparse type converter for time arguments.

  Args:
    string: Time string in HHMMSS format (e.g., "153045").

  Returns:
    datetime.time object representing the parsed time.

  Raises:
    ValueError: If string doesn't match expected format.
  """
  return datetime.strptime(string, OGS_C.TIME_FMT).time()


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
  Round a positive value up to a "nice" number for axis limits.

  Computes the next aesthetically pleasing round number above the input,
  useful for setting plot axis limits.

  Args:
    value: Positive numeric value to round up; zero is not handled specially.
    scale: Rounding mode:
        - 'normal': Round to the next multiple of the leading place value
        - 'log': Round to next power of 10
        - other: Round to next multiple of 10

  Returns:
    Rounded numeric value.

  Example:
    >>> decimeter(47)  # Returns 50
    >>> decimeter(123, 'log')  # Returns 1000
  """
  # Find the order of magnitude (number of digits - 1)
  base = np.floor(np.log10(abs(value)))

  if scale == 'normal':
    # Round up to next "nice" number (e.g., 47 -> 50, 123 -> 200)
    return int(((value // 10 ** base) + 1) * 10 ** base)
  elif scale == 'log':
    # Round up to next power of 10
    return int(10 ** (base + 1))

  # Default: round up to next multiple of 10
  return int(np.ceil(value / 10) * 10)


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
  from matplotlib import colormaps            # Colormap registry

  # Find all unique labels (may include -1 for noise)
  unique = np.unique(labels)
  if len(unique) == 0:
    raise ValueError("Cannot generate colormap for empty labels array.")

  # Create mapping from original labels to sequential indices
  label_to_idx = {lab: i for i, lab in enumerate(unique)}

  # Apply mapping to all labels
  encoded = np.vectorize(label_to_idx.get, otypes=[int])(labels)

  # Create discrete colormap with exactly len(unique) colors
  cmap = colormaps['nipy_spectral'].resampled(len(unique))

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
    output: Optional path to directory where inventory CSV will be saved.

  Returns:
    pd.DataFrame: Columns IDX_EVENTS_STR, (NET.STA.), LONGITUDE_STR,
    LATITUDE_STR, DEPTH_STR, NETWORK_STR, STATION_STR, NETCOLOR_STR,
    STACOLOR_STR. DEPTH_STR contains station elevation in meters. Color values
    are RGBA tuples.

  Raises:
    FileNotFoundError: If the station directory does not exist or contains
      no valid StationXML files.

  Side Effects:
    - Logs warnings for unreadable station files.
    - Writes OGSInventory.csv when output is supplied; output must exist.
  """
  if not stations.is_dir():
    raise FileNotFoundError(f"Station directory not found: {stations}")

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

  if not elements:
    raise FileNotFoundError(
        f"No valid StationXML (*.xml) files found in {stations}"
    )

  INVENTORY = pd.DataFrame(
      elements,
      columns=[
          OGS_C.IDX_EVENTS_STR, OGS_C.LONGITUDE_STR, OGS_C.LATITUDE_STR,
          OGS_C.DEPTH_STR, OGS_C.NETWORK_STR, OGS_C.STATION_STR
      ],
  ).sort_values(by=[OGS_C.IDX_EVENTS_STR]).reset_index(drop=True)

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


def day_directory(target: Path, d_: date | datetime) -> Path:
  """Return the YYYY/MM/DD directory path for a given date under target."""
  return Path(target) / f"{d_.year:04d}" / f"{d_.month:02d}" / f"{d_.day:02d}"


def _scan_day_dir(d_path: Path) -> list[list[Any]]:
  records: list[list[Any]] = []
  for wf in d_path.glob("*.mseed"):
    if wf.name.startswith("."):
      continue
    parts = wf.stem.split(OGS_C.UNDERSCORE_STR + OGS_C.UNDERSCORE_STR)
    if len(parts) >= 2:
      file_date = UTCDateTime(parts[1]).date
      records.append([*parts[0].split(OGS_C.PERIOD_STR), file_date, wf])
  return records


def waveforms(
    waveforms: Path,
    stations: Path,
    start: datetime,
    end: datetime,
    output: Path = Path("."),
    vlines: list[tuple[datetime, str, str]] = [],
    threads: int = OGS_C.DEFAULT_CORES_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """
  Scan directory for waveform files within a specified date range.

  Discovers daily MiniSEED files across date-structured directories
  in parallel using threads determined from the environment (CORES or
  SLURM_CPUS_PER_TASK), organizes them by date and station, and
  generates data availability and station distribution plots.

  Args:
    waveforms: Path to the waveforms directory to scan.
    stations: Path to directory containing StationXML files.
    start: Start date (inclusive) of the date range.
    end: End date (inclusive) of the date range.
    output: Existing directory for metadata CSVs and plots.
    vlines: List of tuples containing datetime objects, labels, and colors
            to mark with vertical lines on the plot.

  Returns:
    tuple[pd.DataFrame, pd.DataFrame]: (WAVEFORMS, INVENTORY).
    WAVEFORMS has NETWORK_STR, STATION_STR, LOC_NAME_STR (SEED location code),
    CHANNEL_STR, DATE_STR (filename start date), FILENAME_STR (Path) columns.
    INVENTORY is the inventory() schema restricted to network/station pairs
    present in WAVEFORMS.

  Side Effects:
    Writes OGSWaveforms.csv, OGSInventory.csv and OGSStations.png. When counts
    are non-empty, writes OGSAvailability.png showing station counts by
    network.

  Note:
    Expects waveform filenames in format:
    NET.STA.LOC.CHA__YYYYMMDDTHHMMSSZ__...mseed beneath YYYY/MM/DD directories.
    Only directories in the inclusive date window are scanned; parsed filename
    dates are not separately filtered.
  """
  # Import plotting utilities (lazy import)
  from . import ogsplotter as OGS_P
  from matplotlib import pyplot as plt

  logger = setup_logger(__name__)

  start_day = start.date()
  end_day = end.date()
  days = [
      start_day + td(days=offset)
      for offset in range((end_day - start_day).days + 1)
  ]

  candidate_dirs: list[Path] = []
  for d in days:
    day_dir = day_directory(waveforms, d)
    if day_dir.is_dir():
      candidate_dirs.append(day_dir)
    else:
      logger.warning("Missing waveform day directory: %s", day_dir)

  elements: list[list[Any]] = []
  if threads > 1 and len(candidate_dirs) > 1:
    with ThreadPoolExecutor(max_workers=threads) as executor:
      for result in executor.map(_scan_day_dir, candidate_dirs):
        elements.extend(result)
  else:
    for d_path in candidate_dirs:
      elements.extend(_scan_day_dir(d_path))

  elements.sort(key=lambda row: (row[4], row[0], row[1], row[3]))

  WAVEFORMS = pd.DataFrame(
      elements,
      columns=[
          OGS_C.NETWORK_STR, OGS_C.STATION_STR, OGS_C.LOC_NAME_STR,
          OGS_C.CHANNEL_STR, OGS_C.DATE_STR, OGS_C.FILENAME_STR
      ]
  )
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
      parser.add_argument('-D', nargs=2, type=is_date, action=SortDatesAction)
      "-D 20220115 20220101" stores the two parsed datetimes in ascending
      order.
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
# SHARED ARGUMENT HELPERS & CLI PARSERS
# =============================================================================


def positive_int(value: str) -> int:
  """Validate that a CLI argument is a positive integer."""
  try:
    parsed_value = int(value)
  except ValueError:
    raise argparse.ArgumentTypeError(f"invalid positive int value: {value!r}")
  if parsed_value <= 0:
    raise argparse.ArgumentTypeError("must be a positive integer")
  return parsed_value


def add_date_range_arguments(
    parser: argparse.ArgumentParser,
    default_dates: Optional[list[datetime]] = None,
) -> Any:
  """
  Add mutually exclusive Gregorian (-D/--dates) and
  Julian (-J/--julian) arguments.
  """
  date_group = parser.add_mutually_exclusive_group(required=False)
  date_group.add_argument(
      '-D', "--dates", dest="dates", required=False, metavar=OGS_C.DATE_STD,
      type=is_date, nargs=2, action=SortDatesAction,
      default=default_dates if default_dates is not None else [
          datetime.min, datetime.max - OGS_C.ONE_DAY
      ],
      help="Specify the beginning and ending (inclusive) Gregorian date (YYYYMMDD) range."
  )
  date_group.add_argument(
      '-J', "--julian", dest="dates", required=False, metavar=OGS_C.DATE_JUL,
      action=SortDatesAction, type=is_julian, nargs=2,
      help="Specify the beginning and ending (inclusive) Julian date (YYYYJJJ) range."
  )
  return date_group


def add_time_arguments(
    parser: argparse.ArgumentParser,
    *flags: str,
    required: bool = False,
    help: str = "Time in HHMMSS format",
    default: Any = None,
    metavar: Optional[str] = None,
) -> Any:
  """Add time argument (-t/--time or custom flags) to an argument parser or group."""
  if not flags:
    flags = ("-t", "--time")
  kwargs: dict[str, Any] = {
      "type": is_time,
      "required": required,
      "help": help,
  }
  if default is not None:
    kwargs["default"] = default
  if metavar is not None:
    kwargs["metavar"] = metavar
  return parser.add_argument(*flags, **kwargs)


def add_file_arguments(
    parser: Any,
    *flags: str,
    required: bool = True,
    nargs: Any = OGS_C.ONE_MORECHAR_STR,
    help: str = "Path to the input file",
    default: Any = None,
    metavar: Optional[str] = None,
) -> Any:
  """Add file input (-f/--file or custom flags) argument to an argument parser or group."""
  if not flags:
    flags = ("-f", "--file")
  kwargs: dict[str, Any] = {
      "type": is_file_path,
      "required": required,
      "help": help,
  }
  if nargs is not None:
    kwargs["nargs"] = nargs
  if default is not None:
    kwargs["default"] = default
  if metavar is not None:
    kwargs["metavar"] = metavar
  return parser.add_argument(*flags, **kwargs)


def add_directory_arguments(
    parser: Any,
    *flags: str,
    required: bool = False,
    default: Any = None,
    help: str = "Directory path",
    metavar: Optional[str] = None,
) -> Any:
  """Add directory argument (-d/--directory or custom flags) to an argument parser or group."""
  if not flags:
    flags = ('-d', "--directory")
  kwargs: dict[str, Any] = {
      "type": is_dir_path,
      "required": required,
      "help": help,
  }
  if default is not None:
    kwargs["default"] = default
  if metavar is not None:
    kwargs["metavar"] = metavar
  return parser.add_argument(*flags, **kwargs)


def add_output_arguments(
    parser: argparse.ArgumentParser,
    *flags: str,
    required: bool = False,
    default: Any = None,
    help: str = "Path or name for output",
    metavar: Optional[str] = None,
) -> Any:
  """Add output argument (-o/--output or custom flags) to an argument parser or group."""
  if not flags:
    flags = ("-o", "--output")
  kwargs: dict[str, Any] = {
      "type": Path,
      "required": required,
      "help": help,
  }
  if default is not None:
    kwargs["default"] = default
  if metavar is not None:
    kwargs["metavar"] = metavar
  return parser.add_argument(*flags, **kwargs)


def add_stations_arguments(
    parser: argparse.ArgumentParser,
    required: bool = True,
    default: Any = DEFAULT_STATION_PATH,
    help: str = "Station metadata directory",
    metavar: Optional[str] = None,
) -> Any:
  """Add stations (-S/--stations) argument to an argument parser."""
  return add_directory_arguments(
      parser,
      "-S", "--stations",
      required=required,
      default=default,
      help=help,
      metavar=metavar,
  )


def add_waveforms_arguments(
    parser: argparse.ArgumentParser,
    required: bool = True,
    default: Any = DEFAULT_WAVE_PATH,
    help: str = "Path to the waveforms directory",
    metavar: Optional[str] = None,
) -> Any:
  """Add waveforms (-W/--waveforms) argument to an argument parser."""
  return add_directory_arguments(
      parser,
      "-W", "--waveforms",
      required=required,
      default=default,
      help=help,
      metavar=metavar,
  )


def add_threads_arguments(
    parser: argparse.ArgumentParser,
    default: int = OGS_C.DEFAULT_CORES_COUNT,
    help: str = "Number of worker threads (default: from SLURM or CPU count)",
    metavar: Optional[str] = None,
) -> Any:
  """Add threads (-t/--threads), defaulting to CORES, SLURM CPUs, or 1."""
  kwargs: dict[str, Any] = {
      "type": positive_int,
      "default": default,
      "help": help,
  }
  if metavar is not None:
    kwargs["metavar"] = metavar
  return parser.add_argument("-t", "--threads", **kwargs)


def add_file_or_dir_arguments(
    parser: argparse.ArgumentParser,
    required: bool = True,
    default_dir: Optional[Path] = None,
) -> Any:
  """Add mutually exclusive input directory (-d/--directory) and file (-f/--file) arguments."""
  path_group = parser.add_mutually_exclusive_group(required=required)
  add_directory_arguments(
      path_group,
      required=False,
      default=default_dir,
      help="Base directory for data files.",
  )
  add_file_arguments(
      path_group,
      required=False,
      default=None,
      metavar=OGS_C.EMPTY_STR,
      help="Path(s) to input data file(s)."
  )
  return path_group


def add_verbosity_arguments(parser: argparse.ArgumentParser) -> Any:
  """Add verbosity (-v/--verbose) and quiet (-q/--quiet) arguments."""
  group = parser.add_mutually_exclusive_group(required=False)
  group.add_argument(
      '-v', "--verbose", action='store_true', default=False,
      help="Enable verbose output"
  )
  group.add_argument(
      "-q", "--quiet", action='store_true', default=False,
      help="Run without standard logging outputs"
  )
  return group


def parse_station_args(
    args: Optional[Sequence[str]] = None
) -> argparse.Namespace:
  """Parse command-line arguments for station inventory extraction."""
  parser = argparse.ArgumentParser(
      description="Discover and summarize per-station waveform inventory."
  )
  parser.add_argument(
      "src_root", type=str, help="Source root path prepended to sys.path"
  )
  add_date_range_arguments(
      parser,
      default_dates=[
          datetime.strptime("20240320", OGS_C.YYYYMMDD_FMT),
          datetime.strptime("20240620", OGS_C.YYYYMMDD_FMT)
      ],
  )
  add_output_arguments(
      parser,
      default=Path("."),
      help="Output directory (default: current directory)"
  )
  add_stations_arguments(parser, required=False)
  add_threads_arguments(parser, default=OGS_C.DEFAULT_CORES_COUNT)
  add_waveforms_arguments(parser, required=False)
  return parser.parse_args(args)


def parse_downloader_args(
    args: Optional[Sequence[str]] = None
) -> argparse.Namespace:
  """Parse command-line arguments for waveform downloading."""
  parser = argparse.ArgumentParser(
      description="Download waveform data from configured FDSN clients"
  )
  add_file_arguments(
      parser, '-K', "--key", default=None, required=False, nargs=None,
      metavar=OGS_C.EMPTY_STR, help="Key to download the data from server."
  )
  parser.add_argument(
      "--network", default=[OGS_C.ALL_WILDCHAR_STR], type=str,
      nargs=OGS_C.ONE_MORECHAR_STR, metavar=OGS_C.EMPTY_STR, required=False,
      help=f"""
          Specify a set of Networks to analyze and negate using a '-' prefix.
          (default: '{OGS_C.ALL_WILDCHAR_STR}').
          Example 0: --network "*" (all networks)\n
          Example 1: --network "OX NI" (exclusively these networks)\n
          Example 2: --network "-OX -NI" (negate these networks)
      """
  )
  parser.add_argument(
      "--station", default=[OGS_C.ALL_WILDCHAR_STR], type=str,
      nargs=OGS_C.ONE_MORECHAR_STR, metavar=OGS_C.EMPTY_STR, required=False,
      help=f"""
          Specify a set of Stations to analyze and negate using a '-' prefix.
          (default: '{OGS_C.ALL_WILDCHAR_STR}').
          Example 0: --station "*"\n
          Example 1: --station "APF VNZE"\n
          Example 2: --station "-ED -OL -SP -VNZE"
      """
  )
  parser.add_argument(
      "--client", metavar=OGS_C.EMPTY_STR, default=OGS_C.OGS_CLIENTS_DEFAULT,
      required=False, type=str, nargs=OGS_C.ONE_MORECHAR_STR,
      help="Client to download the data"
  )
  parser.add_argument(
      "--force", default=False, action='store_true', required=False,
      help="Force running all the pipeline"
  )
  parser.add_argument(
      "--pyrocko", default=False, action='store_true',
      help="Enable PyRocko calls"
  )
  parser.add_argument(
      "--timing", default=False, action='store_true', required=False,
      help="Enable timing"
  )
  parser.add_argument(
      "--timeout", default=OGS_C.OGS_TIMEOUT, type=float, required=False,
      help=f"Timeout for downloading data (default: {OGS_C.OGS_TIMEOUT} sec)"
  )
  parser.add_argument(
      "--retry", default=OGS_C.OGS_RETRY, type=positive_int, required=False,
      help=f"Number of retries for downloading data (default: {OGS_C.OGS_RETRY})"
  )
  domain_group = parser.add_mutually_exclusive_group(required=False)
  domain_group.add_argument(
      "--rectdomain", type=float, nargs=4, default=OGS_C.OGS_STUDY_REGION,
      metavar=("lonW", "lonE", "latS", "latN"),
      help="Rectangular domain to download data: [lonW lonE latS latN]"
  )
  domain_group.add_argument(
      "--circdomain", nargs=4, type=float,
      metavar=("lon", "lat", "min_r", "max_r"),
      help="Circular domain to download data: [center lon, center lat, min r, max r]"
  )
  add_threads_arguments(
      parser,
      metavar=OGS_C.EMPTY_STR,
      help="Number of threads to use for downloading"
  )
  add_time_arguments(
      parser,
      "-c", "--clip",
      required=False,
      help="Specify the time of the center time"
  )
  add_date_range_arguments(
      parser,
      default_dates=[
          datetime.strptime("20240320", OGS_C.YYYYMMDD_FMT),
          datetime.strptime("20240620", OGS_C.YYYYMMDD_FMT)
      ],
  )
  add_stations_arguments(parser, required=False)
  add_waveforms_arguments(parser, required=False)
  add_verbosity_arguments(parser)
  return parser.parse_args(args)


def parse_catalog_args(
    args: Optional[Sequence[str]] = None
) -> argparse.Namespace:
  """Parse command-line arguments for catalog aggregation."""
  parser = argparse.ArgumentParser(description="Parse OGS Manual Catalogs")
  parser.add_argument(
      "-m", "--merge", action='store_true', default=False,
      help="Merge all data files into a single catalog"
  )
  parser.add_argument(
      "-x", "--ext", default=OGS_C.ALL_WILDCHAR_STR, type=str,
      nargs=OGS_C.ONE_MORECHAR_STR, metavar=OGS_C.EMPTY_STR,
      help="File extension to process"
  )
  add_date_range_arguments(
      parser,
      default_dates=[datetime.min, datetime.max - OGS_C.ONE_DAY],
  )
  add_file_or_dir_arguments(parser, required=True)
  add_output_arguments(
      parser,
      default=DATA_PATH / "dataset" / "OGSCatalog",
      help="Name of the catalog"
  )
  add_verbosity_arguments(parser)
  return parser.parse_args(args)


def parse_trainer_args(
    args: Optional[Sequence[str]] = None
) -> argparse.Namespace:
  """Parse command-line arguments for model training."""
  parser = argparse.ArgumentParser(description="Train OGS models")
  add_directory_arguments(
      parser, "-C", "--catalog", required=True,
      help="Path to the catalog directory"
  )
  parser.add_argument(
      "-m", "--model", type=str, default=OGS_C.PHASENET_STR,
      choices=["PhaseNet", "EQTransformer"],
      help="SeisBench model class name (default: PhaseNet)"
  )
  parser.add_argument(
      "-s", "--dataset", type=str, default=OGS_C.INSTANCE_STR,
      choices=[
          OGS_C.INSTANCE_STR, OGS_C.STEAD_STR, OGS_C.SCEDC_STR,
          OGS_C.ORIGINAL_STR, OGS_C.ADRIAARRAY_STR
      ],
      help="Pretrained weights name to fine-tune from (default: instance)"
  )
  parser.add_argument(
      "-b", "--batch_size", type=positive_int, default=256,
      help="Batch size for training"
  )
  parser.add_argument(
      "-d", "--download", action="store_true", help="Enable download mode"
  )
  parser.add_argument(
      "-e", "--epochs", type=positive_int, default=5,
      help="Number of training epochs"
  )
  parser.add_argument(
      "-lr", "--learning_rate", type=float, default=1e-2,
      help="Learning rate for training"
  )
  add_threads_arguments(
      parser, default=4, help="Number of data loader workers"
  )
  add_date_range_arguments(
      parser,
      default_dates=[
          datetime.strptime("20240320", OGS_C.YYYYMMDD_FMT),
          datetime.strptime("20240620", OGS_C.YYYYMMDD_FMT)
      ],
  )
  add_output_arguments(
      parser,
      default=Path("./checkpoints"),
      help="Output directory for model checkpoints"
  )
  parser.add_argument(
      "--prepare-only", action="store_true", default=False,
      help="Prepare dataset only without model training"
  )
  parser.add_argument(
      "--train-only", action="store_true", default=False,
      help="Train model using existing SeisBench dataset without catalog indexing"
  )
  parser.add_argument(
      "--seed", type=int, default=42,
      help="Random seed for repeatable splitting and worker initialization (default: 42)"
  )
  parser.add_argument(
      "--det-weight", type=float, default=1.0,
      help="Weight for EQTransformer detection loss (default: 1.0)"
  )
  parser.add_argument(
      "--p-weight", type=float, default=1.0,
      help="Weight for EQTransformer P-phase loss (default: 1.0)"
  )
  parser.add_argument(
      "--s-weight", type=float, default=1.0,
      help="Weight for EQTransformer S-phase loss (default: 1.0)"
  )
  parser.add_argument(
      "--split-ratio", type=float, default=0.8,
      help="Train/dev split ratio grouped by event (default: 0.8)"
  )
  add_verbosity_arguments(parser)
  add_waveforms_arguments(parser)
  return parser.parse_args(args)


def parse_sequence_args(
    args: Optional[Sequence[str]] = None
) -> argparse.Namespace:
  """Parse command-line arguments for sequence clustering."""
  parser = argparse.ArgumentParser(
      description="OGS Sequence Clustering Tool"
  )
  add_file_arguments(
      parser, "-i", "--input", required=True, nargs=None,
      help="Input file containing seismic event data"
  )
  add_verbosity_arguments(parser)
  return parser.parse_args(args)


def _parse_bulletin_args(
    format_name: str,
    args: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
  """Shared argument parser for legacy bulletin quality check scripts."""
  parser = argparse.ArgumentParser(
      description=f"Run OGS {format_name} quality checks"
  )
  add_file_arguments(parser)
  add_date_range_arguments(
      parser,
      default_dates=[datetime.min, datetime.max - OGS_C.ONE_DAY],
  )
  add_verbosity_arguments(parser)
  return parser.parse_args(args)


def parse_hpl_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
  """Parse command-line arguments for the HPL file processor."""
  return _parse_bulletin_args("HPL", args)


def parse_dat_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
  """Parse command-line arguments for the DAT file processor."""
  return _parse_bulletin_args("DAT", args)


def parse_pun_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
  """Parse command-line arguments for the PUN file processor."""
  return _parse_bulletin_args("PUN", args)


def parse_txt_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
  """Parse command-line arguments for the TXT file processor."""
  return _parse_bulletin_args("TXT", args)


# =============================================================================
# BIPARTITE GRAPH MATCHING CLASSES
# =============================================================================
# Classes for optimal assignment between ground truth and predicted data
# using maximum weight bipartite matching via NetworkX


class OGSBPGraph(ABC):
  """
  Abstract base class for bipartite graph matching between two datasets.

  Provides the framework for constructing bipartite graphs where nodes
  represent data records and edges represent potential matches with
  associated similarity weights.

  Attributes:
    Base: DataFrame containing reference/ground truth records.
    Target: DataFrame containing records to match against Base.
    G: NetworkX Graph representing the bipartite structure.
    E: Set of undirected matched node pairs; endpoint order is not guaranteed.

  Architecture:
    Base nodes: indices 0 to len(Base)-1
    Target nodes: indices len(Base) to len(Base)+len(Target)-1
    Edges: Connect Base[i] to Target[j] if they are potential matches

  Note:
    Subclasses must implement makeMatch() before they can be instantiated,
    including for empty inputs. The constructor calls that implementation
    only when both datasets are non-empty, so any subclass state it needs
    must be initialized before calling super().__init__().
  """

  def __init__(self, Base: pd.DataFrame, Target: pd.DataFrame,
               verbose: bool = False):
    """
    Initialize bipartite graph with Base and Target datasets.

    Args:
        Base: Reference dataset (ground truth picks or events).
        Target: Dataset to match against Base (predictions).
        verbose: Enable DEBUG logging.
    """
    # Reset indices to ensure consistent node numbering
    self.Base = Base.reset_index(drop=True)
    self.Target = Target.reset_index(drop=True)

    # Initialize empty graph and edge set
    self.G = nx.Graph()
    self.E: set[tuple[int, int]] = set()

    self.logger = setup_logger(
        f"{__name__}.{self.__class__.__name__}", verbose=verbose, quiet=False
    )

    # Build graph and compute matching if both datasets are non-empty
    if not self.Base.empty and not self.Target.empty:
      self.makeMatch()

  @abstractmethod
  def makeMatch(self) -> None:
    """
    Construct the bipartite graph and compute maximum weight matching.

    Must be implemented by subclasses to define edge construction logic.

    Raises:
        NotImplementedError: If a subclass explicitly calls this abstract body.
    """
    raise NotImplementedError

  def matched_pairs_array(self) -> np.ndarray:
    """Return matched pairs as an oriented ``int64`` array.

    The returned array has shape ``(n_matches, 2)`` and preserves the node
    interpretation: column 0 is always a Base index and column 1 is always a
    Target index offset by ``len(Base)``.
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
  - Probability: Clipped target/base probability ratio contributes 1% of weight

  Attributes:
    Inherited from OGSBPGraph.

  Example:
    >>> matcher = OGSBPGraphPicks(manual_picks_df, predicted_picks_df)
    >>> matched_pairs = matcher.matched_pairs_array()  # Oriented node pairs

  Note:
    - Base DataFrame should have: TIME_STR, STATION_STR, PHASE_STR
    - Target DataFrame should have: TIME_STR, STATION_STR, PHASE_STR
    - Missing/null probabilities default to 1.0 during scoring
    - Station/time indexing restricts which candidate edges are scored
  """

  def __init__(
      self, Base: pd.DataFrame, Target: pd.DataFrame, verbose: bool = True
  ):
    """
    Normalize input time columns in place and add Base probability if absent.

    Args:
      Base: Manual picks DataFrame (ground truth).
      Target: Predicted picks DataFrame from ML model.
      verbose: Enable DEBUG logging.
    """

    # Ensure PROBABILITY_STR column exists, defaulting to 1.0 if absent
    # (manual picks often don't have probability values)
    if OGS_C.PROBABILITY_STR not in Base.columns:
      Base[OGS_C.PROBABILITY_STR] = 1.0

    # Normalize caller-owned time columns before the parent copies the frames.
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
    1. Index target picks by station and sort their times
    2. For each base pick, binary-search the same-station time window
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
    self.E = nx.max_weight_matching(
        self.G, maxcardinality=False, weight='weight'
    )


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
    - DEPTH_STR is not used: matching uses horizontal geodetic distance
    - Uses time-based pre-filtering for efficiency
  """

  def __init__(self, Base: pd.DataFrame, Target: pd.DataFrame,
               verbose: bool = True):
    """
    Initialize event matcher with time column normalization.

    Mutates caller-owned time columns before the parent copies the frames.
    The ``event_time`` branch attempts a single UTCDateTime conversion of that
    entire Target column; it is not a row-wise alias normalization.

    Args:
      Base: Catalog events DataFrame (ground truth).
      Target: Detected events DataFrame from associator.
      verbose: Enable DEBUG logging.
    """
    # Handle "event_time" column name variant
    if "event_time" in Target.columns:
      Target[OGS_C.TIME_STR] = UTCDateTime(Target["event_time"])

    # Row-wise UTCDateTime conversion of caller-owned time columns.
    if OGS_C.TIME_STR in Base.columns:
      Base[OGS_C.TIME_STR] = [UTCDateTime(x) for x in Base[OGS_C.TIME_STR]]
    if "time" in Target.columns:
      Target[OGS_C.TIME_STR] = [UTCDateTime(x) for x in Target[OGS_C.TIME_STR]]

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
    5. Compute max weight matching (not maximum cardinality)

    Pre-filtering by time significantly reduces the O(n*m) comparison space,
    especially for sparse event catalogs.
    """
    I = len(self.Base)  # Offset for target node indices

    # Vectorized time values for efficient filtering
    base_times = self.Base[OGS_C.TIME_STR].values
    target_times = self.Target[OGS_C.TIME_STR].values

    # Build edges between matching events
    for idxBase, rowBase in self.Base.iterrows():
      # Copy targets, then restrict them to the inclusive time window.
      target_candidates = self.Target.copy()
      # Pre-filter targets by time window (reduces candidates significantly)
      base_time = rowBase[OGS_C.TIME_STR]
      time_mask = np.abs(
          target_times - base_time
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

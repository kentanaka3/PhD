"""
=============================================================================
OGS Catalog Module - Lazy-loading Seismic Event/Pick Catalog + BGMA Review
=============================================================================

OVERVIEW:
This module implements ``OGSCatalog``, a single-class container that lazily
indexes daily seismic event and pick files produced by the OGS processing
pipeline, exposes aggregate ``EVENTS`` and ``PICKS`` DataFrames, and drives
the BGMA (Base/Ground-truth vs. Model Assessment) review workflow used to
compare two catalogs day-by-day.

The class is intentionally large because it bundles four conceptually
distinct responsibilities that all share the same on-disk daily layout and
in-memory DataFrame schemas:

1. CATALOG INDEXING & DAILY I/O
   - ``preload``, ``load_``, ``_load_day``, ``load``, ``postload``, ``get``
   - Discovers dated files under ``events/`` and ``assignments|picks/`` and
     loads them on demand, optionally polygon-filtering events on load.

2. CATALOG VISUALIZATION (single-catalog or comparative)
   - ``plot``, ``plot_events``, ``plot_cumulative_*``,
     ``plot_*_histogram``, plus ``_plot_cumulative`` / ``_plot_histogram``
     helpers.
   - Renders summary maps, cumulative-count curves, and parameter
     histograms with optional target overlays.

3. BGMA EVENT MATCHING & REVIEW
   - Public:  ``bgmaEvents``
   - Private: ``_bgma_events_review``, ``_bgma_events_both``,
              ``_bgma_events_base_only``, ``_bgma_events_target_only``,
              ``_event_feasible_positions``, ``_iter_shared_and_extra_dates``,
              plus per-axis diagnostic plotters
              (``_plot_events_time_diff``, ``_plot_events_mh_map``, ...).
   - Walks every shared/extra date, builds matched / missed / proposed
     event partitions, writes review CSVs, and renders diagnostic figures.

4. BGMA PICK MATCHING & REVIEW
   - Public:  ``bgmaPicks``
   - Private: ``_bgma_picks_both``, ``_bgma_picks_base_only``,
              ``_bgma_picks_target_only``, ``_load_and_clean_picks``,
              ``_clean_picks``, ``_record_unmatched_picks``, plus
              ``_plot_picks_confmtx`` / ``_plot_picks_time_diff`` /
              ``_plot_picks_confidence``.
   - Same per-day cadence as event matching but resolves matches by
     ``(station, phase, time)`` and reports a 3x3 phase confusion matrix.

5. COMBINED WORKFLOW & SET OPERATIONS
   - ``bpgma`` runs the events- and picks-side BGMA reviews back-to-back
     (and optionally per-event waveform diagnostics) for the same target.
   - ``__iadd__`` / ``__isub__`` provide in-place catalog merge/subtract,
     used by the review workflow when materializing review-only subsets.

ARCHITECTURE:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                              ogscatalog.py                              │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                              OGSCatalog                                 │
  │   ┌─────────────────────────────────────────────────────────────────┐   │
  │   │  Daily indexing & caches (paths_, picks/events dicts)           │   │
  │   │  Aggregate frames: EVENTS, PICKS  (built on first ``get``)      │   │
  │   └─────────────────────────────────────────────────────────────────┘   │
  │                              │                                          │
  │     ┌────────────────────────┼─────────────────────────────────┐        │
  │     ▼                        ▼                                 ▼        │
  │  Plotting              BGMA (events)                    BGMA (picks)    │
  │  plot / plot_events    bgmaEvents                       bgmaPicks       │
  │  plot_cumulative_*     ─ shared & extra dates           ─ load+clean    │
  │  plot_*_histogram      ─ feasible-position matching     ─ time-window   │
  │                        ─ matched/missed/proposed         match by S/P   │
  │                        ─ CSV + diagnostic figures       ─ CSV + figs    │
  │                                                                         │
  │                                  ▼                                      │
  │                          bpgma orchestrator                             │
  │                  (events ➜ picks ➜ waveform review)                     │
  └─────────────────────────────────────────────────────────────────────────┘

ON-DISK LAYOUT EXPECTED:
  <input>/
    events/        YYYY-MM-DD.{csv,parquet}     ← event-level rows
    assignments/   YYYY-MM-DD.{csv,parquet}     ← pick-level rows
    picks/         YYYY-MM-DD.{csv,parquet}     ← additional pick rows
  Non-date / hidden stems are silently skipped at index time. Both
  ``assignments`` and ``picks`` feed the same ``picks_`` path cache.

SCHEMA NOTES:
  ``_EVENTS_COLUMNS`` / ``_PICKS_COLUMNS`` define the canonical columns of
  the aggregate frames; ``_EVENTS_MH_COLUMNS`` / ``_EVENTS_MH_WIDE_COLUMNS``
  / ``_PICKS_MH_COLUMNS`` are the BGMA review-frame layouts. Paired columns
  use either the wide layout (``{col}_base`` / ``{col}_target``) or the
  legacy tuple-valued representation; ``_tcol`` abstracts over both.

SEISMIC APPLICATIONS:
  - Daily catalog QC across long date windows without loading everything.
  - Side-by-side comparison of a baseline catalog with a model-produced
    catalog (e.g. ML phase picker / associator output) for false-discovery
    and recall analysis on both events and picks.
  - Waveform-level inspection of missed / proposed events.

USAGE:
  cat = OGSCatalog(input=Path(".../catalog"), start=t0, end=t1, name="OGS")
  events = cat.get("EVENTS")
  cat.plot(targets=[other_catalog])
  cat.bpgma(other_catalog, stations=Path(".../station"))

DEPENDENCIES:
  - numpy, pandas      : array + DataFrame operations
  - matplotlib (.path) : polygon membership for event prefiltering
  - obspy.UTCDateTime  : robust date parsing of file stems
  - ogsconstants       : shared column names and configuration constants
  - ogsutils           : logger + ``contains_points`` polygon kernel
  - ogsplotter         : (lazy import) waveform / map / histogram renderers

AUTHOR: AI2Seism Project
=============================================================================
"""

from __future__ import annotations

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
from collections import Counter
from pathlib import Path
from datetime import datetime, timedelta as td
from typing import Any, Dict, Hashable, Iterator, Literal, Optional, Sequence, cast

# =============================================================================
# THIRD-PARTY LIBRARY IMPORTS
# =============================================================================
import numpy as np                              # Array ops + boolean masks
import pandas as pd                             # All catalog frames
from obspy import UTCDateTime                   # Tolerant date-stem parsing
from matplotlib.path import Path as mplPath     # Polygon containment tests

# =============================================================================
# LOCAL PACKAGE IMPORTS
# =============================================================================
# Support both package import (``from .ogscatalog import ...``) and direct
# script execution (``python ogscatalog.py``) by falling back to flat imports.
try:
  from . import ogsconstants as OGS_C
  from . import ogsutils as OGS_U
except ImportError:
  import ogsconstants as OGS_C
  import ogsutils as OGS_U

# =============================================================================
# MODULE-LEVEL CONSTANTS — frame layouts and BGMA review schemas
# =============================================================================
# Image extension used everywhere in this module for figure output. Switching
# this single alias propagates to every plotting helper below.
IMAGE_EXT = OGS_C.PDF_EXT

# BGMA output-frame column layouts (hoisted out of bgmaEvents/bgmaPicks; pure
# `OGS_C.*` constants, so safe to build once at module load).
_EVENTS_MH_COLUMNS: list[str] = [
  OGS_C.INDEX_STR, OGS_C.TIME_STR, OGS_C.LATITUDE_STR, OGS_C.LONGITUDE_STR,
  OGS_C.DEPTH_STR, OGS_C.ERH_STR, OGS_C.ERZ_STR, OGS_C.GAP_STR,
  OGS_C.MAGNITUDE_L_STR, OGS_C.GROUPS_STR,
]
# Wide-form variant: ``{col}_base`` / ``{col}_target`` columns built from
# ``_EVENTS_MH_COLUMNS`` for matched-event review tables.
_EVENTS_MH_WIDE_COLUMNS: list[str] = [
  *(f"{col}_base" for col in _EVENTS_MH_COLUMNS),
  *(f"{col}_target" for col in _EVENTS_MH_COLUMNS),
]
_PICKS_MH_COLUMNS: list[str] = [
  OGS_C.IDX_PICKS_STR, OGS_C.TIME_STR, OGS_C.PHASE_STR,
  OGS_C.STATION_STR, OGS_C.PROBABILITY_STR,
]
# Phase labels used to shape `bgmaPicks`'s confusion matrix (P-wave row/col,
# S-wave row/col, plus the shared `NONE_STR` missed/proposed axis).
_PICKS_PHASES: tuple[str, ...] = (OGS_C.PWAVE, OGS_C.SWAVE, OGS_C.NONE_STR)
# Row/column labels for `bgmaEvents`'s 2x2 confusion matrix (matched event vs
# missed/proposed on the `NONE_STR` axis).
_EVENTS_PHASES: tuple[str, ...] = (OGS_C.EVENT_STR, OGS_C.NONE_STR)
# Source-label discriminants yielded by `_iter_shared_and_extra_dates`; typed
# as `Literal` so branch sites in `bgmaEvents`/`bgmaPicks` get static
# exhaustiveness coverage instead of free-form strings.
_DateSource = Literal["both", "base_only", "target_only"]
_BOTH: _DateSource = "both"
_BASE_ONLY: _DateSource = "base_only"
_TARGET_ONLY: _DateSource = "target_only"

# Column layouts for the catalog-wide PICKS/EVENTS frames built in __init__.
# These define the canonical schema returned by ``OGSCatalog.get("PICKS")``
# and ``OGSCatalog.get("EVENTS")``.
_PICKS_COLUMNS: list[str] = [
  OGS_C.IDX_PICKS_STR, OGS_C.GROUPS_STR, OGS_C.TIME_STR,
  OGS_C.STATION_STR, OGS_C.PHASE_STR, OGS_C.PROBABILITY_STR,
  OGS_C.EPICENTRAL_DISTANCE_STR, OGS_C.DEPTH_STR,
  OGS_C.AMPLITUDE_STR, OGS_C.STATION_ML_STR,
]
_EVENTS_COLUMNS: list[str] = [
  OGS_C.IDX_EVENTS_STR, OGS_C.TIME_STR, OGS_C.LATITUDE_STR,
  OGS_C.LONGITUDE_STR, OGS_C.DEPTH_STR, OGS_C.GAP_STR, OGS_C.ERZ_STR,
  OGS_C.ERH_STR, OGS_C.ERT_STR, OGS_C.GROUPS_STR, OGS_C.NO_STR,
  OGS_C.NUMBER_P_PICKS_STR, OGS_C.NUMBER_S_PICKS_STR,
  OGS_C.NUMBER_P_AND_S_PICKS_STR, OGS_C.ML_STR, OGS_C.ML_MEDIAN_STR,
  OGS_C.ML_UNC_STR, OGS_C.ML_STATIONS_STR,
]


# =============================================================================
# OGSCatalog — main container class
# =============================================================================
class OGSCatalog:
  """
  Lazy-loading container for OGS daily event and pick catalogs.

  The catalog indexes dated files beneath an OGS run directory, loads daily
  event and pick tables on demand, and materializes aggregate ``EVENTS`` and
  ``PICKS`` frames only when requested. Event days can be spatially filtered
  with ``polygon`` as they are loaded.

  Parameters
  ----------
  input : Path
    Root directory containing dated files in ``events`` and either
    ``assignments`` or ``picks``.
  start : datetime, optional
    Inclusive lower bound used while indexing dated files.
  end : datetime, optional
    Inclusive upper bound used while indexing dated files.
  verbose : bool, optional
    Enable verbose logger output.
  polygon : mplPath, optional
    Polygon applied to loaded event days. If None, no spatial filtering is
    performed.
  output : Path, optional
    Directory used for derived artifacts such as plots and BGMA review files.
  name : str, optional
    Catalog label used in logs and plots. When empty, ``output.name`` is used.

  Attributes
  ----------
  PICKS : pd.DataFrame
    Aggregate picks table, populated lazily from cached daily pick frames.
  EVENTS : pd.DataFrame
    Aggregate events table, populated lazily from cached daily event frames.
  picks_ : dict[date, Path]
    Date-indexed pick file paths discovered during :meth:`preload`.
  events_ : dict[date, Path]
    Date-indexed event file paths discovered during :meth:`preload`.
  picks : dict[date, pd.DataFrame]
    Daily pick frames loaded on demand and cached by date.
  events : dict[date, pd.DataFrame]
    Daily event frames loaded on demand and cached by date.
  polygon : mplPath or None
    Polygon used to spatially filter loaded event days.
  output : Path
    Base directory for generated review tables and figures.
  name : str
    Catalog label used in logs, filenames, and plot legends.

  Methods
  -------
  preload() -> None
    Index dated event and pick files for the configured date window.
  load(key: str) -> Dict[date, pd.DataFrame]
    Populate and return the daily cache for ``"events"`` or ``"picks"``.
  postload(key: str, update: bool = False) -> Dict[date, pd.DataFrame]
    Rebuild a daily cache from the aggregate in-memory DataFrame.
  get(key: str) -> pd.DataFrame
    Return the aggregate ``EVENTS`` or ``PICKS`` DataFrame.
  plot(targets: list[OGSCatalog] = [],
       vlines: list[tuple[datetime, str, str]] = []) -> None
    Generate the standard summary plots for the catalog.
  plot_events(targets: list[OGSCatalog] = [],
              output: Optional[Path] = None) -> None
    Plot event locations for this catalog and optional comparisons.
  plot_events_ms_waveforms(picks: pd.DataFrame,
                           event: pd.Series,
                           waveforms: dict[str, list[Path]],
                           output: Optional[Path] = None) -> None
    Plot missed-event waveforms for BGMA review output.
  plot_events_ps_waveforms(picks: pd.DataFrame,
                           event: pd.Series,
                           waveforms: dict[str, list[Path]],
                           output: Optional[Path] = None) -> None
    Plot proposed-event waveforms for BGMA review output.
  bgmaEvents(target: OGSCatalog, output: Optional[Path] = None) -> None
    Match events between catalogs and write BGMA review artifacts.
  bgmaPicks(target: OGSCatalog, output: Optional[Path] = None) -> None
    Match picks between catalogs and write BGMA review artifacts.
  bpgma(target: OGSCatalog,
        stations: Optional[Path] = None,
        waveforms: Optional[Path] = None,
        vlines: list[tuple[datetime, str, str]] = []) -> None
    Run the combined BGMA workflow for events, picks, and optional waveform
    review.
  """

  def __init__(self,
        input: Path,
        start: datetime = datetime.max,
        end: datetime = datetime.min,
        verbose: bool = False,
        polygon : Optional[mplPath] = mplPath(OGS_C.OGS_POLY_REGION,
                                              closed=True),
        output : Path = OGS_C.THIS_FILE.parent / "data" / "OGSCatalog",
        name: str = OGS_C.EMPTY_STR
      ) -> None:
    """Initialize the lazy-loading catalog and index available days.

    Parameters
    ----------
    input : Path
      Root directory containing dated files in ``events`` and either
      ``assignments`` or ``picks``.
    start : datetime, optional
      Inclusive lower date bound used by :meth:`preload`.
    end : datetime, optional
      Inclusive upper date bound used by :meth:`preload`.
    verbose : bool, optional
      Enable verbose logger output.
    polygon : mplPath, optional
      Polygon applied to loaded event days. If None, no spatial filtering is
      performed.
    output : Path, optional
      Directory created for derived artifacts.
    name : str, optional
      Catalog label used in logs and plots. When empty, ``output.name`` is
      used.
    """
    if not input.exists():
      raise FileNotFoundError(f"Input path {input} does not exist.")
    self.name = output.name if name == OGS_C.EMPTY_STR else name
    self.input = input
    self.start = start
    self.end = end
    self.polygon : Optional[mplPath] = polygon
    self.logger = OGS_U.setup_logger(f"{__name__}.{self.__class__.__name__}",
                                     verbose)
    self.output = output
    self.output.mkdir(parents=True, exist_ok=True)
    (self.output / "img").mkdir(parents=True, exist_ok=True)
    self.picks_: dict[datetime, Path] = {}
    self.events_: dict[datetime, Path] = {}
    self._preload_index: Optional[dict[str, list[tuple[Any, Path]]]] = None
    self.picks: dict[datetime, pd.DataFrame] = {}
    self.events: dict[datetime, pd.DataFrame] = {}
    self.waveforms: Optional[pd.DataFrame] = None
    self.stations: Optional[pd.DataFrame] = None
    self.PICKS: pd.DataFrame = pd.DataFrame(columns=_PICKS_COLUMNS)
    self.EVENTS: pd.DataFrame = pd.DataFrame(columns=_EVENTS_COLUMNS)
    self.preload()

  # -------------------------------------------------------------------------
  # CATALOG INDEXING & DAILY I/O
  # -------------------------------------------------------------------------
  # ``preload`` discovers dated files once; ``load_`` reads one file; the
  # lazy public API (``load`` / ``postload`` / ``get``) is layered on top of
  # ``_load_day`` further down in the file.
  # -------------------------------------------------------------------------

  def preload(self) -> None:
    """Index dated daily files into the path caches.

    Event files are collected from ``events``; pick-like files from both
    ``assignments`` and ``picks`` share ``self.picks_``. Only stems that
    parse as dates within the inclusive ``[start, end]`` window are retained.
    """
    start_d, end_d = self.start.date(), self.end.date()
    if self._preload_index is None:
      self._preload_index = {"events": [], "assignments": [], "picks": []}
      for filepath in self.input.rglob("*"):
        if not filepath.is_file():
          continue
        subdir = filepath.parent.name
        if subdir not in self._preload_index:
          continue
        stem = filepath.stem
        # Skip hidden/system files and non-date stems without noisy tracebacks
        if not stem or stem.startswith(".") or not stem[0].isdigit():
          continue
        try:
          date = UTCDateTime(stem).date
        except Exception:
          self.logger.debug("Skipping non-date filename: %s", stem)
          continue
        self._preload_index[subdir].append((date, filepath))

    for subdir, target in (("events", self.events_),
                           ("assignments", self.picks_),
                           ("picks", self.picks_)):
      for date, filepath in self._preload_index[subdir]:
        if start_d <= date <= end_d:
          target[date] = filepath

  def load_(self, filepath: Path) -> pd.DataFrame:
    """Load one daily file from disk.

    Parameters
    ----------
    filepath : Path
      Daily file path. CSV suffixes are read with :func:`pandas.read_csv`;
      all other suffixes are treated as parquet input.

    Returns
    -------
    pd.DataFrame
      Loaded DataFrame, or an empty frame after logging the read failure.
    """
    try:
      if filepath.suffix == OGS_C.CSV_EXT:
        return pd.read_csv(filepath)
      return pd.read_parquet(filepath)
    except Exception:
      self.logger.exception(f"Error loading {filepath}")
      return pd.DataFrame(columns=[])

  # -------------------------------------------------------------------------
  # CONFUSION-MATRIX & METRIC HELPERS
  # -------------------------------------------------------------------------
  # Small, side-effect-free utilities shared by BGMA events and picks. They
  # build square confusion frames, accumulate counts safely with ``int``
  # coercion, and log recall / false-discovery rates under consistent names.
  # -------------------------------------------------------------------------

  def _empty_cfn_mtx(self, axes: Sequence[str]) -> pd.DataFrame:
    """
    Return a square zero-initialized integer confusion matrix for ``axes``.
    """
    return pd.DataFrame(0, index=list(axes), columns=list(axes), dtype=int)

  def _add(self, mtx: pd.DataFrame, row: Hashable, col: Hashable,
           n: int = 1) -> None:
    """
    Increment one confusion-matrix cell by ``n`` with explicit ``int``
    coercion.
    """
    mtx.at[row, col] = cast(int, mtx.at[row, col]) + int(n)

  def _add_series(
    self, mtx: pd.DataFrame, key: Hashable, counts: pd.Series,
    *, axis: Literal[0, 1]
  ) -> None:
    """Add pre-aggregated counts into one row or column of ``mtx``.

    ``axis=0`` writes column ``key`` using ``counts.index`` as row labels.
    ``axis=1`` writes row ``key`` using ``counts.index`` as column labels.
    Empty input is a no-op.
    """
    if counts.empty:
      return
    valid_axis = mtx.columns if axis == 1 else mtx.index
    valid_set = {str(x) for x in valid_axis}
    for label, n in counts.items():
      slabel = str(label)
      if slabel not in valid_set:
        continue
      if axis == 1:
        self._add(mtx, key, slabel, int(n))
      else:
        self._add(mtx, slabel, key, int(n))

  def _count(self, mtx: pd.DataFrame, row: Hashable, col: Hashable) -> int:
    """Return one confusion-matrix cell as ``int``."""
    return cast(int, mtx.at[row, col])

  @staticmethod
  def _safe_ratio(numerator: int, denominator: int) -> float:
    """
    Return ``numerator / denominator``, or ``0.0`` when ``denominator`` is
    zero.
    """
    return numerator / denominator if denominator else 0.0

  def _log_rate(self, name: str, num: int, den: int,
                suffix: str = "") -> float:
    """Log and return the guarded ratio ``num / den`` under ``name``."""
    rate = self._safe_ratio(num, den)
    self.logger.info("%s%s: %s", name, suffix, rate)
    return rate

  def _log_recall_fdr(
    self,
    correct: int,
    missed: int,
    proposed: int,
    *,
    swapped: int = 0,
    label: str = "",
  ) -> tuple[float, float]:
    """Compute and log overall recall and false discovery rate.

    Parameters
    ----------
    correct : int
      Count of true positives.
    missed : int
      Count of base-only items added to the recall denominator.
    proposed : int
      Count of target-only items used as the false-discovery numerator.
    swapped : int, optional
      Count of phase-mismatched matched picks; excluded from the recall
      numerator and included in both denominators.
    label : str, optional
      Suffix appended to the logged metric names.

    Returns
    -------
    tuple[float, float]
      ``(recall, fdr)``.
    """
    suffix = f" {label}" if label else ""
    recall = self._log_rate(
      "Recall", correct, correct + swapped + missed, suffix
    )
    fdr = self._log_rate(
      "False Discovery Rate", proposed, proposed + correct + swapped, suffix
    )
    return recall, fdr

  def _picks_phase_metrics(
    self, mtx: pd.DataFrame, phase: str, other_phase: str,
  ) -> tuple[float, float]:
    """Compute and log per-phase pick recall/FDR from the 3x3 picks matrix.

    Rows are BASE phases and columns are TARGET phases. ``phase`` selects the
    wave being scored; ``other_phase`` supplies the swapped-phase row/column.
    """
    pp = cast(int, mtx.at[phase, phase])
    po = cast(int, mtx.at[phase, other_phase])
    pn = cast(int, mtx.at[phase, OGS_C.NONE_STR])
    op = cast(int, mtx.at[other_phase, phase])
    np_ = cast(int, mtx.at[OGS_C.NONE_STR, phase])
    suffix = f" {phase}-wave"
    recall = self._log_rate("Recall", pp, pp + po + pn, suffix)
    fdr = self._log_rate("False Discovery Rate", np_, np_ + pp + op, suffix)
    return recall, fdr

  # -------------------------------------------------------------------------
  # PAIRED-COLUMN SCHEMA HELPERS (wide vs. legacy tuple representation)
  # -------------------------------------------------------------------------
  # BGMA review tables can live in two schemas: wide
  # (``{col}_base`` / ``{col}_target``) or legacy (tuple-valued ``col``).
  # ``_tcol`` is the single entry point used by every downstream plotter.
  # -------------------------------------------------------------------------

  def _legacy_tcols(
    self, df: pd.DataFrame, col: str,
  ) -> tuple[pd.Series, pd.Series]:
    """Return cached base/target series for a legacy tuple-valued column."""
    cache = df.attrs.get("_legacy_tcol_cache")
    if not isinstance(cache, dict):
      cache = {}
      df.attrs["_legacy_tcol_cache"] = cache
    cached = cache.get(col)
    if cached is None:
      series = df[col]
      cached = (
        series.apply(lambda value: value[0]),
        series.apply(lambda value: value[1]),
      )
      cache[col] = cached
    return cast(tuple[pd.Series, pd.Series], cached)

  def _tcol(self, df: pd.DataFrame, col: str, idx: Literal[0, 1]) -> pd.Series:
    """Return one side of a paired column from either supported schema.

    Prefers wide columns named ``{col}_base`` / ``{col}_target`` and falls
    back to the legacy tuple-valued ``col`` representation.
    """
    suffix = "base" if idx == 0 else "target"
    wide_col = f"{col}_{suffix}"
    if wide_col in df.columns:
      return df[wide_col]
    return self._legacy_tcols(df, col)[idx]

  def _magnitude_or_none(self, df: pd.DataFrame) -> Optional[pd.Series]:
    """Return scalar ``MAGNITUDE_L`` values when present and not all-null."""
    if OGS_C.MAGNITUDE_L_STR not in df.columns:
      return None
    series = df[OGS_C.MAGNITUDE_L_STR]
    return series if series.notna().any() else None

  def _magnitude_tuple_or_none(
    self, df: pd.DataFrame, idx: Literal[0, 1],
  ) -> Optional[pd.Series]:
    """Return one side of paired ``MAGNITUDE_L`` data when available.

    Supports both wide-schema columns
    ``MAGNITUDE_L_base`` / ``MAGNITUDE_L_target`` and the legacy tuple-valued
    ``MAGNITUDE_L`` column. Returns ``None`` when the selected side is absent
    or entirely null.
    """
    suffix = "base" if idx == 0 else "target"
    wide_col = f"{OGS_C.MAGNITUDE_L_STR}_{suffix}"
    if wide_col not in df.columns and OGS_C.MAGNITUDE_L_STR not in df.columns:
      return None
    series = self._tcol(df, OGS_C.MAGNITUDE_L_STR, idx)
    return series if series.notna().any() else None

  # -------------------------------------------------------------------------
  # POLYGON / SPATIAL FILTERING
  # -------------------------------------------------------------------------
  # Optional containment filtering applied to events as they are loaded; the
  # same polygon also constrains BGMA candidate pairs to a shared domain.
  # -------------------------------------------------------------------------

  def _polygon_vertices(self) -> Optional[np.ndarray[Any, Any]]:
    """Return polygon vertices when spatial filtering is enabled."""
    polygon = self.polygon
    if polygon is None:
      return None
    vertices = getattr(polygon, "vertices", None)
    if not isinstance(vertices, np.ndarray):
      raise TypeError("Configured polygon must expose NumPy vertices.")
    return vertices

  def _event_candidate_mask(
    self,
    events: pd.DataFrame,
    polygon_vertices: Optional[np.ndarray[Any, Any]],
  ) -> np.ndarray[Any, Any]:
    """Return the polygon-membership mask for BGMA event candidates.

    When ``polygon_vertices`` is ``None``, every row remains eligible.
    Otherwise the mask selects events whose hypocenters lie inside the
    supplied polygon.
    """
    if polygon_vertices is None or events.empty:
      return np.ones(len(events.index), dtype=bool)
    return OGS_U.contains_points(
      polygon_vertices,
      events[[OGS_C.LONGITUDE_STR, OGS_C.LATITUDE_STR]].to_numpy()
    )

  def _prefilter_events(
    self,
    events: pd.DataFrame,
    polygon_vertices: Optional[np.ndarray[Any, Any]],
    filtered_frames: list[pd.DataFrame],
    date: datetime,
    label: str,
  ) -> pd.DataFrame:
    """Project one event day into the shared BGMA domain.

    Rows outside ``polygon_vertices`` are diverted to ``filtered_frames`` as
    review-only partitions (``EventsSM`` for BASE, ``EventsSP`` for TARGET)
    and never enter BGMA matching or the 2x2 event confusion matrix.
    """
    if events.empty:
      return events.reset_index(drop=True)
    in_region_mask = self._event_candidate_mask(events, polygon_vertices)
    if not np.all(in_region_mask):
      filtered = events.loc[~in_region_mask].reindex(columns=_EVENTS_MH_COLUMNS)
      filtered_frames.append(filtered.reset_index(drop=True))
      self.logger.warning(
        "DATE %s: %s events filtered by polygon domain: %d",
        date, label, int((~in_region_mask).sum())
      )
    return events.loc[in_region_mask].reset_index(drop=True)

  # -------------------------------------------------------------------------
  # LAZY DAILY CACHE & AGGREGATE FRAME BUILDERS
  # -------------------------------------------------------------------------
  # Day-keyed cache lookup, aggregate concatenation, and the inverse
  # ``postload`` path that rebuilds per-day caches from the aggregate frame.
  # -------------------------------------------------------------------------

  def _load_day(self, key: str, date) -> pd.DataFrame:
    """Return one cached day for ``key``, loading it on first access.

    For ``events``, polygon filtering is applied after load when
    ``self.polygon`` is set. Cached days are returned without disk I/O.

    Parameters
    ----------
    key : str
      Either "events" or "picks".
    date : datetime.date
      The date whose data to load.

    Returns
    -------
    pd.DataFrame
      Daily data for ``date`` after any event-side polygon filtering.
    """
    if key == "events":
      cache, paths = self.events, self.events_
    elif key == "picks":
      cache, paths = self.picks, self.picks_
    else:
      raise ValueError(f"Unknown key: {key}")
    if date in cache:
      return cache[date]
    path = paths[date]
    df = self.load_(path)
    if key == "events":
      if not df.empty:
        polygon_vertices = self._polygon_vertices()
        if polygon_vertices is not None:
          mask = OGS_U.contains_points(
            polygon_vertices,
            df[[OGS_C.LONGITUDE_STR, OGS_C.LATITUDE_STR]].to_numpy()
          )
          df = df[mask]
        if df.empty:
          self.logger.warning(f"All events for {date} filtered out by polygon "
                              f"from {path}")
      else:
        self.logger.warning(f"No events loaded for {date} from {path}")
    cache[date] = df
    return df

  def load(self, key: str,
  ) -> Dict[datetime, pd.DataFrame]:
    """Populate and return the per-day cache for ``key``.

    Only indexed days missing from the cache are loaded; previously cached days
    are returned unchanged.

    Parameters
    ----------
    key : str
      Either "events" or "picks".

    Returns
    -------
    Dict[datetime, pd.DataFrame]
      Mapping of indexed days to daily DataFrames currently held in cache.
    """
    if key not in ("events", "picks"):
      raise ValueError(f"Unknown key: {key}")
    cache = getattr(self, key)
    missing = set(getattr(self, f"{key}_").keys()) - set(cache.keys())
    if missing:
      self.logger.info(f"Loading {self.name} {key} data...")
      for date in missing:
        self._load_day(key, date)
    return cache

  def postload(self, key: str,
               update: bool = False) -> Dict[datetime, pd.DataFrame]:
    """Populate a per-day cache from the aggregate in-memory DataFrame.

    When ``update`` is ``False`` and the aggregate frame is non-empty, rows are
    grouped by ``GROUPS_STR`` and stored back into the daily cache. When
    ``update`` is ``True``, the existing cache is returned unchanged.

    Parameters
    ----------
    key : str
      Either "events" or "picks".
    update : bool, optional
      Whether to skip rebuilding the cache from the aggregate frame.

    Returns
    -------
    Dict[datetime, pd.DataFrame]
      Mapping of cached days to daily DataFrames.
    """
    if key not in ("events", "picks"):
      raise ValueError(f"Unknown key: {key}")
    cache = getattr(self, key)
    df = getattr(self, key.upper())
    if not update and not df.empty:
      for date, day_df in df.groupby(OGS_C.GROUPS_STR):
        cache[UTCDateTime(date).date] = day_df
    return cache

  def get(self, key: str) -> pd.DataFrame:
    """Return the aggregate ``EVENTS`` or ``PICKS`` DataFrame.

    If the aggregate frame is empty, the corresponding daily cache is loaded on
    demand and concatenated once. When no daily data are available, the empty
    aggregate frame is returned after logging a warning.

    Parameters
    ----------
    key : str
      Either "EVENTS" or "PICKS".

    Returns
    -------
    pd.DataFrame
      Aggregate DataFrame currently cached for ``key``.
    """
    if key not in ("EVENTS", "PICKS"):
      raise ValueError(f"Unknown key: {key}")
    if getattr(self, key).empty:
      self.logger.info(f"Loading {self.name} {key} data...")
      daily = self.load(key.lower())
      if daily:
        setattr(self, key,
                pd.concat(daily.values()).reset_index(drop=True))
      else:
        self.logger.warning(f"No {self.name} {key} data loaded.")
    return getattr(self, key)

  # =========================================================================
  # WAVEFORM DIAGNOSTICS (per-event missed/proposed plots)
  # =========================================================================
  # These helpers slice a small pick window around an event origin time and
  # render single-event waveform figures used for BGMA review.
  # =========================================================================

  def _waveform_pick_window(self, event_time: Any) -> pd.DataFrame:
    """Return the waveform pick window using a cached time-sorted index."""
    picks = self.PICKS
    cache = picks.attrs.get("_waveform_pick_window_cache")
    if (not isinstance(cache, dict) or
        cache.get("row_count") != len(picks)):
      indexed_picks = picks.assign(
        _waveform_row_position=np.arange(len(picks))
        ).sort_values(by=OGS_C.TIME_STR, kind="mergesort")
      cache = {
        "row_count": len(picks),
        "indexed_picks": indexed_picks,
        "sorted_times": pd.Index(indexed_picks[OGS_C.TIME_STR]),
      }
      picks.attrs["_waveform_pick_window_cache"] = cache
    indexed_picks = cast(pd.DataFrame, cache["indexed_picks"])
    sorted_times = cast(pd.Index, cache["sorted_times"])
    left = sorted_times.searchsorted(event_time - td(seconds=1), side="left")
    right = sorted_times.searchsorted(event_time + td(seconds=30),
                                      side="right")
    window = cast(pd.DataFrame, indexed_picks.iloc[left:right])
    return window.sort_values(
      by="_waveform_row_position", kind="mergesort"
    ).drop(columns="_waveform_row_position")

  def _plot_event_waveforms(self,
        picks: pd.DataFrame,
        event: pd.Series,
        waveforms: dict[str, list[Path]],
        kind: str,
        label: str,
        output: Optional[Path] = None
      ) -> None:
    """Render a single-event waveform diagnostic figure.

    Parameters
    ----------
    picks : pd.DataFrame
      Picks passed to the waveform plotter for the event under inspection.
    event : pd.Series
      Event metadata used for the title, filename, and local pick window.
    waveforms : dict[str, list[Path]]
      Waveform files grouped by station for the event.
    kind : str
      Short classification tag added to the title and default filename.
    label : str
      Human-readable classification label shown in the plot title.
    output : Optional[Path], optional
      Explicit output path for the rendered figure. When omitted, the figure is
      written under ``self.output / "img"`` using catalog and event metadata.

    Raises
    ------
    ValueError
      If the station inventory has not been loaded.

    Notes
    -----
    This helper creates a waveform diagnostic plot, not a matched catalog plot.
    It also adds a flipped overlay for picks in ``self.PICKS`` between 1 second
    before and 30 seconds after the event origin time, writes the figure, and
    closes the Matplotlib figure before returning.
    """
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    if self.stations is None:
      raise ValueError(
        "Station inventory must be loaded before plotting waveforms"
      )
    stations = cast(pd.DataFrame, self.stations)
    ml = (f" ($M_L$ {event[OGS_C.MAGNITUDE_L_STR]})"
          if OGS_C.MAGNITUDE_L_STR in event else OGS_C.EMPTY_STR)
    plot = OGS_P.event_plotter(
      picks=picks,
      event=event,
      stations=list(stations[OGS_C.STATION_STR].unique()),
      waveforms=waveforms,
      inventory=stations,
      title=(
        f"{label} ({kind}) Event {event[OGS_C.IDX_EVENTS_STR]}" + ml +
        f" | Proposed (PS) Picks {event[OGS_C.TIME_STR] - td(seconds=1)}"
      ),
    )
    window = self._waveform_pick_window(event[OGS_C.TIME_STR])
    plot.add_plot(picks=window, flip=True,
      output=(
        output if output is not None else
        self.output / "img" / (
          f"{self.input.name}_{kind}{event[OGS_C.GROUPS_STR]}"
          f"_{event[OGS_C.IDX_EVENTS_STR]}" + IMAGE_EXT
        )
      )
    )
    plt.close()

  # =========================================================================
  # CATALOG-LEVEL PLOTTING (maps, cumulative curves, histograms)
  # =========================================================================
  # Public summary plots and their shared helpers. All of these read from the
  # aggregate ``EVENTS`` / ``PICKS`` frames and optionally overlay one or
  # more comparison catalogs supplied as ``targets``.
  # =========================================================================

  def plot_events(self, targets: list[OGSCatalog] = [],
                  output: Optional[Path] = None) -> None:
    """Plot event locations for this catalog and optional catalog overlays.

    Parameters
    ----------
    targets : list[OGSCatalog], optional
      Additional catalogs to overlay on the same map for comparison.
    output : Optional[Path], optional
      Explicit output path for the rendered map. When omitted, the figure is
      written under ``self.output / "img"`` using catalog-derived filenames.

    Notes
    -----
    This entrypoint produces catalog location maps, including optional matched
    or comparison catalog overlays supplied via ``targets``. It does not create
    waveform diagnostic plots. The method lazy-loads event tables through
    :meth:`get`, logs and returns when a catalog has no events, writes the map,
    and closes the Matplotlib figure before returning.
    """
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    events = self.get("EVENTS")
    if events.empty:
      self.logger.info("No events to plot.")
      return
    eventsMap = OGS_P.map_plotter(
      OGS_C.OGS_STUDY_REGION,
      x=events[OGS_C.LONGITUDE_STR],
      y=events[OGS_C.LATITUDE_STR],
      legend=True,
      marker='o',
      color="none",
      facecolors='none',
      edgecolors=OGS_C.OGS_BLUE,
      label=self.name,
      output=(
        output if output is not None else
        self.output / "img" / (
          f"{self.input.name}_EventsMap" +
          IMAGE_EXT
        )
      ),
      magnitude=events[OGS_C.MAGNITUDE_L_STR] \
        if OGS_C.MAGNITUDE_L_STR in events.columns else None
    )
    for target, color in zip(targets, OGS_C.PLOT_COLORS[1:len(targets)+1]):
      events = target.get("EVENTS")
      if events.empty:
        self.logger.info(f"No events to plot for {target.name}.")
        continue
      else:
        self.logger.info(f"Plotting events for {target.name}.")
      eventsMap.add_plot(
        x=events[OGS_C.LONGITUDE_STR],
        y=events[OGS_C.LATITUDE_STR],
        legend=True,
        marker='o',
        color="none",
        facecolors='none',
        edgecolors=color,
        label=target.name,
        output=(
          output if output is not None else
          self.output / "img" / (
            f"{self.input.name}_{target.input.name}_EventsMap" +
            IMAGE_EXT
          )
        ),
        magnitude=events[OGS_C.MAGNITUDE_L_STR] \
          if OGS_C.MAGNITUDE_L_STR in events.columns else None
      )
    plt.close()

  def plot(self,
        targets: list[OGSCatalog] = [],
        vlines: list[tuple[datetime, str, str]] = []
      ) -> None:
    """Generate the catalog-level summary plot set.

    Parameters
    ----------
    targets : list[OGSCatalog], optional
      Additional catalogs to include in the comparison summaries.
    vlines : list[tuple[datetime, str, str]], optional
      Vertical reference markers forwarded to the cumulative event and pick
      plots as ``(time, label, color)`` tuples.

    Notes
    -----
    This is the main catalog plotting entrypoint for map, histogram, and
    cumulative matched/comparison plots. It does not generate waveform
    diagnostic figures. Each delegated plot writes its own output file when no
    explicit path is provided by that lower-level method.
    """
    self.plot_events(targets=targets)
    self.plot_erh_histogram(targets=targets)
    self.plot_erz_histogram(targets=targets)
    self.plot_ert_histogram(targets=targets)
    self.plot_magnitude_histogram(targets=targets)
    self.plot_depth_histogram(targets=targets)
    self.plot_cumulative_events(targets=targets, vlines=vlines)
    self.plot_cumulative_picks(targets=targets, vlines=vlines)

  def plot_events_ms_waveforms(self,
        picks: pd.DataFrame,
        event: pd.Series,
        waveforms: dict[str, list[Path]],
        output: Optional[Path] = None
      ) -> None:
    """Plot a waveform diagnostic figure for one Missed (MS) event.

    Parameters
    ----------
    picks : pd.DataFrame
      Picks passed to the waveform plotter for the selected event.
    event : pd.Series
      Event row containing the metadata used in the title and filename.
    waveforms : dict[str, list[Path]]
      Waveform file paths grouped by station.
    output : Optional[Path], optional
      Explicit output path for the rendered figure. When omitted, the helper
      writes under ``self.output / "img"``.

    Notes
    -----
    This method is for per-event waveform diagnostics only. It delegates to
    :meth:`_plot_event_waveforms` and is separate from the catalog-level matched
    or comparison plots produced by :meth:`plot` and :meth:`plot_events`.
    """
    self._plot_event_waveforms(picks, event, waveforms,
                               kind="MS", label="Missed", output=output)

  def plot_events_ps_waveforms(self,
        picks: pd.DataFrame,
        event: pd.Series,
        waveforms: dict[str, list[Path]],
        output: Optional[Path] = None
      ) -> None:
    """Plot a waveform diagnostic figure for one Proposed (PS) event.

    Parameters
    ----------
    picks : pd.DataFrame
      Picks passed to the waveform plotter for the selected event.
    event : pd.Series
      Event row containing the metadata used in the title and filename.
    waveforms : dict[str, list[Path]]
      Waveform file paths grouped by station.
    output : Optional[Path], optional
      Explicit output path for the rendered figure. When omitted, the helper
      writes under ``self.output / "img"``.

    Notes
    -----
    This method is for per-event waveform diagnostics only. It delegates to
    :meth:`_plot_event_waveforms` and is separate from the catalog-level matched
    or comparison plots produced by :meth:`plot` and :meth:`plot_events`.
    """
    self._plot_event_waveforms(picks, event, waveforms,
                               kind="PS", label="Proposed", output=output)

  # -------------------------------------------------------------------------
  # Cumulative-count curves (events / picks over time)
  # -------------------------------------------------------------------------

  def _plot_cumulative(self, kind, title, file_suffix, targets, output,
                       vlines):
    """Shared cumulative plot helper used by the public cumulative wrappers.

    Parameters
    ----------
    kind : str
      Catalog key passed to :meth:`get`; the public wrappers currently use
      ``"PICKS"`` and ``"EVENTS"``.
    title : str
      Figure title used for the base plot and every comparison overlay.
    file_suffix : str
      Suffix appended to the auto-generated filename when ``output`` is not
      provided.
    targets : list[OGSCatalog]
      Additional catalogs overlaid on top of this catalog's cumulative series.
    output : Optional[Path]
      Explicit output path reused for the base figure and every overlay. When
      omitted, files are written under ``self.output / "img"`` as
      ``<self.input.name>_<file_suffix>`` for the base plot and
      ``<self.input.name>_<target.input.name>_<file_suffix>`` for comparison
      outputs.
    vlines : list[tuple[datetime, str, str]]
      Reference lines forwarded to :func:`ogsplotter.day_plotter`.

    Notes
    -----
    :meth:`plot_cumulative_picks` and :meth:`plot_cumulative_events` are thin
    wrappers around this helper; they only fix the table key, title, and output
    suffix.
    """
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    def _path(name_extra=""):
      if output is not None:
        return output
      stem = f"{self.input.name}{name_extra}_{file_suffix}"
      return self.output / "img" / (stem + IMAGE_EXT)
    cache_attr = "_plot_cumulative_groups_cache"
    def _series(cat):
      df = cat.get(kind)
      has_groups = OGS_C.GROUPS_STR in df.columns
      cache = getattr(cat, cache_attr, None)
      if not isinstance(cache, dict):
        cache = {}
        setattr(cat, cache_attr, cache)
      cached = cache.get(kind)
      if (not isinstance(cached, dict)
          or cached.get("frame_id") != id(df)
          or cached.get("row_count") != len(df)
          or cached.get("has_groups") != has_groups):
        series = None
        if not df.empty and has_groups:
          series = df.sort_values(OGS_C.GROUPS_STR)[OGS_C.GROUPS_STR]
        cached = {
          "frame_id": id(df),
          "row_count": len(df),
          "has_groups": has_groups,
          "series": series,
        }
        cache[kind] = cached
      return cached["series"]
    series = _series(self)
    if series is None:
      self.logger.info("No Date data available for histogram.")
      return
    cumulative = OGS_P.day_plotter(
      picks=series, title=title, output=_path(), label=self.name,
      color=OGS_C.OGS_BLUE, vlines=vlines,
    )
    for target, color in zip(targets, OGS_C.PLOT_COLORS[1:]):
      if not isinstance(target, OGSCatalog):
        raise ValueError(f"Can only perform {title} with OGSCatalog")
      tseries = _series(target)
      if tseries is None:
        self.logger.info("No Date data available for histogram.")
        continue
      cumulative.add_plot(
        picks=tseries, title=title,
        output=_path(f"_{target.input.name}"),
        label=target.name, legend=True, color=color,
      )
    plt.close()

  def plot_cumulative_picks(self,
                            targets: list[OGSCatalog] = [],
                            output: Optional[Path] = None,
                            vlines: list[tuple[datetime, str, str]] = []):
    """Public wrapper around :meth:`_plot_cumulative` for pick counts.

    ``targets`` are overlaid as comparison curves and ``vlines`` are forwarded
    unchanged to the shared helper. When ``output`` is omitted, the helper
    writes ``<self.input.name>_CumulativePicks`` under ``self.output / "img"``;
    comparison outputs use the helper's target-suffixed naming convention.
    """
    self._plot_cumulative("PICKS", "Cumulative Picks", "CumulativePicks",
                          targets, output, vlines)

  def plot_cumulative_events(self,
                             targets: list[OGSCatalog] = [],
                             output: Optional[Path] = None,
                             vlines: list[tuple[datetime, str, str]] = []):
    """Public wrapper around :meth:`_plot_cumulative` for event counts.

    ``targets`` are overlaid as comparison curves and ``vlines`` are forwarded
    unchanged to the shared helper. When ``output`` is omitted, the helper
    writes ``<self.input.name>_CumulativeEvents`` under ``self.output / "img"``;
    comparison outputs use the helper's target-suffixed naming convention.
    """
    self._plot_cumulative("EVENTS", "Cumulative Events", "CumulativeEvents",
                          targets, output, vlines)

  # -------------------------------------------------------------------------
  # Per-column histograms (error ellipsoids, depth, magnitude, ...)
  # -------------------------------------------------------------------------

  def _plot_histogram(self,
        column: str,
        xlabel: str,
        title: str,
        file_suffix: str,
        targets: list[OGSCatalog] = [],
        bins: int = OGS_C.NUM_BINS,
        output: Optional[Path] = None,
        **plotter_kwargs
      ) -> None:
    """Shared histogram helper used by the public event-metric wrappers.

    Parameters
    ----------
    column : str
      Event-table column name to histogram.
    xlabel : str
      X-axis label.
    title : str
      Plot title.
    file_suffix : str
      Suffix appended to the auto-generated filename when ``output`` is not
      provided (for example ``"ERZ"`` or ``"MagL"``).
    targets : list[OGSCatalog], optional
      Additional catalogs overlaid on top of this catalog's histogram.
    bins : int, optional
      Number of histogram bins forwarded to
      :func:`ogsplotter.histogram_plotter`.
    output : Optional[Path], optional
      Explicit output path reused for the base figure and every overlay. When
      omitted, files are written under ``self.output / "img"`` as
      ``<self.input.name>_<file_suffix>`` for the base histogram and
      ``<self.input.name>_<target.input.name>_<file_suffix>`` for comparison
      outputs.
    **plotter_kwargs
      Additional keyword arguments passed to ``histogram_plotter()``.

    Notes
    -----
    :meth:`plot_erz_histogram`, :meth:`plot_erh_histogram`,
    :meth:`plot_ert_histogram`, :meth:`plot_depth_histogram`, and
    :meth:`plot_magnitude_histogram` are thin wrappers around this helper; they
    select the column, labels, file suffix, and any plotter-specific kwargs.
    """
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    def _path(name_extra=""):
      if output is not None:
        return output
      stem = f"{self.input.name}{name_extra}_{file_suffix}"
      return self.output / "img" / (stem + IMAGE_EXT)
    events = self.get("EVENTS")
    if events.empty or column not in events.columns:
      self.logger.info(f"No {title} data available for histogram.")
      return
    hist = OGS_P.histogram_plotter(
      data=events[column].dropna(), bins=bins, xlabel=xlabel,
      ylabel="Number of Events", title=title, output=_path(),
      label=self.name, **plotter_kwargs,
    )
    for target, color in zip(targets, OGS_C.PLOT_COLORS[1:]):
      if not isinstance(target, OGSCatalog):
        raise ValueError(f"Can only perform {title} with OGSCatalog")
      target_events = target.get("EVENTS")
      if target_events.empty or column not in target_events.columns:
        self.logger.info(
          f"No {title} data available for histogram for {target.name}.")
        continue
      hist.add_plot(
        data=target_events[column].dropna(), xlabel=xlabel,
        ylabel="Number of Events", title=title, legend=True, alpha=0.5,
        label=target.name, color=color,
        output=_path(f"_{target.input.name}"),
      )
    plt.close()

  def plot_erz_histogram(self, targets=[], bins=OGS_C.NUM_BINS, output=None):
    """Public wrapper around :meth:`_plot_histogram` for event ERZ values.

    ``targets`` are overlaid as comparison histograms, ``bins`` is forwarded
    unchanged, and the default output name is ``<self.input.name>_ERZ`` under
    ``self.output / "img"``.
    """
    self._plot_histogram(
      OGS_C.ERZ_STR, "ERZ (km)", "ERZ Histogram", "ERZ",
      targets=targets, bins=bins, output=output, color=OGS_C.OGS_BLUE,
      xlim=(0, 20))

  def plot_erh_histogram(self, targets=[], bins=OGS_C.NUM_BINS, output=None):
    """Public wrapper around :meth:`_plot_histogram` for event ERH values.

    ``targets`` are overlaid as comparison histograms, ``bins`` is forwarded
    unchanged, and the default output name is ``<self.input.name>_ERH`` under
    ``self.output / "img"``.
    """
    self._plot_histogram(
      OGS_C.ERH_STR, "ERH (km)", "ERH Histogram", "ERH",
      targets=targets, bins=bins, output=output, color=OGS_C.OGS_BLUE,
      xlim=(0, 20), yscale='log')

  def plot_ert_histogram(self, targets=[], bins=OGS_C.NUM_BINS, output=None):
    """Public wrapper around :meth:`_plot_histogram` for event ERT values.

    ``targets`` are overlaid as comparison histograms, ``bins`` is forwarded
    unchanged, and the default output name is ``<self.input.name>_ERT`` under
    ``self.output / "img"``.
    """
    self._plot_histogram(
      OGS_C.ERT_STR, "ERT (s)", "ERT Histogram", "ERT",
      targets=targets, bins=bins, output=output)

  def plot_depth_histogram(self, targets=[], bins=OGS_C.NUM_BINS, output=None):
    """Public wrapper around :meth:`_plot_histogram` for event depth values.

    ``targets`` are overlaid as comparison histograms, ``bins`` is forwarded
    unchanged, and the default output name is ``<self.input.name>_Depth`` under
    ``self.output / "img"``.
    """
    self._plot_histogram(
      OGS_C.DEPTH_STR, "Depth (km)", "Depth Histogram", "Depth",
      targets=targets, bins=bins, output=output, xlim=(0, 50))

  def plot_magnitude_histogram(self, targets=[], bins=OGS_C.NUM_BINS,
                               output=None):
    """Public wrapper around :meth:`_plot_histogram` for event magnitudes.

    ``targets`` are overlaid as comparison histograms, ``bins`` is forwarded
    unchanged, and the default output name is ``<self.input.name>_MagL`` under
    ``self.output / "img"``.
    """
    self._plot_histogram(
      OGS_C.MAGNITUDE_L_STR, "Magnitude ($M_L$)", "Magnitude Histogram",
      "MagL", targets=targets, bins=bins, output=output, yscale='log',
      xlim=(-1, 5))

  # =========================================================================
  # BGMA EVENT MATCHING & REVIEW
  # =========================================================================
  # Day-by-day matching of base vs. target events into three partitions:
  # matched (MH), base-only (SM = silenced/missed), and target-only
  # (SP = surplus/proposed). Drives a 2x2 event confusion matrix and a set
  # of comparison CSVs and figures.
  # =========================================================================

  def _log_review_checks(
    self,
    checks: dict[str, dict[str, object]],
    kind: str,
  ) -> None:
    """Log aggregate review totals for each labeled partition.

    Each ``checks`` entry is expected to provide ``check_sum``,
    ``expected_sum``, and a ``bgma`` mapping used for mismatch detail output.
    The helper also stores the computed ``diff`` back into each review dict.
    """
    sep_pre, sep_diff = (",  ", "  ") if kind == "picks" else (", ", " ")
    mismatch_fmt = (
      f"[REVIEW] MISMATCH: %s {kind} total: %d{sep_pre}expected: %d{sep_diff}diff: %d"
    )
    mismatch_detail_fmt = f"[REVIEW] MISMATCH: %s {kind}:\n%s"
    match_fmt = f"[REVIEW]  MATCH  : %s {kind} total: %d{sep_pre}expected: %d"
    for _label, _review in checks.items():
      _check_sum = cast(int, _review["check_sum"])
      _expected_sum = cast(int, _review["expected_sum"])
      _review["diff"] = _check_sum - _expected_sum
      if _check_sum != _expected_sum:
        self.logger.error(
          mismatch_fmt,
          _label, _check_sum, _expected_sum, _check_sum - _expected_sum,
        )
        self.logger.error(mismatch_detail_fmt, _label, _review["bgma"])
        exit()
      else:
        self.logger.info(match_fmt, _label, _check_sum, _expected_sum)

  def _bgma_events_review(
    self, target: "OGSCatalog", EVENTS_CFN_MTX: pd.DataFrame,
  ) -> tuple[float, float]:
    """Log event-level BGMA totals, persist review partitions, and return rates.

    Counts are derived from ``EVENTS_CFN_MTX`` and checked against
    ``self.EVENTS`` and ``target.EVENTS`` before writing ``EventsMH``,
    ``EventsMS``, and ``EventsPS`` CSV outputs. The returned ``(recall, fdr)``
    pair is used by the caller when titling the event confusion-matrix plot.
    """
    base_events = self.get("EVENTS")
    target_events = target.get("EVENTS")
    base_n = len(base_events.index)
    target_n = len(target_events.index)
    events_mh = cast(pd.DataFrame, self.EventsMH)
    events_ms = cast(pd.DataFrame, self.EventsMS)
    events_sm = cast(
      pd.DataFrame,
      getattr(self, "EventsSM", pd.DataFrame(columns=_EVENTS_MH_COLUMNS)),
    )
    events_ps = cast(pd.DataFrame, self.EventsPS)
    events_sp = cast(
      pd.DataFrame,
      getattr(self, "EventsSP", pd.DataFrame(columns=_EVENTS_MH_COLUMNS)),
    )
    mh_n = len(events_mh.index)
    ms_n = len(events_ms.index)
    sm_n = len(events_sm.index)
    ps_n = len(events_ps.index)
    sp_n = len(events_sp.index)
    matched_count = self._count(EVENTS_CFN_MTX, OGS_C.EVENT_STR, OGS_C.EVENT_STR)
    missed_count = self._count(EVENTS_CFN_MTX, OGS_C.EVENT_STR, OGS_C.NONE_STR)
    proposed_count = self._count(EVENTS_CFN_MTX, OGS_C.NONE_STR, OGS_C.EVENT_STR)
    recall, fdr = self._log_recall_fdr(matched_count, missed_count, proposed_count)
    self.logger.info("\n%s", EVENTS_CFN_MTX)
    self.logger.info(
      "REVIEW " + "=" * 60 + "\n" +
      "Matched    (MH): %d\n"
      "Proposed   (PS): %d  [TARGET EVENTS NOT found in BASE]\n"
      "Skipped    (SP): %d  [TARGET EVENTS FILTER by (LOCATION)]\n"
      "Missed     (MS): %d  [BASE EVENTS NOT found in TARGET]\n"
      "Skimmed    (SM): %d  [BASE EVENTS FILTER by (LOCATION)]\n"
      "BASE total     : %d\n"
      "TARGET total   : %d",
      mh_n,
      ps_n,
      sp_n,
      ms_n,
      sm_n,
      base_n,
      target_n
    )
    event_review_checks = {
      " BASE ": {
        "bgma": {
          OGS_C.EVENT_STR: matched_count,
          OGS_C.NONE_STR: missed_count,
          "FILTERED": sm_n,
        },
        "check_sum": matched_count + missed_count + sm_n,
        "expected_sum": base_n,
      },
      "TARGET": {
        "bgma": {
          OGS_C.EVENT_STR: matched_count,
          OGS_C.NONE_STR: proposed_count,
          "FILTERED": sp_n,
        },
        "check_sum": matched_count + proposed_count + sp_n,
        "expected_sum": target_n,
      },
    }
    self._log_review_checks(event_review_checks, "events")
    self._write_csv(events_mh, target, "EventsMH")
    self._write_csv(events_ms, target, "EventsMS")
    self._write_csv(events_sm, target, "EventsSM")
    self._write_csv(events_ps, target, "EventsPS")
    self._write_csv(events_sp, target, "EventsSP")
    return recall, fdr

  def _finalize_frame(
    self, rows: list[list], columns: list[str], *, sort: bool = True,
  ) -> pd.DataFrame:
    """Build a review DataFrame, optionally sort by TIME_STR, and reset the index.

    Non-empty frames are sorted by ``OGS_C.TIME_STR`` only when ``sort`` is
    ``True``; otherwise row insertion order is preserved. In all cases the
    returned frame uses a dense 0-based index so callers get stable CSV/plot
    output ordering without forcing review-only TIME_STR payloads, such as MH
    tuples, through a chronological sort.
    """
    df = pd.DataFrame(rows, columns=columns)
    if df.empty or not sort:
      return df.reset_index(drop=True)
    return df.sort_values(by=OGS_C.TIME_STR).reset_index(drop=True)

  def _write_csv(
    self, df: pd.DataFrame, target: "OGSCatalog", suffix: str,
  ) -> None:
    """Write ``df`` to the standard review CSV path and log the destination.

    The filename format is ``<base>_<target>_<suffix>.csv`` under
    ``self.output``.
    """
    filepath = (
      self.output / f"{self.input.name}_{target.input.name}_{suffix}.csv"
    )
    df.to_csv(filepath, index=False)
    self.logger.info("%s written.", filepath)

  def _plot_output(
    self, target: "OGSCatalog", label: str, output: Optional[Path] = None,
    *, include_self: bool = True,
  ) -> Path:
    """Resolve the BGMA plot output path.

    If ``output`` is supplied it is returned unchanged. Otherwise the default
    path is created under ``self.output / "img"`` using ``IMAGE_EXT`` and a
    filename stem of ``<base>_<target>_<label>`` when ``include_self`` is
    ``True`` or ``<target>_<label>`` when ``include_self`` is ``False``.
    """
    if output is not None:
      return output
    prefix = (
      f"{self.input.name}_{target.input.name}" if include_self
      else target.input.name
    )
    return self.output / "img" / (f"{prefix}_{label}" + IMAGE_EXT)

  @staticmethod
  def _ps_phase_masks(
    picks_mh: pd.DataFrame,
  ) -> tuple[pd.Series, pd.Series]:
    """Return boolean masks selecting P-wave and S-wave rows from ``picks_mh``.

    Both Series share ``picks_mh``'s index and let pick-plot helpers reuse the
    same phase comparisons instead of rescanning the phase column separately.
    """
    phase_col = picks_mh[OGS_C.PHASE_STR]
    return (phase_col == OGS_C.PWAVE, phase_col == OGS_C.SWAVE)

  def _bgma_events_record_unmatched(
    self, df: pd.DataFrame, cfn_mtx: pd.DataFrame,
    sink: list[pd.DataFrame], *, role: str,
  ) -> None:
    """Count unmatched rows and append them to the side-specific sink.

    ``role=\"base\"`` writes BASE-only rows to ``EventsMS`` on the
    ``EVENT/NONE`` axis. ``role=\"target\"`` writes TARGET-only rows to
    ``EventsPS`` on the ``NONE/EVENT`` axis.
    """
    if role == "base":
      row_label, col_label = OGS_C.EVENT_STR, OGS_C.NONE_STR
    else:
      row_label, col_label = OGS_C.NONE_STR, OGS_C.EVENT_STR
    self._add(cfn_mtx, row_label, col_label, len(df.index))
    sink.append(df.reindex(columns=_EVENTS_MH_COLUMNS).reset_index(drop=True))

  def _event_feasible_positions(
    self,
    base: pd.DataFrame,
    target: pd.DataFrame,
  ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Return row positions with at least one feasible BGMA event edge.

    The feasibility test mirrors ``OGSBPGraphEvents`` edge creation: two
    events must fall within the configured time and distance thresholds.
    Rows with no feasible opposite-side candidate can be routed directly to
    ``EventsMS`` or ``EventsPS`` without paying the graph construction cost.
    """
    I, J = len(base.index), len(target.index)
    if I == 0 or J == 0:
      return np.array([], dtype=int), np.array([], dtype=int)
    time_offset = OGS_C.EVENT_TIME_OFFSET.total_seconds()
    base_times = np.fromiter(
      (UTCDateTime(value).timestamp for value in base[OGS_C.TIME_STR]),
      dtype=float,
      count=I,
    )
    target_times = np.fromiter(
      (UTCDateTime(value).timestamp for value in target[OGS_C.TIME_STR]),
      dtype=float,
      count=J,
    )
    base_keep = np.zeros(I, dtype=bool)
    target_keep = np.zeros(J, dtype=bool)
    for base_pos, (_, base_row) in enumerate(base.iterrows()):
      candidate_pos = np.flatnonzero(
        np.abs(target_times - base_times[base_pos]) <= time_offset
      )
      if not len(candidate_pos):
        continue
      for target_pos in candidate_pos:
        target_row = cast(pd.Series, target.iloc[target_pos])
        if OGS_U.diff_space(base_row, target_row) <= OGS_C.EVENT_DIST_OFFSET:
          base_keep[base_pos] = True
          target_keep[target_pos] = True
    return np.flatnonzero(base_keep), np.flatnonzero(target_keep)

  def _iter_shared_and_extra_dates(
    self,
    target: "OGSCatalog",
    key: str,
  ) -> Iterator[tuple[datetime, _DateSource]]:
    """Yield dates plus the branch label used by BGMA comparison loops.

    The iterator preserves ``self``'s calendar order: each BASE date is tagged
    as ``'both'`` or ``'base_only'`` first, then TARGET-only dates are emitted
    as ``'target_only'``. ``bgmaEvents`` and ``bgmaPicks`` share this routing
    contract.

    Parameters
    ----------
    target : OGSCatalog
      Catalog to compare against.
    key : str
      Either ``"events"`` or ``"picks"``, selecting which calendar to iterate.

    Yields
    ------
    tuple[datetime, _DateSource]
      The date and the source label.

    Raises
    ------
    ValueError
      If ``key`` is not ``"events"`` or ``"picks"``.
    """
    if key == "events":
      base_cal = self.events_
      target_cal = target.events_
    elif key == "picks":
      base_cal = self.picks_
      target_cal = target.picks_
    else:
      raise ValueError(f"Unknown key: {key}")
    base_set, target_set = set(base_cal), set(target_cal)
    for date in base_cal:
      yield date, _BOTH if date in target_set else _BASE_ONLY
    for date in target_cal:
      if date not in base_set:
        yield date, _TARGET_ONLY

  def _bgma_events_base_only(
    self, target: "OGSCatalog", date, EVENTS_CFN_MTX: pd.DataFrame,
    EventsMS: list[pd.DataFrame], EventsSM: list[pd.DataFrame],
  ) -> None:
    """Handle a BASE-only day after projecting BASE onto the shared domain."""
    BASE = self._prefilter_events(
      self._load_day("events", date).reset_index(drop=True),
      target._polygon_vertices(),
      EventsSM,
      date,
      "BASE",
    )
    self.logger.debug(
      "DATE %s NOT in TARGET CATALOG, counting BASE EVENTS as MS.",
      date
    )
    self._bgma_events_record_unmatched(
      BASE, EVENTS_CFN_MTX, EventsMS, role="base",
    )

  def _matched_events_frame(
    self,
    base: pd.DataFrame,
    target: pd.DataFrame,
    base_idx: np.ndarray,
    target_idx: np.ndarray,
  ) -> pd.DataFrame:
    """Return one shared-day wide matched-event frame."""
    base_sel = base.reindex(columns=_EVENTS_MH_COLUMNS).iloc[
      base_idx
    ].reset_index(drop=True)
    target_sel = target.reindex(columns=_EVENTS_MH_COLUMNS).iloc[
      target_idx
    ].reset_index(drop=True)
    return pd.concat(
      [base_sel.add_suffix("_base"), target_sel.add_suffix("_target")],
      axis=1,
    )

  def _bgma_events_both(
    self, target: "OGSCatalog", date,
    EVENTS_CFN_MTX: pd.DataFrame, EventsMH_frames: list[pd.DataFrame],
    EventsMS: list[pd.DataFrame], EventsSM: list[pd.DataFrame],
    EventsPS: list[pd.DataFrame], EventsSP: list[pd.DataFrame],
  ) -> None:
    """Handle a shared day within the shared spatial review domain."""
    BASE = self._prefilter_events(
      self._load_day("events", date).reset_index(drop=True),
      target._polygon_vertices(),
      EventsSM,
      date,
      "BASE",
    )
    TARGET = self._prefilter_events(
      target._load_day("events", date).reset_index(drop=True),
      self._polygon_vertices(),
      EventsSP,
      date,
      "TARGET",
    )
    I, J = len(BASE), len(TARGET)
    self.logger.debug("DATE %s: BASE=%d EVENTS, TARGET=%d EVENTS.", date, I, J)
    feasible_base_pos, feasible_target_pos = self._event_feasible_positions(
      BASE, TARGET
    )
    if (len(feasible_base_pos) != I or len(feasible_target_pos) != J):
      self.logger.debug(
        "DATE %s: pre-pruned BGMA candidates BASE=%d/%d TARGET=%d/%d.",
        date, len(feasible_base_pos), I, len(feasible_target_pos), J,
      )
    if len(feasible_base_pos) and len(feasible_target_pos):
      BASE_FEASIBLE = BASE.iloc[feasible_base_pos].reset_index(drop=True)
      TARGET_FEASIBLE = TARGET.iloc[feasible_target_pos].reset_index(drop=True)
      pairs = OGS_U.OGSBPGraphEvents(
        BASE_FEASIBLE, TARGET_FEASIBLE
      ).matched_pairs_array()
    else:
      BASE_FEASIBLE = BASE.iloc[0:0]
      TARGET_FEASIBLE = TARGET.iloc[0:0]
      pairs = np.empty((0, 2), dtype=int)
    if len(pairs):
      base_idx = feasible_base_pos[pairs[:, 0]]
      target_idx = feasible_target_pos[pairs[:, 1] - len(BASE_FEASIBLE)]
      n_matched = len(pairs)
      unmatched_base_idx = np.delete(np.arange(I), base_idx)
      unmatched_target_idx = np.delete(np.arange(J), target_idx)
      EventsMH_frames.append(self._matched_events_frame(
        BASE, TARGET, base_idx, target_idx,
      ))
    else:
      n_matched = 0
      unmatched_base_idx = np.arange(I)
      unmatched_target_idx = np.arange(J)
    self._add(EVENTS_CFN_MTX, OGS_C.EVENT_STR, OGS_C.EVENT_STR, n_matched)
    self.logger.debug(
      "Matched events: %d (BASE <-> TARGET)", n_matched,
    )
    self._bgma_events_record_unmatched(
      BASE.iloc[unmatched_base_idx], EVENTS_CFN_MTX, EventsMS, role="base",
    )
    self._bgma_events_record_unmatched(
      TARGET.iloc[unmatched_target_idx], EVENTS_CFN_MTX, EventsPS,
      role="target",
    )

  def _bgma_events_target_only(
    self, target: "OGSCatalog", date, EVENTS_CFN_MTX: pd.DataFrame,
    EventsPS: list[pd.DataFrame], EventsSP: list[pd.DataFrame],
  ) -> None:
    """Handle a TARGET-only day after projecting TARGET onto the shared domain."""
    TARGET = self._prefilter_events(
      target._load_day("events", date).reset_index(drop=True),
      self._polygon_vertices(),
      EventsSP,
      date,
      "TARGET",
    )
    if TARGET.empty:
      self.logger.debug("DATE %s in TARGET catalog does NOT have EVENTS", date)
      return
    self.logger.warning(
      "DATE %s in TARGET EVENTS catalog but NOT in BASE", date
    )
    self._bgma_events_record_unmatched(
      TARGET, EVENTS_CFN_MTX, EventsPS, role="target",
    )

  def bgmaEvents(self, target: "OGSCatalog", output=None) -> None:
    """Run BGMA event matching and build matched, missed, and proposed outputs.

    Shared dates are matched with BGMA and written to ``EventsMH`` as wide
    rows: every matched event pair contributes one row with ``{col}_base`` and
    ``{col}_target`` fields so downstream review plots can compare aligned
    attributes without another merge. Events filtered out before matching by
    the shared spatial review domain populate ``EventsSM`` for BASE and
    ``EventsSP`` for TARGET. Unmatched BGMA-eligible BASE rows populate
    ``EventsMS``; unmatched BGMA-eligible TARGET rows populate ``EventsPS``;
    the event confusion matrix tracks only MH/MS/PS outcomes.

    Parameters
    ----------
    target : OGSCatalog
      Catalog to compare against.
    output : Optional[Path], optional
      Optional output override forwarded to the event review plot helpers. If
      omitted, helper-specific filenames are created under ``self.output /
      "img"``.
    """
    if not isinstance(target, OGSCatalog):
      raise ValueError("Can only perform bgmaEvents on OGSCatalog")
    self_name = self.name
    target_name = target.name
    self.logger.info("Starting bgmaEvents: %s vs %s", self_name, target_name)
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt

    EVENTS_CFN_MTX = self._empty_cfn_mtx(_EVENTS_PHASES)
    EventsMH_frames: list[pd.DataFrame] = []
    EventsMS_frames: list[pd.DataFrame] = []
    EventsSM_frames: list[pd.DataFrame] = []
    EventsPS_frames: list[pd.DataFrame] = []
    EventsSP_frames: list[pd.DataFrame] = []
    for date, source in self._iter_shared_and_extra_dates(target, "events"):
      if source == _BASE_ONLY:
        self._bgma_events_base_only(
          target, date, EVENTS_CFN_MTX, EventsMS_frames, EventsSM_frames
        )
      elif source == _BOTH:
        self._bgma_events_both(
          target, date, EVENTS_CFN_MTX, EventsMH_frames,
          EventsMS_frames, EventsSM_frames, EventsPS_frames, EventsSP_frames
        )
      else:  # _TARGET_ONLY
        self._bgma_events_target_only(
          target, date, EVENTS_CFN_MTX, EventsPS_frames, EventsSP_frames
        )
    self.EventsMH = (
      pd.concat(EventsMH_frames, ignore_index=True)
      if EventsMH_frames else pd.DataFrame(columns=_EVENTS_MH_WIDE_COLUMNS)
    )
    self.EventsMS = (
      pd.concat(EventsMS_frames, ignore_index=True)
      if EventsMS_frames else pd.DataFrame(columns=_EVENTS_MH_COLUMNS)
    )
    self.EventsSM = (
      pd.concat(EventsSM_frames, ignore_index=True)
      if EventsSM_frames else pd.DataFrame(columns=_EVENTS_MH_COLUMNS)
    )
    self.EventsPS = (
      pd.concat(EventsPS_frames, ignore_index=True)
      if EventsPS_frames else pd.DataFrame(columns=_EVENTS_MH_COLUMNS)
    )
    self.EventsSP = (
      pd.concat(EventsSP_frames, ignore_index=True)
      if EventsSP_frames else pd.DataFrame(columns=_EVENTS_MH_COLUMNS)
    )
    recall, fdr = self._bgma_events_review(target, EVENTS_CFN_MTX)
    filepath = self._plot_output(target, "EventsConfMtx", output)
    OGS_P.ConfMtx_plotter(
      EVENTS_CFN_MTX.values,
      title="Recall: {:.4f}, FDR: {:.4f}".format(recall, fdr),
      label=_EVENTS_PHASES,
      output=filepath,
      basename=self_name,
      targetname=target_name
    )
    plt.close('all')
    self._plot_events_msps_map(target, output)
    if self.EventsMH.empty:
      self.logger.warning(
        "No MH EVENTS to plot matched-event diagnostics for %s vs %s.",
        self_name,
        target_name,
      )
      plt.close('all')
      return
    self._plot_events_time_diff(target, output)
    self._plot_events_mh_map(target, output)
    self._plot_events_depth_diff(target, output)
    self._plot_events_epidist(target, output)
    self._plot_events_magnitude(target, output)
    plt.close('all')

  # -------------------------------------------------------------------------
  # BGMA event diagnostic plotters
  # -------------------------------------------------------------------------
  # One plotter per diagnostic axis (origin-time delta, hypocenter maps,
  # depth/epi-distance/magnitude). All consume the matched-event review
  # frame ``self.EventsMH`` produced above.
  # -------------------------------------------------------------------------

  def _mh_diff(self, col: str) -> pd.Series:
    """Return BASE-minus-TARGET differences from ``EventsMH`` for one field.

    Expects the wide matched-event frame ``self.EventsMH`` produced by
    ``bgmaEvents``, with paired ``{col}_base`` and ``{col}_target`` columns.
    For ``TIME_STR``, both sides are coerced onto a shared UTC timeline before
    subtraction so mixed tz-aware and tz-naive event timestamps remain
    comparable. The helper does not write output; event diagnostics reuse the
    returned series for matched-event histograms.
    """
    target = self._tcol(self.EventsMH, col, 1)
    base = self._tcol(self.EventsMH, col, 0)
    if col == OGS_C.TIME_STR:
      target_dt = pd.to_datetime(target, utc=True, errors="coerce")
      base_dt = pd.to_datetime(base, utc=True, errors="coerce")
      return cast(pd.Series, (target_dt - base_dt).dt.total_seconds())
    return cast(pd.Series, target - base)

  def _plot_events_time_diff(
    self, target: "OGSCatalog", output: Optional[Path] = None
  ) -> None:
    """Plot matched-event origin-time residuals from ``EventsMH``.

    Expects ``self.EventsMH`` from ``bgmaEvents`` for ``self`` and ``target``.
    Writes the ``EventsTimeDiff`` histogram image through ``_plot_output``.
    """
    from matplotlib import pyplot as plt
    import ogsplotter as OGS_P
    data = cast(pd.Series, self._mh_diff(OGS_C.TIME_STR).dropna())
    if data.empty:
      self.logger.warning(
        "No matched-event time residuals available for plotting."
      )
      plt.close()
      return
    offset = OGS_C.EVENT_TIME_OFFSET.total_seconds()
    OGS_P.histogram_plotter(
      data,
      xlabel="Time Difference (s)",
      title=f"RMSE = {np.sqrt(np.mean(data ** 2)):.4f} s, " +
            f"MAE = {data.abs().mean():.4f} s",
      xlim=(-offset, offset),
      output=self._plot_output(target, "EventsTimeDiff", output),
      legend=True
    )
    plt.close()

  def _plot_events_mh_map(
    self, target: "OGSCatalog", output: Optional[Path] = None
  ) -> None:
    """Plot matched-event locations from the wide ``EventsMH`` table.

    Expects ``self.EventsMH`` produced by ``bgmaEvents``, where matched BASE
    and TARGET event coordinates are stored side by side. Writes the
    ``EventsMH`` map image through ``_plot_output``.
    """
    from matplotlib import pyplot as plt
    import ogsplotter as OGS_P
    magnitude = self._magnitude_tuple_or_none(self.EventsMH, 0)
    myplot = OGS_P.map_plotter(
      domain=OGS_C.OGS_STUDY_REGION,
      x=self._tcol(self.EventsMH, OGS_C.LONGITUDE_STR, 0),
      y=self._tcol(self.EventsMH, OGS_C.LATITUDE_STR, 0),
      facecolors="none", edgecolors=OGS_C.OGS_BLUE, legend=True,
      label=self.name,
      magnitude=magnitude,
    )
    magnitude = self._magnitude_tuple_or_none(self.EventsMH, 1)
    myplot.add_plot(
      self._tcol(self.EventsMH, OGS_C.LONGITUDE_STR, 1),
      self._tcol(self.EventsMH, OGS_C.LATITUDE_STR, 1), color=None,
      label=target.name, legend=True, facecolors="none",
      edgecolors=OGS_C.MEX_PINK,
      magnitude=magnitude,
      output=self._plot_output(target, "EventsMH", output),
    )
    plt.close()

  def _plot_events_msps_map(
    self, target: "OGSCatalog", output: Optional[Path] = None
  ) -> None:
    """Plot unmatched-event locations from ``EventsMS`` and ``EventsPS``.

    Expects ``self.EventsMS`` and ``self.EventsPS`` produced by
    ``bgmaEvents`` for the ``self``/``target`` comparison. Writes the
    ``EventsFalse`` map image through ``_plot_output``.
    """
    from matplotlib import pyplot as plt
    import ogsplotter as OGS_P
    magnitude = self._magnitude_or_none(self.EventsMS)
    myplot = OGS_P.map_plotter(
      domain=OGS_C.OGS_STUDY_REGION,
      x=self.EventsMS[OGS_C.LONGITUDE_STR],
      y=self.EventsMS[OGS_C.LATITUDE_STR],
      label=f"Missed (MS) [{self.name}] {len(self.EventsMS.index)}",
      legend=True,
      magnitude=magnitude,
    )
    magnitude = self._magnitude_or_none(self.EventsPS)
    myplot.add_plot(
      self.EventsPS[OGS_C.LONGITUDE_STR], self.EventsPS[OGS_C.LATITUDE_STR],
      color=None, facecolors="none", edgecolors=OGS_C.MEX_PINK, legend=True,
      label=f"Proposed (PS) [{target.name}] {len(self.EventsPS.index)}",
      magnitude=magnitude,
      output=self._plot_output(target, "EventsFalse", output),
    )
    plt.close()

  def _plot_events_depth_diff(
    self, target: "OGSCatalog", output: Optional[Path] = None
  ) -> None:
    """Plot matched-event depth residuals derived from ``EventsMH``.

    Expects ``self.EventsMH`` from ``bgmaEvents`` for ``self`` and ``target``.
    Writes the ``DepthDiff`` histogram image through ``_plot_output``.
    """
    from matplotlib import pyplot as plt
    import ogsplotter as OGS_P
    OGS_P.histogram_plotter(
      self._mh_diff(OGS_C.DEPTH_STR),
      xlabel=f"Depth Difference (km) [{self.name} - {target.name}]",
      title="Event Depth Difference",
      xlim=(-20, 20),
      output=self._plot_output(target, "DepthDiff", output),
      legend=True
    )
    plt.close()

  def _plot_events_epidist(
    self, target: "OGSCatalog", output: Optional[Path] = None
  ) -> None:
    """Plot matched-event epicentral-distance residuals from ``EventsMH``.

    Expects ``self.EventsMH`` produced by ``bgmaEvents`` so matched BASE and
    TARGET hypocenters are available in paired columns. Writes the
    ``EpiDistDiff`` histogram image through ``_plot_output``.
    """
    from matplotlib import pyplot as plt
    import ogsplotter as OGS_P
    OGS_P.histogram_plotter(
      OGS_P.v_lat_long_to_distance(
        self._tcol(self.EventsMH, OGS_C.LONGITUDE_STR, 0),
        self._tcol(self.EventsMH, OGS_C.LATITUDE_STR, 0),
        np.zeros(len(self.EventsMH.index)),
        self._tcol(self.EventsMH, OGS_C.LONGITUDE_STR, 1),
        self._tcol(self.EventsMH, OGS_C.LATITUDE_STR, 1),
        self._tcol(self.EventsMH, OGS_C.DEPTH_STR, 1),
        dim=2
      ),
      xlim=(0, OGS_C.EVENT_DIST_OFFSET),
      xlabel=f"Epicentral Distance Difference (km) [{self.name} - {target.name}]",
      title="Event Epicentral Distance Difference",
      output=self._plot_output(target, "EpiDistDiff", output),
      legend=True)
    plt.close()

  def _plot_events_magnitude(
    self, target: "OGSCatalog", output: Optional[Path] = None
  ) -> None:
    """Plot matched and unmatched magnitude diagnostics from BGMA outputs.

    Expects ``self.EventsMH``, ``self.EventsMS``, and ``self.EventsPS``
    produced by ``bgmaEvents`` for the ``self``/``target`` comparison. Writes
    ``MagLDist`` for matched-event scatter, optionally ``MagLDiff`` for the
    overlay case, and either ``MSPSMagLDist`` or ``PSMagLDist`` for unmatched
    magnitudes through ``_plot_output``.
    """
    from matplotlib import pyplot as plt
    import ogsplotter as OGS_P
    # Read the matched magnitudes once before the scatter and histogram steps.
    mag0 = self._magnitude_tuple_or_none(self.EventsMH, 0)
    mag1 = self._magnitude_tuple_or_none(self.EventsMH, 1)
    if mag0 is None or mag1 is None:
      return
    # Pre-filter unmatched magnitudes for the later histogram branch(es).
    ps_mag = self.EventsPS[OGS_C.MAGNITUDE_L_STR].dropna()
    ms_mag = self.EventsMS[OGS_C.MAGNITUDE_L_STR].dropna()
    mymags = OGS_P.scatter_plotter(
      mag1,
      mag0,
      xlabel=f"{target.name} Magnitude ($M_L$)",
      ylabel=f"{self.name} Magnitude ($M_L$)",
      title="Magnitude Prediction",
      color=OGS_C.OGS_BLUE,
      legend=True
    )
    x_min = min(mag0.min(), mag1.min())
    x_max = max(mag0.max(), mag1.max())
    mymags.ax.plot([x_min, x_max], [x_min, x_max], color=OGS_C.MEX_PINK,
                   linestyle='--')
    mymags.ax.set_aspect('equal', adjustable='box')
    mymags.ax.grid(True)
    mymags.savefig(self._plot_output(target, "MagLDist", output))
    plt.close()
    # Overlay mode also writes ``MagLDiff`` and combines PS/MS magnitudes.
    overlay = (
      self.input.name in (OGS_C.TXT_EXT, ".all")
      and target.input.name == "OGSLocalMagnitude"
    )
    if overlay:
      # Magnitude Difference Histogram
      data = mag1 - mag0
      OGS_P.histogram_plotter(
        data,
        xlabel=f"Magnitude Difference ($M_L$) [{self.name} - {target.name}]",
        title=(f"RMSE = {np.sqrt(np.mean(data ** 2)):.4f}, " +
               f"MAE = {data.abs().mean():.4f}"),
        xlim=(-1.5, 1.5),
        bins=21,
        output=self._plot_output(target, "MagLDiff", output),
        legend=True
      )
      plt.close()
    if ps_mag.empty:
      return
    # Shared histogram metadata for the unmatched-magnitude output(s).
    ps_kwargs: dict[str, Any] = {
      "xlabel": "Magnitude ($M_L$)",
      "title": "Event Magnitude",
      "color": OGS_C.MEX_PINK,
      "label": f"Proposed (PS) [{target.name}]",
    }
    if overlay:
      mymags = OGS_P.histogram_plotter(
        ps_mag, yscale='log', xlim=[-1., 5.0], **ps_kwargs,
      )
      mymags.add_plot(
        ms_mag,
        label=f"Missed (MS) [{self.name}]",
        color=OGS_C.OGS_BLUE,
        legend=True,
        output=self._plot_output(target, "MSPSMagLDist", output),
      )
    else:
      OGS_P.histogram_plotter(
        ps_mag,
        output=self._plot_output(
          target, "PSMagLDist", output, include_self=False,
        ),
        **ps_kwargs,
      )
    plt.close()

  # =========================================================================
  # BGMA PICK MATCHING & REVIEW
  # =========================================================================
  # Same per-day cadence as the event flow, but matching is by
  # ``(station, phase, time)`` with a tolerance window. Builds the 3x3 phase
  # confusion matrix (P / S / NONE) and the matched (MH) / missed (SM) /
  # proposed (SP) pick partitions.
  # =========================================================================

  def _record_unmatched_picks(
    self,
    frame: pd.DataFrame,
    columns: Sequence[str],
    axis: Literal[0, 1],
    picks_list: list,
    PICKS_CFN_MTX: pd.DataFrame,
  ) -> None:
    """Record one-sided residual picks in both review outputs and the matrix.

    ``frame`` contains picks that survived inventory cleaning but have no BGMA
    partner on the opposite side. ``axis=0`` writes BASE-phase counts into the
    ``NONE`` target column for Missed picks (``PicksMS``). ``axis=1`` writes
    TARGET-phase counts into the ``NONE`` base row for Proposed picks
    (``PicksPS``). The same rows are appended to ``picks_list`` for CSV review
    output.
    """
    if frame.empty:
      return
    self._add_series(
      PICKS_CFN_MTX, OGS_C.NONE_STR,
      frame[OGS_C.PHASE_STR].value_counts(), axis=axis
    )
    picks_list.extend(frame[columns].values.tolist())

  def _load_and_clean_picks(
    self,
    catalog: "OGSCatalog",
    date: datetime,
    inventory: np.ndarray,
    columns: Sequence[str],
    ski_list: list,
    label: str,
  ) -> pd.DataFrame:
    """Load one day of picks and apply the shared BGMA pre-filtering step.

    The returned frame is normalized for matching, while any out-of-inventory
    rows are diverted to the caller-provided filtered review list:
    ``PicksSM`` for BASE or ``PicksSP`` for TARGET.
    """
    frame = catalog._load_day("picks", date).reset_index(drop=True)
    return self._clean_picks(frame, inventory, columns, ski_list, date, label)

  def _clean_picks(
    self,
    picks: pd.DataFrame,
    inventory: np.ndarray,
    columns: Sequence[str],
    ski_list: list,
    date: datetime,
    label: str,
  ) -> pd.DataFrame:
    """Normalize station ids and peel out picks excluded from BGMA.

    Picks may arrive with FDSN station identifiers in
    ``NET.STA.LOC.CHAN`` form. This helper copies the network into
    ``NETWORK`` and reduces ``STATION`` to the station code before testing the
    station against ``inventory``. Rows filtered out here never enter BGMA
    phase accounting; instead they are logged and appended to the filtered
    review bucket passed in ``ski_list``.

    Parameters
    ----------
    picks : pd.DataFrame
      One day's picks to prepare for BGMA.
    inventory : np.ndarray
      Known station codes eligible for matching.
    columns : Sequence[str]
      Review-output schema used when copying filtered rows into ``ski_list``.
    ski_list : list
      Collector for out-of-inventory rows; BASE callers pass ``PicksSM`` and
      TARGET callers pass ``PicksSP``.
    date : datetime
      Day being processed, used in warning messages.
    label : str
      Side label included in warnings, typically ``"BASE"`` or ``"TARGET"``.

    Returns
    -------
    pd.DataFrame
      Cleaned picks that remain eligible for BGMA matching and confusion-
      matrix accounting.
    """
    if picks.empty or OGS_C.STATION_STR not in picks.columns:
      return picks.iloc[0:0]
    split = picks[OGS_C.STATION_STR].str.split(".", expand=False)
    fsdn_split = split.str.len().ge(2)
    if not picks[fsdn_split].empty:
      picks.loc[fsdn_split, OGS_C.NETWORK_STR] = split.str[0]
      picks.loc[fsdn_split, OGS_C.STATION_STR] = split.str[1]

    in_inventory = picks[OGS_C.STATION_STR].isin(inventory)
    if any(~in_inventory):
      missing_stations = cast(
        pd.Series, picks.loc[~in_inventory, OGS_C.STATION_STR]
      ).astype(str).unique().tolist()
      missing_rows = cast(pd.DataFrame, picks.loc[~in_inventory, columns])
      self.logger.warning(
        "DATE %s: %s picks with STATION (%s) not in INVENTORY:",
        date, label, ", ".join(missing_stations)
      )
      self.logger.debug("\n%s", picks[~in_inventory])
      ski_list.extend(missing_rows.values.tolist())
    return cast(pd.DataFrame, picks.loc[in_inventory, :]).reset_index(drop=True)

  def _bgma_picks_base_only(
    self, target: "OGSCatalog", date, INVENTORY, columns, PICKS_CFN_MTX,
  ) -> None:
    """Account for a BASE-only day in the pick review.

    After inventory filtering, every remaining BASE pick is counted as Missed
    (``PicksMS``) by writing its phase into the ``NONE`` target column of the
    confusion matrix. BASE rows removed during cleaning are tracked separately
    in ``PicksSM``.
    """
    self.logger.debug("DATE %s in TARGET catalog has no PICKS, skipping.",
                      date)
    BASE = self._load_and_clean_picks(
      self, date, INVENTORY, columns, self.PicksSM, "BASE"
    )
    self._record_unmatched_picks(
      BASE, columns, 0, self.PicksMS, PICKS_CFN_MTX
    )

  def _bgma_picks_both(
    self, target: "OGSCatalog", date, INVENTORY, columns, PICKS_CFN_MTX,
  ) -> None:
    """Run pick-level BGMA for a date shared by both catalogs.

    Cleaned BASE and TARGET picks are connected through the BGMA graph.
    Same-phase edges become Matched picks (``PicksMH``) on the confusion-
    matrix diagonal, phase-mismatched edges become Swapped picks
    (``PicksSW``) on the off-diagonal, unmatched BASE residues become Missed
    picks (``PicksMS``) in the ``NONE`` target column, and unmatched TARGET
    residues become Proposed picks (``PicksPS``) in the ``NONE`` base row.
    Out-of-inventory rows removed before matching are tracked separately in
    ``PicksSM`` and ``PicksSP``.
    """
    # Normalize NET.STA.LOC.CHAN station ids and divert out-of-inventory rows
    # to the filtered review buckets before graph matching.
    BASE = self._load_and_clean_picks(
      self, date, INVENTORY, columns, self.PicksSM, "BASE"
    )
    TARGET = self._load_and_clean_picks(
      target, date, INVENTORY, columns, self.PicksSP, "TARGET"
    )
    I, J = len(BASE), len(TARGET)
    base_mask = np.ones(I, dtype=bool)
    target_mask = np.ones(J, dtype=bool)
    pairs = OGS_U.OGSBPGraphPicks(BASE, TARGET).matched_pairs_array()
    if len(pairs):
      base_pos = pairs[:, 0]
      target_pos = pairs[:, 1] - I
      base_idx = BASE[OGS_C.IDX_PICKS_STR].to_numpy()[base_pos]
      target_idx = TARGET[OGS_C.IDX_PICKS_STR].to_numpy()[target_pos]
      base_times = BASE.iloc[base_pos][OGS_C.TIME_STR].astype(str).to_numpy()
      target_times = (
        TARGET.iloc[target_pos][OGS_C.TIME_STR].astype(str).to_numpy()
      )
      base_phases = BASE[OGS_C.PHASE_STR].to_numpy()[base_pos]
      target_phases = TARGET[OGS_C.PHASE_STR].to_numpy()[target_pos]
      target_stations = TARGET[OGS_C.STATION_STR].to_numpy()[target_pos]
      base_probs = BASE[OGS_C.PROBABILITY_STR].to_numpy()[base_pos]
      target_probs = TARGET[OGS_C.PROBABILITY_STR].to_numpy()[target_pos]
      match_mask = (base_phases == target_phases)

      if np.any(match_mask):
        self.PicksMH.extend([
          [
            (base_idx[k], target_idx[k]),
            (base_times[k], target_times[k]),
            base_phases[k],
            target_stations[k],
            (base_probs[k], target_probs[k]),
          ] for k in np.flatnonzero(match_mask)
        ])
      if np.any(~match_mask):
        self.PicksSW.extend([
          [
            (base_idx[k], target_idx[k]),
            (base_times[k], target_times[k]),
            (base_phases[k], target_phases[k]),
            target_stations[k],
            (base_probs[k], target_probs[k]),
          ] for k in np.flatnonzero(~match_mask)
        ])

      phase_counts = np.unique(
        np.column_stack((
          np.asarray(base_phases, dtype=str),
          np.asarray(target_phases, dtype=str),
        )),
        axis=0,
        return_counts=True,
      )
      for (bp, tp), n in zip(*phase_counts):
        self._add(PICKS_CFN_MTX, bp, tp, int(n))

      base_mask[base_pos] = False
      target_mask[target_pos] = False

    # Missed (MS): cleaned BASE picks that have no BGMA match in TARGET.
    self._record_unmatched_picks(
      BASE.iloc[base_mask], columns, 0, self.PicksMS, PICKS_CFN_MTX
    )
    # Proposed (PS): cleaned TARGET picks that have no BGMA match in BASE.
    self._record_unmatched_picks(
      TARGET.iloc[target_mask], columns, 1, self.PicksPS, PICKS_CFN_MTX
    )

  def _bgma_picks_target_only(
    self, target: "OGSCatalog", date, INVENTORY, columns, PICKS_CFN_MTX,
  ) -> None:
    """Account for a TARGET-only day in the pick review.

    After inventory filtering, every remaining TARGET pick is counted as
    Proposed (``PicksPS``) by writing its phase into the ``NONE`` base row of
    the confusion matrix. TARGET rows removed during cleaning are tracked
    separately in ``PicksSP``.
    """
    self.logger.warning(
      "DATE %s in TARGET PICKS catalog but NOT in BASE",
      date
    )
    TARGET = self._load_and_clean_picks(
      target, date, INVENTORY, columns, self.PicksSP, "TARGET"
    )
    if TARGET.empty:
      self.logger.warning(
        "DATE %s in TARGET catalog has no picks, skipping.",
        date
      )
      return
    self._record_unmatched_picks(
      TARGET, columns, 1, self.PicksPS, PICKS_CFN_MTX
    )

  def bgmaPicks(self, target: "OGSCatalog", output=None) -> None:
    """Match picks between catalogs and build BGMA review artifacts.

    Rows in ``PICKS_CFN_MTX`` represent BASE phases and columns represent
    TARGET phases. The diagonal counts same-phase matches (``PicksMH``), the
    off-diagonal P<->S cells count swapped-phase matches (``PicksSW``), the
    ``NONE`` target column counts BASE picks missing from TARGET
    (``PicksMS``), and the ``NONE`` base row counts TARGET picks absent from
    BASE (``PicksPS``). Picks filtered out before matching because their
    station is not in the inventory never enter the matrix; they are tracked
    separately as ``PicksSM`` for BASE and ``PicksSP`` for TARGET.

    Parameters
    ----------
    target : OGSCatalog
      Catalog to compare against.
    output : Optional[Path], optional
      Output path for the results. If None, defaults to a file in the output
      directory named "{self.input.name}_{target.input.name}_PicksMH.csv".
    """
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    if not isinstance(target, OGSCatalog):
      raise ValueError("Can only perform bgmaPicks on OGSCatalog")
    if self.stations is None:
      raise ValueError("Station inventory is required for bgmaPicks")
    PICKS_CFN_MTX = self._empty_cfn_mtx(_PICKS_PHASES)
    stations = cast(pd.DataFrame, self.stations)
    INVENTORY = stations[OGS_C.STATION_STR].unique()
    self.PicksMH = list() # Same-phase BASE/TARGET matches.
    self.PicksMS = list() # BASE picks with no BGMA match in TARGET.
    self.PicksSM = list() # BASE picks filtered out before BGMA by inventory.
    self.PicksPS = list() # TARGET picks with no BGMA match in BASE.
    self.PicksSP = list() # TARGET picks filtered out before BGMA by inventory.
    self.PicksSW = list() # Matched BASE/TARGET edges whose phases disagree.
    columns = _PICKS_MH_COLUMNS

    # Walk every shared, BASE-only, and TARGET-only pick day once.
    for date, source in self._iter_shared_and_extra_dates(target, "picks"):
      if source == _BASE_ONLY:
        self._bgma_picks_base_only(
          target, date, INVENTORY, columns, PICKS_CFN_MTX
        )
      elif source == _BOTH:
        self._bgma_picks_both(
          target, date, INVENTORY, columns, PICKS_CFN_MTX
        )
      else:
        self._bgma_picks_target_only(
          target, date, INVENTORY, columns, PICKS_CFN_MTX
        )

    _v = lambda r, c: self._count(PICKS_CFN_MTX, r, c)
    true_pos = _v(OGS_C.PWAVE, OGS_C.PWAVE) + _v(OGS_C.SWAVE, OGS_C.SWAVE)
    swapped  = _v(OGS_C.PWAVE, OGS_C.SWAVE) + _v(OGS_C.SWAVE, OGS_C.PWAVE)
    missed   = _v(OGS_C.PWAVE, OGS_C.NONE_STR) + _v(OGS_C.SWAVE, OGS_C.NONE_STR)
    proposed = _v(OGS_C.NONE_STR, OGS_C.PWAVE) + _v(OGS_C.NONE_STR, OGS_C.SWAVE)
    recall, fdr = self._log_recall_fdr(
      true_pos, missed, proposed, swapped=swapped
    )
    p_recall, p_fdr = self._picks_phase_metrics(
      PICKS_CFN_MTX, OGS_C.PWAVE, OGS_C.SWAVE
    )
    s_recall, s_fdr = self._picks_phase_metrics(
      PICKS_CFN_MTX, OGS_C.SWAVE, OGS_C.PWAVE
    )
    self.logger.info("\n%s", PICKS_CFN_MTX)
    self.logger.info(
      "REVIEW " + "=" * 60 + "\n" +
      "Matched    (MH): %d\n"
      "Swapped    (SW): %d  [BASE PICKS FOUND in TARGET but PHASE MISMATCH]\n"
      "Proposed   (PS): %d  [TARGET PICKS NOT found in BASE]\n"
      "Skipped    (SP): %d  [TARGET PICKS FILTER by (INVENTORY)]\n"
      "Missed     (MS): %d  [BASE PICKS NOT found in TARGET]\n"
      "Skimmed    (SM): %d  [BASE PICKS FILTER by (INVENTORY)]\n"
      "BASE total     : %d\n"
      "TARGET total   : %d",
      len(self.PicksMH),
      len(self.PicksSW),
      len(self.PicksPS),
      len(self.PicksSP),
      len(self.PicksMS),
      len(self.PicksSM),
      len(self.get("PICKS").index),
      len(target.get("PICKS").index)
    )
    # Per-phase review accounting ignores the NONE axis and instead checks that
    # each physical wave total is conserved. For BASE this is row sum
    # (MH + SW + MS) plus filtered BASE picks in PicksSM; for TARGET it is the
    # column sum (MH + SW + PS) plus filtered TARGET picks in PicksSP.
    base_wave_rows = [OGS_C.PWAVE, OGS_C.SWAVE]

    def _phase_counts(frame: pd.DataFrame) -> pd.Series:
      """Return P/S totals for review reconciliation with missing phases as 0.

      The per-phase checks only reconcile physical wave phases, so this helper
      reindexes to the BASE/TARGET P and S rows used by the confusion matrix.
      """
      if frame.empty or OGS_C.PHASE_STR not in frame.columns:
        return pd.Series(0, index=base_wave_rows, dtype=int)
      return frame.groupby(OGS_C.PHASE_STR).size().reindex(
        base_wave_rows, fill_value=0
      )

    def _build_review_checks(
      prefix: str,
      bgma: pd.DataFrame,
      filtered: pd.Series,
      expected: pd.Series,
      axis: Literal[0, 1],
    ) -> dict[str, dict[str, object]]:
      """Build one catalog side's per-phase BGMA review checks.

      ``axis=1`` sums rows because BASE phases live on matrix rows; ``axis=0``
      sums columns because TARGET phases live on matrix columns. Each review
      entry captures the BGMA phase breakdown, the side's filtered count
      (``PicksSM`` or ``PicksSP``), and the expected catalog total for that
      phase.
      """
      totals = bgma.sum(axis=axis).reindex(base_wave_rows, fill_value=0)
      return {
        f"{prefix} {phase}": {
          "bgma": (
            bgma.loc[phase].to_dict()
            if axis == 1 else bgma[phase].to_dict()
          ),
          "filtered": cast(int, filtered.at[phase]),
          "check_sum": cast(int, totals.at[phase]) +
                       cast(int, filtered.at[phase]),
          "expected_sum": cast(int, expected.at[phase]),
        }
        for phase in base_wave_rows
      }

    base_bgma = cast(pd.DataFrame, PICKS_CFN_MTX.loc[base_wave_rows, :])
    target_bgma = cast(pd.DataFrame, PICKS_CFN_MTX.loc[:, base_wave_rows])
    _review_checks: dict[str, dict[str, object]] = {}
    review_specs: list[
      tuple[str, pd.DataFrame, list[Any], pd.DataFrame, Literal[0, 1]]
    ] = [
      (" BASE ", base_bgma, self.PicksSM, self.get("PICKS"), 1),
      ("TARGET", target_bgma, self.PicksSP, target.get("PICKS"), 0),
    ]
    for prefix, bgma, filtered_rows, expected_frame, axis in review_specs:
      _review_checks.update(_build_review_checks(
        prefix, bgma,
        _phase_counts(pd.DataFrame(filtered_rows, columns=columns)),
        _phase_counts(expected_frame),
        axis=axis,
      ))
    self._log_review_checks(_review_checks, "picks")
    for _attr in (
      "PicksMH", "PicksMS", "PicksPS", "PicksSW", "PicksSM", "PicksSP"
    ):
      setattr(self, _attr, self._finalize_frame(getattr(self, _attr), columns))
    picks_mh = cast(pd.DataFrame, self.PicksMH)
    picks_ms = cast(pd.DataFrame, self.PicksMS)
    picks_ps = cast(pd.DataFrame, self.PicksPS)
    picks_sw = cast(pd.DataFrame, self.PicksSW)
    picks_sm = cast(pd.DataFrame, self.PicksSM)
    picks_sp = cast(pd.DataFrame, self.PicksSP)
    self._write_csv(picks_mh, target, "PicksMH")
    self._write_csv(picks_ms, target, "PicksMS")
    self._write_csv(picks_ps, target, "PicksPS")
    self._write_csv(picks_sw, target, "PicksSW")
    self._write_csv(picks_sm, target, "PicksSM")
    self._write_csv(picks_sp, target, "PicksSP")
    self._plot_picks_confmtx(
      target,
      PICKS_CFN_MTX,
      (recall, p_recall, s_recall),
      (fdr, p_fdr, s_fdr),
      output,
    )
    if not self._plot_picks_time_diff(target, output):
      return
    self._plot_picks_confidence(target, output)

  # =========================================================================
  # CATALOG SET OPERATIONS (in-place union / subtraction)
  # =========================================================================
  # Used both by ``__iadd__`` / ``__isub__`` and by the BGMA workflow when
  # building review-only catalogs that exclude already-matched rows.
  # =========================================================================

  def _inplace_set_op(self, target: "OGSCatalog", add: bool) -> None:
    """Mutate this catalog's aggregate state by union or subtraction.

    Only ``self`` is updated; ``target`` is treated as read-only input.
    For each of the date-to-path caches (``picks_`` and ``events_``), add
    mode performs ``self <- self union target`` with ``target`` winning on
    duplicate dates, while subtract mode removes any dates also present in
    ``target``. The corresponding aggregate DataFrames (``PICKS`` and
    ``EVENTS``) are then updated in place: add mode concatenates populated
    frames or copies ``target`` when ``self`` is empty, and subtract mode
    drops rows whose shared identifier values appear in the matching target
    frame. The loaded per-day caches (``picks`` and ``events``) are not
    rebuilt here.
    """
    for cache_attr, df_attr in (("picks_", "PICKS"), ("events_", "EVENTS")):
      cache = getattr(self, cache_attr)
      tcache = getattr(target, cache_attr)
      df = getattr(self, df_attr)
      tdf = getattr(target, df_attr)
      if add:
        setattr(self, cache_attr, {**cache, **tcache})
        if not df.empty and not tdf.empty:
          setattr(self, df_attr, pd.concat([df, tdf], ignore_index=True))
        elif df.empty:
          setattr(self, df_attr, tdf.copy())
      else:
        setattr(self, cache_attr,
                {k: v for k, v in cache.items() if k not in tcache})
        setattr(self, df_attr,
                df[~df[OGS_C.INDEX_STR].isin(tdf[OGS_C.INDEX_STR])])

  # -------------------------------------------------------------------------
  # BGMA pick diagnostic plotters
  # -------------------------------------------------------------------------

  def _plot_picks_confmtx(
    self, target: "OGSCatalog", mtx: pd.DataFrame,
    recall_stats: tuple, fdr_stats: tuple,
    output: Optional[Path] = None,
  ) -> None:
    """Render the pick confusion-matrix summary figure.

    ``mtx`` is the 3x3 BGMA picks matrix with BASE phases on rows and TARGET
    phases on columns. ``recall_stats`` and ``fdr_stats`` are the
    ``(overall, P, S)`` metrics already computed from that matrix. When
    ``output`` is ``None``, the figure is written to the default artifact
    ``<base>_<target>_PicksConfMtx{IMAGE_EXT}`` under ``self.output / "img"``;
    otherwise the supplied path is used unchanged.
    """
    from matplotlib import pyplot as plt
    import ogsplotter as OGS_P
    assert isinstance(self.PicksMH, pd.DataFrame)
    assert isinstance(self.PicksPS, pd.DataFrame)
    recall, p_recall, s_recall = recall_stats
    fdr, p_fdr, s_fdr = fdr_stats
    filepath = self._plot_output(target, "PicksConfMtx", output)
    OGS_P.ConfMtx_plotter(
      mtx.values,
      title="Recall: {:.4f}, Recall P: {:.4f}, Recall S: {:.4f}".format(
        recall, p_recall, s_recall
      ),
      subtitle=" FDR: {:.4f}, FDR P: {:.4f}, FDR S: {:.4f}".format(
        fdr, p_fdr, s_fdr
      ),
      label=mtx.columns.tolist(),
      output=filepath,
      basename=self.name,
      targetname=target.name
    )
    plt.close()

  def _plot_picks_time_diff(
    self, target: "OGSCatalog", output: Optional[Path] = None,
  ) -> bool:
    """Plot matched-pick time deltas from the paired ``PicksMH`` rows.

    This helper assumes ``self.PicksMH`` came from :meth:`bgmaPicks`, where
    ``TIME_STR`` stores ``(base_time, target_time)`` tuples as stringified
    UTC timestamps and ``PHASE_STR`` stores the shared scalar phase used for
    the P/S overlays. When ``output`` is ``None``, the histogram is written to
    ``<base>_<target>_PicksTimeDiff{IMAGE_EXT}`` under ``self.output / "img"``;
    otherwise the supplied path is used unchanged. Returns ``False`` only when
    there are no matched rows to histogram.
    """
    from matplotlib import pyplot as plt
    import ogsplotter as OGS_P
    assert isinstance(self.PicksMH, pd.DataFrame)
    picks_mh = cast(pd.DataFrame, self.PicksMH)
    # ``PicksMH[TIME_STR]`` holds ``(base_time, target_time)`` tuples, so we
    # bulk-parse both sides once and reuse the resulting delta series for the
    # combined, P-wave, and S-wave histograms.
    times = cast(pd.Series, picks_mh[OGS_C.TIME_STR])
    t0 = cast(pd.Series, pd.to_datetime(times.str[0], utc=True,
                                        errors="coerce"))
    t1 = cast(pd.Series, pd.to_datetime(times.str[1], utc=True,
                                        errors="coerce"))
    delta = cast(pd.Series, t1 - t0)
    data = cast(pd.Series, delta.dt.total_seconds())
    if data.empty:
      self.logger.warning(
        "No matched picks to plot time difference histogram."
      )
      return False
    p_mask, s_mask = self._ps_phase_masks(picks_mh)
    def _label(phase: str, data: pd.Series) -> str:
      return (
        f"{phase} Picks: $\\mu$ = {data.mean():.3E}, "
        f"$\\sigma$ = {data.std():.3E},\n"
        f"RMSE = {np.sqrt((data**2).mean()):.4f} s, "
        f"MAE = {data.abs().mean():.4f} s"
      )
    pickdiff = OGS_P.histogram_plotter(
      data,
      xlabel="Time Difference (s)",
      title=(f"RMSE = {np.sqrt((data**2).mean()):.4f} s, "
             f"MAE = {data.abs().mean():.4f} s"),
      legend=True,
      label="Matched (MH)",
      color=OGS_C.MEX_PINK,
      xlim=(-OGS_C.PICK_TIME_OFFSET.total_seconds(),
            OGS_C.PICK_TIME_OFFSET.total_seconds()))
    p_data = data[p_mask]
    pickdiff.add_plot(
      p_data,
      alpha=1,
      step=True,
      color=OGS_C.OGS_BLUE,
      label=_label("P", p_data),
    )
    s_data = data[s_mask]
    pickdiff.add_plot(
      s_data,
      alpha=1,
      color=OGS_C.ALN_GREEN,
      step=True,
      label=_label("S", s_data),
      legend=True,
      output=self._plot_output(target, "PicksTimeDiff", output),
    )
    plt.close()
    return True

  def _plot_picks_confidence(
    self, target: "OGSCatalog", output: Optional[Path] = None,
  ) -> None:
    """Plot confidence histograms for matched and proposed pick diagnostics.

    ``self.PicksMH`` is expected to use the BGMA matched-pick schema, where
    ``PROBABILITY_STR`` stores ``(base_confidence, target_confidence)`` tuples
    and ``PHASE_STR`` stores the matched scalar phase for P/S masks.
    ``self.PicksPS`` contributes proposed TARGET-only picks with scalar
    ``PROBABILITY_STR`` values. When ``output`` is ``None``, the histogram is
    written to ``<base>_<target>_PicksConfDist{IMAGE_EXT}`` under
    ``self.output / "img"``; otherwise the supplied path is used unchanged.
    """
    from matplotlib import pyplot as plt
    import ogsplotter as OGS_P
    assert isinstance(self.PicksMH, pd.DataFrame)
    assert isinstance(self.PicksPS, pd.DataFrame)
    picks_mh = cast(pd.DataFrame, self.PicksMH)
    picks_ps = cast(pd.DataFrame, self.PicksPS)
    target_confidence_col = "_target_confidence_review"
    if target_confidence_col not in picks_mh.columns:
      probabilities = cast(pd.Series, picks_mh[OGS_C.PROBABILITY_STR])
      picks_mh.loc[:, target_confidence_col] = cast(
        pd.Series,
        probabilities.apply(lambda x: x[1]),
      )
    conf_all = cast(pd.Series, picks_mh[target_confidence_col])
    p_mask, s_mask = self._ps_phase_masks(picks_mh)
    # ``conf_all`` scores the TARGET side of matched rows; ``PicksPS`` already
    # stores scalar TARGET confidences for the proposed-only distribution.
    myconf = OGS_P.histogram_plotter(
      conf_all,
      xlabel="Pick Confidence",
      title="Pick Confidence Distribution",
      label="Matched (MH)",
      xlim=(0, 1),
    )
    myconf.add_plot(
      conf_all[p_mask],
      alpha=1,
      step=True,
      color=OGS_C.MEX_PINK,
      label="MH P Picks",
    )
    myconf.add_plot(
      conf_all[s_mask],
      alpha=1,
      color=OGS_C.ALN_GREEN,
      step=True,
      label="MH S Picks",
    )
    myconf.add_plot(
      cast(pd.Series, picks_ps[OGS_C.PROBABILITY_STR]),
      alpha=1,
      color=OGS_C.LIP_ORANGE,
      step=True,
      label="Proposed (PS)",
      legend=True,
      yscale='log',
      output=self._plot_output(target, "PicksConfDist", output),
    )
    plt.close()

  # =========================================================================
  # COMBINED WORKFLOW & OPERATOR OVERLOADS
  # =========================================================================
  # ``bpgma`` runs the full event ➜ pick ➜ (optional) waveform review back
  # to back. The ``__iadd__`` / ``__isub__`` operators expose the catalog
  # set-operation primitive for ad hoc composition outside the workflow.
  # =========================================================================

  def _run_bgma_if_ready(
      self,
      target: "OGSCatalog",
      *,
      cache_attr: str,
      load_key: str,
      get_key: str,
      kind: str,
      runner,
    ) -> None:
    """Run one BGMA pass only when both catalogs expose the needed data.

    This helper is the narrow orchestration gate used by ``bpgma``. It first
    not, there is no BASE data to compare and the pass is skipped silently.
    It then accepts either an already-populated TARGET cache or any data that
    can be resolved lazily via
    ``target.load(load_key)`` / ``target.get(get_key)``. When TARGET still
    resolves to no rows, the skip is logged and ``runner`` is not invoked.

    Side Effects
    ------------
    Emits info-level logging for skip/start decisions and, on success, calls
    ``runner(target)``, which populates the corresponding BGMA review artifacts
    on ``self``.
    """
    if not getattr(self, cache_attr):
      return
    target_cache = getattr(target, cache_attr)
    # Accept either an already-loaded TARGET day cache or a lazily loaded
    # aggregate frame before deciding that this comparison slice is empty.
    if (target_cache == {} and target.load(load_key) == {}
        and target.get(get_key).empty):
      self.logger.info("%s catalog has no %ss to compare.", target.name, kind)
      return
    self.logger.info(
      "Starting BGMA %s comparison between %s and %s.",
      kind, self.name, target.name,
    )
    runner(target)

  def bpgma(self,
        target: "OGSCatalog",
        stations: Optional[Path] = None,
        waveforms: Optional[Path] = None,
        vlines: list[tuple[datetime, str, str]] = []
      ) -> None:
    """Orchestrate BGMA event and pick review for one target catalog.

    Optional context data is loaded before any BGMA pass runs. When both
    ``waveforms`` and ``stations`` are provided, ``OGS_U.waveforms(...)``
    loads waveform metadata and the derived station inventory for the current
    ``self.start`` / ``self.end`` window. When only ``stations`` is provided,
    ``self.waveforms`` is cleared and ``self.stations`` is loaded from that
    inventory file. When ``stations`` is missing, no waveform metadata is
    loaded for this run, even if ``waveforms`` was supplied.

    After those optional loads, BGMA passes are attempted in fixed order:
    events first, then picks. Each pass is gated by ``_run_bgma_if_ready`` and
    therefore runs only when ``self`` already has the relevant cache populated
    and ``target`` resolves to non-empty data. Pick comparison still requires a
    station inventory, so provide ``stations`` whenever pick caches are present.

    Side Effects
    ------------
    Updates ``self.waveforms`` and ``self.stations`` for the current run.
    Successful BGMA passes overwrite the matching review attributes on ``self``
    (for example ``EventsMH`` / ``EventsMS`` / ``EventsPS`` and the ``Picks*``
    outputs) and may write plots or CSV artifacts under ``self.output``.

    Parameters
    ----------
    target : OGSCatalog
      Catalog to compare against.
    stations : Optional[Path]
      Path to station metadata file.
    waveforms : Optional[Path]
      Path to waveform metadata with ``NETWORK``, ``STATION``, ``DATE``, and
      ``FILENAME`` columns. It is used only when ``stations`` is also given.
    vlines : list[tuple[datetime, str, str]]
      Vertical marker lines forwarded to ``OGS_U.waveforms`` when waveform
      metadata is loaded.

    Examples
    --------
    Compare only events after loading event caches on both catalogs::

      base.load("events")
      target.load("events")
      base.bpgma(target)

    Compare events and picks with station metadata::

      base.load("events")
      base.load("picks")
      target.load("events")
      target.load("picks")
      base.bpgma(target, stations=Path("/path/to/stations"))

    Add waveform lookup data used by downstream review helpers::

      base.bpgma(
        target,
        stations=Path("/path/to/stations"),
        waveforms=Path("/path/to/waveforms.csv"),
      )
    """
    if not isinstance(target, OGSCatalog):
      raise ValueError("Can only perform bpgma on OGSCatalog")
    cache = cast(Optional[dict[tuple[Hashable, ...], Any]],
                 getattr(self, "_bpgma_context_cache", None))
    if cache is None:
      cache = {}
      self._bpgma_context_cache = cache
    if waveforms is not None and stations is not None:
      cache_key = (
        "waveforms",
        waveforms,
        stations,
        self.start,
        self.end,
        self.output,
        tuple(vlines),
      )
      if cache_key not in cache:
        loaded_waveforms, loaded_stations = OGS_U.waveforms(
          waveforms,
          stations,
          self.start,
          self.end,
          vlines=vlines,
          output=self.output
        )
        cache[cache_key] = (loaded_waveforms.copy(), loaded_stations.copy())
        self.waveforms, self.stations = loaded_waveforms, loaded_stations
      else:
        cached_waveforms, cached_stations = cast(
          tuple[pd.DataFrame, pd.DataFrame],
          cache[cache_key],
        )
        self.waveforms = cached_waveforms.copy()
        self.stations = cached_stations.copy()
    else:
      self.waveforms = None
      if stations is None:
        self.stations = None
      else:
        cache_key = ("stations", stations, self.output)
        if cache_key not in cache:
          loaded_stations = OGS_U.inventory(stations, output=self.output)
          cache[cache_key] = loaded_stations.copy()
          self.stations = loaded_stations
        else:
          self.stations = cast(pd.DataFrame, cache[cache_key]).copy()
    # Run the broader event pass first, then the station-aware pick review.
    self._run_bgma_if_ready(
      target,
      cache_attr="events_",
      load_key="events",
      get_key="EVENTS",
      kind="event",
      runner=self.bgmaEvents,
    )
    self._run_bgma_if_ready(
      target,
      cache_attr="picks_",
      load_key="picks",
      get_key="PICKS",
      kind="pick",
      runner=self.bgmaPicks,
    )

  def __iadd__(self, target):
    """Merge another catalog into ``self`` in place.

    This is the operator front-end for ``_inplace_set_op(add=True)``. Only
    ``self`` is mutated: the date caches and aggregate ``PICKS`` / ``EVENTS``
    DataFrames are updated, while ``target`` is treated as read-only input.
    Duplicate dates in the cache maps are resolved in favor of ``target`` and
    aggregate frames are concatenated as-is here. Loaded per-day caches are not
    rebuilt by this operator.

    Parameters
    ----------
    target : OGSCatalog
      Catalog to merge in.

    Returns
    -------
    OGSCatalog
      ``self`` after the in-place update.

    Examples
    --------
    ``base += target``
    """
    if not isinstance(target, OGSCatalog):
      raise ValueError("Can only add OGSCatalog to OGSCatalog")
    self._inplace_set_op(target, add=True)
    return self

  def __isub__(self, target):
    """Subtract another catalog from ``self`` in place.

    This is the operator front-end for ``_inplace_set_op(add=False)``. Only
    ``self`` is mutated: any dates present in ``target`` are removed from
    ``self``'s date caches, and rows whose shared identifier values appear in
    ``target`` are filtered out of the aggregate ``PICKS`` / ``EVENTS``
    DataFrames. ``target`` remains read-only and loaded per-day caches are not
    rebuilt here.

    Parameters
    ----------
    target : OGSCatalog
      Catalog to subtract.

    Returns
    -------
    OGSCatalog
      ``self`` after the in-place update.

    Examples
    --------
    ``base -= target``
    """
    if not isinstance(target, OGSCatalog):
      raise ValueError("Can only subtract OGSCatalog from OGSCatalog")
    self._inplace_set_op(target, add=False)
    return self


# =============================================================================
# MANUAL SMOKE TEST ENTRY POINT (not part of the library API)
# =============================================================================

def main():
  """Run a local manual comparison example; this is not library API.

  The paths below are hard-coded for one workstation-specific smoke test. The
  helper builds two catalogs over a shared date range, renders the overview
  plot, and then runs ``bpgma`` with station metadata. Update the paths below
  before running this file directly.
  """
  start = datetime(2024, 3, 20)
  end = datetime(2024, 6, 20)
  stations = Path("/Users/admin/Desktop/OGS_Catalog/station")
  output = Path("/Users/admin/Desktop/Monica/PhD/comparison/OGSCatalog/OGSBackup")
  # Tiny local factory to keep this ad hoc entrypoint readable.
  def _cat(path: Path, name: str) -> OGSCatalog:
    return OGSCatalog(path, start=start, end=end, name=name,
                      verbose=True, output=output)
  BaseCatalog = _cat(
    Path("/Users/admin/Desktop/Monica/PhD/catalog/OGSCatalog/.all"),
    "OGS Catalog")
  TargetCatalog = _cat(
    Path("/Users/admin/Desktop/Monica/PhD/catalog/OGSBackup/OGSLocalMagnitude"),
    "SeisBench Catalog")
  # Plot the broad catalog comparison first, then build the BGMA review set.
  BaseCatalog.plot(targets=[TargetCatalog])
  BaseCatalog.bpgma(
    TargetCatalog,
    stations=stations,
  )


if __name__ == "__main__": main()
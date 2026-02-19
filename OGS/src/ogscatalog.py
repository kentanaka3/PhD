from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta as td
from typing import Dict, Optional

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from matplotlib.path import Path as mplPath

import ogsconstants as OGS_C


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


class OGSCatalog:
  """
  Optimized catalog container for OGS events and picks.

  This implementation emphasizes:
  - Lazy loading of daily files
  - Vectorized time conversion
  - Cached daily views for fast reuse

  Parameters
  ----------
  input : Path
    Path to the catalog directory containing 'events' and 'assignments'
    subdirectories.
  start : datetime, optional
    Start date for filtering events and picks. Default is datetime.max.
  end : datetime, optional
    End date for filtering events and picks. Default is datetime.min.
  verbose : bool, optional
    If True, enables verbose output during processing.
  polygon : mplPath, optional
    Polygon defining the study region for filtering events. Default is
    OGS_C.OGS_POLY_REGION.
  output : Path, optional
    Output directory for storing results. Default is
    OGS_C.THIS_FILE.parent / "data" / "OGSCatalog".
  name : str, optional
    Name of the catalog for identification. Default is an empty string.

  Attributes
  ----------
  PICKS : pd.DataFrame
    DataFrame containing all picks in the catalog.
  EVENTS : pd.DataFrame
    DataFrame containing all events in the catalog.
  events_ : Dict[datetime, Path]
    Mapping of dates to event file paths.
  picks_ : Dict[datetime, Path]
    Mapping of dates to pick file paths.
  events : Dict[datetime, pd.DataFrame]
    Cached daily event DataFrames.
  picks : Dict[datetime, pd.DataFrame]
    Cached daily pick DataFrames.
  polygon : mplPath
    Polygon defining the study region for filtering events.
  verbose : bool
    If True, enables verbose output during processing.
  output : Path
    Output directory for storing results.
  name : str
    Name of the catalog for identification.

  Methods
  -------
  load(key: str) -> Dict[datetime, pd.DataFrame]
    Load daily data for the specified key ('events' or 'picks').
  postload(key: str) -> Dict[datetime, pd.DataFrame]
    Build daily caches from loaded full DataFrames.
  get(key: str) -> pd.DataFrame
    Retrieve the DataFrame for the specified key ('EVENTS' or 'PICKS').
  plot(others: list[OGSCatalog] = [],
        waveforms: Dict[str, Dict[str, list[Path]]] = dict(),
        output: Optional[Path] = None) -> None
    Generate plots comparing this catalog with other catalogs.
  plot_events(others: list[OGSCatalog] = [],
              output: Optional[Path] = None) -> None
    Plot event locations for this catalog and optional comparisons.
  plot_events_fn_waveforms(picks: pd.DataFrame,
                           event: pd.Series,
                           waveforms: Dict[str, Dict[str, list[Path]]],
                           output: Optional[Path] = None) -> None
    Plot waveforms for false negative events.
  bgmaEvents(other: OGSCatalog) -> None
    Perform BGMA event matching with another catalog.
  bgmaPicks(other: OGSCatalog) -> None
    Perform BGMA pick matching with another catalog.
  bpgma(other: OGSCatalog,
        stations: dict[str, tuple[float, float, float, str, str, str, str]]) -> None
    Perform BGMA matching on events and picks.


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
    """Initialize the catalog container.

    Parameters
    ----------
    input : Path
      Root directory containing events, picks, or assignments.
    start : datetime, optional
      Start date for filtering (inclusive).
    end : datetime, optional
      End date for filtering (inclusive).
    verbose : bool, optional
      Enable DEBUG logging if True.
    polygon : mplPath, optional
      Polygon defining the study region.
    output : Path, optional
      Output directory for generated artifacts.
    name : str, optional
      Catalog display name.
    """
    if not input.exists():
      raise FileNotFoundError(f"Input path {input} does not exist.")
    self.name = output.name if name == OGS_C.EMPTY_STR else name
    self.input = input
    self.start = start
    self.end = end
    self.polygon : Optional[mplPath] = polygon
    self.logger = OGS_C.setup_logger(f"{__name__}.{self.__class__.__name__}", verbose)
    self.output = output
    if not self.output.exists():
      self.output.mkdir(parents=True, exist_ok=True)
    (self.output / "img").mkdir(parents=True, exist_ok=True)
    self.picks_ : dict[datetime, Path] = dict() # raw file paths
    self.events_ : dict[datetime, Path] = dict() # raw file paths
    self.picks : dict[datetime, pd.DataFrame] = dict()
    self.events : dict[datetime, pd.DataFrame] = dict()
    self.PICKS : pd.DataFrame = pd.DataFrame(columns=[
      OGS_C.IDX_PICKS_STR, OGS_C.GROUPS_STR, OGS_C.TIME_STR,
      OGS_C.STATION_STR, OGS_C.PHASE_STR, OGS_C.PROBABILITY_STR,
      OGS_C.EPICENTRAL_DISTANCE_STR, OGS_C.DEPTH_STR,
      OGS_C.AMPLITUDE_STR, OGS_C.STATION_ML_STR
    ])
    self.EVENTS : pd.DataFrame = pd.DataFrame(columns=[
      OGS_C.IDX_EVENTS_STR, OGS_C.TIME_STR, OGS_C.LATITUDE_STR,
      OGS_C.LONGITUDE_STR, OGS_C.DEPTH_STR, OGS_C.GAP_STR, OGS_C.ERZ_STR,
      OGS_C.ERH_STR, OGS_C.ERT_STR, OGS_C.GROUPS_STR, OGS_C.NO_STR,
      OGS_C.NUMBER_P_PICKS_STR, OGS_C.NUMBER_S_PICKS_STR,
      OGS_C.NUMBER_P_AND_S_PICKS_STR, OGS_C.ML_STR, OGS_C.ML_MEDIAN_STR,
      OGS_C.ML_UNC_STR, OGS_C.ML_STATIONS_STR
    ])
    self.preload()

  def preload(self) -> None:
    """Preload daily file paths into internal caches."""
    for filepath in self.input.rglob("events/*"):
      if filepath.is_file():
        try:
          date = UTCDateTime(filepath.stem).date
        except Exception as e:
          self.logger.exception(f"Error parsing date from {filepath.stem}")
          continue
        if self.start.date() <= date <= self.end.date():
          self.events_[date] = filepath
    for filepath in self.input.rglob("assignments/*"):
      if filepath.is_file():
        try:
          date = UTCDateTime(filepath.stem).date
        except Exception as e:
          self.logger.exception(f"Error parsing date from {filepath.stem}")
          continue
        if self.start.date() <= date <= self.end.date():
          self.picks_[date] = filepath
    for filepath in self.input.rglob("picks/*"):
      if filepath.is_file():
        try:
          date = UTCDateTime(filepath.stem).date
        except Exception as e:
          self.logger.exception(f"Error parsing date from {filepath.stem}")
          continue
        if self.start.date() <= date <= self.end.date():
          self.picks_[date] = filepath

  def load_(self, filepath : Path) -> pd.DataFrame:
    """Load a file into a DataFrame.

    Parameters
    ----------
    filepath : Path
      File to load (CSV or Parquet).

    Returns
    -------
    pd.DataFrame
      Loaded DataFrame or empty on failure.
    """
    try:
      if filepath.suffix == OGS_C.CSV_EXT:
        return pd.read_csv(filepath)
      else:
        return pd.read_parquet(filepath)
    except Exception:
      self.logger.exception(f"Error loading {filepath}")
      return pd.DataFrame(columns=[])

  def _load_day(self, key: str, date) -> pd.DataFrame:
    """Lazily load a single day's data from disk into the daily cache.

    If the day is already cached, returns immediately without disk I/O.
    For events, applies the polygon filter when ``self.polygon`` is set.

    Parameters
    ----------
    key : str
      Either "events" or "picks".
    date : datetime.date
      The date whose data to load.

    Returns
    -------
    pd.DataFrame
      The loaded (and possibly polygon-filtered) DataFrame for this day.
    """
    if key == "events":
      if date in self.events:
        return self.events[date]
      events = self.load_(self.events_[date])
      if not events.empty:
        if self.polygon is not None:
          mask = contains_points(
            self.polygon.vertices, # type: ignore
            events[[OGS_C.LONGITUDE_STR, OGS_C.LATITUDE_STR]].to_numpy()
          )
          events = events[mask]
        if events.empty:
          self.logger.warning(f"All events for {date} filtered out by polygon from {self.events_[date]}")
      else:
        self.logger.warning(f"No events loaded for {date} from {self.events_[date]}")
      self.events[date] = events
      return self.events[date]
    elif key == "picks":
      if date in self.picks:
        return self.picks[date]
      self.picks[date] = self.load_(self.picks_[date])
      return self.picks[date]
    else:
      raise ValueError(f"Unknown key: {key}")

  def load(self, key: str,
  ) -> Dict[datetime, pd.DataFrame]:
    """Load daily data for the provided key.

    Delegates to :meth:`_load_day` for each date not yet cached, so
    previously loaded days are skipped.

    Parameters
    ----------
    key : str
      Either "events" or "picks".

    Returns
    -------
    Dict[datetime, pd.DataFrame]
      Mapping of date to DataFrame.
    """
    if key == "events":
      missing = set(self.events_.keys()) - set(self.events.keys())
      if missing:
        self.logger.info(f"Loading {self.name} events data...")
        for date in missing:
          self._load_day("events", date)
      return self.events
    elif key == "picks":
      missing = set(self.picks_.keys()) - set(self.picks.keys())
      if missing:
        self.logger.info(f"Loading {self.name} picks data...")
        for date in missing:
          self._load_day("picks", date)
      return self.picks
    else:
      raise ValueError(f"Unknown key: {key}")

  def postload(self, key: str, update: bool = False) -> Dict[datetime, pd.DataFrame]:
    """Build daily caches from in-memory DataFrames.

    Parameters
    ----------
    key : str
      Either "events" or "picks".

    Returns
    -------
    Dict[datetime, pd.DataFrame]
      Mapping of date to DataFrame.
    """
    if key == "events":
      if not update and not self.EVENTS.empty:
        for date, df in self.EVENTS.groupby(OGS_C.GROUPS_STR):
          self.events[UTCDateTime(date).date] = df
      return self.events
    elif key == "picks":
      if not update and not self.PICKS.empty:
        for date, df in self.PICKS.groupby(OGS_C.GROUPS_STR):
          self.picks[UTCDateTime(date).date] = df
      return self.picks
    else:
      raise ValueError(f"Unknown key: {key}")

  def get(self, key: str) -> pd.DataFrame:
    """Return the aggregated DataFrame for a given key.

    Parameters
    ----------
    key : str
      Either "EVENTS" or "PICKS".

    Returns
    -------
    pd.DataFrame
      Aggregated DataFrame for the key.
    """
    if key == "EVENTS":
      if self.EVENTS.empty:
        self.logger.info(f"Loading {self.name} EVENTS data...")
        events = self.load(key.lower())
        if events:
          self.EVENTS = pd.concat(events.values()).reset_index(drop=True)
        else:
          self.logger.warning(f"No {self.name} EVENTS data loaded.")
      return self.EVENTS
    if key == "PICKS":
      if self.PICKS.empty:
        self.logger.info(f"Loading {self.name} PICKS data...")
        picks = self.load(key.lower())
        if picks:
          self.PICKS = pd.concat(picks.values()).reset_index(drop=True)
        else:
          self.logger.warning(f"No {self.name} PICKS data loaded.")
      return self.PICKS
    else: raise ValueError(f"Unknown key: {key}")

  def plot_events(self, others: list[OGSCatalog] = [],
                  output: Optional[Path] = None) -> None:
    """Plot event locations for this catalog and optional comparisons.

    Parameters
    ----------
    others : list[OGSCatalog], optional
      Additional catalogs to overlay.
    output : Optional[Path], optional
      Output path for the figure.
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
      output=output if output is not None else self.output / "img" / f"{self.input.name}_EventsMap.png",
      magnitude=events[OGS_C.MAGNITUDE_L_STR] if OGS_C.MAGNITUDE_L_STR in events.columns else None
    )
    for other, color in zip(others, OGS_C.PLOT_COLORS[1:len(others)+1]):
      events = other.get("EVENTS")
      if events.empty:
        self.logger.info(f"No events to plot for {other.name}.")
        continue
      else:
        self.logger.info(f"Plotting events for {other.name}.")
      eventsMap.add_plot(
        x=events[OGS_C.LONGITUDE_STR],
        y=events[OGS_C.LATITUDE_STR],
        legend=True,
        marker='o',
        color="none",
        facecolors='none',
        edgecolors=color,
        label=other.name,
        output=output if output is not None else self.output / "img" / f"{self.input.name}_{other.input.name}_EventsMap.png",
        magnitude=events[OGS_C.MAGNITUDE_L_STR] if OGS_C.MAGNITUDE_L_STR in events.columns else None
      )
    plt.close()

  def plot(self,
        others: list[OGSCatalog] = [],
        vlines: list[tuple[datetime, str, str]] = []
      ) -> None:
    """Generate summary plots for events and picks.

    Parameters
    ----------
    others : list[OGSCatalog], optional
      Additional catalogs for comparison plots.
    waveforms : dict[str, dict[str, list[Path]]], optional
      Waveform files grouped by day and station.
    output : Optional[Path], optional
      Output path for plot files.
    """
    self.plot_events(others=others)
    self.plot_erh_histogram(others=others)
    self.plot_erz_histogram(others=others)
    self.plot_ert_histogram(others=others)
    self.plot_magnitude_histogram(others=others)
    self.plot_depth_histogram(others=others)
    self.plot_cumulative_events(others=others, vlines=vlines)
    self.plot_cumulative_picks(others=others, vlines=vlines)

  def plot_events_ms_waveforms(self,
        picks: pd.DataFrame,
        event: pd.Series,
        waveforms: dict[str, list[Path]],
        output: Optional[Path] = None
      ) -> None:
    """Plot waveforms for Missed (MS) events.

    Parameters
    ----------
    picks : pd.DataFrame
      Picks to plot for the event.
    event : pd.Series
      Event row containing metadata.
    waveforms : dict[str, list[Path]]
      Waveform file paths grouped by station.
    output : Optional[Path], optional
      Output file path for the plot.
    """
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    myfnplot = OGS_P.event_plotter(
      picks=picks,
      event=event,
      stations=list(self.stations[OGS_C.STATION_STR].unique()),
      waveforms=waveforms,
      inventory=self.stations,
      title=(
        f"Missed (MS) Event {event[OGS_C.IDX_EVENTS_STR]}" +
        (f" ($M_L$ {event[OGS_C.MAGNITUDE_L_STR]})" \
         if OGS_C.MAGNITUDE_L_STR in event else OGS_C.EMPTY_STR) +
        f" | Proposed (PS) Picks {event[OGS_C.TIME_STR] - td(seconds=1)}"
      ),
    )
    myfppicks = self.PICKS[
      self.PICKS[OGS_C.TIME_STR].between( # type: ignore
        event[OGS_C.TIME_STR] - td(seconds=1),
        event[OGS_C.TIME_STR] + td(seconds=30)
      )
    ]
    myfnplot.add_plot(picks=myfppicks, flip=True,
      output=(output if output is not None else self.output / "img" /
              f"{self.input.name}_MS{event[OGS_C.GROUPS_STR]}_{event[OGS_C.IDX_EVENTS_STR]}.png")
    )
    plt.close()

  def plot_events_ps_waveforms(self,
        picks: pd.DataFrame,
        event: pd.Series,
        waveforms: dict[str, list[Path]],
        output: Optional[Path] = None
      ) -> None:
    """Plot waveforms for Proposed (PS) events.

    Parameters
    ----------
    picks : pd.DataFrame
      Picks to plot for the event.
    event : pd.Series
      Event row containing metadata.
    waveforms : dict[str, list[Path]]
      Waveform file paths grouped by station.
    output : Optional[Path], optional
      Output file path for the plot.
    """
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    mypsplot = OGS_P.event_plotter(
      picks=picks,
      event=event,
      stations=list(self.stations[OGS_C.STATION_STR].unique()),
      waveforms=waveforms,
      inventory=self.stations,
      title=(
        f"Proposed (PS) Event {event[OGS_C.IDX_EVENTS_STR]}" +
        (f" ($M_L$ {event[OGS_C.MAGNITUDE_L_STR]})" \
          if OGS_C.MAGNITUDE_L_STR in event else OGS_C.EMPTY_STR) +
        f" | Proposed (PS) Picks {event[OGS_C.TIME_STR] - td(seconds=1)}"
      ),
    )
    mypspicks = self.PICKS[
      self.PICKS[OGS_C.TIME_STR].between( # type: ignore
        event[OGS_C.TIME_STR] - td(seconds=1),
        event[OGS_C.TIME_STR] + td(seconds=30)
      )
    ]
    mypsplot.add_plot(picks=mypspicks, flip=True,
      output=(output if output is not None else self.output / "img" /
              f"{self.input.name}_PS{event[OGS_C.GROUPS_STR]}_{event[OGS_C.IDX_EVENTS_STR]}.png")
    )
    plt.close()

  def plot_cumulative_picks(self,
                            others: list[OGSCatalog] = [],
                            output: Optional[Path] = None,
                            vlines: list[tuple[datetime, str, str]] = []):
    """Plot cumulative picks over time.

    Parameters
    ----------
    others : list[OGSCatalog], optional
      Additional catalogs for comparison.
    output : Optional[Path], optional
      Output path for the plot.
    """
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    picks = self.get("PICKS")
    if picks.empty or OGS_C.GROUPS_STR not in picks.columns:
      self.logger.info("No Date data available for histogram.")
      return
    cumulative = OGS_P.day_plotter(
      picks=picks.sort_values(OGS_C.GROUPS_STR)[OGS_C.GROUPS_STR],
      title=f"Cumulative Picks",
      output=(output if output is not None else self.output / "img" /
              f"{self.input.name}_CumulativePicks.png"),
      label=self.name,
      color=OGS_C.OGS_BLUE,
      vlines=vlines
    )
    for other, color in zip(others, OGS_C.PLOT_COLORS[1:]):
      if not isinstance(other, OGSCatalog):
        raise ValueError("Can only perform cumulative picks with OGSCatalog")
      picks = other.get("PICKS")
      if picks.empty or OGS_C.GROUPS_STR not in picks.columns:
        self.logger.info("No Date data available for histogram.")
        continue
      cumulative.add_plot(
        picks=picks.sort_values(OGS_C.GROUPS_STR)[OGS_C.GROUPS_STR],
        title=f"Cumulative Picks",
        output=(output if output is not None else self.output / "img" /
                f"{self.input.name}_{other.input.name}_CumulativePicks.png"),
        label=other.name,
        legend=True,
        color=color,
      )
    plt.close()

  def plot_cumulative_events(self,
                             others: list[OGSCatalog] = [],
                             output: Optional[Path] = None,
                             vlines: list[tuple[datetime, str, str]] = []):
    """Plot cumulative events over time.

    Parameters
    ----------
    others : list[OGSCatalog], optional
      Additional catalogs for comparison.
    output : Optional[Path], optional
      Output path for the plot.
    """
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    events = self.get("EVENTS")
    if events.empty or OGS_C.GROUPS_STR not in events.columns:
      self.logger.info("No Date data available for histogram.")
      return
    hist = OGS_P.day_plotter(
      picks=events.sort_values(OGS_C.GROUPS_STR)[OGS_C.GROUPS_STR],
      title=f"Cumulative Events",
      output=(output if output is not None else self.output / "img" /
              f"{self.input.name}_CumulativeEvents.png"),
      label=self.name,
      color=OGS_C.OGS_BLUE,
      vlines=vlines,
    )
    for other, color in zip(others, OGS_C.PLOT_COLORS[1:]):
      if not isinstance(other, OGSCatalog):
        raise ValueError("Can only perform cumulative events with OGSCatalog")
      events = other.get("EVENTS")
      if events.empty or OGS_C.GROUPS_STR not in events.columns:
        self.logger.info("No Date data available for histogram.")
        continue
      hist.add_plot(
        picks=events.sort_values(OGS_C.GROUPS_STR)[OGS_C.GROUPS_STR],
        title=f"Cumulative Events",
        output=(output if output is not None else self.output / "img" /
                f"{self.input.name}_{other.input.name}_CumulativeEvents.png"),
        label=other.name,
        legend=True,
        color=color,
      )
    plt.close()

  def _plot_histogram(self,
        column: str,
        xlabel: str,
        title: str,
        file_suffix: str,
        others: list[OGSCatalog] = [],
        bins: int = OGS_C.NUM_BINS,
        output: Optional[Path] = None,
        **plotter_kwargs
      ) -> None:
    """Generic histogram plotter for any event column.

    Parameters
    ----------
    column : str
      DataFrame column name to histogram.
    xlabel : str
      X-axis label.
    title : str
      Plot title.
    file_suffix : str
      Suffix for the output filename (e.g., "ERZ", "MagL").
    others : list[OGSCatalog], optional
      Additional catalogs for comparison.
    bins : int, optional
      Number of histogram bins.
    output : Optional[Path], optional
      Output path for the plot.
    **plotter_kwargs
      Additional keyword arguments passed to ``histogram_plotter()``.
    """
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    events = self.get("EVENTS")
    if events.empty or column not in events.columns:
      self.logger.info(f"No {title} data available for histogram.")
      return
    hist = OGS_P.histogram_plotter(
      data=events[column].dropna(),
      bins=bins,
      xlabel=xlabel,
      ylabel="Number of Events",
      title=title,
      output=(output if output is not None else self.output / "img" /
              f"{self.input.name}_{file_suffix}.png"),
      label=self.name,
      **plotter_kwargs,
    )
    for other, color in zip(others, OGS_C.PLOT_COLORS[1:]):
      if not isinstance(other, OGSCatalog):
        raise ValueError(f"Can only perform {title} with OGSCatalog")
      other_events = other.get("EVENTS")
      if other_events.empty or column not in other_events.columns:
        self.logger.info(
          f"No {title} data available for histogram for {other.name}.")
        continue
      hist.add_plot(
        data=other_events[column].dropna(),
        xlabel=xlabel,
        ylabel="Number of Events",
        title=title,
        legend=True,
        alpha=0.5,
        label=other.name,
        color=color,
        output=(output if output is not None else self.output / "img" /
                f"{self.input.name}_{other.input.name}_{file_suffix}.png")
      )
    plt.close()

  def plot_erz_histogram(self, others=[], bins=OGS_C.NUM_BINS, output=None):
    """Plot ERZ histogram for this catalog and optional comparisons."""
    self._plot_histogram(
      OGS_C.ERZ_STR, "ERZ (km)", "ERZ Histogram", "ERZ",
      others=others, bins=bins, output=output, color=OGS_C.OGS_BLUE,
      xlim=(0, 20))

  def plot_erh_histogram(self, others=[], bins=OGS_C.NUM_BINS, output=None):
    """Plot ERH histogram for this catalog and optional comparisons."""
    self._plot_histogram(
      OGS_C.ERH_STR, "ERH (km)", "ERH Histogram", "ERH",
      others=others, bins=bins, output=output, color=OGS_C.OGS_BLUE,
      xlim=(0, 20), yscale='log')

  def plot_ert_histogram(self, others=[], bins=OGS_C.NUM_BINS, output=None):
    """Plot ERT histogram for this catalog and optional comparisons."""
    self._plot_histogram(
      OGS_C.ERT_STR, "ERT (s)", "ERT Histogram", "ERT",
      others=others, bins=bins, output=output)

  def plot_depth_histogram(self, others=[], bins=OGS_C.NUM_BINS, output=None):
    """Plot depth histogram for this catalog and optional comparisons."""
    self._plot_histogram(
      OGS_C.DEPTH_STR, "Depth (km)", "Depth Histogram", "Depth",
      others=others, bins=bins, output=output, xlim=(0, 50))

  def plot_magnitude_histogram(self, others=[], bins=OGS_C.NUM_BINS,
                               output=None):
    """Plot magnitude histogram for this catalog and optional comparisons."""
    self._plot_histogram(
      OGS_C.MAGNITUDE_L_STR, "Magnitude ($M_L$)", "Magnitude Histogram",
      "MagL", others=others, bins=bins, output=output, yscale='log',
      xlim=(-1, 5))

  def bgmaEvents(self, other: "OGSCatalog") -> None:
    """Match events between catalogs using BGMA.

    Parameters
    ----------
    other : OGSCatalog
      Catalog to compare against.
    """
    self.logger.info("Starting bgmaEvents: %s vs %s", self.name, other.name)
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    if not isinstance(other, OGSCatalog):
      raise ValueError("Can only perform bgmaEvents on OGSCatalog")
    EVENTS_CFN_MTX = pd.DataFrame(0, index=[OGS_C.EVENT_STR, OGS_C.NONE_STR],
                    columns=[OGS_C.EVENT_STR, OGS_C.NONE_STR], dtype=int)
    EventsTP: list[list] = list()
    EventsFN: list[list] = list()
    EventsFP: list[list] = list()
    columns = [OGS_C.INDEX_STR, OGS_C.TIME_STR, OGS_C.LATITUDE_STR,
               OGS_C.LONGITUDE_STR, OGS_C.DEPTH_STR, OGS_C.ERH_STR,
               OGS_C.ERZ_STR, OGS_C.GAP_STR, OGS_C.MAGNITUDE_L_STR,
               OGS_C.GROUPS_STR]
    for date, _ in self.events_.items():
      BASE = self._load_day("events", date).reset_index(drop=True)
      if date not in other.events_:
        self.logger.debug("Date %s not in other catalog, skipping.", date)
        continue
      TARGET = other._load_day("events", date).reset_index(drop=True)
      I = len(BASE)
      self.logger.debug("Date %s: BASE=%d events, TARGET=%d events.",
                        date, I, len(TARGET))
      bpgEvents = OGS_C.OGSBPGraphEvents(BASE, TARGET)
      baseIDs = set(range(I))
      targetIDs = set(range(len(TARGET)))
      for i, j in bpgEvents.E:
        a, b = sorted((i, j))
        b -= I
        EVENTS_CFN_MTX.at[OGS_C.EVENT_STR, OGS_C.EVENT_STR] += 1 # type: ignore
        baseIDs.remove(a)
        targetIDs.remove(b)

        EventsTP.append([
          [BASE.at[a, col] if col in BASE.columns else None,
           TARGET.at[b, col] if col in TARGET.columns else None]
           for col in columns
        ])
      for i in baseIDs:
        EVENTS_CFN_MTX.at[OGS_C.EVENT_STR, OGS_C.NONE_STR] += 1 # type: ignore
        EventsFN.append([
          BASE.at[i, col] if col in BASE.columns else None for col in columns
        ])
      if not TARGET.empty:
        if self.polygon is not None:
          mask = contains_points(
            self.polygon.vertices, # type: ignore
            TARGET[[OGS_C.LONGITUDE_STR, OGS_C.LATITUDE_STR]].to_numpy()
          )
          fp_target = TARGET[mask].reset_index(drop=True)
        else:
          fp_target = TARGET
        for j in targetIDs:
          if j not in fp_target.index: continue
          EVENTS_CFN_MTX.at[OGS_C.NONE_STR, OGS_C.EVENT_STR] += 1 # type: ignore
          EventsFP.append([
            fp_target.at[j, col] if col in fp_target.columns else None
            for col in columns
          ])

    recall = \
      EVENTS_CFN_MTX.at[OGS_C.EVENT_STR, OGS_C.EVENT_STR] / ( # type: ignore
        EVENTS_CFN_MTX.at[OGS_C.EVENT_STR, OGS_C.EVENT_STR] +
        EVENTS_CFN_MTX.at[OGS_C.EVENT_STR, OGS_C.NONE_STR]
      )
    self.logger.info("Recall: %s", recall)
    fdr = \
      EVENTS_CFN_MTX.at[OGS_C.NONE_STR, OGS_C.EVENT_STR] / ( # type: ignore
        EVENTS_CFN_MTX.at[OGS_C.NONE_STR, OGS_C.EVENT_STR] +
        EVENTS_CFN_MTX.at[OGS_C.EVENT_STR, OGS_C.EVENT_STR]
      )
    self.logger.info("False Discovery Rate: %s", fdr)
    self.logger.info("MH: %s PS: %s MS: %s",
      EVENTS_CFN_MTX.at[OGS_C.EVENT_STR, OGS_C.EVENT_STR],
      EVENTS_CFN_MTX.at[OGS_C.NONE_STR, OGS_C.EVENT_STR],
      EVENTS_CFN_MTX.at[OGS_C.EVENT_STR, OGS_C.NONE_STR]
    )
    self.logger.info("\n%s", EVENTS_CFN_MTX)
    self.EventsTP = pd.DataFrame(EventsTP, columns=columns)
    self.EventsFN = pd.DataFrame(EventsFN, columns=columns).sort_values(
      by=OGS_C.TIME_STR
    )
    self.EventsFP = pd.DataFrame(EventsFP, columns=columns).sort_values(
      by=OGS_C.TIME_STR
    )
    filepath = (self.output /
                f"{self.input.name}_{other.input.name}_EventsMH.csv")
    self.EventsTP.to_csv(filepath, index=False)
    self.logger.info("%s written.", filepath)
    filepath = (self.output /
                f"{self.input.name}_{other.input.name}_EventsMS.csv")
    self.EventsFN.to_csv(filepath, index=False)
    self.logger.info("%s written.", filepath)
    filepath = (self.output /
                f"{self.input.name}_{other.input.name}_EventsPS.csv")
    self.EventsFP.to_csv(filepath, index=False)
    self.logger.info("%s written.", filepath)
    filepath = (self.output / "img" /
                f"{self.input.name}_{other.input.name}_EventsConfMtx.png")
    OGS_P.ConfMtx_plotter(
      EVENTS_CFN_MTX.values,
      title="Recall: {:.4f}, FDR: {:.4f}".format(recall, fdr),
      label=EVENTS_CFN_MTX.columns.tolist(),
      output=filepath,
      basename=self.name,
      targetname=other.name
    )
    plt.close()
    # Time Difference Histogram
    data = self.EventsTP[OGS_C.TIME_STR].apply(lambda x: x[1] - x[0])
    OGS_P.histogram_plotter(
      data,
      xlabel="Time Difference (s)",
      title=f"RMSE = {np.sqrt(np.mean(data ** 2)):.4f} s, " +
            f"MAE = {data.abs().mean():.4f} s",
      xlim=(-OGS_C.EVENT_TIME_OFFSET.total_seconds(),
            OGS_C.EVENT_TIME_OFFSET.total_seconds()),
      output=(self.output / "img" /
              f"{self.input.name}_{other.input.name}_EventsTimeDiff.png"),
      legend=True)
    plt.close()
    # Matched (MH) Map
    magnitude = self.EventsTP[OGS_C.MAGNITUDE_L_STR].apply(lambda x: x[0]) \
      if OGS_C.MAGNITUDE_L_STR in self.EventsTP.columns else None
    magnitude = magnitude if magnitude is not None and magnitude.notna().any() else None
    myplot = OGS_P.map_plotter(
      domain=OGS_C.OGS_STUDY_REGION,
      x=self.EventsTP[OGS_C.LONGITUDE_STR].apply(lambda x: x[0]),
      y=self.EventsTP[OGS_C.LATITUDE_STR].apply(lambda x: x[0]),
      facecolors="none", edgecolors=OGS_C.OGS_BLUE, legend=True,
      label=self.name,
      magnitude=magnitude,
    )
    magnitude = self.EventsTP[OGS_C.MAGNITUDE_L_STR].apply(lambda x: x[1]) \
      if OGS_C.MAGNITUDE_L_STR in self.EventsTP.columns else None
    magnitude = magnitude if magnitude is not None and magnitude.notna().any() else None
    myplot.add_plot(
      self.EventsTP[OGS_C.LONGITUDE_STR].apply(lambda x: x[1]),
      self.EventsTP[OGS_C.LATITUDE_STR].apply(lambda x: x[1]), color=None,
      label=other.name, legend=True, facecolors="none",
      edgecolors=OGS_C.MEX_PINK,
      magnitude=magnitude,
      output=(self.output / "img" /
              f"{self.input.name}_{other.input.name}_EventsMH.png")
    )
    plt.close()
    # Missed (MS) and Proposed (PS) Map
    magnitude = self.EventsFN[OGS_C.MAGNITUDE_L_STR] if OGS_C.MAGNITUDE_L_STR in self.EventsFN.columns else None
    magnitude = magnitude if magnitude is not None and magnitude.notna().any() else None
    myplot = OGS_P.map_plotter(
      domain=OGS_C.OGS_STUDY_REGION,
      x=self.EventsFN[OGS_C.LONGITUDE_STR],
      y=self.EventsFN[OGS_C.LATITUDE_STR],
      label=f"Missed (MS) [{self.name}] {len(self.EventsFN.index)}",
      legend=True,
      magnitude=magnitude,
    )
    magnitude = self.EventsFP[OGS_C.MAGNITUDE_L_STR] if OGS_C.MAGNITUDE_L_STR in self.EventsFP.columns else None
    magnitude = magnitude if magnitude is not None and magnitude.notna().any() else None
    myplot.add_plot(
      self.EventsFP[OGS_C.LONGITUDE_STR], self.EventsFP[OGS_C.LATITUDE_STR],
        color=None, facecolors="none", edgecolors=OGS_C.MEX_PINK, legend=True,
        label=f"Proposed (PS) [{other.name}] {len(self.EventsFP.index)}",
        magnitude=magnitude, output=(
          self.output / "img" /
          f"{self.input.name}_{other.input.name}_EventsFalse.png"
        )
    )
    plt.close()
    # Depth Difference Histogram
    OGS_P.histogram_plotter(
      self.EventsTP[OGS_C.DEPTH_STR].apply(lambda x: x[1] - x[0]),
      xlabel=f"Depth Difference (km) [{self.name} - {other.name}]",
      title="Event Depth Difference",
      xlim=(-20, 20),
      output=(self.output / "img" /
              f"{self.input.name}_{other.input.name}_DepthDiff.png"),
      legend=True)
    plt.close()
    # Event Location Scatter Plot
    OGS_P.histogram_plotter(
      OGS_P.v_lat_long_to_distance(
        self.EventsTP[OGS_C.LONGITUDE_STR].apply(lambda x: x[0]),
        self.EventsTP[OGS_C.LATITUDE_STR].apply(lambda x: x[0]),
        self.EventsTP[OGS_C.DEPTH_STR].apply(lambda x: 0),
        self.EventsTP[OGS_C.LONGITUDE_STR].apply(lambda x: x[1]),
        self.EventsTP[OGS_C.LATITUDE_STR].apply(lambda x: x[1]),
        self.EventsTP[OGS_C.DEPTH_STR].apply(lambda x: x[1]),
        dim=2
      ),
      xlim=(0, OGS_C.EVENT_DIST_OFFSET),
      xlabel=f"Epicentral Distance Difference (km) [{self.name} - {other.name}]",
      title="Event Epicentral Distance Difference",
      output=(self.output / "img" /
              f"{self.input.name}_{other.input.name}_EpiDistDiff.png"),
      legend=True)
    plt.close()
    if OGS_C.MAGNITUDE_L_STR in self.EventsTP.columns:
      # Magnitude Scatter Plot
      mymags = OGS_P.scatter_plotter(
        self.EventsTP[OGS_C.MAGNITUDE_L_STR].apply(lambda x: x[1]),
        self.EventsTP[OGS_C.MAGNITUDE_L_STR].apply(lambda x: x[0]),
        xlabel=f"{other.name} Magnitude ($M_L$)",
        ylabel=f"{self.name} Magnitude ($M_L$)",
        title="Magnitude Prediction",
        color=OGS_C.OGS_BLUE,
        legend=True
      )
      x_min = min(
        self.EventsTP[OGS_C.MAGNITUDE_L_STR].apply(lambda x: x[0]).min(),
        self.EventsTP[OGS_C.MAGNITUDE_L_STR].apply(lambda x: x[1]).min()
      )
      x_max = max(
        self.EventsTP[OGS_C.MAGNITUDE_L_STR].apply(lambda x: x[0]).max(),
        self.EventsTP[OGS_C.MAGNITUDE_L_STR].apply(lambda x: x[1]).max()
      )
      mymags.ax.plot([x_min, x_max], [x_min, x_max], color=OGS_C.MEX_PINK,
                     linestyle='--')
      mymags.ax.set_aspect('equal', adjustable='box')
      mymags.ax.grid(True)
      mymags.savefig(
        self.output / "img" /
        f"{self.input.name}_{other.input.name}_MagLDist.png"
      )
      plt.close()
      if (self.input.name in (OGS_C.TXT_EXT, ".all") and \
          other.input.name in ("OGSLocalMagnitude")):
        # Magnitude Difference Histogram
        data = self.EventsTP[OGS_C.MAGNITUDE_L_STR].apply(lambda x: x[1] - x[0])
        OGS_P.histogram_plotter(
          data,
          xlabel=f"Magnitude Difference ($M_L$) [{self.name} - {other.name}]",
          title=(f"RMSE = {np.sqrt(np.mean(data ** 2)):.4f}, " +
                 f"MAE = {data.abs().mean():.4f}"),
          xlim=(-1.5, 1.5),
          bins=21,
          output=(self.output / "img" /
                  f"{self.input.name}_{other.input.name}_MagLDiff.png"),
          legend=True
        )
        plt.close()
        # Event Magnitude Histogram
        mymags = OGS_P.histogram_plotter(
          self.EventsFP[OGS_C.MAGNITUDE_L_STR].dropna(),
          xlabel="Magnitude ($M_L$)",
          title="Event Magnitude",
          color=OGS_C.MEX_PINK,
          yscale='log',
          label=f"Proposed (PS) [{other.name}]",
          xlim=[-1., 5.0],
        )
        mymags.add_plot(
          self.EventsFN[OGS_C.MAGNITUDE_L_STR].dropna(),
          label=f"Missed (MS) [{self.name}]",
          color=OGS_C.OGS_BLUE,
          legend=True,
          output=(self.output / "img" /
                  f"{self.input.name}_{other.input.name}_MSPSMagLDist.png"),
        )
        plt.close()
      else:
        events = self.EventsFP[OGS_C.MAGNITUDE_L_STR].dropna()
        if not events.empty:
          # Event Magnitude Histogram
          OGS_P.histogram_plotter(
            events,
            xlabel="Magnitude ($M_L$)",
            title="Event Magnitude",
            color=OGS_C.MEX_PINK,
            label=f"Proposed (PS) [{other.name}]",
            output=self.output / "img" / f"{other.input.name}_PSMagLDist.png",
          )
          plt.close()
    self.EVENTS = self.get("EVENTS")
    other.EVENTS = other.get("EVENTS")

  def bgmaPicks(self, other: "OGSCatalog",) -> None:
    """Match picks between catalogs using BGMA.

    Parameters
    ----------
    other : OGSCatalog
      Catalog to compare against.
    """
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    if not isinstance(other, OGSCatalog):
      raise ValueError("Can only perform bgmaPicks on OGSCatalog")
    PICKS_CFN_MTX = pd.DataFrame(
      0, index=[OGS_C.PWAVE, OGS_C.SWAVE, OGS_C.NONE_STR],
      columns=[OGS_C.PWAVE, OGS_C.SWAVE, OGS_C.NONE_STR], dtype=int
    )
    INVENTORY = self.stations[OGS_C.STATION_STR].unique()
    self.PicksTP = list()
    self.PicksFN = list()
    self.PicksFP = list()
    columns = [OGS_C.IDX_PICKS_STR, OGS_C.TIME_STR, OGS_C.PHASE_STR,
           OGS_C.STATION_STR, OGS_C.PROBABILITY_STR]
    for date, _ in self.picks_.items():
      BASE = self._load_day("picks", date).reset_index(drop=True)
      BASE[OGS_C.NETWORK_STR] = BASE[OGS_C.STATION_STR].str.split(".").str[0]
      BASE[OGS_C.STATION_STR] = BASE[OGS_C.STATION_STR].str.split(".").str[1]
      BASE = BASE[BASE[OGS_C.STATION_STR].isin(INVENTORY)].reset_index(drop=True)
      if date not in other.picks_:
        self.logger.debug("Date %s not in other picks catalog, skipping.",
                          date)
        continue
      TARGET = other._load_day("picks", date).reset_index(drop=True)
      I = len(BASE)
      self.logger.debug("Date %s: BASE=%d picks, TARGET=%d picks.",
                        date, I, len(TARGET))
      bpgPicks = OGS_C.OGSBPGraphPicks(BASE, TARGET)
      baseIDs = set(range(I))
      targetIDs = set(range(len(TARGET)))
      for i, j in bpgPicks.E:
        a, b = sorted((i, j))
        b -= I
        PICKS_CFN_MTX.at[BASE.at[a, OGS_C.PHASE_STR],
                         TARGET.at[b, OGS_C.PHASE_STR]] += 1 # type: ignore
        if BASE.at[a, OGS_C.PHASE_STR] == TARGET.at[b, OGS_C.PHASE_STR]:
          self.PicksTP.append([
            (BASE.at[a, OGS_C.IDX_PICKS_STR],
             TARGET.at[b, OGS_C.IDX_PICKS_STR]),
            (str(BASE.at[a, OGS_C.TIME_STR]),
             str(TARGET.at[b, OGS_C.TIME_STR])),
            (BASE.at[a, OGS_C.PHASE_STR]),
            (TARGET.at[b, OGS_C.STATION_STR]),
            (BASE.at[a, OGS_C.PROBABILITY_STR],
             TARGET.at[b, OGS_C.PROBABILITY_STR])
          ])
        baseIDs.remove(a)
        targetIDs.remove(b)
      for i in baseIDs:
        PICKS_CFN_MTX.at[BASE.at[i, OGS_C.PHASE_STR], OGS_C.NONE_STR] += 1 # type: ignore
        self.PicksFN.append([BASE.at[i, col] for col in columns])
      for j in targetIDs:
        PICKS_CFN_MTX.at[OGS_C.NONE_STR,
                         TARGET.at[j, OGS_C.PHASE_STR]] += 1 # type: ignore
        self.PicksFP.append([
          TARGET.at[j, OGS_C.IDX_PICKS_STR],
          TARGET.at[j, OGS_C.TIME_STR],
          TARGET.at[j, OGS_C.PHASE_STR],
          TARGET.at[j, OGS_C.STATION_STR],
          TARGET.at[j, OGS_C.PROBABILITY_STR]
        ])
    recall = \
      (PICKS_CFN_MTX.at[OGS_C.PWAVE, OGS_C.PWAVE] + # type: ignore
       PICKS_CFN_MTX.at[OGS_C.SWAVE, OGS_C.SWAVE]) / (
        PICKS_CFN_MTX.at[OGS_C.PWAVE, OGS_C.PWAVE] + # type: ignore
        PICKS_CFN_MTX.at[OGS_C.SWAVE, OGS_C.SWAVE] +
        PICKS_CFN_MTX.at[OGS_C.PWAVE, OGS_C.SWAVE] +
        PICKS_CFN_MTX.at[OGS_C.SWAVE, OGS_C.PWAVE] +
        PICKS_CFN_MTX.at[OGS_C.PWAVE, OGS_C.NONE_STR] +
        PICKS_CFN_MTX.at[OGS_C.SWAVE, OGS_C.NONE_STR]
      )
    self.logger.info("Recall: %s", recall)
    fdr = \
      (PICKS_CFN_MTX.at[OGS_C.NONE_STR, OGS_C.PWAVE] + # type: ignore
       PICKS_CFN_MTX.at[OGS_C.NONE_STR, OGS_C.SWAVE]) / (
        PICKS_CFN_MTX.at[OGS_C.NONE_STR, OGS_C.PWAVE] + # type: ignore
        PICKS_CFN_MTX.at[OGS_C.NONE_STR, OGS_C.SWAVE] +
        PICKS_CFN_MTX.at[OGS_C.PWAVE, OGS_C.PWAVE] +
        PICKS_CFN_MTX.at[OGS_C.SWAVE, OGS_C.SWAVE] +
        PICKS_CFN_MTX.at[OGS_C.PWAVE, OGS_C.SWAVE] +
        PICKS_CFN_MTX.at[OGS_C.SWAVE, OGS_C.PWAVE]
      )
    self.logger.info("False Discovery Rate: %s", fdr)
    p_recall = PICKS_CFN_MTX.at[OGS_C.PWAVE, OGS_C.PWAVE] / ( # type: ignore
      PICKS_CFN_MTX.at[OGS_C.PWAVE, OGS_C.PWAVE] + # type: ignore
      PICKS_CFN_MTX.at[OGS_C.PWAVE, OGS_C.SWAVE] +
      PICKS_CFN_MTX.at[OGS_C.PWAVE, OGS_C.NONE_STR]
    )
    self.logger.info("Recall P-wave: %s", p_recall)
    p_fdr = PICKS_CFN_MTX.at[OGS_C.NONE_STR, OGS_C.PWAVE] / ( # type: ignore
      PICKS_CFN_MTX.at[OGS_C.NONE_STR, OGS_C.PWAVE] + # type: ignore
      PICKS_CFN_MTX.at[OGS_C.PWAVE, OGS_C.PWAVE] +
      PICKS_CFN_MTX.at[OGS_C.SWAVE, OGS_C.PWAVE]
    )
    self.logger.info("False Discovery Rate P-wave: %s", p_fdr)
    s_recall = PICKS_CFN_MTX.at[OGS_C.SWAVE, OGS_C.SWAVE] / ( # type: ignore
      PICKS_CFN_MTX.at[OGS_C.SWAVE, OGS_C.SWAVE] + # type: ignore
      PICKS_CFN_MTX.at[OGS_C.SWAVE, OGS_C.PWAVE] +
      PICKS_CFN_MTX.at[OGS_C.SWAVE, OGS_C.NONE_STR]
    )
    self.logger.info("Recall S-wave: %s", s_recall)
    s_fdr = PICKS_CFN_MTX.at[OGS_C.NONE_STR, OGS_C.SWAVE] / ( # type: ignore
      PICKS_CFN_MTX.at[OGS_C.NONE_STR, OGS_C.SWAVE] + # type: ignore
      PICKS_CFN_MTX.at[OGS_C.PWAVE, OGS_C.SWAVE] +
      PICKS_CFN_MTX.at[OGS_C.SWAVE, OGS_C.SWAVE]
    )
    self.logger.info("False Discovery Rate S-wave: %s", s_fdr)
    self.logger.info("\n%s", PICKS_CFN_MTX)
    self.logger.info("MH: %s PS: %s MS: %s",
                PICKS_CFN_MTX.at[OGS_C.PWAVE, OGS_C.PWAVE] + # type: ignore
                PICKS_CFN_MTX.at[OGS_C.SWAVE, OGS_C.SWAVE],
                PICKS_CFN_MTX.at[OGS_C.NONE_STR, OGS_C.PWAVE] +
                PICKS_CFN_MTX.at[OGS_C.NONE_STR, OGS_C.SWAVE], # type: ignore
                PICKS_CFN_MTX.at[OGS_C.PWAVE, OGS_C.NONE_STR] +
                PICKS_CFN_MTX.at[OGS_C.SWAVE, OGS_C.NONE_STR]) # type: ignore
    self.PicksTP = pd.DataFrame(self.PicksTP, columns=columns).sort_values(
      by=OGS_C.TIME_STR
    )
    self.PicksFN = pd.DataFrame(self.PicksFN, columns=columns).sort_values(
      by=OGS_C.TIME_STR
    ).sort_values(by=OGS_C.TIME_STR)
    self.PicksFP = pd.DataFrame(self.PicksFP, columns=columns).sort_values(
      by=OGS_C.TIME_STR
    ).sort_values(by=OGS_C.TIME_STR)
    filepath = self.output / f"{self.input.name}_{other.input.name}_PicksMH.csv"
    self.PicksTP.to_csv(filepath, index=False)
    self.logger.info("%s written.", filepath)
    filepath = self.output / f"{self.input.name}_{other.input.name}_PicksMS.csv"
    self.PicksFN.to_csv(filepath, index=False)
    self.logger.info("%s written.", filepath)
    filepath = self.output / f"{self.input.name}_{other.input.name}_PicksPS.csv"
    self.PicksFP.to_csv(filepath, index=False)
    self.logger.info("%s written.", filepath)
    filepath = self.output / "img" / (f"{self.input.name}_" + \
               f"{other.input.name}_PicksConfMtx.png")
    OGS_P.ConfMtx_plotter(
      PICKS_CFN_MTX.values,
      title="Recall: {:.4f}, Recall P: {:.4f}, Recall S: {:.4f}".format(
        recall, p_recall, s_recall
      ),
      subtitle=" FDR: {:.4f}, FDR P: {:.4f}, FDR S: {:.4f}".format(
        fdr, p_fdr, s_fdr
      ),
      label=PICKS_CFN_MTX.columns.tolist(),
      output=filepath,
      basename=self.name,
      targetname=other.name
    )
    plt.close()
    # Time Difference Histogram
    data = self.PicksTP[OGS_C.TIME_STR].apply(
      lambda x: UTCDateTime(x[1]) - UTCDateTime(x[0])
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
    data = self.PicksTP.loc[
      self.PicksTP[OGS_C.PHASE_STR] == OGS_C.PWAVE,
      OGS_C.TIME_STR
    ].apply(
      lambda x: UTCDateTime(x[1]) - UTCDateTime(x[0]) # type: ignore
    )
    pickdiff.add_plot(
      data,
      alpha=1,
      step=True,
      color=OGS_C.OGS_BLUE,
      label=f"P Picks: $\mu$ = {data.mean():.3E}, $\sigma$ = {data.std():.3E},\n"
            f"RMSE = {np.sqrt((data**2).mean()):.4f} s, MAE = {data.abs().mean():.4f} s",
    )
    data = self.PicksTP.loc[
      self.PicksTP[OGS_C.PHASE_STR] == OGS_C.SWAVE,
      OGS_C.TIME_STR
    ].apply(
      lambda x: UTCDateTime(x[1]) - UTCDateTime(x[0]) # type: ignore
    )
    pickdiff.add_plot(
      data,
      alpha=1,
      color=OGS_C.ALN_GREEN,
      step=True,
      label=f"S Picks: $\mu$ = {data.mean():.3E}, $\sigma$ = {data.std():.3E},\n"
            f"RMSE = {np.sqrt((data**2).mean()):.4f} s, MAE = {data.abs().mean():.4f} s",
      legend=True,
      output=(self.output / "img" /
              f"{self.input.name}_{other.input.name}_PicksTimeDiff.png"),
    )
    plt.close()
    # Confidence Histogram
    myconf = OGS_P.histogram_plotter(
      self.PicksTP[OGS_C.PROBABILITY_STR].apply(lambda x: x[1]),
      xlabel="Pick Confidence",
      title="Pick Confidence Distribution",
      label="Matched (MH)",
      xlim=(0, 1),
    )
    myconf.add_plot(
      self.PicksTP.loc[
        self.PicksTP[OGS_C.PHASE_STR] == OGS_C.PWAVE,
        OGS_C.PROBABILITY_STR
      ].apply(lambda x: x[1]),
      alpha=1,
      step=True,
      color=OGS_C.MEX_PINK,
      label="MH P Picks",
    )
    myconf.add_plot(
      self.PicksTP.loc[
        self.PicksTP[OGS_C.PHASE_STR] == OGS_C.SWAVE,
        OGS_C.PROBABILITY_STR
      ].apply(lambda x: x[1]),
      alpha=1,
      color=OGS_C.ALN_GREEN,
      step=True,
      label="MH S Picks",
    )
    myconf.add_plot(
      self.PicksFP[OGS_C.PROBABILITY_STR],
      alpha=1,
      color=OGS_C.LIP_ORANGE,
      step=True,
      label="Proposed (PS)",
      legend=True,
      yscale='log',
      output=(self.output / "img" /
              f"{self.input.name}_{other.input.name}_PicksConfDist.png"),
    )
    plt.close()
    self.PICKS = self.get("PICKS")
    other.PICKS = other.get("PICKS")

  def bpgma(self,
        other: "OGSCatalog",
        stations: Optional[Path] = None,
        waveforms: Optional[Path] = None,
        vlines: list[tuple[datetime, str, str]] = []
      ) -> None:
    """Run BGMA comparisons for events and picks.

    Parameters
    ----------
    other : OGSCatalog
      Catalog to compare against.
    stations : Optional[Path]
      Path to station metadata file.
    waveforms : Optional[Path]
      Path to waveform data file with NETWORK, STATION, DATE, FILENAME columns.
    vlines : list[tuple[datetime, str, str]]
      Vertical lines for plotting.
    """
    if not isinstance(other, OGSCatalog):
      raise ValueError("Can only perform bpgma on OGSCatalog")
    self.waveforms, self.stations = OGS_C.waveforms(
      waveforms,
      stations,
      self.start,
      self.end,
      vlines=vlines,
      output=self.output
    ) if waveforms is not None and stations is not None else (None, OGS_C.inventory(stations, output=self.output) if stations is not None else (None, None))
    if self.events_:
      if (other.events_ == {} and other.load("events") == {} and
          other.get("EVENTS").empty):
        self.logger.info("%s catalog has no events to compare.", other.name)
      else:
        self.logger.info("Starting BGMA event comparison between %s and %s.",
                         self.name, other.name)
        self.bgmaEvents(other)
    if self.picks_:
      if (other.picks_ == {} and other.load("picks") == {} and
          other.get("PICKS").empty):
        self.logger.info("%s catalog has no picks to compare.", other.name)
      else:
        self.logger.info("Starting BGMA pick comparison between %s and %s.",
                         self.name, other.name)
        self.bgmaPicks(other)

  def __iadd__(self, other):
    """In-place merge of another catalog.

    Parameters
    ----------
    other : OGSCatalog
      Catalog to merge in.

    Returns
    -------
    OGSCatalog
      Updated catalog.
    """
    if not isinstance(other, OGSCatalog):
      raise ValueError("Can only add OGSCatalog to OGSCatalog")
    self.picks_ = {**self.picks_, **other.picks_}
    if not self.PICKS.empty and not other.PICKS.empty:
      self.PICKS = pd.concat([self.PICKS, other.PICKS], ignore_index=True)
    elif self.PICKS.empty:
      self.PICKS = other.PICKS.copy()
    self.events_ = {**self.events_, **other.events_}
    if not self.EVENTS.empty and not other.EVENTS.empty:
      self.EVENTS = pd.concat([self.EVENTS, other.EVENTS], ignore_index=True)
    elif self.EVENTS.empty:
      self.EVENTS = other.EVENTS.copy()
    return self

  def __isub__(self, other):
    """In-place subtraction of another catalog.

    Parameters
    ----------
    other : OGSCatalog
      Catalog to subtract.

    Returns
    -------
    OGSCatalog
      Updated catalog.
    """
    if not isinstance(other, OGSCatalog):
      raise ValueError("Can only subtract OGSCatalog from OGSCatalog")
    self.picks_ = {k: v for k, v in self.picks_.items()
                   if k not in other.picks_}
    self.PICKS = self.PICKS[~self.PICKS[OGS_C.INDEX_STR].isin(
      other.PICKS[OGS_C.INDEX_STR])]
    self.events_ = {k: v for k, v in self.events_.items()
                    if k not in other.events_}
    self.EVENTS = self.EVENTS[~self.EVENTS[OGS_C.INDEX_STR].isin(
      other.EVENTS[OGS_C.INDEX_STR])]
    return self


def main():
  """Run a basic catalog load and plotting example."""
  start = datetime(2024, 3, 20)
  end = datetime(2024, 6, 20)
  stations = Path("/Users/admin/Desktop/OGS_Catalog/station")
  BaseCatalog: OGSCatalog = OGSCatalog(
    Path("/Users/admin/Desktop/Monica/PhD/catalog/OGSCatalog/.all"),
    start=start,
    end=end,
    name="OGS Catalog",
    verbose=True,
    output=Path("/Users/admin/Desktop/Monica/PhD/comparison/OGSCatalog/OGSBackup")
  )
  TargetCatalog: OGSCatalog = OGSCatalog(
    Path("/Users/admin/Desktop/Monica/PhD/catalog/OGSBackup/OGSLocalMagnitude"),
    start=start,
    end=end,
    name="SeisBench Catalog",
    verbose=True,
    output=Path("/Users/admin/Desktop/Monica/PhD/comparison/OGSCatalog/OGSBackup")
  )
  BaseCatalog.plot(others=[TargetCatalog])
  BaseCatalog.bpgma(
    TargetCatalog,
    stations=stations,
  )


if __name__ == "__main__": main()
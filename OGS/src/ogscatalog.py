from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta as td
from typing import Dict, Optional
import logging

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from matplotlib.path import Path as mplPath

import ogsconstants as OGS_C


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
  def _setup_logger(self, verbose: bool) -> logging.Logger:
    """Create and configure the instance logger.

    Parameters
    ----------
    verbose : bool
      If True, set log level to DEBUG; otherwise INFO.

    Returns
    -------
    logging.Logger
      Configured logger instance.
    """
    logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    if not logger.handlers:
      handler = logging.StreamHandler()
      formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
      )
      handler.setFormatter(formatter)
      logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    return logger

  def __init__(self,
               input: Path,
               start: datetime = datetime.max,
               end: datetime = datetime.min,
               verbose: bool = False,
               polygon : mplPath = mplPath(OGS_C.OGS_POLY_REGION, closed=True),
               output : Path = OGS_C.THIS_FILE.parent / "data" / "OGSCatalog",
               name: str = OGS_C.EMPTY_STR) -> None:
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
    assert input.exists(), \
           f"Input path {input} does not exist."
    self.name = output.name if name == OGS_C.EMPTY_STR else name
    self.input = input
    self.start = start
    self.end = end
    self.polygon : mplPath = polygon
    self.logger = self._setup_logger(verbose)
    self.output = output
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
      OGS_C.ERH_STR, OGS_C.GROUPS_STR, OGS_C.NO_STR,
      OGS_C.NUMBER_P_PICKS_STR, OGS_C.NUMBER_S_PICKS_STR,
      OGS_C.NUMBER_P_AND_S_PICKS_STR, OGS_C.ML_STR, OGS_C.ML_MEDIAN_STR,
      OGS_C.ML_UNC_STR, OGS_C.ML_STATIONS_STR, OGS_C.GROUPS_STR
    ])
    self.preload

  @property
  def preload(self) -> None:
    """Preload daily file paths into internal caches."""
    for filepath in self.input.glob("events/*"):
      if filepath.is_file():
        date = UTCDateTime(filepath.stem).date
        if self.start.date() <= date <= self.end.date():
          self.logger.debug(f"Found event file for date {date}: {filepath}")
          self.events_[date] = filepath
    for filepath in self.input.glob("assignments/*"):
      if filepath.is_file():
        date = UTCDateTime(filepath.stem).date
        if self.start.date() <= date <= self.end.date():
          self.logger.debug(f"Found pick file for date {date}: {filepath}")
          self.picks_[date] = filepath
    for filepath in self.input.glob("picks/*"):
      if filepath.is_file():
        date = UTCDateTime(filepath.stem).date
        if self.start.date() <= date <= self.end.date():
          self.logger.debug(f"Found pick file for date {date}: {filepath}")
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
    if filepath.suffix == OGS_C.CSV_EXT: return pd.read_csv(filepath)
    else:
      try: return pd.read_parquet(filepath)
      except Exception as e:
        self.logger.exception(f"Error loading {filepath}")
        return pd.DataFrame(columns=[])

  def load(self, key: str) -> Dict[datetime, pd.DataFrame]:
    """Load daily data for the provided key.

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
      for date in self.events_.keys():
        self.events[date] = self.load_(self.events_[date])
      return self.events
    elif key == "picks":
      for date in self.picks_.keys():
        self.picks[date] = self.load_(self.picks_[date])
      return self.picks
    else:
      raise ValueError(f"Unknown key: {key}")

  def postload(self, key: str) -> Dict[datetime, pd.DataFrame]:
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
      if not self.EVENTS.empty:
        # TODO: GROUPS_STR creation should be done during load
        self.EVENTS[OGS_C.GROUPS_STR] = \
          self.EVENTS[OGS_C.TIME_STR].dt.date # type: ignore
        for date, df in self.EVENTS.groupby(OGS_C.GROUPS_STR):
          self.events[UTCDateTime(date).date] = df
      return self.events
    elif key == "picks":
      if not self.PICKS.empty:
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
        self.logger.info("Loading EVENTS data...")
        events = self.load(key.lower())
        if events:
          self.EVENTS = pd.concat(events.values()).reset_index(drop=True)
      return self.EVENTS
    if key == "PICKS":
      if self.PICKS.empty:
        self.logger.info("Loading PICKS data...")
        picks = self.load(key.lower())
        if picks:
          self.PICKS = pd.concat(picks.values()).reset_index(drop=True)
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
    )
    for other, color in zip(others, OGS_C.PLOT_COLORS.pop(0)):
      other_events = other.get("EVENTS")
      eventsMap.add_plot(
        x=other_events[OGS_C.LONGITUDE_STR],
        y=other_events[OGS_C.LATITUDE_STR],
        legend=True,
        marker='o',
        facecolors='none',
        edgecolors=color,
        label=other.name,
      )
      plt.close()

  def plot(self,
           others: list[OGSCatalog] = [],
           waveforms: dict[str, dict[str, list[Path]]] = dict(),
           output: Optional[Path] = None
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
    i = 0
    self.waveforms = waveforms
    if not self.get("EVENTS").empty:
      self.plot_events(others, output)
      self.plot_erh_histogram(others=others, output=output)
      self.plot_erz_histogram(others=others, output=output)
      self.plot_ert_histogram(others=others, output=output)
      self.plot_magnitude_histogram(others=others, output=output)
      self.plot_depth_histogram(others=others, output=output)
      self.plot_cumulative_events(others=others, output=output)
      i += 1
    if not self.get("PICKS").empty:
      i += 1
      self.plot_cumulative_picks(others=others, output=output)
    if i == 2:
      if self.waveforms:
        for idx, picks in self.PICKS[self.PICKS[OGS_C.IDX_PICKS_STR].isin(
          self.EventsFN["idx"])].groupby([OGS_C.IDX_PICKS_STR,]): # type: ignore
          event: pd.Series = self.EVENTS[
            self.EVENTS[OGS_C.IDX_EVENTS_STR] == idx
          ].iloc[0]
          event[OGS_C.TIME_STR] = UTCDateTime(event[OGS_C.TIME_STR]).datetime
          waveforms_day: dict[str, list[Path]] = self.waveforms.get(
            event[OGS_C.TIME_STR].date(),
            dict()
          )
          if waveforms_day:
            self.plot_events_fn_waveforms(picks, event, waveforms_day, output=output)

  def plot_events_fn_waveforms(self,
        picks: pd.DataFrame,
        event: pd.Series,
        waveforms: dict[str, list[Path]],
        output: Optional[Path] = None
      ) -> None:
    """Plot waveforms for false negative events.

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
      waveforms=waveforms,
      inventory=self.stations,
      title=f"Missed (MS) Event {event[OGS_C.IDX_EVENTS_STR]} ({event[OGS_C.MAGNITUDE_L_STR] if OGS_C.MAGNITUDE_L_STR in event else OGS_C.EMPTY_STR}) | Proposed (PS) Picks {event[OGS_C.TIME_STR] - td(seconds=1)}",
    )
    myfppicks = self.PicksFP[
      self.PicksFP[OGS_C.TIME_STR].between( # type: ignore
        event[OGS_C.TIME_STR] - td(seconds=1),
        event[OGS_C.TIME_STR] + td(seconds=30)
      )
    ]
    myfnplot.add_plot(picks=myfppicks, flip=True,
      output=output if output is not None else f"{self.input.name}_Event{event[OGS_C.IDX_EVENTS_STR]}_MSPS.png"
    )
    plt.close()

  def plot_cumulative_picks(self, others: list[OGSCatalog] = [], output: Optional[Path] = None):
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
    if picks.empty or OGS_C.TIME_STR not in picks.columns:
      self.logger.info("No Date data available for histogram.")
      return
    if (others and isinstance(others[0], OGSCatalog) and
        OGS_C.GROUPS_STR in others[0].get("PICKS").columns):
      hist = OGS_P.day_plotter(
        picks=others[0].get("PICKS")[OGS_C.GROUPS_STR],
      )
      hist.add_plot(
        picks=picks[OGS_C.GROUPS_STR],
        title=f"Cumulative Picks",
        output=output if output is not None else f"{self.input.name}_{others[0].input.name}_CumulativePicks.png"
      )
    else:
      hist = OGS_P.day_plotter(
        picks=picks[OGS_C.GROUPS_STR],
        title=f"Cumulative Picks",
        output=f"{self.input.name}_CumulativePicks.png"
      )
    plt.close()

  def plot_cumulative_events(self, others: list[OGSCatalog] = [], output: Optional[Path] = None):
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
    if events.empty or OGS_C.TIME_STR not in events.columns:
      self.logger.info("No Date data available for histogram.")
      return
    hist = OGS_P.day_plotter(
      picks=events.sort_values(OGS_C.GROUPS_STR)[OGS_C.GROUPS_STR],
      title=f"Cumulative Events",
      output=f"{self.input.name}_CumulativeEvents.png"
    )
    for other in others:
      if not isinstance(other, OGSCatalog):
        raise ValueError("Can only perform cumulative events with OGSCatalog")
      hist.add_plot(
        picks=other.get("EVENTS").sort_values(OGS_C.GROUPS_STR)[OGS_C.GROUPS_STR],
        title=f"Cumulative Events",
        output=output if output is not None else f"{self.input.name}_{other.input.name}_CumulativeEvents.png"
      )
    plt.close()

  def plot_erz_histogram(self,
                         others: list[OGSCatalog] = [],
                         bins: int = OGS_C.NUM_BINS,
                         output: Optional[Path] = None):
    """Plot ERZ histogram for this catalog and optional comparisons.

    Parameters
    ----------
    others : list[OGSCatalog], optional
      Additional catalogs for comparison.
    bins : int, optional
      Number of histogram bins.
    output : Optional[Path], optional
      Output path for the plot.
    """
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    events = self.get("EVENTS")
    if events.empty or OGS_C.ERZ_STR not in events.columns:
      self.logger.info("No ERZ data available for histogram.")
      return
    hist = OGS_P.histogram_plotter(
      data=events[OGS_C.ERZ_STR].dropna(),
      bins=bins,
      xlabel="ERZ (km)",
      ylabel="Number of Events",
      title=f"ERZ Histogram",
      output=f"{self.input.name}_ERZ.png"
    )
    for other, color in zip(others, OGS_C.PLOT_COLORS.pop(0)):
      if not isinstance(other, OGSCatalog):
        raise ValueError("Can only perform ERZ histogram with OGSCatalog")
      hist.add_plot(
        data=other.get("EVENTS")[OGS_C.ERZ_STR].dropna(),
        xlabel="ERZ (km)",
        ylabel="Number of Events",
        title=f"ERZ Histogram",
        legend=True,
        alpha=1,
        label=other.name,
        color=color,
        facecolor=color,
        output=f"{self.input.name}_{other.input.name}_ERZ.png"
      )
    plt.close()

  def plot_erh_histogram(self,
        others: list[OGSCatalog] = [],
        bins: int = OGS_C.NUM_BINS,
        output: Optional[Path] = None
      ) -> None:
    """Plot ERH histogram for this catalog and optional comparisons.

    Parameters
    ----------
    others : list[OGSCatalog], optional
      Additional catalogs for comparison.
    bins : int, optional
      Number of histogram bins.
    output : Optional[Path], optional
      Output path for the plot.
    """
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    events = self.get("EVENTS")
    if events.empty or OGS_C.ERH_STR not in events.columns:
      self.logger.info("No ERH data available for histogram.")
      print(events)
      return
    hist = OGS_P.histogram_plotter(
      data=events[OGS_C.ERH_STR].dropna(),
      bins=bins,
      xlabel="ERH (km)",
      ylabel="Number of Events",
      title=f"ERH Histogram",
      output=f"{self.input.name}_ERH.png",
      color=OGS_C.OGS_BLUE,
    )
    for other, color in zip(others, OGS_C.PLOT_COLORS.pop(0)):
      if not isinstance(other, OGSCatalog):
        raise ValueError("Can only perform ERH histogram with OGSCatalog")
      hist.add_plot(
        data=other.get("EVENTS")[OGS_C.ERH_STR].dropna(),
        xlabel="ERH (km)",
        ylabel="Number of Events",
        title=f"ERH Histogram",
        legend=True,
        alpha=1,
        label=other.name,
        color=color,
        facecolor=color,
        output=f"{self.input.name}_{other.input.name}_ERH.png"
      )
    plt.close()

  def plot_ert_histogram(self,
        others: list[OGSCatalog] = [],
        bins: int = OGS_C.NUM_BINS,
        output: Optional[Path] = None
      ) -> None:
    """Plot ERT histogram for this catalog and optional comparisons.

    Parameters
    ----------
    others : list[OGSCatalog], optional
      Additional catalogs for comparison.
    bins : int, optional
      Number of histogram bins.
    output : Optional[Path], optional
      Output path for the plot.
    """
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    events = self.get("EVENTS")
    if events.empty or OGS_C.ERT_STR not in events.columns:
      self.logger.info("No ERT data available for histogram.")
      return
    hist = OGS_P.histogram_plotter(
      data=events[OGS_C.ERT_STR].dropna(),
      bins=bins,
      xlabel="ERT (s)",
      ylabel="Number of Events",
      title=f"ERT Histogram",
      output=f"{self.input.name}_ERT.png"
    )
    for other, color in zip(others, OGS_C.PLOT_COLORS.pop(0)):
      if not isinstance(other, OGSCatalog):
        raise ValueError("Can only perform ERT histogram with OGSCatalog")
      hist.add_plot(
        data=other.get("EVENTS")[OGS_C.ERT_STR].dropna(),
        xlabel="ERT (s)",
        ylabel="Number of Events",
        title=f"ERT Histogram",
        legend=True,
        alpha=1,
        label=other.name,
        color=color,
        facecolor=color,
        output=f"{self.input.name}_{other.input.name}_ERT.png"
      )
    plt.close()

  def plot_depth_histogram(self,
        others: list[OGSCatalog] = [],
        bins: int = OGS_C.NUM_BINS,
        output: Optional[Path] = None
      ) -> None:
    """Plot depth histogram for this catalog and optional comparisons.

    Parameters
    ----------
    others : list[OGSCatalog], optional
      Additional catalogs for comparison.
    bins : int, optional
      Number of histogram bins.
    output : Optional[Path], optional
      Output path for the plot.
    """
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    events = self.get("EVENTS")
    if events.empty or OGS_C.DEPTH_STR not in events.columns:
      self.logger.info("No depth data available for histogram.")
      return
    hist = OGS_P.histogram_plotter(
      data=events[OGS_C.DEPTH_STR].dropna(),
      bins=bins,
      xlabel="Depth (km)",
      ylabel="Number of Events",
      title=f"Depth Histogram",
      output=f"{self.input.name}_Depth.png"
    )
    for other, color in zip(others, OGS_C.PLOT_COLORS.pop(0)):
      if not isinstance(other, OGSCatalog):
        raise ValueError("Can only perform Depth histogram with OGSCatalog")
      hist.add_plot(
        data=other.get("EVENTS")[OGS_C.DEPTH_STR].dropna(),
        xlabel="Depth (km)",
        ylabel="Number of Events",
        title=f"Depth Histogram",
        legend=True,
        alpha=1,
        label=other.name,
        color=color,
        facecolor=color,
        output=f"{self.input.name}_{other.input.name}_Depth.png"
      )
    plt.close()

  def plot_magnitude_histogram(self,
        others: list[OGSCatalog] = [],
        bins: int = OGS_C.NUM_BINS,
        output: Optional[Path] = None
      ) -> None:
    """Plot magnitude histogram for this catalog and optional comparisons.

    Parameters
    ----------
    others : list[OGSCatalog], optional
      Additional catalogs for comparison.
    bins : int, optional
      Number of histogram bins.
    output : Optional[Path], optional
      Output path for the plot.
    """
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    events = self.get("EVENTS")
    if events.empty or OGS_C.MAGNITUDE_L_STR not in events.columns:
      self.logger.info("No Magnitude data available for histogram.")
      return
    events[OGS_C.MAGNITUDE_L_STR] = pd.to_numeric(
      events[OGS_C.MAGNITUDE_L_STR], errors='coerce'
    )
    hist = OGS_P.histogram_plotter(
      data=events[OGS_C.MAGNITUDE_L_STR].dropna(),
      bins=bins,
      xlabel="Magnitude ($M_L$)",
      ylabel="Number of Events",
      title=f"Magnitude Histogram",
      output=f"{self.input.name}_Magnitude.png"
    )
    for other, color in zip(others, OGS_C.PLOT_COLORS.pop(0)):
      if not isinstance(other, OGSCatalog):
        raise ValueError("Can only perform Magnitude histogram with OGSCatalog")
      events = self.get("EVENTS")
      if events.empty or OGS_C.MAGNITUDE_L_STR not in events.columns:
        self.logger.info("No Magnitude data available for histogram.")
        return
      hist.add_plot(
        data=events[OGS_C.MAGNITUDE_L_STR].dropna(),
        xlabel="Magnitude ($M_L$)",
        ylabel="Number of Events",
        title=f"Magnitude Histogram",
        legend=True,
        alpha=1,
        label=other.name,
        color=color,
        facecolor=color,
        output=f"{self.input.name}_{other.input.name}_Magnitude.png"
      )
    plt.close()

  def bgmaEvents(self, other: "OGSCatalog") -> None:
    """Match events between catalogs using BGMA.

    Parameters
    ----------
    other : OGSCatalog
      Catalog to compare against.
    """
    import ogsplotter as OGS_P
    from matplotlib import pyplot as plt
    if not isinstance(other, OGSCatalog):
      raise ValueError("Can only perform bgmaEvents on OGSCatalog")
    EVENTS_CFN_MTX = pd.DataFrame(0, index=[OGS_C.EVENT_STR, OGS_C.NONE_STR],
                    columns=[OGS_C.EVENT_STR, OGS_C.NONE_STR], dtype=int)
    self.EventsTP = list()
    self.EventsFN = list()
    self.EventsFP = list()
    columns = [OGS_C.INDEX_STR, OGS_C.TIME_STR, OGS_C.LATITUDE_STR,
           OGS_C.LONGITUDE_STR, OGS_C.DEPTH_STR, OGS_C.ERH_STR,
           OGS_C.ERZ_STR, OGS_C.GAP_STR, OGS_C.MAGNITUDE_L_STR]
    for date, _ in self.events_.items():
      BASE = self.get_(date, "events").reset_index(drop=True)
      TARGET = other.get_(date, "events").reset_index(drop=True)
      I = len(BASE)
      bpgEvents = OGSBPGraphEvents(BASE, TARGET)
      baseIDs = set(range(I))
      targetIDs = set(range(len(TARGET)))
      for i, j in bpgEvents.E:
        a, b = sorted((i, j))
        b -= I
        EVENTS_CFN_MTX.at[OGS_C.EVENT_STR, OGS_C.EVENT_STR] += 1 # type: ignore
        baseIDs.remove(a)
        targetIDs.remove(b)

        self.EventsTP.append([
          (BASE.at[a, col] if col in BASE.columns else None,
           TARGET.at[b, col] if col in TARGET.columns else None)
           for col in columns
        ])
      for i in baseIDs:
        EVENTS_CFN_MTX.at[OGS_C.EVENT_STR, OGS_C.NONE_STR] += 1 # type: ignore
        self.EventsFN.append([
          BASE.at[i, col] if col in BASE.columns else None for col in columns
        ])
      if not TARGET.empty:
        fp_target = TARGET[TARGET[[OGS_C.LONGITUDE_STR, OGS_C.LATITUDE_STR]].apply(
          lambda x: self.polygon.contains_point(
            (x[OGS_C.LONGITUDE_STR], x[OGS_C.LATITUDE_STR])), axis=1)]
        for j in targetIDs:
          if j not in fp_target.index: continue
          EVENTS_CFN_MTX.at[OGS_C.NONE_STR, OGS_C.EVENT_STR] += 1 # type: ignore
          self.EventsFP.append([
            fp_target.at[j, col] if col in fp_target.columns else None
            for col in columns
          ])
    recall = \
      EVENTS_CFN_MTX.at[OGS_C.EVENT_STR, OGS_C.EVENT_STR] / ( # type: ignore
        EVENTS_CFN_MTX.at[OGS_C.EVENT_STR, OGS_C.EVENT_STR] +
        EVENTS_CFN_MTX.at[OGS_C.EVENT_STR, OGS_C.NONE_STR]
      )
    self.logger.info("Recall: %s", recall)
    self.logger.info("MH: %s PS: %s MS: %s",
                EVENTS_CFN_MTX.at[OGS_C.EVENT_STR, OGS_C.EVENT_STR],
                EVENTS_CFN_MTX.at[OGS_C.NONE_STR, OGS_C.EVENT_STR],
                EVENTS_CFN_MTX.at[OGS_C.EVENT_STR, OGS_C.NONE_STR])
    self.logger.info("%s", EVENTS_CFN_MTX)
    self.EventsTP = pd.DataFrame(self.EventsTP, columns=columns)
    self.EventsFN = pd.DataFrame(self.EventsFN, columns=columns).sort_values(
      by=OGS_C.TIME_STR)
    self.EventsFP = pd.DataFrame(self.EventsFP, columns=columns).sort_values(
      by=OGS_C.TIME_STR)
    filepath = f"{self.input.name}_{other.input.name}_EventsMH.csv"
    self.EventsTP.to_csv(filepath, index=False)
    self.logger.info("%s written.", filepath)
    filepath = f"{self.input.name}_{other.input.name}_EventsMS.csv"
    self.EventsFN.to_csv(filepath, index=False)
    self.logger.info("%s written.", filepath)
    filepath = f"{self.input.name}_{other.input.name}_EventsPS.csv"
    self.EventsFP.to_csv(filepath, index=False)
    self.logger.info("%s written.", filepath)
    filepath = f"{self.input.name}_" + \
               f"{other.input.name}_EventsConfMtx.png"
    OGS_P.ConfMtx_plotter(
      EVENTS_CFN_MTX.values,
      title="Recall: {:.4f}".format(recall),
      label=EVENTS_CFN_MTX.columns.tolist(),
      output=filepath
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
      output=f"{self.input.name}_{other.input.name}_EventsTimeDiff.png",
      legend=True)
    plt.close()
    # Missed (MS) Map
    myplot = OGS_P.map_plotter(
      domain=OGS_C.OGS_STUDY_REGION,
      x=self.EventsTP[OGS_C.LONGITUDE_STR].apply(lambda x: x[0]),
      y=self.EventsTP[OGS_C.LATITUDE_STR].apply(lambda x: x[0]),
      facecolors="none", edgecolors=OGS_C.OGS_BLUE, legend=True,
      label=self.name)
    myplot.add_plot(
      self.EventsTP[OGS_C.LONGITUDE_STR].apply(lambda x: x[1]),
      self.EventsTP[OGS_C.LATITUDE_STR].apply(lambda x: x[1]), color=None,
        label="Missed (MS) [SBC]", legend=True, facecolors="none",
        edgecolors=OGS_C.MEX_PINK,
        output=f"{self.input.name}_{other.input.name}_MS.png")
    plt.close()
    # Missed (MS) and Proposed (PS) Map
    myplot = OGS_P.map_plotter(
      domain=OGS_C.OGS_STUDY_REGION,
      x=self.EventsFN[OGS_C.LONGITUDE_STR], y=self.EventsFN[OGS_C.LATITUDE_STR],
      label=f"Missed (MS) [OGS] {len(self.EventsFN.index)}", legend=True,)
    myplot.add_plot(
      self.EventsFP[OGS_C.LONGITUDE_STR], self.EventsFP[OGS_C.LATITUDE_STR], color=None,
        label=f"Proposed (PS) [SBC] {len(self.EventsFP.index)}", legend=True,
        facecolors="none", edgecolors=OGS_C.MEX_PINK,
        output=f"{self.input.name}_{other.input.name}_False.png")
    plt.close()
    # Depth Difference Histogram
    OGS_P.histogram_plotter(
      self.EventsTP[OGS_C.DEPTH_STR].apply(lambda x: x[1] - x[0]),
      xlabel="Depth Difference (km) [OGS - SBC]",
      title="Event Depth Difference",
      xlim=(-OGS_C.EVENT_DIST_OFFSET, OGS_C.EVENT_DIST_OFFSET),
      output=f"{self.input.name}_{other.input.name}_DepthDiff.png",
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
      xlabel="Epicentral Distance Difference (km)",
      title="Event Epicentral Distance Difference",
      output=f"{self.input.name}_{other.input.name}_EpiDistDiff.png",
      legend=True)
    plt.close()
    if OGS_C.MAGNITUDE_L_STR in self.EventsTP.columns:
      # Magnitude Scatter Plot
      mymags = OGS_P.scatter_plotter(
        self.EventsTP[OGS_C.MAGNITUDE_L_STR].apply(lambda x: x[1]),
        self.EventsTP[OGS_C.MAGNITUDE_L_STR].apply(lambda x: x[0]),
        xlabel="SBC Magnitude ($M_L$)",
        ylabel="OGS Magnitude ($M_L$)",
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
        f"{self.input.name}_{other.input.name}_MagDist.png")
      plt.close()
      if self.input.name in (OGS_C.TXT_EXT, ".all") and \
         other.input.name in ("OGSLocalMagnitude"):
        # Magnitude Difference Histogram
        data = self.EventsTP[OGS_C.MAGNITUDE_L_STR].apply(lambda x: x[1] - x[0])
        OGS_P.histogram_plotter(
          data,
          xlabel="Magnitude Difference ($M_L$) [OGS - SBC]",
        title=f"RMSE = {np.sqrt(np.mean(data ** 2)):.4f}, " +
              f"MAE = {data.abs().mean():.4f}",
        xlim=(-1, 1),
        bins=21,
        output=f"{self.input.name}_{other.input.name}_MagDiff.png",
        legend=True)
        plt.close()
        # Event Magnitude Histogram
        mymags = OGS_P.histogram_plotter(
          self.EventsFP[OGS_C.MAGNITUDE_L_STR].dropna(),
          xlabel="Magnitude ($M_L$)",
          title="Event Magnitude",
          color=OGS_C.MEX_PINK,
          yscale='log',
          label="Proposed (PS) [SBC]",
        )
        mymags.add_plot(
          self.EventsFN[OGS_C.MAGNITUDE_L_STR],
          label="Missed (MS) [OGS]",
          color=OGS_C.OGS_BLUE,
          legend=True,
          output=f"{self.input.name}_{other.input.name}_MSPSMagDist.png",
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
            label="Proposed (PS) [SBC]",
            output=f"{other.input.name}_PSMagDist.png",
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
    PICKS_CFN_MTX = pd.DataFrame(0, index=[OGS_C.PWAVE, OGS_C.SWAVE, OGS_C.NONE_STR],
                   columns=[OGS_C.PWAVE, OGS_C.SWAVE, OGS_C.NONE_STR], dtype=int)
    INVENTORY = [key.split(".")[1] for key in self.stations.keys()]
    self.PicksTP = list()
    self.PicksFN = list()
    self.PicksFP = list()
    columns = [OGS_C.IDX_PICKS_STR, OGS_C.TIME_STR, OGS_C.PHASE_STR,
           OGS_C.STATION_STR, OGS_C.PROBABILITY_STR]
    for date, _ in self.picks_.items():
      BASE = self.get_(date, "picks")
      BASE[OGS_C.NETWORK_STR] = BASE[OGS_C.STATION_STR].str.split(".").str[0]
      BASE[OGS_C.STATION_STR] = BASE[OGS_C.STATION_STR].str.split(".").str[1]
      BASE = BASE[BASE[OGS_C.STATION_STR].isin(INVENTORY)].reset_index(drop=True)
      TARGET = other.get_(date, "picks").reset_index(drop=True)
      I = len(BASE)
      bpgPicks = OGSBPGraphPicks(BASE, TARGET)
      baseIDs = set(range(I))
      targetIDs = set(range(len(TARGET)))
      for i, j in bpgPicks.E:
        a, b = sorted((i, j))
        b -= I
        PICKS_CFN_MTX.at[BASE.at[a, OGS_C.PHASE_STR],
                         TARGET.at[b, OGS_C.PHASE_STR]] += 1 # type: ignore
        if BASE.at[a, OGS_C.PHASE_STR] == TARGET.at[b, OGS_C.PHASE_STR]:
          self.PicksTP.append([
            (BASE.at[a, OGS_C.IDX_PICKS_STR], TARGET.at[b, OGS_C.IDX_PICKS_STR]),
            (str(BASE.at[a, OGS_C.TIME_STR]), str(TARGET.at[b, OGS_C.TIME_STR])),
            (BASE.at[a, OGS_C.PHASE_STR]),
            (TARGET.at[b, OGS_C.STATION_STR]),
            (BASE.at[a, OGS_C.PROBABILITY_STR], TARGET.at[b, OGS_C.PROBABILITY_STR])
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
    p_recall = PICKS_CFN_MTX.at[OGS_C.PWAVE, OGS_C.PWAVE] / ( # type: ignore
      PICKS_CFN_MTX.at[OGS_C.PWAVE, OGS_C.PWAVE] + # type: ignore
      PICKS_CFN_MTX.at[OGS_C.PWAVE, OGS_C.SWAVE] +
      PICKS_CFN_MTX.at[OGS_C.PWAVE, OGS_C.NONE_STR]
    )
    self.logger.info("Recall P-wave: %s", p_recall)
    s_recall = PICKS_CFN_MTX.at[OGS_C.SWAVE, OGS_C.SWAVE] / ( # type: ignore
      PICKS_CFN_MTX.at[OGS_C.SWAVE, OGS_C.SWAVE] + # type: ignore
      PICKS_CFN_MTX.at[OGS_C.SWAVE, OGS_C.PWAVE] +
      PICKS_CFN_MTX.at[OGS_C.SWAVE, OGS_C.NONE_STR]
    )
    self.logger.info("Recall S-wave: %s", s_recall)
    self.logger.info("%s", PICKS_CFN_MTX)
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
    filepath = f"{self.input.name}_{other.input.name}_PicksMH.csv"
    self.PicksTP.to_csv(filepath, index=False)
    self.logger.info("%s written.", filepath)
    filepath = f"{self.input.name}_{other.input.name}_PicksMS.csv"
    self.PicksFN.to_csv(filepath, index=False)
    self.logger.info("%s written.", filepath)
    filepath = f"{self.input.name}_{other.input.name}_PicksPS.csv"
    self.PicksFP.to_csv(filepath, index=False)
    self.logger.info("%s written.", filepath)
    filepath = f"{self.input.name}_" + \
               f"{other.input.name}_PicksConfMtx.png"
    OGS_P.ConfMtx_plotter(
      PICKS_CFN_MTX.values,
      title="Recall: {:.4f}, Recall P: {:.4f}, Recall S: {:.4f}".format(
        recall, p_recall, s_recall
      ),
      label=PICKS_CFN_MTX.columns.tolist(),
      output=filepath
    )
    plt.close()
    # Time Difference Histogram
    data = self.PicksTP[OGS_C.TIME_STR].apply(
      lambda x: UTCDateTime(x[1]) - UTCDateTime(x[0]) # type: ignore
    )
    pickdiff = OGS_P.histogram_plotter(
      data,
      xlabel="Time Difference (s)",
      title=f"RMSE = {np.sqrt((data**2).mean()):.4f} s, MAE = {data.abs().mean():.4f} s",
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
      label=f"P Picks: $\mu$ = {data.mean():.3E}, $\sigma$ = {data.std():.3E}",
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
      label=f"S Picks: $\mu$ = {data.mean():.3E}, $\sigma$ = {data.std():.3E}",
      legend=True,
      output=f"{self.input.name}_{other.input.name}_PicksTimeDiff.png",
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
      output=f"{other.input.name}_PicksConfDist.png",
    )
    plt.close()
    self.PICKS = self.get("PICKS")
    other.PICKS = other.get("PICKS")

  def bpgma(self,
        other: "OGSCatalog",
        stations: dict[str, tuple[float, float, float, str, str, str, str]]
      ) -> None:
    """Run BGMA comparisons for events and picks.

    Parameters
    ----------
    other : OGSCatalog
      Catalog to compare against.
    stations : dict[str, tuple[float, float, float, str, str, str, str]]
      Station metadata mapping.
    """
    if not isinstance(other, OGSCatalog):
      raise ValueError("Can only perform bpgma on OGSCatalog")
    self.stations = stations
    if self.events_ != {}:
      if other.events_ == {} and other.get("EVENTS").empty:
        self.logger.info("%s catalog has no events to compare.", other.name)
      else:
        self.bgmaEvents(other)
    if self.picks_ != {}:
      if other.picks_ == {} and other.get("PICKS").empty:
        self.logger.info("%s catalog has no picks to compare.", other.name)
      else:
        self.bgmaPicks(other)

  def plotFNWaveforms(self, other: "OGSCatalog", output: Path) -> None:
    """Plot waveforms for false negative events.

    Parameters
    ----------
    other : OGSCatalog
      Catalog used for comparison.
    output : Path
      Output path for the plot(s).
    """
    pass

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
  start = datetime(2024, 1, 1)
  end = datetime(2024, 12, 31)
  BaseCatalog = OGSCatalog(
    Path("/Users/admin/Desktop/Monica/PhD/catalog/OGSCatalog/.all"),
    start=start,
    end=end,
    name="BaseCatalog",
    verbose=True,
    output=Path("/Volumes/Expansion/KEN/catalog")
  )
  BaseCatalog.plot()

if __name__ == "__main__": main()
import numpy as np
import logging
import obspy as op
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
from pathlib import Path
from obspy import UTCDateTime
from matplotlib.axes import Axes
from obspy.geodetics import gps2dist_azimuth
from datetime import datetime, timedelta as td
from sklearn.metrics import ConfusionMatrixDisplay as ConfMtxDisp
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_map_utils.core.north_arrow import north_arrow

import ogsconstants as OGS_C

def v_lat_long_to_distance(lng1, lat1, depth1, lng2, lat2, depth2, dim=2):
  return [np.sqrt((gps2dist_azimuth(lt1, lg1, lt2, lg2)[0] / 1000.0) ** 2 +
            ((abs(dp2 - dp1) / 1000.0) ** 2 if dim == 3 else 0.))
          for lg1, lt1, dp1, lg2, lt2, dp2 in zip(
            lng1, lat1, depth1, lng2, lat2, depth2)]

class plotter:
  def _setup_logger(self, verbose: bool = False) -> logging.Logger:
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

  def __init__(self, figsize=(20, 10), fig=None, verbose: bool = False,
               **kwargs) -> None:
    plt.rcParams.update({'font.size': 12})
    self.figsize = figsize
    self.fig = fig if fig else plt.figure(figsize=figsize,
                                          layout="compressed")
    self.logger = self._setup_logger(verbose)

  def savefig(self, output=None, **kwargs) -> None:
    if output is not None:
      self.output = output
    if self.output is not None:
      plt.savefig(Path(self.output), bbox_inches='tight', dpi=300, **kwargs)
      self.logger.info("Figure saved to %s", self.output)

class stack_plotter(plotter):
  def __init__(self, x, y, labels, colors, xlabel=None, ylabel=None,
               title=None, fig=None, ax=None, gs=111, legend=False,
               output=None, xlim=None, ylim=None, verbose: bool = False,
               vlines: list[tuple[datetime, str, str]] = [],
               hlines: list[tuple[float, str, str]] = []) -> None:
    super().__init__(fig=fig, verbose=verbose)
    self.ax: Axes = self.fig.add_subplot(gs)
    if xlabel: self.ax.set_xlabel(xlabel)
    if ylabel: self.ax.set_ylabel(ylabel)
    if title: self.ax.set_title(title)
    self.ax.stackplot(x, y, labels=labels, colors=colors)
    self.ax.set_xlim(x[0], x[-1])
    self.ax.set_ylim(0, OGS_C.decimeter(self.ax.get_ylim()[1]))
    for vline in vlines:
      self.ax.axvline(x=vline[0], label=vline[1] or None, # type: ignore
                      color=vline[2] or OGS_C.ALN_GREEN, linestyle='--')
    if legend: self.ax.legend()
    self.ax.grid()
    if output is not None: self.savefig(output=output)

class line_plotter(plotter):
  def __init__(
      self, x, y, xlabel=None, ylabel=None, title=None, fig=None, ax=None,
      color=OGS_C.OGS_BLUE, gs=111, label=None, legend=False, ylim=(-100, 100),
      output=None, xlim=None, verbose: bool = False,
      vlines: list[tuple[datetime, str, str]] = [],
      hlines: list[tuple[float, str, str]] = []) -> None:
    super().__init__(fig=fig, verbose=verbose)
    self.ax: Axes = self.fig.add_subplot(gs)
    if xlabel: self.ax.set_xlabel(xlabel)
    if ylabel: self.ax.set_ylabel(ylabel)
    if title: self.ax.set_title(title)
    if xlim is not None: self.ax.set(xlim=xlim)
    if ylim is not None: self.ax.set(ylim=ylim)
    self.ax.plot(x, y, color=color, label=label)
    if legend: self.ax.legend()
    if ylim is not None: self.ax.set(ylim=ylim)
    for date, label, color in vlines:
      self.ax.axvline(x=date, label=label or None, # type: ignore
                      color=color or OGS_C.ALN_GREEN, linestyle='--')
    for y, label, color in hlines:
      self.ax.axhline(y=y, label=label or None,
                      color=color or OGS_C.ALN_GREEN, linestyle='--')
    if output is not None: self.savefig(output=output)

  def add_plot(self, x, y, xlabel=None, ylabel=None, color=OGS_C.MEX_PINK,
               label=None, legend=None, output=None, savefig=False) -> None:
    self.ax.plot(x, y, color=color, label=label)
    if xlabel: self.ax.set_xlabel(xlabel)
    if ylabel: self.ax.set_ylabel(ylabel)
    if legend is not None: self.ax.legend()
    if output is not None: savefig = True
    if savefig: self.savefig(output=output)

class event_plotter(plotter):
  def __init__(self,
        picks: pd.DataFrame,
        event: pd.Series,
        stations: list[str],
        waveforms: dict[str, list[Path]],
        inventory: pd.DataFrame,
        fig=None, ax=None, gs=None, title=None, color=None, xlabel="Time (s)",
        output=None, ylabel="Epicentral Distance (km)", ylim=150, center=None,
        verbose: bool = False
      ) -> None:
    super().__init__(fig=fig, verbose=verbose)
    self.ax: Axes = self.fig.add_subplot(111)
    self.pick_time = op.UTCDateTime(event[OGS_C.TIME_STR])
    self.ylim = ylim
    self.event = event
    self.waveforms = waveforms
    self.window = td(seconds=30)
    self.inventory = inventory.set_index(OGS_C.INDEX_STR)
    self.sta2sta = dict(zip(
      inventory[OGS_C.STATION_STR], inventory[OGS_C.INDEX_STR]
    ))
    self.t = event[OGS_C.TIME_STR]
    self.offset = td(seconds=1)
    self.antis2c = {}
    for sta, df in picks.groupby(OGS_C.STATION_STR):
      if sta not in stations: continue
      if self.sta2sta[sta] not in self.waveforms: continue # type: ignore
      self.stream: op.Stream = op.read(
        self.waveforms[self.sta2sta[sta]][0], # type: ignore
        starttime=self.pick_time - self.offset,
        endtime=self.pick_time + self.window
      )
      self.stream.detrend()
      self.stream.filter("highpass", freq=2)
      trace = self.stream.select(
        station=self.sta2sta[sta].split(OGS_C.PERIOD_STR)[1]) # type: ignore
      if len(trace) == 0:
        self.logger.warning("No trace found for station %s. Skipping.", sta)
        continue
      trace = trace[0]
      station_x, station_y, station_z, _, _, color, _ = self.inventory.loc[
        self.sta2sta[sta] # type: ignore
      ]
      y_ = np.sqrt((gps2dist_azimuth(
        event[OGS_C.LATITUDE_STR], event[OGS_C.LONGITUDE_STR],
        station_y, station_x)[0] / 1000.0) ** 2 +
        (station_z / 1000.0) ** 2)
      if y_ > self.ylim: continue
      trace_data_max = np.max(np.abs(trace.data))
      trace.data = trace.data / trace_data_max * 5.0 + y_
      self.ax.plot(trace.times(), trace.data, color=color)
      for _, p in df.groupby(OGS_C.PHASE_STR):
        if len(p.index) > 1:
          self.logger.warning(
            "Multiple picks for %s at %s. Using the first one.",
            sta, p[OGS_C.TIME_STR].iloc[0]
          )
        p = p.iloc[0]
        x_ = \
          UTCDateTime(p[OGS_C.TIME_STR]) - (UTCDateTime(self.t) - self.offset)
        if p[OGS_C.PHASE_STR] == "P":
            ls = '-'
            lc = "red"
        else:
            ls = '--'
            lc = "blue"
        self.ax.plot(np.array([x_, x_]), [y_ - 3, y_ + 3], ls=ls, color=lc)
      weights = df[OGS_C.WEIGHT_STR].to_list() \
        if OGS_C.WEIGHT_STR in df.columns \
          else df[OGS_C.PROBABILITY_STR].to_list()
      weights = [float(f"{w:.3f}") for w in weights]
      self.ax.text(31.3, y_, f"{sta} {weights}", fontsize=8)
    if xlabel: self.ax.set_xlabel(xlabel)
    if ylabel: self.ax.set_ylabel(ylabel)
    self.ax.set(ylim=(0, self.ylim))
    self.ax.set_ylim(0)
    self.ax.set_xlim(0, self.window.seconds + self.offset.seconds,)
    xmin, xmax = self.ax.get_xlim()
    self.ax.hlines(y=0, xmin=xmin, xmax=xmax, color=OGS_C.MEX_PINK,
                   linestyles='--')
    self.ax.hlines(y=[-3, 3], xmin=xmin, xmax=xmax, color=OGS_C.ALN_GREEN,
                   linestyles='--')
    if title: self.ax.set_title(title)
    if output is not None: self.savefig(output=output)

  def add_plot(self, picks, color=None, label=None, output=None, savefig=False,
               alpha=1., flip=False, ylim=(-100, 100)) -> None:
    for sta, df in picks.groupby(OGS_C.STATION_STR):
      if sta not in self.sta2sta: continue
      if self.sta2sta[sta] not in self.waveforms: continue
      self.stream: op.Stream = op.read(
        self.waveforms[self.sta2sta[sta]][0], # type: ignore
        starttime=self.pick_time - self.offset,
        endtime=self.pick_time + self.window
      )
      self.stream.detrend()
      self.stream.filter("highpass", freq=2)
      trace = self.stream.select(station=self.sta2sta[sta].split('.')[1])
      if len(trace) == 0:
        self.logger.warning("No trace found for station %s. Skipping.", sta)
        continue
      trace = trace[0]
      station_x, station_y, station_z, _, _, color, _ = self.inventory.loc[
        self.sta2sta[sta] # type: ignore
      ]
      y_ = np.sqrt((gps2dist_azimuth(
        self.event[OGS_C.LATITUDE_STR], self.event[OGS_C.LONGITUDE_STR],
        station_y, station_x)[0] / 1000.0) ** 2 +
        (station_z / 1000.0) ** 2)
      if y_ > self.ylim:
        self.logger.warning(
          "Station %s is too far from the event. Skipping.", sta
        )
        continue
      if flip:
        if y_ < 0: raise ValueError("y_ must be positive for plotting.")
        y_ = -y_
      trace_data_max = np.max(np.abs(trace.data))
      trace.data = trace.data / trace_data_max * 5.0 + y_
      self.ax.plot(trace.times(), trace.data, color=color, alpha=alpha)
      for _, p in df.groupby(OGS_C.PHASE_STR):
        if len(p.index) > 1:
          self.logger.warning(
            "Multiple picks for %s at %s. Using the first one.",
            sta, p[OGS_C.TIME_STR].iloc[0]
          )
        p = p.iloc[0]
        x_ = UTCDateTime(p[OGS_C.TIME_STR]) - (UTCDateTime(self.t) - self.offset)
        if p[OGS_C.PHASE_STR] == OGS_C.PWAVE:
          ls = '-'
          lc = "red"
        else:
          ls = '--'
          lc = "blue"
        self.ax.plot(np.array([x_, x_]), [y_ - 3, y_ + 3], ls=ls, color=lc)
      if not sta in self.sta2color: self.ax.text(31.3, y_, sta, fontsize=8,)
    if flip:
      self.ax.set_ylim(-self.ylim, self.ylim)
      ticks = self.ax.get_yticks()
      self.ax.set_yticklabels([f"{int(abs(tick))}" for tick in ticks])
    plt.tight_layout(pad=0.1)
    if output is not None: savefig = True
    if savefig: self.savefig(output=output)

class day_plotter(plotter):
  def __init__(self, picks, ylabel=None, title=None, ylim=None, label=None,
               output=None, legend=None, yscale=None, color=OGS_C.OGS_BLUE,
               grid=False, verbose: bool = False,
               hlines:list[tuple[float, str, str]]=[],
               vlines:list[tuple[datetime, str, str]]=[]) -> None:
    from obspy import UTCDateTime
    super().__init__(verbose=verbose)
    x = picks.value_counts().sort_index()
    y = np.cumsum(x.values)
    x = [UTCDateTime(xx).date for xx in x.index]
    self.ax: Axes = self.fig.add_subplot(111)
    if ylabel:
      self.ax.set_ylabel(ylabel)
    if title:
      self.ax.set_title(title)
    if color == None:
      self.ax.plot(x, y, label=label)
    else:
      self.ax.plot(x, y, color=color, label=label)
    for label in self.ax.get_xticklabels():
      label.set(rotation=30, horizontalalignment='right')
    self.ax.set_xlim(x[0], x[-1])
    self.ax.set_ylim(0, OGS_C.decimeter(max(y)))
    if ylim is not None:
      self.ax.set_ylim(ylim)
    if legend is not None:
      if legend == False:
        self.ax.get_legend().remove()
      else:
        self.ax.legend()
    if yscale is not None:
      self.ax.set_yscale(yscale)
    if grid:
      self.ax.grid()
    for date, label, color in vlines:
      self.ax.axvline(x=date, label=label or None, # type: ignore
                      color=color or OGS_C.ALN_GREEN, linestyle='--')
    for y, label, color in hlines:
      self.ax.axhline(y=y, label=label or None,
                      color=color or OGS_C.ALN_GREEN, linestyle='--')
    if output is not None:
      self.savefig(output=output)

  def add_plot(self, picks, ylabel=None, title=None, output=None, label=None,
               color=None, legend=None, savefig=False) -> None:
    x = picks.value_counts().sort_index()
    y = np.cumsum(x.values)
    x = [UTCDateTime(xx).date for xx in x.index]
    if color == None:
      self.ax.plot(x, y, label=label)
    else:
      self.ax.plot(x, y, color=color, label=label)
    if ylabel:
      self.ax.set_ylabel(ylabel)
    if title:
      self.ax.set_title(title)
    if legend is not None:
      if legend == False:
        self.ax.get_legend().remove()
      else:
        self.ax.legend()
    self.ax.set_ylim(0, OGS_C.decimeter(max(y[-1], self.ax.get_ylim()[1])))
    if output is not None: savefig = True
    if savefig: self.savefig(output=output)

class map_plotter(plotter):
  def _add_scale_bar(self, ax, extent):
    """Add a cartographic scale bar using matplotlib-scalebar.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
      The axes to add the scale bar to.
    extent : list
      The extent of the map [min_lon, max_lon, min_lat, max_lat].
    """
    # Calculate dx: distance in meters per degree at the map center latitude
    lat_mid = (extent[2] + extent[3]) / 2
    # Distance in meters for 1 degree of longitude at the center latitude
    dx = 111320 * np.cos(np.radians(lat_mid))
    scale_bar = ScaleBar(
      dx=dx,
      units="m",
      dimension="si-length",
      location="lower left",
      scale_loc="bottom",
      length_fraction=0.2,
      width_fraction=0.015,
      font_properties={"size": 10, "weight": "bold"},
      box_alpha=0.85,
      pad=0.5,
      border_pad=0.5,
      sep=5,
    )
    ax.add_artist(scale_bar)

  def _add_north_arrow(self, ax, proj):
    """Add a north arrow using matplotlib-map-utils.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
      The axes to add the north arrow to.
    proj : cartopy.crs.CRS
      The projection of the map.
    """
    north_arrow(
      ax,
      location="upper left",
      scale=0.6,
      rotation={"crs": proj, "reference": "center"},
      shadow=False,
      label={"text": "N", "position": "top", "fontsize": 11, "fontweight": "bold"},
    )

  def __init__(self, domain : list[float], x=None, y=None, text=None,
               xlabel=None, ylabel=None, title=None, fig=None, ax=None, s=20,
               proj=ccrs.PlateCarree(), color=OGS_C.OGS_BLUE, gs=111, label=None,
               legend=False, marker='o', facecolors=OGS_C.OGS_BLUE,
               edgecolors=OGS_C.OGS_BLUE, output=None, scale: float = 50,
               verbose: bool = False) -> None:
    assert len(domain) == 4, "Domain must be a list of four floats: [min_lon, max_lon, min_lat, max_lat]"
    super().__init__(fig=fig, verbose=verbose)
    self.proj = proj
    self.ax: Axes = self.fig.add_subplot(gs, projection=self.proj)
    self.marker = marker
    self.output = output
    self.s = s
    self.scale = scale
    self.legend = legend
    pm = 0.5
    xy = (domain[0], domain[2])
    w = domain[1] - domain[0]
    h = domain[3] - domain[2]
    extent = [domain[0] - pm, domain[1] + pm, domain[2] - pm, domain[3] + pm]
    self.ax.add_patch(mpatches.Polygon(
      OGS_C.OGS_POLY_REGION, closed=True, linewidth=1, color='red', fill=False,
      label="Bulletin Area"))
    rgAx = self.fig.add_axes((.74, 0.01, 0.15, 0.27), projection=self.proj)
    rgAx.add_patch(mpatches.Rectangle(xy, w, h, linewidth=1, color='blue',
                                      fill=False))
    rgAx.add_feature(cfeature.OCEAN, facecolor=("lightblue")) # type: ignore
    rgAx.add_feature(cfeature.BORDERS, linewidth=0.5, # type: ignore
                     edgecolor=OGS_C.MEX_PINK)
    rgAx.add_feature(cfeature.COASTLINE, linewidth=0.5, # type: ignore
                     edgecolor='black')
    rgAx.set_extent([6, 19, 36, 48], crs=self.proj) # type: ignore
    rgAx.set_aspect('equal', adjustable='box')
    ita = rgAx.annotate("Italy", xy=(0.5, 0.55), xycoords='axes fraction',
                        ha='center', va='center', fontsize=20,
                        color=OGS_C.MEX_PINK)
    ita.set(rotation=-30)
    self.ax.add_patch(mpatches.Rectangle(xy, w, h, linewidth=1, color='blue',
                                         fill=False, label="Station Area"))
    self.ax.add_feature(cfeature.OCEAN, facecolor=("lightblue")) # type: ignore
    self.ax.add_feature(cfeature.BORDERS, linewidth=0.5, # type: ignore
                        edgecolor=OGS_C.MEX_PINK)
    self.ax.add_feature(cfeature.COASTLINE, linewidth=0.5, # type: ignore
                        edgecolor='black')
    self.ax.set_extent(extent, crs=proj) # type: ignore
    self.ax.set_aspect('equal', adjustable='box')
    gl = self.ax.gridlines() # type: ignore
    gl.left_labels = True
    gl.top_labels = True
    # Scale bar (using matplotlib-scalebar)
    self._add_scale_bar(self.ax, extent)
    # North arrow (using matplotlib-map-utils)
    self._add_north_arrow(self.ax, proj)
    if x is None or y is None:
      if legend:
        self.ax.legend()
      return
    if type(x) in [tuple, list]:
      if type(x) == pd.Series:
        self.ax.scatter(x.apply(lambda i: i[0]), y.apply(lambda i: i[0]),
                        s=self.s, marker=self.marker, facecolors=facecolors,
                        edgecolors=OGS_C.OGS_BLUE, label="OGS Catalog")
        self.ax.scatter(x.apply(lambda i: i[1]), y.apply(lambda i: i[1]),
                        s=self.s, marker=self.marker, label=label,
                        facecolors=facecolors, edgecolors=OGS_C.MEX_PINK)
        for a, b in zip(x, y):
          self.ax.plot([a[0], a[1]], [b[0], b[1]], color='gray',
                       linewidth=1.5)
    else:
      self.ax.scatter(x, y, s=self.s, marker=self.marker, label=label,
                      facecolors=facecolors, edgecolors=edgecolors)
    if text is not None:
      self.add_text(text, x + 0.1, y + 0.1, color=color, fontsize=12,
                    horizontalalignment='center',
                    verticalalignment='center')
    if legend:
      self.ax.legend()
    if title:
      self.ax.set_title(title)
    if output is not None: self.savefig(output=output)

  def add_plot(self, x, y, xlabel=None, ylabel=None,
               color: str | None = OGS_C.MEX_PINK,
               label=None, facecolors=None, edgecolors=None, legend=None,
               s=None, output=None, savefig=False, marker=None) -> None:
    if marker is not None: self.marker = marker
    self.ax.scatter(x, y, s=self.s if s is None else s, c=color,
                    marker=self.marker, label=label, facecolors=facecolors,
                    edgecolors=edgecolors)
    if xlabel:
      self.ax.set_xlabel(xlabel)
    if ylabel:
      self.ax.set_ylabel(ylabel)
    if legend is not None:
      self.ax.legend()
    if output is not None: savefig = True
    if savefig: self.savefig(output=output)

  def add_text(self, text, x, y, color=OGS_C.OGS_BLUE, fontsize=12,
                horizontalalignment='center', verticalalignment='center',
                output=None, savefig=False) -> None:
    for t, pos_x, pos_y in zip(text, x, y):
      self.ax.text(pos_x, pos_y, t, color=color, fontsize=fontsize,
                   horizontalalignment=horizontalalignment,
                   verticalalignment=verticalalignment)
    if output is not None: savefig = True
    if savefig: self.savefig(output=output)

class scatter_plotter(plotter):
  def __init__(self, x, y, xlabel=None, ylabel=None, title=None, fig=None,
               ax=None, color=OGS_C.OGS_BLUE, gs=111, label=None, legend=False,
               marker='o', aspect=None, edgecolors=OGS_C.OGS_BLUE,
               facecolors=OGS_C.OGS_BLUE, output=None,
               verbose: bool = False) -> None:
    super().__init__(fig=fig, verbose=verbose)
    self.ax: Axes = self.fig.add_subplot(gs)
    if xlabel:
      self.ax.set_xlabel(xlabel)
    if ylabel:
      self.ax.set_ylabel(ylabel)
    if title:
      self.ax.set_title(title)
    if legend:
      self.ax.legend()
    if aspect is not None:
      self.fig.gca().set_aspect(aspect, adjustable='box')
    self.ax.scatter(x, y, color=color, label=label, marker=marker,
                    edgecolors=edgecolors, facecolors=facecolors)
    if output is not None: self.savefig(output=output)

  def add_plot(self, x, y, xlabel=None, ylabel=None, color=OGS_C.MEX_PINK,
               label=None, legend=None, aspect=None, output=None,
               savefig=False) -> None:
    self.ax.plot(x, y, color=color, label=label)
    if xlabel:
      self.ax.set_xlabel(xlabel)
    if ylabel:
      self.ax.set_ylabel(ylabel)
    if legend is not None:
      self.ax.legend()
    if aspect is not None:
      plt.gca().set_aspect(aspect, adjustable='box')
    if output is not None: savefig = True
    if savefig: self.savefig(output=output)

class histogram_plotter(plotter):
  def __init__(self, data, bins=OGS_C.NUM_BINS, xlabel=None,
               ylabel="Number of Events", title=None, fig=None, ax=None,
               color=OGS_C. OGS_BLUE, gs=111, label=None, legend=False,
               xlim=None, edgecolor=None, facecolor=None, yscale=None,
               output=None, verbose: bool = False) -> None:
    super().__init__(fig=fig, figsize=(10, 5), verbose=verbose)
    self.ax = self.fig.add_subplot(gs)
    if xlabel:
      self.ax.set_xlabel(xlabel)
    if ylabel:
      self.ax.set_ylabel(ylabel)
    if title:
      self.ax.set_title(title)
    if xlim is not None:
      self.ax.set(xlim=xlim)
      if type(xlim) != list and type(xlim) != tuple:
        raise ValueError("xlim must be a list or tuple of two floats.")
      if len(xlim) != 2:
        raise ValueError("xlim must be a list or tuple of two floats.")
      if xlim[0] >= xlim[1]:
        raise ValueError("xlim[0] must be less than xlim[1].")
      self.bins = np.linspace(xlim[0], xlim[1], bins + 1)
    else:
      _, self.bins = np.histogram(data, bins=bins)
    y, _, _ = self.ax.hist(
      data,
      bins=self.bins, # type: ignore
      color=color,
      label=label,
      align='mid',
    )
    if legend:
      mean = float(np.mean(data))
      self.ax.axvline(x=mean, c='k', lw=1, alpha=0.5, ls='--',
                      label=f"Mean: {mean:.3E}")
      std = float(np.std(data))
      self.ax.axvline(x=mean + std, c='r', lw=1, alpha=0.5, ls='--',
                      label=f"Standard Deviation: {std:.3E}")
      self.ax.axvline(x=mean - std, c='r', lw=1, alpha=0.5, ls='--')
      self.ax.legend()
    if yscale is not None:
      self.ax.set_yscale(yscale)
    if output is not None: self.savefig(output=output)

  def add_fit(self, func, p0=None, color=OGS_C.MEX_PINK, label=None,
              output=None, savefig=False, xlabel=None, ylabel=None,
              title=None, legend=None, alpha=None) -> None:
    y2 = self.ax.twinx()
    y2.set_ylabel("Probability Density", color=color)
    y2.tick_params(axis='y', labelcolor=color)
    y = func(self.bins, p0)
    y2.plot(self.bins, y, color=color, label=label, alpha=alpha)
    if xlabel:
      self.ax.set_xlabel(xlabel)
    if ylabel:
      self.ax.set_ylabel(ylabel)
    if title:
      self.ax.set_title(title)
    if legend is not None:
      self.ax.legend()
      if legend == False:
        self.ax.get_legend().remove()
    if output is not None: savefig = True
    if savefig: self.savefig(output=output)

  def add_plot(self, data, xlabel=None, ylabel=None, title=None, step=False,
               color=OGS_C.MEX_PINK, label=None, legend=None, alpha=0.5,
               output=None, savefig=False, xscale=None, yscale=None) -> None:
    if step:
      y, _ = np.histogram(data, bins=self.bins)
      self.ax.step(self.bins[:-1], y, color=color, label=label, where='post')
    else:
      y, _, _ = self.ax.hist(
        data,
        bins=self.bins, # type: ignore
        color=color,
        label=label,
        align='mid',
        alpha=alpha
      )
    self.ax.set_ylim(bottom=0.9,
                     top=OGS_C.decimeter(max(*y, self.ax.get_ylim()[1]),
                                         "ken"))
    if xlabel:
      self.ax.set_xlabel(xlabel)
    if ylabel:
      self.ax.set_ylabel(ylabel)
    if title:
      self.ax.set_title(title)
    if legend is not None:
      self.ax.legend()
      if legend == False:
        self.ax.get_legend().remove()
    if xscale is not None:
      self.ax.set_xscale(xscale)
    if yscale is not None:
      self.ax.set_yscale(yscale)
    if output is not None: savefig = True
    if savefig: self.savefig(output=output)

class ConfMtx_plotter(plotter):
  def __init__(self, data, title=None, fig=None, ax=None,
               color=OGS_C.MEX_PINK, gs=111, label=None, legend=False,
               facecolor=None, edgecolor=None, output=None,
               basename=None, targetname=None, verbose: bool = False) -> None:
    super().__init__(fig=fig, figsize=(10, 5), verbose=verbose)
    self.ax: Axes = self.fig.add_subplot(gs)
    if title: self.ax.set_title(title)
    disp = ConfMtxDisp(data, display_labels=label)
    disp.plot(values_format='d', colorbar=True, ax=self.ax)
    for labels in disp.text_.ravel():
      labels.set(color=OGS_C.MEX_PINK, fontsize=12, fontweight="bold")
    disp.im_.set(clim=(0, max(data.flatten())), cmap="Blues", norm="log")
    if basename:
      disp.ax_.set_ylabel(f"{basename}")
    if targetname:
      disp.ax_.set_xlabel(f"{targetname}")
    if output is not None: self.savefig(output=output)
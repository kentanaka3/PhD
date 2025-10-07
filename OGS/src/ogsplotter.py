import numpy as np
import obspy as op
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
from pathlib import Path
from obspy.geodetics import gps2dist_azimuth
from datetime import datetime, timedelta as td
from sklearn.metrics import ConfusionMatrixDisplay as ConfMtxDisp

import ogsconstants as OGS_C

IMG_PATH = Path(__file__).parent.parent / "img"
IMG_PATH.mkdir(parents=True, exist_ok=True)

def v_lat_long_to_distance(lng1, lat1, depth1, lng2, lat2, depth2, dim=2):
  return [np.sqrt((gps2dist_azimuth(lt1, lg1, lt2, lg2)[0] / 1000.0) ** 2 +
            ((abs(dp2 - dp1) / 1000.0) ** 2 if dim == 3 else 0.))
          for lg1, lt1, dp1, lg2, lt2, dp2 in zip(
            lng1, lat1, depth1, lng2, lat2, depth2)]

class plotter:
  def __init__(self, figsize=(20, 10), fig=None, **kwargs) -> None:
    plt.rcParams.update({'font.size': 12})
    self.figsize = figsize
    self.fig = fig if fig else plt.figure(figsize=figsize,
                                          layout="compressed")

  def savefig(self, output=None, **kwargs) -> None:
    if output is not None:
      self.output = output
    if self.output is not None:
      plt.savefig(Path(IMG_PATH, self.output), bbox_inches='tight', dpi=300,
                  **kwargs)
      print(f"Figure saved to {self.output}")

class line_plotter(plotter):
  def __init__(
      self, x, y, xlabel=None, ylabel=None, title=None, fig=None, ax=None,
      color=OGS_C.OGS_BLUE, gs=111, label=None, legend=False,
      output=None) -> None:
    super().__init__(fig=fig)
    self.ax = self.fig.add_subplot(gs)
    if xlabel: self.ax.set_xlabel(xlabel)
    if ylabel: self.ax.set_ylabel(ylabel)
    if title: self.ax.set_title(title)
    self.ax.plot(x, y, color=color, label=label)
    if legend: self.ax.legend()
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
  def __init__(self, picks, event, filepath, inventory, fig=None, ax=None,
               gs=None, title=None, color=None, xlabel="Time (s)", output=None,
               ylabel="Epicentral Distance (km)", ymax=np.inf,
               center=None) -> None:
    super().__init__(fig=fig)
    self.ax = self.fig.add_subplot(111)
    self.pick_time = op.UTCDateTime(event[OGS_C.TIME_STR])
    self.filepath = filepath
    self.inventory = inventory
    self.ymax = ymax
    self.t = event[OGS_C.TIME_STR]
    diff = 0.
    if center is None:
      center = [event[OGS_C.LONGITUDE_STR], event[OGS_C.LATITUDE_STR]]
    else:
      diff = gps2dist_azimuth(
        center[1], center[0],
        event[OGS_C.LATITUDE_STR], event[OGS_C.LONGITUDE_STR])[0] / 1000.0
    self.x, self.y = center
    self.z = event[OGS_C.DEPTH_STR] / 1000.0  # Convert to km
    self.offset = td(seconds=1)
    self.stream = op.read(Path(self.filepath, str(self.pick_time.year),
                               "{:02}".format(self.pick_time.month),
                               "{:02}".format(self.pick_time.day),
                               OGS_C.ALL_WILDCHAR_STR),
                          starttime=self.pick_time - self.offset,
                          endtime=self.pick_time + td(seconds=30)).select(
      component='Z')
    self.stream.detrend()
    self.stream.filter("highpass", freq=2)
    y_max = 0
    self.sta2color = {}
    self.antis2c = {}
    for sta, df in picks.groupby(OGS_C.STATION_STR):
      trace = self.stream.select(station=sta)
      if len(trace) == 0:
        print(f"Warning: No trace found for station {sta}. Skipping.")
        continue
      trace = trace[0]
      normed = trace.event - np.mean(trace.event)
      normed = normed / np.max(np.abs(normed))
      if sta not in self.inventory: continue
      station_x, station_y, station_z, _ = self.inventory[sta]
      y_ = np.sqrt((gps2dist_azimuth(
            station_y, station_x, self.y, self.x
          )[0] / 1000.0) ** 2 + (abs(self.z - station_z) / 1000.0) ** 2
      )
      base = int(np.log10(y_))
      y_max = max(((y_ // (10 ** base)) + 1) * 10 ** base, y_max)
      if color is not None:
        self.ax.plot(trace.times(), 2 * normed + y_, color=color)
        self.sta2color[sta] = color
      else:
        self.ax.plot(trace.times(), 2 * normed + y_)
        self.sta2color[sta] = self.ax.lines[-1].get_color()
      for _, p in df.groupby(OGS_C.PHASE_STR):
        if len(p.index) > 1:
          print(f"Warning: Multiple picks for {sta} at {p[OGS_C.TIME_STR].iloc[0]}. "
                "Using the first one.")
        p = p.iloc[0]
        x_ = datetime(p[OGS_C.TIME_STR]) - (datetime(self.t) - self.offset)
        if p[OGS_C.PHASE_STR] == "P":
            ls = '-'
            lc = "red"
        else:
            ls = '--'
            lc = "blue"
        self.ax.plot(np.array([x_, x_]), [y_ - 3, y_ + 3], 'k', ls=ls,
                     color=lc)
      self.ax.text(31.3, y_, sta, fontsize=8,)
    if xlabel:
      self.ax.set_xlabel(xlabel)
    if ylabel:
      self.ax.set_ylabel(ylabel)
    self.ax.set_ylim(0, min(y_max, self.ymax))
    self.ax.set_ylim(0)
    self.ax.set_xlim(0)
    if diff != 0.:
      xmin, xmax = self.ax.get_xlim()
      self.ax.hlines(y=diff, xmin=xmin, xmax=xmax, color=OGS_C.MEX_PINK,
                     linestyles='--')
      self.ax.hlines(y=-diff, xmin=xmin, xmax=xmax, color=OGS_C.MEX_PINK,
                     linestyles='--')
      self.ax.hlines(y=0, xmin=xmin, xmax=xmax, color=OGS_C.ALN_GREEN,
                    linestyles='--')
      self.y2 = self.ax.secondary_yaxis(
        .95, functions=(lambda x: x - diff, lambda x: x + diff),
        label="Distance from Event Location", color=OGS_C.MEX_PINK)
    if title:
      self.ax.set_title(title)
    if output is not None: self.savefig(output=output)

  def add_plot(self, picks, color=None, label=None, output=None, savefig=False,
               alpha=1., flip=False, ymax=None) -> None:
    if ymax is not None: self.ymax = ymax
    y_max = 0
    for sta, df in picks.groupby(OGS_C.STATION_STR):
      trace = self.stream.select(station=sta)
      if len(trace) == 0:
        print(f"Warning: No trace found for station {sta}. Skipping.")
        continue
      trace = trace[0]
      normed = trace.event - np.mean(trace.event)
      normed = normed / np.max(np.abs(normed))
      station_x, station_y, station_z, _ = self.inventory[sta]
      y_ = np.sqrt((gps2dist_azimuth(
        station_y, station_x, self.y, self.x)[0] / 1000.0) ** 2 +
        (abs(self.z - station_z) / 1000.0) ** 2)
      if flip:
        if y_ < 0: raise ValueError("y_ must be positive for plotting.")
        base = int(np.log10(y_))
        y_max = max(((y_ // (10 ** base)) + 1) * 10 ** base, y_max)
        y_ = -y_
      if y_ > self.ymax:
        print(f"Warning: y_ ({y_}) exceeds ymax ({self.ymax}). Skipping {sta}.")
        continue
      if label is None: label = sta
      tmp_color = self.sta2color.get(sta, color)
      if tmp_color is not None:
        self.ax.plot(trace.times(), 2 * normed + y_, label=label,
                     color=tmp_color, alpha=alpha)
      else:
        self.ax.plot(trace.times(), 2 * normed + y_, label=label, alpha=alpha)
      if flip:
        self.antis2c[sta] = self.ax.lines[-1].get_color()
      else:
        self.sta2color[sta] = self.ax.lines[-1].get_color()
      for _, p in df.groupby(OGS_C.PHASE_STR):
        if len(p.index) > 1:
          print(f"Warning: Multiple picks for {sta} at {p[OGS_C.TIME_STR].iloc[0]}. "
                "Using the first one.")
        p = p.iloc[0]
        x_ = datetime(p[OGS_C.TIME_STR]) - (datetime(self.t) - self.offset)
        if p[OGS_C.PHASE_STR] == OGS_C.PWAVE:
          ls = '-'
          lc = "red"
        else:
          ls = '--'
          lc = "blue"
        self.ax.plot(np.array([x_, x_]), [y_ - 3, y_ + 3], 'k', ls=ls,
                     color=lc)
      if not sta in self.sta2color: self.ax.text(31.3, y_, sta, fontsize=8,)
    if flip:
      y_max = min(y_max, self.ymax)
      self.ax.set_ylim(-y_max, y_max)
      ticks = self.ax.get_yticks()
      self.ax.set_yticklabels([f"{int(abs(tick))}" for tick in ticks])
    plt.tight_layout(pad=0.1)
    if output is not None: savefig = True
    if savefig: self.savefig(output=output)


class stream_plotter(plotter):
  def __init__(self, stream, fig=None, ax=None, gs=111, title=None,
               xlabel="Time (s)", ylabel="Amplitude", color=OGS_C.OGS_BLUE,
               output=None) -> None:
    super().__init__(fig=fig)
    self.ax = self.fig.add_subplot(gs)
    if title:
      self.ax.set_title(title)
    if xlabel:
      self.ax.set_xlabel(xlabel)
    if ylabel:
      self.ax.set_ylabel(ylabel)
    for tr in stream:
      self.ax.plot(tr.times(), tr.data, color=color, label=tr.id)
    if output is not None: self.savefig(output=output)

class day_plotter(plotter):
  def __init__(self, picks, ylabel=None, title=None, ylim=None, label=None,
               output=None, legend=None, yscale=None, color=OGS_C.OGS_BLUE,
               grid=False) -> None:
    from obspy import UTCDateTime
    super().__init__()
    x = picks.value_counts().sort_index()
    y = np.cumsum(x.values)
    x = [UTCDateTime(xx).date for xx in x.index]
    self.ax = self.fig.add_subplot(111)
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
    if output is not None:
      self.savefig(output=output)

  def add_plot(self, picks, ylabel=None, title=None, output=None, label=None,
               color=None, legend=None, savefig=False) -> None:
    x = picks.value_counts().sort_index()
    y = np.cumsum(x.values)
    x = x.index
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
    if output is not None: savefig = True
    if savefig: self.savefig(output=output)

class map_plotter(plotter):
  def __init__(self, domain : list[float], x=None, y=None, text=None,
               xlabel=None, ylabel=None, title=None, fig=None, ax=None, s=20,
               proj=ccrs.PlateCarree(), color=OGS_C.OGS_BLUE, gs=111, label=None,
               legend=False, marker='o', facecolors=OGS_C.OGS_BLUE,
               edgecolors=OGS_C.OGS_BLUE, output=None) -> None:
    assert len(domain) == 4, "Domain must be a list of four floats: [min_lon, max_lon, min_lat, max_lat]"
    super().__init__(fig=fig)
    self.proj = proj
    self.ax = self.fig.add_subplot(gs, projection=self.proj)
    self.marker = marker
    self.output = output
    self.s = s
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

  def add_plot(self, x, y, xlabel=None, ylabel=None, color=OGS_C.MEX_PINK,
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
               facecolors=OGS_C.OGS_BLUE, output=None) -> None:
    super().__init__(fig=fig)
    self.ax = self.fig.add_subplot(gs)
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
               xlim=None, edgecolor=None, facecolor=None,
               output=None) -> None:
    super().__init__(fig=fig, figsize=(10, 5))
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
    self.bins = self.bins - (self.bins[1] - self.bins[0]) / 2.0
    self.ax.hist(data, bins=self.bins, color=color, label=label, # type: ignore
                 align='left')
    if legend:
      mean = float(np.mean(data))
      self.ax.axvline(x=mean, c='k', lw=1, alpha=0.5, ls='--', label="Mean" + (
        " of " + title if title else ""))
      std = float(np.std(data))
      self.ax.axvline(x=mean + std, c='r', lw=1, alpha=0.5, ls='--',
                      label="Standard Deviation" + (" of " + title
                                                    if title else ""))
      self.ax.axvline(x=mean - std, c='r', lw=1, alpha=0.5, ls='--')
      self.ax.legend()
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
               color=OGS_C.MEX_PINK, label=None, legend=None, facecolor=None,
               edgecolor=None, alpha=0.5, output=None, savefig=False) -> None:
    if step:
      y, _ = np.histogram(data, bins=self.bins)
      self.ax.step(self.bins[:-1], y, color=color, label=label, where='mid')
    else:
      self.ax.hist(data, bins=self.bins[:-1], color=color, # type: ignore
                   label=label, facecolor=facecolor, edgecolor=edgecolor,
                   align='right', alpha=alpha)
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

class ConfMtx_plotter(plotter):
  def __init__(self, data, title=None, fig=None, ax=None,
               color=OGS_C.MEX_PINK, gs=111, label=None, legend=False,
               facecolor=None, edgecolor=None, output=None) -> None:
    super().__init__(fig=fig, figsize=(10, 5))
    self.ax = self.fig.add_subplot(gs)
    if title: self.ax.set_title(title)
    disp = ConfMtxDisp(data, display_labels=label)
    disp.plot(values_format='d', colorbar=True, ax=self.ax)
    for labels in disp.text_.ravel():
      labels.set(color=OGS_C.MEX_PINK, fontsize=12, fontweight="bold")
    disp.im_.set(clim=(0, max(data.flatten())), cmap="Blues", norm="log")
    if output is not None: self.savefig(output=output)
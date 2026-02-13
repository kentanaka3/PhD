import os
import sys
import copy
import obspy
import argparse
import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
# Set the project folder
PRJ_PATH = Path(os.path.dirname(__file__)).parent
INC_PATH = os.path.join(PRJ_PATH, "inc")
if INC_PATH not in sys.path:
  sys.path.append(INC_PATH)
  import initializer as ini
  from resources.constants import *
else:
  from resources.constants import *
  import initializer as ini


def circumcenter(P: np.ndarray) -> np.ndarray:
  A, B, C = P
  D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
  Ux = ((A[0]**2 + A[1]**2) * (B[1] - C[1]) + (B[0]**2 + B[1]**2)
        * (C[1] - A[1]) + (C[0]**2 + C[1]**2) * (A[1] - B[1])) / D
  Uy = ((A[0]**2 + A[1]**2) * (C[0] - B[0]) + (B[0]**2 + B[1]**2)
        * (A[0] - C[0]) + (C[0]**2 + C[1]**2) * (B[0] - A[0])) / D
  return np.array([Ux, Uy])


def delaunay_circumcenter(P: np.ndarray) -> np.ndarray:
  TRI = sp.spatial.Delaunay(P)
  return np.asarray([circumcenter(P[simplex]) for simplex in TRI.simplices])


def incenter(P: np.ndarray) -> np.ndarray:
  A, B, C = P
  a = np.linalg.norm(B - C)
  b = np.linalg.norm(C - A)
  c = np.linalg.norm(A - B)
  return (a * A + b * B + c * C) / (a + b + c)


def delaunay_incenter(P: np.ndarray) -> np.ndarray:
  TRI = sp.spatial.Delaunay(P)
  return np.asarray([incenter(P[simplex]) for simplex in TRI.simplices])


def orthocenter(P: np.ndarray) -> np.ndarray:
  A, B, C = P
  a = np.linalg.norm(B - C)
  b = np.linalg.norm(C - A)
  c = np.linalg.norm(A - B)
  return (a**2 * (B + C) + b**2 * (A + C) + c**2 * (A + B)) / (a**2 + b**2 + c**2) - A - B - C


def delaunay_orthocenter(P: np.ndarray) -> np.ndarray:
  TRI = sp.spatial.Delaunay(P)
  return np.asarray([orthocenter(P[simplex]) for simplex in TRI.simplices])


def eulercenter(P: np.ndarray) -> np.ndarray:
  cx, cy = circumcenter(P)
  ox, oy = orthocenter(P)
  return (cx + ox) / 2, (cy + oy) / 2


def delaunay_eulercenter(P: np.ndarray) -> np.ndarray:
  TRI = sp.spatial.Delaunay(P)
  return np.asarray([eulercenter(P[simplex]) for simplex in TRI.simplices])


def centroid(P: np.ndarray) -> np.ndarray:
  return P.mean(axis=0)


def delaunay_centroid(P: np.ndarray) -> np.ndarray:
  TRI = sp.spatial.Delaunay(P)
  return np.asarray([centroid(P[simplex]) for simplex in TRI.simplices])


class StationHierarchy:
  def __init__(self, positions, depth=0):
    self.t = None
    self.x = None
    self.y = None
    self.z = None
    self.depth = depth
    self.graph = nx.Graph()
    self.positions = positions
    for i, pos in enumerate(positions):
      self.graph.add_node(i, pos=pos)
    if len(positions) > 2:
      TESS = sp.spatial.Delaunay(positions)
      CENTROID = nx.Graph()
      for i, spx in enumerate(TESS.simplices):
        nx.add_cycle(self.graph, spx)
        CENTROID.add_node(i, pos=centroid(positions[spx]))
      C_NODES = copy.deepcopy(CENTROID.nodes())
      for node in C_NODES:
        for neighbor in TESS.neighbors[node]:
          if neighbor > 0:
            CENTROID.add_edge(node, neighbor)
      C_COLORS = nx.coloring.equitable_color(CENTROID, num_colors=4)
      colors = set(C_COLORS.values())
      self.t = StationHierarchy(
          np.asarray([CENTROID.nodes[node]['pos'] for node in C_NODES
                      if C_COLORS[node] == 0]), depth=self.depth + 1)
      if 1 in colors:
        self.x = StationHierarchy(
            np.asarray([CENTROID.nodes[node]['pos'] for node in C_NODES
                        if C_COLORS[node] == 1]), depth=self.depth + 1)
      if 2 in colors:
        self.y = StationHierarchy(
            np.asarray([CENTROID.nodes[node]['pos'] for node in C_NODES
                        if C_COLORS[node] == 2]), depth=self.depth + 1)
      if 3 in colors:
        self.z = StationHierarchy(
            np.asarray([CENTROID.nodes[node]['pos'] for node in C_NODES
                        if C_COLORS[node] == 3]), depth=self.depth + 1)
    elif len(positions) == 2:
      self.graph.add_edge(0, 1)

  def draw(self, c='k', a=1.):
    nx.draw(self.graph, nx.get_node_attributes(self.graph, 'pos'),
            node_size=50, node_color=c, alpha=a)
    frac = 2 / 3
    if self.t:
      self.t.draw('r', frac * a)
    if self.x:
      self.x.draw('g', frac * a)
    if self.y:
      self.y.draw('b', frac * a)
    if self.z:
      self.z.draw('m', frac * a)

  def __str__(self):
    offset = "\t" * self.depth
    result = offset + f"Depth {self.depth}: {str(self.positions)}\n"
    if self.t:
      result += self.t.__str__()
    if self.x:
      result += self.x.__str__()
    if self.y:
      result += self.y.__str__()
    if self.z:
      result += self.z.__str__()
    return result

  def CM(self, depth=-1):
    if depth == 0:
      return np.mean(self.positions, axis=0)
    elif depth > 0:
      cm = []
      if self.t:
        cm.append(self.t.CM(depth - 1))
      if self.x:
        cm.append(self.x.CM(depth - 1))
      if self.y:
        cm.append(self.y.CM(depth - 1))
      if self.z:
        cm.append(self.z.CM(depth - 1))
      if not cm:
        return np.mean(self.positions, axis=0)
      return np.mean(np.asarray(cm), axis=0)
    # If depth is -1, return the center of mass of the current hierarchy
    cm = []
    if self.t:
      cm.append(self.t.CM())
    if self.x:
      cm.append(self.x.CM())
    if self.y:
      cm.append(self.y.CM())
    if self.z:
      cm.append(self.z.CM())
    if not cm:
      return np.mean(self.positions, axis=0)
    return np.mean(np.asarray(cm), axis=0)

  def tail(self):
    t = True
    if self.t:
      t = False
      self.t.tail()
    if self.x:
      t = False
      self.x.tail()
    if self.y:
      t = False
      self.y.tail()
    if self.z:
      t = False
      self.z.tail()
    if t:
      print(self.positions)
    return

def gen_points(inventory: obspy.Inventory) -> np.ndarray:
  x = [station.longitude for network in inventory for station in network]
  y = [station.latitude for network in inventory for station in network]
  return np.asarray(list({(x, y) for x, y in np.c_[x, y].tolist()}))

def station_graph(P):
  SH = StationHierarchy(P)
  SH.tail()
  print(SH)
  plt.show()
  return SH

def clustering(filepath: Path) -> None:
  df = pd.read_csv(filepath)
  x = df['x'].values
  y = df['y'].values
  P = np.c_[x, y]
  SH = StationHierarchy(P)
  SH.draw()
  SH.tail()
  print(SH)
  plt.show()
  return

def main(args: argparse.Namespace):
  global DATA_PATH
  DATA_PATH = args.directory.parent
  INVENTORY = obspy.Inventory()
  for station in Path(DATA_PATH, "station").iterdir():
    try:
      S = obspy.read_inventory(station)
    except:
      continue
    INVENTORY.extend(S)
  station_graph(gen_points(INVENTORY))
  # clustering(Path("/Users/admin/Downloads/3MC.csv"))
  return


if __name__ == "__main__":
  main(ini.parse_arguments())

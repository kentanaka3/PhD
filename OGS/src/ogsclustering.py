"""
OGS Clustering Module
Subclasses for sklearn.cluster algorithms with integrated plotting
functionality.
"""

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure
from matplotlib import colormaps

import dadapy as ddpy

from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    AffinityPropagation,
    MeanShift,
    SpectralClustering,
    AgglomerativeClustering,
    DBSCAN,
    HDBSCAN, # type: ignore
    OPTICS,
    Birch,
    BisectingKMeans,
    FeatureAgglomeration,
)

from sklearn.metrics import (
  # Unsupervised metrics (require X, labels)
  silhouette_score,
  calinski_harabasz_score,
  davies_bouldin_score,
  # Supervised metrics (require y_true, labels)
  adjusted_rand_score,
  normalized_mutual_info_score,
  adjusted_mutual_info_score,
  homogeneity_score,
  completeness_score,
  v_measure_score,
  fowlkes_mallows_score,
  # Utility
  pairwise_distances,
)

from typing import Optional, Tuple, Union, Any, List, Callable, Dict

def labels_to_colormap(
    labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Any, Any]:
  """
  Map arbitrary cluster labels (may include -1) to 0..K-1 for colormapping.

  Parameters
  ----------
  labels : np.ndarray
    Cluster labels array.

  Returns
  -------
  tuple
    (encoded_labels, unique_labels, colormap, norm)
  """
  unique = np.unique(labels)
  label_to_idx = {lab: i for i, lab in enumerate(unique)}
  encoded = np.vectorize(label_to_idx.get, otypes=[int])(labels)

  cmap = colormaps.get_cmap("Paired").resampled(len(unique))
  norm = BoundaryNorm(np.arange(-0.5, len(unique) + 0.5), cmap.N)
  return encoded, unique, cmap, norm

def iter_range(values: Any) -> List[Any]:
  if isinstance(values, tuple) and len(values) == 3:
    return list(np.arange(*(values)))
  if isinstance(values, (list, tuple, np.ndarray)): return list(values)
  return []


class BaseClusterer(ABC):
  """
  Abstract base class for clustering algorithms with plotting capabilities.

  Attributes
  ----------
  model : sklearn clustering model
    The underlying sklearn clustering model.
  labels_ : np.ndarray
    Cluster labels after fitting.
  data_ : np.ndarray
    Data used for fitting.
  """

  def __init__(self, **kwargs):
    self.verbose: bool = kwargs.pop('verbose', False)
    self.model = self._create_model(**kwargs)
    self.labels_: Optional[np.ndarray] = None
    self.data_: Optional[np.ndarray] = None
    self._kwargs = kwargs

  @abstractmethod
  def _create_model(self, **kwargs) -> Any:
    """Create the underlying sklearn model."""
    pass

  @property
  def name(self) -> str:
    """Return the name of the clustering algorithm."""
    return self.__class__.__name__

  @property
  def optimize_metric(self) -> Callable[
      [np.ndarray, Callable[[np.ndarray, np.ndarray], Optional[float]]],
      Tuple[dict, Optional[float], Dict[Any, Optional[float]]]]:
    """Return the optimizer callable for this clusterer."""
    return self.__class__._optimize_metric

  @staticmethod
  def _optimize_metric(
      X: np.ndarray,
      metric: Callable[[np.ndarray, np.ndarray], Optional[float]],
      **kwargs) -> Tuple[dict, Optional[float], Dict[Any, Optional[float]]]:
    """
    Default optimizer: no-op for algorithms without a tuning routine.

    Returns
    -------
    tuple
      (best_params, best_score, scores_by_param)
    """
    return {}, None, {}

  def fit_predict(self, X: np.ndarray) -> np.ndarray:
    """
    Fit the model and return cluster labels.

    Parameters
    ----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Cluster labels.
    """
    self.data_ = X
    labels = self.model.fit_predict(X)
    self.labels_ = labels
    return labels

  def plot(self,
           X: Optional[np.ndarray] = None,
           feature_x: int = 0,
           feature_y: int = 1,
           ax: Optional[Axes] = None,
           title: Optional[str] = None,
           xlabel: str = "Feature 1",
           ylabel: str = "Feature 2",
           point_size: Union[int, np.ndarray] = 20,
           alpha: float = 0.7,
           show_legend: bool = True,
           show_noise: bool = True,
           noise_color: str = "gray",
           noise_alpha: float = 0.3,
           figsize: Tuple[int, int] = (10, 8),
           colorbar: bool = True,
           **scatter_kwargs) -> Axes:
    """
    Plot the clustering results.

    Parameters
    ----------
    X : np.ndarray, optional
        Data to plot. If None, uses the fitted data.
    feature_x : int, default=0
        Index of feature for x-axis.
    feature_y : int, default=1
        Index of feature for y-axis.
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.
    title : str, optional
        Plot title. If None, uses algorithm name.
    xlabel : str, default="Feature 1"
        X-axis label.
    ylabel : str, default="Feature 2"
        Y-axis label.
    point_size : int or np.ndarray, default=20
        Size of scatter points.
    alpha : float, default=0.7
        Transparency of points.
    show_legend : bool, default=True
        Whether to show legend.
    show_noise : bool, default=True
        Whether to show noise points (label=-1).
    noise_color : str, default="gray"
        Color for noise points.
    noise_alpha : float, default=0.3
        Alpha for noise points.
    figsize : tuple, default=(10, 8)
        Figure size if creating new figure.
    colorbar : bool, default=True
        Whether to show colorbar.
    **scatter_kwargs
        Additional kwargs passed to scatter.

    Returns
    -------
    Axes
        The matplotlib axes with the plot.
    """
    if self.labels_ is None: raise ValueError(
      "Model must be fitted before plotting. Call fit() first."
    )

    data = X if X is not None else self.data_
    if data is None: raise ValueError("No data available for plotting.")

    if ax is None: fig, ax = plt.subplots(figsize=figsize)

    labels = self.labels_
    x_data = data[:, feature_x]
    y_data = data[:, feature_y]

    # Handle noise points separately
    noise_mask = labels == -1
    cluster_mask = ~noise_mask

    base_scatter_kwargs = dict(scatter_kwargs)
    if show_noise and np.any(noise_mask):
      noise_kwargs = {k: v for k, v in base_scatter_kwargs.items() if k != "s"}
      noise_size = base_scatter_kwargs.get("s", point_size)
      ax.scatter(x_data[noise_mask],
                 y_data[noise_mask],
                 c=noise_color,
                 s=noise_size
                 if isinstance(noise_size, int) else noise_size[noise_mask],
                 alpha=noise_alpha,
                 label="Noise",
                 marker="x",
                 **noise_kwargs)

    if np.any(cluster_mask):
      cluster_labels = labels[cluster_mask]
      encoded, unique, cmap, norm = labels_to_colormap(cluster_labels)

      cluster_kwargs = {k: v for k, v in base_scatter_kwargs.items()
                        if k != "s"}
      cluster_size = base_scatter_kwargs.get("s", point_size)
      sc = ax.scatter(x_data[cluster_mask],
                      y_data[cluster_mask],
                      c=encoded,
                      s=cluster_size if isinstance(cluster_size, int) else
                        cluster_size[cluster_mask],
                      alpha=alpha,
                      cmap=cmap,
                      norm=norm,
                      **cluster_kwargs)

      if colorbar:
        cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(len(unique)))
        cbar.ax.set_yticklabels([str(lab) for lab in unique])
        cbar.set_label("Cluster")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{self.name} Clustering")

    if show_legend and np.any(noise_mask): ax.legend()
    return ax

  def plot_3d(self,
              X: Optional[np.ndarray] = None,
              feature_x: int = 0,
              feature_y: int = 1,
              feature_z: int = 2,
              ax: Optional[Union[Axes, Axes3D]] = None,
              title: Optional[str] = None,
              xlabel: str = "Feature 1",
              ylabel: str = "Feature 2",
              zlabel: str = "Feature 3",
              point_size: Union[int, np.ndarray] = 20,
              alpha: float = 0.7,
              figsize: Tuple[int, int] = (12, 10),
              **scatter_kwargs) -> Axes:
    """
    Plot the clustering results in 3D.

    Parameters
    ----------
    X : np.ndarray, optional
        Data to plot. If None, uses the fitted data.
    feature_x : int, default=0
        Index of feature for x-axis.
    feature_y : int, default=1
        Index of feature for y-axis.
    feature_z : int, default=2
        Index of feature for z-axis.
    ax : Axes, optional
        Matplotlib 3D axes to plot on.
    title : str, optional
        Plot title.
    xlabel, ylabel, zlabel : str
        Axis labels.
    point_size : int or np.ndarray, default=20
        Size of scatter points.
    alpha : float, default=0.7
        Transparency of points.
    figsize : tuple, default=(12, 10)
        Figure size if creating new figure.
    **scatter_kwargs
        Additional kwargs passed to scatter.

    Returns
    -------
    Axes
        The matplotlib 3D axes with the plot.
    """
    if self.labels_ is None: raise ValueError(
      "Model must be fitted before plotting. Call fit() first."
    )

    data = X if X is not None else self.data_
    if data is None: raise ValueError("No data available for plotting.")

    if ax is None:
      fig = plt.figure(figsize=figsize)
      ax = fig.add_subplot(111, projection='3d')

    labels = self.labels_
    cluster_mask = labels != -1

    base_scatter_kwargs = dict(scatter_kwargs)
    if np.any(cluster_mask):
      cluster_labels = labels[cluster_mask]
      encoded, unique, cmap, norm = labels_to_colormap(cluster_labels)

      cluster_kwargs = {k: v for k, v in base_scatter_kwargs.items()
                        if k != "s"}
      cluster_size = base_scatter_kwargs.get("s", point_size)
      cluster_kwargs["s"] = (
        cluster_size if isinstance(cluster_size, int)
        else cluster_size[cluster_mask]
      )
      ax.scatter(data[cluster_mask, feature_x],
                 data[cluster_mask, feature_y],
                 data[cluster_mask, feature_z],
                 c=encoded,
                 alpha=alpha,
                 cmap=cmap,
                 **cluster_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if isinstance(ax, Axes3D):
      ax.set_zlabel(zlabel)
    ax.set_title(title or f"{self.name} Clustering (3D)")
    return ax

  def get_cluster_centers(self) -> Optional[np.ndarray]:
    """Return cluster centers if available."""
    if hasattr(self.model, 'cluster_centers_'):
      return self.model.cluster_centers_
    return None

  def n_clusters(self) -> int:
    """Return the number of clusters found."""
    if self.labels_ is None:
      return 0
    unique_labels = np.unique(self.labels_)
    return len(unique_labels[unique_labels >= 0])

  def __repr__(self) -> str:
    return f"{self.name}({self._kwargs})"


class BaseClusteringScores(ABC):
  """
  Base class for clustering metric objects.

  Subclasses must implement `compute()`.
  """

  def __init__(self,
               X: np.ndarray,
               labels: np.ndarray,
               y_true: Optional[np.ndarray] = None):
    self.X = X
    self.labels = labels
    self.y_true = y_true

  @property
  def name(self) -> str: return self.__class__.__name__

  @abstractmethod
  def compute(self) -> Optional[float]:
    """Compute the metric value; returns None if not applicable."""
    raise NotImplementedError


class SilhouetteScore(BaseClusteringScores):
  def compute(self) -> Optional[float]:
    try: return float(silhouette_score(self.X, self.labels))
    except Exception: return None

class CalinskiHarabaszScore(BaseClusteringScores):
  def compute(self) -> Optional[float]:
    try: return float(calinski_harabasz_score(self.X, self.labels))
    except Exception: return None

class DaviesBouldinScore(BaseClusteringScores):
  def compute(self) -> Optional[float]:
    try: return davies_bouldin_score(self.X, self.labels)
    except Exception: return None

class AdjustedRandScore(BaseClusteringScores):
  def compute(self) -> Optional[float]:
    if self.y_true is None: return None
    try: return adjusted_rand_score(self.y_true, self.labels)
    except Exception: return None

class NormalizedMutualInfoScore(BaseClusteringScores):
  def compute(self) -> Optional[float]:
    if self.y_true is None: return None
    try: return float(normalized_mutual_info_score(self.y_true, self.labels))
    except Exception: return None

class AdjustedMutualInfoScore(BaseClusteringScores):
  def compute(self) -> Optional[float]:
    if self.y_true is None: return None
    try: return float(adjusted_mutual_info_score(self.y_true, self.labels))
    except Exception: return None

class HomogeneityScore(BaseClusteringScores):
  def compute(self) -> Optional[float]:
    if self.y_true is None: return None
    try: return float(homogeneity_score(self.y_true, self.labels))
    except Exception: return None

class CompletenessScore(BaseClusteringScores):
  def compute(self) -> Optional[float]:
    if self.y_true is None: return None
    try: return float(completeness_score(self.y_true, self.labels))
    except Exception: return None

class VMeasureScore(BaseClusteringScores):
  def compute(self) -> Optional[float]:
    if self.y_true is None: return None
    try: return float(v_measure_score(self.y_true, self.labels))
    except Exception: return None

class FowlkesMallowsScore(BaseClusteringScores):
  def compute(self) -> Optional[float]:
    if self.y_true is None: return None
    try: return fowlkes_mallows_score(self.y_true, self.labels)
    except Exception: return None


# =============================================================================
# Centroid-based Clustering
# =============================================================================
class OGSKMeans(BaseClusterer):
  """
  K-Means clustering with plotting capabilities.

  Parameters
  ----------
  n_clusters : int, default=8
    The number of clusters to form.
  init : str, default='k-means++'
    Method for initialization.
  n_init : int or 'auto', default='auto'
    Number of times the k-means algorithm is run.
  max_iter : int, default=300
    Maximum number of iterations.
  tol : float, default=1e-4
    Relative tolerance for convergence.
  random_state : int, optional
    Random state for reproducibility.
  *args, **kwargs
    Additional arguments passed to sklearn.cluster.KMeans.
  """

  def _create_model(self, n_clusters: int = 8, **kwargs) -> KMeans:
    return KMeans(n_clusters=n_clusters, **kwargs)

  def plot(self,
           show_centers: bool = True,
           center_marker: str = "X",
           center_size: int = 200,
           center_color: str = "red",
           *args,
           **kwargs) -> Axes:
    """
    Plot K-Means clustering results with optional cluster centers.

    Parameters
    ----------
    show_centers : bool, default=True
      Whether to show cluster centers.
    center_marker : str, default="X"
      Marker style for centers.
    center_size : int, default=200
      Size of center markers.
    center_color : str, default="red"
      Color of center markers.
    *args, **kwargs
      Arguments passed to BaseClusterer.plot().
    """
    ax = super().plot(*args, **kwargs)
    centers = self.get_cluster_centers()
    if show_centers and centers is not None:
      feature_x = kwargs.get('feature_x', 0)
      feature_y = kwargs.get('feature_y', 1)
      ax.scatter(centers[:, feature_x],
                 centers[:, feature_y],
                 c=center_color,
                 marker=center_marker,
                 s=center_size,
                 edgecolors="black",
                 linewidths=2,
                 label="Centers")
      ax.legend()
    return ax

class OGSMiniBatchKMeans(BaseClusterer):
  """
  Mini-Batch K-Means clustering with plotting capabilities.

  Faster than regular K-Means for large datasets.

  Parameters
  ----------
  n_clusters : int, default=8
    The number of clusters to form.
  batch_size : int, default=1024
    Size of the mini batches.
  **kwargs
    Additional arguments passed to sklearn.cluster.MiniBatchKMeans.
  """

  def _create_model(self,
                    n_clusters: int = 8,
                    batch_size: int = 1024,
                    **kwargs) -> MiniBatchKMeans:
    return MiniBatchKMeans(n_clusters=n_clusters,
                           batch_size=batch_size,
                           **kwargs)

  def plot(self,
           show_centers: bool = True,
           center_marker: str = "X",
           center_size: int = 200,
           center_color: str = "red",
           *args,
           **kwargs) -> Axes:
    """Plot Mini-Batch K-Means results with optional cluster centers."""
    ax = super().plot(*args, **kwargs)
    centers = self.get_cluster_centers()
    if show_centers and centers is not None:
      feature_x = kwargs.get('feature_x', 0)
      feature_y = kwargs.get('feature_y', 1)
      ax.scatter(centers[:, feature_x],
                 centers[:, feature_y],
                 c=center_color,
                 marker=center_marker,
                 s=center_size,
                 edgecolors="black",
                 linewidths=2,
                 label="Centers")
      ax.legend()
    return ax

class OGSBisectingKMeans(BaseClusterer):
  """
  Bisecting K-Means clustering with plotting capabilities.

  Hierarchical approach using K-Means.

  Parameters
  ----------
  n_clusters : int, default=8
    The number of clusters to form.
  **kwargs
    Additional arguments passed to sklearn.cluster.BisectingKMeans.
  """

  def _create_model(self, n_clusters: int = 8, **kwargs) -> BisectingKMeans:
    return BisectingKMeans(n_clusters=n_clusters, **kwargs)

  def plot(self,
           show_centers: bool = True,
           center_marker: str = "X",
           center_size: int = 200,
           center_color: str = "red",
           *args,
           **kwargs) -> Axes:
    """Plot Bisecting K-Means results with optional cluster centers."""
    ax = super().plot(*args, **kwargs)
    centers = self.get_cluster_centers()
    if show_centers and centers is not None:
      feature_x = kwargs.get('feature_x', 0)
      feature_y = kwargs.get('feature_y', 1)
      ax.scatter(centers[:, feature_x],
                 centers[:, feature_y],
                 c=center_color,
                 marker=center_marker,
                 s=center_size,
                 edgecolors="black",
                 linewidths=2,
                 label="Centers")
      ax.legend()
    return ax


# =============================================================================
# Density-based Clustering
# =============================================================================
class OGSDBSCAN(BaseClusterer):
  """
  DBSCAN clustering with plotting capabilities.

  Density-Based Spatial Clustering of Applications with Noise.

  Parameters
  ----------
  eps : float, default=0.5
      Maximum distance between two samples in the same neighborhood.
  min_samples : int, default=5
      Minimum number of samples in a neighborhood for a core point.
  **kwargs
      Additional arguments passed to sklearn.cluster.DBSCAN.
  """

  def _create_model(self,
                    eps: float = 0.5,
                    min_samples: int = 5,
                    **kwargs) -> DBSCAN:
    return DBSCAN(eps=eps, min_samples=min_samples, **kwargs)

  def plot(self, highlight_core: bool = False, *args, **kwargs) -> Axes:
    """
    Plot DBSCAN clustering results.

    Parameters
    ----------
    highlight_core : bool, default=False
        Whether to highlight core samples with different marker.
    *args, **kwargs
        Arguments passed to BaseClusterer.plot().
    """
    ax = super().plot(*args, **kwargs)

    if highlight_core and hasattr(self.model, 'core_sample_indices_'):
      if self.labels_ is None:
        raise ValueError("Model must be fitted before plotting.")
      core_mask = np.zeros(len(self.labels_), dtype=bool)
      core_mask[self.model.core_sample_indices_] = True
      data = kwargs.get('X', self.data_)
      if data is None:
        raise ValueError("No data available for plotting.")
      feature_x = kwargs.get('feature_x', 0)
      feature_y = kwargs.get('feature_y', 1)
      ax.scatter(data[core_mask, feature_x],
                 data[core_mask, feature_y],
                 facecolors='none',
                 edgecolors='black',
                 s=100,
                 linewidths=1.5,
                 label="Core samples")
      ax.legend()
    return ax


class OGSHDBSCAN(BaseClusterer):
  """
  HDBSCAN clustering with plotting capabilities.

  Hierarchical Density-Based Spatial Clustering of Applications with Noise.

  Parameters
  ----------
  min_cluster_size : int, default=5
      Minimum size of clusters.
  min_samples : int, optional
      Number of samples in a neighborhood for a core point.
  cluster_selection_epsilon : float, default=0.0
      Distance threshold for cluster selection.
  **kwargs
      Additional arguments passed to sklearn.cluster.HDBSCAN.
  """

  def _create_model(self, min_cluster_size: int = 5, **kwargs) -> HDBSCAN:
    return HDBSCAN(min_cluster_size=min_cluster_size, **kwargs)

  def plot(self, show_probabilities: bool = False, *args, **kwargs) -> Axes:
    """
    Plot HDBSCAN clustering results.

    Parameters
    ----------
    show_probabilities : bool, default=False
        Whether to use cluster probabilities for point alpha.
    *args, **kwargs
        Arguments passed to BaseClusterer.plot().
    """
    if show_probabilities and hasattr(self.model, 'probabilities_'):
      kwargs['alpha'] = self.model.probabilities_

    return super().plot(*args, **kwargs)

  def get_probabilities(self) -> Optional[np.ndarray]:
    """Return cluster membership probabilities if available."""
    if hasattr(self.model, 'probabilities_'): return self.model.probabilities_
    return None

class OGSOPTICS(BaseClusterer):
  """
  OPTICS clustering with plotting capabilities.

  Ordering Points To Identify the Clustering Structure.

  Parameters
  ----------
  min_samples : int, default=5
    Minimum number of samples in a neighborhood.
  max_eps : float, default=np.inf
    Maximum distance between two samples.
  **kwargs
    Additional arguments passed to sklearn.cluster.OPTICS.
  """

  def _create_model(self, min_samples: int = 5, **kwargs) -> OPTICS:
    return OPTICS(min_samples=min_samples, **kwargs)

  def plot(self, *args, **kwargs) -> Axes:
    """Plot OPTICS clustering results."""
    return super().plot(*args, **kwargs)

  def plot_reachability(self,
                        ax: Optional[Axes] = None,
                        figsize: Tuple[int, int] = (12, 4),
                        title: str = "OPTICS Reachability Plot") -> Axes:
    """
    Plot the reachability diagram.

    Parameters
    ----------
    ax : Axes, optional
      Matplotlib axes to plot on.
    figsize : tuple, default=(12, 4)
      Figure size.
    title : str
      Plot title.

    Returns
    -------
    Axes
    """
    if not hasattr(self.model, 'reachability_'): raise ValueError(
      "Model must be fitted first."
    )

    if ax is None: fig, ax = plt.subplots(figsize=figsize)

    if self.labels_ is None: raise ValueError("Model must be fitted first.")

    reachability = self.model.reachability_
    ordering = self.model.ordering_
    labels = self.labels_[ordering]

    encoded, unique, cmap, norm = labels_to_colormap(labels)

    for i, (reach, lab) in enumerate(zip(reachability[ordering], encoded)):
      color = cmap(norm(lab)) if labels[i] >= 0 else 'gray'
      ax.bar(i, reach, width=1, color=color, edgecolor='none')

    ax.set_xlabel("Sample ordering")
    ax.set_ylabel("Reachability distance")
    ax.set_title(title)
    return ax


class OGSAdvancedDensityPeaks(BaseClusterer):
  """
  Placeholder for advanced density-based clustering algorithms.

  To be implemented.
  """
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def fit_predict(self, X: np.ndarray) -> np.ndarray:
    labels_ = ddpy.Data(X, verbose=self.verbose).compute_clustering_ADP(
      **self._kwargs
    )
    self.labels_ = labels_
    return labels_

  def _create_model(self, **kwargs) -> Any:
    pass

  def plot(self, *args, **kwargs) -> Axes:
    """Plot Advanced Density Peaks clustering results."""
    return super().plot(*args, **kwargs)

# =============================================================================
# Connectivity-based Clustering
# =============================================================================
class OGSAgglomerative(BaseClusterer):
  """
  Agglomerative Clustering with plotting capabilities.

  Hierarchical clustering using a bottom-up approach.

  Parameters
  ----------
  n_clusters : int, default=2
    Number of clusters to find.
  linkage : str, default='ward'
    Linkage criterion: 'ward', 'complete', 'average', 'single'.
  **kwargs
    Additional arguments passed to sklearn.cluster.AgglomerativeClustering.
  """

  def _create_model(self,
                    n_clusters: int = 2,
                    linkage: str = 'ward',
                    **kwargs) -> AgglomerativeClustering:
    return AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,  # type: ignore
        **kwargs)

  def plot(self, *args, **kwargs) -> Axes:
    """Plot Agglomerative Clustering results."""
    return super().plot(*args, **kwargs)

  def plot_dendrogram(self,
                      ax: Optional[Axes] = None,
                      figsize: Tuple[int, int] = (12, 8),
                      truncate_mode: str = 'lastp',
                      p: int = 30,
                      **dendrogram_kwargs) -> Axes:
    """
    Plot dendrogram for hierarchical clustering.

    Requires the model to be fitted with compute_distances=True.

    Parameters
    ----------
    ax : Axes, optional
        Matplotlib axes to plot on.
    figsize : tuple, default=(12, 8)
        Figure size.
    truncate_mode : str, default='lastp'
        Truncation mode for dendrogram.
    p : int, default=30
        Number of leaves to show when truncating.
    **dendrogram_kwargs
        Additional arguments passed to scipy.cluster.hierarchy.dendrogram.
    """
    from scipy.cluster.hierarchy import dendrogram

    if not hasattr(self.model, 'distances_') or self.model.distances_ is None:
      raise ValueError("Dendrogram requires distances. "
                       "Refit with compute_distances=True.")

    if ax is None:
      fig, ax = plt.subplots(figsize=figsize)

    # Create linkage matrix from sklearn model
    counts = np.zeros(self.model.children_.shape[0])
    n_samples = len(self.model.labels_)
    for i, merge in enumerate(self.model.children_):
      current_count = 0
      for child_idx in merge:
        current_count += (1 if child_idx < n_samples
                          else counts[child_idx - n_samples])
      counts[i] = current_count

    linkage_matrix = np.column_stack(
        [self.model.children_, self.model.distances_, counts]).astype(float)

    dendrogram(linkage_matrix,
               ax=ax,
               truncate_mode=truncate_mode,
               p=p,
               **dendrogram_kwargs)

    ax.set_title(f"Dendrogram ({self._kwargs.get('linkage', 'ward')} linkage)")
    ax.set_xlabel("Sample index (or cluster size)")
    ax.set_ylabel("Distance")

    return ax


class OGSFeatureAgglomeration(BaseClusterer):
  """
  Feature Agglomeration with plotting capabilities.

  Similar to AgglomerativeClustering, but clusters features instead of samples.

  Parameters
  ----------
  n_clusters : int, default=2
      Number of clusters to find.
  **kwargs
      Additional arguments passed to sklearn.cluster.FeatureAgglomeration.
  """

  def _create_model(self,
                    n_clusters: int = 2,
                    **kwargs) -> FeatureAgglomeration:
    return FeatureAgglomeration(n_clusters=n_clusters, **kwargs)

  def fit(self, X: np.ndarray) -> "OGSFeatureAgglomeration":
    """Fit and get feature labels."""
    self.data_ = X
    self.model.fit(X)
    self.labels_ = self.model.labels_
    return self

  def transform(self, X: np.ndarray) -> np.ndarray:
    """Transform X to reduced feature space."""
    return self.model.transform(X)

  def plot(self, *args, **kwargs) -> Axes:
    """
    Plot feature clustering as a heatmap.

    Note: This shows feature clusters, not sample clusters.
    """
    if self.labels_ is None:
      raise ValueError("Model must be fitted before plotting.")

    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))

    n_features = len(self.labels_)
    encoded, unique, cmap, norm = labels_to_colormap(self.labels_)

    ax.bar(range(n_features),
           np.ones(n_features),
           color=[cmap(norm(e)) for e in encoded])
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Cluster Assignment")
    ax.set_title(kwargs.get('title', 'Feature Agglomeration Clusters'))
    ax.set_xticks(range(n_features))
    return ax


# =============================================================================
# Message-passing Clustering
# =============================================================================
class OGSAffinityPropagation(BaseClusterer):
  """
  Affinity Propagation clustering with plotting capabilities.

  Creates clusters by sending messages between pairs of samples.

  Parameters
  ----------
  damping : float, default=0.5
    Damping factor between 0.5 and 1.
  preference : float, optional
    Preferences for each point to be an exemplar.
  **kwargs
    Additional arguments passed to sklearn.cluster.AffinityPropagation.
  """

  def _create_model(self,
                    damping: float = 0.5,
                    **kwargs) -> AffinityPropagation:
    return AffinityPropagation(damping=damping, **kwargs)

  def plot(self,
           show_exemplars: bool = True,
           exemplar_marker: str = "D",
           exemplar_size: int = 150,
           *args,
           **kwargs) -> Axes:
    """
    Plot Affinity Propagation results with exemplars.

    Parameters
    ----------
    show_exemplars : bool, default=True
      Whether to highlight cluster exemplars.
    exemplar_marker : str, default="D"
      Marker style for exemplars.
    exemplar_size : int, default=150
      Size of exemplar markers.
    *args, **kwargs
      Arguments passed to BaseClusterer.plot().
    """
    ax = super().plot(*args, **kwargs)

    if show_exemplars and hasattr(self.model, 'cluster_centers_indices_'):
      data = kwargs.get('X', self.data_)
      feature_x = kwargs.get('feature_x', 0)
      feature_y = kwargs.get('feature_y', 1)
      indices = self.model.cluster_centers_indices_

      ax.scatter(data[indices, feature_x],
                 data[indices, feature_y],
                 c='red',
                 marker=exemplar_marker,
                 s=exemplar_size,
                 edgecolors='black',
                 linewidths=2,
                 label="Exemplars",
                 zorder=10)
      ax.legend()
    return ax

  def get_exemplar_indices(self) -> Optional[np.ndarray]:
    """Return indices of cluster exemplars."""
    if hasattr(self.model, 'cluster_centers_indices_'):
      return self.model.cluster_centers_indices_
    return None


class OGSMeanShift(BaseClusterer):
  """
  Mean Shift clustering with plotting capabilities.

  Finds blobs in a smooth density of samples.

  Parameters
  ----------
  bandwidth : float, optional
    Bandwidth used in the RBF kernel. If None, estimated automatically.
  **kwargs
    Additional arguments passed to sklearn.cluster.MeanShift.
  """

  def _create_model(self, **kwargs) -> MeanShift:
    return MeanShift(**kwargs)

  def plot(self,
           show_centers: bool = True,
           center_marker: str = "X",
           center_size: int = 200,
           center_color: str = "red",
           *args,
           **kwargs) -> Axes:
    """
    Plot Mean Shift clustering results.

    Parameters
    ----------
    show_centers : bool, default=True
      Whether to show cluster centers.
    center_marker : str, default="X"
      Marker style for centers.
    center_size : int, default=200
      Size of center markers.
    center_color : str, default="red"
      Color of center markers.
    *args, **kwargs
      Arguments passed to BaseClusterer.plot().
    """
    ax = super().plot(*args, **kwargs)
    centers = self.get_cluster_centers()
    if show_centers and centers is not None:
      feature_x = kwargs.get('feature_x', 0)
      feature_y = kwargs.get('feature_y', 1)
      ax.scatter(centers[:, feature_x],
                 centers[:, feature_y],
                 c=center_color,
                 marker=center_marker,
                 s=center_size,
                 edgecolors='black',
                 linewidths=2,
                 label="Centers")
      ax.legend()

    return ax


# =============================================================================
# Spectral Clustering
# =============================================================================


class OGSSpectralClustering(BaseClusterer):
  """
  Spectral Clustering with plotting capabilities.

  Apply clustering to a projection of the normalized Laplacian.

  Parameters
  ----------
  n_clusters : int, default=8
    Number of clusters to form.
  affinity : str, default='rbf'
    How to construct the affinity matrix.
  **kwargs
    Additional arguments passed to sklearn.cluster.SpectralClustering.
  """

  def _create_model(self, n_clusters: int = 8, **kwargs) -> SpectralClustering:
    return SpectralClustering(n_clusters=n_clusters, **kwargs)

  def plot(self, *args, **kwargs) -> Axes:
    """Plot Spectral Clustering results."""
    return super().plot(*args, **kwargs)


# =============================================================================
# Tree-based Clustering
# =============================================================================


class OGSBirch(BaseClusterer):
  """
  BIRCH clustering with plotting capabilities.

  Memory-efficient, online-learning algorithm.

  Parameters
  ----------
  n_clusters : int, default=3
    Number of clusters after the final clustering step.
  threshold : float, default=0.5
    Radius of the subcluster obtained by merging a new sample.
  branching_factor : int, default=50
    Maximum number of CF subclusters in each node.
  **kwargs
    Additional arguments passed to sklearn.cluster.Birch.
  """

  def _create_model(self,
                    n_clusters: int = 3,
                    threshold: float = 0.5,
                    **kwargs) -> Birch:
    return Birch(n_clusters=n_clusters, threshold=threshold, **kwargs)

  def plot(self, show_subcluster_centers: bool = False, *args,
           **kwargs) -> Axes:
    """
    Plot BIRCH clustering results.

    Parameters
    ----------
    show_subcluster_centers : bool, default=False
      Whether to show subcluster centers.
    *args, **kwargs
      Arguments passed to BaseClusterer.plot().
    """
    ax = super().plot(*args, **kwargs)

    if show_subcluster_centers and hasattr(self.model, 'subcluster_centers_'):
      centers = self.model.subcluster_centers_
      feature_x = kwargs.get('feature_x', 0)
      feature_y = kwargs.get('feature_y', 1)
      ax.scatter(centers[:, feature_x],
                 centers[:, feature_y],
                 c='orange',
                 marker='s',
                 s=50,
                 alpha=0.6,
                 label="Subcluster centers")
      ax.legend()

    return ax


# =============================================================================
# Utility Functions
# =============================================================================


def get_all_eval_metrics() -> dict[str, dict[str, type]]:
  """
  Get a dictionary of all available clustering metric classes.

  Returns
  -------
  dict
    Dictionary mapping metric names to classes.
  """
  return {
    "UnsupervisedScores": {
      # Unsupervised metrics (require X, labels)
      "SilhouetteScore": SilhouetteScore,
      "CalinskiHarabaszScore": CalinskiHarabaszScore,
      "DaviesBouldinScore": DaviesBouldinScore,
    },
    "SupervisedScores": {
      # Supervised metrics (require X, labels, y_true)
      "AdjustedRandScore": AdjustedRandScore,
      "NormalizedMutualInfoScore": NormalizedMutualInfoScore,
      "AdjustedMutualInfoScore": AdjustedMutualInfoScore,
      "HomogeneityScore": HomogeneityScore,
      "CompletenessScore": CompletenessScore,
      "VMeasureScore": VMeasureScore,
      "FowlkesMallowsScore": FowlkesMallowsScore,
    }
  }

def get_all_clusterers() -> dict[str, type[BaseClusterer]]:
  """
  Get a dictionary of all available clustering classes.

  Returns
  -------
  dict
    Dictionary mapping algorithm names to classes.
  """
  return {
    'AdvancedDensityPeaks': OGSAdvancedDensityPeaks,
    'AffinityPropagation': OGSAffinityPropagation,
    'Agglomerative': OGSAgglomerative,
    'Birch': OGSBirch,
    'BisectingKMeans': OGSBisectingKMeans,
    'DBSCAN': OGSDBSCAN,
    'FeatureAgglomeration': OGSFeatureAgglomeration,
    'HDBSCAN': OGSHDBSCAN,
    'KMeans': OGSKMeans,
    'MeanShift': OGSMeanShift,
    'MiniBatchKMeans': OGSMiniBatchKMeans,
    'OPTICS': OGSOPTICS,
    'Spectral': OGSSpectralClustering,
  }

# =============================================================================
# Clustering Zoo
# =============================================================================

class OGSClusteringZoo:
  """
  A factory class to create clustering algorithm instances.

  Example
  -------
  zoo = OGSClusteringZoo()
  kmeans = zoo.create("KMeans", n_clusters=5)
  hdbscan = zoo.create("HDBSCAN", min_cluster_size=10)

  """

  def __init__(self,
               metadata: dict[str, Any] = {},
               verbose: bool = False) -> None:
    self._metadata = metadata or {}
    self.verbose = verbose
    CLUSTERS = get_all_clusterers()
    self._algorithms: dict[str, type[BaseClusterer]] = {
      name: CLUSTERS[name] for name in self.metadata_algorithms
    } if self.metadata_algorithms else CLUSTERS
    METRICS = {k: v for group in get_all_eval_metrics().values()
               for k, v in group.items()}
    self._metrics: dict[str, type[BaseClusteringScores]] = {
      n: METRICS[n] for n in self.metadata_eval_metrics
    } if self.metadata_eval_metrics else {}

  @property
  def metadata_algorithms(self) -> list:
    return self._metadata.get("algorithms", [])

  @property
  def metadata_metric(self) -> Optional[Optional[str]]:
    return self._metadata.get("metric", None)

  @property
  def metadata_eval_metrics(self) -> list:
    return self._metadata.get("eval_metrics", [])

  @property
  def metadata_n_jobs_value(self) -> int:
    return int(self._metadata.get("n_jobs", -1))

  @property
  def metadata_min_cluster_size_value(self) -> Optional[int]:
    value = self._metadata.get("min_cluster_size")
    return int(value) if value is not None else None

  @property
  def metadata_min_samples_value(self) -> Optional[int]:
    value = self._metadata.get("min_samples")
    return int(value) if value is not None else None

  @property
  def metadata_random_state_value(self) -> Optional[int]:
    value = self._metadata.get("random_state")
    return int(value) if value is not None else None

  @property
  def metadata_n_clusters_value(self) -> Optional[int]:
    value = self._metadata.get("n_clusters")
    return int(value) if value is not None else None

  @property
  def metadata_eps_value(self) -> Optional[float]:
    value = self._metadata.get("eps")
    return float(value) if value is not None else None

  @property
  def metadata_damping_value(self) -> Optional[float]:
    value = self._metadata.get("damping")
    return float(value) if value is not None else None

  @property
  def metadata_bandwidth_value(self) -> Optional[float]:
    value = self._metadata.get("bandwidth")
    return float(value) if value is not None else None

  @property
  def metadata_num_clusters_value(self) -> Optional[int]:
    value = self._metadata.get("num_clusters")
    return int(value) if value is not None else None

  @property
  def metadata_Z_value(self) -> Optional[float]:
    value = self._metadata.get("Z")
    return float(value) if value is not None else None

  @property
  def metadata_cluster_size_range(self) -> Optional[Tuple[int, int, int]]:
    value = self._metadata.get("cluster_size_range")
    return tuple(value) if value is not None else None

  @property
  def metadata_bandwidth_range(self) -> Optional[Tuple[float, float, float]]:
    value = self._metadata.get("bandwidth_range")
    return tuple(value) if value is not None else None

  @property
  def metadata_damping_range(self) -> Optional[Tuple[float, float, float]]:
    value = self._metadata.get("damping_range")
    return tuple(value) if value is not None else None

  @property
  def metadata_eps_range(self) -> Optional[Tuple[float, float, float]]:
    value = self._metadata.get("eps_range")
    return tuple(value) if value is not None else None

  @property
  def metadata_min_samples_range(self) -> Optional[Tuple[int, int, int]]:
    value = self._metadata.get("min_samples_range")
    return tuple(value) if value is not None else None

  @property
  def metadata_num_clusters_range(self) -> Optional[Tuple[int, int, int]]:
    value = self._metadata.get("num_clusters_range")
    return tuple(value) if value is not None else None

  @property
  def metadata_sample_size_range(self) -> Optional[Tuple[int, int, int]]:
    value = self._metadata.get("sample_size_range")
    return tuple(value) if value is not None else None

  @property
  def metadata_Z_range(self) -> Optional[Tuple[float, float, float]]:
    value = self._metadata.get("Z_range")
    return tuple(value) if value is not None else None

  @property
  def list(self) -> list[str]:
    """Return a sorted list of available clustering keys."""
    return sorted(get_all_clusterers().keys())

  def register(self, name: str, cls: type) -> None:
    """Register a new clustering class under a name."""
    if name in self._algorithms:
      raise ValueError(f"Clusterer '{name}' already registered.")
    self._algorithms[name] = cls

  def create(self, name: str, **kwargs) -> BaseClusterer:
    """Create a clusterer instance by name."""
    if name not in self._algorithms:
      raise KeyError(f"Unknown clusterer '{name}'.")
    cluster_cls: type[BaseClusterer] = self._algorithms[name]
    cluster_kwargs = self._cluster_kwargs(name)
    cluster_kwargs.update(kwargs)
    return cluster_cls(**cluster_kwargs)

  def _cluster_kwargs(self, algo_name: str) -> dict:
    myDict = {}
    if self.metadata_bandwidth_value is not None and algo_name in {
      "MeanShift"
    }: myDict["bandwidth"] = self.metadata_bandwidth_value
    if self.metadata_damping_value is not None and algo_name in {
      "AffinityPropagation"
    }: myDict["damping"] = self.metadata_damping_value
    if self.metadata_eps_value is not None and algo_name in {"DBSCAN"}:
      myDict["eps"] = self.metadata_eps_value
    if self.metadata_metric not in {None, ""} and algo_name in {
      "Agglomerative",
      "FeatureAgglomeration",
      "DBSCAN",
      "OPTICS",
      "HDBSCAN"
    }: myDict["metric"] = self.metadata_metric
    if self.metadata_min_cluster_size_value is not None and algo_name in {
      "HDBSCAN"
    }: myDict["min_cluster_size"] = self.metadata_min_cluster_size_value
    if self.metadata_min_samples_value is not None and algo_name in {
      "OPTICS",
      "DBSCAN",
      "HDBSCAN"
    }: myDict["min_samples"] = self.metadata_min_samples_value
    if self.metadata_num_clusters_value is not None and algo_name in {
      "KMeans",
      "MiniBatchKMeans",
      "BisectingKMeans",
      "Agglomerative",
      "FeatureAgglomeration",
      "Spectral",
      "Birch",
    }: myDict["n_clusters"] = self.metadata_num_clusters_value
    if self.metadata_n_jobs_value is not None and algo_name in {
      "DBSCAN",
      "OPTICS",
      "Spectral",
      "HDBSCAN"
    }: myDict["n_jobs"] = self.metadata_n_jobs_value
    if self.metadata_random_state_value is not None and algo_name in {
      "KMeans",
      "MiniBatchKMeans",
      "BisectingKMeans",
      "Spectral",
      "AffinityPropagation",
    }: myDict["random_state"] = self.metadata_random_state_value
    if self.metadata_Z_value is not None and algo_name in {
      "AdvancedDensityPeaks"
    }: myDict["Z"] = self.metadata_Z_value
    return myDict

  def _optimize_param(self,
        param_name: str,
        algo_name: str,
        X: np.ndarray,
        metric_name: str,
        values: List[Any],
        base_kwargs: dict
      ) -> dict[str, Any]:
    """
    Optimize a single clustering parameter based on a metric.

    Parameters
    ----------
    param_name : str
      The name of the parameter to optimize.
    cluster_cls : type[BaseClusterer]
      The clustering class to use.
    X : np.ndarray
      The data to be clustered.
    metric_fn : Callable[[np.ndarray, np.ndarray], Optional[float]]
      The metric function to evaluate clustering quality.
    values : List[Any]
      The list of parameter values to test.
    base_kwargs : dict
      Base keyword arguments for the clustering class.

    Returns
    -------
    dict[str, Any]
      A dictionary containing the best parameter value, score, scores by
      parameter, and labels.
    """
    scores: Dict[Any, Optional[float]] = {}
    best_val: Optional[Any] = None
    best_score: Optional[float] = None
    cluster_cls: type[BaseClusterer] = self._algorithms[algo_name]
    for val in values:
      score = self._metrics[metric_name](
        X, cluster_cls(**base_kwargs, **{param_name: val},
                       verbose=self.verbose).fit_predict(X)
      ).compute()
      scores[val] = score
      if score is None: continue
      if best_score is None or (score <= best_score if metric_name in {
        "DaviesBouldinScore"
      } else score >= best_score): best_score, best_val = score, val
    if best_val is None: return {}

    best_clusterer = cluster_cls(**{**base_kwargs, param_name: best_val},
                                 verbose=self.verbose)
    best_clusterer.fit_predict(X)
    return {
      param_name: best_val,
      "clusterer": best_clusterer,
      "score": best_score,
      "scores_by_param": scores,
      "labels": best_clusterer.labels_
    }

  def _optimize_for_metric(self,
                           algo_name: str,
                           X: np.ndarray,
                           metric_name: str) -> dict[str, Any]:
    """
    Optimize clustering parameters for a given evaluation metric.

    Parameters
    ----------
    algo_name : str
      The name of the clustering algorithm.
    X : np.ndarray
      The data to be clustered.
    metric_name : str
      The name of the metric to optimize.

    Returns
    -------
    dict[str, Any]
      A dictionary containing the best parameters and metadata about the 
      optimization.
    """
    base_kwargs = self._cluster_kwargs(algo_name)
    params: dict = {}
    param_name = ""
    values: List[Any] = []
    if algo_name in {"HDBSCAN"}:
      values = iter_range(self.metadata_cluster_size_range)
      if values: param_name = "min_cluster_size"
    elif algo_name in {"KMeans", "MiniBatchKMeans", "BisectingKMeans",
                       "Agglomerative", "FeatureAgglomeration",
                       "Spectral", "Birch"}:
      values = iter_range(self.metadata_num_clusters_range)
      if values: param_name = "n_clusters"
    elif algo_name in {"DBSCAN"}:
      values = iter_range(self.metadata_eps_range)
      if values: param_name = "eps"
    elif algo_name in {"AffinityPropagation"}:
      values = iter_range(self.metadata_damping_range)
      if values: param_name = "damping"
    elif algo_name in {"MeanShift"}:
      values = iter_range(self.metadata_bandwidth_range)
      if values: param_name = "bandwidth"
    elif algo_name in {"OPTICS"}:
      values = iter_range(self.metadata_min_samples_range)
      if values: param_name = "min_samples"
    elif algo_name in {"AdvancedDensityPeaks"}:
      values = iter_range(self.metadata_Z_range)
      if values: param_name = "Z"
    params = self._optimize_param(
      param_name,
      algo_name,
      X,
      metric_name,
      values,
      base_kwargs
    )
    if self.verbose:
      print(f"Optimized {metric_name} for {algo_name}: {param_name} = "
            f"{params.get(param_name)} with score {params.get('score')}")
      for key, value in params["scores_by_param"].items():
        print(f"    {key}: {value}")
    params = {
      "algorithm": algo_name,
      "eval_metric": metric_name,
      **base_kwargs, **params
    }
    return params

  def _init_figure(self,
                   figsize: Tuple[int, int] = (16, 12),
                   **kwargs) -> dict[str, Tuple[Figure, np.ndarray]]:
    n_clusterers = len(self._algorithms)
    cols = min(4, n_clusterers)
    rows = (n_clusterers + cols - 1) // cols

    def build_figure(title: str) -> Tuple[Figure, np.ndarray]:
      fig, axes = plt.subplots(rows, cols, figsize=figsize, **kwargs)
      axes = np.atleast_2d(axes)
      fig.suptitle(title, fontsize=16)
      for idx, cluster_name in enumerate(self._algorithms.keys()):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        ax.set_title(cluster_name)
      # Hide unused subplots
      for idx in range(n_clusterers, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].set_visible(False)
      return fig, axes

    figures: dict[str, Tuple[Figure, np.ndarray]] = {}
    if self._metrics:
      for metric_name in self._metrics:
        figures[metric_name] = build_figure(
          f"Clustering Algorithm Comparison ({metric_name})"
        )
      return figures
    else: return {"": build_figure("Clustering Algorithm Comparison")}

  @abstractmethod
  def run(self,
          X: Optional[np.ndarray] = None,
          figsize: Tuple[int, int] = (16, 8),
          feature_x: int = 0,
          feature_y: int = 1,
          y_true: Optional[np.ndarray] = None,
          **common_kwargs) -> None:
    """
    Compare multiple clustering algorithms on the same dataset.

    Parameters
    ----------
    X : np.ndarray
      Data to cluster.
    configs : dict[str, dict], optional
      Mapping of name -> kwargs for clusterer creation.
    figsize : tuple, default=(16, 8)
      Figure size.
    feature_x : int, default=0
      Feature index for x-axis.
    feature_y : int, default=1
      Feature index for y-axis.
    y_true : np.ndarray, optional
      Ground-truth labels for supervised metrics.
    **common_kwargs
      Common arguments passed to all plot() methods.

    Returns
    -------
    """
    if X is None:
      raise ValueError("X must be provided for clustering comparison.")

    # ----- Computation loop -----
    DATA: dict[str, dict[str, Any]] = {}
    for algo_name in self._algorithms:
      metric_name = ""
      if self._metrics:
        for metric_name in self._metrics:
          params = self._optimize_for_metric(
            algo_name,
            X,
            metric_name
          )
          DATA.setdefault(metric_name, {})[algo_name] = params
      else:
        clusterer: BaseClusterer = self.create(
          algo_name,
          **self._metadata.get(algo_name, {}),
          **self._cluster_kwargs(algo_name)
        )
        params: dict[str, Any] = {
          "algorithm": algo_name,
          "clusterer": clusterer,
          "eval_metric": metric_name,
          "labels": clusterer.fit_predict(X),
          **self._cluster_kwargs(algo_name)
        }
        DATA.setdefault(metric_name, {})[algo_name] = params

    # ----- Plotting loop -----
    for metric_name, (fig, axes) in self._init_figure(figsize=figsize).items():
      for idx, algo_name in enumerate(self._algorithms.keys()):
        row, col = divmod(idx, axes.shape[1])
        ax = axes[row, col]
        params = DATA[metric_name][algo_name]
        if params is None: continue
        clusterer: BaseClusterer = params["clusterer"]
        clusterer.plot(
          X=X,
          feature_x=feature_x,
          feature_y=feature_y,
          ax=ax,
          **common_kwargs
        )

  def _finalize_figure(self,
                       fig: Figure,
                       ax: np.ndarray,
                       **kwargs) -> None:
    plt.tight_layout()

def main():
  # Demo usage
  from sklearn.datasets import make_blobs

  # Generate sample data
  X, y_true = make_blobs(n_samples=300,
                         centers=4,
                         cluster_std=0.60,
                         random_state=42,
                         return_centers=False)
  metadata = {
    "algorithms": [
      "AdvancedDensityPeaks",
      "HDBSCAN",
    ],
    "metric": "euclidean",
    "eval_metrics": [
      'SilhouetteScore',
      'DaviesBouldinScore',
    ],
    "n_jobs": -1,
    "random_state": 42,
    "bandwidth_range": (0.5, 2.0, 0.1),
    "cluster_size_range": (10, 100, 10),
    "damping_range": (0.5, 0.9, 0.1),
    "eps_range": (0.3, 1.0, 0.1),
    "min_samples_range": (5, 50, 5),
    "num_clusters_range": (2, 10, 1),
    "sample_size_range": (100, 300, 20),
    "Z_range": (0.1, 2.0, 0.1),
  }
  # Compare multiple algorithms
  zoo = OGSClusteringZoo(metadata=metadata, verbose=True)
  zoo.run(X)
  plt.show()

if __name__ == "__main__": main()

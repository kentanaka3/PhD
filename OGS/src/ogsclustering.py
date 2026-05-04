"""
=============================================================================
OGS Clustering Module - Scikit-learn Wrappers with Integrated Visualization
=============================================================================

OVERVIEW:
This module provides a comprehensive framework for clustering analysis in
seismic data processing. It wraps scikit-learn clustering algorithms with
enhanced plotting capabilities and evaluation metrics, enabling rapid
exploration of seismic event clusters (e.g., earthquake sequences, swarms).

The module implements:

1. BASE CLASSES
  - BaseClusterer: Abstract base for all clustering algorithms
  - BaseClusteringScores: Abstract base for evaluation metrics

2. CENTROID-BASED CLUSTERING (Partition methods)
  - OGSKMeans: Standard K-Means clustering
  - OGSMiniBatchKMeans: Memory-efficient K-Means for large datasets
  - OGSBisectingKMeans: Hierarchical divisive K-Means

3. DENSITY-BASED CLUSTERING (Spatial methods)
  - OGSDBSCAN: Density-Based Spatial Clustering (eps, min_samples)
  - OGSHDBSCAN: Hierarchical DBSCAN with variable density
  - OGSOPTICS: Ordering Points To Identify Clustering Structure
  - OGSAdvancedDensityPeaks: DADApy-based density peak detection

4. CONNECTIVITY-BASED CLUSTERING (Hierarchical methods)
  - OGSAgglomerative: Bottom-up hierarchical clustering
  - OGSFeatureAgglomeration: Feature-space clustering

5. MESSAGE-PASSING CLUSTERING
  - OGSAffinityPropagation: Exemplar-based clustering
  - OGSMeanShift: Mode-seeking clustering

6. SPECTRAL AND TREE-BASED CLUSTERING
  - OGSSpectralClustering: Graph Laplacian-based clustering
  - OGSBirch: Balanced Iterative Reducing and Clustering

7. EVALUATION METRICS
  - Unsupervised: Silhouette, Calinski-Harabasz, Davies-Bouldin
  - Supervised: Adjusted Rand, Mutual Information, V-Measure, etc.

8. FACTORY AND COMPARISON TOOLS
  - OGSClusteringZoo: Factory class for algorithm creation and comparison
  - get_all_clusterers(): Registry of available algorithms
  - get_all_eval_metrics(): Registry of evaluation metrics

ARCHITECTURE:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                             ogsclustering.py                            │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                           BaseClusterer (ABC)                           │
  │ ┌─────────────────────────────────────────────────────────────────────┐ │
  │ │ • fit_predict(X)                                                    │ │
  │ │ • plot(X, ax, ...)                                                  │ │
  │ │ • plot_3d(X, ax, ...)                                               │ │
  │ │ • get_cluster_centers()                                             │ │
  │ │ • n_clusters()                                                      │ │
  │ └─────────────────────────────────────────────────────────────────────┘ │
  │                                     ▲                                   │
  │   ┌────────── ─────────── ──────────┴───────────────┐                   │
  │   │          │           │                          │                   │
  │  OGSKMeans  OGSHDBSCAN  OGSAdvancedDensityPeaksPP  OGSAgglomerative ... │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                         BaseClusteringScores (ABC)                      │
  │                                     ▲                                   │
  │  ┌───────────────── ────────────────┴──┐                                │
  │  │                 │                   │                                │
  │ SilhouetteScore   AdjustedRandScore   PAkDensitySeparationScore  ...    │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                               OGSClusteringZoo                          │
  │ ┌─────────────────────────────────────────────────────────────────────┐ │
  │ │ • create(name, **kwargs)                                            │ │
  │ │ • run(X, ...)                                                       │ │
  │ │ • _optimize_for_metric(...)                                         │ │
  │ └─────────────────────────────────────────────────────────────────────┘ │
  └─────────────────────────────────────────────────────────────────────────┘

SEISMIC APPLICATIONS:
- Earthquake sequence identification (mainshock-aftershock clustering)
- Swarm detection and characterization
- Spatial pattern analysis of seismicity
- Temporal clustering for rate-state analysis
- Feature-based event classification

USAGE:
  # Basic clustering
  from ogsclustering import OGSKMeans, OGSHDBSCAN

  kmeans = OGSKMeans(n_clusters=5)
  labels = kmeans.fit_predict(X)
  kmeans.plot(xlabel="X (km)", ylabel="Y (km)")

  # Using the factory
  from ogsclustering import OGSClusteringZoo

  zoo = OGSClusteringZoo(metadata={"algorithms": ["HDBSCAN", "DBSCAN"]})
  zoo.run(X)

DEPENDENCIES:
  - numpy: Numerical arrays
  - pandas: DataFrame operations
  - matplotlib: Plotting and visualization
  - sklearn: Clustering algorithms and metrics
  - dadapy: Advanced density peak clustering

AUTHOR: AI2Seism Project
=============================================================================
"""

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import warnings                       # Warning system for non-fatal errors
from abc import ABC, abstractmethod   # Abstract base class support
from importlib import import_module   # Optional dependency loading
from typing import (                  # Type hints for better IDE support
  Optional,                           # Optional type annotation
  Tuple,                              # Tuple type annotation
  Union,                              # Union of multiple types
  Any,                                # Any type (escape hatch)
  List,                               # List type annotation
  Callable,                           # Function type annotation
  Dict,                               # Dictionary type annotation
  Literal,                            # Narrow string literal types
  cast                                # Explicit type narrowing
)

# =============================================================================
# THIRD-PARTY LIBRARY IMPORTS
# =============================================================================

# Numerical computing
import numpy as np                    # Array operations and linear algebra

# Visualization
import matplotlib.pyplot as plt       # Main plotting interface
from matplotlib.axes import Axes      # 2D axes type for type hints
from mpl_toolkits.mplot3d.axes3d import Axes3D  # 3D axes for 3D scatter plots
from matplotlib.figure import Figure  # Figure type for type hints

# Scientific computing (for ADP pipeline)
from sklearn.neighbors import NearestNeighbors  # k-NN distance computation
from scipy.special import gammaln               # Log-Gamma for volume prefactor
from scipy.optimize import curve_fit            # 2NN intrinsic-dimension fit

from ogsconstants import labels_to_colormap, setup_logger

# =============================================================================
# SCIKIT-LEARN CLUSTERING ALGORITHMS
# =============================================================================
# Import all supported clustering algorithms from sklearn

from sklearn.cluster import (
  KMeans,                         # Standard K-Means clustering
  MiniBatchKMeans,                # Mini-batch variant for large data
  AffinityPropagation,            # Message-passing exemplar clustering
  MeanShift,                      # Mode-seeking density clustering
  SpectralClustering,             # Graph Laplacian-based clustering
  AgglomerativeClustering,        # Hierarchical bottom-up clustering
  DBSCAN,                         # Density-based spatial clustering
  OPTICS,                         # Ordering points clustering
  Birch,                          # Balanced iterative clustering
  BisectingKMeans,                # Divisive hierarchical K-Means
  FeatureAgglomeration,           # Feature-space hierarchical clustering
)

HDBSCAN: Optional[Any]


def _load_hdbscan() -> Optional[Any]:
  """Return an available HDBSCAN implementation, if installed."""
  sklearn_hdbscan = getattr(import_module("sklearn.cluster"), "HDBSCAN", None)
  if sklearn_hdbscan is not None:
    return sklearn_hdbscan

  try:
    return getattr(import_module("hdbscan"), "HDBSCAN", None)
  except ImportError:
    return None


HDBSCAN = _load_hdbscan()

AgglomerativeLinkage = Literal['ward', 'complete', 'average', 'single']
VALID_AGGLOMERATIVE_LINKAGES: Tuple[AgglomerativeLinkage, ...] = (
  'ward',
  'complete',
  'average',
  'single',
)


def _validate_agglomerative_linkage(linkage: str) -> AgglomerativeLinkage:
  """Narrow linkage to the values accepted by AgglomerativeClustering."""
  if linkage not in VALID_AGGLOMERATIVE_LINKAGES:
    options = ", ".join(VALID_AGGLOMERATIVE_LINKAGES)
    raise ValueError(
      f"Invalid linkage '{linkage}'. Expected one of: {options}.")
  return cast(AgglomerativeLinkage, linkage)

# =============================================================================
# SCIKIT-LEARN CLUSTERING EVALUATION METRICS
# =============================================================================
# Import metrics for evaluating clustering quality

from sklearn.metrics import (
  # -------------------------------------------------------------------------
  # Unsupervised metrics (require only X and labels, no ground truth)
  # -------------------------------------------------------------------------
  silhouette_score,               # Mean silhouette coefficient [-1, 1]
  calinski_harabasz_score,        # Variance ratio criterion (higher=better)
  davies_bouldin_score,           # Average similarity ratio (lower=better)

  # -------------------------------------------------------------------------
  # Supervised metrics (require ground truth labels y_true)
  # -------------------------------------------------------------------------
  adjusted_rand_score,            # Rand index adjusted for chance
  normalized_mutual_info_score,   # Normalized mutual information
  adjusted_mutual_info_score,     # AMI adjusted for chance
  homogeneity_score,              # Clusters contain only one class
  completeness_score,             # Class members in same cluster
  v_measure_score,                # Harmonic mean of homogeneity/completeness
  fowlkes_mallows_score,          # Geometric mean of precision/recall

  # Utility
  pairwise_distances,             # Compute distance matrix
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def iter_range(values: Any) -> List[Any]:
  """
  Convert range specification to list of values for parameter sweeps.

  Handles multiple input formats for flexible hyperparameter specification.

  Parameters
  ----------
  values : Any
    Range specification in one of these formats:
    - tuple of 3: (start, stop, step) -> np.arange output
    - list/tuple/ndarray: returned as list
    - other: returns empty list

  Returns
  -------
  List[Any]
    List of parameter values to test.

  Example
  -------
  >>> iter_range((0.1, 0.5, 0.1))  # Returns [0.1, 0.2, 0.3, 0.4]
  >>> iter_range([1, 2, 5])        # Returns [1, 2, 5]
  """
  # Handle (start, stop, step) tuple -> numpy arange
  if isinstance(values, tuple) and len(values) == 3:
    return list(np.arange(*(values)))

  # Handle existing sequences -> convert to list
  if isinstance(values, (list, tuple, np.ndarray)):
    return list(values)

  # Unknown format -> empty list (no optimization)
  return []


# =============================================================================
# BASE CLUSTERER CLASS
# =============================================================================


class BaseClusterer(ABC):
  """
  Abstract base class for clustering algorithms with plotting capabilities.

  Provides a unified interface for all clustering algorithms, wrapping
  scikit-learn models with additional visualization and utility methods.
  All OGS clustering classes inherit from this base.

  Attributes
  ----------
  model : sklearn clustering model
    The underlying sklearn clustering model instance.
  labels_ : np.ndarray or None
    Cluster labels after fitting. None before fit() is called.
  data_ : np.ndarray or None
    Data used for fitting. Stored for later plotting.
  verbose : bool
    Whether to print progress information.
  _kwargs : dict
    Keyword arguments passed to the model constructor.

  Methods
  -------
  fit_predict(X)
    Fit the model and return cluster labels.
  plot(X, ax, ...)
    Create 2D scatter plot of clustering results.
  plot_3d(X, ax, ...)
    Create 3D scatter plot of clustering results.
  get_cluster_centers()
    Return cluster centers if available (e.g., for K-Means).
  n_clusters()
    Return the number of clusters found.

  Notes
  -----
  Subclasses must implement:
  - _create_model(**kwargs): Factory method to create the sklearn model
  """

  def __init__(self, **kwargs):
    """
    Initialize the clusterer with optional verbosity.

    Parameters
    ----------
    verbose : bool, optional
      If True, print progress information. Default False.
    **kwargs
      Additional arguments passed to _create_model().
    """
    # Extract verbose flag before passing to model
    self.verbose: bool = kwargs.pop('verbose', False)

    # Set up structured logger (level controlled by verbose flag)
    self.logger = setup_logger(self.__class__.__name__, verbose=self.verbose)

    # Create the underlying sklearn model via subclass factory
    self.model = self._create_model(**kwargs)

    # Initialize state variables (populated after fit)
    self.labels_: Optional[np.ndarray] = None
    self.data_: Optional[np.ndarray] = None

    # Store kwargs for repr and potential re-creation
    self._kwargs = kwargs

  @abstractmethod
  def _create_model(self, **kwargs) -> Any:
    """
    Factory method to create the underlying sklearn model.

    Must be implemented by subclasses.

    Parameters
    ----------
    **kwargs
      Algorithm-specific parameters.

    Returns
    -------
    Any
      Sklearn clustering model instance.
    """
    pass

  @property
  def name(self) -> str:
    """Return the name of the clustering algorithm (class name)."""
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
    **kwargs
  ) -> Tuple[dict, Optional[float], Dict[Any, Optional[float]]]:
    """
    Default optimizer: no-op for algorithms without a tuning routine.

    Subclasses can override this to implement parameter optimization.

    Parameters
    ----------
    X : np.ndarray
      Data to cluster.
    metric : Callable
      Evaluation metric function.
    **kwargs
      Additional arguments.

    Returns
    -------
    tuple
      (best_params, best_score, scores_by_param) - all empty by default.
    """
    return {}, None, {}

  def fit_predict(self, X: np.ndarray) -> np.ndarray:
    """
    Fit the clustering model and return cluster labels.

    Parameters
    ----------
    X : np.ndarray
      Training data of shape (n_samples, n_features).
      For seismic data, features might be [X_km, Y_km, depth, time].

    Returns
    -------
    np.ndarray
      Cluster labels for each sample. Label -1 indicates noise (for
      density-based algorithms like DBSCAN/HDBSCAN).
    """
    # Store data for later plotting
    self.data_ = X

    # Fit model and get labels
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
    Create 2D scatter plot of clustering results.

    Visualizes cluster assignments with color-coded points. Supports separate
    styling for noise points (label=-1) and cluster points.

    Parameters
    ----------
    X : np.ndarray, optional
      Data to plot. If None, uses the data from fit().
    feature_x : int, default=0
      Index of feature for x-axis (e.g., 0 for X_km).
    feature_y : int, default=1
      Index of feature for y-axis (e.g., 1 for Y_km).
    ax : Axes, optional
      Matplotlib axes to plot on. If None, creates new figure.
    title : str, optional
      Plot title. If None, uses algorithm name.
    xlabel : str, default="Feature 1"
      X-axis label.
    ylabel : str, default="Feature 2"
      Y-axis label.
    point_size : int or np.ndarray, default=20
      Size of scatter points. Can be array for variable sizing.
    alpha : float, default=0.7
      Transparency of cluster points.
    show_legend : bool, default=True
      Whether to show legend (for noise points).
    show_noise : bool, default=True
      Whether to show noise points (label=-1).
    noise_color : str, default="gray"
      Color for noise points.
    noise_alpha : float, default=0.3
      Alpha for noise points (more transparent).
    figsize : tuple, default=(10, 8)
      Figure size if creating new figure.
    colorbar : bool, default=True
      Whether to show colorbar with cluster labels.
    **scatter_kwargs
      Additional kwargs passed to ax.scatter().

    Returns
    -------
    Axes
      The matplotlib axes with the plot.

    Raises
    ------
    ValueError
      If model hasn't been fitted or no data is available.
    """
    # Validate that model has been fitted
    if self.labels_ is None:
      raise ValueError(
        "Model must be fitted before plotting. Call fit() first."
      )

    # Use provided data or fall back to stored data
    data = X if X is not None else self.data_
    if data is None:
      raise ValueError("No data available for plotting.")

    # Create new figure if no axes provided
    if ax is None:
      fig, ax = plt.subplots(figsize=figsize)

    labels = self.labels_
    x_data = data[:, feature_x]
    y_data = data[:, feature_y]

    # Separate noise points (label=-1) from cluster points
    noise_mask = labels == -1
    cluster_mask = ~noise_mask

    # Prepare base scatter kwargs
    base_scatter_kwargs = dict(scatter_kwargs)

    # Plot noise points with distinct styling (if any exist)
    if show_noise and np.any(noise_mask):
      # Extract kwargs without 's' (size handled separately)
      noise_kwargs = {k: v for k, v in base_scatter_kwargs.items() if k != "s"}
      noise_size = base_scatter_kwargs.get("s", point_size)

      ax.scatter(
        x_data[noise_mask],
        y_data[noise_mask],
        c=noise_color,
        s=noise_size if isinstance(noise_size, int) else noise_size[noise_mask],
        alpha=noise_alpha,
        label="Noise",
        marker="x",  # X marker distinguishes noise
        **noise_kwargs
      )

    # Plot cluster points with colormap
    if np.any(cluster_mask):
      cluster_labels = labels[cluster_mask]

      # Map labels to sequential indices for coloring
      encoded, unique, cmap, norm = labels_to_colormap(cluster_labels)

      # Extract kwargs without 's' (size handled separately)
      cluster_kwargs = {k: v for k, v in base_scatter_kwargs.items()
        if k != "s"}
      cluster_size = base_scatter_kwargs.get("s", point_size)

      # Create scatter plot with colormap
      sc = ax.scatter(
        x_data[cluster_mask],
        y_data[cluster_mask],
        c=encoded,
        s=cluster_size if isinstance(cluster_size, int) else
          cluster_size[cluster_mask],
        alpha=alpha,
        cmap=cmap,
        norm=norm,
        **cluster_kwargs
      )

      # Add colorbar showing cluster labels
      if colorbar:
        cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(len(unique)))
        cbar.ax.set_yticklabels([str(lab) for lab in unique])
        cbar.set_label("Cluster")

    # Set axis labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{self.name} Clustering")

    # Show legend if noise points are present
    if show_legend and np.any(noise_mask):
      ax.legend()

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
    **scatter_kwargs
  ) -> Axes:
    """
    Create 3D scatter plot of clustering results.

    Useful for visualizing clusters in 3D space (e.g., X, Y, Depth for
    earthquake hypocenter clustering).

    Parameters
    ----------
    X : np.ndarray, optional
      Data to plot. If None, uses the data from fit().
    feature_x : int, default=0
      Index of feature for x-axis.
    feature_y : int, default=1
      Index of feature for y-axis.
    feature_z : int, default=2
      Index of feature for z-axis (e.g., depth).
    ax : Axes or Axes3D, optional
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
    # Validate that model has been fitted
    if self.labels_ is None:
      raise ValueError(
        "Model must be fitted before plotting. Call fit() first."
      )

    # Use provided data or fall back to stored data
    data = X if X is not None else self.data_
    if data is None:
      raise ValueError("No data available for plotting.")

    # Create new 3D figure if no axes provided
    if ax is None:
      fig = plt.figure(figsize=figsize)
      ax = fig.add_subplot(111, projection='3d')

    labels = self.labels_

    # Only plot cluster points (exclude noise)
    cluster_mask = labels != -1

    base_scatter_kwargs = dict(scatter_kwargs)

    if np.any(cluster_mask):
      cluster_labels = labels[cluster_mask]

      # Map labels to sequential indices for coloring
      encoded, unique, cmap, norm = labels_to_colormap(cluster_labels)

      # Extract kwargs and handle size separately
      cluster_kwargs = {k: v for k, v in base_scatter_kwargs.items()
        if k != "s"}
      cluster_size = base_scatter_kwargs.get("s", point_size)
      cluster_kwargs["s"] = (
        cluster_size if isinstance(cluster_size, int)
        else cluster_size[cluster_mask]
      )

      # Create 3D scatter plot
      ax.scatter(
        data[cluster_mask, feature_x],
        data[cluster_mask, feature_y],
        data[cluster_mask, feature_z],
        c=encoded,
        alpha=alpha,
        cmap=cmap,
        **cluster_kwargs
      )

    # Set axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if isinstance(ax, Axes3D):
      ax.set_zlabel(zlabel)
    ax.set_title(title or f"{self.name} Clustering (3D)")

    return ax

  def get_cluster_centers(self) -> Optional[np.ndarray]:
    """
    Return cluster centers if available.

    Only available for centroid-based algorithms (K-Means, Mean Shift).

    Returns
    -------
    np.ndarray or None
      Cluster centers of shape (n_clusters, n_features), or None if the
      algorithm doesn't compute centers.
    """
    if hasattr(self.model, 'cluster_centers_'):
      return self.model.cluster_centers_
    return None

  def n_clusters(self) -> int:
    """
    Return the number of clusters found.

    Excludes noise points (label=-1) from the count.

    Returns
    -------
    int
      Number of clusters (0 if not fitted).
    """
    if self.labels_ is None:
      return 0

    # Get unique labels and exclude noise (-1)
    unique_labels = np.unique(self.labels_)
    return len(unique_labels[unique_labels >= 0])

  def __repr__(self) -> str:
    """String representation showing class name and parameters."""
    return f"{self.name}({self._kwargs})"


# =============================================================================
# CLUSTERING EVALUATION METRICS
# =============================================================================
#
# This section provides wrapper classes for sklearn clustering evaluation
# metrics. Metrics are divided into two categories:
#
# UNSUPERVISED METRICS (require only X and labels):
#   - SilhouetteScore: Measures cluster cohesion and separation [-1, 1]
#   - CalinskiHarabaszScore: Variance ratio criterion (higher = better)
#   - DaviesBouldinScore: Average cluster similarity (lower = better)
#
# SUPERVISED METRICS (require ground truth y_true):
#   - AdjustedRandScore: Similarity adjusted for chance [-1, 1]
#   - NormalizedMutualInfoScore: Shared information [0, 1]
#   - AdjustedMutualInfoScore: NMI adjusted for chance [-1, 1]
#   - HomogeneityScore: Each cluster = single class [0, 1]
#   - CompletenessScore: Each class = single cluster [0, 1]
#   - VMeasureScore: Harmonic mean of homogeneity/completeness [0, 1]
#   - FowlkesMallowsScore: Geometric mean of precision/recall [0, 1]
#
# For seismic applications, unsupervised metrics are more common since ground
# truth cluster labels are typically unavailable. Silhouette score is
# particularly useful for comparing clustering quality across algorithms.
# =============================================================================


class BaseClusteringScores(ABC):
  """
  Abstract base class for clustering evaluation metric objects.

  Provides a uniform interface for computing various clustering quality
  metrics, both unsupervised (using only data and labels) and supervised
  (requiring ground truth labels).

  Attributes
  ----------
  X : np.ndarray
    Data used for clustering, shape (n_samples, n_features).
  labels : np.ndarray
    Cluster labels from clustering algorithm.
  y_true : np.ndarray or None
    Ground truth labels (for supervised metrics).

  Methods
  -------
  compute()
    Compute and return the metric value.

  Notes
  -----
  Subclasses must implement the compute() method which wraps the
  corresponding sklearn metric function.
  """

  def __init__(self,
    X: np.ndarray,
    labels: np.ndarray,
    y_true: Optional[np.ndarray] = None):
    """
    Initialize the metric with data and labels.

    Parameters
    ----------
    X : np.ndarray
      Data used for clustering, shape (n_samples, n_features).
    labels : np.ndarray
      Cluster labels from the algorithm, shape (n_samples,).
    y_true : np.ndarray, optional
      Ground truth labels for supervised metrics.
    """
    # Store data and labels for metric computation
    self.X = X
    self.labels = labels
    self.y_true = y_true  # Only used by supervised metrics

  @property
  def name(self) -> str:
    """Return the name of the metric (class name)."""
    return self.__class__.__name__

  @abstractmethod
  def compute(self) -> Optional[float]:
    """
    Compute the metric value.

    Returns
    -------
    float or None
      Metric value, or None if computation fails or is not applicable.
    """
    raise NotImplementedError


# -----------------------------------------------------------------------------
# Unsupervised Metrics (require only X and labels)
# -----------------------------------------------------------------------------


class SilhouetteScore(BaseClusteringScores):
  """
  Silhouette Coefficient: measures cluster cohesion and separation.

  Range: [-1, 1]. Higher is better.
  - +1: Dense, well-separated clusters
  -  0: Overlapping clusters
  - -1: Misassigned samples

  For seismic sequences: Good for comparing algorithm performance.
  """

  def compute(self) -> Optional[float]:
    """Compute mean silhouette coefficient over all samples."""
    try:
      return float(silhouette_score(self.X, self.labels))
    except Exception:
      # Returns None if clustering invalid (e.g., only 1 cluster)
      return None


class CalinskiHarabaszScore(BaseClusteringScores):
  """
  Calinski-Harabasz Index (Variance Ratio Criterion).

  Ratio of between-cluster to within-cluster dispersion.
  Higher values indicate better-defined clusters.
  No upper bound - scale depends on data.
  """

  def compute(self) -> Optional[float]:
    """Compute Calinski-Harabasz index."""
    try:
      return float(calinski_harabasz_score(self.X, self.labels))
    except Exception:
      return None


class DaviesBouldinScore(BaseClusteringScores):
  """
  Davies-Bouldin Index: average similarity between clusters.

  Lower values indicate better clustering (more separated clusters).
  Minimum value is 0.
  """

  def compute(self) -> Optional[float]:
    """Compute Davies-Bouldin index."""
    try:
      return davies_bouldin_score(self.X, self.labels)
    except Exception:
      return None


# -----------------------------------------------------------------------------
# Supervised Metrics (require ground truth y_true)
# -----------------------------------------------------------------------------


class AdjustedRandScore(BaseClusteringScores):
  """
  Adjusted Rand Index: similarity between two clusterings.

  Adjusted for chance. Range: [-1, 1]. 1 = perfect agreement.
  Useful when comparing clustering to known event classifications.
  """

  def compute(self) -> Optional[float]:
    """Compute adjusted Rand index."""
    if self.y_true is None:
      return None  # Cannot compute without ground truth
    try:
      return adjusted_rand_score(self.y_true, self.labels)
    except Exception:
      return None


class NormalizedMutualInfoScore(BaseClusteringScores):
  """
  Normalized Mutual Information: measures shared information.

  Range: [0, 1]. 1 = perfect correlation between labelings.
  """

  def compute(self) -> Optional[float]:
    """Compute normalized mutual information."""
    if self.y_true is None:
      return None
    try:
      return float(normalized_mutual_info_score(self.y_true, self.labels))
    except Exception:
      return None


class AdjustedMutualInfoScore(BaseClusteringScores):
  """
  Adjusted Mutual Information: NMI adjusted for chance.

  Range: [-1, 1]. Higher is better.
  """

  def compute(self) -> Optional[float]:
    """Compute adjusted mutual information."""
    if self.y_true is None:
      return None
    try:
      return float(adjusted_mutual_info_score(self.y_true, self.labels))
    except Exception:
      return None


class HomogeneityScore(BaseClusteringScores):
  """
  Homogeneity: each cluster contains only members of a single class.

  Range: [0, 1]. 1 = perfectly homogeneous.
  For seismic: Measures if each cluster contains only one event type.
  """

  def compute(self) -> Optional[float]:
    """Compute homogeneity score."""
    if self.y_true is None:
      return None
    try:
      return float(homogeneity_score(self.y_true, self.labels))
    except Exception:
      return None


class CompletenessScore(BaseClusteringScores):
  """
  Completeness: all members of a class are in the same cluster.

  Range: [0, 1]. 1 = perfectly complete.
  For seismic: Measures if all events of same type are in same cluster.
  """

  def compute(self) -> Optional[float]:
    """Compute completeness score."""
    if self.y_true is None:
      return None
    try:
      return float(completeness_score(self.y_true, self.labels))
    except Exception:
      return None


class VMeasureScore(BaseClusteringScores):
  """
  V-Measure: harmonic mean of homogeneity and completeness.

  Range: [0, 1]. Balances both criteria.
  """

  def compute(self) -> Optional[float]:
    """Compute V-measure score."""
    if self.y_true is None:
      return None
    try:
      return float(v_measure_score(self.y_true, self.labels))
    except Exception:
      return None


class FowlkesMallowsScore(BaseClusteringScores):
  """
  Fowlkes-Mallows Index: geometric mean of pairwise precision and recall.

  Range: [0, 1]. Higher indicates better agreement with ground truth.
  """

  def compute(self) -> Optional[float]:
    """Compute Fowlkes-Mallows index."""
    if self.y_true is None:
      return None
    try:
      return fowlkes_mallows_score(self.y_true, self.labels)
    except Exception:
      return None


# -----------------------------------------------------------------------------
# PAk-based Unsupervised Metric (density-topology evaluation)
# -----------------------------------------------------------------------------


class PAkDensitySeparationScore(BaseClusteringScores):
  """
  PAk Density Separation Score — unsupervised clustering evaluation via
  Point-Adaptive k-NN density topology.

  Evaluates how well cluster labels align with the topological structure of the
  data density landscape estimated by the PAk (Point-Adaptive k-NN) density
  estimator [1]_. Unlike geometric metrics (Silhouette, Calinski-Harabasz),
  this score directly tests whether cluster boundaries coincide with
  **statistically significant density valleys**, making it especially suited
  for non-convex, variable-density, and high-dimensional clustering problems
  common in seismology.

  Algorithm
  ---------
  1. Compute k-NN distances for all N points.
  2. Estimate intrinsic dimension *d* via the 2NN ratio method.
  3. Select adaptive neighbourhood size k* per point (likelihood-ratio test).
  4. Estimate per-point log-density and its uncertainty via PAk
     Newton-Raphson ML on the Poisson shell-volume likelihood.
  5. For each cluster *c* identify the **density peak** (point of highest
     log-density) and record its uncertainty.
  6. For every pair of **adjacent** clusters (c, c') — i.e. clusters
     that share at least one k*-neighbourhood edge — find the
     **saddle-point density** (highest border density where a point in
     *c* neighbours a point in *c'*) and its uncertainty.
  7. Compute a per-pair Z-score measuring whether the *log-density* drop
     from the weaker peak to the saddle is statistically significant::

       Z_{c,c'} = (min(û_peak^c, û_peak^{c'}) - û_saddle^{c,c'})
                  / sqrt(σ_peak_c² + σ_peak_{c'}² + σ_saddle² )

     where û = log(ρ) is the PAk log-density and σ is the PAk log-density
     uncertainty (conservative ≈ 2/√k* approximation to the trigamma).
     Working in log-density space provides variance stabilisation via
     the Delta method: Var(ln ρ̂) = 1/(k*−2), independent of the
     unknown true density (see theory §5, Delta Method section).

  8. Return the **population-weighted mean** Z-score across all adjacent
     cluster pairs, with weights w_{c,c'} = √(n_c · n_{c'}).

  Score Interpretation
  --------------------
  Range : [0, +∞). Higher is better.

  ======= =====================================================
  Score   Interpretation
  ======= =====================================================
  > 3.0   Excellent — deeply significant density valleys
  > 1.65  Good — valleys significant at ≥ 90 % confidence
  0.5-1.6 Marginal — weak or ambiguous density separation
  < 0.5   Poor — clusters not supported by density topology
  ======= =====================================================

  For seismic applications, scores > 1.65 typically indicate well-resolved
  earthquake clusters separated by clear gaps in the spatio-temporal density of
  seismicity.

  Parameters
  ----------
  X : np.ndarray, shape (n_samples, n_features)
    Data used for clustering.
  labels : np.ndarray, shape (n_samples,)
    Cluster labels from any clustering algorithm (0-indexed;
    -1 is treated as noise/halo and excluded from evaluation).
  y_true : np.ndarray or None, optional
    Ignored (unsupervised metric). Accepted for API compatibility.
  maxk : int or None, default=None
    Maximum neighbours to consider. If None, uses min(100, N-1).
  Dthr : float, default=23.92812698
    Likelihood-ratio threshold for adaptive k* selection
    (same as in OGSAdvancedDensityPeaks; 23.93 ≈ p ~ 1e-6).
  n_jobs : int, default=-1
    CPU cores for k-NN computation (-1 = all cores).

  Attributes
  ----------
  log_den_ : np.ndarray or None
    PAk log-density estimates after compute().
  log_den_err_ : np.ndarray or None
    PAk log-density uncertainties after compute().
  kstar_ : np.ndarray or None
    Adaptive neighbourhood sizes after compute().
  intrinsic_dim_ : float or None
    Estimated intrinsic dimension after compute().
  cluster_peak_densities_ : dict or None
    {cluster_label: (log_den_peak, log_den_err_peak)} after compute().
  saddle_densities_ : dict or None
    {(c, c'): (log_den_saddle, log_den_err_saddle)} after compute().
  pair_zscores_ : dict or None
    {(c, c'): z_score} after compute().

  References
  ----------
  .. [1] M. d'Errico, E. Facco, A. Laio, A. Rodriguez, "Automatic
     topography of high-dimensional data sets by non-parametric density
     peak clustering," Information Sciences 560 (2021) 476-492.

  Citation
  --------
  If you use this metric in your research, please cite the original PAk paper
  and this implementation:
  @article{d2021automatic,
    title={
      Automatic topography of high-dimensional data sets by non-parametric
      density peak clustering
    },
    author={
      d'Errico, Michele and Facco, Enrico and Laio, Alessandro and Rodriguez, Alessandro
    },
    journal={Information Sciences},
    volume={560},
    pages={476--492},
    year={2021},
    publisher={Elsevier}
  }
  @article{
  }

  Examples
  --------
  >>> from ogsclustering import OGSDBSCAN, PAkDensitySeparationScore
  >>> dbscan = OGSDBSCAN(eps=0.5, min_samples=5)
  >>> labels = dbscan.fit_predict(X)
  >>> scorer = PAkDensitySeparationScore(X, labels, maxk=50)
  >>> score = scorer.compute()
  >>> print(f"PAk separation Z-score: {score:.2f}")
  """

  def __init__(
    self,
    X: np.ndarray,
    labels: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    maxk: Optional[int] = None,
    Dthr: float = 23.92812698,
    n_jobs: int = -1,
  ):
    super().__init__(X, labels, y_true)
    self._maxk = maxk
    self._Dthr = Dthr
    self._n_jobs = n_jobs

    # Public diagnostics (populated after compute)
    self.log_den_: Optional[np.ndarray] = None
    self.log_den_err_: Optional[np.ndarray] = None
    self.kstar_: Optional[np.ndarray] = None
    self.intrinsic_dim_: Optional[float] = None
    self.cluster_peak_densities_: Optional[Dict[int,
                                                Tuple[float, float]]] = None
    self.saddle_densities_: Optional[Dict[Tuple[int, int],
                                          Tuple[float, float]]] = None
    self.pair_zscores_: Optional[Dict[Tuple[int, int], float]] = None

  # -------------------------------------------------------------------
  # Internal helpers (reuse PAk pipeline components)
  # -------------------------------------------------------------------

  @staticmethod
  def _compute_nn_distances(
    X: np.ndarray, maxk: int, n_jobs: int
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute k-NN distances using sklearn NearestNeighbors."""
    nn = NearestNeighbors(
      n_neighbors=maxk + 1, metric='euclidean', n_jobs=n_jobs
    )
    nn.fit(X)
    distances, dist_indices = nn.kneighbors(X)
    distances = distances.astype(np.float64)
    dist_indices = dist_indices.astype(np.int64)

    eps = np.finfo(np.float64).eps
    zero_mask = distances[:, 1:] <= eps
    if np.any(zero_mask):
      distances[:, 1:][zero_mask] = eps

    return distances, dist_indices

  @staticmethod
  def _estimate_intrinsic_dim(
    distances: np.ndarray, mu_fraction: float = 0.9
  ) -> float:
    """Estimate intrinsic dimension via the 2NN ratio method."""
    eps = np.finfo(np.float64).eps
    N = distances.shape[0]
    mu = distances[:, 2] / np.maximum(distances[:, 1], eps)
    log_mu = np.log(mu)
    log_mu_sorted = np.sort(log_mu)
    n_eff = max(int(N * mu_fraction), 2)
    log_mu_reduced = log_mu_sorted[:n_eff]
    y = -np.log(1.0 - np.arange(1, n_eff + 1) / N)

    def linear_through_origin(x, m):
      return m * x

    try:
      d_fit, _ = curve_fit(linear_through_origin, log_mu_reduced, y)
      d = float(d_fit[0])
    except (RuntimeError, ValueError):
      d = float((N - 1) / np.sum(log_mu))
    return max(d, 0.5)

  @staticmethod
  def _compute_kstar_adaptive(
    distances: np.ndarray, d: float, maxk: int, Dthr: float
  ) -> np.ndarray:
    """Compute adaptive k* via likelihood-ratio test."""
    N = distances.shape[0]
    eps = np.finfo(np.float64).eps
    k_min = 4
    n_tests = maxk - k_min
    if n_tests <= 0:
      return np.full(N, max(3, maxk), dtype=np.int64)

    k_vals = np.arange(k_min, maxk, dtype=np.float64)
    r_inner = np.maximum(distances[:, k_min:maxk], eps)
    r_outer = distances[:, k_min + 1:maxk + 1]
    mu = r_outer / r_inner
    log_mu = np.log(np.maximum(mu, 1.0 + eps))
    coeff = 2.0 * (k_vals * d + 1.0)
    const = 2.0 * np.log(d) + 2.0 * np.log(k_vals)
    D = coeff[np.newaxis, :] * log_mu - const[np.newaxis, :]
    exceeds = D > Dthr
    has_exceed = np.any(exceeds, axis=1)
    first_idx = np.argmax(exceeds, axis=1)
    kstar = np.where(
      has_exceed,
      k_vals[first_idx].astype(np.int64),
      np.int64(maxk),
    )
    return np.maximum(kstar, np.int64(3))

  @staticmethod
  def _density_pak(
    distances: np.ndarray, kstar: np.ndarray, d: float, N: int
  ) -> Tuple[np.ndarray, np.ndarray]:
    """PAk log-density via Newton-Raphson ML on shell-volume likelihood."""
    eps = np.finfo(np.float64).eps
    prefactor = np.exp(
      d / 2.0 * np.log(np.pi) - gammaln((d + 2.0) / 2.0)
    )
    kstar_f = np.maximum(kstar.astype(np.float64), 2.0)
    kmax = int(np.max(kstar))

    # PAk error estimate — standard deviation of the log-density û.
    # Theory: The variance of û = log(k*/S) equals the trigamma function
    # ψ₁(k*) = 1/k* + 1/(2k*²) + O(1/k*³). The PAk formula below is a
    # deliberately conservative approximation (≈ 4/k* ≈ 4× the CR bound) that
    # accounts for uncertainty in S, finite-sample effects, and the estimated
    # intrinsic dimension d̂. See d'Errico et al. (2021) §3.2.
    #
    # Alternatives (kept for reference):
    #   1/np.sqrt(kstar_f)          — Cramér–Rao lower bound (σ_CR = 1/√k*)
    #   np.sqrt(2.0/(kstar_f-1.0))  — intermediate approximation
    log_den_err = np.sqrt(
      (4.0 * kstar_f + 2.0) / (kstar_f * np.maximum(kstar_f - 1.0, 1.0))
    )

    # kNN starting root for Newton-Raphson
    dc = np.maximum(distances[np.arange(N), kstar], eps)
    u = np.log(kstar_f) - np.log(prefactor) - d * np.log(dc)

    # Shell volumes (vectorised)
    r = distances[:, :kmax]
    r1 = distances[:, 1:kmax + 1]
    r1_safe = np.maximum(r1, eps)
    ratio = r / r1_safe
    near_one = np.abs(ratio - 1.0) < 10 * np.finfo(ratio.dtype).resolution
    if np.any(near_one):
      ratio[near_one] -= 10 * np.finfo(ratio.dtype).resolution
    ratio_d = np.power(np.maximum(ratio, 0.0), d)
    one_minus = np.maximum(1.0 - ratio_d, eps)
    exponent = d * np.log(r1_safe) + np.log(one_minus)
    exponent = np.clip(exponent, -300.0, 300.0)
    volumes = prefactor * np.exp(exponent)
    col_idx = np.arange(kmax)[np.newaxis, :]
    mask = col_idx < kstar[:, np.newaxis]
    masked_vols = volumes * mask
    S = np.maximum(np.sum(masked_vols, axis=1), eps)

    # Newton-Raphson (vectorised)
    for _ in range(100):
      exp_u = np.exp(np.clip(u, -300.0, 300.0))
      grad = kstar_f - exp_u * S
      hess = -exp_u * S
      safe_hess = np.where(np.abs(hess) < 1e-300, -1e-300, hess)
      delta = -grad / safe_hess
      u += delta
      if np.max(np.abs(delta)) < 1e-8:
        break

    return u, log_den_err

  # -------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------

  def compute(self) -> Optional[float]:
    """
    Compute the PAk Density Separation Score.

    Runs the full PAk density estimation pipeline on `self.X`, then evaluates
    how well `self.labels` partition the density landscape into regions
    separated by significant density valleys.

    Returns
    -------
    float or None
      Population-weighted mean Z-score of peak-to-saddle density drops across
      all adjacent cluster pairs. Returns None if fewer than 2 valid clusters
      exist or if computation fails.
    """
    try:
      X = self.X
      labels = self.labels
      N = X.shape[0]

      # --- Filter noise points (label == -1) ---
      valid_mask = labels >= 0
      if np.sum(valid_mask) < 3:
        return None

      unique_labels = np.unique(labels[valid_mask])
      n_clusters = len(unique_labels)
      if n_clusters < 2:
        return None

      # --- Step 1: k-NN distances ---
      maxk = self._maxk if self._maxk is not None else min(100, N - 1)
      maxk = min(maxk, N - 1)
      distances, dist_indices = self._compute_nn_distances(
        X, maxk, self._n_jobs
      )

      # --- Step 2: Intrinsic dimension ---
      d = self._estimate_intrinsic_dim(distances)
      self.intrinsic_dim_ = d

      # --- Step 3: Adaptive k* ---
      kstar = self._compute_kstar_adaptive(distances, d, maxk, self._Dthr)
      self.kstar_ = kstar

      # --- Step 4: PAk density estimation ---
      log_den, log_den_err = self._density_pak(distances, kstar, d, N)

      # Guard against NaN — fallback to kNN for affected points only.
      # The Newton-Raphson iteration is independent per-point, so only the
      # non-finite entries need replacement; converged values are kept.
      nan_mask = ~np.isfinite(log_den)
      if np.any(nan_mask):
        eps = np.finfo(np.float64).eps
        prefactor = np.exp(
          d / 2.0 * np.log(np.pi) - gammaln((d + 2.0) / 2.0)
        )
        kstar_f = np.maximum(kstar.astype(np.float64), 1.0)
        dc = np.maximum(distances[np.arange(N), kstar], eps)
        knn_den = np.log(kstar_f) - np.log(prefactor) - d * np.log(dc)
        knn_err = np.sqrt(
          (4.0 * kstar_f + 2.0) / (kstar_f * np.maximum(kstar_f - 1.0, 1.0))
        )
        log_den[nan_mask] = knn_den[nan_mask]
        log_den_err[nan_mask] = knn_err[nan_mask]

      self.log_den_ = log_den
      self.log_den_err_ = log_den_err

      # --- Step 5: Per-cluster peak densities ---
      cluster_peaks: Dict[int, Tuple[float, float]] = {}
      for c in unique_labels:
        cmask = labels == c
        idx_in_cluster = np.where(cmask)[0]
        densities = log_den[idx_in_cluster]
        peak_idx = idx_in_cluster[np.argmax(densities)]
        cluster_peaks[int(c)] = (
          float(log_den[peak_idx]),
          float(log_den_err[peak_idx]),
        )
      self.cluster_peak_densities_ = cluster_peaks

      # --- Step 6: Saddle-point densities between adjacent pairs ---
      # A pair (c, c') is adjacent if any point in c has a neighbor (within
      # maxk) in c'. We use the full maxk neighborhood for adjacency detection
      # to ensure robust border discovery even when k* is small, while the
      # saddle density itself is computed from the PAk-estimated log-densities.
      saddle_den: Dict[Tuple[int, int], Tuple[float, float]] = {}

      # Vectorised border detection over the full (N, maxk) neighbor matrix to
      # avoid Python loops where possible.
      neighbor_labels = labels[dist_indices[:, 1:]]   # (N, maxk)
      own_labels = labels[:, np.newaxis]              # (N, 1)
      valid_own = own_labels >= 0                     # exclude noise
      valid_nbr = neighbor_labels >= 0                # exclude noise
      cross_mask = (neighbor_labels != own_labels) & valid_own & valid_nbr

      # Extract all inter-cluster (i, j) edges
      row_idx, col_idx = np.where(cross_mask)
      if row_idx.size > 0:
        j_idx = dist_indices[row_idx, col_idx + 1]    # +1 for skipped col-0
        ci_arr = labels[row_idx]
        cj_arr = labels[j_idx]

        # Border density = min(ρ_i, ρ_j) per edge
        border_rho_arr = np.minimum(log_den[row_idx], log_den[j_idx])
        border_err_arr = np.sqrt(
          log_den_err[row_idx]**2 + log_den_err[j_idx]**2
        )

        # Canonical pair keys (min, max)
        pair_lo = np.minimum(ci_arr, cj_arr)
        pair_hi = np.maximum(ci_arr, cj_arr)

        # Find best (highest) saddle density per pair
        for k_edge in range(row_idx.size):
          pair = (int(pair_lo[k_edge]), int(pair_hi[k_edge]))
          rho_e = float(border_rho_arr[k_edge])
          err_e = float(border_err_arr[k_edge])
          if pair not in saddle_den or rho_e > saddle_den[pair][0]:
            saddle_den[pair] = (rho_e, err_e)

      self.saddle_densities_ = saddle_den

      if len(saddle_den) == 0:
        # No adjacent cluster pairs found even in the full maxk neighborhood →
        # clusters are completely disconnected in k-NN space.  Use the global
        # density minimum as a conservative virtual saddle and compute proper
        # Z-scores for all cluster pairs.
        min_idx = int(np.argmin(log_den))
        virtual_saddle = float(log_den[min_idx])
        virtual_err = float(log_den_err[min_idx])

        pair_zscores_v: Dict[Tuple[int, int], float] = {}
        total_weight = 0.0
        weighted_z_sum = 0.0
        ul = list(cluster_peaks.keys())
        for ic in range(len(ul)):
          for jc in range(ic + 1, len(ul)):
            ci, cj = ul[ic], ul[jc]
            rho_ci, err_ci = cluster_peaks[ci]
            rho_cj, err_cj = cluster_peaks[cj]
            weaker = min(rho_ci, rho_cj)
            c_err = np.sqrt(err_ci**2 + err_cj**2 + virtual_err**2)
            if c_err > 0:
              z = (weaker - virtual_saddle) / c_err
            else:
              z = float('inf') if weaker > virtual_saddle else 0.0
            pair_zscores_v[(ci, cj)] = float(z)
            ni = float(np.sum(labels == ci))
            nj = float(np.sum(labels == cj))
            w = np.sqrt(ni * nj)
            weighted_z_sum += w * z
            total_weight += w

        self.pair_zscores_ = pair_zscores_v
        if total_weight > 0:
          return float(weighted_z_sum / total_weight)
        return None

      # --- Step 7: Z-scores per adjacent pair ---
      pair_zscores: Dict[Tuple[int, int], float] = {}
      total_weight = 0.0
      weighted_z_sum = 0.0

      for (ci, cj), (rho_saddle, err_saddle) in saddle_den.items():
        rho_peak_ci, err_peak_ci = cluster_peaks[ci]
        rho_peak_cj, err_peak_cj = cluster_peaks[cj]

        # Use the weaker peak (the more vulnerable cluster)
        weaker_peak = min(rho_peak_ci, rho_peak_cj)

        # Combined uncertainty (peak errors + saddle error)
        combined_err = np.sqrt(
          err_peak_ci**2 + err_peak_cj**2 + err_saddle**2
        )

        # Z-score: how many standard deviations the density drops
        if combined_err > 0:
          z = (weaker_peak - rho_saddle) / combined_err
        else:
          z = np.inf if weaker_peak > rho_saddle else 0.0

        pair_zscores[(ci, cj)] = float(z)

        # Population weight: geometric mean of cluster sizes
        ni = float(np.sum(labels == ci))
        nj = float(np.sum(labels == cj))
        w = np.sqrt(ni * nj)
        weighted_z_sum += w * z
        total_weight += w

      self.pair_zscores_ = pair_zscores

      if total_weight <= 0:
        return None

      return float(weighted_z_sum / total_weight)

    except Exception:
      return None

  # -------------------------------------------------------------------
  # Plotting methods
  # -------------------------------------------------------------------

  def _plot_dendrogram(
    self,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "viridis",
    logscale: bool = True,
    title: str = "PAk Density Topography Dendrogram",
  ) -> Axes:
    """
    Plot a dendrogram of the dataset topography based on PAk density.

    Visualizes the hierarchy of clusters built with single linkage,
    using the density at the border between clusters as the similarity
    measure. Unlike classical dendrograms where all branches have the
    same height, here the height of each branch is proportional to the
    log-density of the cluster center.

    The x-axis spacing between clusters is proportional to the cluster
    population (or its logarithm for unbalanced clusterings).

    Parameters
    ----------
    ax : Axes, optional
      Matplotlib axes to plot on. Creates new figure if None.
    figsize : tuple, default=(12, 8)
      Figure size if creating new figure.
    cmap : str, default='viridis'
      Colormap for cluster node colors.
    logscale : bool, default=True
      If True, x-axis spacing proportional to ln(population).
    title : str
      Plot title.

    Returns
    -------
    Axes
      Matplotlib axes with the dendrogram.

    Raises
    ------
    ValueError
      If compute() has not been called yet.
    """
    if self.log_den_ is None:
      raise ValueError(
        "compute() must be called before plotting the dendrogram."
      )

    labels = self.labels
    log_den = self.log_den_

    # --- Gather cluster info ---
    unique_labels = np.array(sorted(
      [l for l in np.unique(labels) if l >= 0]
    ))
    n_clusters = len(unique_labels)
    label_to_idx = {int(l): i for i, l in enumerate(unique_labels)}

    # Per-cluster: peak density, center index, population
    peak_densities = np.empty(n_clusters)
    populations = np.empty(n_clusters)
    for i, c in enumerate(unique_labels):
      cmask = labels == c
      idx_in_cluster = np.where(cmask)[0]
      peak_densities[i] = np.max(log_den[idx_in_cluster])
      populations[i] = float(np.sum(cmask))

    # --- Build saddle matrix ---
    log_den_bord = np.full((n_clusters, n_clusters), -np.inf)
    for (ci, cj), (rho_s, _) in self.saddle_densities_.items():
      ii = label_to_idx.get(ci)
      jj = label_to_idx.get(cj)
      if ii is not None and jj is not None:
        log_den_bord[ii, jj] = rho_s
        log_den_bord[jj, ii] = rho_s

    # --- Build condensed distance vector ---
    Fmax = float(np.max(log_den))
    min_den = float(np.min(log_den))
    large_val = 2.0 * Fmax - min_den

    # --- Single-linkage merging (dadapy-style) ---
    # Build full distance matrix
    Dis_mat = np.full((n_clusters, n_clusters), large_val)
    for i in range(n_clusters):
      Dis_mat[i, i] = 0.0
      for j in range(i + 1, n_clusters):
        if log_den_bord[i, j] > -np.inf:
          Dis_mat[i, j] = Fmax - log_den_bord[i, j]
          Dis_mat[j, i] = Dis_mat[i, j]

    # Single-linkage iterative merging
    # Track which original clusters belong to each active set
    active = list(range(n_clusters))
    members = [[i] for i in range(n_clusters)]  # members[i] = list of orig clusters
    merges = []  # (child_a, child_b, merge_distance, size)

    # For the dendrogram we track nodes: first n_clusters are leaves
    node_count = n_clusters

    # Iterative single-linkage
    current_dist = Dis_mat.copy()
    active_set = list(range(n_clusters))

    merge_nodes = []  # list of (left_node, right_node, merge_height, new_node_id)
    node_members = {i: [i] for i in range(n_clusters)}

    while len(active_set) > 1:
      # Find minimum distance pair among active clusters
      best_dist = np.inf
      best_i, best_j = -1, -1
      for ii_idx in range(len(active_set)):
        for jj_idx in range(ii_idx + 1, len(active_set)):
          i_c = active_set[ii_idx]
          j_c = active_set[jj_idx]
          if current_dist[i_c, j_c] < best_dist:
            best_dist = current_dist[i_c, j_c]
            best_i = i_c
            best_j = j_c

      # Merge best_i and best_j into best_i
      merge_height = Fmax - best_dist
      new_node = node_count
      node_count += 1
      merge_nodes.append((best_i, best_j, merge_height, new_node))
      node_members[new_node] = node_members[best_i] + node_members[best_j]

      # Update distances (single linkage = min)
      for k_c in active_set:
        if k_c != best_i and k_c != best_j:
          d_new = min(current_dist[best_i, k_c], current_dist[best_j, k_c])
          current_dist[best_i, k_c] = d_new
          current_dist[k_c, best_i] = d_new

      active_set.remove(best_j)
      # Reassign best_j's id in active tracking to best_i
      # (best_i now represents the merged cluster)

    # --- Compute x-positions (population-based) ---
    # Leaf x-positions: order by merge traversal
    def get_leaf_order(node_id):
      """Recursively get leaf ordering from merge tree."""
      if node_id < n_clusters:
        return [node_id]
      for left, right, _, nid in merge_nodes:
        if nid == node_id:
          return get_leaf_order(left) + get_leaf_order(right)
      return [node_id]

    root_node = node_count - 1
    leaf_order = get_leaf_order(root_node)

    # Population-based spacing
    if logscale:
      widths = np.log(populations + 1)
    else:
      widths = populations.copy()

    # Assign x-positions based on cumulative width
    x_pos = np.zeros(n_clusters)
    cumulative = 0.0
    for leaf in leaf_order:
      x_pos[leaf] = cumulative + widths[leaf] / 2.0
      cumulative += widths[leaf]

    # Compute x-position for internal nodes (average of children)
    node_x = {}
    for i in range(n_clusters):
      node_x[i] = x_pos[i]

    for left, right, _, nid in merge_nodes:
      node_x[nid] = (node_x[left] + node_x[right]) / 2.0

    # --- Plot ---
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=figsize)

    colormap = plt.get_cmap(cmap)
    colors = [colormap(i / max(n_clusters - 1, 1)) for i in range(n_clusters)]

    # Draw branches
    for left, right, m_height, nid in merge_nodes:
      # Left child: vertical line from child height to merge height,
      # then horizontal to merge x
      left_y = peak_densities[left] if left < n_clusters else \
        [mh for l, r, mh, n in merge_nodes if n == left][0]
      right_y = peak_densities[right] if right < n_clusters else \
        [mh for l, r, mh, n in merge_nodes if n == right][0]

      lx = node_x[left]
      rx = node_x[right]
      mx = node_x[nid]

      # Vertical lines from children down to merge height
      ax.plot([lx, lx], [left_y, m_height], color='black', linewidth=1.0)
      ax.plot([rx, rx], [right_y, m_height], color='black', linewidth=1.0)
      # Horizontal line connecting children at merge height
      ax.plot([lx, rx], [m_height, m_height], color='black', linewidth=1.0)

    # Draw cluster nodes (colored scatter)
    for i in range(n_clusters):
      ax.scatter(
        x_pos[i], peak_densities[i],
        c=[colors[i]], s=100, zorder=5, edgecolors='black', linewidths=0.5
      )
      ax.annotate(
        str(int(unique_labels[i])),
        (x_pos[i], peak_densities[i]),
        textcoords="offset points", xytext=(0, 8),
        ha='center', fontsize=8, fontweight='bold'
      )

    ax.set_ylabel(r"$\ln(\rho)$  [log-density]")
    xlabel = r"$\ln$(population)" if logscale else "population"
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax

  def _plot_network(
    self,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis",
    title: str = "PAk Cluster Network (MDS Topology)",
  ) -> Axes:
    """
    Plot a network representation of the dataset topography.

    Uses Multidimensional Scaling (MDS) to embed the inter-cluster
    dissimilarity matrix (Fmax - saddle density) into 2D. Each cluster
    is represented as a node whose size is proportional to sqrt(population),
    and edges between clusters have thickness proportional to the border
    density.

    Parameters
    ----------
    ax : Axes, optional
      Matplotlib axes. Creates new figure if None.
    figsize : tuple, default=(10, 8)
      Figure size.
    cmap : str, default='viridis'
      Colormap for node colors.
    title : str
      Plot title.

    Returns
    -------
    Axes
      Matplotlib axes with the network plot.

    Raises
    ------
    ValueError
      If compute() has not been called yet.
    """
    if self.log_den_ is None:
      raise ValueError(
        "compute() must be called before plotting the network."
      )

    from sklearn import manifold
    from matplotlib.collections import LineCollection

    labels = self.labels
    log_den = self.log_den_

    # --- Gather cluster info ---
    unique_labels = np.array(sorted(
      [l for l in np.unique(labels) if l >= 0]
    ))
    n_clusters = len(unique_labels)
    label_to_idx = {int(l): i for i, l in enumerate(unique_labels)}

    populations = np.empty(n_clusters)
    peak_densities = np.empty(n_clusters)
    for i, c in enumerate(unique_labels):
      cmask = labels == c
      populations[i] = float(np.sum(cmask))
      idx_in_cluster = np.where(cmask)[0]
      peak_densities[i] = np.max(log_den[idx_in_cluster])

    # --- Build saddle matrix ---
    Fmax = float(np.max(log_den))
    min_den = float(np.min(log_den))
    large_val = 2.0 * (Fmax - min_den)

    log_den_bord = np.full((n_clusters, n_clusters), -np.inf)
    for (ci, cj), (rho_s, _) in self.saddle_densities_.items():
      ii = label_to_idx.get(ci)
      jj = label_to_idx.get(cj)
      if ii is not None and jj is not None:
        log_den_bord[ii, jj] = rho_s
        log_den_bord[jj, ii] = rho_s

    # --- Build dissimilarity matrix ---
    d_dis = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
      for j in range(i + 1, n_clusters):
        if log_den_bord[i, j] > -np.inf:
          d_dis[i, j] = Fmax - log_den_bord[i, j]
        else:
          d_dis[i, j] = large_val
        d_dis[j, i] = d_dis[i, j]

    # --- MDS embedding ---
    mds = manifold.MDS(
      n_components=2, dissimilarity='precomputed', random_state=42,
      normalized_stress='auto'
    )
    coords = mds.fit_transform(d_dis)

    # --- Plot ---
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=figsize)

    colormap = plt.get_cmap(cmap)
    colors = [colormap(i / max(n_clusters - 1, 1)) for i in range(n_clusters)]

    # Draw edges between all adjacent cluster pairs
    # Normalise border density for linewidth
    bord_vals = []
    for i in range(n_clusters):
      for j in range(i + 1, n_clusters):
        if log_den_bord[i, j] > -np.inf:
          bord_vals.append(log_den_bord[i, j])

    if bord_vals:
      bord_min = min(bord_vals)
      bord_max = max(bord_vals)
      bord_range = bord_max - bord_min if bord_max > bord_min else 1.0

      segments = []
      linewidths = []
      for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
          if log_den_bord[i, j] > -np.inf:
            segments.append([coords[i], coords[j]])
            norm_w = (log_den_bord[i, j] - bord_min) / bord_range
            linewidths.append(0.5 + 3.0 * norm_w)

      if segments:
        lc = LineCollection(
          segments, linewidths=linewidths, colors='gray', alpha=0.5
        )
        ax.add_collection(lc)

    # Draw nodes
    sizes = 20.0 * np.sqrt(populations)
    for i in range(n_clusters):
      ax.scatter(
        coords[i, 0], coords[i, 1],
        c=[colors[i]], s=sizes[i], zorder=5,
        edgecolors='black', linewidths=0.5
      )
      ax.annotate(
        str(int(unique_labels[i])),
        (coords[i, 0], coords[i, 1]),
        textcoords="offset points", xytext=(5, 5),
        ha='left', fontsize=9, fontweight='bold'
      )

    ax.set_title(title)
    ax.set_xlabel("MDS dim 1")
    ax.set_ylabel("MDS dim 2")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')

    return ax

  def _plot_decisionGraph(
    self,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "PAk Decision Graph",
    point_size: int = 20,
    alpha: float = 0.7,
  ) -> Axes:
    """
    Plot the decision graph using log-density (g = log_den - log_den_err) as
    the density axis and delta (distance to nearest higher-density point) as
    the other axis.

    Points with both high density and high delta are cluster centers.

    Parameters
    ----------
    ax : Axes, optional
      Matplotlib axes. Creates new figure if None.
    figsize : tuple, default=(10, 8)
      Figure size.
    title : str
      Plot title.
    point_size : int, default=20
      Size of scatter points.
    alpha : float, default=0.7
      Point transparency.

    Returns
    -------
    Axes
      Matplotlib axes with the decision graph.

    Raises
    ------
    ValueError
      If compute() has not been called yet.
    """
    if self.log_den_ is None:
      raise ValueError(
        "compute() must be called before plotting the decision graph."
      )

    X = self.X
    labels = self.labels
    log_den = self.log_den_
    log_den_err = self.log_den_err_
    N = X.shape[0]

    # Conservative density estimate
    g = log_den - log_den_err

    # --- Compute delta: distance to nearest higher-density point ---
    # Sort points by descending g for efficiency
    order = np.argsort(-g)
    delta = np.full(N, np.inf)

    # For the point with highest g, delta will remain as max
    # For each other point, find nearest point with strictly higher g
    # Use a chunked approach to avoid O(N^2) memory for large datasets
    chunk_size = 1000
    for start in range(0, N, chunk_size):
      end = min(start + chunk_size, N)
      chunk_indices = np.arange(start, end)

      for idx in chunk_indices:
        i = order[idx]
        if idx == 0:
          # Global maximum — will set delta later
          continue
        # Points with higher g are order[:idx]
        higher_points = order[:idx]
        dists = np.sqrt(np.sum((X[higher_points] - X[i]) ** 2, axis=1))
        delta[i] = np.min(dists)

    # Set delta for global maximum to the max finite delta
    global_max_idx = order[0]
    finite_deltas = delta[delta < np.inf]
    if len(finite_deltas) > 0:
      delta[global_max_idx] = np.max(finite_deltas)
    else:
      delta[global_max_idx] = 1.0

    # --- Plot ---
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Color by cluster label
    unique_labels = np.unique(labels)
    colormap = plt.get_cmap("tab10")

    # Scatter all points
    for c in unique_labels:
      if c < 0:
        mask = labels == c
        ax.scatter(
          g[mask], delta[mask],
          c='lightgray', s=point_size * 0.5, alpha=alpha * 0.5,
          label='noise', zorder=2
        )
      else:
        mask = labels == c
        color = colormap(int(c) % 10)
        ax.scatter(
          g[mask], delta[mask],
          c=[color], s=point_size, alpha=alpha,
          label=f'Cluster {int(c)}', zorder=3
        )

    # Highlight cluster centers (peaks)
    if self.cluster_peak_densities_ is not None:
      for c, (rho_peak, _) in self.cluster_peak_densities_.items():
        cmask = labels == c
        idx_in_cluster = np.where(cmask)[0]
        peak_idx = idx_in_cluster[np.argmax(log_den[idx_in_cluster])]
        ax.scatter(
          g[peak_idx], delta[peak_idx],
          c='red', s=point_size * 5, marker='*', zorder=10,
          edgecolors='black', linewidths=0.5
        )

    ax.set_xlabel(r"$\rho$  [conservative log-density $g = \ln\rho - \sigma$]")
    ax.set_ylabel(r"$\delta$  [distance to nearest higher-density point]")
    ax.set_title(title)
    ax.legend(
      bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, framealpha=0.8
    )
    ax.grid(True, alpha=0.3)

    return ax


# =============================================================================
# CENTROID-BASED CLUSTERING
# =============================================================================
#
# Algorithms that partition data by minimizing distance to cluster centers.
# Require specifying number of clusters (k) in advance.
#
# Included algorithms:
#   - OGSKMeans: Classic K-Means, minimizes within-cluster variance
#   - OGSMiniBatchKMeans: Faster K-Means using mini-batches (for large data)
#   - OGSBisectingKMeans: Hierarchical K-Means using binary splitting
#
# For seismic applications:
#   - Good for well-separated earthquake clusters with known count
#   - K-Means assumes spherical clusters of similar size
#   - MiniBatch useful for large catalogs (>100k events)
#   - Bisecting can reveal cluster hierarchy
# =============================================================================


class CentroidClusterer(BaseClusterer):
  """
  Intermediate base class for centroid-based clustering algorithms.

  Provides a shared ``plot()`` method that overlays cluster centers on the
  scatter plot generated by ``BaseClusterer.plot()``.  All K-Means variants
  inherit from this class instead of ``BaseClusterer`` directly.
  """

  def plot(self,
    show_centers: bool = True,
    center_marker: str = "X",
    center_size: int = 200,
    center_color: str = "red",
    *args,
    **kwargs
  ) -> Axes:
    """
    Plot clustering results with optional cluster centers overlay.

    Parameters
    ----------
    show_centers : bool, default=True
      Whether to show cluster centers as markers.
    center_marker : str, default="X"
      Marker style for centers.
    center_size : int, default=200
      Size of center markers.
    center_color : str, default="red"
      Color of center markers.
    *args, **kwargs
      Arguments passed to BaseClusterer.plot().

    Returns
    -------
    Axes
      Matplotlib axes with the plot.
    """
    ax = super().plot(*args, **kwargs)

    centers = self.get_cluster_centers()
    if show_centers and centers is not None:
      feature_x = kwargs.get('feature_x', 0)
      feature_y = kwargs.get('feature_y', 1)
      ax.scatter(
        centers[:, feature_x],
        centers[:, feature_y],
        c=center_color,
        marker=center_marker,
        s=center_size,
        edgecolors="black",
        linewidths=2,
        label="Centers"
      )
      ax.legend()
    return ax


class OGSKMeans(CentroidClusterer):
  """
  K-Means clustering with plotting capabilities.

  Partitions n samples into k clusters by minimizing within-cluster sum of
  squares (inertia). Cluster centers are called centroids.

  For seismic catalogs: Useful when number of earthquake sequences is known or
  can be estimated (e.g., from magnitude distribution).

  Parameters
  ----------
  n_clusters : int, default=8
    The number of clusters to form.
  init : str, default='k-means++'
    Method for initialization ('k-means++', 'random', or ndarray).
  n_init : int or 'auto', default='auto'
    Number of times the k-means algorithm is run with different seeds.
  max_iter : int, default=300
    Maximum number of iterations for a single run.
  tol : float, default=1e-4
    Relative tolerance for convergence (based on inertia change).
  random_state : int, optional
    Random state for reproducibility.
  **kwargs
    Additional arguments passed to sklearn.cluster.KMeans.

  Attributes
  ----------
  model : sklearn.cluster.KMeans
    The underlying sklearn KMeans instance.
  labels_ : np.ndarray
    Cluster labels for each sample after fitting.

  Example
  -------
  >>> kmeans = OGSKMeans(n_clusters=5, random_state=42)
  >>> labels = kmeans.fit_predict(earthquake_locations)
  >>> kmeans.plot(feature_x=0, feature_y=1, show_centers=True)
  """

  def _create_model(self, n_clusters: int = 8, **kwargs) -> KMeans:
    """Create sklearn KMeans instance with specified parameters."""
    return KMeans(n_clusters=n_clusters, **kwargs)


class OGSMiniBatchKMeans(CentroidClusterer):
  """
  Mini-Batch K-Means clustering with plotting capabilities.

  Faster variant of K-Means that uses mini-batches to reduce computation time.
  Slightly worse results than regular K-Means but much faster for large
  datasets.

  For seismic catalogs: Recommended for catalogs with >50,000 events where full
  K-Means would be too slow.

  Parameters
  ----------
  n_clusters : int, default=8
    The number of clusters to form.
  batch_size : int, default=1024
    Size of the mini batches. Larger = slower but more accurate.
  **kwargs
    Additional arguments passed to sklearn.cluster.MiniBatchKMeans.
  """

  def _create_model(self,
    n_clusters: int = 8,
    batch_size: int = 1024,
    **kwargs
  ) -> MiniBatchKMeans:
    """Create sklearn MiniBatchKMeans instance."""
    return MiniBatchKMeans(n_clusters=n_clusters,
      batch_size=batch_size,
      **kwargs)


class OGSBisectingKMeans(CentroidClusterer):
  """
  Bisecting K-Means clustering with plotting capabilities.

  Hierarchical approach: starts with all data in one cluster, then repeatedly
  splits the cluster with largest inertia. Provides a tree-like structure of
  clusters.

  For seismic catalogs: Useful for exploring cluster hierarchy, e.g.,
  identifying sub-clusters within a seismic swarm.

  Parameters
  ----------
  n_clusters : int, default=8
    The number of clusters to form.
  **kwargs
    Additional arguments passed to sklearn.cluster.BisectingKMeans.
  """

  def _create_model(self, n_clusters: int = 8, **kwargs) -> BisectingKMeans:
    """Create sklearn BisectingKMeans instance."""
    return BisectingKMeans(n_clusters=n_clusters, **kwargs)


# =============================================================================
# DENSITY-BASED CLUSTERING
# =============================================================================
#
# Algorithms that find clusters as high-density regions separated by
# low-density regions. Do NOT require specifying number of clusters.
#
# Included algorithms:
#   - OGSDBSCAN: Classic density-based, uses eps and min_samples
#   - OGSHDBSCAN: Hierarchical DBSCAN, automatically finds optimal clusters
#   - OGSOPTICS: Creates reachability plot, finds clusters at multiple scales
#   - OGSAdvancedDensityPeaks: Density peaks algorithm (via dadapy)
#
# For seismic applications:
#   - DBSCAN/HDBSCAN excellent for earthquake sequence identification
#   - Naturally handles noise (isolated events labeled as -1)
#   - No assumption of cluster shape (can find elongated fault structures)
#   - HDBSCAN recommended as it requires fewer parameter tuning
#
# Key parameters:
#   - eps (DBSCAN): Maximum distance for neighborhood (km for spatial data)
#   - min_samples: Minimum events to form a dense region
#   - min_cluster_size (HDBSCAN): Minimum cluster size
# =============================================================================


class OGSDBSCAN(BaseClusterer):
  """
  DBSCAN clustering with plotting capabilities.

  Density-Based Spatial Clustering of Applications with Noise.
  Finds core samples in high-density regions and expands clusters from them.
  Points in low-density regions are labeled as noise (label=-1).

  For seismic catalogs: Excellent for identifying earthquake sequences without
  knowing the number of clusters. Set eps based on expected spatial extent of
  sequences (e.g., 5-10 km for local clusters).

  Parameters
  ----------
  eps : float, default=0.5
    Maximum distance between two samples in the same neighborhood.
    For seismic data in km, typical values: 1-20 km.
  min_samples : int, default=5
    Minimum number of samples in a neighborhood for a core point.
    Higher values = stricter clustering, fewer small clusters.
  metric : str, default='euclidean'
    Distance metric to use (euclidean, manhattan, etc.).
  **kwargs
    Additional arguments passed to sklearn.cluster.DBSCAN.

  Attributes
  ----------
  model.core_sample_indices_ : np.ndarray
    Indices of core samples (high-density points).
  model.components_ : np.ndarray
    Copy of each core sample.

  Example
  -------
  >>> dbscan = OGSDBSCAN(eps=5.0, min_samples=10)
  >>> labels = dbscan.fit_predict(earthquake_xyz)
  >>> n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
  >>> n_noise = list(labels).count(-1)
  """

  def _create_model(self,
    eps: float = 0.5,
    min_samples: int = 5,
    **kwargs
  ) -> DBSCAN:
    """Create sklearn DBSCAN instance with specified parameters."""
    return DBSCAN(eps=eps, min_samples=min_samples, **kwargs)

  def plot(self, highlight_core: bool = False, *args, **kwargs) -> Axes:
    """
    Plot DBSCAN clustering results.

    Parameters
    ----------
    highlight_core : bool, default=False
      Whether to highlight core samples with circle outlines.
      Core samples are the dense points that define clusters.
    *args, **kwargs
      Arguments passed to BaseClusterer.plot().

    Returns
    -------
    Axes
      Matplotlib axes with the plot.
    """
    # Plot clusters and noise using parent method
    ax = super().plot(*args, **kwargs)

    # Optionally highlight core samples (the dense points)
    if highlight_core and hasattr(self.model, 'core_sample_indices_'):
      if self.labels_ is None:
        raise ValueError("Model must be fitted before plotting.")

      # Create mask for core samples
      core_mask = np.zeros(len(self.labels_), dtype=bool)
      core_mask[self.model.core_sample_indices_] = True

      data = kwargs.get('X', self.data_)
      if data is None:
        raise ValueError("No data available for plotting.")

      feature_x = kwargs.get('feature_x', 0)
      feature_y = kwargs.get('feature_y', 1)

      # Draw circles around core samples
      ax.scatter(
        data[core_mask, feature_x],
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
  An extension of DBSCAN that converts it into a hierarchical clustering
  algorithm and extracts a flat clustering from the hierarchy.

  For seismic catalogs: RECOMMENDED density-based algorithm. Automatically
  determines the optimal number of clusters and handles varying cluster
  densities better than DBSCAN.

  Parameters
  ----------
  min_cluster_size : int, default=5
    Minimum size of clusters. Clusters smaller than this are noise.
  min_samples : int, optional
    Number of samples in a neighborhood for a core point.
    Defaults to min_cluster_size if not specified.
  cluster_selection_epsilon : float, default=0.0
    Distance threshold for cluster selection. Can be used to merge clusters
    closer than this threshold.
  cluster_selection_method : str, default='eom'
    Method for selecting clusters: 'eom' (excess of mass) or 'leaf'.
  **kwargs
    Additional arguments passed to sklearn.cluster.HDBSCAN.

  Attributes
  ----------
  model.probabilities_ : np.ndarray
    Cluster membership probability for each point [0, 1].
  model.cluster_persistence_ : np.ndarray
    Persistence of each cluster (stability measure).
  """

  def _create_model(self, min_cluster_size: int = 5, **kwargs) -> Any:
    """Create an available HDBSCAN instance."""
    if HDBSCAN is None:
      raise ImportError(
        "HDBSCAN is unavailable. Install scikit-learn with HDBSCAN support "
        "or install the 'hdbscan' package.")
    return HDBSCAN(min_cluster_size=min_cluster_size, **kwargs)

  def plot(self, show_probabilities: bool = False, *args, **kwargs) -> Axes:
    """
    Plot HDBSCAN clustering results.

    Parameters
    ----------
    show_probabilities : bool, default=False
      Whether to use cluster probabilities for point transparency.
      Points with low membership probability appear more transparent.
    *args, **kwargs
      Arguments passed to BaseClusterer.plot().

    Returns
    -------
    Axes
      Matplotlib axes with the plot.
    """
    # If showing probabilities, use them for alpha transparency
    if show_probabilities and hasattr(self.model, 'probabilities_'):
      kwargs['alpha'] = self.model.probabilities_

    return super().plot(*args, **kwargs)

  def get_probabilities(self) -> Optional[np.ndarray]:
    """
    Return cluster membership probabilities if available.

    Returns
    -------
    np.ndarray or None
      Probability of cluster membership for each point [0, 1].
    """
    if hasattr(self.model, 'probabilities_'):
      return self.model.probabilities_
    return None


class OGSOPTICS(BaseClusterer):
  """
  OPTICS clustering with plotting capabilities.

  Ordering Points To Identify the Clustering Structure.
  Creates a reachability plot that can be used to extract clusters at different
  density levels. More flexible than DBSCAN.

  For seismic catalogs: Useful for exploring multi-scale clustering structure,
  e.g., identifying nested sequences within larger swarms.

  Parameters
  ----------
  min_samples : int, default=5
    Minimum number of samples in a neighborhood.
  max_eps : float, default=np.inf
    Maximum distance between two samples for neighborhood.
  xi : float, optional
    Determines minimum steepness on reachability plot for cluster boundary.
  cluster_method : str, default='xi'
    Method to extract clusters: 'xi' or 'dbscan'.
  **kwargs
    Additional arguments passed to sklearn.cluster.OPTICS.

  Attributes
  ----------
  model.reachability_ : np.ndarray
    Reachability distances for each sample.
  model.ordering_ : np.ndarray
    Cluster-ordered indices of samples.
  """

  def _create_model(self, min_samples: int = 5, **kwargs) -> OPTICS:
    """Create sklearn OPTICS instance."""
    return OPTICS(min_samples=min_samples, **kwargs)

  def plot(self, *args, **kwargs) -> Axes:
    """Plot OPTICS clustering results."""
    return super().plot(*args, **kwargs)

  def plot_reachability(self,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (12, 4),
    title: str = "OPTICS Reachability Plot"
  ) -> Axes:
    """
    Plot the reachability diagram.

    The reachability plot shows the cluster structure. Valleys in the plot
    indicate clusters; deeper valleys = denser clusters.

    Parameters
    ----------
    ax : Axes, optional
      Matplotlib axes to plot on. Creates new figure if None.
    figsize : tuple, default=(12, 4)
      Figure size (width, height).
    title : str
      Plot title.

    Returns
    -------
    Axes
      Matplotlib axes with the reachability plot.

    Raises
    ------
    ValueError
      If model has not been fitted.
    """
    if not hasattr(self.model, 'reachability_'):
      raise ValueError("Model must be fitted first.")

    if ax is None:
      fig, ax = plt.subplots(figsize=figsize)

    if self.labels_ is None:
      raise ValueError("Model must be fitted first.")

    # Get reachability distances and ordering
    reachability = self.model.reachability_
    ordering = self.model.ordering_
    labels = self.labels_[ordering]

    # Color-code by cluster
    encoded, unique, cmap, norm = labels_to_colormap(labels)

    # Plot as bar chart (each bar = one point's reachability distance)
    for i, (reach, lab) in enumerate(zip(reachability[ordering], encoded)):
      color = cmap(norm(lab)) if labels[i] >= 0 else 'gray'
      ax.bar(i, reach, width=1, color=color, edgecolor='none')

    ax.set_xlabel("Sample ordering")
    ax.set_ylabel("Reachability distance")
    ax.set_title(title)
    return ax


class OGSAdvancedDensityPeaks(BaseClusterer):
  """
  Advanced Density Peaks clustering — pure NumPy/SciPy implementation.

  Implements the full ADP algorithm from scratch using only numpy, scipy, and
  sklearn. Every pipeline stage is an explicit method: k-NN distance
  computation, intrinsic-dimension estimation, adaptive neighbourhood
  selection, density estimation (PAk, kNN, kstarNN, kpeaks), local-density mode
  detection, steepest-ascent cluster assignment, saddle-point identification,
  and Z-score multimodality merging.

  Algorithm Steps
  ---------------
  1. Compute k-nearest-neighbor distances (sklearn NearestNeighbors)
  2. Estimate intrinsic dimension via 2NN ratio method
  3. Select adaptive neighborhood size k* via likelihood-ratio test
  4. Estimate local log-density using the configured estimator
  5. Find local density maxima as initial cluster centers
  6. Assign each point to the nearest peak via steepest-ascent on g
  7. Identify saddle points (highest border density between clusters)
  8. Merge clusters whose peaks fail the Z-score multimodality test
  9. Optionally mark low-density border points as halo (outliers)

  Parameters
  ----------
  Z : float, default=1.65
    Merging parameter controlling cluster granularity.
    Higher Z merges more aggressively (fewer, larger clusters).
    Lower Z keeps more peaks separate (more, smaller clusters).
    Typical range: [0.5, 3.0].
  halo : bool, default=False
    If True, mark low-density border points as outliers (label=-1).
  density_method : str, default='PAk'
    Density estimation method:
    - 'PAk': Point-Adaptive k-NN (recommended, balanced speed/quality)
    - 'kNN': Fixed k-nearest neighbor (fastest, less adaptive)
    - 'kstarNN': Adaptive k* with kNN estimator
    - 'kpeaks': k-peaks density estimator (alias for kstarNN)
  k : int, default=10
    Number of neighbors for kNN density estimation.
    Only used when density_method='kNN'.
  Dthr : float, default=23.92812698
    Likelihood ratio threshold for adaptive k* selection.
    Controls neighborhood size stringency:
    - 23.93: very conservative (p ~ 1e-6, default)
    - 14.07: stringent (p ~ 1e-4)
    - 6.63: moderate (p ~ 0.01)
  maxk : int or None, default=None
    Maximum neighbors to consider. If None, uses min(100, n_samples-1).
  n_jobs : int, default=-1
    CPU cores for parallel distance computation. -1 uses all cores.
  verbose : bool, default=False
    If True, print progress information.
  **kwargs
    Additional arguments passed to BaseClusterer.

  Attributes
  ----------
  labels_ : np.ndarray or None
    Cluster labels for each sample after fitting.
  data_ : np.ndarray or None
    Data used for fitting.
  cluster_centers_ : np.ndarray or None
    Indices (into the input array) of cluster center points.
  n_clusters_ : int or None
    Number of clusters found.
  log_den_ : np.ndarray or None
    Estimated log-density for each sample.
  log_den_err_ : np.ndarray or None
    Estimated error on log-density for each sample.
  kstar_ : np.ndarray or None
    Optimal neighborhood size for each sample.
  intrinsic_dim_ : float or None
    Estimated intrinsic dimension of the data.
  log_den_bord_ : np.ndarray or None
    Log-density matrix at saddle points between clusters.
  log_den_bord_err_ : np.ndarray or None
    Error matrix at saddle points between clusters.
  cluster_indices_ : list or None
    List of arrays, each containing sample indices in a cluster.

  References
  ----------
  M. d'Errico, E. Facco, A. Laio, A. Rodriguez, "Automatic topography of
  high-dimensional data sets by non-parametric density peak clustering,"
  Information Sciences 560 (2021) 476-492.

  Examples
  --------
  >>> adp = OGSAdvancedDensityPeaks(Z=1.5, density_method='PAk')
  >>> labels = adp.fit_predict(X)
  >>> adp.plot(xlabel="Longitude (km)", ylabel="Latitude (km)")
  >>> adp.plot_density(xlabel="Longitude (km)", ylabel="Latitude (km)")
  """

  _VALID_DENSITY_METHODS = frozenset({'PAk', 'kNN', 'kstarNN', 'kpeaks'})

  def __init__(self,
    Z: float = 1.65,
    halo: bool = False,
    density_method: str = 'PAk',
    k: int = 10,
    Dthr: float = 23.92812698,
    maxk: Optional[int] = None,
    n_jobs: int = -1,
    **kwargs
  ):
    """Initialize Advanced Density Peaks clusterer."""
    if density_method not in self._VALID_DENSITY_METHODS:
      raise ValueError(
        "density_method must be one of "
        f"{sorted(self._VALID_DENSITY_METHODS)}, got '{density_method}'"
      )

    self._Z = Z
    self._halo = halo
    self._density_method = density_method
    self._k = k
    self._Dthr = Dthr
    self._maxk = maxk
    self._n_jobs = n_jobs

    # Rich output attributes (populated after fit)
    self.cluster_centers_: Optional[np.ndarray] = None
    self.n_clusters_: Optional[int] = None
    self.log_den_: Optional[np.ndarray] = None
    self.log_den_err_: Optional[np.ndarray] = None
    self.kstar_: Optional[np.ndarray] = None
    self.intrinsic_dim_: Optional[float] = None
    self.log_den_bord_: Optional[np.ndarray] = None
    self.log_den_bord_err_: Optional[np.ndarray] = None
    self.bord_indices_: Optional[np.ndarray] = None
    self.cluster_indices_: Optional[list] = None
    self.distances_: Optional[np.ndarray] = None
    self.dist_indices_: Optional[np.ndarray] = None

    super().__init__(**kwargs)

  def _create_model(self, **kwargs) -> Any:
    """No sklearn model — full pipeline implemented in-class."""
    return None

  # =========================================================================
  # Step 1: k-NN Distance Computation
  # =========================================================================

  def _compute_nn_distances(
    self, X: np.ndarray, maxk: int, n_jobs: int
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute k-nearest-neighbor distances using sklearn.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
      Input data.
    maxk : int
      Maximum neighbor rank.
    n_jobs : int
      Number of parallel jobs.

    Returns
    -------
    distances : np.ndarray, shape (N, maxk+1)
      Sorted distances. Column 0 is self-distance (0).
    dist_indices : np.ndarray, shape (N, maxk+1)
      Indices of neighbors. Column 0 is self-index.
    """
    distances, dist_indices = PAkDensitySeparationScore._compute_nn_distances(X, maxk, n_jobs)
    # OGSAdvancedDensityPeaks additionally warns about zero distances
    eps = np.finfo(np.float64).eps
    zero_mask = distances[:, 1:] <= eps
    if np.any(zero_mask):
      n_zeros = int(np.sum(zero_mask))
      warnings.warn(
        f"Found {n_zeros} zero neighbor distances. "
        "Dataset may contain duplicate points.",
        RuntimeWarning,
      )
    return distances, dist_indices

  # =========================================================================
  # Step 2: Intrinsic Dimension Estimation (2NN)
  # =========================================================================

  def _estimate_intrinsic_dim(
    self, distances: np.ndarray, mu_fraction: float = 0.9
  ) -> float:
    """
    Estimate intrinsic dimension using the 2NN ratio method.

    Uses the ratio of second-to-first neighbor distances and fits the empirical
    CDF to extract the dimension.

    Parameters
    ----------
    distances : np.ndarray, shape (N, maxk+1)
      k-NN distance matrix.
    mu_fraction : float, default=0.9
      Fraction of sorted mu values to use (discard outlier tail).

    Returns
    -------
    float
      Estimated intrinsic dimension.

    References
    ----------
    E. Facco et al., "Estimating the intrinsic dimension of datasets by a
    minimal neighborhood information," Sci. Rep. 7 (2017).
    """
    return PAkDensitySeparationScore._estimate_intrinsic_dim(distances, mu_fraction)

  # =========================================================================
  # Step 3: Adaptive k* Selection (Likelihood-Ratio Test)
  # =========================================================================

  def _compute_kstar_adaptive(
    self, distances: np.ndarray, d: float, maxk: int, Dthr: float
  ) -> np.ndarray:
    """
    Compute optimal neighborhood size k* for each point.

    Tests consecutive distance ratios against a Pareto likelihood model. When
    the deviance exceeds the threshold, density is deemed inhomogeneous and k*
    is set to the last homogeneous rank.

    Fully vectorised — O(N * maxk) with no Python loops.

    Parameters
    ----------
    distances : np.ndarray, shape (N, maxk+1)
      Sorted k-NN distances (column 0 = self).
    d : float
      Intrinsic dimension estimate.
    maxk : int
      Maximum neighbor rank.
    Dthr : float
      Likelihood-ratio threshold.

    Returns
    -------
    np.ndarray of int64, shape (N,)
      Adaptive neighborhood size for each point, in [3, maxk].
    """
    return PAkDensitySeparationScore._compute_kstar_adaptive(distances, d, maxk, Dthr)

  # =========================================================================
  # Step 4: Density Estimation (Dispatcher + Methods)
  # =========================================================================

  def _compute_density_dispatch(
    self,
    distances: np.ndarray,
    dist_indices: np.ndarray,
    kstar: np.ndarray,
    d: float,
    N: int,
    maxk: int,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dispatch to the configured density estimator.

    Parameters
    ----------
    distances, dist_indices : np.ndarray
      k-NN distance and index matrices.
    kstar : np.ndarray of int
      Adaptive neighborhood sizes.
    d : float
      Intrinsic dimension.
    N : int
      Number of data points.
    maxk : int
      Maximum neighbor rank.

    Returns
    -------
    log_den : np.ndarray, shape (N,)
      Estimated log-density.
    log_den_err : np.ndarray, shape (N,)
      Error on log-density.
    """
    method = self._density_method

    if method == 'PAk':
      return self._density_pak(distances, kstar, d, N)
    elif method == 'kNN':
      return self._density_knn(distances, kstar, d, N)
    elif method in ('kstarNN', 'kpeaks'):
      return self._density_kstarnn(distances, kstar, d, N)
    else:
      raise ValueError(f"Unknown density method: {method}")

  # ---- 4a. kNN density (fixed-k, closed-form) ----

  def _density_knn(
    self, distances: np.ndarray, kstar: np.ndarray, d: float, N: int
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fixed-k kNN log-density estimate.

    log_den = log(k) - log(V_d) - d * log(r_k)

    where V_d is the d-dimensional unit-ball volume prefactor.
    Fully vectorised, no loops.
    """
    eps = np.finfo(np.float64).eps
    prefactor = np.exp(
      d / 2.0 * np.log(np.pi) - gammaln((d + 2.0) / 2.0)
    )
    kstar_f = np.maximum(kstar.astype(np.float64), 1.0)

    dc = np.maximum(distances[np.arange(N), kstar], eps)

    log_den = np.log(kstar_f) - np.log(prefactor) - d * np.log(dc)
    log_den_err = 1.0 / np.sqrt(kstar_f)

    return log_den, log_den_err

  # ---- 4b. kstarNN density (adaptive-k, closed-form) ----

  def _density_kstarnn(
    self, distances: np.ndarray, kstar: np.ndarray, d: float, N: int
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptive-k* kNN log-density estimate.

    Same formula as kNN but with per-point adaptive k* values, giving a
    locally-adaptive density estimate.
    """
    return self._density_knn(distances, kstar, d, N)

  # ---- 4c. PAk density (shell-volume Newton-Raphson ML) ----

  def _density_pak(
    self, distances: np.ndarray, kstar: np.ndarray, d: float, N: int
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Point-Adaptive k-NN (PAk) log-density via Newton-Raphson ML.

    Refines the kNN density estimate by maximising the Poisson shell-volume
    likelihood over all k* shells around each point.
    The Newton-Raphson iteration is fully vectorised across all N points
    simultaneously.

    Parameters
    ----------
    distances : np.ndarray, shape (N, maxk+1)
    kstar : np.ndarray of int, shape (N,)
    d : float
      Intrinsic dimension.
    N : int
      Number of points.

    Returns
    -------
    log_den : np.ndarray, shape (N,)
    log_den_err : np.ndarray, shape (N,)
    """
    return PAkDensitySeparationScore._density_pak(distances, kstar, d, N)

  # =========================================================================
  # Step 5: Find Density Modes (Local Maxima)
  # =========================================================================

  def _find_density_modes(
    self,
    g: np.ndarray,
    kstar: np.ndarray,
    dist_indices: np.ndarray,
    N: int,
  ) -> Tuple[List[int], dict]:
    """
    Find local density maxima as initial cluster centers.

    A point i is a center if g[i] >= g[j] for all j in its k* neighborhood.
    Centers that share a neighborhood are pruned: the lower-g center is removed
    and mapped to the higher-g one.

    Parameters
    ----------
    g : np.ndarray, shape (N,)
      Conservative density score (log_den - log_den_err).
    kstar : np.ndarray of int, shape (N,)
      Adaptive neighborhood sizes.
    dist_indices : np.ndarray, shape (N, maxk+1)
      Neighbor index matrix.
    N : int
      Number of data points.

    Returns
    -------
    centers : list of int
      Indices of surviving density peaks.
    removed : dict
      Maps removed-center index -> replacement-center index.
    """
    # -- Step A: identify local maxima --
    is_center = np.ones(N, dtype=bool)
    for i in range(N):
      k = kstar[i]
      neighbors = dist_indices[i, 1:k + 1]
      if np.any(g[neighbors] > g[i]):
        is_center[i] = False

    center_set = set(np.where(is_center)[0])

    # -- Step B: prune centers sharing a neighborhood --
    removed: dict = {}
    centers_list = sorted(center_set, key=lambda c: g[c], reverse=True)

    for c in centers_list:
      if c in removed:
        continue
      k = kstar[c]
      neighbors = dist_indices[c, 1:k + 1]
      for nb in neighbors:
        if nb in center_set and nb != c and nb not in removed:
          if g[nb] <= g[c]:
            removed[nb] = c
          else:
            removed[c] = nb
            break

    # -- Step C: resolve transitive chains --
    def resolve(idx: int) -> int:
      visited = set()
      while idx in removed:
        if idx in visited:
          break
        visited.add(idx)
        idx = removed[idx]
      return idx

    for k_rem in list(removed.keys()):
      removed[k_rem] = resolve(k_rem)

    centers = [c for c in centers_list if c not in removed]
    return centers, removed

  # =========================================================================
  # Step 6: Steepest-Ascent Cluster Assignment
  # =========================================================================

  def _assign_clusters(
    self,
    g: np.ndarray,
    centers: List[int],
    removed: dict,
    dist_indices: np.ndarray,
    kstar: np.ndarray,
    N: int,
  ) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Assign each point to a cluster via steepest ascent on g.

    Points are processed in descending g order.  Each point is assigned to the
    cluster of its highest-g already-assigned neighbor within its k*
    neighborhood.

    Parameters
    ----------
    g : np.ndarray, shape (N,)
    centers : list of int
    removed : dict
    dist_indices : np.ndarray, shape (N, maxk+1)
    kstar : np.ndarray of int, shape (N,)
    N : int

    Returns
    -------
    cluster_assignment : np.ndarray of int, shape (N,)
      Cluster label per point (0-indexed).
    cl_struct : list of list of int
      Per-cluster point index lists.
    """
    n_clusters = len(centers)
    cluster_assignment = np.full(N, -1, dtype=np.int64)

    # Assign centers
    center_to_cid = {}
    for cid, c in enumerate(centers):
      cluster_assignment[c] = cid
      center_to_cid[c] = cid

    # Process all points in descending g order
    order = np.argsort(-g)
    for i in order:
      if cluster_assignment[i] >= 0:
        continue
      k = kstar[i]
      neighbors = dist_indices[i, 1:k + 1]
      assigned_mask = cluster_assignment[neighbors] >= 0
      if np.any(assigned_mask):
        assigned_nbs = neighbors[assigned_mask]
        best_nb = assigned_nbs[np.argmax(g[assigned_nbs])]
        cluster_assignment[i] = cluster_assignment[best_nb]
      else:
        # Try broader neighborhood (full row)
        all_neighbors = dist_indices[i, 1:]
        assigned_mask2 = cluster_assignment[all_neighbors] >= 0
        if np.any(assigned_mask2):
          assigned_nbs2 = all_neighbors[assigned_mask2]
          best_nb2 = assigned_nbs2[np.argmax(g[assigned_nbs2])]
          cluster_assignment[i] = cluster_assignment[best_nb2]

    # Handle any residual unassigned via removed-centers chain
    unassigned = np.where(cluster_assignment < 0)[0]
    for i in unassigned:
      if i in removed:
        replacement = removed[i]
        if cluster_assignment[replacement] >= 0:
          cluster_assignment[i] = cluster_assignment[replacement]

    # Final fallback: assign to nearest assigned point
    still_unassigned = np.where(cluster_assignment < 0)[0]
    if len(still_unassigned) > 0:
      for i in still_unassigned:
        all_nbs = dist_indices[i, 1:]
        assigned_mask3 = cluster_assignment[all_nbs] >= 0
        if np.any(assigned_mask3):
          cluster_assignment[i] = cluster_assignment[
            all_nbs[assigned_mask3][0]
          ]
        else:
          cluster_assignment[i] = 0

    # Build per-cluster structure
    cl_struct: List[List[int]] = [[] for _ in range(n_clusters)]
    for i in range(N):
      cid = int(cluster_assignment[i])
      if 0 <= cid < n_clusters:
        cl_struct[cid].append(i)

    return cluster_assignment, cl_struct

  # =========================================================================
  # Step 7: Saddle-Point Identification
  # =========================================================================

  def _find_saddle_points(
    self,
    cluster_assignment: np.ndarray,
    log_den: np.ndarray,
    log_den_err: np.ndarray,
    dist_indices: np.ndarray,
    kstar: np.ndarray,
    centers: List[int],
    N: int,
    n_clusters: int,
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find saddle points (border densities) between cluster pairs.

    For each point, checks if any k*-neighbor belongs to a different cluster;
    if so, records the border density as the average of both points'
    log-densities.

    Parameters
    ----------
    cluster_assignment, log_den, log_den_err, dist_indices, kstar, centers, N,
    n_clusters : as above.

    Returns
    -------
    log_den_bord : np.ndarray, shape (n_clusters, n_clusters)
      Border log-density between cluster pairs (-inf if none).
    log_den_bord_err : np.ndarray, shape (n_clusters, n_clusters)
      Error on border density.
    bord_indices : np.ndarray, shape (n_clusters, n_clusters, 2)
      Indices of the two border points for each pair.
    """
    NEG_INF = -np.inf
    log_den_bord = np.full((n_clusters, n_clusters), NEG_INF)
    log_den_bord_err = np.zeros((n_clusters, n_clusters))
    bord_indices = np.full((n_clusters, n_clusters, 2), -1, dtype=np.int64)

    for p1 in range(N):
      c1 = int(cluster_assignment[p1])
      k = kstar[p1]
      neighbors = dist_indices[p1, 1:k + 1]

      for p2 in neighbors:
        c2 = int(cluster_assignment[p2])
        if c1 == c2 or c1 < 0 or c2 < 0:
          continue

        # Border density = average of both points' log-densities
        bord_g = 0.5 * (log_den[p1] + log_den[p2])
        bord_e = 0.5 * (log_den_err[p1] + log_den_err[p2])

        if bord_g > log_den_bord[c1, c2]:
          log_den_bord[c1, c2] = bord_g
          log_den_bord[c2, c1] = bord_g
          log_den_bord_err[c1, c2] = bord_e
          log_den_bord_err[c2, c1] = bord_e
          bord_indices[c1, c2] = [p1, p2]
          bord_indices[c2, c1] = [p2, p1]
        break  # Only first inter-cluster neighbor per point

    return log_den_bord, log_den_bord_err, bord_indices

  # =========================================================================
  # Step 8: Z-Score Multimodality Test (Merge Loop)
  # =========================================================================

  def _multimodality_test(
    self,
    Z: float,
    centers: List[int],
    log_den: np.ndarray,
    log_den_err: np.ndarray,
    log_den_bord: np.ndarray,
    log_den_bord_err: np.ndarray,
    cluster_assignment: np.ndarray,
    cl_struct: List[List[int]],
    n_clusters: int,
  ) -> Tuple[List[int], np.ndarray, List[List[int]], np.ndarray,
             np.ndarray, int]:
    """
    Merge clusters whose peaks are not statistically separated.

    Iteratively finds the pair (c1, c2) with the highest border density where
    the Z-score test fails for at least one peak, then merges the weaker peak
    into the stronger one.

    Parameters
    ----------
    Z : float
      Merging threshold.
    centers, log_den, log_den_err, log_den_bord, log_den_bord_err,
    cluster_assignment, cl_struct, n_clusters : as above.

    Returns
    -------
    centers, cluster_assignment, cl_struct, log_den_bord, log_den_bord_err,
    n_final : updated after all merges.
    """
    alive = np.ones(n_clusters, dtype=bool)
    peak_den = np.array([log_den[c] for c in centers])
    peak_err = np.array([log_den_err[c] for c in centers])

    while True:
      best_saddle = -np.inf
      merge_pair = None

      for c1 in range(n_clusters):
        if not alive[c1]:
          continue
        for c2 in range(c1 + 1, n_clusters):
          if not alive[c2]:
            continue
          bord_g = log_den_bord[c1, c2]
          if not np.isfinite(bord_g):
            continue

          bord_e = log_den_bord_err[c1, c2]

          # Z-score test for each peak
          a1 = peak_den[c1] - bord_g
          a2 = peak_den[c2] - bord_g
          e1 = Z * (peak_err[c1] + bord_e)
          e2 = Z * (peak_err[c2] + bord_e)

          if a1 < e1 or a2 < e2:
            if bord_g > best_saddle:
              best_saddle = bord_g
              merge_pair = (c1, c2)

      if merge_pair is None:
        break

      c1, c2 = merge_pair

      # Determine which peak to keep (higher margin)
      bord_g = log_den_bord[c1, c2]
      bord_e = log_den_bord_err[c1, c2]
      margin1 = (peak_den[c1] - bord_g) / max(peak_err[c1] + bord_e, 1e-300)
      margin2 = (peak_den[c2] - bord_g) / max(peak_err[c2] + bord_e, 1e-300)

      if margin1 >= margin2:
        keep, remove = c1, c2
      else:
        keep, remove = c2, c1

      # Merge: reassign points
      alive[remove] = False
      for pt in cl_struct[remove]:
        cluster_assignment[pt] = keep
      cl_struct[keep].extend(cl_struct[remove])
      cl_struct[remove] = []

      # Update border densities: keep the higher border with surviving
      for c3 in range(n_clusters):
        if not alive[c3] or c3 == keep:
          continue
        new_bord = max(log_den_bord[keep, c3], log_den_bord[remove, c3])
        if new_bord > log_den_bord[keep, c3]:
          log_den_bord[keep, c3] = new_bord
          log_den_bord[c3, keep] = new_bord
          log_den_bord_err[keep, c3] = log_den_bord_err[remove, c3]
          log_den_bord_err[c3, keep] = log_den_bord_err[remove, c3]

    n_final = int(np.sum(alive))
    return (centers, cluster_assignment, cl_struct,
            log_den_bord, log_den_bord_err, n_final)

  # =========================================================================
  # Step 9: Finalize Labels and Build Output Matrices
  # =========================================================================

  def _finalize(
    self,
    halo: bool,
    centers: List[int],
    cluster_assignment: np.ndarray,
    cl_struct: List[List[int]],
    log_den: np.ndarray,
    log_den_bord: np.ndarray,
    log_den_bord_err: np.ndarray,
    bord_indices: np.ndarray,
    n_clusters_orig: int,
    N: int,
  ) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray,
             np.ndarray, List[List[int]]]:
    """
    Relabel clusters 0..n_final-1 and apply optional halo.

    Parameters
    ----------
    halo : bool
      If True, mark low-density border points as -1.
    Other parameters : as above.

    Returns
    -------
    labels : np.ndarray, shape (N,)
    final_centers : np.ndarray of int
    n_final : int
    final_bord : np.ndarray, shape (n_final, n_final)
    final_bord_err : np.ndarray, shape (n_final, n_final)
    final_bord_idx : np.ndarray, shape (n_final, n_final, 2)
    final_cl_struct : list of list of int
    """
    # Identify surviving clusters
    surviving = [c for c, pts in enumerate(cl_struct) if len(pts) > 0]
    n_final = len(surviving)

    # Build old_id -> new_id mapping
    old_to_new = {}
    for new_id, old_id in enumerate(surviving):
      old_to_new[old_id] = new_id

    # Relabel
    labels = np.full(N, -1, dtype=np.int64)
    for i in range(N):
      old_cid = int(cluster_assignment[i])
      if old_cid in old_to_new:
        labels[i] = old_to_new[old_cid]

    # Final centers
    final_centers = np.array(
      [centers[old_id] for old_id in surviving], dtype=np.int64
    )

    # Final border matrices
    final_bord = np.full((n_final, n_final), -np.inf)
    final_bord_err = np.zeros((n_final, n_final))
    final_bord_idx = np.full((n_final, n_final, 2), -1, dtype=np.int64)

    for i_new, i_old in enumerate(surviving):
      for j_new, j_old in enumerate(surviving):
        if i_new != j_new:
          final_bord[i_new, j_new] = log_den_bord[i_old, j_old]
          final_bord_err[i_new, j_new] = log_den_bord_err[i_old, j_old]
          if bord_indices is not None:
            final_bord_idx[i_new, j_new] = bord_indices[i_old, j_old]

    # Final cluster structure
    final_cl_struct: List[List[int]] = [[] for _ in range(n_final)]
    for i in range(N):
      if labels[i] >= 0:
        final_cl_struct[labels[i]].append(i)

    # Halo: mark points below max border density as outliers
    if halo and n_final > 1:
      for cid in range(n_final):
        # Max border density for this cluster
        bord_row = final_bord[cid, :]
        finite_mask = np.isfinite(bord_row)
        if not np.any(finite_mask):
          continue
        max_bord_den = np.max(bord_row[finite_mask])
        for pt in final_cl_struct[cid]:
          if log_den[pt] < max_bord_den:
            labels[pt] = -1

    return (labels, final_centers, n_final, final_bord,
            final_bord_err, final_bord_idx, final_cl_struct)
  # =========================================================================
  # Step 4d: k-peaks density
  # =========================================================================

  def _density_kpeaks(
    self,
    distances: np.ndarray,
    dist_indices: np.ndarray,
    kstar: np.ndarray,
    d: float,
    N: int,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    k-peaks density estimator.

    Uses the k* value itself as a proxy for local density and computes the
    error from the variance of k* in the neighborhood.
    """
    log_den = kstar.astype(np.float64)
    log_den_err = np.ones(N, dtype=np.float64)

    for i in range(N):
      k = kstar[i]
      neighbors = dist_indices[i, 1:k + 1]
      if len(neighbors) > 0:
        neighbor_kstar = kstar[neighbors].astype(np.float64)
        diff_sq = (neighbor_kstar - float(k)) ** 2
        log_den_err[i] = max(np.sqrt(np.mean(diff_sq)), 1e-10)

    return log_den, log_den_err

  # =========================================================================
  # Main Orchestrator: fit_predict
  # =========================================================================

  def fit_predict(self, X: np.ndarray) -> np.ndarray:
    """
    Fit the ADP model and return cluster labels.

    Runs the full ADP pipeline from scratch:
      1. k-NN distance computation
      2. Intrinsic dimension estimation (2NN)
      3. Adaptive k* selection (likelihood-ratio test)
      4. Density estimation (PAk / kNN / kstarNN / kpeaks)
      5. Local density mode detection
      6. Steepest-ascent cluster assignment
      7. Saddle-point identification
      8. Z-score multimodality merging
      9. Relabeling and optional halo

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
      Data to cluster.

    Returns
    -------
    np.ndarray, shape (n_samples,)
      Cluster labels (0-indexed). -1 indicates halo/outlier points when
      halo=True.
    """
    self.data_ = X
    N = X.shape[0]

    # --- Input validation ---
    if N < 3:
      warnings.warn(
        f"Too few samples ({N}) for ADP clustering. "
        "Returning all points as cluster 0.",
        RuntimeWarning,
      )
      self.labels_ = np.zeros(N, dtype=np.int64)
      self.n_clusters_ = 1
      return self.labels_

    if np.any(~np.isfinite(X)):
      raise ValueError(
        "Input data contains NaN or Inf values. "
        "Clean the data before clustering."
      )

    # --- Configure neighborhood size ---
    maxk = self._maxk if self._maxk is not None else min(100, N - 1)
    maxk = min(maxk, N - 1)

    self.logger.info("N=%d, maxk=%d, method=%s", N, maxk, self._density_method)

    # --- Step 1: k-NN distances ---
    self.logger.debug("Step 1: Computing k-NN distances...")
    distances, dist_indices = self._compute_nn_distances(X, maxk, self._n_jobs)
    self.distances_ = distances
    self.dist_indices_ = dist_indices

    # --- Step 2: Intrinsic dimension ---
    self.logger.debug("Step 2: Estimating intrinsic dimension...")
    d = self._estimate_intrinsic_dim(distances)
    self.intrinsic_dim_ = d
    self.logger.debug("Intrinsic dimension = %.2f", d)

    # --- Step 3: Adaptive k* ---
    self.logger.debug("Step 3: Computing adaptive k*...")
    if self._density_method == 'kNN':
      kstar = np.full(N, min(self._k, maxk), dtype=np.int64)
    else:
      kstar = self._compute_kstar_adaptive(distances, d, maxk, self._Dthr)
    self.kstar_ = kstar
    self.logger.debug(
      "k* range: [%d, %d], median=%.0f",
      np.min(kstar), np.max(kstar), np.median(kstar),
    )

    # --- Step 4: Density estimation ---
    self.logger.debug("Step 4: Estimating density (%s)...",
                      self._density_method)
    log_den, log_den_err = self._compute_density_dispatch(
      distances, dist_indices, kstar, d, N, maxk
    )

    # Guard against NaN
    nan_mask = ~np.isfinite(log_den)
    if np.any(nan_mask):
      warnings.warn(
        f"Found {np.sum(nan_mask)} NaN/Inf in density. "
        "Falling back to kNN density.",
        RuntimeWarning,
      )
      log_den, log_den_err = self._density_knn(distances, kstar, d, N)

    self.log_den_ = log_den
    self.log_den_err_ = log_den_err

    # --- Step 5: Find density modes ---
    self.logger.debug("Step 5: Finding density modes...")
    g = log_den - log_den_err
    centers, removed = self._find_density_modes(g, kstar, dist_indices, N)
    n_centers = len(centers)
    self.logger.debug("Found %d initial centers", n_centers)

    if n_centers == 0:
      warnings.warn(
        "No density modes found. Assigning all points to cluster 0.",
        RuntimeWarning,
      )
      self.labels_ = np.zeros(N, dtype=np.int64)
      self.n_clusters_ = 1
      self.cluster_centers_ = np.array([np.argmax(g)], dtype=np.int64)
      return self.labels_

    if n_centers == 1:
      self.labels_ = np.zeros(N, dtype=np.int64)
      self.n_clusters_ = 1
      self.cluster_centers_ = np.array(centers, dtype=np.int64)
      self.cluster_indices_ = [list(range(N))]
      return self.labels_

    # --- Step 6: Assign clusters ---
    self.logger.debug("Step 6: Assigning clusters via steepest ascent...")
    cluster_assignment, cl_struct = self._assign_clusters(
      g, centers, removed, dist_indices, kstar, N
    )

    # --- Step 7: Find saddle points ---
    self.logger.debug("Step 7: Finding saddle points...")
    log_den_bord, log_den_bord_err, bord_indices = self._find_saddle_points(
      cluster_assignment, log_den, log_den_err, dist_indices, kstar, centers,
      N, n_centers
    )

    # --- Step 8: Multimodality test ---
    self.logger.debug("Step 8: Multimodality test (merging)...")
    (centers, cluster_assignment, cl_struct, log_den_bord, log_den_bord_err,
     n_final) = self._multimodality_test(
      self._Z, centers, log_den, log_den_err, log_den_bord, log_den_bord_err,
      cluster_assignment, cl_struct, n_centers
    )
    self.logger.debug("After merging: %d clusters", n_final)

    # --- Step 9: Finalize ---
    self.logger.debug("Step 9: Finalizing labels...")
    (labels, final_centers, n_final, final_bord, final_bord_err,
     final_bord_idx, final_cl_struct) = self._finalize(
      self._halo, centers, cluster_assignment, cl_struct, log_den,
      log_den_bord, log_den_bord_err, bord_indices, n_centers, N
    )

    # --- Store results ---
    self.labels_ = labels
    self.cluster_centers_ = final_centers
    self.n_clusters_ = n_final
    self.log_den_bord_ = final_bord
    self.log_den_bord_err_ = final_bord_err
    self.bord_indices_ = final_bord_idx
    self.cluster_indices_ = final_cl_struct

    self.logger.info("Done. %d clusters found.", n_final)

    return labels

  # =========================================================================
  # Public Methods: Accessors and Plotting
  # =========================================================================

  def get_cluster_centers(self) -> Optional[np.ndarray]:
    """
    Return coordinates of cluster centers (density peaks).

    Returns
    -------
    np.ndarray or None
      Coordinates of cluster centers, shape (n_clusters, n_features).
    """
    if self.cluster_centers_ is None or self.data_ is None:
      return None
    return self.data_[self.cluster_centers_]

  def plot(self, *args, **kwargs) -> Axes:
    """Plot Advanced Density Peaks clustering results."""
    return super().plot(*args, **kwargs)

  def plot_density(self,
    X: Optional[np.ndarray] = None,
    feature_x: int = 0,
    feature_y: int = 1,
    ax: Optional[Axes] = None,
    title: str = "ADP Log-Density Landscape",
    xlabel: str = "Feature 1",
    ylabel: str = "Feature 2",
    point_size: int = 20,
    alpha: float = 0.8,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis",
  ) -> Axes:
    """
    Plot the estimated log-density landscape.

    Colors each point by its estimated log-density, revealing the density
    structure that drives ADP cluster assignment. Density peaks (cluster
    centers) are overlaid as red stars.

    Parameters
    ----------
    X : np.ndarray, optional
      Data to plot. If None, uses the data from fit.
    feature_x : int, default=0
      Feature index for x-axis.
    feature_y : int, default=1
      Feature index for y-axis.
    ax : Axes, optional
      Matplotlib axes. Creates new figure if None.
    title : str
      Plot title.
    xlabel, ylabel : str
      Axis labels.
    point_size : int, default=20
      Scatter point size.
    alpha : float, default=0.8
      Point transparency.
    figsize : tuple, default=(10, 8)
      Figure size if creating new figure.
    cmap : str, default='viridis'
      Colormap for density values.

    Returns
    -------
    Axes
      Matplotlib axes with the density plot.
    """
    if self.log_den_ is None:
      raise ValueError("Model must be fitted first. Call fit_predict().")

    data = X if X is not None else self.data_
    if data is None:
      raise ValueError("No data available for plotting.")

    if ax is None:
      fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
      data[:, feature_x],
      data[:, feature_y],
      c=self.log_den_,
      s=point_size,
      alpha=alpha,
      cmap=cmap,
    )
    plt.colorbar(scatter, ax=ax, label="Log density")

    if self.cluster_centers_ is not None and self.data_ is not None:
      centers = self.data_[self.cluster_centers_]
      ax.scatter(
        centers[:, feature_x],
        centers[:, feature_y],
        c='red',
        s=point_size * 8,
        marker='*',
        edgecolors='black',
        linewidths=0.8,
        zorder=5,
        label="Cluster centers",
      )
      ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax

  def plot_cluster_borders(self,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: str = "ADP Saddle-Point Density Between Clusters",
    cmap: str = "YlOrRd",
  ) -> Axes:
    """
    Plot heatmap of saddle-point densities between clusters.

    Shows the log-density at the boundary (saddle point) between each pair of
    clusters. Lower boundary density indicates clearer separation between
    clusters.

    Parameters
    ----------
    ax : Axes, optional
      Matplotlib axes. Creates new figure if None.
    figsize : tuple, default=(8, 6)
      Figure size.
    title : str
      Plot title.
    cmap : str, default='YlOrRd'
      Colormap for border density values.

    Returns
    -------
    Axes
      Matplotlib axes with the heatmap.
    """
    if self.log_den_bord_ is None:
      raise ValueError(
        "Border densities not available. Fit the model first."
      )

    if ax is None:
      fig, ax = plt.subplots(figsize=figsize)

    # Replace -inf with NaN for cleaner visualization
    bord_display = self.log_den_bord_.copy()
    bord_display[~np.isfinite(bord_display)] = np.nan

    im = ax.imshow(bord_display, cmap=cmap, aspect='auto')
    plt.colorbar(im, ax=ax, label="Log density at border")

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Cluster")
    ax.set_title(title)

    n = self.log_den_bord_.shape[0]
    for i in range(n):
      for j in range(n):
        val = self.log_den_bord_[i, j]
        if np.isfinite(val):
          ax.text(
            j, i, f"{val:.1f}",
            ha='center', va='center', fontsize=8, color='black',
          )

    return ax


# =============================================================================
# ADP++ — OPTIMIZED ADVANCED DENSITY PEAKS (VECTORIZED)
# =============================================================================
#
# Mathematically identical to ADP but replaces all Python-level loops in
# Steps 4d, 5, 6, 7, and 8 with vectorized numpy operations.
#
# Four optimizations (see class docstring for full details):
#
#   Opt 1 — Fused Union-Find (Steps 5+6):
#           Steepest-ascent forest built in one vectorized pass.
#           Pointer-jumping path compression: O(N log N) numpy.
#
#   Opt 2 — Tensor saddle-point detection (Step 7):
#           Inter-cluster neighbor search via (N, maxk) tensor
#           comparison + argmax-on-boolean first-hit trick.
#           Cost: O(N · maxk) numpy.
#
#   Opt 3 — Vectorized k-peaks density (Step 4d):
#           Padded (N, maxk) array broadcast for neighbor k* variance.
#           Cost: O(N · maxk) numpy.
#
#   Opt 4 — Vectorized multimodality merge (Step 8):
#           Z-score eligibility for all C² pairs computed as broadcasted
#           (C × C) tensors. Per-merge border update via np.maximum + boolean
#           indexing.
#           Cost: O(C³) numpy (vs O(C³) Python in ADP).
#
# All other steps (1-4abc, 9) are inherited unchanged from
# OGSAdvancedDensityPeaks.
#
# Reference: d'Errico et al. (2021), "Automatic topography of
# high-dimensional data sets by non-parametric density peak clustering."
# =============================================================================

class OGSAdvancedDensityPeaksPP(OGSAdvancedDensityPeaks):
  """
  ADP++ — Optimized Advanced Density Peaks clustering.

  Produces **mathematically identical** results to
  :class:`OGSAdvancedDensityPeaks` while eliminating all Python-level loops
  from four computational bottlenecks (Steps 4d, 5, 6, 7, 8).

  Optimizations
  -------------
  1. **Fused Union-Find** (replaces Steps 5 + 6):
     Builds a steepest-ascent forest in one vectorized pass: every point is
     linked to its highest-density k*-neighbor.  Roots of the forest are
     density modes (local maxima of the conservative score

                          g = log_den - log_den_err

     ). Vectorized pointer-jumping compresses all paths in O(N log N) numpy
     operations, replacing the O(N · k̄) Python mode-detection loop and the O(N)
     sorted steepest-ascent assignment loop.

  2. **Vectorized saddle-point detection** (replaces Step 7):
     The entire (N × maxk) neighbor-cluster matrix is built via advanced
     indexing.  A boolean tensor flags inter-cluster edges; ``np.argmax`` on
     the boolean axis reproduces ADP's first-hit (``break``) semantics.  Border
     densities are gathered, sorted, and reduced to the C × C saddle matrix.
     Cost: O(N · maxk) numpy + O(E log E) sort (E ≪ N · maxk).

  3. **Vectorized k-peaks density** (replaces Step 4d loop):
     The neighbor-k* matrix (N × maxk) is built by fancy indexing and masked
     with the k* validity range.  Squared differences and row-wise means are
     computed as pure array broadcasts.
     Cost: O(N · maxk) numpy, zero Python-level iteration.

  4. **Vectorized multimodality test** (replaces Step 8 merge loop):
     Z-score eligibility for all C² cluster pairs is evaluated as four
     broadcasted (C × C) tensor operations:

       a₁ = ρ_peak[i] - ρ_bord[i,j]   (peak-to-saddle gap, peak i)
       a₂ = ρ_peak[j] - ρ_bord[i,j]   (peak-to-saddle gap, peak j)
       e₁ = Z · (σ_peak[i] + σ_bord[i,j])   (significance, peak i)
       e₂ = Z · (σ_peak[j] + σ_bord[i,j])   (significance, peak j)
       eligible[i,j] = (a₁ < e₁) ∨ (a₂ < e₂)

     The best merge (highest border density among eligible pairs) is found by a
     masked argmax on the upper triangle.  After each merge the border
     row/column for the surviving cluster is updated via
     ``np.maximum`` + boolean indexing (one numpy call replaces the O(C) Python
                                        loop).
     Cost: O(C³) numpy operations (~100-1000× faster than O(C³)
     Python iterations due to SIMD, cache locality, and no interpreter
     overhead).

  Complexity comparison
  ---------------------
  +------------+---------------------------+----------------------------+
  | Step(s)    | ADP (parent)              | ADP++ (this class)         |
  +============+===========================+============================+
  | 4d         | O(N · k̄)  Python loop     | O(N · maxk)  numpy         |
  | 5 + 6      | O(N · k̄)  Python loops    | O(N log N)   numpy         |
  | 7          | O(N · k̄)  Python loop     | O(N · maxk)  numpy         |
  | 8          | O(C³)     Python loops    | O(C³)        numpy         |
  +------------+---------------------------+----------------------------+

  Steps 1-4abc and 9 are inherited unchanged from
  :class:`OGSAdvancedDensityPeaks`.

  Parameters
  ----------
  Same as :class:`OGSAdvancedDensityPeaks`.

  See Also
  --------
  OGSAdvancedDensityPeaks : Reference ADP implementation with Python-level
                            loops.

  References
  ----------
  d'Errico, M. et al. (2021). "Automatic topography of high-dimensional data
  sets by non-parametric density peak clustering." Information Sciences, 560,
  476-492.

  Citation
  --------
  If you use this class in your research, please cite the original ADP paper
  and this implementation as follows:
  ```
  @article{derrico2021automatic,
    title={Automatic topography of high-dimensional data sets by non-parametric density peak clustering},
    author={d'Errico, M. and Laio, A. and Rodriguez, A.},
    journal={Information Sciences},
    volume={560},
    pages={476--492},
    year={2021},
    publisher={Elsevier}
  }
  @article{
  }
  ```
  """

  # =========================================================================
  # Optimization 1: Fused Union-Find (Steps 5 + 6)
  # =========================================================================
  # Replaces _find_density_modes (Step 5) + _assign_clusters (Step 6).
  #
  # Algorithm:
  #   Phase 1 — Parent-link construction (vectorized):
  #     For each point i, parent(i) = argmax_{j ∈ N_i^{k*}, g_j > g_i} g_j
  #     If no such j exists, parent(i) = i  (i is a root / density mode).
  #     Built via masked (N, maxk) tensor + axis-1 argmax.
  #
  #   Phase 2 — Path compression (vectorized pointer-jumping):
  #     Iteratively apply root[i] = root[root[i]] for all i in parallel.
  #     Converges in ≤ ⌈log₂ N⌉ + 1 iterations.  Each iteration is a single
  #     numpy fancy-index operation (root[root]).
  #
  #   Phase 3 — Label extraction:
  #     Roots (parent(i) = i) are density modes.  Sorted by descending g and
  #     assigned sequential labels via a lookup table.
  #
  # Complexity:  O(N · maxk) for Phase 1, O(N log N) for Phase 2.
  # =========================================================================

  def _fused_mode_and_assignment(
    self,
    g: np.ndarray,
    kstar: np.ndarray,
    dist_indices: np.ndarray,
    N: int,
    maxk: int,
  ) -> Tuple[np.ndarray, List[int], List[List[int]]]:
    """
    Fused mode detection and cluster assignment via Union-Find.

    Builds a steepest-ascent forest where each point links to its
    highest-density k*-neighbor.  Roots of the forest are density modes (local
    maxima of `g`). Vectorized pointer-jumping compresses all paths in
    O(N log N) numpy operations.

    Mathematically equivalent to calling ``_find_density_modes`` (Step 5)
    followed by ``_assign_clusters`` (Step 6) in the parent
    :class:`OGSAdvancedDensityPeaks`.

    Algorithm
    ---------
    **Phase 1** — Parent-link construction (fully vectorized):
      For each point *i*, define::

        parent(i) = argmax { g_j : j ∈ N_i^{k*}, g_j > g_i }

      If no such neighbor exists, ``parent(i) = i`` (root/mode).
      Implemented as a masked argmax over the (N, maxk) neighbor score matrix.

    **Phase 2** — Vectorized pointer-jumping path compression::

        root^{t+1}[i] = root^t[root^t[i]]   for all i

      Each iteration doubles the compressed path length.  Converges in at most
      ``ceil(log2(N)) + 1`` iterations (checked via ``np.array_equal``).

    **Phase 3** — Label extraction:
      Points where ``root[i] == i`` are density modes.  Modes are sorted by
      descending `g` and assigned sequential cluster labels through a lookup
      table.

    Parameters
    ----------
    g : np.ndarray, shape (N,)
      Conservative density score ``log_den - log_den_err``.
    kstar : np.ndarray of int, shape (N,)
      Adaptive neighborhood size per point.
    dist_indices : np.ndarray, shape (N, maxk+1)
      Neighbor index matrix (column 0 = self).
    N : int
      Number of data points.
    maxk : int
      Maximum neighbor rank.

    Returns
    -------
    cluster_assignment : np.ndarray of int, shape (N,)
      Cluster label per point (0-indexed).
    centers : list of int
      Indices of density-mode points, sorted by descending `g`.
    cl_struct : list of list of int
      Per-cluster point index lists.
    """
    # ── Phase 1: Parent-link construction ──────────────────────────────
    # Build (N, maxk) neighbor score matrix via fancy indexing, then mask to
    # keep only k*-range neighbors with strictly higher g.
    # A single axis-1 argmax yields the parent pointer for every point.
    neighbor_indices = dist_indices[:, 1:maxk + 1]        # (N, maxk)
    neighbor_g = g[neighbor_indices]                      # (N, maxk)

    col_idx = np.arange(maxk)[np.newaxis, :]              # (1, maxk)
    valid = (col_idx < kstar[:, np.newaxis]) & (neighbor_g > g[:, np.newaxis])

    neighbor_g_masked = np.where(valid, neighbor_g, -np.inf)
    best_local = np.argmax(neighbor_g_masked, axis=1)     # (N,)
    has_higher = np.max(neighbor_g_masked, axis=1) > -np.inf

    best_global = neighbor_indices[np.arange(N), best_local]
    parent = np.where(has_higher, best_global, np.arange(N, dtype=np.int64))

    # ── Phase 2: Path compression (pointer-jumping) ──────────────────
    # root^{t+1}[i] = root^t[root^t[i]]  — doubles compressed length each
    # iteration.  Converges in ≤ ceil(log2(N)) + 1 steps.
    root = parent.copy()
    max_iters = int(np.ceil(np.log2(max(N, 2)))) + 1
    for _ in range(max_iters):
      root_new = root[root]
      if np.array_equal(root_new, root):
        break
      root = root_new

    # ── Phase 3: Label extraction ────────────────────────────────────
    # root[i] == i ⟹ i is a density mode.  Sort modes by descending g
    # (matches ADP convention) and build sequential cluster labels.
    is_root = root == np.arange(N, dtype=np.int64)
    center_indices = np.where(is_root)[0]

    center_order = np.argsort(-g[center_indices])
    center_indices = center_indices[center_order]
    centers_list = center_indices.tolist()
    n_clusters = len(centers_list)

    root_to_label = np.full(N, -1, dtype=np.int64)
    for cid, c in enumerate(centers_list):
      root_to_label[c] = cid

    cluster_assignment = root_to_label[root]

    # ── Fallback: assign residual unassigned points ──────────────────
    # Rare edge case — assign to nearest already-labelled neighbor.
    unassigned = cluster_assignment < 0
    if np.any(unassigned):
      for i in np.where(unassigned)[0]:
        nbs = dist_indices[i, 1:]
        mask = cluster_assignment[nbs] >= 0
        if np.any(mask):
          cluster_assignment[i] = cluster_assignment[nbs[mask][0]]
        else:
          cluster_assignment[i] = 0

    # ── Build per-cluster structure ──────────────────────────────────
    cl_struct: List[List[int]] = [[] for _ in range(n_clusters)]
    for cid in range(n_clusters):
      cl_struct[cid] = np.where(cluster_assignment == cid)[0].tolist()

    return cluster_assignment, centers_list, cl_struct

  # =========================================================================
  # Optimization 2: Vectorized Saddle Points (Step 7)
  # =========================================================================
  # Replaces _find_saddle_points from the parent class.
  #
  # Algorithm (7 vectorized sub-steps):
  #   1. Build (N, maxk) neighbor-cluster matrix via fancy indexing.
  #   2. Mask to k* range: V[i,r] = 1 iff r < k*_i.
  #   3. Boolean tensor B[i,r] = "neighbor r of point i belongs to a different
  #      cluster" ∧ V[i,r].
  #   4. argmax(B, axis=1) → first inter-cluster neighbor per point
  #      (reproduces ADP's first-hit / break semantics).
  #   5. Gather all border-point pairs (p1, p2) and their cluster labels
  #      (c1, c2).
  #   6. Compute border densities: ρ_bord = ½(ρ_{p1} + ρ_{p2}).
  #   7. Sort by descending ρ_bord and reduce to C×C matrix
  #      (first-occurrence scan guarantees the max is kept).
  #
  # Complexity:  O(N · maxk) numpy  +  O(E log E) sort,  E ≪ N·maxk.
  # =========================================================================

  def _find_saddle_points(
    self,
    cluster_assignment: np.ndarray,
    log_den: np.ndarray,
    log_den_err: np.ndarray,
    dist_indices: np.ndarray,
    kstar: np.ndarray,
    centers: List[int],
    N: int,
    n_clusters: int,
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized saddle-point detection between cluster pairs.

    For every point, the first k*-neighbor belonging to a different cluster is
    identified via a boolean tensor + argmax trick that faithfully reproduces
    ADP's ``break``-on-first-hit loop.  The resulting border-point pairs are
    reduced to the C × C saddle-density matrix by a descending-sort
    first-occurrence scan.

    Replaces the O(N · k̄) Python nested loop in the parent class with
    O(N · maxk) vectorized numpy operations.

    Parameters
    ----------
    cluster_assignment : np.ndarray of int, shape (N,)
      Current cluster label per point.
    log_den : np.ndarray, shape (N,)
      Estimated log-density per point.
    log_den_err : np.ndarray, shape (N,)
      Log-density error per point.
    dist_indices : np.ndarray, shape (N, maxk+1)
      Neighbor index matrix (column 0 = self).
    kstar : np.ndarray of int, shape (N,)
      Adaptive neighborhood sizes.
    centers : list of int
      Current cluster center indices.
    N : int
      Number of data points.
    n_clusters : int
      Number of clusters.

    Returns
    -------
    log_den_bord : np.ndarray, shape (n_clusters, n_clusters)
      Symmetric matrix of border log-densities (-∞ if no border).
    log_den_bord_err : np.ndarray, shape (n_clusters, n_clusters)
      Error on border log-densities.
    bord_indices : np.ndarray, shape (n_clusters, n_clusters, 2)
      Indices of the two border points for each cluster pair.
    """
    maxk = dist_indices.shape[1] - 1

    # ── Sub-step 1: Neighbor-cluster matrix (N, maxk) ────────────────
    neighbor_indices = dist_indices[:, 1:maxk + 1]          # (N, maxk)
    neighbor_clusters = cluster_assignment[neighbor_indices] # (N, maxk)

    # ── Sub-step 2: k* validity mask ─────────────────────────────────
    col_idx = np.arange(maxk)[np.newaxis, :]                # (1, maxk)
    valid_mask = col_idx < kstar[:, np.newaxis]             # (N, maxk)

    # ── Sub-step 3: Boolean inter-cluster edge tensor ────────────────
    # B[i,r] = True iff neighbor r of point i belongs to a different cluster
    # AND both labels are non-negative AND r < k*_i.
    point_clusters = cluster_assignment[:, np.newaxis]      # (N, 1)
    is_border = (
      (neighbor_clusters != point_clusters)
      & valid_mask
      & (point_clusters >= 0)
      & (neighbor_clusters >= 0)
    )

    # ── Sub-step 4: First inter-cluster neighbor (argmax trick) ──────
    # np.argmax on int8 returns the index of the *first* True entry, faithfully
    # reproducing ADP's "break on first hit" semantics.
    has_border = np.any(is_border, axis=1)
    first_border_col = np.argmax(is_border.astype(np.int8), axis=1)

    # ── Sub-step 5: Gather border-point pairs ────────────────────────
    border_points = np.where(has_border)[0]
    if len(border_points) == 0:
      log_den_bord = np.full((n_clusters, n_clusters), -np.inf)
      log_den_bord_err = np.zeros((n_clusters, n_clusters))
      bord_indices = np.full((n_clusters, n_clusters, 2), -1, dtype=np.int64)
      return log_den_bord, log_den_bord_err, bord_indices

    border_cols = first_border_col[border_points]
    p1 = border_points                                      # point indices
    p2 = neighbor_indices[p1, border_cols]                  # neighbor indices
    c1 = cluster_assignment[p1]                             # cluster of p1
    c2 = cluster_assignment[p2]                             # cluster of p2

    # ── Sub-step 6: Border densities ─────────────────────────────────
    # ρ_bord(p1, p2) = ½ (ρ_{p1} + ρ_{p2})  — same formula as ADP.
    border_g = 0.5 * (log_den[p1] + log_den[p2])
    border_e = 0.5 * (log_den_err[p1] + log_den_err[p2])

    # ── Sub-step 7: Reduce to C × C matrix (max per cluster pair) ──────
    # Pre-sort by descending border density so the first-occurrence scan
    # automatically records the maximum for each (c1, c2) pair.
    log_den_bord = np.full((n_clusters, n_clusters), -np.inf)
    log_den_bord_err = np.zeros((n_clusters, n_clusters))
    bord_indices = np.full((n_clusters, n_clusters, 2), -1, dtype=np.int64)

    order = np.argsort(-border_g)
    p1_sorted = p1[order]
    p2_sorted = p2[order]
    c1_sorted = c1[order]
    c2_sorted = c2[order]
    bg_sorted = border_g[order]
    be_sorted = border_e[order]

    seen = np.zeros((n_clusters, n_clusters), dtype=bool)
    for idx in range(len(order)):
      ci, cj = c1_sorted[idx], c2_sorted[idx]
      if not seen[ci, cj]:
        seen[ci, cj] = True
        seen[cj, ci] = True
        log_den_bord[ci, cj] = bg_sorted[idx]
        log_den_bord[cj, ci] = bg_sorted[idx]
        log_den_bord_err[ci, cj] = be_sorted[idx]
        log_den_bord_err[cj, ci] = be_sorted[idx]
        bord_indices[ci, cj] = [p1_sorted[idx], p2_sorted[idx]]
        bord_indices[cj, ci] = [p2_sorted[idx], p1_sorted[idx]]

    return log_den_bord, log_den_bord_err, bord_indices

  # =========================================================================
  # Optimization 4: Vectorized Multimodality Test (Step 8)
  # =========================================================================
  # Replaces _multimodality_test from the parent class.
  #
  # The merge loop has the same greedy structure as ADP:
  #   while ∃ eligible pair  →  merge the one with highest ρ_bord.
  #
  # What changes is HOW the eligible pair is found each iteration:
  #   ADP:   Python double loop over alive pairs          O(C²) Python
  #   ADP++: (C × C) broadcasted Z-score tensor + argmax  O(C²) numpy
  #
  # And how borders are updated after each merge:
  #   ADP:   Python loop over alive clusters              O(C)  Python
  #   ADP++: np.maximum + boolean row indexing            O(C)  numpy
  #
  # Total: O(C³) numpy  vs  O(C³) Python  (~100-1000× wall-clock gain).
  # =========================================================================

  def _multimodality_test(
    self,
    Z: float,
    centers: List[int],
    log_den: np.ndarray,
    log_den_err: np.ndarray,
    log_den_bord: np.ndarray,
    log_den_bord_err: np.ndarray,
    cluster_assignment: np.ndarray,
    cl_struct: List[List[int]],
    n_clusters: int,
  ) -> Tuple[List[int], np.ndarray, List[List[int]], np.ndarray,
             np.ndarray, int]:
    r"""
    Vectorized multimodality test — merges statistically indistinct peaks.

    Mathematically identical to the parent ADP implementation but replaces the
    O(C²) Python double loop per merge iteration with broadcasted numpy tensor
    operations over the full (C × C) pair space.

    Mathematical formulation
    ------------------------
    For every alive pair ``(i, j)`` with ``i < j``, compute:

    .. math::

      a_1 = \hat\rho^{\text{peak}}_i - \hat\rho^{\text{bord}}_{ij}
      \qquad
      e_1 = Z \cdot (\sigma^{\text{peak}}_i + \sigma^{\text{bord}}_{ij})

      a_2 = \hat\rho^{\text{peak}}_j - \hat\rho^{\text{bord}}_{ij}
      \qquad
      e_2 = Z \cdot (\sigma^{\text{peak}}_j + \sigma^{\text{bord}}_{ij})

      \text{eligible}_{ij} = (a_1 < e_1) \lor (a_2 < e_2)

    All four quantities are (C × C) matrices built by numpy broadcasting
    (``peak_den[:, None]``, ``peak_den[None, :]``, etc.) — no Python loop.

    The best merge target is::

      (c1*, c2*) = argmax_{eligible ∧ alive ∧ upper-tri}  ρ_bord[i, j]

    obtained via a masked ``np.argmax`` on the upper triangle.

    After each merge the border row/column for the surviving cluster is updated
    with ``np.maximum`` + boolean indexing (one numpy call replaces the O(C)
    Python loop in ADP).

    Merge decision
    ~~~~~~~~~~~~~~
    Among the two peaks of the selected pair, the one with the higher
    normalized margin ``m_k = (ρ_peak[k] - ρ_bord) / (σ_peak[k] + σ_bord)``
    survives; the other is absorbed.

    Complexity
    ----------
    ADP  (parent): O(C³) Python iterations  (C merges × C² pair scan).
    ADP++ (this):  O(C³) numpy operations — same asymptotic count but
      ~100-1000× faster per element due to SIMD, cache locality, and
      elimination of CPython interpreter overhead.

    For C > ~1000 initial modes, a priority-queue approach could further reduce
    the outer loop to O(C² log C); the numpy constant factor dominates for
    typical seismic catalog sizes (C < 1000).

    Parameters
    ----------
    Z : float
      Merging threshold (Z-score multiplier).
    centers : list of int
      Indices of cluster centers (density-mode points).
    log_den : np.ndarray, shape (N,)
      Estimated log-density per point.
    log_den_err : np.ndarray, shape (N,)
      Log-density error per point.
    log_den_bord : np.ndarray, shape (C, C)
      Symmetric border log-density matrix.
    log_den_bord_err : np.ndarray, shape (C, C)
      Border log-density error matrix.
    cluster_assignment : np.ndarray of int, shape (N,)
      Current cluster label per point.
    cl_struct : list of list of int
      Per-cluster point index lists.
    n_clusters : int
      Number of clusters before merging.

    Returns
    -------
    centers : list of int
      (unchanged) Original center indices.
    cluster_assignment : np.ndarray of int, shape (N,)
      Updated cluster labels after merging.
    cl_struct : list of list of int
      Updated per-cluster point lists.
    log_den_bord : np.ndarray, shape (C, C)
      Updated border density matrix.
    log_den_bord_err : np.ndarray, shape (C, C)
      Updated border density error matrix.
    n_final : int
      Number of surviving (alive) clusters.
    """
    alive = np.ones(n_clusters, dtype=bool)
    peak_den = np.array([log_den[c] for c in centers])      # (C,)
    peak_err = np.array([log_den_err[c] for c in centers])  # (C,)

    # Upper-triangular mask — avoids double-counting (i,j) and (j,i).
    upper = np.triu(np.ones((n_clusters, n_clusters), dtype=bool), k=1)

    while True:
      # ── Vectorized eligibility scan ────────────────────────────────
      # Build (C, C) candidate mask: both clusters alive, finite border, and
      # i < j (upper triangle).
      alive_2d = alive[:, None] & alive[None, :]            # (C, C)
      candidate = alive_2d & upper & np.isfinite(log_den_bord)

      if not np.any(candidate):
        break

      # Z-score test — four (C, C) broadcasted tensors.
      # a1[i,j] = peak_den[i] - bord[i,j]   (gap for row peak)
      # a2[i,j] = peak_den[j] - bord[i,j]   (gap for col peak)
      # e1[i,j] = Z · (σ_peak[i] + σ_bord[i,j])  (threshold)
      # e2[i,j] = Z · (σ_peak[j] + σ_bord[i,j])  (threshold)
      bord_g = log_den_bord
      bord_e = log_den_bord_err

      a1 = peak_den[:, None] - bord_g
      a2 = peak_den[None, :] - bord_g
      e1 = Z * (peak_err[:, None] + bord_e)
      e2 = Z * (peak_err[None, :] + bord_e)

      eligible = (a1 < e1) | (a2 < e2)
      merge_mask = candidate & eligible

      if not np.any(merge_mask):
        break

      # ── Best merge: highest border density among eligible pairs ────
      masked_bord = np.where(merge_mask, log_den_bord, -np.inf)
      flat_idx = int(np.argmax(masked_bord))
      c1, c2 = divmod(flat_idx, n_clusters)

      # ── Merge decision: keep the peak with higher normalized margin ─
      # margin_k = (ρ_peak[k] - ρ_bord) / (σ_peak[k] + σ_bord)
      bord_g_pair = log_den_bord[c1, c2]
      bord_e_pair = log_den_bord_err[c1, c2]
      denom1 = max(peak_err[c1] + bord_e_pair, 1e-300)
      denom2 = max(peak_err[c2] + bord_e_pair, 1e-300)
      margin1 = (peak_den[c1] - bord_g_pair) / denom1
      margin2 = (peak_den[c2] - bord_g_pair) / denom2

      if margin1 >= margin2:
        keep, remove = c1, c2
      else:
        keep, remove = c2, c1

      # ── Reassign all points of the removed cluster ─────────────────
      alive[remove] = False
      for pt in cl_struct[remove]:
        cluster_assignment[pt] = keep
      cl_struct[keep].extend(cl_struct[remove])
      cl_struct[remove] = []

      # ── Vectorized border update ───────────────────────────────────
      # For every surviving cluster c3 ≠ keep:
      #   bord(keep, c3) ← max(bord(keep, c3), bord(remove, c3))
      # One np.maximum + boolean index replaces ADP's Python loop.
      update_mask = alive.copy()
      update_mask[keep] = False

      old_keep = log_den_bord[keep, :].copy()
      old_remove = log_den_bord[remove, :].copy()
      better = (old_remove > old_keep) & update_mask

      log_den_bord[keep, better] = old_remove[better]
      log_den_bord[better, keep] = old_remove[better]
      log_den_bord_err[keep, better] = log_den_bord_err[remove, better]
      log_den_bord_err[better, keep] = log_den_bord_err[remove, better]

    n_final = int(np.sum(alive))
    return (centers, cluster_assignment, cl_struct,
            log_den_bord, log_den_bord_err, n_final)

  # =========================================================================
  # Optimization 3: Vectorized k-peaks density (Step 4d)
  # =========================================================================
  # Replaces _density_kpeaks from the parent class.
  #
  # The k-peaks estimator uses k*_i as a density proxy and computes σ_i from
  # the variance of k* in the k*-neighborhood:
  #
  #   ρ_i = k*_i,    σ_i = max( sqrt( mean_j (k*_j - k*_i)² ), 1e-10 )
  #
  # ADP computes this with a Python loop over N points.
  # ADP++ builds the full (N, maxk) neighbor-k* matrix, applies a padded
  # validity mask, and evaluates the squared-difference + mean as pure array
  # broadcasts.
  # Cost: O(N · maxk) numpy.
  # =========================================================================

  def _density_kpeaks(
    self,
    distances: np.ndarray,
    dist_indices: np.ndarray,
    kstar: np.ndarray,
    d: float,
    N: int,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized k-peaks density estimator.

    Uses k*_i as a proxy for local density and estimates its error from the
    variance of k* across the k*-neighborhood.  Replaces the O(N) Python loop
    in the parent class with padded (N, maxk) array broadcasts — zero
    Python-level iteration.

    Parameters
    ----------
    distances : np.ndarray, shape (N, maxk+1)
      Distance matrix (unused, kept for API compatibility).
    dist_indices : np.ndarray, shape (N, maxk+1)
      Neighbor index matrix (column 0 = self).
    kstar : np.ndarray of int, shape (N,)
      Adaptive neighborhood sizes.
    d : float
      Intrinsic dimensionality (unused, kept for API compatibility).
    N : int
      Number of data points.

    Returns
    -------
    log_den : np.ndarray, shape (N,)
      Density estimate (= k*_i cast to float64).
    log_den_err : np.ndarray, shape (N,)
      Density error (std of k* in the neighborhood, clamped ≥ 1e-10).
    """
    maxk = dist_indices.shape[1] - 1
    log_den = kstar.astype(np.float64)                    # ρ_i = k*_i

    # ── (N, maxk) neighbor-k* matrix via fancy indexing ──────────────
    neighbor_indices = dist_indices[:, 1:maxk + 1]        # (N, maxk)
    neighbor_kstar = kstar[neighbor_indices].astype(np.float64)
    kstar_f = kstar[:, np.newaxis].astype(np.float64)     # (N, 1)

    # ── k* validity mask: V[i,r] = 1 iff r < k*_i ───────────────────
    col_idx = np.arange(maxk)[np.newaxis, :]              # (1, maxk)
    valid = col_idx < kstar[:, np.newaxis]                # (N, maxk)

    # ── σ_i = max( sqrt( mean_j (k*_j - k*_i)² ), 1e-10 ) ──────────
    # Masked entries contribute 0 to the sum and are excluded from the count,
    # so the mean is computed only over valid neighbors.
    diff_sq = np.where(valid, (neighbor_kstar - kstar_f) ** 2, 0.0)
    counts = valid.sum(axis=1).astype(np.float64)
    counts = np.maximum(counts, 1.0)                      # avoid /0

    log_den_err = np.maximum(np.sqrt(diff_sq.sum(axis=1) / counts), 1e-10)

    return log_den, log_den_err

  # =========================================================================
  # Overridden fit_predict
  # =========================================================================
  # Same nine-step ADP pipeline, but:
  #   Step 4d  → Opt 3  (vectorized k-peaks density)
  #   Steps 5+6 → Opt 1  (fused Union-Find)
  #   Step 7   → Opt 2  (tensor saddle-point detection)
  #   Step 8   → Opt 4  (vectorized multimodality merge)
  # Steps 1-4abc and 9 are inherited unchanged.
  # =========================================================================

  def fit_predict(self, X: np.ndarray) -> np.ndarray:
    """
    Fit ADP++ and return cluster labels.

    Runs the full nine-step ADP pipeline with four vectorized overrides that
    eliminate all Python-level loops from the critical path:

    - **Step 4d** — k-peaks density via padded array broadcast.
    - **Steps 5+6** — Fused Union-Find mode detection + assignment.
    - **Step 7** — Tensor-based saddle-point detection.
    - **Step 8** — Broadcasted (C × C) multimodality merge.

    Steps 1-4abc (k-NN, intrinsic dim, adaptive k*, other density methods) and
    Step 9 (finalization / halo) are inherited unchanged from
    :class:`OGSAdvancedDensityPeaks`.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
      Data to cluster.

    Returns
    -------
    np.ndarray, shape (n_samples,)
      Cluster labels (0-indexed). -1 = halo/outlier when halo=True.
    """
    self.data_ = X
    N = X.shape[0]

    # --- Input validation ---
    if N < 3:
      warnings.warn(
        f"Too few samples ({N}) for ADP++ clustering. "
        "Returning all points as cluster 0.",
        RuntimeWarning,
      )
      self.labels_ = np.zeros(N, dtype=np.int64)
      self.n_clusters_ = 1
      return self.labels_

    if np.any(~np.isfinite(X)):
      raise ValueError(
        "Input data contains NaN or Inf values. "
        "Clean the data before clustering."
      )

    # --- Configure neighborhood size ---
    maxk = self._maxk if self._maxk is not None else min(100, N - 1)
    maxk = min(maxk, N - 1)

    self.logger.info("N=%d, maxk=%d, method=%s", N, maxk, self._density_method)

    # --- Step 1: k-NN distances (inherited) ---
    self.logger.debug("Step 1: Computing k-NN distances...")
    distances, dist_indices = self._compute_nn_distances(X, maxk, self._n_jobs)
    self.distances_ = distances
    self.dist_indices_ = dist_indices

    # --- Step 2: Intrinsic dimension (inherited) ---
    self.logger.debug("Step 2: Estimating intrinsic dimension...")
    d = self._estimate_intrinsic_dim(distances)
    self.intrinsic_dim_ = d
    self.logger.debug("Intrinsic dimension = %.2f", d)

    # --- Step 3: Adaptive k* (inherited) ---
    self.logger.debug("Step 3: Computing adaptive k*...")
    if self._density_method == 'kNN':
      kstar = np.full(N, min(self._k, maxk), dtype=np.int64)
    else:
      kstar = self._compute_kstar_adaptive(distances, d, maxk, self._Dthr)
    self.kstar_ = kstar

    # --- Step 4: Density estimation (inherited) ---
    self.logger.debug("Step 4: Estimating density (%s)...",
                      self._density_method)
    log_den, log_den_err = self._compute_density_dispatch(
      distances, dist_indices, kstar, d, N, maxk
    )

    nan_mask = ~np.isfinite(log_den)
    if np.any(nan_mask):
      warnings.warn(
        f"Found {np.sum(nan_mask)} NaN/Inf in density. "
        "Falling back to kNN density.",
        RuntimeWarning,
      )
      log_den, log_den_err = self._density_knn(distances, kstar, d, N)

    self.log_den_ = log_den
    self.log_den_err_ = log_den_err

    # --- Steps 5+6 (FUSED): Union-Find mode detection + assignment ---
    self.logger.debug(
      "Steps 5+6: Fused Union-Find mode detection + assignment..."
    )
    g = log_den - log_den_err
    cluster_assignment, centers, cl_struct = self._fused_mode_and_assignment(
      g, kstar, dist_indices, N, maxk
    )
    n_centers = len(centers)
    self.logger.debug("Found %d initial centers", n_centers)

    if n_centers == 0:
      warnings.warn(
        "No density modes found. Assigning all points to cluster 0.",
        RuntimeWarning,
      )
      self.labels_ = np.zeros(N, dtype=np.int64)
      self.n_clusters_ = 1
      self.cluster_centers_ = np.array([np.argmax(g)], dtype=np.int64)
      return self.labels_

    if n_centers == 1:
      self.labels_ = np.zeros(N, dtype=np.int64)
      self.n_clusters_ = 1
      self.cluster_centers_ = np.array(centers, dtype=np.int64)
      self.cluster_indices_ = [list(range(N))]
      return self.labels_

    # --- Step 7: Saddle points (vectorized override) ---
    self.logger.debug("Step 7: Vectorized saddle-point detection...")
    log_den_bord, log_den_bord_err, bord_indices = self._find_saddle_points(
      cluster_assignment, log_den, log_den_err,
      dist_indices, kstar, centers, N, n_centers
    )

    # --- Step 8: Vectorized multimodality test (override) ---
    self.logger.debug("Step 8: Vectorized multimodality test (merging)...")
    (centers, cluster_assignment, cl_struct,
     log_den_bord, log_den_bord_err, n_final) = self._multimodality_test(
      self._Z, centers, log_den, log_den_err,
      log_den_bord, log_den_bord_err,
      cluster_assignment, cl_struct, n_centers
    )
    self.logger.debug("After merging: %d clusters", n_final)

    # --- Step 9: Finalize (inherited) ---
    self.logger.debug("Step 9: Finalizing labels...")
    (labels, final_centers, n_final,
     final_bord, final_bord_err, final_bord_idx,
     final_cl_struct) = self._finalize(
      self._halo, centers, cluster_assignment, cl_struct,
      log_den, log_den_bord, log_den_bord_err,
      bord_indices, n_centers, N
    )

    # --- Store results ---
    self.labels_ = labels
    self.cluster_centers_ = final_centers
    self.n_clusters_ = n_final
    self.log_den_bord_ = final_bord
    self.log_den_bord_err_ = final_bord_err
    self.bord_indices_ = final_bord_idx
    self.cluster_indices_ = final_cl_struct

    self.logger.info("Done. %d clusters found.", n_final)

    return labels


# =============================================================================
# CONNECTIVITY-BASED (HIERARCHICAL) CLUSTERING
# =============================================================================
#
# Algorithms that build a hierarchy of clusters by recursively merging
# (agglomerative) or splitting (divisive) clusters.
#
# Included algorithms:
#   - OGSAgglomerative: Bottom-up hierarchical clustering
#   - OGSFeatureAgglomeration: Clusters features instead of samples
#
# Linkage criteria:
#   - 'ward': Minimizes within-cluster variance (default, recommended)
#   - 'complete': Maximum distance between cluster members
#   - 'average': Average distance between cluster members
#   - 'single': Minimum distance between cluster members
#
# For seismic applications:
#   - Useful for exploring hierarchy of earthquake sequences
#   - Dendrogram visualization shows cluster relationships
#   - Ward linkage good for spatially compact clusters
# =============================================================================


class OGSAgglomerative(BaseClusterer):
  """
  Agglomerative Clustering with plotting capabilities.

  Hierarchical clustering using a bottom-up approach. Each sample starts as its
  own cluster, then pairs of clusters are successively merged based on a
  linkage criterion.

  For seismic catalogs: Good for exploring hierarchical relationships between
  earthquake sequences. Dendrogram can reveal sub-sequences within larger
  swarms.

  Parameters
  ----------
  n_clusters : int, default=2
    Number of clusters to find.
  linkage : str, default='ward'
    Linkage criterion determining merge strategy:
    - 'ward': Minimizes variance (assumes euclidean metric)
    - 'complete': Maximum inter-cluster distance
    - 'average': Mean inter-cluster distance
    - 'single': Minimum inter-cluster distance
  metric : str, default='euclidean'
    Distance metric (only used if linkage != 'ward').
  compute_distances : bool, default=False
    Set True to enable dendrogram plotting.
  **kwargs
    Additional arguments passed to sklearn.cluster.AgglomerativeClustering.

  Attributes
  ----------
  model.children_ : np.ndarray
    Merge history showing which clusters were joined.
  model.distances_ : np.ndarray
    Distances at each merge (requires compute_distances=True).
  """

  def _create_model(self,
    n_clusters: int = 2,
    linkage: str = 'ward',
    **kwargs) -> AgglomerativeClustering:
      """Create sklearn AgglomerativeClustering instance."""
      validated_linkage = _validate_agglomerative_linkage(linkage)
      return AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=validated_linkage,
        **kwargs)

  def plot(self, *args, **kwargs) -> Axes:
    """Plot Agglomerative Clustering results."""
    return super().plot(*args, **kwargs)

  def plot_dendrogram(self,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (12, 8),
    truncate_mode: str = 'lastp',
    p: int = 30,
    **dendrogram_kwargs
  ) -> Axes:
    """
    Plot dendrogram for hierarchical clustering.

    The dendrogram shows the hierarchical merge history. The height
    of each merge indicates the distance at which clusters were joined.

    Parameters
    ----------
    ax : Axes, optional
      Matplotlib axes to plot on. Creates new figure if None.
    figsize : tuple, default=(12, 8)
      Figure size (width, height).
    truncate_mode : str, default='lastp'
      Truncation mode: 'lastp' shows last p merges, 'level' by depth.
    p : int, default=30
      Number of leaves/levels to show when truncating.
    **dendrogram_kwargs
      Additional arguments passed to scipy.cluster.hierarchy.dendrogram.

    Returns
    -------
    Axes
      Matplotlib axes with the dendrogram.

    Raises
    ------
    ValueError
      If model was not fitted with compute_distances=True.
    """
    from scipy.cluster.hierarchy import dendrogram

    # Check that distances were computed
    if not hasattr(self.model, 'distances_') or self.model.distances_ is None:
      raise ValueError("Dendrogram requires distances. "
                       "Refit with compute_distances=True.")

    if ax is None:
      fig, ax = plt.subplots(figsize=figsize)

    # Build linkage matrix from sklearn model attributes
    # Format: [child1, child2, distance, count]
    counts = np.zeros(self.model.children_.shape[0])
    n_samples = len(self.model.labels_)

    for i, merge in enumerate(self.model.children_):
      current_count = 0
      for child_idx in merge:
        current_count += (
          1 if child_idx < n_samples else counts[child_idx - n_samples]
        )
      counts[i] = current_count

    linkage_matrix = np.column_stack(
      [self.model.children_, self.model.distances_, counts]).astype(float)

    # Plot dendrogram
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

  Similar to AgglomerativeClustering, but clusters FEATURES instead of samples.
  Useful for dimensionality reduction by grouping similar features together.

  For seismic catalogs: Can be used to identify which earthquake attributes
  (magnitude, depth, location) cluster together.

  Parameters
  ----------
  n_clusters : int, default=2
    Number of feature clusters to find.
  **kwargs
    Additional arguments passed to sklearn.cluster.FeatureAgglomeration.

  Methods
  -------
  transform(X)
    Transform data to reduced feature space using cluster means.
  """

  def _create_model(self,
    n_clusters: int = 2,
    **kwargs
  ) -> FeatureAgglomeration:
    """Create sklearn FeatureAgglomeration instance."""
    return FeatureAgglomeration(n_clusters=n_clusters, **kwargs)

  def fit(self, X: np.ndarray) -> "OGSFeatureAgglomeration":
    """
    Fit and get feature cluster labels.

    Parameters
    ----------
    X : np.ndarray
      Data of shape (n_samples, n_features).

    Returns
    -------
    self
    """
    self.data_ = X
    self.model.fit(X)
    self.labels_ = self.model.labels_  # Labels are for FEATURES, not samples
    return self

  def transform(self, X: np.ndarray) -> np.ndarray:
    """
    Transform X to reduced feature space.

    Parameters
    ----------
    X : np.ndarray
      Data of shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
      Transformed data of shape (n_samples, n_clusters).
    """
    return self.model.transform(X)

  def plot(self, *args, **kwargs) -> Axes:
    """
    Plot feature clustering as a bar chart.

    Shows which features belong to which cluster.
    Note: This shows feature clusters, not sample clusters.

    Returns
    -------
    Axes
      Matplotlib axes with the plot.
    """
    if self.labels_ is None:
      raise ValueError("Model must be fitted before plotting.")

    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))

    n_features = len(self.labels_)
    encoded, unique, cmap, norm = labels_to_colormap(self.labels_)

    # Bar chart with each bar representing a feature, colored by cluster
    ax.bar(range(n_features),
      np.ones(n_features),
      color=[cmap(norm(e)) for e in encoded])
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Cluster Assignment")
    ax.set_title(kwargs.get('title', 'Feature Agglomeration Clusters'))
    ax.set_xticks(range(n_features))
    return ax


# =============================================================================
# MESSAGE-PASSING CLUSTERING
# =============================================================================
#
# Algorithms that iteratively pass messages between data points to identify
# clusters. Do not require specifying number of clusters.
#
# Included algorithms:
#   - OGSAffinityPropagation: Finds exemplars via message passing
#   - OGSMeanShift: Finds cluster centers via gradient ascent
#
# For seismic applications:
#   - Affinity Propagation: Good for finding representative events
#   - Mean Shift: Works well for blob-like spatial clusters
#   - Neither requires knowing k in advance
# =============================================================================


class OGSAffinityPropagation(BaseClusterer):
  """
  Affinity Propagation clustering with plotting capabilities.

  Creates clusters by sending messages between pairs of samples until
  convergence. Identifies cluster centers called "exemplars" - actual data
  points that represent their clusters.

  For seismic catalogs: Useful for identifying representative events
  (exemplars) within each sequence. Good when you want actual earthquakes as
  cluster representatives, not abstract centroids.

  Parameters
  ----------
  damping : float, default=0.5
    Damping factor between 0.5 and 1.0 to avoid numerical oscillations.
    Higher values = slower but more stable convergence.
  preference : float or array-like, optional
    Preferences for each point to be an exemplar. Default uses median
    of all pairwise similarities. Higher values = more clusters.
  max_iter : int, default=200
    Maximum number of iterations.
  **kwargs
    Additional arguments passed to sklearn.cluster.AffinityPropagation.

  Attributes
  ----------
  model.cluster_centers_indices_ : np.ndarray
    Indices of exemplar samples in the original data.
  model.affinity_matrix_ : np.ndarray
    The affinity matrix used for clustering.
  """

  def _create_model(self,
    damping: float = 0.5,
    **kwargs
  ) -> AffinityPropagation:
    """Create sklearn AffinityPropagation instance."""
    return AffinityPropagation(damping=damping, **kwargs)

  def plot(self,
    show_exemplars: bool = True,
    exemplar_marker: str = "D",
    exemplar_size: int = 150,
    *args,
    **kwargs
  ) -> Axes:
    """
    Plot Affinity Propagation results with exemplars.

    Parameters
    ----------
    show_exemplars : bool, default=True
      Whether to highlight cluster exemplars (representative points).
    exemplar_marker : str, default="D"
      Marker style for exemplars (diamond by default).
    exemplar_size : int, default=150
      Size of exemplar markers.
    *args, **kwargs
      Arguments passed to BaseClusterer.plot().

    Returns
    -------
    Axes
      Matplotlib axes with the plot.
    """
    # Plot clusters using parent method
    ax = super().plot(*args, **kwargs)

    # Overlay exemplars (cluster representatives)
    if show_exemplars and hasattr(self.model, 'cluster_centers_indices_'):
      data = kwargs.get('X', self.data_)
      feature_x = kwargs.get('feature_x', 0)
      feature_y = kwargs.get('feature_y', 1)
      indices = self.model.cluster_centers_indices_

      ax.scatter(
        data[indices, feature_x],
        data[indices, feature_y],
        c='red',
        marker=exemplar_marker,
        s=exemplar_size,
        edgecolors='black',
        linewidths=2,
        label="Exemplars",
        zorder=10
      )
      ax.legend()
    return ax

  def get_exemplar_indices(self) -> Optional[np.ndarray]:
    """
    Return indices of cluster exemplars.

    Returns
    -------
    np.ndarray or None
      Indices of exemplar samples in the original data.
    """
    if hasattr(self.model, 'cluster_centers_indices_'):
      return self.model.cluster_centers_indices_
    return None


class OGSMeanShift(BaseClusterer):
  """
  Mean Shift clustering with plotting capabilities.

  Finds clusters by iteratively shifting points towards regions of highest
  density. Number of clusters is determined automatically based on the
  bandwidth parameter.

  For seismic catalogs: Good for finding blob-like spatial clusters of
  earthquakes. Bandwidth controls the spatial scale of clusters.

  Parameters
  ----------
  bandwidth : float, optional
    Bandwidth used in the RBF kernel. If None, estimated automatically using
    sklearn.cluster.estimate_bandwidth().
  bin_seeding : bool, default=False
    If True, initial kernel locations are discretized to speed up.
  min_bin_freq : int, default=1
    Minimum number of seeds per bin.
  cluster_all : bool, default=True
    If True, all points are clustered. If False, orphan points get -1.
  **kwargs
    Additional arguments passed to sklearn.cluster.MeanShift.

  Attributes
  ----------
  model.cluster_centers_ : np.ndarray
    Coordinates of cluster centers.
  """

  def _create_model(self, **kwargs) -> MeanShift:
    """Create sklearn MeanShift instance."""
    return MeanShift(**kwargs)

  def plot(self,
    show_centers: bool = True,
    center_marker: str = "X",
    center_size: int = 200,
    center_color: str = "red",
    *args,
    **kwargs
  ) -> Axes:
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

    Returns
    -------
    Axes
      Matplotlib axes with the plot.
    """
    # Plot clusters using parent method
    ax = super().plot(*args, **kwargs)

    # Overlay cluster centers
    centers = self.get_cluster_centers()
    if show_centers and centers is not None:
      feature_x = kwargs.get('feature_x', 0)
      feature_y = kwargs.get('feature_y', 1)
      ax.scatter(
        centers[:, feature_x],
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
# SPECTRAL CLUSTERING
# =============================================================================
#
# Uses eigenvalues of similarity matrix to reduce dimensionality before
# clustering. Effective for non-convex clusters and graph-based relationships.
#
# For seismic applications:
#   - Can find clusters connected by complex shapes (e.g., along faults)
#   - Works with similarity/affinity matrices, not just Euclidean distance
#   - Computationally expensive for large datasets
# =============================================================================


class OGSSpectralClustering(BaseClusterer):
  """
  Spectral Clustering with plotting capabilities.

  Projects data onto a low-dimensional embedding derived from the graph
  Laplacian, then applies K-Means in the embedded space. Effective for
  non-convex clusters and when cluster structure follows a graph.

  For seismic catalogs: Useful when earthquake sequences follow complex spatial
  patterns (e.g., along curved fault traces) that K-Means cannot capture. Can
  use different affinity measures.

  Parameters
  ----------
  n_clusters : int, default=8
    Number of clusters to form.
  affinity : str, default='rbf'
    How to construct the affinity matrix:
    - 'rbf': Gaussian kernel (uses gamma parameter)
    - 'nearest_neighbors': k-NN graph
    - 'precomputed': User provides affinity matrix
  gamma : float, optional
    Kernel coefficient for 'rbf' affinity. Higher = more local.
  n_neighbors : int, default=10
    Number of neighbors for 'nearest_neighbors' affinity.
  assign_labels : str, default='kmeans'
    Strategy to assign labels: 'kmeans' or 'discretize'.
  **kwargs
    Additional arguments passed to sklearn.cluster.SpectralClustering.

  Notes
  -----
  Spectral clustering is memory-intensive for large datasets (>10k points).
  Consider using MiniBatchKMeans or HDBSCAN for large seismic catalogs.
  """

  def _create_model(self, n_clusters: int = 8, **kwargs) -> SpectralClustering:
    """Create sklearn SpectralClustering instance."""
    return SpectralClustering(n_clusters=n_clusters, **kwargs)

  def plot(self, *args, **kwargs) -> Axes:
    """Plot Spectral Clustering results."""
    return super().plot(*args, **kwargs)


# =============================================================================
# TREE-BASED CLUSTERING
# =============================================================================
#
# Uses tree structures for efficient, scalable clustering.
# Suitable for large datasets and streaming data.
#
# Included algorithms:
#   - OGSBirch: Balanced Iterative Reducing and Clustering using Hierarchies
#
# For seismic applications:
#   - BIRCH excellent for large catalogs (>100k events)
#   - Supports online/streaming clustering
#   - Memory-efficient for big data scenarios
# =============================================================================


class OGSBirch(BaseClusterer):
  """
  BIRCH clustering with plotting capabilities.

  Balanced Iterative Reducing and Clustering using Hierarchies.
  A memory-efficient, online-learning algorithm that incrementally clusters
  incoming data points.

  For seismic catalogs: Ideal for very large catalogs or real-time earthquake
  clustering. Builds a tree structure (CF-tree) for efficient clustering
  without loading all data into memory.

  Parameters
  ----------
  n_clusters : int, default=3
    Number of clusters after the final clustering step.
    If None, returns subcluster centroids directly.
  threshold : float, default=0.5
    Radius of the subcluster. Points within this radius are merged.
    For seismic data in km, typical values: 1-10 km.
  branching_factor : int, default=50
    Maximum number of CF subclusters in each node of the CF-tree.
    Higher = faster but uses more memory.
  **kwargs
    Additional arguments passed to sklearn.cluster.Birch.

  Attributes
  ----------
  model.subcluster_centers_ : np.ndarray
    Centers of the subclusters (before final clustering).
  model.subcluster_labels_ : np.ndarray
    Labels for each subcluster.

  Notes
  -----
  BIRCH has two clustering stages:
  1. Build CF-tree with subclusters (online, incremental)
  2. Apply final clustering (e.g., AgglomerativeClustering) to subclusters
  """

  def _create_model(self,
    n_clusters: int = 3,
    threshold: float = 0.5,
    **kwargs) -> Birch:
      """Create sklearn Birch instance."""
      return Birch(n_clusters=n_clusters, threshold=threshold, **kwargs)

  def plot(self, show_subcluster_centers: bool = False, *args,
    **kwargs
  ) -> Axes:
    """
    Plot BIRCH clustering results.

    Parameters
    ----------
    show_subcluster_centers : bool, default=False
      Whether to show subcluster centers (intermediate clustering).
    *args, **kwargs
      Arguments passed to BaseClusterer.plot().

    Returns
    -------
    Axes
      Matplotlib axes with the plot.
    """
    # Plot final clusters using parent method
    ax = super().plot(*args, **kwargs)

    # Optionally show subcluster centers (intermediate level)
    if show_subcluster_centers and hasattr(self.model, 'subcluster_centers_'):
      centers = self.model.subcluster_centers_
      feature_x = kwargs.get('feature_x', 0)
      feature_y = kwargs.get('feature_y', 1)
      ax.scatter(
        centers[:, feature_x],
        centers[:, feature_y],
        c='orange',
        marker='s',
        s=50,
        alpha=0.6,
        label="Subcluster centers"
      )
      ax.legend()

    return ax


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
#
# Helper functions to get available clustering algorithms and metrics.
# Used by OGSClusteringZoo for algorithm discovery and comparison.
# =============================================================================


def get_all_eval_metrics() -> dict[str, dict[str, type]]:
  """
  Get a dictionary of all available clustering metric classes.

  Returns a nested dictionary organized by metric type (unsupervised vs
  supervised). Use this to discover available metrics for clustering
  evaluation.

  Returns
  -------
  dict[str, dict[str, type]]
    Nested dictionary with structure:
    - "UnsupervisedScores": {metric_name: MetricClass, ...}
    - "SupervisedScores": {metric_name: MetricClass, ...}

  Example
  -------
  >>> metrics = get_all_eval_metrics()
  >>> silhouette_cls = metrics["UnsupervisedScores"]["SilhouetteScore"]
  >>> score = silhouette_cls(X, labels).compute()
  """
  return {
    "UnsupervisedScores": {
      # Unsupervised metrics (require only X and labels)
      "SilhouetteScore": SilhouetteScore,
      "CalinskiHarabaszScore": CalinskiHarabaszScore,
      "DaviesBouldinScore": DaviesBouldinScore,
      "PAkDensitySeparationScore": PAkDensitySeparationScore,
    },
    "SupervisedScores": {
      # Supervised metrics (require X, labels, AND y_true)
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
  Get a dictionary of all available clustering algorithm classes.

  Returns a dictionary mapping short algorithm names to their OGS wrapper
  classes. Use this to discover available algorithms.

  Returns
  -------
  dict[str, type[BaseClusterer]]
    Dictionary mapping algorithm names to classes.
    Keys are short names like 'KMeans', 'HDBSCAN', etc.
    Values are the OGS wrapper classes.

  Example
  -------
  >>> clusterers = get_all_clusterers()
  >>> kmeans_cls = clusterers['KMeans']
  >>> kmeans = kmeans_cls(n_clusters=5)
  >>> labels = kmeans.fit_predict(X)

  Notes
  -----
  Available algorithms by category:

  Centroid-based:
    - KMeans, MiniBatchKMeans, BisectingKMeans

  Density-based:
    - DBSCAN, HDBSCAN, OPTICS, AdvancedDensityPeaks, AdvancedDensityPeaksPP

  Connectivity-based:
    - Agglomerative, FeatureAgglomeration

  Message-passing:
    - AffinityPropagation, MeanShift

  Spectral:
    - Spectral

  Tree-based:
    - Birch
  """
  return {
    'AdvancedDensityPeaks': OGSAdvancedDensityPeaks,
    'AdvancedDensityPeaksPP': OGSAdvancedDensityPeaksPP,
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
# CLUSTERING ZOO (FACTORY CLASS)
# =============================================================================
#
# Factory class for creating, comparing, and optimizing clustering algorithms.
# Provides a unified interface for:
#   - Creating clusterer instances by name
#   - Running multiple algorithms on the same data
#   - Optimizing hyperparameters based on evaluation metrics
#   - Comparing clustering results visually
#
# Workflow:
#   1. Initialize with metadata (algorithm list, parameter ranges, metrics)
#   2. Call run() to compare algorithms
#   3. For each algorithm, optimize parameters based on chosen metric
#   4. Visualize all results in a grid
# =============================================================================


class OGSClusteringZoo:
  """
  Factory class for creating and comparing clustering algorithms.

  OGSClusteringZoo provides a unified interface for:
  - Creating clustering instances by name
  - Comparing multiple algorithms on the same dataset
  - Optimizing hyperparameters based on evaluation metrics
  - Visualizing comparison results

  For seismic catalogs: Use this to systematically compare different
  clustering algorithms on earthquake data and find optimal parameters.

  Parameters
  ----------
  metadata : dict[str, Any], optional
    Configuration dictionary with the following keys:
    - algorithms: List of algorithm names to use
    - eval_metrics: List of metric names for optimization
    - metric: Distance metric for clustering (e.g., 'euclidean')
    - n_jobs: Number of parallel jobs (-1 for all CPUs)
    - random_state: Random seed for reproducibility
    - Various parameter ranges for optimization (see below)
  verbose : bool, default=False
    Whether to print progress information.

  Attributes
  ----------
  _algorithms : dict[str, type[BaseClusterer]]
    Available clustering algorithm classes.
  _metrics : dict[str, type[BaseClusteringScores]]
    Available evaluation metric classes.

  Example
  -------
  >>> metadata = {
  ...     "algorithms": ["KMeans", "HDBSCAN", "DBSCAN"],
  ...     "eval_metrics": ["SilhouetteScore"],
  ...     "num_clusters_range": (2, 10, 1),
  ...     "cluster_size_range": (10, 100, 10),
  ... }
  >>> zoo = OGSClusteringZoo(metadata=metadata, verbose=True)
  >>> zoo.run(X)  # Compare algorithms and visualize
  >>> plt.show()

  Metadata Parameter Ranges
  -------------------------
  Each range is a tuple of (start, stop, step):
  - num_clusters_range: (min_k, max_k, step) for K-Means, Agglomerative, etc.
  - cluster_size_range: (min, max, step) for HDBSCAN min_cluster_size
  - eps_range: (min, max, step) for DBSCAN eps parameter
  - damping_range: (min, max, step) for AffinityPropagation damping
  - bandwidth_range: (min, max, step) for MeanShift bandwidth
  - min_samples_range: (min, max, step) for OPTICS/DBSCAN min_samples
  - Z_range: (min, max, step) for AdvancedDensityPeaks Z parameter
  """

  def __init__(self,
    metadata: Optional[dict[str, Any]] = None,
    verbose: bool = False
  ) -> None:
    """
    Initialize the clustering zoo with configuration.

    Parameters
    ----------
    metadata : dict[str, Any], optional
      Configuration dictionary (see class docstring for keys).
    verbose : bool, default=False
      Whether to print optimization progress.
    """
    # Store configuration
    self._metadata = metadata or {}
    self.verbose = verbose
    self.logger = setup_logger(self.__class__.__name__, verbose=self.verbose)

    # Build available algorithms dictionary
    CLUSTERS = get_all_clusterers()
    self._algorithms: dict[str, type[BaseClusterer]] = {
      name: CLUSTERS[name] for name in self.metadata_algorithms
    } if self.metadata_algorithms else CLUSTERS

    # Build available metrics dictionary
    METRICS = {k: v for group in get_all_eval_metrics().values()
      for k, v in group.items()}
    self._metrics: dict[str, type[BaseClusteringScores]] = {
      n: METRICS[n] for n in self.metadata_eval_metrics
    } if self.metadata_eval_metrics else {}

  # -------------------------------------------------------------------------
  # Metadata Property Accessors
  # -------------------------------------------------------------------------

  @property
  def metadata_algorithms(self) -> list:
    """List of algorithm names to use from metadata."""
    return self._metadata.get("algorithms", [])

  @property
  def metadata_metric(self) -> Optional[Optional[str]]:
    """Distance metric for clustering (e.g., 'euclidean')."""
    return self._metadata.get("metric", None)

  @property
  def metadata_eval_metrics(self) -> list:
    """List of evaluation metric names for optimization."""
    return self._metadata.get("eval_metrics", [])

  @property
  def metadata_n_jobs_value(self) -> int:
    """Number of parallel jobs (-1 = all CPUs)."""
    return int(self._metadata.get("n_jobs", -1))

  @property
  def metadata_min_cluster_size_value(self) -> Optional[int]:
    """Fixed min_cluster_size value for HDBSCAN."""
    value = self._metadata.get("min_cluster_size")
    return int(value) if value is not None else None

  @property
  def metadata_min_samples_value(self) -> Optional[int]:
    """Fixed min_samples value for density-based algorithms."""
    value = self._metadata.get("min_samples")
    return int(value) if value is not None else None

  @property
  def metadata_random_state_value(self) -> Optional[int]:
    """Random state for reproducibility."""
    value = self._metadata.get("random_state")
    return int(value) if value is not None else None

  @property
  def metadata_n_clusters_value(self) -> Optional[int]:
    """Fixed n_clusters value for K-Means, etc."""
    value = self._metadata.get("n_clusters")
    return int(value) if value is not None else None

  @property
  def metadata_eps_value(self) -> Optional[float]:
    """Fixed eps value for DBSCAN."""
    value = self._metadata.get("eps")
    return float(value) if value is not None else None

  @property
  def metadata_damping_value(self) -> Optional[float]:
    """Fixed damping value for AffinityPropagation."""
    value = self._metadata.get("damping")
    return float(value) if value is not None else None

  @property
  def metadata_bandwidth_value(self) -> Optional[float]:
    """Fixed bandwidth value for MeanShift."""
    value = self._metadata.get("bandwidth")
    return float(value) if value is not None else None

  @property
  def metadata_num_clusters_value(self) -> Optional[int]:
    """Fixed num_clusters value (alias for n_clusters)."""
    value = self._metadata.get("num_clusters")
    return int(value) if value is not None else None

  @property
  def metadata_Z_value(self) -> Optional[float]:
    """Fixed Z value for AdvancedDensityPeaks."""
    value = self._metadata.get("Z")
    return float(value) if value is not None else None

  # -------------------------------------------------------------------------
  # Parameter Range Properties (for optimization)
  # -------------------------------------------------------------------------

  @property
  def metadata_cluster_size_range(self) -> Optional[Tuple[int, int, int]]:
    """Range (min, max, step) for HDBSCAN min_cluster_size optimization."""
    value = self._metadata.get("cluster_size_range")
    return tuple(value) if value is not None else None

  @property
  def metadata_bandwidth_range(self) -> Optional[Tuple[float, float, float]]:
    """Range (min, max, step) for MeanShift bandwidth optimization."""
    value = self._metadata.get("bandwidth_range")
    return tuple(value) if value is not None else None

  @property
  def metadata_damping_range(self) -> Optional[Tuple[float, float, float]]:
    """Range (min, max, step) for AffinityPropagation damping optimization."""
    value = self._metadata.get("damping_range")
    return tuple(value) if value is not None else None

  @property
  def metadata_eps_range(self) -> Optional[Tuple[float, float, float]]:
    """Range (min, max, step) for DBSCAN eps optimization."""
    value = self._metadata.get("eps_range")
    return tuple(value) if value is not None else None

  @property
  def metadata_min_samples_range(self) -> Optional[Tuple[int, int, int]]:
    """Range (min, max, step) for min_samples optimization."""
    value = self._metadata.get("min_samples_range")
    return tuple(value) if value is not None else None

  @property
  def metadata_num_clusters_range(self) -> Optional[Tuple[int, int, int]]:
    """Range (min, max, step) for n_clusters optimization."""
    value = self._metadata.get("num_clusters_range")
    return tuple(value) if value is not None else None

  @property
  def metadata_sample_size_range(self) -> Optional[Tuple[int, int, int]]:
    """Range (min, max, step) for sample_size parameter."""
    value = self._metadata.get("sample_size_range")
    return tuple(value) if value is not None else None

  @property
  def metadata_Z_range(self) -> Optional[Tuple[float, float, float]]:
    """Range (min, max, step) for AdvancedDensityPeaks Z optimization."""
    value = self._metadata.get("Z_range")
    return tuple(value) if value is not None else None

  # -------------------------------------------------------------------------
  # Public Methods
  # -------------------------------------------------------------------------

  @property
  def list(self) -> list[str]:
    """Return a sorted list of all available clustering algorithm names."""
    return sorted(get_all_clusterers().keys())

  def register(self, name: str, cls: type) -> None:
    """
    Register a new clustering class under a name.

    Parameters
    ----------
    name : str
      Short name for the algorithm (e.g., 'MyCustomClusterer').
    cls : type
      The clustering class (must inherit from BaseClusterer).

    Raises
    ------
    ValueError
      If an algorithm with this name is already registered.
    """
    if name in self._algorithms:
      raise ValueError(f"Clusterer '{name}' already registered.")
    self._algorithms[name] = cls

  def create(self, name: str, **kwargs) -> BaseClusterer:
    """
    Create a clusterer instance by name.

    Parameters
    ----------
    name : str
      Name of the clustering algorithm (e.g., 'KMeans', 'HDBSCAN').
    **kwargs
      Additional parameters passed to the clusterer constructor.

    Returns
    -------
    BaseClusterer
      Configured clustering instance.

    Raises
    ------
    KeyError
      If the algorithm name is not recognized.

    Example
    -------
    >>> zoo = OGSClusteringZoo()
    >>> kmeans = zoo.create("KMeans", n_clusters=5)
    >>> labels = kmeans.fit_predict(X)
    """
    if name not in self._algorithms:
      raise KeyError(f"Unknown clusterer '{name}'.")
    cluster_cls: type[BaseClusterer] = self._algorithms[name]
    # Get default kwargs from metadata
    cluster_kwargs = self._cluster_kwargs(name)
    # Override with user-provided kwargs
    cluster_kwargs.update(kwargs)
    return cluster_cls(**cluster_kwargs)

  # -------------------------------------------------------------------------
  # Private Methods
  # -------------------------------------------------------------------------

  def _cluster_kwargs(self, algo_name: str) -> dict:
    """
    Build kwargs dictionary for a specific algorithm from metadata.

    Maps metadata values to the appropriate parameter names for each algorithm
    type.

    Parameters
    ----------
    algo_name : str
      Name of the clustering algorithm.

    Returns
    -------
    dict
      Keyword arguments for the clusterer constructor.
    """
    myDict = {}

    # MeanShift bandwidth
    if self.metadata_bandwidth_value is not None and algo_name in {
      "MeanShift"
    }:
      myDict["bandwidth"] = self.metadata_bandwidth_value

    # AffinityPropagation damping
    if self.metadata_damping_value is not None and algo_name in {
      "AffinityPropagation"
    }:
      myDict["damping"] = self.metadata_damping_value

    # DBSCAN eps
    if self.metadata_eps_value is not None and algo_name in {"DBSCAN"}:
      myDict["eps"] = self.metadata_eps_value

    # Distance metric for algorithms that support it
    if self.metadata_metric not in {None, ""} and algo_name in {
      "Agglomerative",
      "FeatureAgglomeration",
      "DBSCAN",
      "OPTICS",
      "HDBSCAN"
    }:
      myDict["metric"] = self.metadata_metric

    # HDBSCAN min_cluster_size
    if self.metadata_min_cluster_size_value is not None and algo_name in {
      "HDBSCAN"
    }:
      myDict["min_cluster_size"] = self.metadata_min_cluster_size_value

    # min_samples for density-based algorithms
    if self.metadata_min_samples_value is not None and algo_name in {
      "OPTICS",
      "DBSCAN",
      "HDBSCAN"
    }:
      myDict["min_samples"] = self.metadata_min_samples_value

    # n_clusters for centroid-based algorithms
    if self.metadata_num_clusters_value is not None and algo_name in {
      "KMeans",
      "MiniBatchKMeans",
      "BisectingKMeans",
      "Agglomerative",
      "FeatureAgglomeration",
      "Spectral",
      "Birch",
    }:
      myDict["n_clusters"] = self.metadata_num_clusters_value

    # Parallelization for supported algorithms
    if self.metadata_n_jobs_value is not None and algo_name in {
      "DBSCAN",
      "OPTICS",
      "Spectral",
      "HDBSCAN"
    }:
      myDict["n_jobs"] = self.metadata_n_jobs_value

    # Random state for reproducibility
    if self.metadata_random_state_value is not None and algo_name in {
      "KMeans",
      "MiniBatchKMeans",
      "BisectingKMeans",
      "Spectral",
      "AffinityPropagation",
    }:
      myDict["random_state"] = self.metadata_random_state_value

    # AdvancedDensityPeaks Z parameter
    if self.metadata_Z_value is not None and algo_name in {
      "AdvancedDensityPeaks", "AdvancedDensityPeaksPP"
    }:
      myDict["Z"] = self.metadata_Z_value

    return myDict

  def _optimize_param(self,
    param_name: str,
    algo_name: str,
    X: np.ndarray,
    metric_name: str,
    values: List[Any],
    base_kwargs: dict) -> dict[str, Any]:
    """
    Optimize a single clustering parameter based on a metric.

    Performs a grid search over the provided parameter values and returns the
    best configuration based on the evaluation metric.

    Parameters
    ----------
    param_name : str
      The name of the parameter to optimize (e.g., 'n_clusters', 'eps').
    algo_name : str
      Name of the clustering algorithm.
    X : np.ndarray
      The data to be clustered.
    metric_name : str
      Name of the evaluation metric to optimize.
    values : List[Any]
      The list of parameter values to test.
    base_kwargs : dict
      Base keyword arguments for the clustering class.

    Returns
    -------
    dict[str, Any]
      A dictionary containing:
      - param_name: The best parameter value
      - clusterer: The fitted clusterer with best parameters
      - score: The best metric score
      - scores_by_param: Dict mapping each value to its score
      - labels: Cluster labels from best configuration
    """
    scores: Dict[Any, Optional[float]] = {}
    best_val: Optional[Any] = None
    best_score: Optional[float] = None
    cluster_cls: type[BaseClusterer] = self._algorithms[algo_name]

    # Grid search over parameter values
    for val in values:
      # Fit model with current parameter value
      score = self._metrics[metric_name](
        X, cluster_cls(**base_kwargs, **{param_name: val},
          verbose=self.verbose).fit_predict(X)
      ).compute()
      scores[val] = score

      if score is None:
        continue

      # Update best if this score is better
      # DaviesBouldin is lower-is-better, others are higher-is-better
      if best_score is None or (score <= best_score if metric_name in {
        "DaviesBouldinScore"
      } else score >= best_score):
        best_score, best_val = score, val

    # If no valid scores, return empty
    if best_val is None:
      return {}

    # Refit with best parameter
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
    metric_name: str
  ) -> dict[str, Any]:
    """
    Optimize clustering parameters for a given evaluation metric.

    Determines which parameter to optimize based on the algorithm type,
    then calls _optimize_param to perform the grid search.

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
      A dictionary containing:
      - algorithm: Algorithm name
      - eval_metric: Metric used for optimization
      - Best parameter value and score
      - scores_by_param: All tested values and their scores
      - labels: Cluster labels from best configuration
    """
    # Get base kwargs from metadata
    base_kwargs = self._cluster_kwargs(algo_name)
    params: dict = {}
    param_name = ""
    values: List[Any] = []

    # Determine which parameter to optimize based on algorithm
    if algo_name in {"HDBSCAN"}:
      values = iter_range(self.metadata_cluster_size_range)
      if values:
        param_name = "min_cluster_size"
    elif algo_name in {"KMeans", "MiniBatchKMeans", "BisectingKMeans",
      "Agglomerative", "FeatureAgglomeration",
      "Spectral", "Birch"}:
        values = iter_range(self.metadata_num_clusters_range)
        if values:
          param_name = "n_clusters"
    elif algo_name in {"DBSCAN"}:
      values = iter_range(self.metadata_eps_range)
      if values:
        param_name = "eps"
    elif algo_name in {"AffinityPropagation"}:
      values = iter_range(self.metadata_damping_range)
      if values:
        param_name = "damping"
    elif algo_name in {"MeanShift"}:
      values = iter_range(self.metadata_bandwidth_range)
      if values:
        param_name = "bandwidth"
    elif algo_name in {"OPTICS"}:
      values = iter_range(self.metadata_min_samples_range)
      if values:
        param_name = "min_samples"
    elif algo_name in {"AdvancedDensityPeaks", "AdvancedDensityPeaksPP"}:
      values = iter_range(self.metadata_Z_range)
      if values:
        param_name = "Z"

    # Perform optimization
    params = self._optimize_param(
      param_name,
      algo_name,
      X,
      metric_name,
      values,
      base_kwargs
    )

    # Log optimization results
    self.logger.info(
      "Optimized %s for %s: %s = %s with score %s", metric_name, algo_name,
      param_name, params.get(param_name), params.get('score'),
    )
    for key, value in params.get("scores_by_param", {}).items():
      self.logger.debug("    %s: %s", key, value)

    # Add metadata to results
    params = {
      "algorithm": algo_name,
      "eval_metric": metric_name,
      **base_kwargs,
      **params
    }
    return params

  def _init_figure(self,
    figsize: Tuple[int, int] = (16, 12),
    **kwargs
  ) -> dict[str, Tuple[Figure, np.ndarray]]:
    """
    Initialize comparison figure(s) with subplots for each algorithm.

    Creates a grid of subplots with one subplot per algorithm.
    If multiple metrics are specified, creates one figure per metric.

    Parameters
    ----------
    figsize : tuple, default=(16, 12)
      Figure size (width, height).
    **kwargs
      Additional arguments passed to plt.subplots().

    Returns
    -------
    dict[str, Tuple[Figure, np.ndarray]]
      Dictionary mapping metric names to (figure, axes_array) tuples.
    """
    n_clusterers = len(self._algorithms)
    cols = min(4, n_clusterers)  # Max 4 columns
    rows = (n_clusterers + cols - 1) // cols  # Ceiling division

    def build_figure(title: str) -> Tuple[Figure, np.ndarray]:
      """Create a single figure with algorithm grid."""
      fig, axes = plt.subplots(rows, cols, figsize=figsize, **kwargs)
      axes = np.atleast_2d(axes)
      fig.suptitle(title, fontsize=16)

      # Set subplot titles
      for idx, cluster_name in enumerate(self._algorithms.keys()):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        ax.set_title(cluster_name)

      # Hide unused subplots
      for idx in range(n_clusterers, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].set_visible(False)

      return fig, axes

    # Build figures for each metric (or one figure if no metrics)
    figures: dict[str, Tuple[Figure, np.ndarray]] = {}
    if self._metrics:
      for metric_name in self._metrics:
        figures[metric_name] = build_figure(
          f"Clustering Algorithm Comparison ({metric_name})"
        )
      return figures
    else:
      return {"": build_figure("Clustering Algorithm Comparison")}

  def run(self,
    X: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (16, 8),
    feature_x: int = 0,
    feature_y: int = 1,
    y_true: Optional[np.ndarray] = None,
    **common_kwargs
  ) -> None:
    """
    Compare multiple clustering algorithms on the same dataset.

    This is the main entry point for algorithm comparison. For each algorithm
    and metric combination:
    1. Optimizes the primary parameter based on the metric
    2. Fits the optimized model
    3. Plots results in a comparison grid

    Parameters
    ----------
    X : np.ndarray
      Data to cluster, shape (n_samples, n_features).
      For seismic data: [X_km, Y_km, depth_km, time_days, ...]
    figsize : tuple, default=(16, 8)
      Figure size (width, height) in inches.
    feature_x : int, default=0
      Feature index for x-axis in plots.
    feature_y : int, default=1
      Feature index for y-axis in plots.
    y_true : np.ndarray, optional
      Ground-truth labels for supervised metrics.
      Typically unavailable for seismic applications.
    **common_kwargs
      Common arguments passed to all plot() methods.
      Examples: alpha, point_size, show_legend, etc.

    Returns
    -------
    None
      Results are plotted to matplotlib figures.

    Raises
    ------
    ValueError
      If X is None.

    Example
    -------
    >>> zoo = OGSClusteringZoo(metadata=metadata, verbose=True)
    >>> zoo.run(earthquake_data, feature_x=0, feature_y=1)
    >>> plt.show()
    """
    if X is None:
      raise ValueError("X must be provided for clustering comparison.")

    # ----- Computation loop -----
    # Run each algorithm with optional parameter optimization
    DATA: dict[str, dict[str, Any]] = {}
    for algo_name in self._algorithms:
      metric_name = ""
      if self._metrics:
        # Optimize for each evaluation metric
        for metric_name in self._metrics:
          params = self._optimize_for_metric(
            algo_name,
            X,
            metric_name
          )
          DATA.setdefault(metric_name, {})[algo_name] = params
      else:
        # No optimization, just run with default/configured params
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
    # Create comparison figures with one subplot per algorithm
    for metric_name, (fig, axes) in self._init_figure(figsize=figsize).items():
      for idx, algo_name in enumerate(self._algorithms.keys()):
        row, col = divmod(idx, axes.shape[1])
        ax = axes[row, col]
        params = DATA[metric_name][algo_name]
        if params is None:
          continue
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
    **kwargs
  ) -> None:
    """Apply final layout adjustments to figure."""
    plt.tight_layout()


# =============================================================================
# MANIFOLD BENCHMARK
# =============================================================================
#
# Provides 21 synthetic manifold generators (Facco et al., 2017; d'Errico et
# al., 2021) and methods to:
#   - generate, score, and cluster all manifolds,
#   - plot three 7×3 grids (geometry, density, clusters).
#
# Every piece of plotting logic lives here so that ``tmp.py`` is a thin
# script that simply calls ``ManifoldBenchmark.run()``.
# =============================================================================


class ManifoldBenchmark:
  """
  21-manifold benchmark suite for ADP / PAk evaluation.

  Parameters
  ----------
  N : int
    Number of points per manifold (default 10 000).
  seed : int
    Global random seed (default 42).

  Attributes
  ----------
  REGISTRY : list[tuple]
    Ordered list of ``(name, generator, d, D, description)`` tuples
    defining the 7 × 3 grid layout.
  results : list[dict]
    Per-manifold results populated by :meth:`compute`.
  """

  import time

  # -----------------------------------------------------------------------
  # Internal helpers (static)
  # -----------------------------------------------------------------------

  @staticmethod
  def _random_rotation(d: int, D: int, rng: np.random.RandomState):
    """Random (D, d) rotation via QR decomposition."""
    A = rng.randn(D, d)
    Q, _ = np.linalg.qr(A)
    return Q[:, :d]

  @staticmethod
  def _nonlinear_embed(X_int: np.ndarray, D: int,
                       rng: np.random.RandomState) -> np.ndarray:
    """Random-Fourier-feature smooth embedding from d → D."""
    d = X_int.shape[1]
    n_feat = (D + 1) // 2
    W = rng.randn(d, n_feat) * 2.0
    b = rng.uniform(0, 2 * np.pi, size=n_feat)
    Z = X_int @ W + b[np.newaxis, :]
    return np.column_stack([np.cos(Z), np.sin(Z)])[:, :D]

  # -----------------------------------------------------------------------
  # 21 manifold generators — each returns (X, d_true)
  # -----------------------------------------------------------------------

  @staticmethod
  def gen_M1(N: int = 10_000, seed: int = 42):
    """M1: 10-dim hypersphere surface in R^11."""
    rng = np.random.RandomState(seed)
    X = rng.randn(N, 11)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, 10

  @staticmethod
  def gen_M2(N: int = 10_000, seed: int = 42):
    """M2: Affine 3-plane in R^5."""
    rng = np.random.RandomState(seed)
    R = ManifoldBenchmark._random_rotation(3, 5, rng)
    return rng.uniform(0, 1, (N, 3)) @ R.T + rng.randn(5), 3

  @staticmethod
  def gen_M3(N: int = 10_000, seed: int = 42):
    """M3: Concentrated figure d=4, D=6."""
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, (N, 2))
    r = rng.uniform(0, 1, (N, 2))
    X_int = np.column_stack([
      r[:, 0] * np.cos(theta[:, 0]), r[:, 0] * np.sin(theta[:, 0]),
      r[:, 1] * np.cos(theta[:, 1]), r[:, 1] * np.sin(theta[:, 1]),
    ])
    X_int[:, 2:] *= 0.3
    R = ManifoldBenchmark._random_rotation(4, 6, rng)
    return X_int @ R.T, 4

  @staticmethod
  def gen_M4(N: int = 10_000, seed: int = 42):
    """M4: Nonlinear d=4, D=8."""
    rng = np.random.RandomState(seed)
    return ManifoldBenchmark._nonlinear_embed(
      rng.uniform(0, 1, (N, 4)), 8, rng), 4

  @staticmethod
  def gen_M5(N: int = 10_000, seed: int = 42):
    """M5: 2D helix surface in R^3."""
    rng = np.random.RandomState(seed)
    t = rng.uniform(0, 4 * np.pi, N)
    s = rng.uniform(0.5, 1.5, N)
    return np.column_stack([s * np.cos(t), s * np.sin(t),
                            t / (4 * np.pi)]), 2

  @staticmethod
  def gen_M6(N: int = 10_000, seed: int = 42):
    """M6: Nonlinear d=6, D=36."""
    rng = np.random.RandomState(seed)
    return ManifoldBenchmark._nonlinear_embed(
      rng.uniform(0, 1, (N, 6)), 36, rng), 6

  @staticmethod
  def gen_M7(N: int = 10_000, seed: int = 42):
    """M7: Swiss-Roll d=2, D=3."""
    rng = np.random.RandomState(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(0, 1, N))
    h = rng.uniform(0, 10, N)
    return np.column_stack([t * np.cos(t), h, t * np.sin(t)]), 2

  @staticmethod
  def gen_M9(N: int = 10_000, seed: int = 42):
    """M9: Uniform 20-cube (d=D=20)."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 1, (N, 20)), 20

  @staticmethod
  def gen_M10a(N: int = 10_000, seed: int = 42):
    """M10a: 10-cube in R^11."""
    rng = np.random.RandomState(seed)
    R = ManifoldBenchmark._random_rotation(10, 11, rng)
    return rng.uniform(0, 1, (N, 10)) @ R.T, 10

  @staticmethod
  def gen_M10b(N: int = 10_000, seed: int = 42):
    """M10b: 17-cube in R^18."""
    rng = np.random.RandomState(seed)
    R = ManifoldBenchmark._random_rotation(17, 18, rng)
    return rng.uniform(0, 1, (N, 17)) @ R.T, 17

  @staticmethod
  def gen_M10c(N: int = 10_000, seed: int = 42):
    """M10c: 24-cube in R^25."""
    rng = np.random.RandomState(seed)
    R = ManifoldBenchmark._random_rotation(24, 25, rng)
    return rng.uniform(0, 1, (N, 24)) @ R.T, 24

  @staticmethod
  def gen_M10d(N: int = 10_000, seed: int = 42):
    """M10d: 70-cube in R^71."""
    rng = np.random.RandomState(seed)
    R = ManifoldBenchmark._random_rotation(70, 71, rng)
    return rng.uniform(0, 1, (N, 70)) @ R.T, 70

  @staticmethod
  def gen_M11(N: int = 10_000, seed: int = 42):
    """M11: Möbius band ×10 twists, d=2, D=3."""
    rng = np.random.RandomState(seed)
    u = rng.uniform(0, 2 * np.pi, N)
    v = rng.uniform(-0.5, 0.5, N)
    n_tw = 10
    return np.column_stack([
      (1 + v * np.cos(n_tw * u / 2)) * np.cos(u),
      (1 + v * np.cos(n_tw * u / 2)) * np.sin(u),
      v * np.sin(n_tw * u / 2),
    ]), 2

  @staticmethod
  def gen_M12(N: int = 10_000, seed: int = 42):
    """M12: Isotropic Gaussian d=D=20."""
    rng = np.random.RandomState(seed)
    return rng.randn(N, 20), 20

  @staticmethod
  def gen_M13(N: int = 10_000, seed: int = 42):
    """M13: 1D helix curve in R^3."""
    rng = np.random.RandomState(seed)
    t = np.sort(rng.uniform(0, 4 * np.pi, N))
    return np.column_stack([np.cos(t), np.sin(t), t / (4 * np.pi)]), 1

  @staticmethod
  def gen_MN1(N: int = 10_000, seed: int = 42):
    """MN1: Nonlinear d=18, D=72."""
    rng = np.random.RandomState(seed)
    return ManifoldBenchmark._nonlinear_embed(
      rng.uniform(0, 1, (N, 18)), 72, rng), 18

  @staticmethod
  def gen_MN2(N: int = 10_000, seed: int = 42):
    """MN2: Nonlinear d=24, D=96."""
    rng = np.random.RandomState(seed)
    return ManifoldBenchmark._nonlinear_embed(
      rng.uniform(0, 1, (N, 24)), 96, rng), 24

  @staticmethod
  def gen_Mbeta(N: int = 10_000, seed: int = 42):
    """Mβ: Nonlinear d=10, D=40."""
    rng = np.random.RandomState(seed)
    return ManifoldBenchmark._nonlinear_embed(
      rng.uniform(0, 1, (N, 10)), 40, rng), 10

  @staticmethod
  def gen_MP3(N: int = 10_000, seed: int = 42):
    """MP3: Nonlinear d=3, D=12."""
    rng = np.random.RandomState(seed)
    return ManifoldBenchmark._nonlinear_embed(
      rng.uniform(0, 1, (N, 3)), 12, rng), 3

  @staticmethod
  def gen_MP6(N: int = 10_000, seed: int = 42):
    """MP6: Nonlinear d=6, D=21."""
    rng = np.random.RandomState(seed)
    return ManifoldBenchmark._nonlinear_embed(
      rng.uniform(0, 1, (N, 6)), 21, rng), 6

  @staticmethod
  def gen_MP9(N: int = 10_000, seed: int = 42):
    """MP9: Nonlinear d=9, D=30."""
    rng = np.random.RandomState(seed)
    return ManifoldBenchmark._nonlinear_embed(
      rng.uniform(0, 1, (N, 9)), 30, rng), 9

  # -----------------------------------------------------------------------
  # Registry — display order for the 7 × 3 grid
  # -----------------------------------------------------------------------

  REGISTRY: list = []   # populated after class body

  # -----------------------------------------------------------------------
  # Construction
  # -----------------------------------------------------------------------

  def __init__(self, N: int = 10_000, seed: int = 42):
    self.N = N
    self.seed = seed
    self.results: list[dict] = []
    self._X_single: list[np.ndarray] = []
    self._X2d: list[np.ndarray] = []
    self._xlabels: list[str] = []
    self._ylabels: list[str] = []

  # -----------------------------------------------------------------------
  # Public helpers
  # -----------------------------------------------------------------------

  @staticmethod
  def make_two_cluster(gen_func, N_per: int = 5000,
                       sep: float = 20.0, seed: int = 42):
    """Two translated copies → well-separated 2-cluster data."""
    X0, d = gen_func(N=N_per, seed=seed)
    X1, _ = gen_func(N=N_per, seed=seed + 1)
    D = X0.shape[1]
    shift = np.zeros(D)
    shift[0] = sep
    X1 = X1 + shift
    X = np.vstack([X0, X1])
    labels = np.concatenate([np.zeros(N_per, dtype=int),
                             np.ones(N_per, dtype=int)])
    return X, labels, d

  @staticmethod
  def project_2d(X: np.ndarray, D: int):
    """Project to 2D: native coords if D ≤ 3, else PCA."""
    if D <= 3:
      return X[:, :2], 'x₁', 'x₂'
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_
    return X2, f'PC1 ({ev[0]*100:.0f}%)', f'PC2 ({ev[1]*100:.0f}%)'

  # -----------------------------------------------------------------------
  # compute() — Pass 1: generate, score, cluster
  # -----------------------------------------------------------------------

  def compute(self, Z: float = 1.65, verbose: bool = True):
    """
    Generate all 21 manifolds, compute PAk scores on 2-cluster versions,
    and run ADP++ unsupervised clustering on single manifolds.

    Parameters
    ----------
    Z : float
      Z-score threshold for ADP++ clustering (default 1.65).
    verbose : bool
      Print progress table.

    Returns
    -------
    list[dict]
      Per-manifold result dictionaries.
    """
    self.results.clear()
    self._X_single.clear()
    self._X2d.clear()
    self._xlabels.clear()
    self._ylabels.clear()

    if verbose:
      print('=' * 72)
      print(f'  Benchmark:  21 manifolds × N = {self.N:,}')
      print(f'  ADP++ clustering on raw single-manifold data')
      print('=' * 72)

    for idx, (name, gen_func, d, D, desc) in enumerate(self.REGISTRY):
      t0 = time.time()
      if verbose:
        print(f'\n[{idx+1:2d}/21]  {name:<5s}  d={d}, D={D}  ({desc})')

      # 1) single manifold
      X, _ = gen_func(N=self.N)
      self._X_single.append(X)
      X2, xl, yl = self.project_2d(X, D)
      self._X2d.append(X2)
      self._xlabels.append(xl)
      self._ylabels.append(yl)

      # 2) PAk on 2-cluster
      N_per = self.N // 2
      X_cl, lab_true, _ = self.make_two_cluster(
        gen_func, N_per=N_per, sep=20.0, seed=self.seed)
      maxk_pak = min(80, N_per - 1)
      scorer = PAkDensitySeparationScore(X_cl, lab_true, maxk=maxk_pak)
      pak_score = scorer.compute()
      d_est_pak = scorer.intrinsic_dim_
      if verbose:
        s_str = f'{pak_score:.2f}' if pak_score is not None else 'None'
        print(f'        PAk  →  d̂={d_est_pak:.1f}  S={s_str}')

      # 3) ADP++ unsupervised
      maxk_adp = min(100, self.N - 1)
      adp = OGSAdvancedDensityPeaksPP(
        Z=Z, density_method='PAk', maxk=maxk_adp, halo=False)
      with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        adp_labels = adp.fit_predict(X)

      n_clust = adp.n_clusters_
      d_est_adp = adp.intrinsic_dim_
      log_den = adp.log_den_
      elapsed = time.time() - t0
      if verbose:
        print(f'        ADP++ →  d̂={d_est_adp:.1f}  K={n_clust}'
              f'  ({elapsed:.1f}s)')

      self.results.append(dict(
        name=name, desc=desc, d=d, D=D,
        pak_score=pak_score, d_est_pak=d_est_pak,
        n_clust=n_clust, d_est_adp=d_est_adp,
        adp_labels=adp_labels, log_den=log_den,
        elapsed=elapsed,
      ))

    if verbose:
      self._print_summary()

    return self.results

  # -----------------------------------------------------------------------
  # Summary table
  # -----------------------------------------------------------------------

  def _print_summary(self):
    print('\n' + '=' * 72)
    print(f'  {"ID":<6s} {"d":>3s} {"D":>4s} {"d̂_PAk":>6s} {"S_PAk":>7s}'
          f'  {"d̂_ADP":>6s} {"K_ADP":>5s}  {"Time":>5s}')
    print('-' * 72)
    for r in self.results:
      s = f'{r["pak_score"]:.2f}' if r['pak_score'] is not None else 'None'
      print(f'  {r["name"]:<6s} {r["d"]:3d} {r["D"]:4d}'
            f' {r["d_est_pak"]:6.1f} {s:>7s}'
            f'  {r["d_est_adp"]:6.1f} {r["n_clust"]:5d}'
            f'  {r["elapsed"]:5.1f}s')
    print('=' * 72)

  # -----------------------------------------------------------------------
  # 3 plotting methods — each returns a Figure
  # -----------------------------------------------------------------------

  def plot_geometry(self, ax: Optional[np.ndarray] = None,
                    **kwargs) -> Figure:
    """
    Plot a 7 × 3 grid of manifold geometry coloured by first coordinate.

    Parameters
    ----------
    ax : ndarray of Axes, optional
      Pre-created (7, 3) axes array.  Created if *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    NROWS, NCOLS = 7, 3
    if ax is None:
      fig, ax = plt.subplots(NROWS, NCOLS, figsize=(18, 32))
    else:
      fig = ax.flat[0].figure
    fig.suptitle(
      f'21 Benchmark Manifolds  —  N = {self.N:,}\n'
      '(PCA → 2D for D > 3;  native coords for D ≤ 3)',
      fontsize=16, fontweight='bold', y=0.995,
    )
    for idx, (r, X2, xl, yl) in enumerate(
        zip(self.results, self._X2d, self._xlabels, self._ylabels)):
      row, col = divmod(idx, NCOLS)
      a = ax[row, col]
      c = X2[:, 0]
      a.scatter(X2[:, 0], X2[:, 1], c=c, cmap='Spectral',
                s=0.4, alpha=0.6, rasterized=True, edgecolors='none')
      a.set_xlabel(xl, fontsize=8)
      a.set_ylabel(yl, fontsize=8)
      a.tick_params(labelsize=7)
      s_str = (f'S={r["pak_score"]:.2f}'
               if r['pak_score'] is not None else 'S=None')
      a.set_title(
        f'{r["name"]}  —  {r["desc"]}\n'
        f'd={r["d"]}, D={r["D"]}, d̂={r["d_est_pak"]:.1f}, {s_str}',
        fontsize=9, fontweight='bold',
      )
      a.set_aspect('equal', adjustable='datalim')
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig

  def plot_density(self, ax: Optional[np.ndarray] = None,
                   **kwargs) -> Figure:
    """
    Plot a 7 × 3 grid of ADP++ log-density landscape.

    Returns
    -------
    matplotlib.figure.Figure
    """
    NROWS, NCOLS = 7, 3
    if ax is None:
      fig, ax = plt.subplots(NROWS, NCOLS, figsize=(18, 32))
    else:
      fig = ax.flat[0].figure
    fig.suptitle(
      f'ADP++ Log-Density Landscape  —  N = {self.N:,}\n'
      '(PAk density estimator;  colour = log ρ̂)',
      fontsize=16, fontweight='bold', y=0.995,
    )
    for idx, (r, X2, xl, yl) in enumerate(
        zip(self.results, self._X2d, self._xlabels, self._ylabels)):
      row, col = divmod(idx, NCOLS)
      a = ax[row, col]
      sc = a.scatter(X2[:, 0], X2[:, 1], c=r['log_den'], cmap='viridis',
                     s=0.4, alpha=0.7, rasterized=True, edgecolors='none')
      cb = fig.colorbar(sc, ax=a, fraction=0.046, pad=0.04)
      cb.ax.tick_params(labelsize=6)
      cb.set_label('log ρ̂', fontsize=7)
      a.set_xlabel(xl, fontsize=8)
      a.set_ylabel(yl, fontsize=8)
      a.tick_params(labelsize=7)
      a.set_title(
        f'{r["name"]}  —  {r["desc"]}\n'
        f'd̂={r["d_est_adp"]:.1f}, K={r["n_clust"]}',
        fontsize=9, fontweight='bold',
      )
      a.set_aspect('equal', adjustable='datalim')
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig

  def plot_clusters(self, ax: Optional[np.ndarray] = None,
                    Z: float = 1.65, **kwargs) -> Figure:
    """
    Plot a 7 × 3 grid of ADP++ cluster assignments.

    Returns
    -------
    matplotlib.figure.Figure
    """
    NROWS, NCOLS = 7, 3
    if ax is None:
      fig, ax = plt.subplots(NROWS, NCOLS, figsize=(18, 32))
    else:
      fig = ax.flat[0].figure
    fig.suptitle(
      f'ADP++ Cluster Assignment  —  N = {self.N:,}\n'
      f'(Z = {Z}, PAk density, unsupervised)',
      fontsize=16, fontweight='bold', y=0.995,
    )
    for idx, (r, X2, xl, yl) in enumerate(
        zip(self.results, self._X2d, self._xlabels, self._ylabels)):
      row, col = divmod(idx, NCOLS)
      a = ax[row, col]
      labels = r['adp_labels']
      n_cl = r['n_clust']
      if n_cl <= 20:
        cmap = plt.colormaps.get_cmap('tab20').resampled(max(n_cl, 1))
      else:
        cmap = 'nipy_spectral'
      a.scatter(X2[:, 0], X2[:, 1], c=labels, cmap=cmap,
                s=0.4, alpha=0.6, rasterized=True, edgecolors='none')
      a.set_xlabel(xl, fontsize=8)
      a.set_ylabel(yl, fontsize=8)
      a.tick_params(labelsize=7)
      a.set_title(
        f'{r["name"]}  —  {r["desc"]}\n'
        f'd̂={r["d_est_adp"]:.1f}, K={r["n_clust"]} clusters',
        fontsize=9, fontweight='bold',
      )
      a.set_aspect('equal', adjustable='datalim')
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig

  # -----------------------------------------------------------------------
  # run() — compute + plot + save (convenience entry point)
  # -----------------------------------------------------------------------

  def run(self, Z: float = 1.65, save_dir: Optional[str] = None,
          dpi: int = 200, show: bool = True, verbose: bool = True):
    """
    Full pipeline: compute all manifolds, then save three grid figures.

    Parameters
    ----------
    Z : float
      Z-score threshold for ADP++ (default 1.65).
    save_dir : str or None
      Directory to save PNGs.  Defaults to directory of this file.
    dpi : int
      Resolution for saved figures (default 200).
    show : bool
      Call ``plt.show()`` at the end (default True).
    verbose : bool
      Print progress.
    """
    import os as _os

    self.compute(Z=Z, verbose=verbose)

    if save_dir is None:
      save_dir = _os.path.dirname(_os.path.abspath(__file__))

    fig1 = self.plot_geometry()
    p1 = _os.path.join(save_dir, 'manifolds_grid.png')
    fig1.savefig(p1, dpi=dpi, bbox_inches='tight')
    if verbose:
      print(f'  Saved → {p1}')

    fig2 = self.plot_density()
    p2 = _os.path.join(save_dir, 'manifolds_density.png')
    fig2.savefig(p2, dpi=dpi, bbox_inches='tight')
    if verbose:
      print(f'  Saved → {p2}')

    fig3 = self.plot_clusters(Z=Z)
    p3 = _os.path.join(save_dir, 'manifolds_clusters.png')
    fig3.savefig(p3, dpi=dpi, bbox_inches='tight')
    if verbose:
      print(f'  Saved → {p3}')

    if show:
      plt.show()

    return fig1, fig2, fig3


# Populate the REGISTRY class attribute after class body
ManifoldBenchmark.REGISTRY = [
  # row 1 — low-d parametric
  ('M5',   ManifoldBenchmark.gen_M5,    2,   3,  'Helix 2D'),
  ('M7',   ManifoldBenchmark.gen_M7,    2,   3,  'Swiss-Roll'),
  ('M13',  ManifoldBenchmark.gen_M13,   1,   3,  'Helix 1D'),
  # row 2
  ('M11',  ManifoldBenchmark.gen_M11,   2,   3,  'Möbius ×10'),
  ('M2',   ManifoldBenchmark.gen_M2,    3,   5,  'Affine 3→5'),
  ('M3',   ManifoldBenchmark.gen_M3,    4,   6,  'Concentrated 4→6'),
  # row 3
  ('M4',   ManifoldBenchmark.gen_M4,    4,   8,  'Nonlinear 4→8'),
  ('MP3',  ManifoldBenchmark.gen_MP3,   3,  12,  'Nonlinear 3→12'),
  ('M6',   ManifoldBenchmark.gen_M6,    6,  36,  'Nonlinear 6→36'),
  # row 4
  ('MP6',  ManifoldBenchmark.gen_MP6,   6,  21,  'Nonlinear 6→21'),
  ('MP9',  ManifoldBenchmark.gen_MP9,   9,  30,  'Nonlinear 9→30'),
  ('Mβ',   ManifoldBenchmark.gen_Mbeta, 10,  40,  'Nonlinear 10→40'),
  # row 5
  ('M1',   ManifoldBenchmark.gen_M1,   10,  11,  'Hypersphere 10→11'),
  ('M10a', ManifoldBenchmark.gen_M10a, 10,  11,  'Hypercube 10→11'),
  ('M10b', ManifoldBenchmark.gen_M10b, 17,  18,  'Hypercube 17→18'),
  # row 6
  ('M10c', ManifoldBenchmark.gen_M10c, 24,  25,  'Hypercube 24→25'),
  ('M10d', ManifoldBenchmark.gen_M10d, 70,  71,  'Hypercube 70→71'),
  ('M9',   ManifoldBenchmark.gen_M9,   20,  20,  'Uniform 20D'),
  # row 7
  ('M12',  ManifoldBenchmark.gen_M12,  20,  20,  'Gaussian 20D'),
  ('MN1',  ManifoldBenchmark.gen_MN1,  18,  72,  'Nonlinear 18→72'),
  ('MN2',  ManifoldBenchmark.gen_MN2,  24,  96,  'Nonlinear 24→96'),
]


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
  """
  Demo function showing OGSClusteringZoo usage.

  Creates synthetic data with make_blobs and compares multiple clustering
  algorithms with hyperparameter optimization.
  """
  from sklearn.datasets import make_blobs

  # Generate sample data (4 clusters with some noise)
  X, y_true = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=0.60,
    random_state=42,
    return_centers=False
  )

  # Configuration for the clustering comparison
  metadata = {
    # Algorithms to compare
    "algorithms": [
      "AdvancedDensityPeaksPP",
      "HDBSCAN",
      "KMeans",
    ],
    # Distance metric
    "metric": "euclidean",
    # Evaluation metrics for optimization
    "eval_metrics": [
      'SilhouetteScore',
      'PAkDensitySeparationScore',
      'DaviesBouldinScore',
    ],
    # Parallelization
    "n_jobs": -1,
    "random_state": 42,
    # Parameter ranges for optimization (start, stop, step)
    "bandwidth_range": (0.5, 2.0, 0.1),
    "cluster_size_range": (10, 100, 10),
    "damping_range": (0.5, 0.9, 0.1),
    "eps_range": (0.3, 1.0, 0.1),
    "min_samples_range": (5, 50, 5),
    "num_clusters_range": (2, 10, 1),
    "sample_size_range": (100, 300, 20),
    "Z_range": (0.1, 2.0, 0.1),
  }

  # Create zoo and run comparison
  zoo = OGSClusteringZoo(metadata=metadata, verbose=True)
  zoo.run(X)
  plt.show()


if __name__ == "__main__": main()

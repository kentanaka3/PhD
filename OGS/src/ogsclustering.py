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
                  ┌─────────────────────────────────────┐
                  │          ogsclustering.py           │
                  ├─────────────────────────────────────┤
                  │         BaseClusterer (ABC)         │
                  │ ┌─────────────────────────────────┐ │
                  │ │ • fit_predict(X)                │ │
                  │ │ • plot(X, ax, ...)              │ │
                  │ │ • plot_3d(X, ax, ...)           │ │
                  │ │ • get_cluster_centers()         │ │
                  │ │ • n_clusters()                  │ │
                  │ └─────────────────────────────────┘ │
                  │           ▲                         │
                  │     ┌─────┴─────┐                   │
                  │     │           │                   │
                  │  OGSKMeans  OGSDBSCAN  ...          │
                  ├─────────────────────────────────────┤
                  │      BaseClusteringScores (ABC)     │
                  │           ▲                         │
                  │  ┌────────┴────────┐                │
                  │  │                 │                │
                  │ SilhouetteScore   AdjustedRandScore │
                  ├─────────────────────────────────────┤
                  │          OGSClusteringZoo           │
                  │ ┌─────────────────────────────────┐ │
                  │ │ • create(name, **kwargs)        │ │
                  │ │ • run(X, ...)                   │ │
                  │ │ • _optimize_for_metric(...)     │ │
                  │ └─────────────────────────────────┘ │
                  └─────────────────────────────────────┘

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
from abc import ABC, abstractmethod         # Abstract base class support
from typing import (                         # Type hints for better IDE support
  Optional,                                # Optional type annotation
  Tuple,                                   # Tuple type annotation
  Union,                                   # Union of multiple types
  Any,                                     # Any type (escape hatch)
  List,                                    # List type annotation
  Callable,                                # Function type annotation
  Dict                                     # Dictionary type annotation
)

# =============================================================================
# THIRD-PARTY LIBRARY IMPORTS
# =============================================================================

# Numerical computing
import numpy as np                           # Array operations and linear algebra
import pandas as pd                          # DataFrame operations (unused but available)

# Visualization
import matplotlib.pyplot as plt              # Main plotting interface
from matplotlib.axes import Axes             # 2D axes type for type hints
from mpl_toolkits.mplot3d.axes3d import Axes3D  # 3D axes for 3D scatter plots
from matplotlib.colors import BoundaryNorm  # Discrete colormap normalization
from matplotlib.figure import Figure        # Figure type for type hints
from matplotlib import colormaps            # Colormap registry

# Advanced density-based clustering (DADApy library)
import dadapy as ddpy                        # Density peak clustering algorithms

# =============================================================================
# SCIKIT-LEARN CLUSTERING ALGORITHMS
# =============================================================================
# Import all supported clustering algorithms from sklearn

from sklearn.cluster import (
  KMeans,                                  # Standard K-Means clustering
  MiniBatchKMeans,                         # Mini-batch variant for large data
  AffinityPropagation,                     # Message-passing exemplar clustering
  MeanShift,                               # Mode-seeking density clustering
  SpectralClustering,                      # Graph Laplacian-based clustering
  AgglomerativeClustering,                 # Hierarchical bottom-up clustering
  DBSCAN,                                  # Density-based spatial clustering
  HDBSCAN,                                 # Hierarchical DBSCAN  # type: ignore
  OPTICS,                                  # Ordering points clustering
  Birch,                                   # Balanced iterative clustering
  BisectingKMeans,                         # Divisive hierarchical K-Means
  FeatureAgglomeration,                    # Feature-space hierarchical clustering
)

# =============================================================================
# SCIKIT-LEARN CLUSTERING EVALUATION METRICS
# =============================================================================
# Import metrics for evaluating clustering quality

from sklearn.metrics import (
  # -------------------------------------------------------------------------
  # Unsupervised metrics (require only X and labels, no ground truth)
  # -------------------------------------------------------------------------
  silhouette_score,                        # Mean silhouette coefficient [-1, 1]
  calinski_harabasz_score,                 # Variance ratio criterion (higher=better)
  davies_bouldin_score,                    # Average similarity ratio (lower=better)

  # -------------------------------------------------------------------------
  # Supervised metrics (require ground truth labels y_true)
  # -------------------------------------------------------------------------
  adjusted_rand_score,                     # Rand index adjusted for chance
  normalized_mutual_info_score,            # Normalized mutual information
  adjusted_mutual_info_score,              # AMI adjusted for chance
  homogeneity_score,                       # Clusters contain only one class
  completeness_score,                      # Class members in same cluster
  v_measure_score,                         # Harmonic mean of homogeneity/completeness
  fowlkes_mallows_score,                   # Geometric mean of precision/recall

  # Utility
  pairwise_distances,                      # Compute distance matrix
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def labels_to_colormap(
  labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Any, Any]:
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
  # Find all unique labels (may include -1 for noise)
  unique = np.unique(labels)

  # Create mapping from original labels to sequential indices
  label_to_idx = {lab: i for i, lab in enumerate(unique)}

  # Apply mapping to all labels
  encoded = np.vectorize(label_to_idx.get, otypes=[int])(labels)

  # Create discrete colormap with exactly len(unique) colors
  cmap = colormaps.get_cmap("Paired").resampled(len(unique))

  # Create boundary norm for discrete color assignment
  # Boundaries at -0.5, 0.5, 1.5, ... ensure each integer maps to one color
  norm = BoundaryNorm(np.arange(-0.5, len(unique) + 0.5), cmap.N)

  return encoded, unique, cmap, norm


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
  >>> iter_range([1, 2, 5])         # Returns [1, 2, 5]
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
      Cluster labels for each sample. Label -1 indicates noise
      (for density-based algorithms like DBSCAN/HDBSCAN).
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

    Visualizes cluster assignments with color-coded points. Supports
    separate styling for noise points (label=-1) and cluster points.

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
        Cluster centers of shape (n_clusters, n_features), or None
        if the algorithm doesn't compute centers.
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
# For seismic applications, unsupervised metrics are more common since
# ground truth cluster labels are typically unavailable. Silhouette score
# is particularly useful for comparing clustering quality across algorithms.
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
  - 0: Overlapping clusters
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


class OGSKMeans(BaseClusterer):
  """
  K-Means clustering with plotting capabilities.

  Partitions n samples into k clusters by minimizing within-cluster
  sum of squares (inertia). Cluster centers are called centroids.

  For seismic catalogs: Useful when number of earthquake sequences
  is known or can be estimated (e.g., from magnitude distribution).

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

  def plot(self,
    show_centers: bool = True,
    center_marker: str = "X",
    center_size: int = 200,
    center_color: str = "red",
    *args,
    **kwargs
  ) -> Axes:
    """
    Plot K-Means clustering results with optional cluster centers.

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
    # First plot the clustered points using parent method
    ax = super().plot(*args, **kwargs)

    # Overlay cluster centers if requested
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


class OGSMiniBatchKMeans(BaseClusterer):
  """
  Mini-Batch K-Means clustering with plotting capabilities.

  Faster variant of K-Means that uses mini-batches to reduce computation
  time. Slightly worse results than regular K-Means but much faster for
  large datasets.

  For seismic catalogs: Recommended for catalogs with >50,000 events
  where full K-Means would be too slow.

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

  def plot(self,
    show_centers: bool = True,
    center_marker: str = "X",
    center_size: int = 200,
    center_color: str = "red",
    *args,
    **kwargs
  ) -> Axes:
    """Plot Mini-Batch K-Means results with optional cluster centers."""
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
        edgecolors="black",
        linewidths=2,
        label="Centers"
      )
      ax.legend()
    return ax


class OGSBisectingKMeans(BaseClusterer):
  """
  Bisecting K-Means clustering with plotting capabilities.

  Hierarchical approach: starts with all data in one cluster, then
  repeatedly splits the cluster with largest inertia. Provides a
  tree-like structure of clusters.

  For seismic catalogs: Useful for exploring cluster hierarchy,
  e.g., identifying sub-clusters within a seismic swarm.

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

  def plot(self,
    show_centers: bool = True,
    center_marker: str = "X",
    center_size: int = 200,
    center_color: str = "red",
    *args,
    **kwargs
  ) -> Axes:
    """Plot Bisecting K-Means results with optional cluster centers."""
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
        edgecolors="black",
        linewidths=2,
        label="Centers"
      )
      ax.legend()
    return ax


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

  For seismic catalogs: Excellent for identifying earthquake sequences
  without knowing the number of clusters. Set eps based on expected
  spatial extent of sequences (e.g., 5-10 km for local clusters).

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
    Distance threshold for cluster selection. Can be used to
    merge clusters closer than this threshold.
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

  def _create_model(self, min_cluster_size: int = 5, **kwargs) -> HDBSCAN:
    """Create sklearn HDBSCAN instance."""
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
  Creates a reachability plot that can be used to extract clusters
  at different density levels. More flexible than DBSCAN.

  For seismic catalogs: Useful for exploring multi-scale clustering
  structure, e.g., identifying nested sequences within larger swarms.

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

    The reachability plot shows the cluster structure. Valleys in the
    plot indicate clusters; deeper valleys = denser clusters.

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
  Advanced Density Peaks clustering (via dadapy library).

  Implementation of the Density Peaks algorithm with advanced
  features from the dadapy package.

  For seismic catalogs: Alternative density-based approach that
  finds cluster centers as local density maxima.

  Parameters
  ----------
  **kwargs
    Arguments passed to dadapy.Data.compute_clustering_ADP().

  Notes
  -----
  Requires the dadapy library to be installed.
  This is a placeholder implementation that may need extension.
  """

  def __init__(self, **kwargs):
    """Initialize Advanced Density Peaks clusterer."""
    super().__init__(**kwargs)

  def fit_predict(self, X: np.ndarray) -> np.ndarray:
    """
    Fit the model and return cluster labels.

    Uses dadapy's ADP (Advanced Density Peaks) algorithm.

    Parameters
    ----------
    X : np.ndarray
      Data to cluster, shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
      Cluster labels for each sample.
    """
    # Use dadapy's Data class for ADP clustering
    labels_ = ddpy.Data(X, verbose=self.verbose).compute_clustering_ADP(
      **self._kwargs
    )
    self.labels_ = labels_
    return labels_

  def _create_model(self, **kwargs) -> Any:
    """No sklearn model - uses dadapy directly."""
    pass

  def plot(self, *args, **kwargs) -> Axes:
    """Plot Advanced Density Peaks clustering results."""
    return super().plot(*args, **kwargs)

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

  Hierarchical clustering using a bottom-up approach. Each sample
  starts as its own cluster, then pairs of clusters are successively
  merged based on a linkage criterion.

  For seismic catalogs: Good for exploring hierarchical relationships
  between earthquake sequences. Dendrogram can reveal sub-sequences
  within larger swarms.

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

  Similar to AgglomerativeClustering, but clusters FEATURES instead
  of samples. Useful for dimensionality reduction by grouping similar
  features together.

  For seismic catalogs: Can be used to identify which earthquake
  attributes (magnitude, depth, location) cluster together.

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
# Algorithms that iteratively pass messages between data points to
# identify clusters. Do not require specifying number of clusters.
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
  convergence. Identifies cluster centers called "exemplars" - actual
  data points that represent their clusters.

  For seismic catalogs: Useful for identifying representative events
  (exemplars) within each sequence. Good when you want actual earthquakes
  as cluster representatives, not abstract centroids.

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

  Finds clusters by iteratively shifting points towards regions of
  highest density. Number of clusters is determined automatically
  based on the bandwidth parameter.

  For seismic catalogs: Good for finding blob-like spatial clusters
  of earthquakes. Bandwidth controls the spatial scale of clusters.

  Parameters
  ----------
  bandwidth : float, optional
    Bandwidth used in the RBF kernel. If None, estimated automatically
    using sklearn.cluster.estimate_bandwidth().
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

  For seismic catalogs: Useful when earthquake sequences follow complex
  spatial patterns (e.g., along curved fault traces) that K-Means cannot
  capture. Can use different affinity measures.

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
  A memory-efficient, online-learning algorithm that incrementally
  clusters incoming data points.

  For seismic catalogs: Ideal for very large catalogs or real-time
  earthquake clustering. Builds a tree structure (CF-tree) for
  efficient clustering without loading all data into memory.

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

  Returns a nested dictionary organized by metric type (unsupervised
  vs supervised). Use this to discover available metrics for clustering
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

  Returns a dictionary mapping short algorithm names to their
  OGS wrapper classes. Use this to discover available algorithms.

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
    - DBSCAN, HDBSCAN, OPTICS, AdvancedDensityPeaks

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
    metadata: dict[str, Any] = {},
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

    Maps metadata values to the appropriate parameter names for each
    algorithm type.

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
      "AdvancedDensityPeaks"
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

    Performs a grid search over the provided parameter values and
    returns the best configuration based on the evaluation metric.

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
    elif algo_name in {"AdvancedDensityPeaks"}:
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

    # Print optimization results if verbose
    if self.verbose:
      print(f"Optimized {metric_name} for {algo_name}: {param_name} = "
        f"{params.get(param_name)} with score {params.get('score')}")
      for key, value in params["scores_by_param"].items():
        print(f"    {key}: {value}")

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

  @abstractmethod
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

    This is the main entry point for algorithm comparison. For each
    algorithm and metric combination:
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
# MAIN ENTRY POINT
# =============================================================================


def main():
  """
  Demo function showing OGSClusteringZoo usage.

  Creates synthetic data with make_blobs and compares multiple
  clustering algorithms with hyperparameter optimization.
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
      "AdvancedDensityPeaks",
      "HDBSCAN",
    ],
    # Distance metric
    "metric": "euclidean",
    # Evaluation metrics for optimization
    "eval_metrics": [
      'SilhouetteScore',
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

"""
=============================================================================
OGS Clustering Test Suite - Unit Tests for Seismic Event Clustering & ADP/PAk
=============================================================================

OVERVIEW:
Unit test suite for ``ogsclustering.py``. Validates clustering utilities,
algorithm wrappers, high-performance Advanced Density Peaks (ADP), and PAk
density peak pointer-jumping algorithms.

TEST CASES & INVARIANTS:
  1. TestOGSClusteringUtils:
     - Range iterator tuple and list parsing; validation of invalid bounds.
     - Noise-aware categorical colormap generation for cluster visualizations.
  2. TestOGSClusteringModels:
     - Model instantiation and parameter validation across K-Means and DBSCAN.
     - Model zoo factory registration, kwargs propagation, and metric tuning.
     - Unfitted model center handling and exception handling.
  3. TestOGSClusteringPlotting:
     - Verification that 2D and 3D plotting calls enforce fitted model state.
  4. Advanced Density Peaks & PAk:
     - Density estimation, parent-link assignment, and pointer compression.

USAGE:
python -m unittest OGS/test/testogsclustering.py

DEPENDENCIES:
- unittest: unit test framework
  - numpy / pandas: feature arrays and coordinates
  - scikit-learn / dadapy: clustering algorithms and distance metrics
  - ogsclustering: clustering implementations under test

AUTHORS:
  - 健
  - Istituto Nazionale di Oceanografia e di Geofisica Sperimentale (OGS)
    Centro di Ricerche Sismologiche (CRS)
  - Università degli Studi di Trieste (UniTS)
    Dipartimento di Matematica, Informatica e Geoscienze (MIGe)
    Applied Data Science and Artificial Intelligence (ADSAI)
  - Terabit Network for Research and Academic Big Data in Italy (TeRABIT)
    Consorzio Interuniversitario del Nord-Est per il Calcolo Automatico (CINECA)
=============================================================================
"""

from OGS.src import ogsclustering as OGSCL
from sklearn.datasets import make_blobs
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from pathlib import Path
import unittest
from unittest.mock import patch
import warnings
import tempfile
from functools import cache

import numpy as np

import matplotlib
matplotlib.use("Agg")


# Silence warnings during tests
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# MODULE-LEVEL HELPERS
# =============================================================================


def _make_3blob_data():
  """Helper: 600-point 3-blob dataset with well-separated clusters."""
  return make_blobs(
      n_samples=600, centers=3, n_features=2, cluster_std=0.6, random_state=42,
  )


# =============================================================================
# HELPER: create clustered datasets from manifold generators
# =============================================================================


def make_well_separated(
    gen_func, N_per_cluster: int = 500, n_clusters: int = 2,
    separation: float = 20.0, seed: int = 42
):
  """
  Create well-separated clusters by generating n_clusters copies of a
  manifold, each translated by `separation` along orthogonal axes.

  Returns
  -------
  X : np.ndarray, shape (N_per_cluster * n_clusters, D)
  labels : np.ndarray of int, shape (N_per_cluster * n_clusters,)
  d_true : int   (intrinsic dimension)
  """
  parts = []
  labels_list = []
  d_true = None
  for c in range(n_clusters):
    X_c, d_true = gen_func(N=N_per_cluster, seed=seed + c)
    D = X_c.shape[1]
    shift = np.zeros(D)
    shift[c % D] = separation * (c + 1)
    X_c = X_c + shift
    parts.append(X_c)
    labels_list.append(np.full(N_per_cluster, c, dtype=int))
  X = np.vstack(parts)
  labels = np.concatenate(labels_list)
  return X, labels, d_true


# =============================================================================
# DADAPY TUTORIAL DATA HELPERS
# =============================================================================

DATA_URL_DIHEDRALS = "https://figshare.com/ndownloader/files/36359700"
DATA_URL_DISTANCES = "https://figshare.com/ndownloader/files/36359697"
TUTORIAL_DATA_DIR = Path(__file__).resolve().parent / "data"

def _tutorial_file(url: str, filename: str) -> Path:
  """Require a local tutorial dataset; never download or replace files."""
  filepath = TUTORIAL_DATA_DIR / filename
  instructions = (
      f"Download manually from {url} and save as OGS/test/data/{filename}."
  )
  if not filepath.is_file():
    raise FileNotFoundError(f"Tutorial dataset is missing: {filepath}. {instructions}")
  if filepath.stat().st_size == 0:
    raise ValueError(f"Tutorial dataset is empty: {filepath}. {instructions}")
  return filepath


def _load_dihedrals():
  """Load and select 15 dihedral angles from CLN025 trajectory."""
  path = _tutorial_file(
      DATA_URL_DIHEDRALS,
      'cln025traj_dihedrals_decimated_equilibrated.npy',
  )
  all_dihedrals = np.load(path)
  coords = [1, 4, 5, 7, 10, 12, 13, 14, 15, 16, 17, 18, 19, 24, 25]
  selected = all_dihedrals[:, coords]
  return selected


def _load_distances():
  """Load heavy atom distances from CLN025 trajectory."""
  path = _tutorial_file(
      DATA_URL_DISTANCES,
      'cln025traj_distances_decimated_equilibrated.npy',
  )
  return np.load(path)


@cache
def _tutorial_data():
  """Load tutorial arrays once per process, only when requested by a test."""
  return _load_dihedrals(), _load_distances()


# =========================================================================
# GENERAL OGSClustering Tests (utils, models, plotting, metrics)
# =========================================================================


class TestOGSClusteringUtils(unittest.TestCase):
  def test_iter_range_tuple(self):
    values = (0.0, 1.0, 0.2)
    expected = list(np.arange(*values))
    self.assertEqual(OGSCL.iter_range(values), expected)

  def test_iter_range_list(self):
    values = [1, 2, 3]
    self.assertEqual(OGSCL.iter_range(values), values)

  def test_iter_range_invalid(self):
    self.assertEqual(OGSCL.iter_range(5), [])

  def test_labels_to_colormap(self):
    labels = np.array([2, 2, -1, 3])
    encoded, unique, cmap, norm = OGSCL.labels_to_colormap(labels)
    self.assertTrue(np.array_equal(unique, np.array([-1, 2, 3])))
    self.assertTrue(np.array_equal(encoded, np.array([1, 1, 0, 2])))
    self.assertIsNotNone(cmap)
    self.assertIsNotNone(norm)
    self.assertEqual(cmap.N, 3)
    np.testing.assert_array_equal(norm.boundaries, [-0.5, 0.5, 1.5, 2.5])
    np.testing.assert_array_equal(norm(encoded), encoded)

  def test_labels_to_colormap_no_noise(self):
    labels = np.array([0, 1, 1])
    encoded, unique, cmap, norm = OGSCL.labels_to_colormap(labels)
    self.assertTrue(np.array_equal(unique, np.array([0, 1])))
    self.assertTrue(np.array_equal(encoded, np.array([0, 1, 1])))
    self.assertIsNotNone(cmap)
    self.assertIsNotNone(norm)
    self.assertEqual(cmap.N, 2)
    np.testing.assert_array_equal(norm.boundaries, [-0.5, 0.5, 1.5])

  def test_labels_to_colormap_all_noise(self):
    labels = np.array([-1, -1, -1])
    encoded, unique, cmap, norm = OGSCL.labels_to_colormap(labels)
    self.assertTrue(np.array_equal(unique, np.array([-1])))
    self.assertTrue(np.array_equal(encoded, np.array([0, 0, 0])))
    self.assertIsNotNone(cmap)
    self.assertIsNotNone(norm)
    self.assertEqual(cmap.N, 1)
    np.testing.assert_array_equal(norm.boundaries, [-0.5, 0.5])

  def test_labels_to_colormap_empty_input_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "empty labels array"):
      OGSCL.labels_to_colormap(np.array([], dtype=int))


class TestOGSClusteringModels(unittest.TestCase):
  def test_kmeans_basic_fit(self):
    X, _ = make_blobs(n_samples=120, centers=3, n_features=2,
                      random_state=0)
    model = OGSCL.OGSKMeans(n_clusters=3, random_state=0)
    labels = model.fit_predict(X)
    self.assertEqual(len(labels), X.shape[0])
    self.assertEqual(model.n_clusters(), 3)
    centers = model.get_cluster_centers()
    self.assertIsNotNone(centers)
    self.assertEqual(centers.shape, (3, 2))

  def test_n_clusters_unfitted(self):
    model = OGSCL.OGSKMeans(n_clusters=2)
    self.assertEqual(model.n_clusters(), 0)

  def test_get_cluster_centers_none(self):
    X, _ = make_blobs(n_samples=50, centers=2, n_features=2,
                      random_state=1)
    model = OGSCL.OGSDBSCAN(eps=0.1, min_samples=3)
    model.fit_predict(X)
    self.assertIsNone(model.get_cluster_centers())

  def test_zoo_create_kmeans(self):
    metadata = {"algorithms": ["KMeans"], "num_clusters": 2,
                "random_state": 0}
    zoo: OGSCL.OGSClusteringZoo = OGSCL.OGSClusteringZoo(metadata=metadata)
    clusterer = zoo.create("KMeans")
    self.assertIsInstance(clusterer, OGSCL.OGSKMeans)
    self.assertEqual(clusterer.model.n_clusters, 2)

  def test_zoo_create_dbscan_params(self):
    metadata = {
        "algorithms": ["DBSCAN"],
        "eps": 0.3,
        "min_samples": 4
    }
    zoo: OGSCL.OGSClusteringZoo = OGSCL.OGSClusteringZoo(metadata=metadata)
    clusterer = zoo.create("DBSCAN")
    self.assertIsInstance(clusterer, OGSCL.OGSDBSCAN)
    self.assertEqual(clusterer.model.eps, 0.3)
    self.assertEqual(clusterer.model.min_samples, 4)

  def test_zoo_list_contains_kmeans(self):
    zoo: OGSCL.OGSClusteringZoo = OGSCL.OGSClusteringZoo()
    self.assertIn("KMeans", zoo.list)

  def test_zoo_create_unknown_raises(self):
    zoo: OGSCL.OGSClusteringZoo = OGSCL.OGSClusteringZoo()
    with self.assertRaises(KeyError):
      zoo.create("NotAClusterer")

  def test_zoo_cluster_kwargs_metric(self):
    metadata = {"algorithms": ["DBSCAN"], "metric": "euclidean"}
    zoo: OGSCL.OGSClusteringZoo = OGSCL.OGSClusteringZoo(metadata=metadata)
    clusterer = zoo.create("DBSCAN")
    self.assertEqual(clusterer.model.metric, "euclidean")

  def test_zoo_optimize_for_metric_kmeans(self):
    X, _ = make_blobs(n_samples=80, centers=3, n_features=2,
                      random_state=2)
    metadata = {
        "algorithms": ["KMeans"],
        "eval_metrics": ["SilhouetteScore"],
        "num_clusters_range": (2, 5, 1),
        "random_state": 0
    }
    zoo: OGSCL.OGSClusteringZoo = OGSCL.OGSClusteringZoo(metadata=metadata)
    params = zoo._optimize_for_metric("KMeans", X, "SilhouetteScore")
    self.assertEqual(params["algorithm"], "KMeans")
    self.assertEqual(params["eval_metric"], "SilhouetteScore")
    self.assertIn("clusterer", params)
    self.assertEqual(len(params["labels"]), X.shape[0])


class TestOGSClusteringPlotting(unittest.TestCase):
  def test_plot_requires_fit(self):
    model = OGSCL.OGSKMeans(n_clusters=2, random_state=0)
    with self.assertRaises(ValueError):
      model.plot()

  def test_plot_3d_requires_fit(self):
    model = OGSCL.OGSKMeans(n_clusters=2, random_state=0)
    with self.assertRaises(ValueError):
      model.plot_3d()

  def test_plot_returns_axes(self):
    X, _ = make_blobs(n_samples=60, centers=2, n_features=2,
                      random_state=3)
    model = OGSCL.OGSKMeans(n_clusters=2, random_state=0)
    model.fit_predict(X)
    ax = model.plot()
    self.assertIsInstance(ax, Axes)
    plt.close(ax.figure)

  def test_plot_3d_returns_axes(self):
    X, _ = make_blobs(n_samples=60, centers=2, n_features=3,
                      random_state=4)
    model = OGSCL.OGSKMeans(n_clusters=2, random_state=0)
    model.fit_predict(X)
    ax = model.plot_3d()
    self.assertIsInstance(ax, Axes)
    plt.close(ax.figure)

  def test_dbscan_highlight_core(self):
    X, _ = make_blobs(n_samples=80, centers=2, n_features=2,
                      random_state=5)
    model = OGSCL.OGSDBSCAN(eps=0.5, min_samples=5)
    model.fit_predict(X)
    ax = model.plot(highlight_core=True)
    self.assertIsInstance(ax, Axes)
    plt.close(ax.figure)

  def test_zoo_run_basic(self):
    X, _ = make_blobs(n_samples=50, centers=2, n_features=2,
                      random_state=6)
    metadata = {"algorithms": ["KMeans", "DBSCAN"], "eps": 0.4}
    zoo: OGSCL.OGSClusteringZoo = OGSCL.OGSClusteringZoo(metadata=metadata)
    zoo.run(X, feature_x=0, feature_y=1)
    plt.close("all")


class TestOGSClusteringMetrics(unittest.TestCase):
  def test_unsupervised_metrics(self):
    X, _ = make_blobs(n_samples=60, centers=2, n_features=2,
                      random_state=7)
    model = OGSCL.OGSKMeans(n_clusters=2, random_state=0)
    labels = model.fit_predict(X)
    metric = OGSCL.SilhouetteScore(X, labels).compute()
    self.assertIsInstance(metric, float)

  def test_supervised_metrics_none_without_labels(self):
    X, _ = make_blobs(n_samples=40, centers=2, n_features=2,
                      random_state=8)
    model = OGSCL.OGSKMeans(n_clusters=2, random_state=0)
    labels = model.fit_predict(X)
    metric = OGSCL.AdjustedRandScore(X, labels, None).compute()
    self.assertIsNone(metric)

  def test_supervised_metrics_with_labels(self):
    X, y = make_blobs(n_samples=40, centers=2, n_features=2,
                      random_state=9)
    model = OGSCL.OGSKMeans(n_clusters=2, random_state=0)
    labels = model.fit_predict(X)
    metric = OGSCL.AdjustedRandScore(X, labels, y).compute()
    self.assertIsInstance(metric, float)


# =========================================================================
# Advanced Density Peaks Tests
# =========================================================================


class TestOGSAdvancedDensityPeaks(unittest.TestCase):
  # -- 1. Import and instantiation --
  def test_instantiation_defaults(self):
    adp = OGSCL.OGSAdvancedDensityPeaks()
    self.assertIsNotNone(adp)

  def test_valid_density_methods(self):
    for method in ('PAk', 'kNN', 'kstarNN', 'kpeaks'):
      adp = OGSCL.OGSAdvancedDensityPeaks(density_method=method)
      self.assertIsNotNone(adp)

  def test_invalid_density_method_raises(self):
    with self.assertRaises(ValueError):
      OGSCL.OGSAdvancedDensityPeaks(density_method='健')

  # -- 2. PAk density on 3-blob data --
  def test_pak_3blobs(self):
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaks(Z=1.65, density_method='PAk')
    labels = adp.fit_predict(X)
    self.assertEqual(len(labels), X.shape[0])
    self.assertGreaterEqual(adp.n_clusters_, 1)

  # -- 3. kNN density method --
  def test_knn_3blobs(self):
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaks(density_method='kNN', k=10)
    labels = adp.fit_predict(X)
    self.assertEqual(len(labels), X.shape[0])
    self.assertGreaterEqual(adp.n_clusters_, 1)

  # -- 4. kstarNN density method --
  def test_kstarnn_3blobs(self):
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaks(density_method='kstarNN')
    labels = adp.fit_predict(X)
    self.assertEqual(len(labels), X.shape[0])
    self.assertGreaterEqual(adp.n_clusters_, 1)

  # -- 5. kpeaks density method --
  def test_kpeaks_3blobs(self):
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaks(density_method='kpeaks')
    labels = adp.fit_predict(X)
    self.assertEqual(len(labels), X.shape[0])
    self.assertGreaterEqual(adp.n_clusters_, 1)

  # -- 6. halo=True --
  def test_halo_mode(self):
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaks(halo=True, density_method='PAk')
    labels = adp.fit_predict(X)
    self.assertEqual(len(labels), X.shape[0])
    self.assertGreaterEqual(adp.n_clusters_, 1)
    # Halo labels are either cluster IDs (>=0) or -1 (noise)
    self.assertTrue(np.all(labels >= -1))

  # -- 7. Too few points --
  def test_too_few_points(self):
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    adp = OGSCL.OGSAdvancedDensityPeaks()
    with self.assertWarns(RuntimeWarning):
      labels = adp.fit_predict(X)
    self.assertEqual(len(labels), 2)

  # -- 8. NaN input --
  def test_nan_input_raises(self):
    X = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
    adp = OGSCL.OGSAdvancedDensityPeaks()
    with self.assertRaises(ValueError):
      adp.fit_predict(X)

  # -- 9. Full attributes check --
  def test_attributes_after_fit(self):
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaks(density_method='PAk')
    adp.fit_predict(X)
    N = X.shape[0]
    nc = adp.n_clusters_

    # labels
    self.assertEqual(adp.labels_.shape, (N,))
    self.assertEqual(adp.labels_.dtype, np.int64)

    # data
    self.assertTrue(np.array_equal(adp.data_, X))

    # cluster centers (indices)
    self.assertEqual(adp.cluster_centers_.shape, (nc,))

    # log density
    self.assertEqual(adp.log_den_.shape, (N,))
    self.assertEqual(adp.log_den_err_.shape, (N,))

    # kstar
    self.assertEqual(adp.kstar_.shape, (N,))

    # intrinsic dimension
    self.assertIsInstance(adp.intrinsic_dim_, float)
    self.assertGreater(adp.intrinsic_dim_, 0.0)

    # border matrices
    self.assertEqual(adp.log_den_bord_.shape, (nc, nc))
    self.assertEqual(adp.log_den_bord_err_.shape, (nc, nc))

    # cluster indices
    self.assertEqual(len(adp.cluster_indices_), nc)

    # distances
    self.assertEqual(adp.distances_.shape[0], N)
    self.assertEqual(adp.dist_indices_.shape[0], N)

  # -- 10. Zoo registry --
  def test_zoo_registry_contains_adp(self):
    zoo = OGSCL.OGSClusteringZoo()
    self.assertIn("AdvancedDensityPeaks", zoo.list)

  # -- 11. Zoo create and fit --
  def test_zoo_create_and_fit(self):
    X, _ = _make_3blob_data()
    metadata = {"algorithms": ["AdvancedDensityPeaks"], "Z": 1.65}
    zoo = OGSCL.OGSClusteringZoo(metadata=metadata)
    clusterer = zoo.create("AdvancedDensityPeaks")
    self.assertIsInstance(clusterer, OGSCL.OGSAdvancedDensityPeaks)
    labels = clusterer.fit_predict(X)
    self.assertEqual(len(labels), X.shape[0])
    self.assertGreaterEqual(clusterer.n_clusters_, 1)

  # -- 12. plot() returns Axes --
  def test_plot_returns_axes(self):
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaks(density_method='PAk')
    adp.fit_predict(X)
    ax = adp.plot(xlabel="X", ylabel="Y")
    self.assertIsInstance(ax, Axes)
    plt.close(ax.figure)

  # -- 13. plot_density() returns Axes --
  def test_plot_density_returns_axes(self):
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaks(density_method='PAk')
    adp.fit_predict(X)
    ax = adp.plot_density(xlabel="X", ylabel="Y")
    self.assertIsInstance(ax, Axes)
    plt.close(ax.figure)

  # -- 14. plot_cluster_borders() returns Axes --
  def test_plot_cluster_borders_returns_axes(self):
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaks(density_method='PAk')
    adp.fit_predict(X)
    ax = adp.plot_cluster_borders()
    self.assertIsInstance(ax, Axes)
    plt.close(ax.figure)

  # -- 15. get_cluster_centers() --
  def test_get_cluster_centers(self):
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaks(density_method='PAk')
    adp.fit_predict(X)
    centers = adp.get_cluster_centers()
    self.assertIsNotNone(centers)
    self.assertEqual(centers.shape[1], 2)
    self.assertEqual(centers.shape[0], adp.n_clusters_)

  # -- 16. n_clusters() method --
  def test_n_clusters_method(self):
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaks(density_method='PAk')
    adp.fit_predict(X)
    self.assertEqual(adp.n_clusters(), adp.n_clusters_)


# =========================================================================
# ADP++ (Optimized Advanced Density Peaks) Tests
# =========================================================================

class TestOGSAdvancedDensityPeaksPP(unittest.TestCase):
  """Tests for the vectorized ADP++ implementation."""

  # -- 1. Import and instantiation --
  def test_instantiation_defaults(self):
    adp = OGSCL.OGSAdvancedDensityPeaksPP()
    self.assertIsNotNone(adp)

  def test_valid_density_methods(self):
    for method in ('PAk', 'kNN', 'kstarNN', 'kpeaks'):
      adp = OGSCL.OGSAdvancedDensityPeaksPP(density_method=method)
      self.assertIsNotNone(adp)

  def test_invalid_density_method_raises(self):
    with self.assertRaises(ValueError):
      OGSCL.OGSAdvancedDensityPeaksPP(density_method='健')

  # -- 2. PAk density on 3-blob data --
  def test_pak_3blobs(self):
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaksPP(Z=1.65, density_method='PAk')
    labels = adp.fit_predict(X)
    self.assertEqual(len(labels), X.shape[0])
    self.assertGreaterEqual(adp.n_clusters_, 1)

  # -- 3. kNN density method --
  def test_knn_3blobs(self):
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaksPP(density_method='kNN', k=10)
    labels = adp.fit_predict(X)
    self.assertEqual(len(labels), X.shape[0])
    self.assertGreaterEqual(adp.n_clusters_, 1)

  # -- 4. kstarNN density method --
  def test_kstarnn_3blobs(self):
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaksPP(density_method='kstarNN')
    labels = adp.fit_predict(X)
    self.assertEqual(len(labels), X.shape[0])
    self.assertGreaterEqual(adp.n_clusters_, 1)

  # -- 5. kpeaks density method --
  def test_kpeaks_3blobs(self):
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaksPP(density_method='kpeaks')
    labels = adp.fit_predict(X)
    self.assertEqual(len(labels), X.shape[0])
    self.assertGreaterEqual(adp.n_clusters_, 1)

  # -- 6. halo=True --
  def test_halo_mode(self):
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaksPP(halo=True, density_method='PAk')
    labels = adp.fit_predict(X)
    self.assertEqual(len(labels), X.shape[0])
    self.assertGreaterEqual(adp.n_clusters_, 1)
    self.assertTrue(np.all(labels >= -1))

  # -- 7. Too few points --
  def test_too_few_points(self):
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    adp = OGSCL.OGSAdvancedDensityPeaksPP()
    with self.assertWarns(RuntimeWarning):
      labels = adp.fit_predict(X)
    self.assertEqual(len(labels), 2)

  # -- 8. NaN input --
  def test_nan_input_raises(self):
    X = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
    adp = OGSCL.OGSAdvancedDensityPeaksPP()
    with self.assertRaises(ValueError):
      adp.fit_predict(X)

  # -- 9. Full attributes check --
  def test_attributes_after_fit(self):
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaksPP(density_method='PAk')
    adp.fit_predict(X)
    N = X.shape[0]
    nc = adp.n_clusters_

    self.assertEqual(adp.labels_.shape, (N,))
    self.assertEqual(adp.labels_.dtype, np.int64)
    self.assertTrue(np.array_equal(adp.data_, X))
    self.assertEqual(adp.cluster_centers_.shape, (nc,))
    self.assertEqual(adp.log_den_.shape, (N,))
    self.assertEqual(adp.log_den_err_.shape, (N,))
    self.assertEqual(adp.kstar_.shape, (N,))
    self.assertIsInstance(adp.intrinsic_dim_, float)
    self.assertGreater(adp.intrinsic_dim_, 0.0)
    self.assertEqual(adp.log_den_bord_.shape, (nc, nc))
    self.assertEqual(adp.log_den_bord_err_.shape, (nc, nc))
    self.assertEqual(len(adp.cluster_indices_), nc)
    self.assertEqual(adp.distances_.shape[0], N)
    self.assertEqual(adp.dist_indices_.shape[0], N)

  # -- 10. Zoo registry --
  def test_zoo_registry_contains_adppp(self):
    zoo = OGSCL.OGSClusteringZoo()
    self.assertIn("AdvancedDensityPeaksPP", zoo.list)

  # -- 11. Zoo create and fit --
  def test_zoo_create_and_fit(self):
    X, _ = _make_3blob_data()
    metadata = {"algorithms": ["AdvancedDensityPeaksPP"], "Z": 1.65}
    zoo = OGSCL.OGSClusteringZoo(metadata=metadata)
    clusterer = zoo.create("AdvancedDensityPeaksPP")
    self.assertIsInstance(clusterer, OGSCL.OGSAdvancedDensityPeaksPP)
    labels = clusterer.fit_predict(X)
    self.assertEqual(len(labels), X.shape[0])
    self.assertGreaterEqual(clusterer.n_clusters_, 1)

  # -- 12. plot() returns Axes --
  def test_plot_returns_axes(self):
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaksPP(density_method='PAk')
    adp.fit_predict(X)
    ax = adp.plot(xlabel="X", ylabel="Y")
    self.assertIsInstance(ax, Axes)
    plt.close(ax.figure)

  # -- 13. plot_density() returns Axes --
  def test_plot_density_returns_axes(self):
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaksPP(density_method='PAk')
    adp.fit_predict(X)
    ax = adp.plot_density(xlabel="X", ylabel="Y")
    self.assertIsInstance(ax, Axes)
    plt.close(ax.figure)

  # -- 14. plot_cluster_borders() returns Axes --
  def test_plot_cluster_borders_returns_axes(self):
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaksPP(density_method='PAk')
    adp.fit_predict(X)
    ax = adp.plot_cluster_borders()
    self.assertIsInstance(ax, Axes)
    plt.close(ax.figure)

  # -- 15. get_cluster_centers() --
  def test_get_cluster_centers(self):
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaksPP(density_method='PAk')
    adp.fit_predict(X)
    centers = adp.get_cluster_centers()
    self.assertIsNotNone(centers)
    self.assertEqual(centers.shape[1], 2)
    self.assertEqual(centers.shape[0], adp.n_clusters_)

  # -- 16. n_clusters() method --
  def test_n_clusters_method(self):
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaksPP(density_method='PAk')
    adp.fit_predict(X)
    self.assertEqual(adp.n_clusters(), adp.n_clusters_)

  # -- 17. Equivalence [HIGH]: ADP and ADP++ produce same results --
  def test_equivalence_pak(self):
    """ADP and ADP++ must produce identical labels with PAk density."""
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaks(Z=1.65, density_method='PAk')
    adppp = OGSCL.OGSAdvancedDensityPeaksPP(Z=1.65, density_method='PAk')
    labels_adp = adp.fit_predict(X)
    labels_adppp = adppp.fit_predict(X)
    self.assertTrue(
        np.array_equal(labels_adp, labels_adppp),
        f"ADP and ADP++ labels differ (PAk). "
        f"ADP clusters: {adp.n_clusters_}, ADP++ clusters: {adppp.n_clusters_}"
    )

  def test_equivalence_knn(self):
    """ADP and ADP++ must produce identical labels with kNN density."""
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaks(Z=1.65, density_method='kNN', k=10)
    adppp = OGSCL.OGSAdvancedDensityPeaksPP(Z=1.65, density_method='kNN', k=10)
    labels_adp = adp.fit_predict(X)
    labels_adppp = adppp.fit_predict(X)
    self.assertTrue(
        np.array_equal(labels_adp, labels_adppp),
        f"ADP and ADP++ labels differ (kNN). "
        f"ADP clusters: {adp.n_clusters_}, ADP++ clusters: {adppp.n_clusters_}"
    )

  def test_equivalence_kstarnn(self):
    """ADP and ADP++ must produce identical labels with kstarNN density."""
    X, _ = _make_3blob_data()
    adp = OGSCL.OGSAdvancedDensityPeaks(Z=1.65, density_method='kstarNN')
    adppp = OGSCL.OGSAdvancedDensityPeaksPP(Z=1.65, density_method='kstarNN')
    labels_adp = adp.fit_predict(X)
    labels_adppp = adppp.fit_predict(X)
    self.assertTrue(
        np.array_equal(labels_adp, labels_adppp),
        f"ADP and ADP++ labels differ (kstarNN). "
        f"ADP clusters: {adp.n_clusters_}, ADP++ clusters: {adppp.n_clusters_}"
    )


# =========================================================================
# ManifoldBenchmark Tests
# =========================================================================

class TestManifoldBenchmark(unittest.TestCase):
  """Tests for the ManifoldBenchmark class."""

  def test_instantiation(self):
    """ManifoldBenchmark can be instantiated with default parameters."""
    mb = OGSCL.ManifoldBenchmark()
    self.assertIsNotNone(mb)
    self.assertEqual(mb.N, 10_000)
    self.assertEqual(mb.seed, 42)

  def test_instantiation_custom(self):
    """ManifoldBenchmark accepts custom N and seed."""
    mb = OGSCL.ManifoldBenchmark(N=500, seed=99)
    self.assertEqual(mb.N, 500)
    self.assertEqual(mb.seed, 99)

  def test_registry_has_21_entries(self):
    """REGISTRY must contain exactly 21 manifold definitions."""
    self.assertEqual(len(OGSCL.ManifoldBenchmark.REGISTRY), 21)

  def test_registry_entry_structure(self):
    """Each REGISTRY entry should be (name, gen_func, d, D, description)."""
    for entry in OGSCL.ManifoldBenchmark.REGISTRY:
      self.assertEqual(len(entry), 5)
      name, gen_func, d, D, desc = entry
      self.assertIsInstance(name, str)
      self.assertTrue(callable(gen_func))
      self.assertIsInstance(d, int)
      self.assertIsInstance(D, int)
      self.assertIsInstance(desc, str)
      self.assertGreater(d, 0)
      self.assertGreaterEqual(D, d)

  def test_each_generator_runs(self):
    """Each generator in REGISTRY should run with N=100 and return (X, d)."""
    for name, gen_func, d_true, D, desc in OGSCL.ManifoldBenchmark.REGISTRY:
      X, d = gen_func(N=100, seed=42)
      self.assertEqual(
          X.shape[0], 100, f"{name}: expected 100 rows, got {X.shape[0]}"
      )
      self.assertEqual(
          X.shape[1], D, f"{name}: expected D={D} columns, got {X.shape[1]}"
      )
      self.assertEqual(
          d, d_true, f"{name}: expected d={d_true}, got {d}"
      )
      self.assertTrue(
          np.all(np.isfinite(X)), f"{name}: X contains non-finite values"
      )

  def test_make_two_cluster(self):
    """make_two_cluster should produce correct shapes and labels."""
    X, labels, d = OGSCL.ManifoldBenchmark.make_two_cluster(
        OGSCL.ManifoldBenchmark.gen_M5, N_per=50
    )
    self.assertEqual(X.shape[0], 100)  # 2 * 50
    self.assertEqual(labels.shape[0], 100)
    self.assertEqual(d, 2)  # M5 has d=2
    self.assertTrue(np.array_equal(np.unique(labels), [0, 1]))

  def test_project_2d_low_dim(self):
    """project_2d on D<=3 data should return first 2 columns."""
    X, d = OGSCL.ManifoldBenchmark.gen_M5(N=100, seed=42)
    X2, xl, yl = OGSCL.ManifoldBenchmark.project_2d(X, 3)
    self.assertEqual(X2.shape, (100, 2))
    self.assertTrue(np.array_equal(X2, X[:, :2]))

  def test_project_2d_high_dim(self):
    """project_2d on D>3 data should use PCA."""
    X, d = OGSCL.ManifoldBenchmark.gen_M2(N=100, seed=42)
    X2, xl, yl = OGSCL.ManifoldBenchmark.project_2d(X, 5)
    self.assertEqual(X2.shape, (100, 2))
    self.assertIn('PC1', xl)
    self.assertIn('PC2', yl)

  def test_compute_runs(self):
    """compute() should run on a tiny N=100 dataset without error."""
    mb = OGSCL.ManifoldBenchmark(N=100, seed=42)
    results = mb.compute(verbose=False)
    self.assertEqual(len(results), 21)
    for r in results:
      self.assertIn('name', r)
      self.assertIn('pak_score', r)
      self.assertIn('n_clust', r)

  def test_plot_geometry_returns_figure(self):
    """plot_geometry() should return a Figure object."""
    mb = OGSCL.ManifoldBenchmark(N=100, seed=42)
    mb.compute(verbose=False)
    fig = mb.plot_geometry()
    self.assertIsInstance(fig, Figure)
    plt.close(fig)

  def test_plot_density_returns_figure(self):
    """plot_density() should return a Figure object."""
    mb = OGSCL.ManifoldBenchmark(N=100, seed=42)
    mb.compute(verbose=False)
    fig = mb.plot_density()
    self.assertIsInstance(fig, Figure)
    plt.close(fig)

  def test_plot_clusters_returns_figure(self):
    """plot_clusters() should return a Figure object."""
    mb = OGSCL.ManifoldBenchmark(N=100, seed=42)
    mb.compute(verbose=False)
    fig = mb.plot_clusters()
    self.assertIsInstance(fig, Figure)
    plt.close(fig)


# =========================================================================
# PAk Score Tests
# =========================================================================


class TestPAkScoreEdgeCases(unittest.TestCase):
  """Edge-case tests for PAkDensitySeparationScore."""

  def test_single_cluster_returns_none(self):
    """A single cluster (no pairs) must return None."""
    rng = np.random.RandomState(0)
    X = rng.randn(200, 3)
    labels = np.zeros(200, dtype=int)
    scorer = OGSCL.PAkDensitySeparationScore(X, labels)
    self.assertIsNone(scorer.compute())

  def test_all_noise_returns_none(self):
    """All labels = -1 -> fewer than 3 valid points -> None."""
    rng = np.random.RandomState(0)
    X = rng.randn(200, 3)
    labels = np.full(200, -1, dtype=int)
    scorer = OGSCL.PAkDensitySeparationScore(X, labels)
    self.assertIsNone(scorer.compute())

  def test_too_few_points_returns_none(self):
    """Fewer than 3 valid points -> None."""
    X = np.array([[0, 0], [1, 1]])
    labels = np.array([0, 1])
    scorer = OGSCL.PAkDensitySeparationScore(X, labels)
    self.assertIsNone(scorer.compute())

  def test_noise_labels_excluded(self):
    """Points with label=-1 must be excluded but not break the scorer."""
    rng = np.random.RandomState(42)
    X0 = rng.randn(100, 2) + np.array([0, 0])
    X1 = rng.randn(100, 2) + np.array([10, 10])
    X = np.vstack([X0, X1])
    labels = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
    # Mark some as noise
    labels[:10] = -1
    labels[100:110] = -1
    scorer = OGSCL.PAkDensitySeparationScore(X, labels)
    score = scorer.compute()
    self.assertIsNotNone(score)
    self.assertGreater(score, 0)

  def test_returns_float(self):
    """Score must be a float, not None, for valid 2-cluster data."""
    rng = np.random.RandomState(42)
    X = np.vstack([rng.randn(100, 3), rng.randn(100, 3) + 10])
    labels = np.array([0] * 100 + [1] * 100)
    scorer = OGSCL.PAkDensitySeparationScore(X, labels)
    score = scorer.compute()
    self.assertIsInstance(score, float)
    self.assertTrue(np.isfinite(score))

  def test_diagnostics_populated(self):
    """After compute(), all diagnostic attributes must be populated."""
    rng = np.random.RandomState(42)
    X = np.vstack([rng.randn(100, 3), rng.randn(100, 3) + 10])
    labels = np.array([0] * 100 + [1] * 100)
    scorer = OGSCL.PAkDensitySeparationScore(X, labels)
    scorer.compute()
    self.assertIsNotNone(scorer.log_den_)
    self.assertIsNotNone(scorer.log_den_err_)
    self.assertIsNotNone(scorer.kstar_)
    self.assertIsNotNone(scorer.intrinsic_dim_)
    self.assertIsNotNone(scorer.cluster_peak_densities_)
    self.assertIsNotNone(scorer.saddle_densities_)
    self.assertIsNotNone(scorer.pair_zscores_)
    self.assertEqual(len(scorer.log_den_), 200)
    self.assertEqual(len(scorer.log_den_err_), 200)
    self.assertEqual(len(scorer.kstar_), 200)


class TestPAkScoreOrdering(unittest.TestCase):
  """
  Core property: well-separated clusters must score higher than random labels,
  which should score higher than heavily overlapping clusters.
  """

  @classmethod
  def setUpClass(cls):
    """Generate shared datasets once."""
    rng = np.random.RandomState(42)
    # Well-separated 3D Gaussians
    cls.X_sep = np.vstack([
        rng.randn(300, 3) * 0.5 + np.array([0, 0, 0]),
        rng.randn(300, 3) * 0.5 + np.array([8, 8, 8]),
        rng.randn(300, 3) * 0.5 + np.array([-8, 8, 0]),
    ])
    cls.labels_sep = np.array([0] * 300 + [1] * 300 + [2] * 300)

    # Same data, random labels
    cls.labels_random = rng.randint(0, 3, size=900)

    # Overlapping 3D Gaussians (same centers, large sigma)
    cls.X_overlap = np.vstack([
        rng.randn(300, 3) * 5.0 + np.array([0, 0, 0]),
        rng.randn(300, 3) * 5.0 + np.array([1, 1, 1]),
        rng.randn(300, 3) * 5.0 + np.array([2, 2, 2]),
    ])
    cls.labels_overlap = np.array([0] * 300 + [1] * 300 + [2] * 300)

  def test_separated_higher_than_random(self):
    s_sep = OGSCL.PAkDensitySeparationScore(self.X_sep, self.labels_sep)
    s_rnd = OGSCL.PAkDensitySeparationScore(self.X_sep, self.labels_random)
    score_sep = s_sep.compute()
    score_rnd = s_rnd.compute()
    self.assertIsNotNone(score_sep)
    self.assertIsNotNone(score_rnd)
    self.assertGreater(
        score_sep, score_rnd,
        f"Separated ({score_sep:.2f}) should beat random ({score_rnd:.2f})"
    )

  def test_separated_higher_than_overlap(self):
    s_sep = OGSCL.PAkDensitySeparationScore(self.X_sep, self.labels_sep)
    s_ovl = OGSCL.PAkDensitySeparationScore(
        self.X_overlap, self.labels_overlap
    )
    score_sep = s_sep.compute()
    score_ovl = s_ovl.compute()
    self.assertIsNotNone(score_sep)
    self.assertIsNotNone(score_ovl)
    self.assertGreater(
        score_sep, score_ovl,
        f"Separated ({score_sep:.2f}) should beat overlap ({score_ovl:.2f})"
    )

  def test_well_separated_above_threshold(self):
    """Well-separated Gaussians should produce Z > 1.65 (90% confidence)."""
    s = OGSCL.PAkDensitySeparationScore(self.X_sep, self.labels_sep)
    score = s.compute()
    self.assertIsNotNone(score)
    self.assertGreater(
        score, 1.65, f"Well-separated score={score:.2f} should exceed 1.65"
    )


class TestPAkScoreIntrinsicDimension(unittest.TestCase):
  """
  Verify that the 2NN intrinsic dimension estimator embedded in the scorer
  produces reasonable estimates for each manifold.
  """

  def _check_intrinsic_dim(self, name, gen_func, d_true, tol_factor=0.5):
    """
    Run the scorer on a single-manifold 2-cluster dataset and check
    that the estimated intrinsic dimension is within tolerance.

    We allow |d̂ - d_true| ≤ tol_factor * d_true + 2 to account for
    finite-sample and boundary effects, especially at high d.
    """
    X, labels, _ = make_well_separated(
        gen_func, N_per_cluster=500, n_clusters=2, seed=42
    )
    scorer = OGSCL.PAkDensitySeparationScore(X, labels, maxk=min(80, 499))
    scorer.compute()

    d_est = scorer.intrinsic_dim_
    self.assertIsNotNone(d_est, f"{name}: intrinsic_dim_ is None")

    tol = tol_factor * d_true + 2.0
    self.assertAlmostEqual(
        d_est, d_true, delta=tol,
        msg=f"{name}: d_hat={d_est:.1f} vs d_true={d_true} (tol={tol:.1f})"
    )

  def test_M1_hypersphere(self):
    self._check_intrinsic_dim('M1', OGSCL.ManifoldBenchmark.gen_M1, 10)

  def test_M2_affine(self):
    self._check_intrinsic_dim('M2', OGSCL.ManifoldBenchmark.gen_M2, 3)

  def test_M5_helix_2d(self):
    self._check_intrinsic_dim('M5', OGSCL.ManifoldBenchmark.gen_M5, 2)

  def test_M7_swiss_roll(self):
    self._check_intrinsic_dim('M7', OGSCL.ManifoldBenchmark.gen_M7, 2)

  def test_M10a_hypercube(self):
    self._check_intrinsic_dim('M10a', OGSCL.ManifoldBenchmark.gen_M10a, 10)

  def test_M11_moebius(self):
    self._check_intrinsic_dim('M11', OGSCL.ManifoldBenchmark.gen_M11, 2)

  def test_M13_helix_1d(self):
    self._check_intrinsic_dim('M13', OGSCL.ManifoldBenchmark.gen_M13, 1)

  def test_MP3_nonlinear(self):
    self._check_intrinsic_dim('MP3', OGSCL.ManifoldBenchmark.gen_MP3, 3)

  def test_MP6_nonlinear(self):
    self._check_intrinsic_dim('MP6', OGSCL.ManifoldBenchmark.gen_MP6, 6)


class TestPAkScoreManifoldClusters(unittest.TestCase):
  """
  For each manifold, verify that:
    - Well-separated clusters -> Z > 0 (finite, positive)
    - Well-separated > random labels on the same data
  """

  def _run_manifold_test(self, name, gen_func):
    """Test well-separated vs random-label Z-scores for a manifold."""
    N_per = 400
    X, labels_true, d_true = make_well_separated(
        gen_func, N_per_cluster=N_per, n_clusters=2, separation=20.0, seed=42
    )
    rng = np.random.RandomState(99)
    labels_random = rng.randint(0, 2, size=len(labels_true))

    maxk = min(80, N_per - 1)

    scorer_true = OGSCL.PAkDensitySeparationScore(X, labels_true, maxk=maxk)
    score_true = scorer_true.compute()

    scorer_rand = OGSCL.PAkDensitySeparationScore(X, labels_random, maxk=maxk)
    score_rand = scorer_rand.compute()

    # True labels should give a positive score
    self.assertIsNotNone(score_true, f"{name}: score with true labels is None")
    self.assertGreater(
        score_true, 0, f"{name}: score_true={score_true:.2f} should be > 0"
    )
    self.assertTrue(
        np.isfinite(score_true), f"{name}: score_true is not finite"
    )

    # True labels should beat random labels
    if score_rand is not None:
      self.assertGreater(
          score_true, score_rand,
          f"{name}: true ({score_true:.2f}) should beat random "
          f"({score_rand:.2f})"
      )

  # --- Low-dimensional manifolds (fast) ---
  def test_M2_affine_3d(self):
    self._run_manifold_test('M2', OGSCL.ManifoldBenchmark.gen_M2)

  def test_M5_helix_2d(self):
    self._run_manifold_test('M5', OGSCL.ManifoldBenchmark.gen_M5)

  def test_M7_swiss_roll(self):
    self._run_manifold_test('M7', OGSCL.ManifoldBenchmark.gen_M7)

  def test_M11_moebius(self):
    self._run_manifold_test('M11', OGSCL.ManifoldBenchmark.gen_M11)

  def test_M13_helix_1d(self):
    self._run_manifold_test('M13', OGSCL.ManifoldBenchmark.gen_M13)

  # --- Medium-dimensional manifolds ---
  def test_M1_hypersphere(self):
    self._run_manifold_test('M1', OGSCL.ManifoldBenchmark.gen_M1)

  def test_M3_concentrated(self):
    self._run_manifold_test('M3', OGSCL.ManifoldBenchmark.gen_M3)

  def test_M4_nonlinear_4d(self):
    self._run_manifold_test('M4', OGSCL.ManifoldBenchmark.gen_M4)

  def test_M10a_hypercube_10d(self):
    self._run_manifold_test('M10a', OGSCL.ManifoldBenchmark.gen_M10a)

  def test_MP3_nonlinear_3d(self):
    self._run_manifold_test('MP3', OGSCL.ManifoldBenchmark.gen_MP3)

  def test_MP6_nonlinear_6d(self):
    self._run_manifold_test('MP6', OGSCL.ManifoldBenchmark.gen_MP6)

  def test_MP9_nonlinear_9d(self):
    self._run_manifold_test('MP9', OGSCL.ManifoldBenchmark.gen_MP9)

  # --- High-dimensional manifolds ---
  def test_M6_nonlinear_36d(self):
    self._run_manifold_test('M6', OGSCL.ManifoldBenchmark.gen_M6)

  def test_M9_affine_20d(self):
    self._run_manifold_test('M9', OGSCL.ManifoldBenchmark.gen_M9)

  def test_M10b_hypercube_17d(self):
    self._run_manifold_test('M10b', OGSCL.ManifoldBenchmark.gen_M10b)

  def test_M10c_hypercube_24d(self):
    self._run_manifold_test('M10c', OGSCL.ManifoldBenchmark.gen_M10c)

  def test_M12_gaussian_20d(self):
    self._run_manifold_test('M12', OGSCL.ManifoldBenchmark.gen_M12)

  def test_Mbeta_nonlinear_40d(self):
    self._run_manifold_test('Mbeta', OGSCL.ManifoldBenchmark.gen_Mbeta)

  def test_MN1_nonlinear_72d(self):
    self._run_manifold_test('MN1', OGSCL.ManifoldBenchmark.gen_MN1)

  def test_MN2_nonlinear_96d(self):
    self._run_manifold_test('MN2', OGSCL.ManifoldBenchmark.gen_MN2)

  def test_M10d_hypercube_70d(self):
    self._run_manifold_test('M10d', OGSCL.ManifoldBenchmark.gen_M10d)


class TestPAkScoreNonConvex(unittest.TestCase):
  """
  Test that PAk correctly scores non-convex cluster shapes where geometric
  metrics (Silhouette, CH) are known to fail.

  Two interleaved half-moons: geometrically overlapping in the bounding box
  but separated by a density valley. PAk should give a high score (clusters
  follow density); Silhouette gives a low/moderate score because the
  Euclidean centroid model is inappropriate.
  """

  @classmethod
  def setUpClass(cls):
    """Generate interleaved half-moons."""
    rng = np.random.RandomState(42)
    N = 500

    # Moon 1: upper arc
    theta1 = np.linspace(0, np.pi, N)
    X1 = np.column_stack([np.cos(theta1), np.sin(theta1)])
    X1 += rng.randn(N, 2) * 0.05

    # Moon 2: lower arc, shifted
    theta2 = np.linspace(0, np.pi, N)
    X2 = np.column_stack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5])
    X2 += rng.randn(N, 2) * 0.05

    cls.X_moons = np.vstack([X1, X2])
    cls.labels_moons = np.array([0] * N + [1] * N)

  def test_pak_positive_on_moons(self):
    """PAk score > 0 for well-defined density-separated moons."""
    scorer = OGSCL.PAkDensitySeparationScore(self.X_moons, self.labels_moons)
    score = scorer.compute()
    self.assertIsNotNone(score)
    self.assertGreater(
        score, 0, f"PAk score on moons = {score:.2f}, expected > 0"
    )

  def test_pak_strongly_positive_on_moons(self):
    """
    On half-moons, PAk should give a high Z-score (> 1.0) because the two arcs
    are density-separated even though they are geometrically interleaved.
    """
    pak_true = OGSCL.PAkDensitySeparationScore(
        self.X_moons, self.labels_moons
    ).compute()
    self.assertIsNotNone(pak_true)
    self.assertGreater(
        pak_true, 1.0, f"PAk on moons = {pak_true:.2f}, expected > 1.0"
    )


class TestPAkScoreMultiCluster(unittest.TestCase):
  """Test with >2 clusters to verify correct pair-wise Z-score aggregation."""

  def test_five_gaussians(self):
    """Five well-separated Gaussian clusters -> high Z."""
    rng = np.random.RandomState(42)
    n_clusters = 5
    N_per = 200
    centers = np.array([
        [0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0], [5, 5, 10]
    ], dtype=float)
    parts = []
    labs = []
    for c in range(n_clusters):
      parts.append(rng.randn(N_per, 3) * 0.5 + centers[c])
      labs.append(np.full(N_per, c, dtype=int))
    X = np.vstack(parts)
    labels = np.concatenate(labs)

    scorer = OGSCL.PAkDensitySeparationScore(X, labels)
    score = scorer.compute()

    self.assertIsNotNone(score)
    self.assertGreater(
        score, 1.0, f"5-Gaussian score={score:.2f} should be > 1.0"
    )

    # Check correct number of pair Z-scores
    n_pairs = len(scorer.pair_zscores_)
    self.assertGreater(n_pairs, 0, "Should have at least 1 adjacent pair")
    # At most C(5,2)=10 pairs
    self.assertLessEqual(n_pairs, 10)

  def test_z_scores_all_finite(self):
    """All individual pair Z-scores must be finite."""
    rng = np.random.RandomState(42)
    X = np.vstack([
        rng.randn(150, 3) + np.array([0, 0, 0]),
        rng.randn(150, 3) + np.array([8, 0, 0]),
        rng.randn(150, 3) + np.array([0, 8, 0]),
    ])
    labels = np.array([0] * 150 + [1] * 150 + [2] * 150)
    scorer = OGSCL.PAkDensitySeparationScore(X, labels)
    scorer.compute()
    if scorer.pair_zscores_:
      for pair, z in scorer.pair_zscores_.items():
        self.assertTrue(
            np.isfinite(z), f"Z-score for pair {pair} = {z} is not finite"
        )


class TestPAkScoreSwissRollNonConvex(unittest.TestCase):
  """
  Swiss-Roll is the canonical test for non-convex cluster evaluation.
  Two pieces of the roll (inner and outer spiral) are geometrically
  interleaved but density-separated.
  """

  def test_swiss_roll_inner_outer(self):
    """Two non-convex spiral pieces should be density-separated."""
    rng = np.random.RandomState(42)
    N = 800

    t1 = 1.5 * np.pi + 1.0 * np.pi * rng.uniform(0, 1, N)
    h1 = rng.uniform(0, 10, N)
    X1 = np.column_stack([t1 * np.cos(t1), h1, t1 * np.sin(t1)])

    t2 = 3.5 * np.pi + 1.0 * np.pi * rng.uniform(0, 1, N)
    h2 = rng.uniform(0, 10, N)
    X2 = np.column_stack([t2 * np.cos(t2), h2, t2 * np.sin(t2)])

    X = np.vstack([X1, X2])
    labels = np.array([0] * N + [1] * N)

    scorer = OGSCL.PAkDensitySeparationScore(X, labels)
    score = scorer.compute()

    self.assertIsNotNone(score)
    self.assertGreater(
        score, 0, f"Swiss-Roll spiral score={score:.2f} should be > 0"
    )


class TestPAkScoreWeightedAggregation(unittest.TestCase):
  """Verify that the population-weighting formula works correctly."""

  def test_weight_formula(self):
    """
    Manually compute weighted mean and compare with scorer output.
    """
    rng = np.random.RandomState(42)
    # Unequal cluster sizes: 100 vs 400
    X = np.vstack([
        rng.randn(100, 2) + np.array([0, 0]),
        rng.randn(400, 2) + np.array([10, 0]),
    ])
    labels = np.array([0] * 100 + [1] * 400)

    scorer = OGSCL.PAkDensitySeparationScore(X, labels)
    score = scorer.compute()

    if score is not None and scorer.pair_zscores_:
      # Manual recomputation
      total_w = 0.0
      wz_sum = 0.0
      for (ci, cj), z in scorer.pair_zscores_.items():
        ni = float(np.sum(labels == ci))
        nj = float(np.sum(labels == cj))
        w = np.sqrt(ni * nj)
        wz_sum += w * z
        total_w += w
      expected = wz_sum / total_w
      self.assertAlmostEqual(
          score, expected, places=10, msg="Weighted mean mismatch"
      )


class TestPAkScoreMaxk(unittest.TestCase):
  """Test custom maxk parameter."""

  def test_small_maxk(self):
    """Score must still work with a small maxk."""
    rng = np.random.RandomState(42)
    X = np.vstack([rng.randn(100, 2), rng.randn(100, 2) + 8])
    labels = np.array([0] * 100 + [1] * 100)
    scorer = OGSCL.PAkDensitySeparationScore(X, labels, maxk=10)
    score = scorer.compute()
    self.assertIsNotNone(score)
    self.assertGreater(score, 0)

  def test_large_maxk(self):
    """Score must still work with maxk close to N."""
    rng = np.random.RandomState(42)
    X = np.vstack([rng.randn(50, 2), rng.randn(50, 2) + 8])
    labels = np.array([0] * 50 + [1] * 50)
    scorer = OGSCL.PAkDensitySeparationScore(X, labels, maxk=90)
    score = scorer.compute()
    self.assertIsNotNone(score)
    self.assertGreaterEqual(score, 0)


class TestPAkScoreDthr(unittest.TestCase):
  """Test that the Dthr parameter modulates k* (and thus the score)."""

  def test_dthr_affects_kstar(self):
    """Lower Dthr -> smaller k*; higher -> larger k*."""
    rng = np.random.RandomState(42)
    X = np.vstack([rng.randn(200, 3), rng.randn(200, 3) + 10])
    labels = np.array([0] * 200 + [1] * 200)

    # Conservative Dthr
    s_hi = OGSCL.PAkDensitySeparationScore(X, labels, Dthr=23.93)
    s_hi.compute()
    kstar_hi = s_hi.kstar_

    # Relaxed Dthr
    s_lo = OGSCL.PAkDensitySeparationScore(X, labels, Dthr=6.63)
    s_lo.compute()
    kstar_lo = s_lo.kstar_

    self.assertIsNotNone(kstar_hi)
    self.assertIsNotNone(kstar_lo)

    mean_k_hi = np.mean(kstar_hi)
    mean_k_lo = np.mean(kstar_lo)

    self.assertGreater(mean_k_hi, 2)
    self.assertGreater(mean_k_lo, 2)


# =========================================================================
# DADApy Tutorial Tests
# =========================================================================


class TestTutorialFiles(unittest.TestCase):
  def test_missing_dataset_reports_manual_download_instructions(self):
    with tempfile.TemporaryDirectory() as directory, \
        patch(__name__ + ".TUTORIAL_DATA_DIR", Path(directory)):
      with self.assertRaises(FileNotFoundError) as error:
        _tutorial_file(DATA_URL_DIHEDRALS, "missing.npy")
      self.assertIn(DATA_URL_DIHEDRALS, str(error.exception))
      self.assertIn("OGS/test/data/missing.npy", str(error.exception))
      self.assertEqual(list(Path(directory).iterdir()), [])

  def test_empty_dataset_is_reported_and_preserved(self):
    with tempfile.TemporaryDirectory() as directory, \
        patch(__name__ + ".TUTORIAL_DATA_DIR", Path(directory)):
      path = Path(directory) / "empty.npy"
      path.touch()
      with self.assertRaises(ValueError) as error:
        _tutorial_file(DATA_URL_DISTANCES, path.name)
      self.assertIn(DATA_URL_DISTANCES, str(error.exception))
      self.assertIn("OGS/test/data/empty.npy", str(error.exception))
      self.assertTrue(path.is_file())

  def test_existing_dataset_is_returned_without_modification(self):
    with tempfile.TemporaryDirectory() as directory, \
        patch(__name__ + ".TUTORIAL_DATA_DIR", Path(directory)):
      path = Path(directory) / "existing.npy"
      path.write_bytes(b"fixture marker")
      self.assertEqual(_tutorial_file(DATA_URL_DIHEDRALS, path.name), path)
      self.assertEqual(path.read_bytes(), b"fixture marker")


class TestTutorialDataCache(unittest.TestCase):
  def setUp(self):
    _tutorial_data.cache_clear()
    self.addCleanup(_tutorial_data.cache_clear)

  def test_successful_load_is_shared_between_callers(self):
    dihedrals = np.zeros((2, 15))
    distances = np.zeros((2, 3))
    with patch(__name__ + "._load_dihedrals", return_value=dihedrals) as load_dih, \
        patch(__name__ + "._load_distances", return_value=distances) as load_dist:
      first = _tutorial_data()
      second = _tutorial_data()
    self.assertIs(first, second)
    self.assertIs(first[0], dihedrals)
    self.assertIs(first[1], distances)
    load_dih.assert_called_once_with()
    load_dist.assert_called_once_with()

  def test_loading_failure_propagates_without_caching_a_fallback(self):
    with patch(__name__ + "._load_dihedrals", return_value=np.zeros((2, 15))), \
        patch(__name__ + "._load_distances",
              side_effect=OSError("tutorial load failed")) as load_dist:
      for _ in range(2):
        with self.assertRaisesRegex(OSError, "tutorial load failed"):
          _tutorial_data()
    self.assertEqual(load_dist.call_count, 2)
    self.assertEqual(_tutorial_data.cache_info().currsize, 0)


class TestDataLoading(unittest.TestCase):
  """Verify that the datasets have expected shapes and are finite."""

  @classmethod
  def setUpClass(cls):
    cls.dihedrals, cls.distances = _tutorial_data()

  def test_dihedrals_shape_real(self):
    """Selected dihedral angles should be (3758, 15) for real data."""
    self.assertEqual(self.dihedrals.shape, (3758, 15))

  def test_distances_shape_real(self):
    """Heavy atom distances should be (3758, 4278) for real data."""
    self.assertEqual(self.distances.shape, (3758, 4278))

  def test_dihedrals_2d(self):
    """Dihedral data should be 2-D with >= 10 features."""
    self.assertEqual(self.dihedrals.ndim, 2)
    self.assertGreaterEqual(self.dihedrals.shape[1], 10)

  def test_distances_2d(self):
    """Distance data should be 2-D."""
    self.assertEqual(self.distances.ndim, 2)

  def test_dihedrals_finite(self):
    """All dihedral values must be finite."""
    self.assertTrue(np.all(np.isfinite(self.dihedrals)))

  def test_distances_finite(self):
    """All distance values must be finite."""
    self.assertTrue(np.all(np.isfinite(self.distances)))


class TestIntrinsicDimensionDihedrals(unittest.TestCase):
  """
  Intrinsic dimension estimation on dihedral-like data.
  """

  _scorer = None
  _id = None

  @classmethod
  def setUpClass(cls):
    dihedrals, _ = _tutorial_data()
    data = dihedrals.copy() + np.pi
    cls._n_features = data.shape[1]
    N = data.shape[0]
    labels = np.zeros(N, dtype=np.int64)
    labels[N // 2:] = 1
    cls._scorer = OGSCL.PAkDensitySeparationScore(data, labels)
    cls._scorer.compute()
    cls._id = cls._scorer.intrinsic_dim_

  @classmethod
  def tearDownClass(cls):
    cls._scorer = None

  def test_intrinsic_dim_positive(self):
    """Intrinsic dimension must be positive."""
    self.assertIsNotNone(self._id)
    self.assertGreater(self._id, 0)

  def test_intrinsic_dim_reasonable(self):
    """ID should be within a sensible range for the data dimensionality."""
    self.assertGreater(self._id, 0.5)
    self.assertLess(self._id, self._n_features + 5)


class TestIntrinsicDimensionDistances(unittest.TestCase):
  """
  Intrinsic dimension estimation on distance-like data.
  """

  _id = None

  @classmethod
  def setUpClass(cls):
    _, distances = _tutorial_data()
    data = distances.copy()
    cls._n_features = data.shape[1]
    N = data.shape[0]
    labels = np.zeros(N, dtype=np.int64)
    labels[N // 2:] = 1
    scorer = OGSCL.PAkDensitySeparationScore(data, labels)
    scorer.compute()
    cls._id = scorer.intrinsic_dim_

  def test_intrinsic_dim_reasonable(self):
    """ID should be within a sensible range for the data dimensionality."""
    self.assertIsNotNone(self._id)
    self.assertGreater(self._id, 0.5)
    self.assertLess(self._id, self._n_features + 5)


class TestADPClusteringDihedrals(unittest.TestCase):
  """
  ADP clustering on dihedral angle data with Z=4.5 (as in tutorial).
  The tutorial finds 3 clusters, with the largest containing 2516/3758 points.
  """

  _adp = None
  _labels = None
  _N = None

  @classmethod
  def setUpClass(cls):
    dihedrals, _ = _tutorial_data()
    data = dihedrals.copy() + np.pi
    cls._N = data.shape[0]
    cls._adp = OGSCL.OGSAdvancedDensityPeaks(
        Z=4.5, halo=False, density_method='PAk',
    )
    cls._labels = cls._adp.fit_predict(data)

  @classmethod
  def tearDownClass(cls):
    cls._adp = None
    cls._labels = None

  def test_finds_clusters(self):
    """ADP should find at least 2 clusters."""
    self.assertIsNotNone(self._adp.n_clusters_)
    self.assertGreaterEqual(self._adp.n_clusters_, 2)

  def test_labels_cover_all_points(self):
    """With halo=False, every point should have label >= 0."""
    self.assertTrue(np.all(self._labels >= 0))

  def test_cluster_centers_exist(self):
    """Cluster centers should be populated after fit."""
    self.assertIsNotNone(self._adp.cluster_centers_)
    self.assertGreater(len(self._adp.cluster_centers_), 0)

  def test_log_density_computed(self):
    """Log-density should be computed and have correct length."""
    self.assertIsNotNone(self._adp.log_den_)
    self.assertEqual(len(self._adp.log_den_), self._N)

  def test_intrinsic_dim_computed(self):
    """Intrinsic dimension should be computed and reasonable."""
    self.assertIsNotNone(self._adp.intrinsic_dim_)
    self.assertGreater(self._adp.intrinsic_dim_, 1)
    self.assertLess(self._adp.intrinsic_dim_, 20)

  def test_kstar_computed(self):
    """Adaptive k* should be computed for each point."""
    self.assertIsNotNone(self._adp.kstar_)
    self.assertEqual(len(self._adp.kstar_), self._N)

  def test_largest_cluster_dominates(self):
    """The largest cluster should contain a substantial fraction of points."""
    _, counts = np.unique(self._labels, return_counts=True)
    self.assertGreater(np.max(counts) / self._N, 0.25)


class TestADPClusteringDistances(unittest.TestCase):
  """
  ADP clustering on heavy atom distances with Z=3.5.
  """

  _adp = None
  _labels = None
  _N = None

  @classmethod
  def setUpClass(cls):
    _, distances = _tutorial_data()
    data = distances.copy()
    cls._N = data.shape[0]
    cls._adp = OGSCL.OGSAdvancedDensityPeaks(
        Z=3.5, halo=False, density_method='PAk',
    )
    cls._labels = cls._adp.fit_predict(data)

  @classmethod
  def tearDownClass(cls):
    cls._adp = None
    cls._labels = None

  def test_finds_clusters(self):
    """ADP should find at least 2 clusters."""
    self.assertIsNotNone(self._adp.n_clusters_)
    self.assertGreaterEqual(self._adp.n_clusters_, 2)

  def test_labels_cover_all_points(self):
    """With halo=False, every point should have label >= 0."""
    self.assertTrue(np.all(self._labels >= 0))

  def test_cluster_centers_exist(self):
    """Cluster centers should be populated after fit."""
    self.assertIsNotNone(self._adp.cluster_centers_)
    self.assertGreater(len(self._adp.cluster_centers_), 0)

  def test_largest_cluster_dominates(self):
    """The largest cluster should contain a substantial fraction of points."""
    _, counts = np.unique(self._labels, return_counts=True)
    self.assertGreater(np.max(counts) / self._N, 0.25)


class TestCrossRepresentationConsistency(unittest.TestCase):
  """
  Compare cluster assignments from dihedral and distance representations.
  """

  _labels_dih = None
  _labels_dist = None
  _n_clusters_dih = None
  _n_clusters_dist = None

  @classmethod
  def setUpClass(cls):
    # Dihedrals
    dihedrals, distances = _tutorial_data()
    data_dih = dihedrals.copy() + np.pi
    adp_dih = OGSCL.OGSAdvancedDensityPeaks(
        Z=4.5, halo=False, density_method='PAk',
    )
    cls._labels_dih = adp_dih.fit_predict(data_dih)
    cls._n_clusters_dih = adp_dih.n_clusters_

    # Distances
    data_dist = distances.copy()
    adp_dist = OGSCL.OGSAdvancedDensityPeaks(
        Z=3.5, halo=False, density_method='PAk',
    )
    cls._labels_dist = adp_dist.fit_predict(data_dist)
    cls._n_clusters_dist = adp_dist.n_clusters_

  @classmethod
  def tearDownClass(cls):
    cls._labels_dih = None
    cls._labels_dist = None

  def test_both_find_similar_n_clusters(self):
    """Both representations should find between 2 and 10 clusters."""
    self.assertGreaterEqual(self._n_clusters_dih, 2)
    self.assertLessEqual(self._n_clusters_dih, 10)
    self.assertGreaterEqual(self._n_clusters_dist, 2)
    self.assertLessEqual(self._n_clusters_dist, 10)

  def test_cluster_agreement_above_chance(self):
    """
    After optimal label permutation, agreement should exceed chance level.
    Only runs exhaustive search if n_clusters is small enough (< 8).
    """
    from itertools import permutations

    labels1 = self._labels_dih
    labels2 = self._labels_dist

    n_c = max(labels1.max(), labels2.max()) + 1

    if n_c > 7:
      best_agreement = self._greedy_agreement(labels1, labels2, n_c)
    else:
      best_agreement = 0.0
      for perm in permutations(range(n_c)):
        remapped = np.array(
            [perm[l] if l < len(perm) else l for l in labels2]
        )
        agreement = np.mean(labels1 == remapped)
        best_agreement = max(best_agreement, agreement)

    chance = 1.0 / max(n_c, 1)
    self.assertGreater(
        best_agreement, chance,
        f"Best agreement {best_agreement:.2%} is below chance {chance:.2%}",
    )

  @staticmethod
  def _greedy_agreement(labels1, labels2, n_c):
    """Greedy best-match agreement for large n_c."""
    from collections import Counter
    best = 0.0
    used = set()
    for c1 in range(n_c):
      mask1 = labels1 == c1
      if not np.any(mask1):
        continue
      best_c2 = -1
      best_overlap = -1
      for c2 in range(n_c):
        if c2 in used:
          continue
        overlap = np.sum(mask1 & (labels2 == c2))
        if overlap > best_overlap:
          best_overlap = overlap
          best_c2 = c2
      if best_c2 >= 0:
        used.add(best_c2)
        best += best_overlap
    return best / len(labels1)


class TestPAkScoreOnADPClustering(unittest.TestCase):
  """
  Run PAkDensitySeparationScore on ADP clustering results from dihedrals.
  """

  _score = None
  _scorer = None

  @classmethod
  def setUpClass(cls):
    dihedrals, _ = _tutorial_data()
    data = dihedrals.copy() + np.pi
    adp = OGSCL.OGSAdvancedDensityPeaks(
        Z=4.5, halo=False, density_method='PAk',
    )
    labels = adp.fit_predict(data)

    cls._scorer = OGSCL.PAkDensitySeparationScore(data, labels)
    cls._score = cls._scorer.compute()
    cls._data = data

  @classmethod
  def tearDownClass(cls):
    cls._scorer = None

  def test_score_positive(self):
    """PAk separation score should be positive for ADP clustering."""
    self.assertIsNotNone(self._score)
    self.assertGreater(self._score, 0)

  def test_score_is_finite(self):
    """PAk separation score should be a finite number."""
    self.assertIsNotNone(self._score)
    self.assertTrue(np.isfinite(self._score))

  def test_diagnostics_populated(self):
    """All diagnostic attributes should be populated after compute()."""
    self.assertIsNotNone(self._scorer.log_den_)
    self.assertIsNotNone(self._scorer.log_den_err_)
    self.assertIsNotNone(self._scorer.kstar_)
    self.assertIsNotNone(self._scorer.intrinsic_dim_)
    self.assertIsNotNone(self._scorer.cluster_peak_densities_)
    self.assertIsNotNone(self._scorer.saddle_densities_)
    self.assertIsNotNone(self._scorer.pair_zscores_)

  def test_score_better_than_random(self):
    """True ADP labels should score higher than random labels."""
    data = self._data
    rng = np.random.RandomState(42)
    random_labels = rng.randint(0, 3, size=data.shape[0])
    random_scorer = OGSCL.PAkDensitySeparationScore(data, random_labels)
    random_score = random_scorer.compute()

    if random_score is not None:
      self.assertGreater(
          self._score, random_score,
          "ADP score should exceed random labeling score",
      )


class TestPlottingMethods(unittest.TestCase):
  """
  Test that dendrogram, network, and decision graph plots run without error.
  Uses a small subset (first 500 points) for speed.
  """

  _scorer = None

  @classmethod
  def setUpClass(cls):
    dihedrals, _ = _tutorial_data()
    data = dihedrals[:500].copy() + np.pi
    adp = OGSCL.OGSAdvancedDensityPeaks(
        Z=4.5, halo=False, density_method='PAk',
    )
    labels = adp.fit_predict(data)
    cls._scorer = OGSCL.PAkDensitySeparationScore(data, labels)
    cls._scorer.compute()

  @classmethod
  def tearDownClass(cls):
    plt.close('all')
    cls._scorer = None

  def tearDown(self):
    plt.close('all')

  # --- _plot_dendrogram ---

  def test_plot_dendrogram_runs(self):
    """_plot_dendrogram should execute and return an Axes."""
    ax = self._scorer._plot_dendrogram()
    self.assertIsInstance(ax, matplotlib.axes.Axes)

  def test_plot_dendrogram_with_ax(self):
    """_plot_dendrogram should accept a pre-created Axes."""
    fig, ax = plt.subplots()
    returned_ax = self._scorer._plot_dendrogram(ax=ax)
    self.assertIs(returned_ax, ax)

  # --- _plot_network ---

  def test_plot_network_runs(self):
    """_plot_network should execute and return an Axes."""
    ax = self._scorer._plot_network()
    self.assertIsInstance(ax, matplotlib.axes.Axes)

  def test_plot_network_with_ax(self):
    """_plot_network should accept a pre-created Axes."""
    fig, ax = plt.subplots()
    returned_ax = self._scorer._plot_network(ax=ax)
    self.assertIs(returned_ax, ax)

  # --- _plot_decisionGraph ---

  def test_plot_decisionGraph_runs(self):
    """_plot_decisionGraph should execute and return an Axes."""
    ax = self._scorer._plot_decisionGraph()
    self.assertIsInstance(ax, matplotlib.axes.Axes)

  def test_plot_decisionGraph_with_ax(self):
    """_plot_decisionGraph should accept a pre-created Axes."""
    fig, ax = plt.subplots()
    returned_ax = self._scorer._plot_decisionGraph(ax=ax)
    self.assertIs(returned_ax, ax)


class TestDensityEstimation(unittest.TestCase):
  """
  Validate PAk density estimation properties on dihedral data.
  """

  _adp = None
  _N = None

  @classmethod
  def setUpClass(cls):
    dihedrals, _ = _tutorial_data()
    data = dihedrals.copy() + np.pi
    cls._N = data.shape[0]
    cls._adp = OGSCL.OGSAdvancedDensityPeaks(
        Z=4.5, halo=False, density_method='PAk',
    )
    cls._adp.fit_predict(data)

  @classmethod
  def tearDownClass(cls):
    cls._adp = None

  def test_log_density_finite(self):
    """All log-density values should be finite."""
    self.assertTrue(np.all(np.isfinite(self._adp.log_den_)))

  def test_log_density_err_positive(self):
    """All log-density error estimates should be positive."""
    self.assertTrue(np.all(self._adp.log_den_err_ > 0))

  def test_kstar_in_range(self):
    """All k* values should be in [3, maxk]."""
    maxk = min(100, self._N - 1)
    self.assertTrue(np.all(self._adp.kstar_ >= 3))
    self.assertTrue(np.all(self._adp.kstar_ <= maxk))

  def test_cluster_peaks_are_local_maxima(self):
    """Each cluster center should have the highest log-density in its cluster."""
    labels = self._adp.labels_
    log_den = self._adp.log_den_
    centers = self._adp.cluster_centers_

    for center_idx in centers:
      cluster_label = labels[center_idx]
      cluster_mask = labels == cluster_label
      cluster_densities = log_den[cluster_mask]
      center_density = log_den[center_idx]
      self.assertAlmostEqual(
          center_density, np.max(cluster_densities), places=10,
          msg=(
              f"Center {center_idx} (cluster {cluster_label}) density "
              f"{center_density:.4f} != max {np.max(cluster_densities):.4f}"
          ),
      )


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
  unittest.main()

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.datasets import make_blobs

THIS_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(THIS_DIR + "/../src"))

matplotlib.use("Agg")

import ogsclustering as OGSCL


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

  def test_labels_to_colormap_no_noise(self):
    labels = np.array([0, 1, 1])
    encoded, unique, cmap, norm = OGSCL.labels_to_colormap(labels)
    self.assertTrue(np.array_equal(unique, np.array([0, 1])))
    self.assertTrue(np.array_equal(encoded, np.array([0, 1, 1])))
    self.assertIsNotNone(cmap)
    self.assertIsNotNone(norm)

  def test_labels_to_colormap_all_noise(self):
    labels = np.array([-1, -1, -1])
    encoded, unique, cmap, norm = OGSCL.labels_to_colormap(labels)
    self.assertTrue(np.array_equal(unique, np.array([-1])))
    self.assertTrue(np.array_equal(encoded, np.array([0, 0, 0])))
    self.assertIsNotNone(cmap)
    self.assertIsNotNone(norm)


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
    model.fit(X)
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
    model.fit(X)
    ax = model.plot()
    self.assertIsInstance(ax, Axes)
    plt.close(ax.figure)

  def test_plot_3d_returns_axes(self):
    X, _ = make_blobs(n_samples=60, centers=2, n_features=3,
                      random_state=4)
    model = OGSCL.OGSKMeans(n_clusters=2, random_state=0)
    model.fit(X)
    ax = model.plot_3d()
    self.assertIsInstance(ax, Axes)
    plt.close(ax.figure)

  def test_dbscan_highlight_core(self):
    X, _ = make_blobs(n_samples=80, centers=2, n_features=2,
                      random_state=5)
    model = OGSCL.OGSDBSCAN(eps=0.5, min_samples=5)
    model.fit(X)
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
    metric = OGSCL.SilhouetteMetric(X, labels).compute()
    self.assertIsInstance(metric, float)

  def test_supervised_metrics_none_without_labels(self):
    X, _ = make_blobs(n_samples=40, centers=2, n_features=2,
                      random_state=8)
    model = OGSCL.OGSKMeans(n_clusters=2, random_state=0)
    labels = model.fit_predict(X)
    metric = OGSCL.AdjustedRandMetric(X, labels, None).compute()
    self.assertIsNone(metric)

  def test_supervised_metrics_with_labels(self):
    X, y = make_blobs(n_samples=40, centers=2, n_features=2,
                      random_state=9)
    model = OGSCL.OGSKMeans(n_clusters=2, random_state=0)
    labels = model.fit_predict(X)
    metric = OGSCL.AdjustedRandMetric(X, labels, y).compute()
    self.assertIsInstance(metric, float)


if __name__ == "__main__":
    unittest.main()

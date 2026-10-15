from functools import cache
import logging

import numpy as np
import pytest
from scipy.special import gammaln

from OGS.src.ogsclustering import (
    ManifoldBenchmark,
    OGSAdvancedDensityPeaks,
    OGSAdvancedDensityPeaksPP,
    PAkDensitySeparationScore,
)


CASES = [(entry[0], entry[1]) for entry in ManifoldBenchmark.REGISTRY]
MODELS = [OGSAdvancedDensityPeaks, OGSAdvancedDensityPeaksPP]


@cache
def geometry(name):
    generator = dict(CASES)[name]
    coordinates, _ = generator(N=192, seed=42)
    distances, indices = PAkDensitySeparationScore._compute_nn_distances(
        coordinates, 40, 1
    )
    dimension = PAkDensitySeparationScore._estimate_intrinsic_dim(distances)
    ranks = PAkDensitySeparationScore._compute_kstar_adaptive(
        distances, dimension, 40, 23.92812698
    )
    return coordinates, distances, indices, dimension, ranks


@cache
def fitted(name, model_type):
    coordinates, _, _, _, _ = geometry(name)
    model = model_type(Z=0.05, halo=False, maxk=40, n_jobs=1)
    model.logger.setLevel(logging.ERROR)
    model.fit_predict(coordinates)
    return model


@pytest.mark.parametrize("name,generator", CASES, ids=[c[0] for c in CASES])
def test_density_matches_homogeneous_likelihood(name, generator):
    _, distances, _, dimension, ranks = geometry(name)
    actual, error = PAkDensitySeparationScore._density_pak(
        distances, ranks, dimension, len(distances)
    )
    log_prefactor = dimension / 2 * np.log(np.pi) - gammaln(dimension / 2 + 1)
    expected = (
        np.log(ranks)
        - log_prefactor
        - dimension * np.log(distances[np.arange(len(distances)), ranks])
    )
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-10)
    np.testing.assert_allclose(error**2, (4 * ranks + 2) / (ranks * (ranks - 1)))


@pytest.mark.parametrize("scale", [1e-100, 0.001, 0.1, 10.0, 1e100])
@pytest.mark.parametrize("name,generator", CASES, ids=[c[0] for c in CASES])
def test_density_uniform_scale_covariance(name, generator, scale):
    _, distances, _, dimension, ranks = geometry(name)
    original, original_error = PAkDensitySeparationScore._density_pak(
        distances, ranks, dimension, len(distances)
    )
    transformed, transformed_error = PAkDensitySeparationScore._density_pak(
        distances * scale, ranks, dimension, len(distances)
    )
    np.testing.assert_allclose(
        transformed, original - dimension * np.log(scale),
        atol=1e-8, rtol=1e-10,
    )
    np.testing.assert_array_equal(transformed_error, original_error)


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_density_log_space_handles_extreme_dimension_and_tied_shells(scale):
    distances = np.array([
        [0.0, 1.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0, 4.0],
    ]) * scale
    ranks = np.array([2, 3, 4])
    dimension = 1200.0
    actual, error = PAkDensitySeparationScore._density_pak(
        distances, ranks, dimension, len(distances)
    )
    log_volume = (
        dimension / 2 * np.log(np.pi)
        - gammaln(dimension / 2 + 1)
        + dimension * np.log(distances[np.arange(len(distances)), ranks])
    )
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, np.log(ranks) - log_volume)
    np.testing.assert_allclose(error**2, (4 * ranks + 2) / (ranks * (ranks - 1)))


@pytest.mark.parametrize("name,generator", CASES, ids=[c[0] for c in CASES])
def test_rank_statistic_first_crossing_and_threshold_monotonicity(name, generator):
    _, distances, _, dimension, _ = geometry(name)
    results = []
    for threshold in [6.63, 23.92812698, 100.0]:
        expected = []
        for row in distances:
            crossings = [
                k for k in range(4, 40)
                if (
                    2 * (k * dimension + 1) * np.log(row[k + 1] / row[k])
                    - 2 * np.log(dimension) - 2 * np.log(k)
                ) > threshold
            ]
            expected.append(crossings[0] if crossings else 40)
        actual = PAkDensitySeparationScore._compute_kstar_adaptive(
            distances, dimension, 40, threshold
        )
        np.testing.assert_array_equal(actual, expected)
        results.append(actual)
    assert np.all(results[0] <= results[1])
    assert np.all(results[1] <= results[2])


@pytest.mark.parametrize("name,generator", CASES, ids=[c[0] for c in CASES])
def test_pointer_jumping_matches_explicit_strict_ascent(name, generator):
    _, distances, indices, dimension, ranks = geometry(name)
    density, error = PAkDensitySeparationScore._density_pak(
        distances, ranks, dimension, len(distances)
    )
    score = density - error
    parent = np.arange(len(score))
    for point in range(len(score)):
        neighbors = indices[point, 1:ranks[point] + 1]
        higher = neighbors[score[neighbors] > score[point]]
        if len(higher):
            parent[point] = higher[np.argmax(score[higher])]
    roots = parent.copy()
    for point in range(len(score)):
        visited = set()
        while roots[point] != parent[roots[point]]:
            assert roots[point] not in visited
            visited.add(int(roots[point]))
            roots[point] = parent[roots[point]]
    labels, centers, members = OGSAdvancedDensityPeaksPP()._fused_mode_and_assignment(
        score, ranks, indices, len(score), 40
    )
    np.testing.assert_array_equal(np.asarray(centers)[labels], roots)
    assert sorted(point for cluster in members for point in cluster) == list(range(len(score)))


@pytest.mark.parametrize("model_type", MODELS, ids=["ADP", "ADP++"])
@pytest.mark.parametrize("name,generator", CASES, ids=[c[0] for c in CASES])
def test_final_partition_is_complete(name, generator, model_type):
    model = fitted(name, model_type)
    assert sorted(point for cluster in model.cluster_indices_ for point in cluster) == list(range(192))
    for label, members in enumerate(model.cluster_indices_):
        assert np.all(model.labels_[members] == label)
        assert model.cluster_centers_[label] in members


@pytest.mark.parametrize("model_type", MODELS, ids=["ADP", "ADP++"])
@pytest.mark.parametrize("name,generator", CASES, ids=[c[0] for c in CASES])
def test_final_saddle_endpoints_explain_density_and_error(name, generator, model_type):
    model = fitted(name, model_type)
    if model.log_den_bord_ is None or not np.any(np.isfinite(model.log_den_bord_)):
        pytest.skip("No finite final saddle on this fixture")
    for left, right in zip(*np.where(np.isfinite(model.log_den_bord_))):
        endpoints = model.bord_indices_[left, right]
        assert np.all(endpoints >= 0)
        assert np.array_equal(model.labels_[endpoints], [left, right])
        np.testing.assert_allclose(
            model.log_den_bord_[left, right], np.mean(model.log_den_[endpoints]),
            atol=1e-12, rtol=1e-12,
        )
        np.testing.assert_allclose(
            model.log_den_bord_err_[left, right],
            np.mean(model.log_den_err_[endpoints]),
            atol=1e-12, rtol=1e-12,
        )
        np.testing.assert_array_equal(
            model.bord_indices_[right, left], endpoints[::-1]
        )


@pytest.mark.parametrize("model_type", MODELS, ids=["ADP", "ADP++"])
@pytest.mark.parametrize("keep", [0, 1])
@pytest.mark.parametrize("removed_border", [4.0, 5.0, 6.0])
def test_merge_preserves_selected_edge_record(model_type, keep, removed_border):
    remove = 1 - keep
    density = np.array([9.0, 9.0, 15.0])
    density[keep] = 10.0
    borders = np.full((3, 3), -np.inf)
    errors = np.zeros((3, 3))
    endpoints = np.full((3, 3, 2), -1, dtype=int)
    for left, right, value, error in [
        (0, 1, 8.5, 0.1),
        (keep, 2, 5.0, 0.2),
        (remove, 2, removed_border, 0.3),
    ]:
        borders[left, right] = borders[right, left] = value
        errors[left, right] = errors[right, left] = error
        endpoints[left, right] = [left, right]
        endpoints[right, left] = [right, left]
    result = model_type()._multimodality_test(
        1.0, [0, 1, 2], density, np.ones(3), borders, errors, endpoints,
        np.arange(3), [[0], [1], [2]], 3,
    )
    assert result[-1] == 2
    np.testing.assert_array_equal(result[1], [keep, keep, 2])
    selected = remove if removed_border > 5.0 else keep
    assert borders[keep, 2] == max(5.0, removed_border)
    assert errors[keep, 2] == (0.3 if selected == remove else 0.2)
    assert borders[2, keep] == borders[keep, 2]
    assert errors[2, keep] == errors[keep, 2]
    np.testing.assert_array_equal(endpoints[keep, 2], [selected, 2])
    np.testing.assert_array_equal(endpoints[2, keep], [2, selected])


@pytest.mark.parametrize("model_type", MODELS, ids=["ADP", "ADP++"])
def test_merge_provenance_survives_multiple_absorptions(model_type):
    model = model_type()
    density = np.array([10.0, 9.0, 8.0, 15.0])
    borders = np.full((4, 4), -np.inf)
    errors = np.zeros((4, 4))
    endpoints = np.full((4, 4, 2), -1, dtype=int)
    for left, right, value, error in [
        (0, 1, 8.5, 0.1), (1, 2, 7.5, 0.1),
        (0, 3, 4.0, 0.2), (1, 3, 5.0, 0.3), (2, 3, 6.0, 0.4),
    ]:
        borders[left, right] = borders[right, left] = value
        errors[left, right] = errors[right, left] = error
        endpoints[left, right] = [left, right]
        endpoints[right, left] = [right, left]
    centers, labels, members, borders, errors, n_final = model._multimodality_test(
        1.0, [0, 1, 2, 3], density, np.ones(4), borders, errors, endpoints,
        np.arange(4), [[0], [1], [2], [3]], 4,
    )
    assert n_final == 2
    final = model._finalize(
        False, centers, labels, members, density, borders, errors, endpoints, 4, 4
    )
    np.testing.assert_array_equal(final[0], [0, 0, 0, 1])
    assert final[3][0, 1] == final[3][1, 0] == 6.0
    assert final[4][0, 1] == final[4][1, 0] == 0.4
    np.testing.assert_array_equal(final[5][0, 1], [2, 3])
    np.testing.assert_array_equal(final[5][1, 0], [3, 2])


@pytest.mark.parametrize("name,generator", CASES, ids=[c[0] for c in CASES])
def test_score_aggregation_matches_operational_equation(name, generator):
    coordinates, labels, _ = ManifoldBenchmark.make_two_cluster(
        generator, N_per=96, sep=20.0, seed=42
    )
    scorer = PAkDensitySeparationScore(coordinates, labels, maxk=40, n_jobs=1)
    result = scorer.compute()
    assert result is not None
    assert np.isfinite(result)
    assert result >= 0
    weighted = 0.0
    total = 0.0
    for pair, actual_contrast in scorer.pair_zscores_.items():
        left, right = pair
        peak_left, error_left = scorer.cluster_peak_densities_[left]
        peak_right, error_right = scorer.cluster_peak_densities_[right]
        if pair in scorer.saddle_densities_:
            saddle, saddle_error = scorer.saddle_densities_[pair]
        else:
            minimum = np.argmin(scorer.log_den_)
            saddle = scorer.log_den_[minimum]
            saddle_error = scorer.log_den_err_[minimum]
        expected = (
            min(peak_left, peak_right) - saddle
        ) / np.sqrt(error_left**2 + error_right**2 + saddle_error**2)
        np.testing.assert_allclose(actual_contrast, expected)
        weight = np.sqrt(np.sum(labels == left) * np.sum(labels == right))
        weighted += weight * expected
        total += weight
    np.testing.assert_allclose(result, weighted / total)


@pytest.mark.parametrize("model_type", MODELS, ids=["ADP", "ADP++"])
def test_three_sample_fit_does_not_read_nonexistent_neighbors(model_type):
    model = model_type(n_jobs=1)
    model.fit_predict(np.array([[0.0], [1.0], [3.0]]))
    assert np.all(model.kstar_ <= model.distances_.shape[1] - 1)


@pytest.mark.parametrize("model_type", MODELS, ids=["ADP", "ADP++"])
def test_refit_one_cluster_does_not_retain_previous_saddles(model_type):
    coordinates, _ = ManifoldBenchmark.gen_M5(N=192, seed=42)
    model = model_type(Z=0.0, halo=False, maxk=40, n_jobs=1)
    model.logger.setLevel(logging.ERROR)
    model.fit_predict(coordinates)
    assert model.n_clusters_ > 1
    model.fit_predict((2.0 ** np.arange(12))[:, None])
    assert model.n_clusters_ == 1
    assert model.log_den_bord_ is None or model.log_den_bord_.shape == (1, 1)
    assert model.bord_indices_ is None or model.bord_indices_.shape == (1, 1, 2)


@pytest.mark.parametrize("model_type", MODELS, ids=["ADP", "ADP++"])
def test_merge_decisions_are_invariant_to_log_density_offset(model_type):
    coordinates, _ = ManifoldBenchmark.gen_M5(N=192, seed=42)
    model = model_type(Z=0.05, halo=False, maxk=40, n_jobs=1)
    distances, indices = model._compute_nn_distances(coordinates, 40, 1)
    dimension = model._estimate_intrinsic_dim(distances)
    ranks = model._compute_kstar_adaptive(distances, dimension, 40, 23.92812698)
    density, error = model._density_pak(distances, ranks, dimension, len(coordinates))
    score = density - error
    if model_type is OGSAdvancedDensityPeaksPP:
        initial, centers, members = model._fused_mode_and_assignment(
            score, ranks, indices, len(coordinates), 40
        )
    else:
        centers, removed = model._find_density_modes(score, ranks, indices, len(coordinates))
        initial, members = model._assign_clusters(
            score, centers, removed, indices, ranks, len(coordinates)
        )
    borders, border_error, border_indices = model._find_saddle_points(
        initial, density, error, indices, ranks, centers,
        len(coordinates), len(centers),
    )
    results = []
    for offset in [0.0, -100.0, 100.0]:
        result = model._multimodality_test(
            0.05, centers.copy(), density + offset, error,
            borders.copy() + offset, border_error.copy(),
            border_indices.copy(),
            initial.copy(), [points.copy() for points in members], len(centers),
        )
        results.append(result[1])
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(results[0], results[2])


def test_registry_score_is_invariant_to_uniform_change_of_units():
    coordinates, labels, _ = ManifoldBenchmark.make_two_cluster(
        ManifoldBenchmark.gen_MP9, N_per=96, sep=20.0, seed=42
    )
    original = PAkDensitySeparationScore(
        coordinates, labels, maxk=40, n_jobs=1
    )
    transformed = PAkDensitySeparationScore(
        coordinates * 0.001, labels, maxk=40, n_jobs=1
    )
    original_score = original.compute()
    transformed_score = transformed.compute()
    np.testing.assert_allclose(
        original.intrinsic_dim_, transformed.intrinsic_dim_, rtol=1e-8
    )
    np.testing.assert_array_equal(original.kstar_, transformed.kstar_)
    np.testing.assert_allclose(original_score, transformed_score, atol=1e-8, rtol=1e-8)

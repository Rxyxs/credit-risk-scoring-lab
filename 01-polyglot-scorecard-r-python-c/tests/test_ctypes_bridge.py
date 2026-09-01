import numpy as np
import pytest

from src.ctypes_bridge import ScoreEngine, score_batch_numpy, score_batch_python


def _tiny_example():
    # 2 features, max 3 bins. Aplicante 0: bin(0,1) -> puntos 10+50=60+base.
    # Aplicante 1: bin(2,0) -> puntos 30+40=70+base.
    points_table = np.array([
        [10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0],
    ])
    bin_indices = np.array([[0, 1], [2, 0]], dtype=np.int32)
    base_points = 500.0
    expected = np.array([500 + 10 + 50, 500 + 30 + 40])
    return bin_indices, points_table, base_points, expected


def test_numpy_reference_matches_hand_computed():
    bin_indices, points_table, base_points, expected = _tiny_example()
    result = score_batch_numpy(bin_indices, points_table, base_points)
    np.testing.assert_allclose(result, expected)


def test_python_reference_matches_hand_computed():
    bin_indices, points_table, base_points, expected = _tiny_example()
    result = score_batch_python(bin_indices, points_table, base_points)
    np.testing.assert_allclose(result, expected)


def test_c_engine_matches_numpy_and_hand_computed():
    try:
        engine = ScoreEngine()
    except FileNotFoundError:
        pytest.skip("score_engine.dll no compilada -- correr powershell -File c/build.ps1 primero")

    bin_indices, points_table, base_points, expected = _tiny_example()
    result = engine.score_batch(bin_indices, points_table, base_points)
    np.testing.assert_allclose(result, expected)


def test_c_engine_matches_numpy_on_random_batch():
    try:
        engine = ScoreEngine()
    except FileNotFoundError:
        pytest.skip("score_engine.dll no compilada -- correr powershell -File c/build.ps1 primero")

    rng = np.random.default_rng(0)
    n_rows, n_features, max_bins = 500, 6, 5
    points_table = rng.normal(0, 10, size=(n_features, max_bins))
    bin_indices = rng.integers(0, max_bins, size=(n_rows, n_features)).astype(np.int32)
    base_points = 560.0

    c_result = engine.score_batch(bin_indices, points_table, base_points)
    np_result = score_batch_numpy(bin_indices, points_table, base_points)
    np.testing.assert_allclose(c_result, np_result, atol=1e-9)

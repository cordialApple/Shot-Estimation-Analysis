import numpy as np

from shotlab import metrics, store, tuning
from shotlab.calibration import Platt


def test_one_standard_error_prefers_simpler_within_band():
    results = {
        "params": [
            {"m": 30, "c": 10},
            {"m": 3000, "c": 1},
            {"m": 3000, "c": 10},
            {"m": 300, "c": 1},
        ],
        "mean_test_log_loss": np.array([-0.6200, -0.6203, -0.6202, -0.6300]),
        "std_test_log_loss": np.array([0.002, 0.002, 0.002, 0.002]),
    }
    for i in range(4):
        results[f"split{i}_test_log_loss"] = None
    order = [["m", "desc"], ["c", "asc"]]
    assert tuning.one_standard_error_index(results, order) == 1


def test_one_standard_error_keeps_best_when_band_is_tight():
    results = {
        "params": [{"m": 30}, {"m": 3000}],
        "mean_test_log_loss": np.array([-0.62, -0.64]),
        "std_test_log_loss": np.array([0.001, 0.001]),
    }
    for i in range(5):
        results[f"split{i}_test_log_loss"] = None
    assert tuning.one_standard_error_index(results, [["m", "desc"]]) == 0


def test_platt_recovers_known_distortion():
    rng = np.random.default_rng(0)
    truth = rng.uniform(0.1, 0.9, 200000)
    y = (rng.random(truth.size) < truth).astype(int)
    log_odds = np.log(truth / (1 - truth))
    distorted = 1 / (1 + np.exp(-(2.0 * log_odds + 0.5)))
    fixed = Platt().fit(distorted, y).apply(distorted)
    assert abs(fixed - truth).max() < 0.02
    slope, intercept = metrics.calibration_line(y, fixed)
    assert abs(slope - 1) < 0.02
    assert abs(intercept) < 0.02


def test_cluster_bootstrap_covers_zero_for_noise():
    rng = np.random.default_rng(1)
    diff = rng.normal(0, 1, 20000)
    clusters = rng.integers(0, 300, diff.size)
    out = metrics.cluster_bootstrap(diff, clusters, draws=500, seed=0)
    assert out["ci_low"] < 0 < out["ci_high"]
    shifted = metrics.cluster_bootstrap(diff + 0.1, clusters, draws=500, seed=0)
    assert shifted["ci_low"] > 0
    assert shifted["p_value"] < 0.01


def test_holm_is_monotone_and_bounded():
    adjusted = metrics.holm([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_trivial_baselines_match_plan():
    y = np.array([1] * 458 + [0] * 542)
    always_make = metrics.threshold_metrics(y, np.ones(1000))
    assert abs(always_make["f1"] - 0.628) < 0.001
    always_miss = metrics.threshold_metrics(y, np.zeros(1000))
    assert abs(always_miss["accuracy"] - 0.542) < 0.001


def test_config_hash_is_order_independent():
    assert store.config_hash({"a": 1, "b": [1, 2]}) == store.config_hash({"b": [1, 2], "a": 1})
    assert store.config_hash({"a": 1}) != store.config_hash({"a": 2})

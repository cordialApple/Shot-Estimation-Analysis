import json
import time

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score

from shotlab.data import RESULTS, config

EPS = 1e-12
THRESHOLD = 0.5


def clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def per_row_log_loss(y, p):
    p = clip(p)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def per_row_brier(y, p):
    return (np.asarray(p, dtype=float) - y) ** 2


def equal_count_bins(p, bins):
    order = np.argsort(p, kind="stable")
    labels = np.empty(len(p), dtype=int)
    labels[order] = np.arange(len(p)) * bins // len(p)
    return labels


def reliability_table(y, p, bins=20):
    frame = pd.DataFrame({"y": y, "p": p, "bin": equal_count_bins(np.asarray(p), bins)})
    table = frame.groupby("bin").agg(
        n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")
    )
    table["se"] = np.sqrt(table["observed"] * (1 - table["observed"]) / table["n"])
    return table


def murphy(y, p, bins=20):
    """Murphy decomposition of the Brier score over equal-count bins."""
    table = reliability_table(y, p, bins)
    weight = table["n"] / table["n"].sum()
    base = np.mean(y)
    gap = table["predicted"] - table["observed"]
    lift = table["observed"] - base
    return {
        "reliability": float((weight * gap**2).sum()),
        "resolution": float((weight * lift**2).sum()),
        "uncertainty": float(base * (1 - base)),
    }


def calibration_line(y, p):
    """Slope and intercept of a logistic fit of y on logit(p)."""
    x = logit(clip(p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=1000).fit(x, y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def expected_calibration_error(y, p, bins=15):
    table = reliability_table(y, p, bins)
    weight = table["n"] / table["n"].sum()
    gap = (table["predicted"] - table["observed"]).abs()
    return float((weight * gap).sum())


def threshold_metrics(y, p, threshold=THRESHOLD):
    predicted = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float((tp + tn) / len(y)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float((recall + specificity) / 2),
        "mcc": float(matthews_corrcoef(y, predicted)),
    }


def summarize(y, p, base_rate):
    y = np.asarray(y, dtype=int)
    loss = float(per_row_log_loss(y, p).mean())
    base_loss = float(per_row_log_loss(y, np.full(len(y), base_rate)).mean())
    slope, intercept = calibration_line(y, p)
    return {
        "log_loss": loss,
        "skill": 1 - loss / base_loss,
        "brier": float(per_row_brier(y, p).mean()),
        "auc": float(roc_auc_score(y, p)),
        "ece": expected_calibration_error(y, p),
        "cal_slope": slope,
        "cal_intercept": intercept,
        **murphy(y, p),
        **threshold_metrics(y, p),
    }


def cluster_bootstrap(diff, clusters, draws=None, seed=None):
    """Paired bootstrap of a mean per-row difference, resampling whole clusters."""
    draws = draws or config()["bootstrap_draws"]
    seed = config()["seed"] if seed is None else seed
    codes, _ = pd.factorize(np.asarray(clusters))
    groups = codes.max() + 1
    sums = np.bincount(codes, weights=diff, minlength=groups)
    counts = np.bincount(codes, minlength=groups).astype(float)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(groups, np.full(groups, 1 / groups), size=draws)
    stats = (weights @ sums) / (weights @ counts)
    low, high = np.percentile(stats, [2.5, 97.5])
    below = (stats <= 0).mean()
    above = (stats >= 0).mean()
    p_value = min(1.0, 2 * min(below, above))
    return {
        "mean": float(np.mean(diff)),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(p_value),
    }


def holm(p_values):
    """Holm step-down adjusted p-values, in the order given."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def log_heldout_scoring(what, detail):
    """Appends to the ledger of every time held-out outcomes were scored."""
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "what": what, "detail": detail}
    with open(RESULTS / "scoring_ledger.jsonl", "a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(entry) + "\n")

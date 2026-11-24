import json

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

from shotlab import features, metrics, splits, store
from shotlab.calibration import Isotonic
from shotlab.data import ARTIFACTS, TARGET, config
from shotlab.features import LADDER
from shotlab.runs import CROSS_SEASON, DIRECTIONS, TUNING

EXPLAIN = ARTIFACTS / "explain"
SLICE_MIN = 500
SLICE_ORDER = {
    "shot type": ["dunk", "layup", "tip", "floater", "post", "jumper", "heave"],
    "distance": ["two, 0–4 ft", "two, 4–14 ft", "two, 14+ ft", "three, under 24 ft", "three, 24–27 ft", "three, 27+ ft"],
    "closest defender": ["0–2 ft", "2–4 ft", "4–6 ft", "6–10 ft", "10+ ft"],
    "shot clock": ["exactly 0", "0–4 s", "4–12 s", "12–20 s", "20–24 s", "exactly 24"],
    "contesters": ["0", "1", "2", "3+"],
    "shooter history": ["0 shots", "1–49", "50–299", "300+"],
}
CONTRASTS = [("tabfm", "hgb"), ("tabicl", "hgb"), ("hgb", "logreg")]
SIMPLICITY = ["logreg", "hgb", "tabicl", "tabfm"]
RUNG_LABELS = ["location", "+ shot type", "+ creation", "+ defense", "+ closeout", "+ identity (primary)", "+ team and month"]
RUNG_ARMS = [f"ablation:ladder:{i + 1}_{LADDER[i]}" for i in range(len(LADDER) - 1)] + ["primary", "ablation:plus_team_month"]
_catalog = None


def catalog(refresh=False):
    global _catalog
    if _catalog is None or refresh:
        _catalog = store.catalog()
    return _catalog


def labeled_table():
    """Feature table with outcomes, panel flags and the columns the slices need."""
    manifest = splits.load_manifest()
    table = features.load_features().merge(manifest[["shot_id", "panel", "tuning_query"]], on="shot_id", validate="one_to_one")
    table["team_month"] = table["off_team_id"].astype(str) + "|" + table["season"].astype(str) + "|" + table["month_block"].astype(str)
    return table


def files(model, direction, arm="primary", role="eval", n_train=0, replicate=0):
    listing = catalog()
    keep = ((listing["model"] == model) & (listing["direction"] == direction) & (listing["arm"] == arm)
            & (listing["role"] == role) & (listing["n_train"] == n_train) & (listing["replicate"] == replicate))
    return listing[keep]


def has(model, direction, **keys):
    return len(files(model, direction, **keys)) > 0


def predictions(model, direction, seed=0, **keys):
    listing = files(model, direction, **keys)
    if len(listing) != 1:
        raise ValueError(f"expected one prediction file for {model} {direction} {keys}, found {len(listing)}")
    frame = pd.read_parquet(listing["path"].iloc[0])
    if (frame["context_seed"] >= 0).any() and seed is not None:
        frame = frame[frame["context_seed"] == seed]
    return frame[["shot_id", "fold", "context_seed", "p_raw", "p_cal"]]


def base_rate(table, direction):
    return float(table.loc[table["season"].isin(DIRECTIONS[direction][0]), TARGET].mean())


def scored(table, model, direction, scope="panel", **keys):
    """Held-out rows joined to one model's predictions. Fails if any row of the scope has no prediction."""
    held = table[table["season"] == DIRECTIONS[direction][1]]
    if scope == "panel":
        held = held[held["panel"]]
    frame = held.merge(predictions(model, direction, **keys), on="shot_id", how="left", validate="one_to_one")
    missing = int(frame["p_raw"].isna().sum())
    if missing:
        raise ValueError(f"{model} {direction} {keys}: {missing} of {len(frame)} {scope} shots have no prediction")
    frame["y"] = frame[TARGET].astype(int)
    return frame.sort_values("shot_id").reset_index(drop=True)


def headline(table, models, scope="panel", arm="primary"):
    rows = {}
    for model in models:
        for direction in CROSS_SEASON:
            frame = scored(table, model, direction, scope, arm=arm)
            for name, column in [("raw", "p_raw"), ("calibrated", "p_cal")]:
                rows[(model, direction, name)] = metrics.summarize(frame["y"].to_numpy(), frame[column].to_numpy(), base_rate(table, direction))
    metrics.log_heldout_scoring("headline", {"models": list(models), "scope": scope, "arm": arm})
    return pd.DataFrame(rows).T.rename_axis(["model", "direction", "probability"])


def baselines(table):
    """Constant predictions: the training-season make rate, always make, always miss."""
    rows = {}
    for direction in CROSS_SEASON:
        held = table[(table["season"] == DIRECTIONS[direction][1]) & table["panel"]]
        y = held[TARGET].astype(int).to_numpy()
        rate = np.full(len(y), base_rate(table, direction))
        rows[("base rate", direction)] = {"log_loss": float(metrics.per_row_log_loss(y, rate).mean()),
                                         "brier": float(metrics.per_row_brier(y, rate).mean()),
                                         **metrics.threshold_metrics(y, rate)}
        rows[("always make", direction)] = metrics.threshold_metrics(y, np.ones(len(y)))
        rows[("always miss", direction)] = metrics.threshold_metrics(y, np.zeros(len(y)))
    return pd.DataFrame(rows).T.rename_axis(["model", "direction"])


def loss_difference(table, first, second, direction, column="p_cal"):
    """Per-shot log loss of the first model minus the second on the shared panel. Negative favours the first."""
    a, b = scored(table, first, direction), scored(table, second, direction)
    assert a["shot_id"].equals(b["shot_id"])
    y = a["y"].to_numpy()
    diff = metrics.per_row_log_loss(y, a[column].to_numpy()) - metrics.per_row_log_loss(y, b[column].to_numpy())
    return a[["shot_id", "shooter_id", "team_month"]].assign(diff=diff, direction=direction)


def contrasts(table, pairs=CONTRASTS, clusters="shooter_id", column="p_cal"):
    """Cluster-bootstrap intervals per direction and pooled; Holm correction on the pooled p-values."""
    rows = []
    for first, second in pairs:
        parts = [loss_difference(table, first, second, direction, column) for direction in CROSS_SEASON]
        for label, part in list(zip(CROSS_SEASON, parts)) + [("pooled", pd.concat(parts, ignore_index=True))]:
            rows.append({"contrast": f"{first} − {second}", "direction": label,
                         **metrics.cluster_bootstrap(part["diff"].to_numpy(), part[clusters].to_numpy())})
    out = pd.DataFrame(rows)
    pooled = out["direction"] == "pooled"
    out.loc[pooled, "p_holm"] = metrics.holm(out.loc[pooled, "p_value"].to_numpy())
    signs = out[~pooled].groupby("contrast")["mean"].agg(lambda means: len(set(np.sign(means))) == 1)
    out["same_sign_both_directions"] = out["contrast"].map(signs)
    out["tie_under_threshold"] = out["mean"].abs() < config()["tie_threshold_log_loss"]
    metrics.log_heldout_scoring("contrasts", {"pairs": [list(p) for p in pairs], "clusters": clusters, "column": column})
    return out


def band(values, edges, labels):
    return pd.cut(values, edges, labels=labels, right=False).astype(str)


def slice_columns(table, direction):
    """The pre-registered slices of the held-out panel."""
    held = table[(table["season"] == DIRECTIONS[direction][1]) & table["panel"]].sort_values("shot_id").reset_index(drop=True)
    train = table[table["season"].isin(DIRECTIONS[direction][0])]
    history = held["shooter_id"].map(train["shooter_id"].value_counts()).fillna(0)
    three = held["three"] == 1
    clock = held["shotclock"]
    slices = pd.DataFrame({
        "shot type": held["shottype"].astype(str),
        "distance": np.where(three, band(held["distance"], [0, 24, 27, 200], ["three, under 24 ft", "three, 24–27 ft", "three, 27+ ft"]),
                             band(held["distance"], [0, 4, 14, 200], ["two, 0–4 ft", "two, 4–14 ft", "two, 14+ ft"])),
        "closest defender": band(held["closestdefdist"], [0, 2, 4, 6, 10, 200], ["0–2 ft", "2–4 ft", "4–6 ft", "6–10 ft", "10+ ft"]),
        "shot clock": np.select([clock == 0, clock == 24], ["exactly 0", "exactly 24"],
                                band(clock, [0, 4, 12, 20, 25], ["0–4 s", "4–12 s", "12–20 s", "20–24 s"])),
        "contesters": held["num_contesters"].astype(int).map({0: "0", 1: "1", 2: "2", 3: "3+"}),
        "shooter history": band(history, [0, 1, 50, 300, 10**6], ["0 shots", "1–49", "50–299", "300+"]),
    })
    return held, slices


def slice_table(table, models, column="p_cal"):
    rows = []
    for direction in CROSS_SEASON:
        held, slices = slice_columns(table, direction)
        y = held[TARGET].astype(int).to_numpy()
        for model in models:
            p = scored(table, model, direction)[column].to_numpy()
            loss = metrics.per_row_log_loss(y, p)
            for family in slices:
                grouped = pd.DataFrame({"level": slices[family], "y": y, "p": p, "loss": loss}).groupby("level")
                part = grouped.agg(shots=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"), log_loss=("loss", "mean"))
                part = part[part["shots"] >= SLICE_MIN].reset_index()
                rows.append(part.assign(slice=family, model=model, direction=direction))
    metrics.log_heldout_scoring("slices", {"models": list(models), "column": column})
    out = pd.concat(rows, ignore_index=True)
    out["gap"] = out["predicted"] - out["observed"]
    out["rank"] = [list(SLICE_ORDER).index(s) * 100 + SLICE_ORDER[s].index(level) for s, level in zip(out["slice"], out["level"])]
    return out.sort_values(["rank", "model", "direction"], kind="stable").reset_index(drop=True)


def arm_scores(table, models, arms, column="p_cal"):
    """Panel log loss and AUC of each available arm, both directions."""
    rows = []
    for model in models:
        for arm in arms:
            for direction in CROSS_SEASON:
                if not has(model, direction, arm=arm):
                    continue
                frame = scored(table, model, direction, arm=arm)
                rows.append({"model": model, "arm": arm, "direction": direction,
                             **{k: v for k, v in metrics.summarize(frame["y"].to_numpy(), frame[column].to_numpy(), base_rate(table, direction)).items()
                                if k in ("log_loss", "auc", "brier", "cal_slope", "ece")}})
    metrics.log_heldout_scoring("arms", {"models": list(models), "arms": list(arms), "column": column})
    return pd.DataFrame(rows)


def learning_curve(table, models, column="p_cal"):
    curve = config()["learning_curve"]
    runs = [(size, replicate) for size in curve["sizes"] for replicate in range(curve["replicates"][size])]
    rows = []
    for model in models:
        for direction in CROSS_SEASON:
            for size, replicate in runs:
                if not has(model, direction, arm="learning_curve", n_train=size, replicate=replicate):
                    continue
                frame = scored(table, model, direction, arm="learning_curve", n_train=size, replicate=replicate)
                rows.append({"model": model, "direction": direction, "n_train": size, "replicate": replicate,
                             "log_loss": float(metrics.per_row_log_loss(frame["y"].to_numpy(), frame[column].to_numpy()).mean())})
            if has(model, direction):
                frame = scored(table, model, direction)
                rows.append({"model": model, "direction": direction, "n_train": int((table["season"] == DIRECTIONS[direction][0][0]).sum()),
                             "replicate": 0, "log_loss": float(metrics.per_row_log_loss(frame["y"].to_numpy(), frame[column].to_numpy()).mean())})
    metrics.log_heldout_scoring("learning_curve", {"models": list(models), "column": column})
    return pd.DataFrame(rows)


def seed_spread(table, model, column="p_cal"):
    rows = []
    for direction in CROSS_SEASON:
        frame = table[["shot_id", TARGET]].merge(predictions(model, direction, seed=None), on="shot_id")
        loss = metrics.per_row_log_loss(frame[TARGET].astype(int).to_numpy(), frame[column].to_numpy())
        for seed, value in pd.Series(loss).groupby(frame["context_seed"].to_numpy()).mean().items():
            rows.append({"model": model, "direction": direction, "context_seed": int(seed), "log_loss": float(value)})
    metrics.log_heldout_scoring("seed_spread", {"model": model, "column": column})
    return pd.DataFrame(rows)


def inner_oof(table, model, season):
    """Month-blocked out-of-fold predictions for one season, from the direction that trains on it."""
    direction = next(d for d in CROSS_SEASON if DIRECTIONS[d][0] == [season])
    frame = predictions(model, direction, role="inner_oof")
    return frame.merge(table[["shot_id", TARGET, "panel"]], on="shot_id", validate="one_to_one").assign(y=lambda f: f[TARGET].astype(int))


def isotonic_sensitivity(table, models):
    """Held-out panel log loss with Platt against isotonic, both fitted on training-season out-of-fold rows."""
    rows = []
    for model in models:
        for direction in CROSS_SEASON:
            oof = inner_oof(table, model, DIRECTIONS[direction][0][0])
            frame = scored(table, model, direction)
            isotonic = Isotonic().fit(oof["p_raw"].to_numpy(), oof["y"].to_numpy()).apply(frame["p_raw"].to_numpy())
            y = frame["y"].to_numpy()
            rows.append({"model": model, "direction": direction,
                         "raw": float(metrics.per_row_log_loss(y, frame["p_raw"].to_numpy()).mean()),
                         "Platt": float(metrics.per_row_log_loss(y, frame["p_cal"].to_numpy()).mean()),
                         "isotonic": float(metrics.per_row_log_loss(y, isotonic).mean())})
    metrics.log_heldout_scoring("isotonic_sensitivity", {"models": list(models)})
    return pd.DataFrame(rows)


def season_shift(table, models, column="p_cal"):
    """Same scored rows, two models: one trained on the other season, one fitted within the season (out-of-fold)."""
    rows = []
    for model in models:
        for direction in CROSS_SEASON:
            season = DIRECTIONS[direction][1]
            across = scored(table, model, direction)[["shot_id", "y", column]]
            within = inner_oof(table, model, season)[["shot_id", column]].rename(columns={column: "within"})
            both = across.merge(within, on="shot_id")
            y = both["y"].to_numpy()
            rows.append({"model": model, "scored_season": season, "shots": len(both),
                         "within_season": float(metrics.per_row_log_loss(y, both["within"].to_numpy()).mean()),
                         "cross_season": float(metrics.per_row_log_loss(y, both[column].to_numpy()).mean())})
    metrics.log_heldout_scoring("season_shift", {"models": list(models), "column": column})
    out = pd.DataFrame(rows)
    out["cost_of_shift"] = out["cross_season"] - out["within_season"]
    return out


def logit_matrix(frames, column="p_cal"):
    return np.column_stack([logit(metrics.clip(frame[column].to_numpy())) for frame in frames])


def stacking(table, models, column="p_cal"):
    """Blend weights fitted on training-season out-of-fold rows shared by all models, applied to the held-out panel."""
    rows, weights = [], {}
    for direction in CROSS_SEASON:
        oofs = [inner_oof(table, model, DIRECTIONS[direction][0][0]).set_index("shot_id") for model in models]
        shared = sorted(set.intersection(*[set(frame.index) for frame in oofs]))
        blend = LogisticRegression(C=1e6, max_iter=1000).fit(logit_matrix([frame.loc[shared] for frame in oofs], column),
                                                             oofs[0].loc[shared, "y"].to_numpy())
        held = [scored(table, model, direction) for model in models]
        y = held[0]["y"].to_numpy()
        stacked = blend.predict_proba(logit_matrix(held, column))[:, 1]
        weights[direction] = dict(zip(models, blend.coef_[0]), intercept=float(blend.intercept_[0]), fit_rows=len(shared))
        single = {model: float(metrics.per_row_log_loss(y, frame[column].to_numpy()).mean()) for model, frame in zip(models, held)}
        rows.append({"direction": direction, **single, "blend": float(metrics.per_row_log_loss(y, stacked).mean())})
    metrics.log_heldout_scoring("stacking", {"models": list(models), "column": column})
    return pd.DataFrame(rows).set_index("direction"), pd.DataFrame(weights)


def agreement(table, models, column="p_cal"):
    """Correlation of held-out log-odds between models, mean of the two directions."""
    matrices = []
    for direction in CROSS_SEASON:
        logits = logit_matrix([scored(table, model, direction) for model in models], column)
        matrices.append(pd.DataFrame(np.corrcoef(logits, rowvar=False), index=models, columns=models))
    return sum(matrices) / len(matrices)


def explanation(model, direction, name):
    return pd.read_parquet(EXPLAIN / model / f"{direction}__{name}.parquet")


def permutation_importance(table, models):
    """Increase in held-out log loss when one feature block is shuffled, on the shared probe (raw probabilities)."""
    rows = []
    outcomes = table.set_index("shot_id")[TARGET]
    for model in models:
        for direction in CROSS_SEASON:
            frame = explanation(model, direction, "permutation")
            y = outcomes.reindex(frame["shot_id"]).astype(int).to_numpy()
            frame = frame.assign(loss=metrics.per_row_log_loss(y, frame["p"].to_numpy()))
            by_block = frame.groupby(["block", "repeat"])["loss"].mean().groupby("block").mean()
            for block, value in (by_block.drop("none") - by_block["none"]).items():
                rows.append({"model": model, "direction": direction, "block": block, "importance": float(value)})
    metrics.log_heldout_scoring("permutation_importance", {"models": list(models)})
    return pd.DataFrame(rows)


def selections(model):
    rows = {}
    for path in sorted((TUNING / model).glob("*selection*.json")):
        selection = json.loads(path.read_text())
        run = selection["run_config"]
        if run["arm"] == "primary" and not run.get("n_train"):
            rows[run["direction"]] = selection
    return rows

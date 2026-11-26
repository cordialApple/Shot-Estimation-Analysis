import json

import pandas as pd

from shotlab import features, store
from shotlab.data import ROOT, config

DIRECTION = "S1+S2_to_T"
SIMPLICITY = ["logreg", "hgb", "tabicl", "tabfm"]
OUTPUT = ROOT / "predictions" / "test_predictions.csv"


def choose(mean_log_loss, candidates):
    """Lowest mean cross-season panel log loss; models within the tie threshold lose to a simpler one."""
    best = min(mean_log_loss[model] for model in candidates)
    tied = [model for model in candidates if mean_log_loss[model] <= best + config()["tie_threshold_log_loss"]]
    return next(model for model in SIMPLICITY if model in tied)


def test_predictions(mean_log_loss):
    """Writes calibrated test-season probabilities: one column per model that predicted T, plus p_final."""
    listing = store.catalog()
    listing = listing[(listing["direction"] == DIRECTION) & (listing["arm"] == "primary") & (listing["role"] == "eval")]
    table = features.load_features()
    out = table.loc[table["season"] == "T", ["shot_id"]].sort_values("shot_id").reset_index(drop=True)
    for model in SIMPLICITY:
        paths = listing.loc[listing["model"] == model, "path"]
        if len(paths) != 1:
            continue
        frame = pd.read_parquet(paths.iloc[0])
        frame = frame[frame["context_seed"] <= 0]
        out = out.merge(frame[["shot_id", "p_cal"]].rename(columns={"p_cal": f"p_{model}"}), on="shot_id", how="left", validate="one_to_one")
        if out[f"p_{model}"].isna().any():
            raise ValueError(f"{model} is missing test-season shots")
    candidates = [model for model in SIMPLICITY if f"p_{model}" in out and model in mean_log_loss]
    chosen = choose(mean_log_loss, candidates)
    out["p_final"] = out[f"p_{chosen}"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False, float_format="%.6f")
    note = {"final_model": chosen, "candidates": candidates, "mean_cross_season_log_loss": {m: float(mean_log_loss[m]) for m in candidates},
            "rows": len(out), "rule": "lowest mean calibrated panel log loss; ties under the threshold go to the simpler model"}
    OUTPUT.with_suffix(".json").write_text(json.dumps(note, indent=2))
    return out, note

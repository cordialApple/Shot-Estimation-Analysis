import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from shotlab import features, store, tuning
from shotlab.calibration import Platt
from shotlab.data import ARTIFACTS, config, load_grid
from shotlab.features import LADDER, PRIMARY_BLOCKS
from shotlab.models import MAKERS
from shotlab.splits import nested_subset

TUNING = ARTIFACTS / "tuning"
MODELS = ARTIFACTS / "models"
DIRECTIONS = {
    "S1_to_S2": (["S1"], "S2"),
    "S2_to_S1": (["S2"], "S1"),
    "S1+S2_to_T": (["S1", "S2"], "T"),
}
CROSS_SEASON = ["S1_to_S2", "S2_to_S1"]
RATE_BLOCKS = {"identity", "team_month"}


def training_frame(table, direction, n_train=None, replicate=0):
    seasons, _ = DIRECTIONS[direction]
    frame = table[table["season"].isin(seasons)].reset_index(drop=True)
    if n_train:
        keep = nested_subset(frame["shot_id"].to_numpy(), n_train, replicate)
        frame = frame[keep].reset_index(drop=True)
    return frame


def run_config(model, direction, arm, blocks, grid_spec, n_train=None, replicate=0):
    return {
        "model": model,
        "arm": arm,
        "direction": direction,
        "blocks": list(blocks),
        "grid": grid_spec,
        "n_train": n_train,
        "replicate": replicate,
    }


def selection_path(cfg):
    return TUNING / cfg["model"] / f"{store.stem(cfg, 'selection')}.json"


def model_path(cfg):
    return MODELS / cfg["model"] / f"{store.stem(cfg, 'model')}.joblib"


def load_selection(
    model,
    direction,
    arm="primary",
    blocks=PRIMARY_BLOCKS,
    n_train=None,
    replicate=0,
    grid_spec=None,
):
    grid_spec = grid_spec or load_grid(model)
    cfg = run_config(model, direction, arm, blocks, grid_spec, n_train, replicate)
    return json.loads(selection_path(cfg).read_text())


def run_trained(
    model,
    direction,
    arm="primary",
    blocks=PRIMARY_BLOCKS,
    grid_spec=None,
    n_train=None,
    replicate=0,
    n_jobs=1,
    keep_model=True,
    verbose=2,
):
    grid_spec = grid_spec or load_grid(model)
    cfg = run_config(model, direction, arm, blocks, grid_spec, n_train, replicate)
    if store.exists(cfg, "eval") and selection_path(cfg).exists():
        return json.loads(selection_path(cfg).read_text())
    started = time.time()
    table = features.load_features()
    train = training_frame(table, direction, n_train, replicate)
    X = features.model_input(train)
    y = features.target(train)
    search = tuning.run_grid(MAKERS[model](blocks), grid_spec, X, y, n_jobs=n_jobs, verbose=verbose)
    results = tuning.results_table(search)
    best = search.best_estimator_
    tuned = MAKERS[model](blocks).set_params(**search.best_params_)
    oof, fold = tuning.out_of_fold(tuned, X, y)
    platt = Platt().fit(oof, y)
    inner = pd.DataFrame(
        {"shot_id": train["shot_id"], "fold": fold, "p_raw": oof, "p_cal": platt.apply(oof)}
    )
    store.write(inner, cfg, "inner_oof")
    held_out = features.season_frame(table, DIRECTIONS[direction][1])
    raw = best.predict_proba(features.model_input(held_out))[:, 1]
    seconds = time.time() - started
    scored = pd.DataFrame({"shot_id": held_out["shot_id"], "p_raw": raw, "p_cal": platt.apply(raw)})
    store.write(scored, cfg, "eval", extra={"seconds": seconds})
    by_mean = results.loc[results["best_by_mean"]].iloc[0]
    selected = results.loc[results["selected"]].iloc[0]
    best_index = int(np.argmax(search.cv_results_["mean_test_log_loss"]))
    selection = {
        "run_config": cfg,
        "config_hash": store.config_hash(cfg),
        "best_params": search.best_params_,
        "best_by_mean_params": search.cv_results_["params"][best_index],
        "cv_log_loss_selected": -float(selected["mean_test_log_loss"]),
        "cv_log_loss_best": -float(by_mean["mean_test_log_loss"]),
        "cv_auc_selected": float(selected["mean_test_auc"]),
        "configurations": int(len(results)),
        "train_rows": int(len(train)),
        "base_rate": float(y.mean()),
        "platt": platt.params(),
        "seconds": seconds,
    }
    TUNING.joinpath(model).mkdir(parents=True, exist_ok=True)
    results.to_parquet(selection_path(cfg).with_suffix(".parquet"), index=False)
    selection_path(cfg).write_text(json.dumps(selection, indent=2, default=str))
    if keep_model:
        MODELS.joinpath(model).mkdir(parents=True, exist_ok=True)
        joblib.dump(best, model_path(cfg))
    return selection


def load_model(model, direction, arm="primary", blocks=PRIMARY_BLOCKS):
    cfg = run_config(model, direction, arm, blocks, load_grid(model))
    return joblib.load(model_path(cfg))


def ablation_arms():
    arms = {}
    for i in range(len(LADDER) - 1):
        arms[f"ladder:{i + 1}_{LADDER[i]}"] = LADDER[: i + 1]
    arms["plus_team_month"] = PRIMARY_BLOCKS + ["team_month"]
    return arms


def ablation_grid(model, direction, blocks):
    """Structure hyperparameters fixed at the primary selection; only smoothing m is re-tuned."""
    primary = load_grid(model)
    chosen = load_selection(model, direction)["best_params"]
    if RATE_BLOCKS & set(blocks):
        smoothing = primary["param_grid"]["rates__m"]
    else:
        smoothing = [chosen["rates__m"]]
    grid = {name: [value] for name, value in chosen.items()}
    grid["rates__m"] = smoothing
    return {"param_grid": grid, "simplicity_order": primary["simplicity_order"]}


def run_suite(suite, model, n_jobs=1):
    if suite == "primary":
        for direction in CROSS_SEASON:
            run_trained(model, direction, n_jobs=n_jobs)
    elif suite == "final":
        run_trained(model, "S1+S2_to_T", n_jobs=n_jobs)
    elif suite == "ablation":
        for direction in CROSS_SEASON:
            for arm, blocks in ablation_arms().items():
                run_trained(
                    model,
                    direction,
                    arm=f"ablation:{arm}",
                    blocks=blocks,
                    grid_spec=ablation_grid(model, direction, blocks),
                    n_jobs=n_jobs,
                    keep_model=False,
                )
            run_trained(
                model,
                direction,
                arm="ablation:ladder:1_location_retuned",
                blocks=LADDER[:1],
                n_jobs=n_jobs,
                keep_model=False,
            )
    elif suite == "learning_curve":
        curve = config()["learning_curve"]
        for direction in CROSS_SEASON:
            for size in curve["sizes"]:
                for replicate in range(curve["replicates"][size]):
                    run_trained(
                        model,
                        direction,
                        arm="learning_curve",
                        n_train=size,
                        replicate=replicate,
                        n_jobs=n_jobs,
                        keep_model=False,
                        verbose=0,
                    )
    else:
        raise ValueError(suite)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=["primary", "final", "ablation", "learning_curve"])
    parser.add_argument("model", choices=sorted(MAKERS))
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    run_suite(args.suite, args.model, args.n_jobs)


if __name__ == "__main__":
    main()

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shotlab import features, metrics, plots, runs, store
from shotlab.data import TARGET
from shotlab.features import BLOCKS
from shotlab.runs import TUNING

KINDS = {"layup": ("layup", 0), "mid-range jumper": ("jumper", 0), "three-point jumper": ("jumper", 1)}
NEUTRAL_PLAYER = "average player"
CLOSEOUT_COLUMNS = BLOCKS["closeout"]
HEADLINE = ["log_loss", "skill", "brier", "auc", "ece", "cal_slope", "cal_intercept"]


def selection_files(model, arm="primary"):
    """Selection records of one model and arm, keyed by direction."""
    found = {}
    for path in sorted((TUNING / model).glob("*selection*.json")):
        selection = json.loads(path.read_text())
        run = selection["run_config"]
        if run["arm"] == arm and not run.get("n_train"):
            found[run["direction"]] = (selection, pd.read_parquet(path.with_suffix(".parquet")))
    return found


def selection_summary(model):
    rows = {}
    for direction, (selection, _) in selection_files(model).items():
        rows[plots.DIRECTION_LABELS[direction]] = {
            **{f"selected {k.split('__')[-1]}": v for k, v in selection["best_params"].items()},
            "inner CV log loss, selected": selection["cv_log_loss_selected"],
            "inner CV log loss, best": selection["cv_log_loss_best"],
            "inner CV AUC, selected": selection["cv_auc_selected"],
            "Platt slope": selection["platt"]["slope"], "Platt intercept": selection["platt"]["intercept"],
            "configurations": selection["configurations"], "training shots": selection["train_rows"],
            "minutes": selection.get("seconds", selection.get("tuning_seconds", np.nan)) / 60,
        }
    return pd.DataFrame(rows)


def one_standard_error(cv, score="mean_test_log_loss", spread="std_test_log_loss", folds=5):
    """Best mean inner log loss and the threshold that still counts as tied with it."""
    best = cv.loc[cv[score].idxmax()]
    return -best[score], -best[score] + best[spread] / np.sqrt(folds)


def inner_oof(model, direction, table, arm="primary"):
    """Month-blocked out-of-fold predictions on the training season, joined to outcomes."""
    frame = store.read(model=model, arm=arm, direction=direction, role="inner_oof", n_train=0)
    return frame.merge(table[["shot_id", TARGET, "shot_family"]], on="shot_id", validate="one_to_one")


def within_season(model, table):
    rows = {}
    for direction in selection_files(model):
        oof = inner_oof(model, direction, table)
        y = oof[TARGET].to_numpy().astype(int)
        for name, column in [("raw", "p_raw"), ("Platt", "p_cal")]:
            rows[(plots.DIRECTION_LABELS[direction].split(" → ")[0].replace("train ", "within "), name)] = {
                k: v for k, v in metrics.summarize(y, oof[column].to_numpy(), y.mean()).items() if k in HEADLINE}
    return pd.DataFrame(rows).T[HEADLINE]


def fold_scores(model, table):
    rows = []
    for direction in selection_files(model):
        oof = inner_oof(model, direction, table)
        loss = metrics.per_row_log_loss(oof[TARGET].to_numpy().astype(int), oof["p_raw"].to_numpy())
        rows.append(pd.Series(loss).groupby(oof["fold"].to_numpy()).mean().rename(plots.DIRECTION_LABELS[direction]))
    return pd.concat(rows, axis=1).rename_axis("month block")


def calibration_figure(model, table):
    directions = list(selection_files(model))
    fig, axes = plt.subplots(1, len(directions), figsize=(4.6 * len(directions), 3.8), sharey=True, squeeze=False)
    for ax, direction in zip(axes[0], directions):
        oof = inner_oof(model, direction, table)
        plots.reliability(ax, oof[TARGET].to_numpy().astype(int), oof["p_raw"].to_numpy(), color=plots.COLORS[model])
        ax.set_title(f"{plots.LABELS[model]}, out-of-fold, {direction.split('_')[0]}")
    for ax in axes[0][1:]:
        ax.set_ylabel("")
    fig.tight_layout()
    return fig


def defender_profile(train, shottype, three, distances):
    """The median shot of one kind with one defender slid across distances, taken by an unseen player (edge 0).

    The contested flag follows what real shots of that kind show at each defender distance.
    """
    kind = features.model_input(train[(train["shottype"] == shottype) & (train["three"] == three)])
    medians = kind.select_dtypes("number").median()
    reference = kind.iloc[[0]].astype({column: float for column in medians.index})
    reference[medians.index] = medians.to_numpy()
    reference[["shooter_id", "closestdef_id"]] = NEUTRAL_PLAYER
    profile = pd.concat([reference] * len(distances), ignore_index=True)
    contested = (kind["num_contesters"] > 0).groupby(kind["closestdefdist"].round()).mean().ge(0.5).astype(float)
    profile["closestdefdist"] = distances
    profile["num_contesters"] = contested.reindex(np.round(distances)).ffill().bfill().to_numpy()
    profile["distcont2"] = np.nan
    profile[CLOSEOUT_COLUMNS] = 0.0
    return profile


def defender_profile_figure(model, table):
    """What-if curves: slide the closest defender for the median shot of each kind, one line per training season."""
    distances = np.linspace(0, 12, 49)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for (label, (shottype, three)), color in zip(KINDS.items(), plots.KIND_COLORS):
        for direction, dash in zip(selection_files(model), ["-", "--"]):
            profile = defender_profile(runs.training_frame(table, direction), shottype, three, distances)
            ax.plot(distances, runs.load_model(model, direction).predict_proba(profile)[:, 1], color=color, linestyle=dash,
                    label=label if dash == "-" else None)
    ax.set_xlabel("closest defender distance at release (ft)")
    ax.set_ylabel("predicted make probability")
    ax.set_title(f"{plots.LABELS[model]}: the value of space, by shot kind")
    ax.legend()
    fig.tight_layout()
    return fig

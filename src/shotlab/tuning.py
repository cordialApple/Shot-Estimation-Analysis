import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_predict

from shotlab.splits import month_folds

SCORING = {"log_loss": "neg_log_loss", "brier": "neg_brier_score", "auc": "roc_auc"}
PRIMARY = "log_loss"


def one_standard_error_index(results, simplicity_order):
    """Most regularized configuration within one standard error of the best log loss."""
    mean = np.asarray(results[f"mean_test_{PRIMARY}"])
    suffix = f"_test_{PRIMARY}"
    split_keys = [key for key in results if key.startswith("split") and key.endswith(suffix)]
    folds = len(split_keys)
    standard_error = np.asarray(results[f"std_test_{PRIMARY}"]) / np.sqrt(folds)
    best = int(np.argmax(mean))
    eligible = np.flatnonzero(mean >= mean[best] - standard_error[best])
    table = pd.DataFrame({"index": eligible})
    sort_columns, ascending = [], []
    for parameter, direction in simplicity_order:
        table[parameter] = [results["params"][i][parameter] for i in eligible]
        sort_columns.append(parameter)
        ascending.append(direction == "asc")
    table["score"] = mean[eligible]
    sort_columns.append("score")
    ascending.append(False)
    ordered = table.sort_values(sort_columns, ascending=ascending, kind="stable")
    return int(ordered["index"].iloc[0])


def run_grid(pipeline, grid_spec, X, y, n_jobs=1, verbose=2):
    folds = month_folds(X["month_block"])
    order = grid_spec["simplicity_order"]
    search = GridSearchCV(
        pipeline,
        grid_spec["param_grid"],
        scoring=SCORING,
        cv=folds,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=lambda results: one_standard_error_index(results, order),
        error_score="raise",
    )
    return search.fit(X, y)


def results_table(search):
    results = pd.DataFrame(search.cv_results_)
    prefixes = ("param_", "mean_test_", "std_test_", "mean_fit_time")
    keep = [c for c in results if c.startswith(prefixes)]
    table = results[keep].copy()
    table["selected"] = np.arange(len(table)) == search.best_index_
    score = table[f"mean_test_{PRIMARY}"]
    table["best_by_mean"] = score == score.max()
    return table


def out_of_fold(pipeline, X, y):
    """Month-blocked out-of-fold probabilities and the fold of each row."""
    folds = month_folds(X["month_block"])
    probabilities = cross_val_predict(pipeline, X, y, cv=folds, method="predict_proba")[:, 1]
    return probabilities, X["month_block"].to_numpy()

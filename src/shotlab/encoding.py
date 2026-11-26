import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from shotlab.features import KEYS

ENTITIES = {
    "shooter_edge": "shooter_id",
    "defender_edge": "closestdef_id",
    "off_team_edge": "off_team_id",
    "def_team_edge": "def_team_id",
}
FAMILY = "shot_family"
FOLD = "month_block"


class RateEncoder(BaseEstimator, TransformerMixin):
    """Laplace-smoothed make rate per entity and shot family, minus the family rate.

    edge = (makes - attempts * family_rate) / (attempts + m). Training rows are
    cross-fitted over month blocks in fit_transform; transform uses full-fit tables.
    """

    def __init__(self, m=300.0):
        self.m = m

    def fit(self, X, y):
        frame = pd.DataFrame({"family": X[FAMILY].astype(str).to_numpy(), "y": np.asarray(y, dtype=float)})
        self.prior_ = frame.groupby("family")["y"].mean()
        self.tables_ = {}
        for name, key in ENTITIES.items():
            frame["key"] = X[key].to_numpy()
            stats = frame.groupby(["key", "family"])["y"].agg(["sum", "size"])
            prior = self.prior_.reindex(stats.index.get_level_values("family")).to_numpy()
            self.tables_[name] = (stats["sum"] - stats["size"] * prior) / (stats["size"] + self.m)
        return self

    def transform(self, X):
        family = X[FAMILY].astype(str).to_numpy()
        out = X.drop(columns=KEYS)
        for name, key in ENTITIES.items():
            index = pd.MultiIndex.from_arrays([X[key].to_numpy(), family])
            out[name] = self.tables_[name].reindex(index).fillna(0.0).to_numpy()
        return out

    def fit_transform(self, X, y):
        self.fit(X, y)
        frame = pd.DataFrame({
            "family": X[FAMILY].astype(str).to_numpy(), "fold": X[FOLD].to_numpy(), "y": np.asarray(y, dtype=float)})

        def out_of_fold(keys):
            total = frame.groupby(keys)["y"]
            held = frame.groupby(keys + ["fold"])["y"]
            return (total.transform("sum") - held.transform("sum")).to_numpy(), (
                total.transform("size") - held.transform("size")).to_numpy()

        family_makes, family_attempts = out_of_fold(["family"])
        prior = np.divide(family_makes, family_attempts, out=np.full(len(frame), frame["y"].mean()), where=family_attempts > 0)
        out = X.drop(columns=KEYS)
        for name, key in ENTITIES.items():
            frame["key"] = X[key].to_numpy()
            makes, attempts = out_of_fold(["key", "family"])
            out[name] = (makes - attempts * prior) / (attempts + self.m)
        return out


class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[list(self.columns)]

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from shotlab.data import config
from shotlab.encoding import RateEncoder, SelectColumns
from shotlab.features import (
    CATEGORICAL,
    GAMESTATES,
    MAX_CONTESTERS,
    PRIMARY_BLOCKS,
    SHOTTYPES,
    columns_for,
)

SPLINE_COLUMNS = [
    "distance",
    "shot_angle",
    "shotclock",
    "closestdefdist",
    "distcont2",
    "shooterspeed",
    "dribblesbefore",
    "closeout_1",
    "closeout_4",
]
ONE_HOT = {
    "shottype": SHOTTYPES,
    "gamestate": GAMESTATES,
    "num_contesters": [float(i) for i in range(MAX_CONTESTERS + 1)],
}
NO_SECOND_CONTESTER_FT = 15.0
CLIP_QUANTILES = (0.005, 0.995)
SHOTTYPE_INTERACTIONS = ["closestdefdist", "num_contesters", "distance"]
PAIR_INTERACTIONS = [
    ("three", "closestdefdist"),
    ("three", "catch_and_shoot"),
    ("three", "closeout_1"),
    ("catch_and_shoot", "closestdefdist"),
]


class LinearDesign(BaseEstimator, TransformerMixin):
    """Design matrix for logistic regression: splines, one-hot levels, optional interactions.

    Works on any subset of feature columns so ablation arms reuse it.
    """

    def __init__(self, n_knots=5, interactions=True):
        self.n_knots = n_knots
        self.interactions = interactions

    def _numeric(self, X):
        categorical = [c for c in CATEGORICAL if c in X]
        frame = X.drop(columns=categorical).astype(float)
        if "distcont2" in frame:
            frame["distcont2"] = frame["distcont2"].fillna(NO_SECOND_CONTESTER_FT)
        return frame

    def fit(self, X, y=None):
        numeric = self._numeric(X)
        self.spline_columns_ = [c for c in SPLINE_COLUMNS if c in numeric]
        self.linear_columns_ = [
            c for c in numeric if c not in self.spline_columns_ and c not in ONE_HOT
        ]
        self.one_hot_columns_ = [c for c in ONE_HOT if c in X]
        self.bounds_ = numeric[self.spline_columns_].quantile(list(CLIP_QUANTILES))
        self.center_ = numeric.mean()
        self.scale_ = numeric.std().replace(0, 1.0)
        self.splines_ = {}
        for column in self.spline_columns_:
            spline = SplineTransformer(
                n_knots=self.n_knots,
                degree=3,
                knots="uniform",
                extrapolation="constant",
                include_bias=False,
            )
            self.splines_[column] = spline.fit(self._clipped(numeric, column))
        self.feature_names_ = list(self._build(X).columns)
        return self

    def _clipped(self, numeric, column):
        low, high = self.bounds_[column]
        return numeric[[column]].clip(low, high).to_numpy()

    def _build(self, X):
        numeric = self._numeric(X)
        parts = {}
        for column in self.spline_columns_:
            basis = self.splines_[column].transform(self._clipped(numeric, column))
            for i in range(basis.shape[1]):
                parts[f"{column}_s{i}"] = basis[:, i]
        for column in self.linear_columns_:
            parts[column] = numeric[column].to_numpy()
        dummies = {}
        for column in self.one_hot_columns_:
            values = X[column].to_numpy()
            for level in ONE_HOT[column][1:]:
                label = int(level) if isinstance(level, float) else level
                dummies[f"{column}={label}"] = (values == level).astype(float)
        parts.update(dummies)
        if self.interactions:
            scaled = (numeric - self.center_) / self.scale_
            if "shottype" in X:
                for name, dummy in dummies.items():
                    if not name.startswith("shottype="):
                        continue
                    for column in SHOTTYPE_INTERACTIONS:
                        if column in scaled:
                            parts[f"{name}:{column}"] = dummy * scaled[column].to_numpy()
            for left, right in PAIR_INTERACTIONS:
                if left in numeric and right in scaled:
                    parts[f"{left}:{right}"] = numeric[left].to_numpy() * scaled[right].to_numpy()
        return pd.DataFrame(parts, index=X.index)

    def transform(self, X):
        return self._build(X).to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_)


def make_logreg(blocks=PRIMARY_BLOCKS):
    return Pipeline(
        [
            ("rates", RateEncoder()),
            ("select", SelectColumns(columns_for(blocks))),
            ("design", LinearDesign()),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )


def make_hgb(blocks=PRIMARY_BLOCKS):
    model = HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        early_stopping=False,
        random_state=config()["seed"],
    )
    return Pipeline(
        [
            ("rates", RateEncoder()),
            ("select", SelectColumns(columns_for(blocks))),
            ("model", model),
        ]
    )


MAKERS = {"logreg": make_logreg, "hgb": make_hgb}

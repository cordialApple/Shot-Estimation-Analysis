import numpy as np
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from shotlab.metrics import clip


class Platt:
    """Two-parameter logistic map on the logit of the raw probability."""

    def fit(self, p, y):
        x = logit(clip(p)).reshape(-1, 1)
        model = LogisticRegression(C=1e6, max_iter=1000).fit(x, y)
        self.slope_ = float(model.coef_[0, 0])
        self.intercept_ = float(model.intercept_[0])
        return self

    def apply(self, p):
        return expit(self.slope_ * logit(clip(p)) + self.intercept_)

    def params(self):
        return {"slope": self.slope_, "intercept": self.intercept_}


class Isotonic:
    def fit(self, p, y):
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.model_ = model.fit(np.asarray(p), y)
        return self

    def apply(self, p):
        return self.model_.predict(np.asarray(p))

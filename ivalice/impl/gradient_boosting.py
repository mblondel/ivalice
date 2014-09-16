"""Functional gradient boosting"""

# Author: Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

import numpy as np
from scipy import stats

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state


# Taken from https://github.com/nudomarinero/wquantiles (MIT license)
def _weighted_quantile(data, weights, quantile):
    # Sort the data
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
    # Compute the auxiliary arrays
    Sn = np.cumsum(sorted_weights)
    # TODO: Check that the weights do not sum zero
    Pn = (Sn-0.5*sorted_weights)/np.sum(sorted_weights)
    # Get the value of the weighted median
    return np.interp(quantile, Pn, sorted_data)


def _weighted_median(data, weights):
    return _weighted_quantile(data, weights, 0.5)


class _QuantileEstimator(BaseEstimator):
    """An estimator predicting the alpha-quantile of the training targets."""
    def __init__(self, alpha=0.9):
        if not 0 < alpha < 1.0:
            raise ValueError("`alpha` must be in (0, 1.0) but was %r" % alpha)
        self.alpha = alpha

    def fit(self, X, y):
        self.quantile = stats.scoreatpercentile(y, self.alpha * 100.0)
        return self

    def predict(self, X):
        y = np.empty(X.shape[0], dtype=np.float64)
        y.fill(self.quantile)
        return y


class _MeanEstimator(BaseEstimator):
    """An estimator predicting the mean of the training targets."""
    def fit(self, X, y):
        self.mean = np.mean(y)
        return self

    def predict(self, X):
        y = np.empty(X.shape[0], dtype=np.float64)
        y.fill(self.mean)
        return y


class _SquareLoss(object):

    def init_estimator(self):
        return _MeanEstimator()

    def negative_gradient(self, y, y_pred):
        return y - y_pred

    def step_size(self, y, y_pred, h_pred):
        return 1.0


class _AbsoluteLoss(object):

    def init_estimator(self):
        return _QuantileEstimator(alpha=0.5)

    def negative_gradient(self, y, y_pred):
        return np.sign(y - y_pred)

    def step_size(self, y, y_pred, h_pred):
        cond = h_pred != 0
        diff = y - y_pred
        diff[cond] /= h_pred[cond]
        diff[~cond] = 0
        return _weighted_median(diff, np.abs(h_pred))


class _SquaredHingeLoss(object):

    def __init__(self, max_steps=1):
        self.max_steps = max_steps

    def init_estimator(self):
        return _MeanEstimator()

    def negative_gradient(self, y, y_pred):
        return 2 * np.maximum(1 - y * y_pred, 0) * y

    def step_size(self, y, y_pred, h_pred):
        rho = 0

        y_h_pred = y * h_pred
        h_pred_sq = h_pred ** 2

        for it in xrange(self.max_steps):
            error = 1 - y * (y_pred + rho * h_pred)
            Lp = -np.sum(np.maximum(error, 0) * y_h_pred)
            Lpp = np.sum((error > 0) * h_pred_sq)

            if Lpp == 0:
                break

            rho -= Lp / Lpp

        return rho


class _BaseGB(BaseEstimator):

    def _fit(self, X, y, loss, rng):
        if self.subsample != 1.0:
            n = int(X.shape[0] * self.subsample)
            ind = rng.permutation(X.shape[0])[:n]
            X = X[ind]
            y = y[ind]

        y_pred = self.predict(X)
        negative_gradient = loss.negative_gradient(y, y_pred)

        est = clone(self.estimator)
        est.fit(X, negative_gradient)
        self.estimators_.append(est)

        h_pred = est.predict(X)

        step_size = getattr(self, "step_size", "constant")

        if step_size == "line_search":
            step_size = loss.step_size(y, y_pred, h_pred)
        elif step_size == "constant":
            step_size = 1.0
        else:
            raise ValueError("Unknown step size.")

        i = len(self.estimators_) - 1
        self.estimator_weights_[i] *= step_size

    def fit(self, X, y):
        rng = check_random_state(self.random_state)
        loss = self._get_loss()

        self.estimator_weights_ = np.ones(self.n_estimators, dtype=np.float64)
        self.estimator_weights_[1:] *= self.learning_rate

        estimator = loss.init_estimator()
        self.estimators_ = [estimator.fit(X, y)]

        for i in xrange(1, self.n_estimators):
            self._fit(X, y, loss, rng)

            if self.callback is not None:
                self.callback(self)

        return self

    def predict(self, X):
        pred = np.zeros(X.shape[0], dtype=np.float64)
        for i, est in enumerate(self.estimators_):
            pred += self.estimator_weights_[i] * est.predict(X)
        return pred


class GBClassifier(_BaseGB, ClassifierMixin):

    def __init__(self, estimator, n_estimators=100,
                 step_size="line_search", learning_rate=0.1,
                 loss="squared_hinge", subsample=1.0,
                 callback=None, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.step_size = step_size
        self.learning_rate = learning_rate
        self.loss = loss
        self.subsample = subsample
        self.callback = callback
        self.random_state = random_state

    def _get_loss(self):
        losses = dict(squared_hinge=_SquaredHingeLoss())
        return losses[self.loss]

    def fit(self, X, y):
        self._lb = LabelBinarizer(neg_label=-1)
        y = self._lb.fit_transform(y)[:, 0]
        return super(GBClassifier, self).fit(X, y)

    def predict(self, X):
        pred = super(GBClassifier, self).predict(X).reshape(-1, 1)
        return self._lb.inverse_transform(pred)


class GBRegressor(_BaseGB, RegressorMixin):

    def __init__(self, estimator, n_estimators=100,
                 step_size="line_search", learning_rate=0.1,
                 loss="squared", subsample=1.0,
                 callback=None, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.step_size = step_size
        self.learning_rate = learning_rate
        self.loss = loss
        self.subsample = subsample
        self.callback = callback
        self.random_state = random_state

    def _get_loss(self):
        losses = dict(squared=_SquareLoss(),
                      absolute=_AbsoluteLoss())
        return losses[self.loss]

import numpy as np
from scipy import stats

from sklearn.base import BaseEstimator, RegressorMixin, clone
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


class QuantileEstimator(BaseEstimator):
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


class MeanEstimator(BaseEstimator):
    """An estimator predicting the mean of the training targets."""
    def fit(self, X, y):
        self.mean = np.mean(y)
        return self

    def predict(self, X):
        y = np.empty(X.shape[0], dtype=np.float64)
        y.fill(self.mean)
        return y


class SquareLoss(object):

    def init_estimator(self):
        return MeanEstimator()

    def negative_gradient(self, y, y_pred):
        return y - y_pred

    def step_size(self, y, y_pred, h_pred):
        return 1.0


class AbsoluteLoss(object):

    def init_estimator(self):
        return QuantileEstimator(alpha=0.5)

    def negative_gradient(self, y, y_pred):
        return np.sign(y - y_pred)

    def step_size(self, y, y_pred, h_pred):
        cond = h_pred != 0
        diff = y - y_pred
        diff[cond] /= h_pred[cond]
        diff[~cond] = 0
        return _weighted_median(diff, np.abs(h_pred))


class GBRegressor(BaseEstimator):

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
        losses = dict(squared=SquareLoss(),
                      absolute=AbsoluteLoss())
        return losses[self.loss]

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

        if self.step_size == "line_search":
            step_size = loss.step_size(y, y_pred, h_pred)
        elif self.step_size == "constant":
            step_size = 1.0
        else:
            raise ValueError("Unknown step size.")

        i = len(self.estimators_) - 1
        self.estimator_weights_[i] *= step_size
        #if step_size == 0:
            #self.estimator_weights_[i:] = 0
        #else:
            #self.estimator_weights_[i] *= step_size

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



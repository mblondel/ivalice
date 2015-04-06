"""LambdaMART"""

# Author: Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

import numpy as np
import numba

from .gradient_boosting import _BaseGB, _MeanEstimator


@numba.njit("void(f8[:], f8[:], f8[:], f8, i4, f8[:])")
def _negative_gradient(y, y_pred, c, idcg, max_rank, g):
    n_samples = y.shape[0]

    for i in xrange(max_rank):
        for j in xrange(i + 1, n_samples):
            S = np.sign(y[i] - y[j])

            if S == 0:
                continue

            score_diff = y_pred[i] - y_pred[j]

            diff = y[j] * (c[i] - c[j]) + y[i] * (c[j] - c[i])
            ndcg_diff = abs(diff / idcg)

            if ndcg_diff == 0:
                continue

            rho = 1.0 / (1.0 + np.exp(S * score_diff))
            #rho = expit(-S * score_diff)
            g[i] += S * ndcg_diff * rho
            g[j] -= S * ndcg_diff * rho


def _dcg_score(y_true, y_score, max_rank=10, gains="exponential"):
    order = np.lexsort((y_true, -y_score))

    if max_rank is not None:
        order = order[:max_rank]

    y_true = np.take(y_true, order)

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def _ndcg_score(y_true, y_score, max_rank=10, gains="exponential"):
    best = _dcg_score(y_true, y_true, max_rank, gains)
    actual = _dcg_score(y_true, y_score, max_rank, gains)
    return actual / best


class _NDCGLoss(object):

    def __init__(self, max_rank=10):
        self.max_rank = max_rank

    def init_estimator(self):
        return _MeanEstimator()

    def negative_gradient(self, y, y_pred):
        n_samples = y.shape[0]

        max_rank = self.max_rank if self.max_rank is not None else n_samples

        #order = np.argsort(y_pred)[::-1]
        order = np.lexsort((y, -y_pred))
        y = np.take(y, order)
        y_pred = np.take(y_pred, order)

        ind = np.arange(n_samples)
        c = 1. / np.log2(ind + 2)  # discount factors
        c[max_rank:] = 0

        g = np.zeros(n_samples, dtype=np.float64)

        y_sorted = np.sort(y)[::-1]
        idcg = np.sum(y_sorted * c)

        _negative_gradient(y, y_pred, c, idcg, max_rank, g)

        if np.any(np.isnan(g)):
            print "g contains NaNs"

        inv_ix = np.empty_like(order)
        inv_ix[order] = np.arange(len(order))
        g = g[inv_ix]

        return g


class LambdaMART(_BaseGB):

    def __init__(self, estimator, n_estimators=100, learning_rate=1.0,
                 loss="ndcg", max_rank=10, gains="exponential",
                 subsample=1.0, callback=None, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.max_rank = max_rank
        self.gains = gains
        self.subsample = subsample
        self.callback = callback
        self.random_state = random_state

    def _get_loss(self):
        losses = dict(ndcg=_NDCGLoss(max_rank=self.max_rank))
        return losses[self.loss]

    def fit(self, X, y):
        if self.gains == "exponential":
            y = 2 ** y - 1

        return super(LambdaMART, self).fit(X, y)

    def score(self, X, y):
        y_pred = self.predict(X)
        return _ndcg_score(y, y_pred, max_rank=self.max_rank, gains=self.gains)

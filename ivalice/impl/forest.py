"""Random Forests"""

# Author: Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

import numpy as np

from sklearn.base import BaseEstimator, clone
from sklearn.base import RegressorMixin
from sklearn.utils import check_random_state

from .tree import TreeRegressor


MAX_INT = np.iinfo(np.int32).max


def _fit_random_tree(tree, X, y, sample_weight, bootstrap, rng):
    if bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            sample_weight = sample_weight.copy()

        indices = rng.randint(0, n_samples, n_samples)
        sample_counts = np.bincount(indices, minlength=n_samples)
        sample_weight *= sample_counts

        tree.fit(X, y, sample_weight=sample_weight)
        tree.indices_ = sample_counts > 0.

    else:
        tree.fit(X, y, sample_weight=sample_weight)


class _BaseRF(BaseEstimator):

    def _fit(self, X, y, sample_weight, tree):
        rng = check_random_state(self.random_state)
        self.estimators_ = []


        for k in xrange(self.n_estimators):
            tree = clone(tree)
            tree.set_params(random_state=rng.randint(MAX_INT))
            _fit_random_tree(tree, X, y, sample_weight, self.bootstrap, rng)
            self.estimators_.append(tree)

        return self


class RFRegressor(_BaseRF, RegressorMixin):

    def __init__(self, n_estimators=10, max_features=None, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, bootstrap=True,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        tree = TreeRegressor(max_features=self.max_features,
                             max_depth=self.max_depth,
                             min_samples_split=self.min_samples_split,
                             min_samples_leaf=self.min_samples_leaf)
        return self._fit(X, y, sample_weight, tree)

    def predict(self, X):
        pred = np.array([tree.predict(X) for tree in self.estimators_])
        return np.mean(pred, axis=0)

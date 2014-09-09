"""McRank"""

# Author: Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

import numpy as np

from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder


DEFAULT_CLF = GradientBoostingClassifier(loss="deviance")


class _BaseMcRank(BaseEstimator):

    def __init__(self, estimator=DEFAULT_CLF):
        self.estimator = estimator

    @property
    def classes_(self):
        return self._label_encoder.classes_

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(np.abs(y - y_pred))

    def predict(self, X):
        """Predict expected target value for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples]
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = self.predict_proba(X)
        classes = np.repeat(self.classes_, n_samples)
        classes = classes.reshape(n_classes, n_samples).T
        # pred[i] = \sum_m P(y_i = m) * m
        return np.average(classes, axis=1, weights=proba)


class McRank(_BaseMcRank):

    def set_estimator_params(self, **params):
        self.estimator_.set_params(**params)

    def fit(self, X, y):
        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(y)

        if not hasattr(self, "estimator_") or \
           not getattr(self.estimator, "warm_start", False):
            self.estimator_ = clone(self.estimator)

        self.estimator_.fit(X, y)

        return self

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        return self.estimator_.predict_proba(X)


class OrdinalMcRank(_BaseMcRank):

    def _fit(self, X, y, m, est):
        cond = y <= m
        y_bin = y.copy()
        y_bin[cond] = 0
        y_bin[~cond] = 1
        est.fit(X, y_bin)

    def set_estimator_params(self, **params):
        for est in self.estimators_:
            est.set_params(**params)

    def fit(self, X, y):
        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(y)

        n_classifiers = len(self.classes_) - 1

        if not hasattr(self, "estimators_") or \
           not getattr(self.estimator, "warm_start", False):
            self.estimators_ = [clone(self.estimator)
                                for m in xrange(n_classifiers)]

        for m in xrange(n_classifiers):
            self._fit(X, y, m, self.estimators_[m])

        return self

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # 2d array of shape (n_classes-1) x n_samples containing
        # cumulative probabilities P(y_i <= k)
        P = np.array([e.predict_proba(X)[:, 0] for e in self.estimators_])

        # 2d array of shape n_classes x n_samples containing
        # cumulative probabilities P(y_i <= k)
        P = np.vstack((P, np.ones(n_samples)))

        proba = np.zeros((n_samples, n_classes), dtype=np.float64)

        proba[:, 0] = P[0] # P(y = 0) = P(y <= 0)

        for m in xrange(1, n_classes):
            proba[:, m] = P[m] - P[m - 1] # P(y = m) = P(y <= m) - P(y <= m - 1)

        return proba

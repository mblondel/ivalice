"""AdaBoost"""

# Author: Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score


class AdaBoostClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, estimator, n_estimators=10):
        self.estimator = estimator
        self.n_estimators = n_estimators

    def fit(self, X, y):
        n_samples = X.shape[0]

        weights = np.ones(n_samples, dtype=np.float64) / n_samples

        self._lb = LabelBinarizer(neg_label=-1)
        y = self._lb.fit_transform(y).ravel()

        self.estimators_ = np.zeros(self.n_estimators, dtype=np.object)
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)

        y_pred_ = np.zeros(n_samples, dtype=np.float64)

        for it in xrange(self.n_estimators):
            est = clone(self.estimator)
            est = est.fit(X, y, sample_weight=weights)

            y_pred = est.predict(X)
            err = 1 - accuracy_score(y, y_pred, sample_weight=weights)

            if err == 0:
                self.estimator_weights_[it] = 1
                self.estimators_[it] = est
                break

            alpha = 0.5 * np.log((1 - err) / err)

            #weights *= np.exp(- alpha * y * y_pred)
            #weights /= weights.sum()

            y_pred_ += alpha * y_pred
            weights = np.exp(-y * y_pred_)
            #weights = 1.0 / (1 + np.exp(y * y_pred_))  # logit boost
            weights /= weights.sum()

            self.estimator_weights_[it] = alpha
            self.estimators_[it] = est


        return self

    def predict(self, X):
        y_pred = np.zeros(X.shape[0], dtype=np.float64)
        for it in xrange(self.n_estimators):
            if self.estimator_weights_[it] != 0:
                pred = self.estimators_[it].predict(X)
                y_pred += self.estimator_weights_[it] * pred
        y_pred = np.sign(y_pred)
        return self._lb.inverse_transform(y_pred.reshape(-1, 1))

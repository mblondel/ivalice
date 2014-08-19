import numpy as np

from sklearn.tree import DecisionTreeRegressor as skDecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.utils.testing import assert_almost_equal

from ivalice.impl.tree import DecisionTreeRegressor


def _make_datasets(n_times):
    for n in xrange(n_times):
        yield make_regression(n_samples=20, n_features=10, random_state=n)


def test_fully_developed():
    sk = 0
    iv = 0

    for X, y in _make_datasets(10):
        reg = skDecisionTreeRegressor(max_depth=None)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        sk += np.mean((y - y_pred) ** 2)

        reg = DecisionTreeRegressor(max_depth=None)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        iv += np.mean((y - y_pred) ** 2)

    assert_almost_equal(sk, iv)


def test_max_depth():
    sk = 0
    iv = 0

    for X, y in _make_datasets(10):
        reg = skDecisionTreeRegressor(max_depth=5)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        sk += np.mean((y - y_pred) ** 2)

        reg = DecisionTreeRegressor(max_depth=5)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        iv += np.mean((y - y_pred) ** 2)

    assert_almost_equal(sk, iv)


def test_max_depth_min_samples():
    sk = 0
    iv = 0

    for X, y in _make_datasets(10):
        reg = skDecisionTreeRegressor(max_depth=5,
                                      min_samples_split=4,
                                      min_samples_leaf=2)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        sk += np.mean((y - y_pred) ** 2)

        reg = DecisionTreeRegressor(max_depth=5,
                                    min_samples_split=4,
                                    min_samples_leaf=2)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        iv += np.mean((y - y_pred) ** 2)

    assert_almost_equal(sk, iv)


def test_max_decision_stump():
    sk = 0
    iv = 0

    for X, y in _make_datasets(10):
        reg = skDecisionTreeRegressor(max_depth=1)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        sk += np.mean((y - y_pred) ** 2)

        reg = DecisionTreeRegressor(max_depth=1)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        iv += np.mean((y - y_pred) ** 2)

    assert_almost_equal(sk, iv)

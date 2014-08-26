import numpy as np

from sklearn.tree import DecisionTreeRegressor as skDecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier as skDecisionTreeClassifier
from sklearn.datasets import make_regression
from sklearn.datasets import load_iris
from sklearn.utils.testing import assert_almost_equal

from ivalice.impl.tree import DecisionTreeRegressor
from ivalice.impl.tree import DecisionTreeClassifier


def _make_regression_datasets(n_times):
    for n in xrange(n_times):
        yield make_regression(n_samples=20, n_features=10, random_state=n)


def _make_classification_datasets(n_times):
    iris = load_iris()
    X, y = iris.data, iris.target
    for n in xrange(n_times):
        rng = np.random.RandomState(n)
        ind = rng.permutation(X.shape[0])[:20]
        yield X[ind], y[ind]


def test_mse_fully_developed():
    sk = 0
    iv = 0

    for X, y in _make_regression_datasets(10):
        reg = skDecisionTreeRegressor(max_depth=None)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        sk += np.mean((y - y_pred) ** 2)

        reg = DecisionTreeRegressor(max_depth=None)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        iv += np.mean((y - y_pred) ** 2)

    assert_almost_equal(sk, iv)


def test_mse_max_depth():
    for max_depth in (5, 1):
        sk = 0
        iv = 0

        for X, y in _make_regression_datasets(10):
            reg = skDecisionTreeRegressor(max_depth=max_depth)
            reg.fit(X, y)
            y_pred = reg.predict(X)
            sk += np.mean((y - y_pred) ** 2)

            reg = DecisionTreeRegressor(max_depth=max_depth)
            reg.fit(X, y)
            y_pred = reg.predict(X)
            iv += np.mean((y - y_pred) ** 2)

        assert_almost_equal(sk, iv)


def test_mse_max_depth_min_samples():
    sk = 0
    iv = 0

    for X, y in _make_regression_datasets(10):
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


def test_mse_max_depth_max_features():
    sk = 0
    iv = 0

    n_times = 30
    for X, y in _make_regression_datasets(n_times):
        reg = skDecisionTreeRegressor(max_depth=5,
                                      max_features=4,
                                      random_state=0)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        sk += np.mean((y - y_pred) ** 2)

        reg = DecisionTreeRegressor(max_depth=5,
                                    max_features=4,
                                    random_state=0)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        iv += np.mean((y - y_pred) ** 2)

    sk /= n_times
    iv /= n_times

    assert_almost_equal(sk, 405.5, 1)
    assert_almost_equal(iv, 394.8, 1)


def test_classif_max_depth():
    for criterion in ("gini", "entropy"):
        sk = 0
        iv = 0

        for X, y in _make_classification_datasets(10):
            clf = skDecisionTreeClassifier(criterion=criterion, max_depth=5)
            clf.fit(X, y)
            y_pred = clf.predict(X)
            sk += np.mean(y == y_pred)

            clf = DecisionTreeClassifier(criterion=criterion, max_depth=5)
            clf.fit(X, y)
            y_pred = clf.predict(X)
            iv += np.mean(y == y_pred)

        sk /= 10
        iv /= 10

    assert_almost_equal(sk, iv)

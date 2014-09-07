import numpy as np

from sklearn.tree import DecisionTreeRegressor as skRegTree
from sklearn.tree import DecisionTreeClassifier as skClassifTree

from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn.datasets import load_diabetes

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

from sklearn.utils.testing import assert_almost_equal

from ivalice.regression import TreeRegressor
from ivalice.classification import TreeClassifier


def _make_regression_datasets(n_times, sw=False):
    for n in xrange(n_times):
        X, y = make_regression(n_samples=100, n_features=10, random_state=n)
        if sw:
            rng = np.random.RandomState(n)
            w = rng.rand(X.shape[0])
            w[w <= 0.5] = 0.0
            yield X, y, w
        else:
            yield X, y


def _make_classification_datasets(n_times, sw=False):
    for n in xrange(n_times):
        X, y = make_classification(n_samples=20,
                                   n_features=10,
                                   n_informative=10,
                                   n_redundant=0,
                                   n_classes=3,
                                   random_state=n)
        if sw:
            rng = np.random.RandomState(n)
            w = np.round(rng.rand(X.shape[0]))
            w[w <= 0.5] = 0.0
            yield X, y, w
        else:
            yield X, y


def test_mse_fully_developed():
    sk = 0
    iv = 0

    for X, y in _make_regression_datasets(10):
        reg = skRegTree(max_depth=None)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        sk += np.mean((y - y_pred) ** 2)

        reg = TreeRegressor(max_depth=None)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        iv += np.mean((y - y_pred) ** 2)

    assert_almost_equal(sk, iv)


def test_mse_max_depth():
    for max_depth in (5, 1):
        sk = 0
        iv = 0

        for X, y in _make_regression_datasets(10):
            reg = skRegTree(max_depth=max_depth)
            reg.fit(X, y)
            y_pred = reg.predict(X)
            sk += np.mean((y - y_pred) ** 2)

            reg = TreeRegressor(max_depth=max_depth)
            reg.fit(X, y)
            y_pred = reg.predict(X)
            iv += np.mean((y - y_pred) ** 2)

        assert_almost_equal(sk, iv)


def test_mse_min_samples():
    sk = 0
    iv = 0

    for X, y in _make_regression_datasets(10):
        reg = skRegTree(max_depth=5,
                        min_samples_split=4,
                        min_samples_leaf=2)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        sk += np.mean((y - y_pred) ** 2)

        reg = TreeRegressor(max_depth=5,
                                    min_samples_split=4,
                                    min_samples_leaf=2)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        iv += np.mean((y - y_pred) ** 2)

    assert_almost_equal(sk, iv)


def test_mse_max_features():
    sk = 0
    iv = 0

    n_times = 30
    for X, y in _make_regression_datasets(n_times):
        reg = skRegTree(max_depth=5,
                        max_features=4,
                        random_state=0)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        sk += np.mean((y - y_pred) ** 2)

        reg = TreeRegressor(max_depth=5,
                                    max_features=4,
                                    random_state=0)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        iv += np.mean((y - y_pred) ** 2)

    sk /= n_times
    iv /= n_times

    assert_almost_equal(sk, 4588.4, 1)
    assert_almost_equal(iv, 4921.1, 1)


def test_mse_sample_weight():
    sk = 0
    iv = 0

    n_times = 10
    for X, y, w in _make_regression_datasets(n_times, sw=True):
        reg = skRegTree(max_depth=5)
        reg.fit(X, y, w)
        y_pred = reg.predict(X)
        sk += mean_squared_error(y, y_pred, sample_weight=w)

        reg = TreeRegressor(max_depth=5)
        reg.fit(X, y, w)
        y_pred = reg.predict(X)
        iv += mean_squared_error(y, y_pred, sample_weight=w)

    sk /= n_times
    iv /= n_times

    assert_almost_equal(sk, iv)


def test_mse_duplicate_features():
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    reg = skRegTree(max_depth=5)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    sk = np.mean((y - y_pred) ** 2)

    reg = TreeRegressor(max_depth=5)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    iv = np.mean((y - y_pred) ** 2)

    assert_almost_equal(sk, iv)


def test_classif_max_depth():
    for criterion in ("gini", "entropy"):
        sk = 0
        iv = 0

        for X, y in _make_classification_datasets(10):
            clf = skClassifTree(criterion=criterion,
                                max_depth=5,
                                random_state=1)
            clf.fit(X, y)
            y_pred = clf.predict(X)
            sk += np.mean(y == y_pred)

            clf = TreeClassifier(criterion=criterion,
                                         max_depth=5,
                                         random_state=1)
            clf.fit(X, y)
            y_pred = clf.predict(X)
            iv += np.mean(y == y_pred)

        sk /= 10
        iv /= 10

    assert_almost_equal(sk, iv)


def test_classif_sample_weight():
    for criterion in ("gini", "entropy"):
        sk = 0
        iv = 0

        for X, y, w in _make_classification_datasets(10, sw=True):
            clf = skClassifTree(criterion=criterion, max_depth=5)
            clf.fit(X, y, w)
            y_pred = clf.predict(X)
            sk += accuracy_score(y, y_pred, sample_weight=w)

            clf = TreeClassifier(criterion=criterion, max_depth=5)
            clf.fit(X, y, w)
            y_pred = clf.predict(X)
            iv += accuracy_score(y, y_pred, sample_weight=w)

        sk /= 10
        iv /= 10

    assert_almost_equal(sk, iv)

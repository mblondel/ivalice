import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.datasets import load_iris
from sklearn.datasets import load_linnerud
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_almost_equal

from ivalice.classification import GBClassifier
from ivalice.regression import GBRegressor

bunch = load_diabetes()
X, y = bunch.data, bunch.target

X_tr, X_te, y_tr, y_te = train_test_split(X, y,
                                          train_size=0.75,
                                          test_size=0.25,
                                          random_state=0)

iris = load_iris()
cond = iris.target <= 1
X_bin, y_bin = iris.data[cond], iris.target[cond]

X_bin_tr, X_bin_te, y_bin_tr, y_bin_te = train_test_split(X_bin, y_bin,
                                                          train_size=0.75,
                                                          test_size=0.25,
                                                          random_state=0)

iris = load_iris()
X_mult, y_mult = iris.data, iris.target

X_mult_tr, X_mult_te, y_mult_tr, y_mult_te = train_test_split(X_mult, y_mult,
                                                          train_size=0.75,
                                                          test_size=0.25,
                                                          random_state=0)


def test_squared_loss():
    reg = GradientBoostingRegressor(learning_rate=0.1, max_depth=3,
                                    random_state=0)
    reg.fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    sk = np.sqrt(np.mean((y_pred - y_te) ** 2))

    reg = DecisionTreeRegressor(max_features=1.0, max_depth=3, random_state=0)
    reg = GBRegressor(reg, n_estimators=100, learning_rate=0.1)
    reg.fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    iv = np.sqrt(np.mean((y_pred - y_te) ** 2))

    assert_almost_equal(sk, iv, 0)


def test_absolute_loss():
    # Check absolute loss with scikit-learn implementation.
    reg = GradientBoostingRegressor(learning_rate=0.1, loss="lad",
                                    random_state=0)
    reg.fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    sk = np.mean(np.abs(y_pred - y_te))

    reg = DecisionTreeRegressor(max_features=1.0, max_depth=3, random_state=0)
    reg = GBRegressor(reg, n_estimators=100, learning_rate=0.1, loss="absolute")
    reg.fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    iv = np.mean(np.abs(y_pred - y_te))

    assert_almost_equal(sk, iv, 0)


def test_absolute_loss_constant():
    # Check absolute loss with scikit-learn implementation.
    reg = DecisionTreeRegressor(max_features=1.0, max_depth=3, random_state=0)
    reg = GBRegressor(reg, n_estimators=100, learning_rate=0.1, loss="absolute",
                      step_size="constant")
    reg.fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    iv = np.mean(np.abs(y_pred - y_te))

    assert_almost_equal(iv, 55.6, 1)


def test_subsample():
    reg = DecisionTreeRegressor(max_features=1.0, max_depth=3,
                                random_state=0)
    reg = GBRegressor(reg, n_estimators=100, learning_rate=0.1, subsample=0.6,
                      random_state=0)
    reg.fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    mse = np.sqrt(np.mean((y_pred - y_te) ** 2))
    assert_almost_equal(mse, 62.8, 1)


def test_squared_hinge_loss():
    # With line search.
    clf = DecisionTreeClassifier(max_features=1.0, max_depth=3)
    clf = GBClassifier(clf, n_estimators=10, step_size="line_search")
    clf.fit(X_bin_tr, y_bin_tr)
    assert_almost_equal(clf.score(X_bin_te, y_bin_te), 1.0)

    # With constant step size.
    clf = DecisionTreeClassifier(max_features=1.0, max_depth=3)
    clf = GBClassifier(clf, n_estimators=10, step_size="constant",
                       learning_rate=0.1)
    clf.fit(X_bin_te, y_bin_te)
    assert_almost_equal(clf.score(X_bin_te, y_bin_te), 1.0)


def test_squared_hinge_loss_ovr():
    # With line search.
    clf = DecisionTreeClassifier(max_features=1.0, max_depth=3)
    clf = GBClassifier(clf, n_estimators=10, step_size="line_search")
    clf.fit(X_mult_tr, y_mult_tr)
    assert_almost_equal(clf.score(X_mult_te, y_mult_te), 0.974, 3)

    # With constant step size.
    clf = DecisionTreeClassifier(max_features=1.0, max_depth=3)
    clf = GBClassifier(clf, n_estimators=10, step_size="constant",
                       learning_rate=0.1)
    clf.fit(X_mult_te, y_mult_te)
    assert_almost_equal(clf.score(X_mult_te, y_mult_te), 1.0)


def test_log_loss():
    # With line search.
    clf = DecisionTreeClassifier(max_features=1.0, max_depth=3)
    clf = GBClassifier(clf, n_estimators=10, step_size="line_search",
                       loss="log")
    clf.fit(X_bin_tr, y_bin_tr)
    assert_almost_equal(clf.score(X_bin_te, y_bin_te), 1.0)

    # With constant step size.
    clf = DecisionTreeClassifier(max_features=1.0, max_depth=3)
    clf = GBClassifier(clf, n_estimators=10, step_size="constant",
                       loss="log", learning_rate=0.1)
    clf.fit(X_bin_te, y_bin_te)
    assert_almost_equal(clf.score(X_bin_te, y_bin_te), 1.0)


def test_multioutput_regression():
    data = load_linnerud()
    X, Y = data.data, data.target

    reg = DecisionTreeRegressor(max_features=1.0, max_depth=3)
    reg = GBRegressor(reg, n_estimators=10, step_size="line_search")
    Y_pred = reg.fit(X, Y).predict(X)

    acc = [0.697, 0.744, 0.631]
    acc2 = [r2_score(Y[:, k], Y_pred[:, k]) for k in xrange(Y.shape[1])]

    assert_array_almost_equal(acc, acc2, 3)

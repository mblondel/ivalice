import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.utils.testing import assert_almost_equal

from ivalice.impl.gradient_boosting import GBRegressor

bunch = load_diabetes()
X, y = bunch.data, bunch.target

X_tr, X_te, y_tr, y_te = train_test_split(X, y,
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

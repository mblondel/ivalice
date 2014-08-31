import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor as skRF
from sklearn.utils.testing import assert_almost_equal

from ivalice.impl.forest import RFRegressor

diabetes = load_diabetes()
X_d, y_d = diabetes.data, diabetes.target


def test_regression():
    rf = skRF(n_estimators=100,
              max_features=0.6,
              max_depth=3,
              bootstrap=False,
              random_state=0)
    rf.fit(X_d, y_d)
    y_pred = rf.predict(X_d)
    sk = np.mean((y_d - y_pred) ** 2)


    rf = RFRegressor(n_estimators=100,
                     max_features=0.6,
                     max_depth=3,
                     bootstrap=False,
                     random_state=0)

    rf.fit(X_d, y_d)
    y_pred = rf.predict(X_d)
    iv = np.mean((y_d - y_pred) ** 2)

    assert_almost_equal(sk, 2692.3, 1)
    assert_almost_equal(iv, 2689.9, 1)

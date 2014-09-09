import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_equal

from ivalice.ranking import McRank
from ivalice.ranking import OrdinalMcRank


bunch = load_diabetes()
X, y = bunch.data, bunch.target
y = np.round(y, decimals=-2)


def test_mcrank():
    gb = GradientBoostingClassifier(n_estimators=10,
                                    loss="deviance",
                                    random_state=0)
    mc = McRank(gb)
    mc.fit(X, y)
    assert_almost_equal(mc.score(X, y), 48.08, 2)


def test_mcrank_set_estimator_params():
    gb = GradientBoostingClassifier(n_estimators=5,
                                    loss="deviance",
                                    random_state=0)
    mc = McRank(gb)
    mc.set_estimator_params(n_estimators=10)
    assert_equal(gb.n_estimators, 10)


def test_mcrank_warm_start():
    gb = GradientBoostingClassifier(n_estimators=5,
                                    loss="deviance",
                                    warm_start=True,
                                    random_state=0)
    mc = McRank(gb)
    mc.fit(X, y)
    assert_almost_equal(mc.score(X, y), 56.06, 1)

    mc.set_estimator_params(n_estimators=10)
    mc.fit(X, y)
    assert_almost_equal(mc.score(X, y), 48.08, 2)


def test_ordinal_mcrank():
    gb = GradientBoostingClassifier(n_estimators=10,
                                    loss="deviance",
                                    random_state=0)
    mc = OrdinalMcRank(gb)
    mc.fit(X, y)
    assert_almost_equal(mc.score(X, y), 48.62, 2)


def test_ordinal_mcrank_set_estimator_params():
    gb = GradientBoostingClassifier(n_estimators=5,
                                    loss="deviance",
                                    random_state=0)
    mc = OrdinalMcRank(gb)
    mc.set_estimator_params(n_estimators=10)
    assert_equal(gb.n_estimators, 10)


def test_ordinal_mcrank_warm_start():
    gb = GradientBoostingClassifier(n_estimators=5,
                                    loss="deviance",
                                    warm_start=True,
                                    random_state=0)

    mc = OrdinalMcRank(gb)
    mc.fit(X, y)
    assert_almost_equal(mc.score(X, y), 56.35, 2)

    mc.set_estimator_params(n_estimators=10)
    mc.fit(X, y)
    assert_almost_equal(mc.score(X, y), 48.62, 2)

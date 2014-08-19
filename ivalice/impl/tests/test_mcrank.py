import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.testing import assert_almost_equal

from ivalice.impl.mcrank import McRank
from ivalice.impl.mcrank import OrdinalMcRank


bunch = load_diabetes()
X, y = bunch.data, bunch.target
y = np.round(y, decimals=-2)


def test_mcrank():
    gb = GradientBoostingClassifier(n_estimators=10,
                                    loss="deviance",
                                    random_state=0)
    mc = McRank(gb)
    mc.fit(X, y)
    assert_almost_equal(mc.score(X, y), 48.1, 1)


def test_ordinal_mcrank():
    gb = GradientBoostingClassifier(n_estimators=10,
                                    loss="deviance",
                                    random_state=0)
    mc = OrdinalMcRank(gb)
    mc.fit(X, y)
    assert_almost_equal(mc.score(X, y), 48.6, 1)

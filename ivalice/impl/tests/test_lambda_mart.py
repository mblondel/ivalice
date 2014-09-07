import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.testing import assert_almost_equal

from ivalice.impl.lambda_mart import LambdaMART

data = load_diabetes()
X, y = data.data, data.target
y /= (y.max() - y.min())


def test_lambda_mart_ndcg():
    for gains in ("linear", "exponential"):
        reg = DecisionTreeRegressor()
        lm = LambdaMART(reg, n_estimators=10, max_rank=10, gains=gains)
        lm.fit(X, y)
        ndcg = lm.score(X, y)
        assert_almost_equal(ndcg, 1.0)

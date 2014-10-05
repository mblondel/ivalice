import numpy as np

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.testing import assert_equal

from ivalice.classification import AdaBoostClassifier

X_bin, y_bin = make_classification(n_samples=200, n_classes=2, random_state=0)


def test_adaboost_binary():
    tree = DecisionTreeClassifier(max_depth=1, random_state=0)
    clf = AdaBoostClassifier(tree, n_estimators=10)
    clf.fit(X_bin, y_bin)
    assert_equal(clf.score(X_bin, y_bin), 0.96)

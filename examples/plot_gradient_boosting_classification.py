"""
================================
Gradient boosting classification
================================

This example compares the squared hinge and log losses in gradient boosting.
"""

print __doc__

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ivalice.classification import GBClassifier

n_estimators = 10

class Callback(object):

    def __init__(self, X_tr, y_tr, X_te, y_te):
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.X_te = X_te
        self.y_te = y_te
        self.accuracy_tr = []
        self.accuracy_te = []

    def __call__(self, est):
        y_pred_tr = est.predict(X_tr)
        y_pred_te = est.predict(X_te)
        self.accuracy_tr.append(np.mean(self.y_tr == y_pred_tr))
        self.accuracy_te.append(np.mean(self.y_te == y_pred_te))

data = load_iris()

X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target,
                                          train_size=0.5, test_size=0.5,
                                          random_state=0)

tree = DecisionTreeClassifier(max_depth=1)  # decision stumps

estimators = (
    ("squared hinge", "b", GBClassifier(tree, n_estimators=n_estimators,
                                        step_size="constant", learning_rate=0.1,
                                        loss="squared_hinge")),

    ("log", "g", GBClassifier(tree, n_estimators=n_estimators,
                              step_size="constant", learning_rate=0.1, loss="log")),

)

it = np.arange(n_estimators) + 1

for name, color, clf in estimators:
    clf.callback = Callback(X_tr, y_tr, X_te, y_te)
    clf.fit(X_tr, y_tr)

    plt.plot(it, clf.callback.accuracy_tr, label=name + " (train)", color=color,
             linestyle="-", linewidth=2)
    plt.plot(it, clf.callback.accuracy_te, label=name + " (test)", color=color,
             linestyle="--", linewidth=2)

plt.xlabel("Boosting iteration")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")

plt.show()

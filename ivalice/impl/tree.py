import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin

TREE_LEAF = -1
UNDEFINED = -2

class Tree(object):

    def __init__(self):
        self.threshold = []
        self.feature = []
        self.value = []
        self.children_left = []
        self.children_right = []

    def finalize(self):
        for attr in ("threshold", "feature", "value",
                     "children_left", "children_right", ):
            setattr(self, attr, np.array(getattr(self, attr)))
        return self


def _apply(X, tree):
    nodes = []
    for i in range(X.shape[0]):
        node = 0
        # While node not a leaf
        while tree.children_left[node] != TREE_LEAF:
            if X[i, tree.feature[node]] <= tree.threshold[node]:
                node = tree.children_left[node]
            else:
                node = tree.children_right[node]
        nodes.append(node)
    return np.array(nodes)


def _impurity(X_t, y_t, j, s, min_samples_leaf):
    cond = X_t[:, j] > s

    if len(cond) == 0:
        return np.inf

    X_L = X_t[cond]
    X_R = X_t[~cond]
    y_L = y_t[cond]
    y_R = y_t[~cond]

    N_t = len(X_t)
    N_L = len(X_L)
    N_R = len(X_R)

    if N_L < min_samples_leaf or N_R < min_samples_leaf:
        return np.inf

    y_hat_L = np.mean(y_L)
    y_hat_R = np.mean(y_R)

    err_L = np.sum((y_L - y_hat_L) ** 2)
    err_R = np.sum((y_R - y_hat_R) ** 2)

    return (err_L + err_R) / N_t


def _fit(X, y, max_depth=3, min_samples_split=2, min_samples_leaf=1):
    n_samples, n_features = X.shape

    root_indices = np.arange(n_samples)

    stack = [(root_indices, 0, 0, 0)]
    tree = Tree()

    node_t = 0

    while len(stack) > 0:
        indices_t, left_t, depth_t, parent_t = stack.pop()

        if node_t > 0:
            if left_t:
                tree.children_left[parent_t] = node_t
            else:
                tree.children_right[parent_t] = node_t

        X_t, y_t = X[indices_t], y[indices_t]
        N_t = len(X_t)

        if depth_t == max_depth or N_t < min_samples_split:
            tree.threshold.append(UNDEFINED)
            tree.feature.append(UNDEFINED)
            tree.value.append(np.mean(y_t))
            tree.children_left.append(TREE_LEAF)
            tree.children_right.append(TREE_LEAF)
            node_t += 1

            continue

        best_imp = np.inf
        best_thresh = None
        best_j = None

        for j in xrange(n_features):
            values = np.unique(X_t[:, j])

            for k in xrange(len(values) - 1):
                thresh = (values[k + 1] - values[k]) / 2.0 + values[k]
                imp = _impurity(X_t, y_t, j, thresh, min_samples_leaf)

                if imp < best_imp:
                    best_imp = imp
                    best_thresh = thresh
                    best_j = j

        if best_thresh is None:
            tree.threshold.append(UNDEFINED)
            tree.feature.append(UNDEFINED)
            tree.value.append(np.mean(y_t))
            tree.children_left.append(TREE_LEAF)
            tree.children_right.append(TREE_LEAF)
            node_t += 1

            continue

        indices_L = indices_t[X[indices_t, best_j] <= best_thresh]
        indices_R = indices_t[X[indices_t, best_j] > best_thresh]

        tree.threshold.append(best_thresh)
        tree.feature.append(best_j)
        tree.value.append(np.mean(y_t))

        # Node ids of children are not known yet.
        tree.children_left.append(0)
        tree.children_right.append(0)

        stack.append((indices_L, 1, depth_t + 1, node_t))
        stack.append((indices_R, 0, depth_t + 1, node_t))

        node_t += 1

    return tree.finalize()


class DecisionTreeRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        self.tree_ = _fit(X, y,
                          max_depth=self.max_depth,
                          min_samples_split=self.min_samples_split,
                          min_samples_leaf=self.min_samples_leaf)
        return self

    def predict(self, X):
        return self.tree_.value.take(_apply(X, self.tree_))

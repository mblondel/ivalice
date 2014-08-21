"""Regression trees"""

# Author: Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

import numpy as np
import numba

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin

from .sort import heapsort, quicksort

TREE_LEAF = -1
UNDEFINED = -2

DOUBLE_MAX = np.finfo(np.float64).max


class Tree(object):

    def __init__(self):
        self.threshold = []
        self.feature = []
        self.value = []
        self.children_left = []
        self.children_right = []

    def add_node(self, threshold, feature, value,
                 child_left=0, child_right=0):
        self.threshold.append(threshold)
        self.feature.append(feature)
        self.value.append(value)
        self.children_left.append(child_left)
        self.children_right.append(child_right)

    def add_terminal_node(self, value):
        self.threshold.append(UNDEFINED)
        self.feature.append(UNDEFINED)
        self.value.append(value)
        self.children_left.append(TREE_LEAF)
        self.children_right.append(TREE_LEAF)

    def finalize(self):
        self.threshold = np.array(self.threshold, dtype=np.float64)
        self.feature = np.array(self.feature, dtype=np.int32)
        self.value = np.array(self.value, dtype=np.float64)
        self.children_left = np.array(self.children_left, dtype=np.int32)
        self.children_right = np.array(self.children_right, dtype=np.int32)
        return self


@numba.njit("void(f8[:,:], i4[:], f8[:], i4[:], i4[:], i4[:])")
def _apply(X, feature, threshold, children_left, children_right, out):
    for i in range(X.shape[0]):
        node = 0
        # While node not a leaf
        while children_left[node] != TREE_LEAF:
            if X[i, feature[node]] <= threshold[node]:
                node = children_left[node]
            else:
                node = children_right[node]
        out[i] = node


@numba.njit("f8(f8[:], f8[:], i4[:], i4, i4, f8, i4)")
def _impurity(Xj, y, indices, start_t, end_t, s, min_samples_leaf):
    N_t = end_t - start_t
    N_L = 0
    N_R = 0
    y_hat_L = 0
    y_hat_R = 0

    for i in xrange(start_t, end_t):
        if Xj[i] > s:
            N_R += 1
            y_hat_R += y[indices[i]]
        else:
            N_L += 1
            y_hat_L += y[indices[i]]

    if N_R < min_samples_leaf or N_L < min_samples_leaf:
        return DOUBLE_MAX

    y_hat_L /= N_L
    y_hat_R /= N_R

    err_L = 0
    err_R = 0

    for i in xrange(start_t, end_t):
        if Xj[i] > s:
            err_R += (y[indices[i]] - y_hat_R) ** 2
        else:
            err_L += (y[indices[i]] - y_hat_L) ** 2

    return (err_L + err_R) / N_t


@numba.njit("void(f8[:,:], f8[:], i4[:], f8[:], i4, i4, i4, f8[:], i4[:])")
def _best_split(X, y, indices, Xj, start_t, end_t, min_samples_leaf,
                out_f8, out_i4):
    n_features = X.shape[1]

    best_imp = DOUBLE_MAX
    best_thresh = 0
    out_i4[0] = -1
    best_j = out_i4[0]

    size_t = end_t - start_t

    for j in xrange(n_features):

        for p in xrange(start_t, end_t):
            Xj[p] = X[indices[p], j]

        heapsort(Xj[start_t:end_t], indices[start_t:end_t], size_t)

        # FIXME: take care of duplicate feature values.
        for k in xrange(start_t, end_t - 1):
            thresh = (Xj[k + 1] - Xj[k]) / 2.0 + Xj[k]

            # FIXME: impurity can be computed efficiently from last
            # iteration.
            imp = _impurity(Xj, y, indices, start_t, end_t, thresh,
                            min_samples_leaf)

            if imp < best_imp:
                best_imp = imp
                best_thresh = thresh
                best_j = j
                pos_t = k + 1

    out_f8[0] = best_thresh
    out_i4[0] = best_j
    out_i4[1] = pos_t

# TODO:
# - implement introsort
# - pre-allocate stack
# - implement gini and entropy criteria
def _fit(X, y, max_depth=3, min_samples_split=2, min_samples_leaf=1):
    n_samples, n_features = X.shape

    indices = np.arange(n_samples).astype(np.int32)

    stack = [(0, n_samples, 0, 0, 0)]
    tree = Tree()

    node_t = 0

    # Buffers
    Xj = np.zeros(n_samples, dtype=np.float64)
    out_f8 = np.zeros(1, dtype=np.float64)
    out_i4 = np.zeros(2, dtype=np.int32)

    while len(stack) > 0:
        start_t, end_t, left_t, depth_t, parent_t = stack.pop()

        if node_t > 0:
            if left_t:
                tree.children_left[parent_t] = node_t
            else:
                tree.children_right[parent_t] = node_t

        indices_t = indices[start_t:end_t]
        mean_y_t = np.mean(y[indices_t])

        N_t = end_t - start_t

        if depth_t == max_depth or N_t < min_samples_split:
            tree.add_terminal_node(mean_y_t)
            node_t += 1
            continue

        _best_split(X, y, indices, Xj, start_t, end_t, min_samples_leaf,
                    out_f8, out_i4)
        best_thresh = out_f8[0]
        best_j = out_i4[0]
        pos_t = out_i4[1]

        if best_j == -1:
            tree.add_terminal_node(mean_y_t)
            node_t += 1
            continue

        # FIXME: move to _best_split
        cmp_func = lambda a,b: cmp(X[a, best_j], X[b, best_j])
        indices[start_t:end_t] = sorted(indices[start_t:end_t], cmp=cmp_func)

        tree.add_node(threshold=best_thresh,
                      feature=best_j,
                      value=mean_y_t)

        stack.append((start_t, pos_t, 1, depth_t + 1, node_t))
        stack.append((pos_t, end_t, 0, depth_t + 1, node_t))

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
        nodes = np.empty(X.shape[0], dtype=np.int32)
        _apply(X, self.tree_.feature, self.tree_.threshold,
               self.tree_.children_left, self.tree_.children_right, nodes)
        return self.tree_.value.take(nodes)

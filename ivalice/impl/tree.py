"""Classification and regression trees"""

# Author: Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

import numpy as np
import numba

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from .sort import heapsort

TREE_LEAF = -1
UNDEFINED = -2

DOUBLE_MAX = np.finfo(np.float64).max

MSE_CRITERION = 0
GINI_CRITERION = 1
ENTROPY_CRITERION = 2


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


@numba.njit("f8(f8[:], i4[:], i4, i4)")
def _assign_mse(y, indices, start_t, end_t):
    N_t = end_t - start_t
    s = 0
    for i in xrange(start_t, end_t):
        s += y[indices[i]]
    return s / N_t


@numba.njit("f8(f8[:], f8[:], i4[:], i4, i4, f8, i4)")
def _impurity_mse(Xj, y, indices, start_t, end_t, s, min_samples_leaf):
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


@numba.njit("void(f8[:], i4[:], i4, i4, i4[:])")
def _assign_classification(y, indices, start_t, end_t, buf):
    n_classes = buf.shape[0]

    for k in xrange(n_classes):
        buf[k] = 0

    for i in xrange(start_t, end_t):
        idx = int(y[indices[i]])
        buf[idx] += 1


@numba.njit("i4(f8[:], f8[:], i4[:], i4, i4, f8, i4[:], i4[:])")
def _compute_counts(Xj, y, indices, start_t, end_t, s, count_L, count_R):
    n_classes = count_L.shape[0]
    N_L = 0

    for k in xrange(n_classes):
        count_L[k] = 0
        count_R[k] = 0

    for i in xrange(start_t, end_t):
        if Xj[i] > s:
            idx = int(y[indices[i]])
            count_R[idx] += 1
        else:
            N_L += 1
            idx = int(y[indices[i]])
            count_L[idx] += 1

    return N_L


@numba.njit("f8(f8[:], f8[:], i4[:], i4, i4, f8, i4[:], i4[:], i4)")
def _impurity_gini(Xj, y, indices, start_t, end_t, s,
                   count_L, count_R, min_samples_leaf):
    n_classes = count_L.shape[0]
    N_t = end_t - start_t

    N_L = _compute_counts(Xj, y, indices, start_t, end_t, s, count_L, count_R)
    N_R = N_t - N_L

    if N_R < min_samples_leaf or N_L < min_samples_leaf:
        return DOUBLE_MAX

    gini_L = 0
    gini_R = 0
    for k in xrange(n_classes):
        proba_L = count_L[k] / float(N_t)
        proba_R = count_R[k] / float(N_t)

        gini_L += proba_L * (1 - proba_L)
        gini_R += proba_R * (1 - proba_R)

    #return float(N_L) / N_t * gini_L + float(N_R) / N_t * gini_R
    return N_L * gini_L + N_R * gini_R


@numba.njit("f8(f8[:], f8[:], i4[:], i4, i4, f8, i4[:], i4[:], i4)")
def _impurity_entropy(Xj, y, indices, start_t, end_t, s,
                      count_L, count_R, min_samples_leaf):
    n_classes = count_L.shape[0]
    N_t = end_t - start_t

    N_L = _compute_counts(Xj, y, indices, start_t, end_t, s, count_L, count_R)
    N_R = N_t - N_L

    if N_R < min_samples_leaf or N_L < min_samples_leaf:
        return DOUBLE_MAX

    ent_L = 0
    ent_R = 0
    for k in xrange(n_classes):
        proba_L = count_L[k] / float(N_t)
        proba_R = count_R[k] / float(N_t)

        if proba_L > 0:
            ent_L -= proba_L * np.log2(proba_L)

        if proba_R > 0:
            ent_R -= proba_R * np.log2(proba_R)

    #return float(N_L) / N_t * ent_L + float(N_R) / N_t * ent_R
    return N_L * ent_L + N_R * ent_R


@numba.njit("void(f8[:,:], f8[:], i4[:], f8[:], i4, i4, i4, i4, i4[:], i4[:], f8[:], i4[:])")
def _best_split(X, y, indices, Xj, start_t, end_t, criterion,
                min_samples_leaf, count_L, count_R, out_f8, out_i4):
    n_features = X.shape[1]

    best_imp = DOUBLE_MAX
    best_thresh = 0
    best_j = -1
    pos_t = -1

    size_t = end_t - start_t

    for j in xrange(n_features):

        for p in xrange(start_t, end_t):
            Xj[p] = X[indices[p], j]

        # FIXME: use introsort.
        heapsort(Xj[start_t:end_t], indices[start_t:end_t], size_t)

        # FIXME: take care of duplicate feature values.
        for k in xrange(start_t, end_t - 1):
            thresh = (Xj[k + 1] - Xj[k]) / 2.0 + Xj[k]

            # FIXME: impurity can be computed efficiently from last
            # iteration.
            if criterion == MSE_CRITERION:
                imp = _impurity_mse(Xj, y, indices, start_t, end_t, thresh,
                                    min_samples_leaf)
            elif criterion == GINI_CRITERION:
                imp = _impurity_gini(Xj, y, indices, start_t, end_t, thresh,
                                     count_L, count_R, min_samples_leaf)
            else:
                imp = _impurity_entropy(Xj, y, indices, start_t, end_t, thresh,
                                        count_L, count_R, min_samples_leaf)

            if imp < best_imp:
                best_imp = imp
                best_thresh = thresh
                best_j = j
                pos_t = k + 1

    out_f8[0] = best_thresh
    out_i4[0] = best_j
    out_i4[1] = pos_t

    best_j = out_i4[0]  # workaround some bug in Numba

    if best_j != -1:
        # Reorder indices for the best split.
        for p in xrange(start_t, end_t):
            Xj[p] = X[indices[p], best_j]

        heapsort(Xj[start_t:end_t], indices[start_t:end_t], size_t)


def _build(X, y, criterion, max_depth=3, min_samples_split=2,
           min_samples_leaf=1):
    n_samples, n_features = X.shape

    # FIXME: pre-allocate stack?
    stack = [(0, n_samples, 0, 0, 0)]
    tree = Tree()
    node_t = 0
    indices = np.arange(n_samples).astype(np.int32)

    # Buffers
    Xj = np.zeros(n_samples, dtype=np.float64)
    out_f8 = np.zeros(1, dtype=np.float64)
    out_i4 = np.zeros(2, dtype=np.int32)

    if criterion >= GINI_CRITERION:  # Classification case
        enc = LabelEncoder()
        y = enc.fit_transform(y).astype(np.float64)
        # Arrays which will contain the number of samples in each class.
        count_L = np.zeros(len(enc.classes_), dtype=np.int32)
        count_R = np.zeros(len(enc.classes_), dtype=np.int32)
    else:
        count_L = np.zeros(0, dtype=np.int32)
        count_R = np.zeros(0, dtype=np.int32)

    while len(stack) > 0:
        # Pick node from the stack.
        start_t, end_t, left_t, depth_t, parent_t = stack.pop()

        if node_t > 0:
            # Adjust children node id of parent.
            if left_t:
                tree.children_left[parent_t] = node_t
            else:
                tree.children_right[parent_t] = node_t

        # Node value.
        if criterion == MSE_CRITERION:
            y_hat_t = _assign_mse(y, indices, start_t, end_t)
        else:
            _assign_classification(y, indices, start_t, end_t, count_L)
            y_hat_t = np.argmax(count_L)

        # Number of samples which reached that node.
        N_t = end_t - start_t

        # Terminal node if max_depth or min_samples_split conditions are met.
        if depth_t == max_depth or N_t < min_samples_split:
            tree.add_terminal_node(y_hat_t)
            node_t += 1
            continue

        # Find best split across all features.
        _best_split(X, y, indices, Xj, start_t, end_t, criterion,
                    min_samples_leaf, count_L, count_R, out_f8, out_i4)
        best_thresh, best_j, pos_t = out_f8[0], out_i4[0], out_i4[1]

        # No best split found: terminal node.
        if best_j == -1:
            tree.add_terminal_node(y_hat_t)
            node_t += 1
            continue

        # Add node to the tree.
        tree.add_node(threshold=best_thresh,
                      feature=best_j,
                      value=y_hat_t)

        # Add left and right children to the stack.
        stack.append((start_t, pos_t, 1, depth_t + 1, node_t))
        stack.append((pos_t, end_t, 0, depth_t + 1, node_t))

        node_t += 1

    if criterion >= GINI_CRITERION:
        tree.value = enc.inverse_transform(tree.value)

    return tree.finalize()


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _get_criterion(self):
        return {"gini": GINI_CRITERION,
                "entropy": ENTROPY_CRITERION}[self.criterion]

    def fit(self, X, y):
        self.tree_ = _build(X, y, self._get_criterion(),
                            max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                            min_samples_leaf=self.min_samples_leaf)
        self.tree_.value = self.tree_.value.astype(np.int32)
        return self

    def predict(self, X):
        nodes = np.empty(X.shape[0], dtype=np.int32)
        _apply(X, self.tree_.feature, self.tree_.threshold,
               self.tree_.children_left, self.tree_.children_right, nodes)
        return self.tree_.value.take(nodes)


class DecisionTreeRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        self.tree_ = _build(X, y, MSE_CRITERION,
                            max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                            min_samples_leaf=self.min_samples_leaf)
        return self

    def predict(self, X):
        nodes = np.empty(X.shape[0], dtype=np.int32)
        _apply(X, self.tree_.feature, self.tree_.threshold,
               self.tree_.children_left, self.tree_.children_right, nodes)
        return self.tree_.value.take(nodes)

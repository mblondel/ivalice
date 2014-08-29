"""Classification and regression trees"""

# Author: Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

import numbers

import numpy as np
import numba

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

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


@numba.njit("f8(f8[:], f8[:], i4[:], i4, i4)")
def _assign_mse(y, sample_weight, samples, start_t, end_t):
    N_t = 0
    s = 0
    for ii in xrange(start_t, end_t):
        i = samples[ii]
        s += y[i] * sample_weight[i]
        N_t += sample_weight[i]
    return s / N_t


@numba.njit("f8(f8[:], f8[:], f8[:], i4[:], i4, i4, f8, f8[:])")
def _impurity_mse(Xj, y, sample_weight, samples, start_t, end_t, s, out_f8):
    N_L = 0
    N_R = 0
    y_hat_L = 0
    y_hat_R = 0

    for ii in xrange(start_t, end_t):
        i = samples[ii]
        if Xj[ii] > s:
            N_R += sample_weight[i]
            y_hat_R += y[i] * sample_weight[i]
        else:
            N_L += sample_weight[i]
            y_hat_L += y[i] * sample_weight[i]

    N_t = N_L + N_R

    if N_L == 0 or N_R == 0:
        return DOUBLE_MAX

    y_hat_L /= N_L
    y_hat_R /= N_R

    err_L = 0
    err_R = 0

    for ii in xrange(start_t, end_t):
        i = samples[ii]
        if Xj[ii] > s:
            err_R += sample_weight[i] * ((y[i] - y_hat_R) ** 2)
        else:
            err_L += sample_weight[i] * ((y[i] - y_hat_L) ** 2)

    out_f8[0] = N_L
    out_f8[1] = N_R
    out_f8[2] = N_t

    return (err_L + err_R) / N_t


@numba.njit("void(f8[:], f8[:], i4[:], i4, i4, f8[:])")
def _assign_classification(y, sample_weight, samples, start_t, end_t, buf):
    n_classes = buf.shape[0]

    for k in xrange(n_classes):
        buf[k] = 0

    for ii in xrange(start_t, end_t):
        i = samples[ii]
        idx = int(y[i])
        buf[idx] += sample_weight[i]


@numba.njit("void(f8[:], f8[:], f8[:], i4[:], i4, i4, f8, f8[:], f8[:], f8[:])")
def _compute_counts(Xj, y, sample_weight, samples, start_t, end_t, s,
                    count_L, count_R, out_f8):
    n_classes = count_L.shape[0]
    N_L = 0
    N_R = 0

    for k in xrange(n_classes):
        count_L[k] = 0
        count_R[k] = 0

    for ii in xrange(start_t, end_t):
        i = samples[ii]
        if Xj[ii] > s:
            N_R += sample_weight[i]
            idx = int(y[i])
            count_R[idx] += sample_weight[i]
        else:
            N_L += sample_weight[i]
            idx = int(y[i])
            count_L[idx] += sample_weight[i]

    out_f8[0] = N_L
    out_f8[1] = N_R
    out_f8[2] = N_L + N_R


@numba.njit("f8(f8[:], f8[:], f8[:], i4[:], i4, i4, f8, f8[:], f8[:], f8[:])")
def _impurity_gini(Xj, y, sample_weight, samples, start_t, end_t, s,
                   count_L, count_R, out_f8):
    n_classes = count_L.shape[0]

    _compute_counts(Xj, y, sample_weight, samples, start_t, end_t, s,
                    count_L, count_R, out_f8)
    N_L = out_f8[0]
    N_R = out_f8[1]
    N_t = out_f8[2]

    if N_L == 0 and N_R == 0:
        return DOUBLE_MAX

    gini_L = 0
    gini_R = 0
    for k in xrange(n_classes):
        proba_L = count_L[k] / N_t
        proba_R = count_R[k] / N_t

        gini_L += proba_L * (1 - proba_L)
        gini_R += proba_R * (1 - proba_R)

    #return float(N_L) / N_t * gini_L + float(N_R) / N_t * gini_R
    return N_L * gini_L + N_R * gini_R


@numba.njit("f8(f8[:], f8[:], f8[:], i4[:], i4, i4, f8, f8[:], f8[:], f8[:])")
def _impurity_entropy(Xj, y, sample_weight, samples, start_t, end_t, s,
                      count_L, count_R, out_f8):
    n_classes = count_L.shape[0]

    _compute_counts(Xj, y, sample_weight, samples, start_t, end_t, s,
                    count_L, count_R, out_f8)
    N_L = out_f8[0]
    N_R = out_f8[1]
    N_t = out_f8[2]

    if N_L == 0 or N_R == 0:
        return DOUBLE_MAX

    ent_L = 0
    ent_R = 0
    for k in xrange(n_classes):
        proba_L = count_L[k] / N_t
        proba_R = count_R[k] / N_t

        if proba_L > 0:
            ent_L -= proba_L * np.log2(proba_L)

        if proba_R > 0:
            ent_R -= proba_R * np.log2(proba_R)

    #return float(N_L) / N_t * ent_L + float(N_R) / N_t * ent_R
    return N_L * ent_L + N_R * ent_R


@numba.njit("void(f8[:,:], f8[:], f8[:], i4[:], i4[:], f8[:], i4, i4, i4, "
            "i4, f8[:], f8[:], f8[:])")
def _best_split(X, y, sample_weight, samples, features, Xj, start_t, end_t,
                criterion, min_samples_leaf, count_L, count_R, out_f8):
    best_imp = DOUBLE_MAX
    best_thresh = 0.0
    best_j = -1
    pos_t = -1
    N_L = 0.0
    N_R = 0.0
    N_t = 0.0

    size_t = end_t - start_t

    for j in features:

        for p in xrange(start_t, end_t):
            Xj[p] = X[samples[p], j]

        # Sort samples in nodes_t by their value for feature j.
        heapsort(Xj[start_t:end_t], samples[start_t:end_t], size_t)
        # FIXME: use introsort.

        # FIXME: take care of duplicate feature values.
        for k in xrange(start_t, end_t - 1):
            N_L = 1 + k - start_t
            N_R = size_t - N_L

            if N_R < min_samples_leaf or N_L < min_samples_leaf:
                continue

            # Choose splitting threshold.
            # Any value between Xj[k+1] and Xj[k] is fine.
            thresh = (Xj[k + 1] - Xj[k]) / 2.0 + Xj[k]

            # FIXME: impurity can be computed efficiently from last
            # iteration.
            if criterion == MSE_CRITERION:
                imp = _impurity_mse(Xj, y, sample_weight, samples, start_t,
                                    end_t, thresh, out_f8)
            elif criterion == GINI_CRITERION:
                imp = _impurity_gini(Xj, y, sample_weight, samples, start_t,
                                     end_t, thresh, count_L, count_R, out_f8)
            else:
                imp = _impurity_entropy(Xj, y, sample_weight, samples, start_t,
                                        end_t, thresh, count_L, count_R, out_f8)

            if imp < best_imp:
                best_imp = imp
                best_thresh = thresh
                best_j = j
                pos_t = k + 1
                N_L = out_f8[0]
                N_R = out_f8[1]
                N_t = out_f8[2]

    out_f8[0] = N_L
    out_f8[1] = N_R
    out_f8[2] = N_t
    out_f8[3] = best_thresh
    out_f8[4] = best_j
    out_f8[5] = pos_t

    if best_j != -1:
        # Reorder samples for the best split.
        for p in xrange(start_t, end_t):
            Xj[p] = X[samples[p], best_j]

        heapsort(Xj[start_t:end_t], samples[start_t:end_t], size_t)


def _build_tree(X, y, sample_weight, criterion, max_features=None, max_depth=3,
                min_samples_split=2, min_samples_leaf=1, random_state=None):
    n_samples, n_features = X.shape

    # FIXME: pre-allocate stack?
    tree = Tree()
    node_t = 0
    samples = np.arange(n_samples).astype(np.int32)
    samples = samples[sample_weight > 0]
    features = np.arange(n_features).astype(np.int32)
    stack = [(0, len(samples), 0, 0, np.sum(sample_weight), 0)]

    # Buffers
    Xj = np.zeros(n_samples, dtype=np.float64)
    out_f8 = np.zeros(6, dtype=np.float64)

    if criterion >= GINI_CRITERION:  # Classification case
        enc = LabelEncoder()
        y = enc.fit_transform(y).astype(np.float64)
        # Arrays which will contain the number of samples in each class.
        count_L = np.zeros(len(enc.classes_), dtype=np.float64)
        count_R = np.zeros(len(enc.classes_), dtype=np.float64)
    else:
        count_L = np.zeros(0, dtype=np.float64)
        count_R = np.zeros(0, dtype=np.float64)

    while len(stack) > 0:
        # Pick node from the stack.
        start_t, end_t, left_t, depth_t, N_t, parent_t = stack.pop()

        if node_t > 0:
            # Adjust children node id of parent.
            if left_t:
                tree.children_left[parent_t] = node_t
            else:
                tree.children_right[parent_t] = node_t

        # Node value.
        if criterion == MSE_CRITERION:
            y_hat_t = _assign_mse(y, sample_weight, samples, start_t, end_t)
        else:
            _assign_classification(y, sample_weight, samples, start_t, end_t,
                                   count_L)
            y_hat_t = np.argmax(count_L)

        size_t = end_t - start_t

        # Terminal node if max_depth or min_samples_split conditions are met.
        if depth_t == max_depth or size_t < min_samples_split:
            tree.add_terminal_node(y_hat_t)
            node_t += 1
            continue

        # Find best split across all features.
        if max_features != n_features:
            random_state.shuffle(features)

        _best_split(X, y, sample_weight, samples, features[:max_features], Xj,
                    start_t, end_t, criterion, min_samples_leaf,
                    count_L, count_R, out_f8)
        N_L, N_R, _, best_thresh, best_j, pos_t = out_f8
        best_j = int(best_j)
        pos_t = int(pos_t)

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
        stack.append((start_t, pos_t, 1, depth_t + 1, N_L, node_t))
        stack.append((pos_t, end_t, 0, depth_t + 1, N_R, node_t))

        node_t += 1

    if criterion >= GINI_CRITERION:
        tree.value = enc.inverse_transform(tree.value)

    return tree.finalize()


class _BaseTree(BaseEstimator):

    def _get_max_features(self, X):
        n_features = X.shape[1]

        if self.max_features is None:
            max_features = n_features
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * n_features))
            else:
                raise ValueError("max_features should be positive!")

        return max_features


class DecisionTreeClassifier(_BaseTree, ClassifierMixin):

    def __init__(self, criterion="gini", max_features=None, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, random_state=None):
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _get_criterion(self):
        return {"gini": GINI_CRITERION,
                "entropy": ENTROPY_CRITERION}[self.criterion]

    def fit(self, X, y, sample_weight=None):
        rng = check_random_state(self.random_state)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)

        self.tree_ = _build_tree(X, y, sample_weight,
                                 criterion=self._get_criterion(),
                                 max_features=self._get_max_features(X),
                                 max_depth=self.max_depth,
                                 min_samples_split=self.min_samples_split,
                                 min_samples_leaf=self.min_samples_leaf,
                                 random_state=rng)
        self.tree_.value = self.tree_.value.astype(np.int32)
        return self

    def predict(self, X):
        nodes = np.empty(X.shape[0], dtype=np.int32)
        _apply(X, self.tree_.feature, self.tree_.threshold,
               self.tree_.children_left, self.tree_.children_right, nodes)
        return self.tree_.value.take(nodes)


class DecisionTreeRegressor(_BaseTree, RegressorMixin):

    def __init__(self, max_features=None, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, random_state=None):
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        rng = check_random_state(self.random_state)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)

        self.tree_ = _build_tree(X, y, sample_weight,
                                 criterion=MSE_CRITERION,
                                 max_features=self._get_max_features(X),
                                 max_depth=self.max_depth,
                                 min_samples_split=self.min_samples_split,
                                 min_samples_leaf=self.min_samples_leaf,
                                 random_state=rng)
        return self

    def predict(self, X):
        nodes = np.empty(X.shape[0], dtype=np.int32)
        _apply(X, self.tree_.feature, self.tree_.threshold,
               self.tree_.children_left, self.tree_.children_right, nodes)
        return self.tree_.value.take(nodes)

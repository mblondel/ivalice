import numpy as np

from sklearn.utils.testing import assert_array_equal

from ivalice.impl.sort import quicksort

def test_quicksort():
    rng = np.random.RandomState(0)
    values = rng.rand(500)
    indices = np.arange(len(values)).astype(np.int32)

    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_indices = indices[sorted_idx]

    quicksort(values, indices, 0, len(values) - 1)

    assert_array_equal(sorted_values, values)
    assert_array_equal(sorted_indices, indices)

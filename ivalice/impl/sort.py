"""Efficient sorting routines"""

# Authors: Jake Vanderplas <jakevdp@cs.washington.edu> (quicksort)
#          Lars Buitinck <L.J.Buitinck@uva.nl> (heapsort)
#          Mathieu Blondel <mathieu@mblondel.org> (Numba port)
# License: BSD 3 clause

import numba


@numba.njit("void(f8[:], i4[:], i4, i4)")
def _dual_swap(values, indices, i1, i2):
    dtmp = values[i1]
    values[i1] = values[i2]
    values[i2] = dtmp

    itmp = indices[i1]
    indices[i1] = indices[i2]
    indices[i2] = itmp


@numba.njit("i4(f8[:], i4, i4)")
def _median3(values, start, end):
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    size = end - start + 1
    mid = start + size / 2

    a = values[start]
    b = values[mid]
    c = values[end]

    if a < b:
        if b < c:
            return mid
        elif a < c:
            return end
        else:
            return start
    elif b < c:
        if a < c:
            return start
        else:
            return end
    else:
        return mid


@numba.njit("i4(f8[:], i4[:], i4, i4)")
def _partition(values, indices, start, end):
    #pivot_idx = start + (end - start + 1) / 2
    pivot_idx = _median3(values, start, end)
    _dual_swap(values, indices, start, pivot_idx)
    pivot = values[start]
    i = start + 1
    j = start + 1

    while j <= end:
        if values[j] <= pivot:
            _dual_swap(values, indices, i, j)
            i += 1
        j += 1

    _dual_swap(values, indices, start, i - 1)

    return i - 1


@numba.njit("void(f8[:], i4[:], i4)")
def _sort2(values, indices, start):
    end = start + 1
    if values[start] > values[end]:
        _dual_swap(values, indices, start, end)


@numba.njit("void(f8[:], i4[:], i4)")
def _sort3(values, indices, start):
    mid = start + 1
    end = start + 2
    if values[start] > values[mid]:
        _dual_swap(values, indices, start, mid)
    if values[mid] > values[end]:
        _dual_swap(values, indices, mid, end)
        if values[start] > values[mid]:
            _dual_swap(values, indices, start, mid)


# As of Numba v0.13.2, recursion is not supported in Numba yet.
@numba.jit("void(f8[:], i4[:], i4, i4)")
def quicksort(values, indices, start, end):
    size = end - start + 1

    if size == 2:
        _sort2(values, indices, start)
    elif size == 3:
        _sort3(values, indices, start)
    if size > 1:
        i = _partition(values, indices, start, end)
        quicksort(values, indices, start, i - 1)
        quicksort(values, indices, i + 1, end)


@numba.njit("void(f8[:], i4[:], i4, i4)")
def _sift_down(values, indices, start, end):
    # Restore heap order in Xf[start:end] by moving the max element to start.

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and values[maxind] < values[child]:
            maxind = child
        if child + 1 < end and values[maxind] < values[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            _dual_swap(values, indices, root, maxind)
            root = maxind


@numba.njit("void(f8[:], i4[:], i4)")
def heapsort(values, indices, size):
    # heapify
    start = (size - 2) / 2
    end = size
    while True:
        _sift_down(values, indices, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = size - 1
    while end > 0:
        _dual_swap(values, indices, 0, end)
        _sift_down(values, indices, 0, end)
        end -= 1

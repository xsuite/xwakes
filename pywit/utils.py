from typing import Sequence, Union

import numpy as np


def round_sigfigs(arr: Union[float, Sequence[float]], sigfigs: int):
    """
    Rounds all the floats in an array to a the given number of
    significant digits.
    Warning: this routine is slow when working with very long
    arrays. If the final goal is to get unique elements to the given
    precision, it is better to use unique_sigfigs.
    :param arr: A float or an array-like sequence of floats.
    :param sigfigs: The number of significant digits (integer).
    :return: The same array (or float) rounded appropriately.
    """
    if np.isscalar(arr):
        return np.round(arr, sigfigs - 1 - int(np.floor(np.log10(np.abs(arr)))))
    else:
        return np.array([round_sigfigs(value, sigfigs) for value in arr])


def unique_sigfigs(arr: np.ndarray, sigfigs: int):
    """
    Given an array, returns it sorted and with only the elements that
    are unique after rounding to a given number of significant digits.
    :param arr: An array of floats
    :param sigfigs: The number of significant digits (integer)
    :return: the sorted array with unique elements
    """
    arr = np.unique(arr.flat)
    sorted_indices = np.argsort(arr.flat)
    log_diff = np.floor(np.log10(np.append(np.inf, np.diff(arr[sorted_indices]))))
    log_arr = np.floor(np.log10(arr[sorted_indices]))
    return arr.flat[sorted_indices[log_diff - log_arr >= -sigfigs]]

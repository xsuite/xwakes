from typing import Sequence, Union

import numpy as np


def round_sigfigs(arr: Union[float, Sequence[float]], sigfigs: int):
    """
    Rounds all the floats in an array to a the given number of
    significant digits.
    Warning: this routine is slow when working with very long
    arrays. In this case it is better to use unique_sigfigs
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
    Given an array returns all the elements that are unique after rounding to a given number of significant digits
    :param arr: An array of floats
    :param sigfigs: The number of significant digits (integer)
    """
    arr = np.unique(arr.flat)
    i = np.argsort(arr.flat)
    mag_d = np.floor(np.log10(np.append(np.inf, np.diff(arr[i]))))
    mag_arr = np.floor(np.log10(arr[i]))
    return arr.flat[i[mag_d - mag_arr >= -sigfigs]]

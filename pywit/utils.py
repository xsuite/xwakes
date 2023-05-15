from typing import Sequence, Union
from copy import deepcopy

import numpy as np


def round_sigfigs(arr: Union[float,Sequence[float]], sigfigs: int):
    """
    Rounds all the floats in an array to a the given number of
    significant digits.
    :param arr: A float or an array-like sequence of floats.
    :param sigfigs: The number of significant digits (integer).
    :return: The same array (or float) rounded appropriately.
    """
    if np.isscalar(arr):
        return np.round(arr, sigfigs - 1 - int(np.floor(np.log10(np.abs(arr)))))
    else:
        return np.array([round_sigfigs(value, sigfigs) for value in arr])


def create_list(a, n=1):
    """
    if a is a scalar, return a list containing n times the element a
    otherwise return a
    """
    if not (np.isscalar(a)):
        pass
    else:
        b = []
        for i in range(n):
            b.append(a)
        a = deepcopy(b)

    return a

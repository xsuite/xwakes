from typing import Sequence, Union

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

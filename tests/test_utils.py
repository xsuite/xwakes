from pywit.utils import round_sigfigs, unique_sigfigs

from numpy import testing
import numpy as np


def test_round_sigfigs():
    assert round_sigfigs(1.23456e7, 4) == 1.235e7

    arr = [1.23, 1.2345e-15, 6.789e15, -1020.34e3]
    testing.assert_equal(round_sigfigs(arr, 3),
                         np.array([1.23, 1.23e-15, 6.79e15, -1.02e6]))


def test_unique_sigfigs():
    arr = np.array([1, 3, 2, 3+1e-3, 0.5])

    testing.assert_equal(unique_sigfigs(arr, 4), np.array([0.5, 1., 2., 3., 3.001]))

    testing.assert_equal(unique_sigfigs(arr, 3), np.array([0.5, 1., 2., 3.]))



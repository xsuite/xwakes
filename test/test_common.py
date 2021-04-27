from math import sqrt, log

from numpy import vectorize, sin, cos

functions = [vectorize(f) for f in [lambda x: 1/x - log(x) + 1j * (x % 3),
                                    lambda x: x % 2 + x % 4 + 1 + 1j * log(x),
                                    lambda x: x / sqrt(x + x ** 2) + 1j * sqrt(abs(x - 5000)) / 10,
                                    lambda x: 5 * sin(x / 1500) + 1j * 7 * (cos(x / 2000)) + 3]]


def relative_error(a: complex, b: complex) -> float:
    if a == 0 and b == 0:
        return 0
    assert a != 0, "First argument cannot be zero"
    return abs(a - b) / abs(a)

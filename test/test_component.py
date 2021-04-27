from pywib.component import Component
from test.test_common import functions
from pywib.parameters import *

from itertools import product
from random import choice

from pytest import raises
from numpy import linspace, testing


def test_valid_addition():
    x = Component(choice(functions), choice(functions), 'x', (0, 0), (0, 0))
    a = Component(choice(functions), None, 'x', (0, 0), (0, 0))
    y = x + x
    b = a + a
    xs = linspace(1, 10000, 200)

    testing.assert_allclose(y.impedance(xs), 2 * x.impedance(xs), rtol=REL_TOL, atol=ABS_TOL)
    testing.assert_allclose(y.wake(xs), 2 * x.wake(xs), rtol=REL_TOL, atol=ABS_TOL)
    testing.assert_allclose(b.impedance(xs), 2 * a.impedance(xs), rtol=REL_TOL, atol=ABS_TOL)
    assert not b.wake

    assert x.is_compatible(y)


def test_valid_sum():
    values = linspace(1, 10000, 200)
    wake_aggregates = [0 for _ in range(200)]
    imp_aggregates = [0 for _ in range(200)]
    components = []
    for i in range(1, 100):
        x = Component(choice(functions), choice(functions), 'x', (0, 0), (0, 0))
        for j, value in enumerate(values):
            wake_aggregates[j] += x.wake(value)
            imp_aggregates[j] += x.impedance(value)
        components.append(x)

    y = sum(components)
    for impedance, wake, value in zip(imp_aggregates, wake_aggregates, values):
        assert (impedance - y.impedance(value)) < ABS_TOL
        assert (wake - y.wake(value)) < ABS_TOL

    assert x.is_compatible(y)


def test_invalid_addition():
    x = Component(choice(functions), choice(functions), 'x', (0, 0), (0, 0))

    for other in [Component(choice(functions), choice(functions), 'z', (0, 0), (0, 0)),
                  Component(choice(functions), choice(functions), 'x', (0, 1), (0, 0)),
                  Component(choice(functions), choice(functions), 'x', (0, 0), (1, 0))]:
        with raises(AssertionError):
            x + other


def test_invalid_initialization():
    # Verifies that an Component object needs to be initialized with at least one function
    with raises(AssertionError):
        Component(None, None, 'x', (0, 0), (0, 0))

    # Verifies that the plane has to be a valid character
    with raises(AssertionError):
        Component(lambda x: x, lambda x: x, plane='w', source_exponents=(0, 0), test_exponents=(0, 0))
    with raises(AssertionError):
        Component(lambda x: x, lambda x: x, plane='', source_exponents=(0, 0), test_exponents=(0, 0))

    # Verifies that the various exponents need to be given in the correct format
    with raises(AssertionError):
        Component(lambda x: x, lambda x: x, plane='x')
    with raises(AssertionError):
        Component(lambda x: x, lambda x: x, plane='x', source_exponents=(0, 0, 0), test_exponents=(0, 0))


def test_valid_multiplication():
    components = [Component(choice(functions), choice(functions), 'x', (0, 0), (0, 0)) for _ in range(4)]
    xs = linspace(1, 10000, 20)
    with raises(AssertionError):
        components[0] * components[1]

    for real, imaginary in product(linspace(1, 10000, 20), linspace(1, 10000, 20)):
        w = real + imaginary * 1j
        scaled_left = [w * c for c in components]
        scaled_right = [c * w for c in components]
        assert all(c.is_compatible(sl) for c, sl in zip(components, scaled_left))
        assert all(c.is_compatible(sr) for c, sr in zip(components, scaled_right))
        for scaled in (scaled_left, scaled_right):
            for c, s in zip(components, scaled):
                testing.assert_allclose(w * c.impedance(xs), s.impedance(xs), rtol=REL_TOL, atol=ABS_TOL)
                testing.assert_allclose(w * c.wake(xs), s.wake(xs), rtol=REL_TOL, atol=ABS_TOL)


def test_valid_division():
    components = [Component(choice(functions), choice(functions), 'x', (0, 0), (0, 0)) for _ in range(4)]
    with raises(TypeError):
        components[0] / components[1]
    with raises(TypeError):
        2 / components[0]

    xs = linspace(1, 10000, 20)

    for real, imaginary in product(linspace(1, 10000, 20), linspace(1, 10000, 20)):
        w = real + imaginary * 1j
        divided = [c / w for c in components]
        assert all(c.is_compatible(d) for c, d in zip(components, divided))
        for c, d in zip(components, divided):
            testing.assert_allclose(c.impedance(xs) / w, d.impedance(xs), rtol=REL_TOL, atol=ABS_TOL)
            testing.assert_allclose(c.wake(xs) / w, d.wake(xs), rtol=REL_TOL, atol=ABS_TOL)


def test_is_compatible():
    x = Component(choice(functions), choice(functions), 'x', (0, 0), (0, 0))
    for i, (plane, source, test) in enumerate(product(('x', 'y', 'z'),
                                                      ((0, 0), (0, 1), (1, 0), (1, 1)),
                                                      ((0, 0), (0, 1), (1, 0), (1, 1)))):
        y = Component(choice(functions), choice(functions), plane, source, test)
        if i == 0:
            assert y.is_compatible(x) and x.is_compatible(y)
        else:
            assert not y.is_compatible(x) and not x.is_compatible(y)
        assert x is not y

    y = x
    assert y is x


def test_component_sorting():
    f = lambda x: x
    components_unordered = [Component(f, f, plane, source, test) for plane, source, test in
                            product(('z', 'x', 'y'),
                            ((1, 0), (1, 1), (0, 0), (0, 1)),
                            ((1, 1), (0, 1), (0, 0), (1, 0)))]
    components_ordered = [Component(f, f, plane, source, test) for plane, source, test in
                          product(('x', 'y', 'z'),
                          ((0, 0), (0, 1), (1, 0), (1, 1)),
                          ((0, 0), (0, 1), (1, 0), (1, 1)))]
    for attempt, correct in zip(components_unordered, components_ordered):
        assert not attempt.is_compatible(correct)
    for attempt, correct in zip(sorted(components_unordered, key=lambda x: (x.plane, x.source_exponents,
                                                                            x.test_exponents)), components_ordered):
        assert attempt.is_compatible(correct), f"{attempt.plane, attempt.source_exponents, attempt.test_exponents}\n" \
                                               f"{correct.plane, correct.source_exponents, correct.test_exponents}"


def test_component_commutativity():
    x, y = (Component(choice(functions), choice(functions), 'x', (0, 0), (0, 0)) for _ in range(2))
    a = y + x
    b = x + y
    xs = linspace(1, 10000, 500)
    testing.assert_allclose(a.impedance(xs), b.impedance(xs), rtol=REL_TOL, atol=ABS_TOL)
    testing.assert_allclose(a.wake(xs), b.wake(xs), rtol=REL_TOL, atol=ABS_TOL)


def test_component_associativity():
    x = Component(choice(functions), choice(functions), 'x', (0, 0), (0, 0))
    y = Component(choice(functions), choice(functions), 'x', (0, 0), (0, 0))
    z = Component(choice(functions), choice(functions), 'x', (0, 0), (0, 0))
    a = x + y
    b = y + z
    c = a + z
    d = x + b
    xs = linspace(1, 10000, 500)
    testing.assert_allclose(c.impedance(xs), d.impedance(xs), rtol=REL_TOL, atol=ABS_TOL)
    testing.assert_allclose(c.wake(xs), d.wake(xs), rtol=REL_TOL, atol=ABS_TOL)


def test_distributivity():
    xs = linspace(1, 10000, 50)
    for _ in range(10):
        for scalar in linspace(0.1, 100, 20):
            x = Component(choice(functions), choice(functions), 'x', (0, 0), (0, 0))
            y = Component(choice(functions), choice(functions), 'x', (0, 0), (0, 0))
            a = scalar * (x + y)
            b = (scalar * x) + (scalar * y)
            testing.assert_allclose(a.impedance(xs), b.impedance(xs), rtol=REL_TOL, atol=ABS_TOL)
            testing.assert_allclose(a.wake(xs), b.wake(xs), rtol=REL_TOL, atol=ABS_TOL)

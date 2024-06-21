from pywit.component import (Component,
                             mix_fine_and_rough_sampling)
from test_common import functions
from pywit.parameters import *

from itertools import product
from random import choice

from pytest import raises,mark
from numpy import linspace, testing
import numpy as np


def test_mix_fine_and_rough_sampling_no_rois():
    start = 1
    stop = 10
    rough_points = 10
    assert np.allclose(
        np.geomspace(start, stop, rough_points),
        mix_fine_and_rough_sampling(start=start,
                                    stop=stop,
                                    rough_points=rough_points,
                                    fine_points=0,
                                    rois=[]))

def test_mix_fine_and_rough_sampling_with_rois_no_overlap():
    start = 1
    stop = 10
    rough_points = 10
    fine_points = 100
    # here there is no overlap between the rois and the rough points
    # so all the values in the rois should be included in the total array
    rois = [(5.1, 5.9), (6.1, 6.9)]
    target_val = np.sort(
        np.concatenate((np.geomspace(start, stop, rough_points),
                        np.linspace(rois[0][0], rois[0][1], fine_points),
                        np.linspace(rois[1][0], rois[1][1], fine_points))))
    assert np.allclose(
        target_val,
        mix_fine_and_rough_sampling(start=start,
                                    stop=stop,
                                    rough_points=rough_points,
                                    fine_points=fine_points,
                                    rois=rois))

def test_mix_fine_and_rough_sampling_with_rois_with_overlap():
    start = 1
    stop = 10
    rough_points = 10
    fine_points = 100
    rough_points_arr = np.geomspace(start, stop, rough_points)
    # in the first ROI the extrema overlap with two vales in the rough_points
    # up to the 7th significant digit so they should not appear in the total array
    rois = [(rough_points_arr[4] + 1e-7,
             rough_points_arr[5] + 1e-7),
             (6.1, 6.9)]
    target_val = np.sort(
        np.concatenate((rough_points_arr,
                        np.linspace(rois[0][0], rois[0][1], fine_points)[1:-1],
                        np.linspace(rois[1][0], rois[1][1], fine_points))))
    assert np.allclose(
        target_val,
        mix_fine_and_rough_sampling(start=start,
                                    stop=stop,
                                    rough_points=rough_points,
                                    fine_points=fine_points,
                                    rois=rois))

def simple_component_from_rois(f_rois=None, t_rois=None):
    return Component(impedance=lambda f: f+1j*f, wake=lambda z: z,
                     plane='z', source_exponents=(0, 0),
                     test_exponents=(0, 0), f_rois=f_rois, t_rois=t_rois)


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


@mark.parametrize(
    "f_rois, start, precision_factor, rough_points, expected_mesh",
    [
        [[],                            1e7, 10,  2, [1e7, 1e9]],
        [[(1e8, 2e8)],                  1e7, 0,   2, [1e7, 1e9]],
        [[(1e8, 2e8)],                  1e7, 1,   2, [1e7, 1e8, 2e8, 1e9]],
        [[(1e8, 2e8), (1e8, 2e8)],      1e7, 1,   2, [1e7, 1e8, 2e8, 1e9]],
        [[(3e7, 4e7), (2e9, 3e9)],      1e8, 1.5, 2, [1e8, 1e9]],
        [[(9e7, 1.1e8)],                1e8, 1.5, 2, [1e8, 1.05e8, 1.1e8, 1e9]],
        [[(9e8, 1.1e9)],                1e8, 1.5, 2, [1e8, 9e8, 9.5e8, 1e9]],
        [[(1e8, 2e8)],                  1e7, 2.5, 2, [1e7, 1e8, 1.25e8, 1.5e8, 1.75e8, 2e8, 1e9]],
        [[(1e8, 2e8), (3e8, 4e8)],      1e7, 1.5, 2, [1e7, 1e8, 1.5e8, 2e8, 3e8, 3.5e8, 4e8, 1e9]],
        [[(1e8, 2e8), (1.5e8, 2.5e8)],  1e7, 5/3, 3, [1e7, 1e8, 1.25e8, 1.5e8, 1.75e8, 2e8, 2.25e8, 2.5e8, 1e9]],
    ],
)
def test_impedance_wake_to_array_rois(f_rois, start, precision_factor,
                                 rough_points, expected_mesh):
    imp_component = simple_component_from_rois(f_rois=f_rois)
    frequencies, _ = imp_component.impedance_to_array(
                                    rough_points=rough_points,
                                    start=start, stop=1e9,
                                    precision_factor=precision_factor)

    testing.assert_equal(frequencies, expected_mesh)

    scaling_factor = 1e-15 # just to get meaningful times
    t_rois = [(i*scaling_factor, f*scaling_factor) for i,f in f_rois]
    wake_component = simple_component_from_rois(t_rois=t_rois)
    times, _ = wake_component.wake_to_array(
                                    rough_points=rough_points,
                                    start=start*scaling_factor, stop=1e-6,
                                    precision_factor=precision_factor)

    testing.assert_allclose(times, [e*scaling_factor for e in expected_mesh],
                            rtol=1e-15)

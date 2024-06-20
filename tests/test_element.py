from pywit.element import Element, Component
from test_common import functions
from pywit.parameters import *

from typing import List
from random import choice, uniform
from itertools import product

from pytest import raises
from numpy import linspace, pi, testing


def create_n_elements(n: int, all_components=False) -> List[Element]:
    """
    Creates a list of n Element objects whose component lists contain one component for each plane.
    :param n: The number of elements in the output list
    :param all_components: A flag indicating whether or not the elements in the output list should have components
    corresponding to every possible combination of source_exponents, test_exponents and plane
    :return: A list of mock-elements for testing
    """
    ls, bxs, bys = [[uniform(1, 100) for _ in range(n)] for _ in range(3)]
    fs, gs = [[[choice(functions) for _ in range(27 if all_components else 3)] for _ in range(n)] for _ in range(2)]
    if all_components:
        cs = [[Component(choice(functions), choice(functions), plane, source, test) for plane, source, test in
               product(['x', 'y', 'z'], [(0, 0), (0, 1), (1, 0)], [(0, 0), (0, 1), (1, 0)])] for _ in range(n)]
    else:
        cs = [[Component(f, g, plane, (choice([0, 1]), choice([0, 1])), (choice([0, 1]), choice([0, 1]))) for f, g, plane in
               zip(fs[i], gs[i], ['x', 'y', 'z'])] for i in range(n)]
    return [Element(l, bx, by, cl) for l, bx, by, cl in zip(ls, bxs, bys, cs)]


def check_if_elements_equal(e1: Element, e2: Element, start: float = 1, stop: float = 10000, points: int = 50,
                            verbose: bool = False) -> None:
    """
    Asserts that all the components of two elements have the same parameters. Also asserts that the functions
    of the components give the same values, within some tolerance, when evaluated over a range specified by arguments
    of this function.
    :param e1: The first element to be compared
    :param e2: The second element to be compared
    :param start: The first value the functions of the components will be evaluated at
    :param stop: The last value the functions of the components will be evaluated at
    :param points: The number of points the functions of the components will be evaluated in
    :param verbose: A flag indicating whether or not to give printouts which can be helpful in debugging
    """
    xs = linspace(start, stop, points)
    assert e1.is_compatible(e2, verbose=True)
    for c1, c2 in zip(e1.components, e2.components):
        if verbose:
            print(f"{c1}\n{c2}\n\n")
        c1.is_compatible(c2)
        if c1.impedance and c2.impedance:
            testing.assert_allclose(c1.impedance(xs), c2.impedance(xs), rtol=REL_TOL, atol=ABS_TOL)
        if c1.wake and c2.wake:
            testing.assert_allclose(c1.wake(xs), c2.wake(xs), rtol=REL_TOL, atol=ABS_TOL)


def test_initialization():
    """
    Verifies that components can be initialized without throwing exceptions if done correctly, and that exceptions are
    thrown if invalid initialization arguments are given
    """
    for length, beta_x, beta_y in product(*[linspace(-1, 10, 23) for _ in range(3)]):
        if all(e > 0 for e in (length, beta_x, beta_y)):
            Element(length, beta_x, beta_y)
        else:
            with raises(AssertionError):
                Element(length, beta_x, beta_y)

    x = Component(choice(functions), choice(functions), 'x', (0, 0), (0, 0))
    y = Component(choice(functions), choice(functions), 'y', (0, 0), (0, 0))

    Element(1, 1, 1, [x])
    Element(1, 1, 1, [x, y])
    with raises(AssertionError):
        Element(beta_x=1, beta_y=1, components=[x])
    with raises(AssertionError):
        Element(length=1, beta_y=1, components=[x])
    with raises(AssertionError):
        Element(length=1, beta_x=1, components=[x])


def test_addition():
    """
    Verifies that addition of elements works as intended
    """
    for _ in range(10):
        l1, l2, bx1, bx2, by1, by2 = (uniform(1, 100) for _ in range(6))
        fs1, gs1 = [choice(functions) for _ in range(3)], [choice(functions) for _ in range(3)]
        fs2, gs2 = [choice(functions) for _ in range(3)], [choice(functions) for _ in range(3)]
        cs1 = [Component(f, g, plane, (0, 0), (0, 0)) for f, g, plane in zip(fs1, gs1, ['x', 'y', 'z'])]
        cs2 = [Component(f, g, plane, (0, 0), (0, 0)) for f, g, plane in zip(fs2, gs2, ['x', 'y', 'z'])]
        e1 = Element(l1, bx1, by1, cs1)
        e2 = Element(l2, bx2, by2, cs2)
        e_left = e1 + e2
        e_right = e2 + e1
        ratios = [e1.beta_x / e_left.beta_x, e1.beta_y / e_left.beta_y,
                  e2.beta_x / e_left.beta_x, e2.beta_y / e_left.beta_y]
        for e in e_left, e_right:
            assert abs(e.length - l1 - l2) < ABS_TOL
            assert abs(e.beta_x - (l1 * bx1 + l2 * bx2) / e.length) < ABS_TOL
            assert abs(e.beta_y - (l1 * by1 + l2 * by2) / e.length) < ABS_TOL
            assert len(e.components) == 3
            for c, c1, c2 in zip(e.components, cs1, cs2):
                left_coeff = (ratios[0] ** c1.power_x) * (ratios[1] ** c1.power_y)
                right_coeff = (ratios[2] ** c2.power_x) * (ratios[3] ** c2.power_y)
                for i in linspace(1, 10000, 20):
                    assert abs(c.impedance(i) - left_coeff * c1.impedance(i) - right_coeff * c2.impedance(i)) < ABS_TOL
                    assert abs(c.wake(i) - left_coeff * c1.wake(i) - right_coeff * c2.wake(i)) < ABS_TOL


def test_associativity():
    """
    Verifies that addition of elements is associative
    """
    for _ in range(10):
        es = create_n_elements(3)
        sum1 = es[0] + (es[1] + es[2])
        sum2 = (es[0] + es[1]) + es[2]
        check_if_elements_equal(sum1, sum2, points=100)


def test_scalar_multiplication():
    """
    Verifies that multiplication of an element by a scalar works as intended
    """
    for n in range(1, 10):
        l = uniform(1, 100)
        e = Element(l, 2, 2, [Component(choice(functions), choice(functions), plane, (0, 0), (0, 0))
                              for plane in ('x', 'y', 'z')])
        summed_e = sum(e for _ in range(n))
        scaled_e = n * e
        check_if_elements_equal(summed_e, scaled_e, points=100)


def test_rotation_by_zero():
    """
    Verifies that rotation by an angle theta=0 leaves an element unchanged
    """
    for ac in [False, True]:
        es = create_n_elements(10, all_components=ac)
        rotated_es = [e.rotated(0) for e in es]
        for e1, e2 in zip(es, rotated_es):
            check_if_elements_equal(e1, e2)


def test_angle_addition():
    """
    Verifies that rotating an element by theta=(a + b) is equivalent to rotating by theta=a and then theta=b
    """
    es = create_n_elements(3, all_components=True)
    for alpha in linspace(0, 3 * pi / 2, 3):
        for beta in linspace(0, 3 * pi / 2, 3):
            theta = alpha + beta
            rotated_once = [e.rotated(theta) for e in es]
            rotated_twice = [e.rotated(alpha).rotated(beta) for e in es]
            for e1, e2 in zip(rotated_once, rotated_twice):
                check_if_elements_equal(e1, e2)


def test_rotation_commutativity():
    """
    Verifies rotation of elements is a commutative operation
    """
    es = create_n_elements(10)
    for alpha in linspace(0, 3 * pi / 2, 3):
        for beta in linspace(0, 3 * pi / 2, 3):
            for e1, e2 in [(e.rotated(alpha).rotated(beta), e.rotated(beta).rotated(alpha)) for e in es]:
                check_if_elements_equal(e1, e2)


def test_sum():
    """
    Verifies that the sum()-syntax works as intended
    """
    for ac in [False, True]:
        a, b, c = create_n_elements(3, all_components=ac)
        check_if_elements_equal(sum([a, b, c]), a + b + c)


def test_exponent_conservation():
    """
    Verifies that the sums (a + b) and (c + d), corresponding to source_exponents and test_exponents, of components are
    conserved after rotation of an element
    """
    es = []
    exp_sums = []
    for a, b, c, d in product((0, 1), (0, 1), (0, 1), (0, 1)):
        components = [Component(lambda x: x, None, plane, (a, b), (c, d)) for plane in ('x', 'y', 'z')]
        es.append(Element(1, 1, 1, components))
        exp_sums.append((a + b, c + d))

    for theta in linspace(0, 2 * pi, 100):
        for e, exp_sum in zip(es, exp_sums):
            rotated_e = e.rotated(theta)
            for c in rotated_e.components:
                sum1, sum2 = sum(c.source_exponents), sum(c.test_exponents)
                assert (sum1, sum2) == exp_sum


# def test_rotation_addition_order():
#     for theta in linspace(0, 2 * pi, 40):
#         es = create_n_elements(2)
#         sum_rotate = sum(es).rotated(theta)
#         rotate_sum = sum(e.rotated(theta) for e in es)
#         check_if_elements_equal(sum_rotate, rotate_sum, verbose=True)


# def test_simplified_rotation_addition():
#     # Passes if the ratio (beta_x / beta_y) is the same for all elements, fails otherwise
#     es = [Element(1, 1, beta, [Component(lambda x: x, lambda x: x, 'x', (0, 0), (0, 0))]) for beta in (1, 2)]
#     for theta in linspace(0, 2 * pi, 100):
#         sum_rotate = sum(es).rotated(theta)
#         rotate_sum = sum(e.rotated(theta) for e in es)
#         check_if_elements_equal(sum_rotate, rotate_sum)

from typing import Tuple
from pywit.component import Component

import scipy as sp
import numpy as np

from copy import deepcopy

from scipy.constants import c as c_light, mu_0
from scipy.special import erfc, zeta

free_space_impedance = mu_0 * c_light

def transverse_dipolar_impedance_tesla_cavity(freq, alpha, L, a, g):
    return 2 / ((2 * np.pi * freq / c_light) * a ** 2) * 1j * free_space_impedance / (
                np.pi * (2 * np.pi * freq / c_light) * a ** 2) * (
                1 + (1 + 1j) * alpha * L / a * np.sqrt(np.pi / ((2 * np.pi * freq / c_light) * g))) ** (-1)


def transverse_dipolar_wake_tesla_cavity(time, alpha, L, a, g):
    s00 = g/8 * (a/(alpha * L)) ** 2
    return ((4*free_space_impedance*c_light*s00)/(np.pi*a**4) * np.heaviside(time, 0) *
            (1 - (1 + np.sqrt(c_light*time/s00)) * np.exp(-np.sqrt(c_light*time/s00))))


def longitudinal_impedance_tesla_cavity(freq, alpha, L, a, g):
    return (1j * free_space_impedance / (np.pi * (2 * np.pi * freq / c_light) * a ** 2) *
            (1 + (1 + 1j) * alpha * L / a * np.sqrt(np.pi / ((2 * np.pi * freq / c_light) * g))) ** (-1))


def longitudinal_wake_tesla_cavity(time, alpha, L, a, g):
    s00 = g/8 * (a/(alpha * L)) ** 2

    return ((free_space_impedance * c_light/(np.pi*a**2)) * np.heaviside(time, 0) *
            np.exp(-np.pi*c_light*time/(4*s00)) * erfc(np.sqrt(np.pi*c_light*time/(4*s00))))


def zero_function(x, alpha, L, a, g):
    return np.zeros_like(x)


def create_tesla_cavity(plane: str, exponents: Tuple[int, int, int, int], a: float, g: float, L: float) -> Component:
    """
    Creates a single component object modeling a periodic accelerating stucture.
    Follow K. Bane formalism developped in SLAC-PUB-9663, "Short-range dipole wakefields
    in accelerating structures for the NLC".
    :param plane: the plane the component corresponds to
    :param exponents: four integers corresponding to (source_x, source_y, test_x, test_y) aka (a, b, c, d)
    :param a: accelerating structure iris gap in m
    :param g: individual cell gap in m
    :param L: period length in m
    :return: A component object of a periodic accelerating structure
    """

    # Material properties required for the skin depth computation are derived from the input Layer attributes
    # material_resistivity = layer.dc_resistivity
    # material_relative_permeability = layer.magnetic_susceptibility
    # material_permeability = material_relative_permeability * scipy.constants.mu_0

    # Create the skin depth as a function offrequency and layer properties
    # delta_skin = lambda freq: (material_resistivity/ (2*pi*abs(freq) * material_permeability)) ** (1/2)

    gamma = g / L
    alpha1 = 0.4648
    alpha = 1 - alpha1 * np.sqrt(gamma) - (1 - 2*alpha1) * gamma

    s00 = g/8 * (a/(alpha * L)) ** 2

    # Longitudinal impedance and wake
    if plane == 'z' and exponents == (0, 0, 0, 0):
        impedance = longitudinal_impedance_tesla_cavity
        wake = longitudinal_wake_tesla_cavity
    # Transverse dipolar impedance and wake
    elif (plane == 'x' and exponents == (1, 0, 0, 0)) or (plane == 'y' and exponents == (0, 1, 0, 0)):
        impedance = transverse_dipolar_impedance_tesla_cavity
        wake = transverse_dipolar_wake_tesla_cavity
    else:
        print("Warning: resistive wall impedance not implemented for component {}{}. Set to zero".format(plane,
                                                                                                         exponents))
        impedance = zero_function
        wake = zero_function

    return Component(impedance=np.vectorize(lambda f: impedance(f, alpha, L, a, g)),
                     wake=np.vectorize(lambda t: wake(t, alpha, L, a, g)),
                     plane=plane, source_exponents=exponents[:2], test_exponents=exponents[2:])


def shunt_impedance_flat_taper_stupakov_formula(a, b, tantheta, w, fcutoff, comp=None):
    if comp is None:
        comp = ['zlong', 'zxdip', 'zydip', 'zxqua', 'zyqua']

    scalar_input = False
    if np.isscalar(comp):
        scalar_input = True
        comp = [comp]

    z_0 = mu_0 * c_light  # free space impedance
    r_over_q = []

    for comp in comp:

        if comp.endswith('long'):
            cst = 4. * mu_0 * fcutoff / 2.  # factor 4 due to use of half-gaps here
            I = 7. * zeta(3, 1) / (2. * np.pi ** 2) * tantheta * (
                    b - a)  # approx. integral (sp.zeta(3.,1.) is Riemann zeta function at x=3)
            # I=7.*1.202057/(2.*np.pi**2)*tantheta*(b-a);

        elif comp.endswith('ydip'):
            cst = z_0 * w * np.pi / 4.
            I = tantheta * (1. / (a ** 2) - 1. / (b ** 2)) / (2. * np.pi)  # approx. integral

        elif comp.endswith('xqua'):
            cst = -z_0 * np.pi / 4.
            I = tantheta * (1. / a - 1. / b) / (np.pi ** 2)  # approx. integral

        elif comp.endswith('xdip'):
            cst = z_0 * np.pi / 4.
            I = tantheta * (1. / a - 1. / b) / (np.pi ** 2)  # approx. integral

        elif comp.endswith('yqua'):
            cst = z_0 * np.pi / 4.
            I = tantheta * (1. / a - 1. / b) / (np.pi ** 2)  # approx. integral
        else:
            # mock backup values
            cst = 0
            I = 0

        # shunt impedance /Q
        r_over_q.append(cst * I)

    if scalar_input:
        return r_over_q[0]
    else:
        return np.array(r_over_q)


def integrand_stupakov(g, w, Gint, gpow):
    """
    computes integrand for Stupakov integral for a rectangular linear taper at a given
    vertical half-gap(s) g (can be an array)
    Note that (g')^2 has been taken out of the integral.

    w is the width of the taper.
    Gint is the indice of the G function to use (0 is for G0=F in Stupakov's paper)
    gpow is the power to which 1/g is taken
    See Phys. Rev. STAB 10, 094401 (2007)
    """

    return g_stupakov(g / w, Gint) / (g ** gpow)


def g_stupakov(x, g_int):
    """
    computes F (when Gint=0) or G_[Gint] function from Stupakov's formulas
    for a rectangular linear taper at a given x=g/w ratio
    x can be an array
    See Phys. Rev. STAB 10, 094401 (2007)
    """

    x = np.array(create_list(x))
    g = np.zeros(len(x))

    for ix, xelem in enumerate(x):

        old_g = 1.
        eps = 1e-5  # relative precision of the summation
        incr = 10  # increment for sum computation
        m = np.arange(incr)
        mlimit = 1e6

        while (abs(g[ix] - old_g) > eps * abs(g[ix])) and (m[-1] < mlimit):
            Garray = g_element(m, xelem, g_int)
            old_g = g[ix]
            g[ix] += np.sum(Garray)
            m += incr

        # print Gint,m[-1]
        if (m[-1] >= mlimit):
            print("Pb in G_Stupakov: maximum number of elements reached !", m[-1], xelem, ", err=",
                  abs((g[ix] - old_g) / g[ix]))

    return g


def g_element(m, x, g_int):
    """
    computes each element of series defining F (when Gint=0) or G_[Gint] function
    from Stupakov's formulas for a rectangular linear taper at a given x=g/w ratio
    m is an array of integers where we compute those monomials
    See Phys. Rev. STAB 10, 094401 (2007)
    """

    if g_int == 0:
        val = (2 * m + 1) * np.pi * x / 2
        res2 = ((1 - np.exp(-2 * val)) / (1 + np.exp(-2 * val)) * (
                2 * (np.exp(-val)) / (1 + np.exp(-2 * val))) ** 2) / (2 * m + 1)

        return res2

    elif g_int == 1:
        val = (2 * m + 1) * np.pi * x / 2
        res2 = x ** 3 * (2 * m + 1) * (4 * np.exp(-2 * val)) * (1 + np.exp(-2 * val)) / ((1 - np.exp(-2 * val)) ** 3)
        return res2

    elif g_int == 2:
        val = (2 * m + 1) * np.pi * x / 2
        res2 = x ** 2 * (2 * m + 1) * (
            ((1 - np.exp(-2 * val)) / (1 + np.exp(-2 * val)) * (2 * (np.exp(-val)) / (1 + np.exp(-2 * val))) ** 2))
        return res2

    elif g_int == 3:
        val = m * np.pi * x
        res2 = x ** 2 * (2 * m) * ((1 - np.exp(-2 * val)) / (1 + np.exp(-2 * val)) * (
                2 * (np.exp(-val)) / (1 + np.exp(-2 * val))) ** 2)
        return res2

    else:
        print("Pb in G_element for Stupakov's formulas: Gint not 0, 1, 2 or 3 !")


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
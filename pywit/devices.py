from typing import Tuple
from pywit.component import Component
from pywit.element import Element
from pywit.utilities import create_resonator_component
from pywit.interface import component_names
from pywit.utils import create_list

import numpy as np
from scipy import integrate

from scipy.constants import c as c_light, mu_0
from scipy.special import erfc, zeta

free_space_impedance = mu_0 * c_light


def create_tesla_cavity_component(plane: str, exponents: Tuple[int, int, int, int], a: float, g: float,
                                  L: float) -> Component:
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

    # Create the skin depth as a function of frequency and layer properties
    # delta_skin = lambda freq: (material_resistivity/ (2*pi*abs(freq) * material_permeability)) ** (1/2)

    gamma = g / L
    alpha1 = 0.4648
    alpha = 1 - alpha1 * np.sqrt(gamma) - (1 - 2*alpha1) * gamma

    s00 = g/8 * (a/(alpha * L)) ** 2

    # Longitudinal impedance and wake
    if plane == 'z' and exponents == (0, 0, 0, 0):
        def longitudinal_impedance_tesla_cavity(freq):
            return (1j * free_space_impedance / (np.pi * (2 * np.pi * freq / c_light) * a ** 2) *
                    (1 + (1 + 1j) * alpha * L / a * np.sqrt(np.pi / ((2 * np.pi * freq / c_light) * g))) ** (-1))

        def longitudinal_wake_tesla_cavity(time):
            return ((free_space_impedance * c_light / (np.pi * a ** 2)) * np.heaviside(time, 0) *
                    np.exp(-np.pi * c_light * time / (4 * s00)) * erfc(np.sqrt(np.pi * c_light * time / (4 * s00))))

        impedance = longitudinal_impedance_tesla_cavity
        wake = longitudinal_wake_tesla_cavity
    # Transverse dipolar impedance and wake
    elif (plane == 'x' and exponents == (1, 0, 0, 0)) or (plane == 'y' and exponents == (0, 1, 0, 0)):

        def transverse_dipolar_impedance_tesla_cavity(freq):
            return 2 / ((2 * np.pi * freq / c_light) * a ** 2) * 1j * free_space_impedance / (
                    np.pi * (2 * np.pi * freq / c_light) * a ** 2) * (
                           1 + (1 + 1j) * alpha * L / a * np.sqrt(np.pi / ((2 * np.pi * freq / c_light) * g))) ** (-1)

        def transverse_dipolar_wake_tesla_cavity(time):
            return ((4 * free_space_impedance * c_light * s00) / (np.pi * a ** 4) * np.heaviside(time, 0) *
                    (1 - (1 + np.sqrt(c_light * time / s00)) * np.exp(-np.sqrt(c_light * time / s00))))

        impedance = transverse_dipolar_impedance_tesla_cavity
        wake = transverse_dipolar_wake_tesla_cavity
    else:
        print("Warning: tesla cavity impedance not implemented for component {}{}. Set to zero".format(plane,
                                                                                                       exponents))

        def zero_function(x):
            return np.zeros_like(x)

        impedance = zero_function
        wake = zero_function

    return Component(impedance=np.vectorize(lambda f: impedance(f)),
                     wake=np.vectorize(lambda t: wake(t)),
                     plane=plane, source_exponents=exponents[:2], test_exponents=exponents[2:])


def g_addend(m, x, g_index):
    """
    Computes the m-th addend of series defining F (when Gint=0) or G_[Gint] function from Stupakov's formulas for a
    rectangular linear taper at a given halfagp/width. m can be an array, in which case the function returns an array.
    See Phys. Rev. STAB 10, 094401 (2007)
    :param m: the index of the addend (it can be an array)
    :param x: halfagp/width ratio
    :param g_index: the index of the G function
    """

    if g_index == 0:
        val = (2 * m + 1) * np.pi * x / 2
        res2 = ((1 - np.exp(-2 * val)) / (1 + np.exp(-2 * val)) * (
                2 * (np.exp(-val)) / (1 + np.exp(-2 * val))) ** 2) / (2 * m + 1)

        return res2

    elif g_index == 1:
        val = (2 * m + 1) * np.pi * x / 2
        res2 = x ** 3 * (2 * m + 1) * (4 * np.exp(-2 * val)) * (1 + np.exp(-2 * val)) / ((1 - np.exp(-2 * val)) ** 3)
        return res2

    elif g_index == 2:
        val = (2 * m + 1) * np.pi * x / 2
        res2 = x ** 2 * (2 * m + 1) * (
            ((1 - np.exp(-2 * val)) / (1 + np.exp(-2 * val)) * (2 * (np.exp(-val)) / (1 + np.exp(-2 * val))) ** 2))
        return res2

    elif g_index == 3:
        val = m * np.pi * x
        res2 = x ** 2 * (2 * m) * ((1 - np.exp(-2 * val)) / (1 + np.exp(-2 * val)) * (
                2 * (np.exp(-val)) / (1 + np.exp(-2 * val))) ** 2)
        return res2

    else:
        print("Pb in G_element for Stupakov's formulas: Gint not 0, 1, 2 or 3 !")


def g_stupakov(x: float, g_index: int):
    """
    Computes F (when Gint=0) or G_[Gint] function from Stupakov's formulas for a rectangular linear taper at a given
    x=halfagp/width ratio. See Phys. Rev. STAB 10, 094401 (2007)
    :param x: the halfagp/width ratio
    :param g_index: the index of the G function
    """
    x = np.array(create_list(x))
    g = np.zeros(len(x))

    for ix, xelem in enumerate(x):

        old_g = 1.
        eps = 1e-5  # relative precision of the summation
        incr = 10  # increment for sum computation
        m = np.arange(incr)
        m_limit = 1e6

        while (abs(g[ix] - old_g) > eps * abs(g[ix])) and (m[-1] < m_limit):
            g_array = g_addend(m, xelem, g_index)
            old_g = g[ix]
            g[ix] += np.sum(g_array)
            m += incr

        if m[-1] >= m_limit:
            print("Warning: maximum number of elements reached in g_stupakov!", m[-1], xelem, ", err=",
                  abs((g[ix] - old_g) / g[ix]))

    return g


def integrand_stupakov(g: float, w: float, g_index: int, g_power: int):
    """
    Computes the integrand for the Stupakov integral for a rectangular linear taper at a given vertical half-gap g
    Note that (g')^2 has been taken out of the integral. See Phys. Rev. STAB 10, 094401 (2007)
    :param g: the small halfagp of the taper
    :param w: the width of the taper
    :param g_index: the indice of the G function to use (0 is for G0=F in Stupakov's paper)
    :param g_power: is the power to which 1/g is taken
    """

    return g_stupakov(g / w, g_index) / (g ** g_power)


def shunt_impedance_flat_taper_stupakov_formula(half_gap_small: float, half_gap_big: float, taper_slope: float,
                                                half_width: float, cutoff_frequency: float = None,
                                                component_id: str = None, approximate_integrals: bool = False) -> float:
    """
    Computes the shunt impedance in Ohm(/m if not longitudinal) of one single rectangular linear taper using Stupakov's
    formulae (Phys. Rev. STAB 10, 094401 - 2007), multiplied by Z0*c/(4*pi) to convert to SI units.
    Taper is in the vertical plane.
    We use here half apertures (half-gap and half-width) whereas Stupakov's paper is expressed with full apertures. This
    does not make any difference except for an additional factor 4 here for longitudinal impedance.

    The formula is valid under the conditions of low frequency and length of taper much larger than its transverse
    dimensions

    :param half_gap_small: small vertical half-gap
    :param half_gap_big: large vertical half-gap
    :param taper_slope: the slope of the taper
    :param half_width: width of the taper (constant)
    :param cutoff_frequency: the cutoff frequency (used only for the longitudinal component)
    :param component_id: is a list with the names or a single name of the components for which ones computes the R/Q
    (ex: zlong, zydip, zxquad, etc.)
    :param approximate_integrals: use approximated formulas to compute the integrals. It can be used if one assumes
    small half_gap_big/half_width ratio
    :return: the shunt impedance of each component in comp
    """
    z_0 = mu_0 * c_light  # free space impedance

    if cutoff_frequency is None and component_id == 'zlong':
        raise ValueError("cutoff_frequency must be specified when component_id is 'zlong'")

    if component_id == 'zlong':
        g_index = 0
        g_power = 0
        cst = 4. * mu_0 * cutoff_frequency / 2.  # factor 4 due to use of half-gaps here
        i = 7. * zeta(3, 1) / (2. * np.pi ** 2) * taper_slope * (
                half_gap_big - half_gap_small)  # approx. integral (sp.zeta(3.,1.) is Riemann zeta function at x=3)
        # I=7.*1.202057/(2.*np.pi**2)*tantheta*(b-a);

    elif component_id == 'zydip':
        g_index = 1
        g_power = 3
        cst = z_0 * half_width * np.pi / 4.
        i = taper_slope * (1. / (half_gap_small ** 2) - 1. / (half_gap_big ** 2)) / (
                    2. * np.pi)  # approx. integral

    elif component_id == 'zxqua':
        g_index = 2
        g_power = 2
        cst = -z_0 * np.pi / 4.
        i = taper_slope * (1. / half_gap_small - 1. / half_gap_big) / (np.pi ** 2)  # approx. integral

    elif component_id == 'zxdip':
        g_index = 3
        g_power = 2
        cst = z_0 * np.pi / 4.
        i = taper_slope * (1. / half_gap_small - 1. / half_gap_big) / (np.pi ** 2)  # approx. integral

    elif component_id == 'zyqua':
        g_index = 2
        g_power = 2
        cst = z_0 * np.pi / 4.
        i = taper_slope * (1. / half_gap_small - 1. / half_gap_big) / (np.pi ** 2)  # approx. integral
    else:
        # mock backup values
        g_power = 0
        g_index = 0
        cst = 0
        i = 0

    if not approximate_integrals:
        # computes numerically the integral instead of using its approximation
        i, err = integrate.quadrature(integrand_stupakov, half_gap_small, half_gap_big,
                                      args=(half_width, g_index, g_power), tol=1.e-3, maxiter=200)
        i *= taper_slope  # put back g' factor that was dropped

    return cst * i


def create_flat_taper_stupakov_formula_component(half_gap_small: float, half_gap_big: float, taper_slope: float,
                                                 half_width: float, plane: str, exponents: Tuple[int, int, int, int],
                                                 cutoff_frequency: float = None, component_id: str = None) -> Component:
    """
    Creates a component using the flat taper Stupakov formula
    :param half_gap_small: small vertical half-gap
    :param half_gap_big: large vertical half-gap
    :param taper_slope: the slope of the taper
    :param half_width: width of the taper (constant)
    :param plane: the plane the component corresponds to
    :param exponents: four integers corresponding to (source_x, source_y, test_x, test_y) aka (a, b, c, d)
    :param cutoff_frequency: the cutoff frequency (used only for the longitudinal component)
    :param component_id: the components for which the R/Q is computed (ex: zlong, zydip, zxqua, etc.)
    :return: A component object of a flat taper
    """
    if component_id not in component_names.keys():
        raise ValueError(f"component_id must be one of the following: {component_names.keys()}")

    r_shunt = shunt_impedance_flat_taper_stupakov_formula(half_gap_small=half_gap_small, half_gap_big=half_gap_big,
                                                          taper_slope=taper_slope, half_width=half_width,
                                                          cutoff_frequency=cutoff_frequency, component_id=component_id)

    return create_resonator_component(plane=plane, exponents=exponents, r=r_shunt, q=1, f_r=cutoff_frequency)


def create_flat_taper_stupakov_formula_element(half_gap_small: float, half_gap_big: float, taper_slope: float,
                                               half_width: float, beta_x: float, beta_y: float,
                                               cutoff_frequency: float = None, component_ids: str = None,
                                               name: str = "Flat taper", tag: str = "",
                                               description: str = "") -> Element:
    """
    Creates a component using the flat taper Stupakov formula
    :param half_gap_small: small vertical half-gap
    :param half_gap_big: large vertical half-gap
    :param taper_slope: the slope of the taper
    :param half_width: width of the taper (constant)
    :param beta_x: The size of the beta function in the x-plane at the position of the taper
    :param beta_y: The size of the beta function in the y-plane at the position of the taper
    :param cutoff_frequency: the cutoff frequency (used only for the longitudinal component)
    :param component_ids: a list of components to be computed
    :param name: A user-specified name of the Element
    :param tag: A string corresponding to a specific Element
    :param description: A description for the Element
    :return: A component object of a flat taper
    """
    length = (half_gap_big - half_gap_small) * taper_slope

    if component_ids is None:
        component_ids = ['zlong', 'zxdip', 'zydip', 'zxqua', 'zyqua']

    components = []
    for component_id in component_ids:
        plane, exponents, _ = component_names[component_id]
        components.append(create_flat_taper_stupakov_formula_component(half_gap_small=half_gap_small,
                                                                       half_gap_big=half_gap_big,
                                                                       taper_slope=taper_slope,
                                                                       half_width=half_width,
                                                                       plane=plane, exponents=exponents,
                                                                       cutoff_frequency=cutoff_frequency,
                                                                       component_id=component_id))

    return Element(length=length, beta_x=beta_x, beta_y=beta_y, components=components, name=name, tag=tag,
                   description=description)

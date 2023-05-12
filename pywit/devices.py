from typing import Tuple
from pywit.component import Component

import numpy as np

from scipy.constants import c as c_light, mu_0
from scipy.special import erfc, zeta

free_space_impedance = mu_0 * c_light


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
        print("Warning: resistive wall impedance not implemented for component {}{}. Set to zero".format(plane,
                                                                                                         exponents))

        def zero_function(x):
            return np.zeros_like(x)

        impedance = zero_function
        wake = zero_function

    return Component(impedance=np.vectorize(lambda f: impedance(f)),
                     wake=np.vectorize(lambda t: wake(t)),
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

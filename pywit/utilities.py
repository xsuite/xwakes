from pywit.component import Component
from pywit.element import Element
from pywit.interface import Layer

from yaml import load, SafeLoader
from typing import Tuple, Dict

from numpy import vectorize, sqrt, exp, pi, sin, cos, abs, sign, heaviside
import scipy.constants

def string_to_params(name: str, include_is_impedance: bool = True):
    """
    Converts a string describing some specific component to the set of parameters which give this component
    :param name: The string description of the component to be converted. Either of the form <pabcd> where p is either
    'x', 'y' or 'z', and abcd are four integers giving the source- and test exponents of the component. Optionally,
    a single character, 'z' or 'w' can be included at the front of the string if it is necessary to indicate that the
    given component is either a wake- or impedance component. In this case, the include_is_impedance flag must be set.
    :param include_is_impedance: A flag indicating whether or not "name" includes a 'z' or 'w' at the beginning.
    :return: A tuple containing the necessary information to describe the type of the given component
    """
    if include_is_impedance:
        is_impedance = name[0] == 'z'
        name = name[1:]

    plane = name[0].lower()
    exponents = tuple(int(n) for n in name[1:])

    return (is_impedance, plane, exponents) if include_is_impedance else (plane, exponents)


def create_component_from_config(identifier: str) -> Component:
    """
    Creates a Component object as specified by a .yaml config file
    :param identifier: A unique identifier-string corresponding to a component specification in config/component.yaml
    :return: A Component object initialized according to the specification
    """
    with open("config/component.yaml", "r") as f:
        cdict = load(f, Loader=SafeLoader)[identifier]
        wake = vectorize(lambda x: eval(cdict['wake'], {'exp': exp, 'sqrt': sqrt, 'pi': pi, 'x': x})) if 'wake' in cdict else None
        impedance = vectorize(lambda x: eval(cdict['impedance'], {'exp': exp, 'sqrt': sqrt, 'pi': pi, 'x': x})) if 'impedance' in cdict else None
        name = cdict['name'] if 'name' in cdict else ""
        plane = cdict['plane']
        source_exponents = (int(cdict['source_exponents'][0]), int(cdict['source_exponents'][-1]))
        test_exponents = (int(cdict['test_exponents'][0]), int(cdict['test_exponents'][-1]))
        return Component(impedance, wake, plane, source_exponents, test_exponents, name)


def create_element_from_config(identifier: str) -> Element:
    """
    Creates an Element object as specified by a .yaml config file
    :param identifier: A unique identifier-string corresponding to an element specification in config/element.yaml
    :return: An Element object initialized according to the specifications
    """
    with open("config/element.yaml", "r") as f:
        edict = load(f, Loader=SafeLoader)[identifier]
        length = float(edict['length']) if 'length' in edict else 0
        beta_x = float(edict['beta_x']) if 'beta_x' in edict else 0
        beta_y = float(edict['beta_y']) if 'beta_y' in edict else 0
        name = edict['name'] if 'name' in edict else ""
        tag = edict['tag'] if 'tag' in edict else ""
        components = [create_component_from_config(c_id) for c_id in edict['components'].split()] \
            if 'components' in edict else []
        return Element(length, beta_x, beta_y, components, name, tag)


def create_resonator_component(plane: str, exponents: Tuple[int, int, int, int],
                               r: float, q: float, f_r: float) -> Component:
    """
    Creates a single component object belonging to a resonator
    :param plane: the plane the component corresponds to
    :param exponents: four integers corresponding to (source_x, source_y, test_x, test_y) aka (a, b, c, d)
    :param r: the shunt impedance of the given component of the resonator
    :param q: the quality factor of the given component of the resonator
    :param f_r: the resonance frequency of the given component of the resonator
    :return: A component object of a resonator, specified by the input arguments
    """
    root_term = sqrt(1 - 1 / (4 * q ** 2) + 0J)
    omega_r = 2 * pi * f_r
    if plane == 'z':
        impedance = lambda f: r / (1 - 1j * q * (f_r / f - f / f_r))
        omega_bar = omega_r * root_term
        alpha = omega_r / (2 * q)
        wake = lambda t: (omega_r * r * exp(-alpha * t) * (
                    cos(omega_bar * t) - alpha * sin(omega_bar * t) / omega_bar) / q).real
    else:
        impedance = lambda f: (f_r * r) / (f * (1 - 1j * q * (f_r / f - f / f_r)))
        wake = lambda t: (omega_r * r * exp(-omega_r * t / (2 * q)) * sin(omega_r * root_term * t) / (q * root_term)).real

    d = f_r / (2 * q)
    # TODO: add ROI(s) for wake

    return Component(vectorize(impedance), vectorize(wake), plane, source_exponents=exponents[:2], test_exponents=exponents[2:],
                     f_rois=[(f_r - d, f_r + d)])


def create_resonator_element(length: float, beta_x: float, beta_y: float,
                             rs: Dict[str, float], qs: Dict[str, float], fs: Dict[str, float],
                             tag: str = 'resonator', description: str = '') -> Element:
    """
    Creates an element object representing a resonator.
    :param length: The length, in meters, of the resonator element
    :param beta_x: The value of the beta function in the x-direction at the position of the resonator element
    :param beta_y: The value of the beta function in the y-direction at the position of the resonator element
    :param rs: A dictionary where the keys correspond to a plane followed by four exponents, i.e. "y0100", and the
    values give the Shunt impedance corresponding to this particular component
    :param qs: A dictionary where the keys correspond to a plane followed by four exponents, i.e. "y0100", and the
    values give the quality factor of the specified component of the resonator
    :param fs: A dictionary where the keys correspond to a plane followed by four exponents, i.e. "y0100", and the
    values give the resonance frequency corresponding to the particular component
    :param tag: An optional short string used to place elements into categories
    :param description: An optional short description of the element
    :return: An element object as specified by the user-input
    """
    assert set(rs.keys()) == set(qs.keys()) == set(fs.keys()), "The three input dictionaries describing the " \
                                                               "resonator do not all have identical keys"
    components = []
    for key in rs.keys():
        plane, exponents = string_to_params(key, include_is_impedance=False)
        components.append(create_resonator_component(plane, exponents, rs[key], qs[key], fs[key]))

    return Element(length, beta_x, beta_y, components, tag=tag, description=description)


def create_resistive_wall_component(plane: str, exponents: Tuple[int, int, int, int],
                                    layer: Layer, radius: float) -> Component:
    """
    Creates a single component object modeling a resistive wall impedance/wake
    Only longitudinal and transverse dipolar impedances are supported.
    :param plane: the plane the component corresponds to
    :param exponents: four integers corresponding to (source_x, source_y, test_x, test_y) aka (a, b, c, d)
    :param layer: the chamber material, as a pywit Layer object
    :param radius: the chamber radius in m
    :return: A component object of a resistive wall, specified by the input arguments
    """

    c_light = scipy.constants.speed_of_light  # m s-1
    free_space_impedance = scipy.constants.mu_0 * c_light

    # Material properties required for the skin depth computation are derived from the input Layer attributes
    material_resistivity = layer.dc_resistivity
    material_relative_permeability = layer.magnetic_susceptibility
    material_permeability = material_relative_permeability * scipy.constants.mu_0

    # Create the skin depth as a function offrequency and layer properties
    delta_skin = lambda f: (material_resistivity/ (2*pi*abs(f) * material_permeability)) ** (1/2)

    # Longitudinal impedance and wake
    if (plane == 'z' and exponents == (0, 0, 0, 0)):
        impedance = lambda f: (1/2)* (1+sign(f)*1j) * material_resistivity / (pi * radius) * (1 / delta_skin(f))
        wake = lambda t: - (c_light) / (2*pi*radius) * (free_space_impedance * material_resistivity/pi)**(1/2) * 1/(t**(1/2))
    # Transverse dipolar impedance
    elif (plane == 'x' and exponents == (1, 0, 0, 0)) or (plane == 'y' and exponents == (0, 1, 0, 0)):
        impedance = lambda f: (c_light/(2*pi*f)) * (1+sign(f)*1j) * material_resistivity / (pi * radius**3) * (1 / delta_skin(f))
        wake = lambda t: - (c_light) / (2*pi*radius**3) * (free_space_impedance * material_resistivity/pi)**(1/2) * 1/(t**(3/2))
    else:
        print("Warning: resistive wall impedance not implemented for component {}{}. Set to zero".format(plane, exponents))
        impedance = lambda f: 0
        wake = lambda f: 0

    return Component(vectorize(impedance), vectorize(wake), plane, source_exponents=exponents[:2], test_exponents=exponents[2:])

def create_TESLA_cavity(plane: str, exponents: Tuple[int, int, int, int],
                                    a: float, g: float, L: float) -> Component:
    """
    Creates a single component object modeling a periodic accelerating stucture.
    Follow K. Bane formalism developped in SLAC-PUB-9663, "Short-range dipole wakefields
    in accelerating structures for the NLC".
    :param plane: the plane the component corresponds to
    :param exponents: four integers corresponding to (source_x, source_y, test_x, test_y) aka (a, b, c, d)
    :param a: accelerating structure irig gap in m
    :param g: individual cell gap in m
    :param L: period length in m
    :return: A component object of a periodic accelerating structure
    """

    c_light = scipy.constants.speed_of_light  # m s-1
    free_space_impedance = scipy.constants.mu_0 * c_light
    erfc = scipy.special.erfc

    # Material properties required for the skin depth computation are derived from the input Layer attributes
    # material_resistivity = layer.dc_resistivity
    # material_relative_permeability = layer.magnetic_susceptibility
    # material_permeability = material_relative_permeability * scipy.constants.mu_0

    # Create the skin depth as a function offrequency and layer properties
    # delta_skin = lambda f: (material_resistivity/ (2*pi*abs(f) * material_permeability)) ** (1/2)

    gamma = g / L
    alpha1 = 0.4648
    alpha = 1 - alpha1 * sqrt(gamma) - (1 - 2*alpha1) * gamma

    s00 = g/8 * (a/(alpha * L)) ** 2

    # Longitudinal impedance and wake
    if (plane == 'z' and exponents == (0, 0, 0, 0)):
        impedance = lambda f: 1j*free_space_impedance/(pi* (2*pi*f/c_light) * a**2) * (1 + (1+1j) * alpha * L / a * sqrt(pi/((2*pi*f/c_light) * g)))**(-1)
        wake = lambda t: (free_space_impedance * c_light/(pi*a**2)) * heaviside(t, 0) * exp(-pi*c_light*t/(4*s00)) * erfc(sqrt(pi*c_light*t/(4*s00)))
    # Transverse dipolar impedance and wake
    elif (plane == 'x' and exponents == (1, 0, 0, 0)) or (plane == 'y' and exponents == (0, 1, 0, 0)):
        impedance = lambda f: 2 / ((2*pi*f/c_light) * a**2) * 1j*free_space_impedance/(pi* (2*pi*f/c_light) * a**2) * (1 + (1+1j) * alpha * L / a * sqrt(pi/((2*pi*f/c_light) * g)))**(-1)
        wake = lambda t: (4*free_space_impedance*c_light*s00)/(pi*a**4) * heaviside(t, 0) * (1 - (1 + sqrt(c_light*t/s00)) * exp(-sqrt(c_light*t/s00)))
    else:
        print("Warning: resistive wall impedance not implemented for component {}{}. Set to zero".format(plane, exponents))
        impedance = lambda f: 0
        wake = lambda f: 0

    return Component(vectorize(impedance), vectorize(wake), plane, source_exponents=exponents[:2], test_exponents=exponents[2:])
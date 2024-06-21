from .component import Component
from .element import Element
from .interface import Layer, FlatIW2DInput, RoundIW2DInput
from .interface import component_names

from yaml import load, SafeLoader
from typing import Tuple, Dict, List, Union, Sequence, Optional, Callable
from collections import defaultdict

from numpy import (vectorize, sqrt, exp, pi, sin, cos, abs, sign,
                   inf, floor, linspace, ones, isscalar, array)
from numpy.typing import ArrayLike
import scipy.constants
from scipy import special as sp
import numpy as np

if hasattr(np, 'trapezoid'):
    trapz = np.trapezoid # numpy 2.0
else:
    trapz = np.trapz

c_light = scipy.constants.speed_of_light  # m s-1
mu0 = scipy.constants.mu_0
Z0 = mu0 * c_light
eps0 = scipy.constants.epsilon_0


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
    is_impedance = False
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
        wake = vectorize(lambda x: eval(cdict['wake'], {'exp': exp, 'sqrt': sqrt,
                                                        'pi': pi, 'x': x})) if 'wake' in cdict else None
        impedance = vectorize(lambda x: eval(cdict['impedance'], {'exp': exp, 'sqrt': sqrt,
                                                                  'pi': pi, 'x': x})) if 'impedance' in cdict else None
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


def compute_resonator_f_roi_half_width(q: float, f_r: float, f_roi_level: float = 0.5):
    aux = np.sqrt((1 - f_roi_level) / f_roi_level)

    return (aux + np.sqrt(aux**2 + 4*q**2))*f_r/(2*q) - f_r


def create_resonator_component(plane: str, exponents: Tuple[int, int, int, int],
                               r: float, q: float, f_r: float, f_roi_level: float = 0.5) -> Component:
    """
    Creates a single component object belonging to a resonator
    :param plane: the plane the component corresponds to
    :param exponents: four integers corresponding to (source_x, source_y, test_x, test_y) aka (a, b, c, d)
    :param r: the shunt impedance of the given component of the resonator
    :param q: the quality factor of the given component of the resonator
    :param f_r: the resonance frequency of the given component of the resonator
    :param f_roi_level: fraction of the peak ok the resonator which is covered by the ROI. I.e. the roi will cover
    the frequencies for which the resonator impedance is larger than f_roi_level*r
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
        wake = lambda t: (omega_r * r * exp(-omega_r * t / (2 * q)) * sin(omega_r * root_term * t) /
                          (q * root_term)).real

    if q > 1:
        d = compute_resonator_f_roi_half_width(q=q, f_r=f_r, f_roi_level=f_roi_level)
        f_rois = [(f_r - d, f_r + d)]
    else:
        f_rois = []
    # TODO: add ROI(s) for wake

    return Component(impedance, wake, plane, source_exponents=exponents[:2],
                     test_exponents=exponents[2:],
                     f_rois=f_rois)


def create_resonator_element(length: float, beta_x: float, beta_y: float,
                             rs: Dict[str, float], qs: Dict[str, float], fs: Dict[str, float],
                             f_roi_levels: Dict[str, float] = None, tag: str = 'resonator',
                             description: str = '') -> Element:
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
    :param f_roi_levels: A dictionary where the keys correspond to a plane followed by four exponents, i.e. "y0100", and
    the values give the fraction of the peak ok the resonator which is covered by the ROI. I.e. the roi will cover
    the frequencies for which the resonator impedance is larger than f_roi_level*r
    :param tag: An optional short string used to place elements into categories
    :param description: An optional short description of the element
    :return: An element object as specified by the user-input
    """
    if f_roi_levels is None:
        f_roi_levels = {}
        for key in rs.keys():
            f_roi_levels[key] = 0.5

    assert set(rs.keys()) == set(qs.keys()) == set(fs.keys()) == set(f_roi_levels.keys()), "The three input " \
                                                                                         "dictionaries describing " \
                                                                                         "the  resonator do not all " \
                                                                                         "have identical keys"
    components = []
    for key in rs.keys():
        plane, exponents = string_to_params(key, include_is_impedance=False)
        components.append(create_resonator_component(plane, exponents, rs[key], qs[key], fs[key],
                                                     f_roi_level=f_roi_levels[key]))

    return Element(length, beta_x, beta_y, components, tag=tag, description=description)


def create_many_resonators_element(length: float, beta_x: float, beta_y: float,
                                   params_dict: Dict[str, List[Dict[str, float]]], tag: str = 'resonator',
                                   description: str = '') -> Element:
    """
    Creates an element object representing many resonators.
    :param length: The length, in meters, of the element
    :param beta_x: The value of the beta function in the x-direction at the position of the element
    :param beta_y: The value of the beta function in the y-direction at the position of the element
    :param params_dict: a dictionary associating to each component a list of dictionaries containing the
    parameters of a resonator component. E.g.:
    params_dict = {
        'z0000':
            [
                {'r': 10, 'q': 10, 'f': 50, 'f_roi_level: 0.5},
                {'r': 40, 'q': 100, 'f': 60}
            ],
        'x1000':
        ...
    }
    f_roi_level is optional
    :param tag: An optional short string used to place elements into categories
    :param description: An optional short description of the element
    :return: An element object as specified by the user-input
    """
    all_components = []
    for component_id, component_params_list in params_dict.items():
        plane, exponents = string_to_params(component_id, include_is_impedance=False)
        for component_params in component_params_list:
            assert ('r' in component_params.keys() and 'q' in component_params.keys() and
                    'f' in component_params.keys()), "each of the the component dictionaries must contain r, q and f"

            f_roi_level = component_params.get('f_roi_level', 0.5)
            all_components.append(create_resonator_component(plane, exponents, component_params['r'],
                                                             component_params['q'], component_params['f'],
                                                             f_roi_level=f_roi_level))

    comp_dict = defaultdict(lambda: 0)
    for c in all_components:
        comp_dict[(c.plane, c.source_exponents, c.test_exponents)] += c

    components = sorted(comp_dict.values(), key=lambda x: (x.plane, x.source_exponents, x.test_exponents))

    return Element(length, beta_x, beta_y, components, tag=tag, description=description)


def create_classic_thick_wall_component(plane: str, exponents: Tuple[int, int, int, int],
                                        layer: Layer, radius: float) -> Component:
    """
    Creates a single component object modeling a resistive wall impedance/wake,
    based on the "classic thick wall formula" (see e.g. A. W. Chao, chap. 2 in
    "Physics of Collective Beams Instabilities in High Energy Accelerators",
    John Wiley and Sons, 1993).
    Only longitudinal and transverse dipolar impedances are supported here.
    :param plane: the plane the component corresponds to
    :param exponents: four integers corresponding to (source_x, source_y, test_x, test_y) aka (a, b, c, d)
    :param layer: the chamber material, as a wit Layer object
    :param radius: the chamber radius in m
    :return: A component object
    """

    # Material properties required for the skin depth computation are derived from the input Layer attributes
    material_resistivity = layer.dc_resistivity
    material_relative_permeability = 1. + layer.magnetic_susceptibility
    material_permeability = material_relative_permeability * mu0

    # Create the skin depth as a function offrequency and layer properties
    delta_skin = lambda f: (material_resistivity / (2*pi*abs(f) * material_permeability)) ** (1/2)

    # Longitudinal impedance and wake
    if plane == 'z' and exponents == (0, 0, 0, 0):
        impedance = lambda f: (1/2) * (1+sign(f)*1j) * material_resistivity / (pi * radius) * (1 / delta_skin(f))
        wake = lambda t: - c_light / (2*pi*radius) * (Z0 * material_resistivity/pi)**(1/2) * 1/(t**(1/2))
    # Transverse dipolar impedance
    elif (plane == 'x' and exponents == (1, 0, 0, 0)) or (plane == 'y' and exponents == (0, 1, 0, 0)):
        impedance = lambda f: ((c_light/(2*pi*f)) * (1+sign(f)*1j) *
                               material_resistivity / (pi * radius**3) *
                               (1 / delta_skin(f)))
        wake = lambda t: -c_light / (2*pi*radius**3) * (Z0 * material_resistivity/pi)**(1/2) * 1/(t**(3/2))
    else:
        print("Warning: resistive wall impedance not implemented for component {}{}. Set to zero".format(plane,
                                                                                                         exponents))
        impedance = lambda f: 0
        wake = lambda f: 0

    return Component(vectorize(impedance), vectorize(wake), plane, source_exponents=exponents[:2],
                     test_exponents=exponents[2:])


def _zlong_round_single_layer_approx(frequencies: ArrayLike, gamma: float,
                                     layer: Layer, radius: float, length: float) -> ArrayLike:
    """
    Function to compute the longitudinal resistive-wall impedance from
    the single-layer, approximated formula for a cylindrical structure,
    by E. Metral (see e.g. Eqs. 13-14 in N. Mounet and E. Metral, IPAC'10, TUPD053,
    https://accelconf.web.cern.ch/IPAC10/papers/tupd053.pdf, and
    Eq. 21 in F. Roncarolo et al, Phys. Rev. ST Accel. Beams 12, 084401, 2009,
    https://doi.org/10.1103/PhysRevSTAB.12.084401)
    :param frequencies: the frequencies (array) (in Hz)
    :param gamma: relativistic mass factor
    :param layer: a layer with material properties (only resistivity,
    relaxation time and magnetic susceptibility are taken into account
    at this stage)
    :param radius: the radius of the structure (in m)
    :param length: the total length of the resistive object (in m)
    :return: Zlong, the longitudinal impedance at these frequencies
    """
    beta = sqrt(1.-1./gamma**2)
    omega = 2*pi*frequencies
    k = omega/(beta*c_light)

    rho = layer.dc_resistivity
    tau = layer.resistivity_relaxation_time
    mu1 = 1.+layer.magnetic_susceptibility
    eps1 = 1. - 1j/(eps0*rho*omega*(1.+1j*omega*tau))
    nu = k*sqrt(1.-beta**2*eps1*mu1)

    coef_long = 1j*omega*mu0*length/(2.*pi*beta**2*gamma**2)

    x1 = k*radius/gamma
    x1sq = x1**2
    x2 = nu*radius

    zlong = coef_long * (sp.k0(x1)/sp.i0(x1) - 1./(x1sq*(1./2.+eps1*sp.kve(1, x2)/(x2*sp.kve(0, x2)))))

    return zlong


def _zdip_round_single_layer_approx(frequencies: ArrayLike, gamma: float,
                                    layer: Layer, radius: float, length: float) -> ArrayLike:
    """
    Function to compute the transverse dipolar resistive-wall impedance from
    the single-layer, approximated formula for a cylindrical structure,
    Eqs. 13-14 in N. Mounet and E. Metral, IPAC'10, TUPD053,
    https://accelconf.web.cern.ch/IPAC10/papers/tupd053.pdf, and
    Eq. 21 in F. Roncarolo et al, Phys. Rev. ST Accel. Beams 12, 084401, 2009,
    https://doi.org/10.1103/PhysRevSTAB.12.084401)
    :param frequencies: the frequencies (array) (in Hz)
    :param gamma: relativistic mass factor
    :param layer: a layer with material properties (only resistivity,
    relaxation time and magnetic susceptibility are taken into account
    at this stage)
    :param radius: the radius of the structure (in m)
    :param length: the total length of the resistive object (in m)
    :return: Zdip, the transverse dipolar impedance at these frequencies
    """
    beta = sqrt(1.-1./gamma**2)
    omega = 2*pi*frequencies
    k = omega/(beta*c_light)

    rho = layer.dc_resistivity
    tau = layer.resistivity_relaxation_time
    mu1 = 1.+layer.magnetic_susceptibility
    eps1 = 1. - 1j/(eps0*rho*omega*(1.+1j*omega*tau))
    nu = k*sqrt(1.-beta**2*eps1*mu1)

    coef_dip = 1j*k**2*Z0*length/(4.*pi*beta*gamma**4)

    x1 = k*radius/gamma
    x1sq = x1**2
    x2 = nu*radius

    zdip = coef_dip * (sp.k1(x1)/sp.i1(x1) + 4.*beta**2*gamma**2/(x1sq*(2.+x2*sp.kve(0, x2)/(mu1*sp.kve(1, x2)))))

    return zdip


def create_resistive_wall_single_layer_approx_component(plane: str, exponents: Tuple[int, int, int, int],
                                                        input_data: Union[FlatIW2DInput, RoundIW2DInput]) -> Component:
    """
    Creates a single component object modeling a resistive wall impedance,
    based on the single-layer approximated formulas by E. Metral (see e.g.
    Eqs. 13-14 in N. Mounet and E. Metral, IPAC'10, TUPD053,
    https://accelconf.web.cern.ch/IPAC10/papers/tupd053.pdf, and
    Eq. 21 in F. Roncarolo et al, Phys. Rev. ST Accel. Beams 12, 084401, 2009,
    https://doi.org/10.1103/PhysRevSTAB.12.084401)
    :param plane: the plane the component corresponds to
    :param exponents: four integers corresponding to (source_x, source_y, test_x, test_y) aka (a, b, c, d)
    :param input_data: an IW2D input object (flat or round). If the input
    is of type FlatIW2DInput and symmetric, we apply to the round formula the
    Yokoya factors for an infinitely flat structure (see e.g. K. Yokoya,
    KEK Preprint 92-196 (1993), and Part. Accel. 41 (1993) pp.221-248,
    https://cds.cern.ch/record/248630/files/p221.pdf),
    while for a single plate we use those from A. Burov and V. Danilov,
    PRL 82,11 (1999), https://doi.org/10.1103/PhysRevLett.82.2286. Other
    kinds of asymmetric structure will raise an error.
    If the input is of type RoundIW2DInput, the structure is in principle round
    but the Yokoya factors put in the input will be used.
    :return: A component object
    """
    gamma = input_data.relativistic_gamma
    length = input_data.length

    if isinstance(input_data, FlatIW2DInput):
        if len(input_data.top_layers) > 1:
            raise NotImplementedError("Input data can have only one layer")
        yok_long = 1.
        layer = input_data.top_layers[0]
        radius = input_data.top_half_gap
        if input_data.top_bottom_symmetry:
            yok_dipx = pi**2/24.
            yok_dipy = pi**2/12.
            yok_quax = -pi**2/24.
            yok_quay = pi**2/24.
        elif input_data.bottom_half_gap == inf:
            yok_dipx = 0.25
            yok_dipy = 0.25
            yok_quax = -0.25
            yok_quay = 0.25
        else:
            raise NotImplementedError("For asymmetric structures, only the case of a single plate is implemented; "
                                      "hence the bottom half gap must be infinite")
    elif isinstance(input_data, RoundIW2DInput):
        radius = input_data.inner_layer_radius
        if len(input_data.layers) > 1:
            raise NotImplementedError("Input data can have only one layer")
        layer = input_data.layers[0]
        yok_long = input_data.yokoya_factors[0]
        yok_dipx = input_data.yokoya_factors[1]
        yok_dipy = input_data.yokoya_factors[2]
        yok_quax = input_data.yokoya_factors[3]
        yok_quay = input_data.yokoya_factors[4]
    else:
        raise NotImplementedError("Input of type neither FlatIW2DInput nor RoundIW2DInput cannot be handled")

    # Longitudinal impedance
    if plane == 'z' and exponents == (0, 0, 0, 0):
        impedance = lambda f: yok_long*_zlong_round_single_layer_approx(
                                            f, gamma, layer, radius, length)
    # Transverse impedances
    elif plane == 'x' and exponents == (1, 0, 0, 0):
        impedance = lambda f: yok_dipx*_zdip_round_single_layer_approx(
                                            f, gamma, layer, radius, length)
    elif plane == 'y' and exponents == (0, 1, 0, 0):
        impedance = lambda f: yok_dipy*_zdip_round_single_layer_approx(
                                            f, gamma, layer, radius, length)
    elif plane == 'x' and exponents == (0, 0, 1, 0):
        impedance = lambda f: yok_quax*_zdip_round_single_layer_approx(
                                            f, gamma, layer, radius, length)
    elif plane == 'y' and exponents == (0, 0, 0, 1):
        impedance = lambda f: yok_quay*_zdip_round_single_layer_approx(
                                            f, gamma, layer, radius, length)
    else:
        impedance = lambda f: 0

    return Component(impedance=impedance, plane=plane, source_exponents=exponents[:2],
                     test_exponents=exponents[2:])


def create_resistive_wall_single_layer_approx_element(
            input_data: Union[FlatIW2DInput, RoundIW2DInput],
            beta_x: float, beta_y: float,
            component_ids: Sequence[str] = ('zlong', 'zxdip', 'zydip', 'zxqua', 'zyqua'),
            name: str = "", tag: str = "", description: str = "") -> Element:
    """
    Creates an element object modeling a resistive wall impedance,
    based on the single-layer approximated formulas by E. Metral (see e.g.
    Eqs. 13-14 in N. Mounet and E. Metral, IPAC'10, TUPD053,
    https://accelconf.web.cern.ch/IPAC10/papers/tupd053.pdf, and
    Eq. 21 in F. Roncarolo et al, Phys. Rev. ST Accel. Beams 12, 084401, 2009,
    https://doi.org/10.1103/PhysRevSTAB.12.084401)
    :param input_data: an IW2D input object (flat or round). If the input
    is of type FlatIW2DInput and symmetric, we apply to the round formula the
    Yokoya factors for an infinitely flat structure (see e.g. K. Yokoya,
    KEK Preprint 92-196 (1993), and Part. Accel. 41 (1993) pp.221-248,
    https://cds.cern.ch/record/248630/files/p221.pdf),
    while for a single plate we use those from A. Burov and V. Danilov,
    PRL 82,11 (1999), https://doi.org/10.1103/PhysRevLett.82.2286. Other
    kinds of asymmetric structure will raise an error.
    If the input is of type RoundIW2DInput, the structure is in principle round
    but the Yokoya factors put in the input will be used.
    :param beta_x: The beta function in the x-plane at the position of the element
    :param beta_y: The beta function in the y-plane at the position of the element
    :param component_ids: a list of components to be computed
    :param name: A user-specified name for the Element
    :param tag: A string to tag the Element
    :param description: A description for the Element
    :return: An Element object representing the structure
    """
    components = []
    length = input_data.length
    for component_id in component_ids:
        _, plane, exponents = component_names[component_id]
        components.append(create_resistive_wall_single_layer_approx_component(
                          plane, exponents, input_data))

    return Element(length=length, beta_x=beta_x, beta_y=beta_y, components=components, name=name, tag=tag,
                   description=description)


def _zlong_round_taper_RW_approx(frequencies: ArrayLike, gamma: float,
                                 layer: Layer, radius_small: float,
                                 radius_large: float, length: float,
                                 step_size: float = 1e-3) -> ArrayLike:
    """
    Function to compute the longitudinal resistive-wall impedance for a
    round taper, integrating the radius-dependent approximated formula
    for a cylindrical structure (see_zlong_round_single_layer_approx above),
    over the length of the taper.
    :param frequencies: the frequencies (array) (in Hz)
    :param gamma: relativistic mass factor
    :param layer: a layer with material properties (only resistivity,
    relaxation time and magnetic susceptibility are taken into account
    at this stage)
    :param radius_small: the smallest radius of the taper (in m)
    :param radius_large: the largest radius of the taper (in m)
    :param length: the total length of the taper (in m)
    :param step_size: the step size (in the radial direction) for the
    integration (in m)
    :return: the longitudinal impedance at these frequencies
    """
    if isscalar(frequencies):
        frequencies = array(frequencies)
    beta = sqrt(1.-1./gamma**2)
    omega = 2*pi*frequencies.reshape((-1, 1))
    k = omega/(beta*c_light)

    rho = layer.dc_resistivity
    tau = layer.resistivity_relaxation_time
    mu1 = 1.+layer.magnetic_susceptibility
    eps1 = 1. - 1j/(eps0*rho*omega*(1.+1j*omega*tau))
    nu = k*sqrt(1.-beta**2*eps1*mu1)

    coef_long = 1j*omega*mu0/(2.*pi*beta**2*gamma**2)

    npts = int(floor(abs(radius_large-radius_small)/step_size)+1)
    radii = linspace(radius_small, radius_large, npts).reshape((1, -1))
    one_array = ones(radii.shape)

    x1 = k.dot(radii)/gamma
    x1sq = x1**2
    x2 = nu.dot(radii)
    zlong = (coef_long.dot(length / float(npts) * one_array) *
             (sp.k0(x1) / sp.i0(x1) - 1. / (x1sq * (1. / 2. + eps1.dot(one_array) *
                                            sp.kve(1, x2) / (x2 * sp.kve(0, x2)))))
            )

    return trapz(zlong, axis=1)


def _zdip_round_taper_RW_approx(frequencies: ArrayLike, gamma: float,
                                layer: Layer, radius_small: float,
                                radius_large: float, length: float,
                                step_size: float = 1e-3) -> ArrayLike:
    """
    Function to compute the transverse dip. resistive-wall impedance for a
    round taper, integrating the radius-dependent approximated formula
    for a cylindrical structure (see_zdip_round_single_layer_approx above),
    over the length of the taper.
    :param frequencies: the frequencies (array) (in Hz)
    :param gamma: relativistic mass factor
    :param layer: a layer with material properties (only resistivity,
    relaxation time and magnetic susceptibility are taken into account
    at this stage)
    :param radius_small: the smallest radius of the taper (in m)
    :param radius_large: the largest radius of the taper (in m)
    :param length: the total length of the taper (in m)
    :param step_size: the step size (in the radial direction) for the
    integration (in m)
    :return: the transverse dipolar impedance at these frequencies
    """
    if isscalar(frequencies):
        frequencies = array(frequencies)
    beta = sqrt(1.-1./gamma**2)
    omega = 2*pi*frequencies.reshape((-1,1))
    k = omega/(beta*c_light)

    rho = layer.dc_resistivity
    tau = layer.resistivity_relaxation_time
    mu1 = 1.+layer.magnetic_susceptibility
    eps1 = 1. - 1j/(eps0*rho*omega*(1.+1j*omega*tau))
    nu = k*sqrt(1.-beta**2*eps1*mu1)

    coef_dip = 1j*k**2*Z0/(4.*pi*beta*gamma**4)

    npts = int(floor(abs(radius_large-radius_small)/step_size)+1)
    radii = linspace(radius_small,radius_large,npts).reshape((1,-1))
    one_array = ones(radii.shape)

    x1 = k.dot(radii)/gamma
    x1sq = x1**2
    x2 = nu.dot(radii)
    zdip = (
            coef_dip.dot(length / float(npts) * one_array) *
            (sp.k1(x1) / sp.i1(x1) + 4 * beta**2 * gamma**2 / (x1sq * (2 + x2 * sp.kve(0, x2) / (mu1 * sp.kve(1, x2)))))
           )

    return trapz(zdip, axis=1)


def create_taper_RW_approx_component(plane: str, exponents: Tuple[int, int, int, int],
                                     input_data: Union[FlatIW2DInput, RoundIW2DInput],
                                     radius_small: float, radius_large: float,
                                     step_size: float = 1e-3) -> Component:
    """
    Creates a single component object modeling a round or flat taper (flatness
    along the horizontal direction, change of half-gap along the vertical one)
    resistive-wall impedance, using the integration of the radius-dependent
    approximated formula for a cylindrical structure (see
    the above functions), over the length of the taper.
    :param plane: the plane the component corresponds to
    :param exponents: four integers corresponding to (source_x, source_y, test_x, test_y) aka (a, b, c, d)
    :param input_data: an IW2D input object (flat or round). If the input
    is of type FlatIW2DInput and symmetric, we apply to the round formula the
    Yokoya factors for an infinitely flat structure (see e.g. K. Yokoya,
    KEK Preprint 92-196 (1993), and Part. Accel. 41 (1993) pp.221-248,
    https://cds.cern.ch/record/248630/files/p221.pdf),
    while for a single plate we use those from A. Burov and V. Danilov,
    PRL 82,11 (1999), https://doi.org/10.1103/PhysRevLett.82.2286. Other
    kinds of asymmetric structure will raise an error.
    If the input is of type RoundIW2DInput, the structure is in principle
    round but the Yokoya factors put in the input will be used.
    Note that the radius or half-gaps in input_data are not used (replaced
    by the scan from radius_small to radius_large, for the integration).
    :param radius_small: the smallest radius of the taper (in m)
    :param radius_large: the largest radius of the taper (in m)
    :param step_size: the step size (in the radial or vertical direction)
    for the integration (in m)
    :return: A component object
    """
    gamma = input_data.relativistic_gamma
    length = input_data.length

    if isinstance(input_data, FlatIW2DInput):
        if len(input_data.top_layers) > 1:
            raise NotImplementedError("Input data can have only one layer")
        yok_long = 1.
        layer = input_data.top_layers[0]
        radius = input_data.top_half_gap
        if input_data.top_bottom_symmetry:
            yok_dipx = pi**2/24.
            yok_dipy = pi**2/12.
            yok_quax = -pi**2/24.
            yok_quay = pi**2/24.
        elif input_data.bottom_half_gap == inf:
            yok_dipx = 0.25
            yok_dipy = 0.25
            yok_quax = -0.25
            yok_quay = 0.25
        else:
            raise NotImplementedError("For asymmetric structures, only the case of a single plate is implemented; "
                                      "hence the bottom half gap must be infinite")
    elif isinstance(input_data, RoundIW2DInput):
        radius = input_data.inner_layer_radius
        if len(input_data.layers) > 1:
            raise NotImplementedError("Input data can have only one layer")
        layer = input_data.layers[0]
        yok_long = input_data.yokoya_factors[0]
        yok_dipx = input_data.yokoya_factors[1]
        yok_dipy = input_data.yokoya_factors[2]
        yok_quax = input_data.yokoya_factors[3]
        yok_quay = input_data.yokoya_factors[4]
    else:
        raise NotImplementedError("Input of type neither FlatIW2DInput nor RoundIW2DInput cannot be handled")

    # Longitudinal impedance
    if plane == 'z' and exponents == (0, 0, 0, 0):
        impedance = lambda f: yok_long*_zlong_round_taper_RW_approx(
                        f, gamma, layer, radius_small, radius_large,
                        length, step_size=step_size)
    # Transverse impedances
    elif plane == 'x' and exponents == (1, 0, 0, 0):
        impedance = lambda f: yok_dipx*_zdip_round_taper_RW_approx(
                        f, gamma, layer, radius_small, radius_large,
                        length, step_size=step_size)
    elif plane == 'y' and exponents == (0, 1, 0, 0):
        impedance = lambda f: yok_dipy*_zdip_round_taper_RW_approx(
                        f, gamma, layer, radius_small, radius_large,
                        length, step_size=step_size)
    elif plane == 'x' and exponents == (0, 0, 1, 0):
        impedance = lambda f: yok_quax*_zdip_round_taper_RW_approx(
                        f, gamma, layer, radius_small, radius_large,
                        length, step_size=step_size)
    elif plane == 'y' and exponents == (0, 0, 0, 1):
        impedance = lambda f: yok_quay*_zdip_round_taper_RW_approx(
                        f, gamma, layer, radius_small, radius_large,
                        length, step_size=step_size)
    else:
        impedance = lambda f: 0

    return Component(impedance=impedance, plane=plane, source_exponents=exponents[:2],
                     test_exponents=exponents[2:])


def create_taper_RW_approx_element(
            input_data: Union[FlatIW2DInput, RoundIW2DInput],
            beta_x: float, beta_y: float,
            radius_small: float, radius_large: float, step_size: float=1e-3,
            component_ids: Sequence[str] = ('zlong', 'zxdip', 'zydip', 'zxqua', 'zyqua'),
            name: str = "", tag: str = "", description: str = "") -> Element:
    """
    Creates an element object modeling a round or flat taper (flatness
    along the horizontal direction, change of half-gap along the vertical one)
    resistive-wall impedance, using the integration of the radius-dependent
    approximated formula for a cylindrical structure (see
    the above functions), over the length of the taper.
    :param input_data: an IW2D input object (flat or round). If the input
    is of type FlatIW2DInput and symmetric, we apply to the round formula the
    Yokoya factors for an infinitely flat structure (see e.g. K. Yokoya,
    KEK Preprint 92-196 (1993), and Part. Accel. 41 (1993) pp.221-248,
    https://cds.cern.ch/record/248630/files/p221.pdf),
    while for a single plate we use those from A. Burov and V. Danilov,
    PRL 82,11 (1999), https://doi.org/10.1103/PhysRevLett.82.2286. Other
    kinds of asymmetric structure will raise an error.
    If the input is of type RoundIW2DInput, the structure is in principle round
    but the Yokoya factors put in the input will be used.
    Note that the radius or half-gaps in input_data are not used (replaced
    by the scan from radius_small to radius_large, for the integration).
    :param beta_x: The beta function in the x-plane at the position of the element
    :param beta_y: The beta function in the y-plane at the position of the element
    :param radius_small: the smallest radius of the taper (in m)
    :param radius_large: the largest radius of the taper (in m)
    :param step_size: the step size (in the radial or vertical direction)
    for the integration (in m)
    :param component_ids: a list of components to be computed
    :param name: A user-specified name for the Element
    :param tag: A string to tag the Element
    :param description: A description for the Element
    :return: An Element object representing the structure
    """
    components = []
    length = input_data.length
    for component_id in component_ids:
        _, plane, exponents = component_names[component_id]
        components.append(create_taper_RW_approx_component(plane=plane, exponents=exponents, input_data=input_data,
                                                           radius_small=radius_small, radius_large=radius_large,
                                                           step_size=step_size))

    return Element(length=length, beta_x=beta_x, beta_y=beta_y, components=components, name=name, tag=tag,
                   description=description)


def create_interpolated_impedance_component(interpolation_frequencies: ArrayLike,
                                            impedance: Optional[Callable] = None,
                                            wake: Optional[Callable] = None, plane: str = '',
                                            source_exponents: Tuple[int, int] = (-1, -1),
                                            test_exponents: Tuple[int, int] = (-1, -1),
                                            name: str = "Unnamed Component",
                                            f_rois: Optional[List[Tuple[float, float]]] = None,
                                            t_rois: Optional[List[Tuple[float, float]]] = None):
    """
    Creates a component in which the impedance function is evaluated directly only on few points and it is interpolated
    everywhere else. This helps when the impedance function is very slow to evaluate.
    :param interpolation_frequencies: the frequencies where the function is evaluated for the interpolation
    :param impedance: A callable function representing the impedance function of the Component. Can be undefined if
    the wake function is defined.
    :param wake: A callable function representing the wake function of the Component. Can be undefined if
    the impedance function is defined.
    :param plane: The plane of the Component, either 'x', 'y' or 'z'. Must be specified for valid initialization
    :param source_exponents: The exponents in the x and y planes experienced by the source particle. Also
    referred to as 'a' and 'b'. Must be specified for valid initialization
    :param test_exponents: The exponents in the x and y planes experienced by the source particle. Also
    referred to as 'a' and 'b'. Must be specified for valid initialization
    :param name: An optional user-specified name of the component
    :param f_rois: a list of frequency regions of interest
    :param t_rois: a list of time regions of interest
    """
    def interpolated_impedance(x):
        return np.interp(x, interpolation_frequencies, impedance(interpolation_frequencies))

    return Component(impedance=interpolated_impedance, wake=wake, plane=plane,
                     source_exponents=source_exponents, test_exponents=test_exponents,
                     name=name, f_rois=f_rois,
                     t_rois=t_rois)

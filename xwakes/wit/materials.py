from .interface import Layer
from .utils import round_sigfigs

from pathlib import Path
import numpy as np
import json
from typing import Callable, Tuple


def layer_from_dict(thickness: float, material_dict: dict) -> Layer:
    """
    Define a layer from a dictionary containing the materials properties.

    :param thickness: layer thickness in m
    :type thickness: float
    :param material_dict: dictionary of material properties. 'dc_resistivity', 'resistivity_relaxation_time',
    're_dielectric_constant', 'magnetic_susceptibility', 'permeability_relaxation_frequency' are required
    :type material_dict: dict
    :return: Layer of the provided material
    :rtype: Layer
    """

    # Check that the provided dict has the required entries to create a Layer object
    # If not raise an AssertionError and indicate which property is missing
    required_material_properties = np.array(['dc_resistivity', 'resistivity_relaxation_time',
                                             're_dielectric_constant', 'magnetic_susceptibility',
                                             'permeability_relaxation_frequency'])
    # missing_properties_list is an array of bool. False indicates a missing key
    missing_properties_list = np.array([key not in material_dict for key in required_material_properties])

    assert not any(missing_properties_list), '{} missing from the input dictionary'.format(
            ", ".join(required_material_properties[np.asarray(missing_properties_list)]))

    return Layer(thickness=thickness,
                 dc_resistivity=material_dict['dc_resistivity'],
                 resistivity_relaxation_time=material_dict['resistivity_relaxation_time'],
                 re_dielectric_constant=material_dict['re_dielectric_constant'],
                 magnetic_susceptibility=material_dict['magnetic_susceptibility'],
                 permeability_relaxation_frequency=material_dict['permeability_relaxation_frequency'])


def layer_from_json_material_library(thickness: float, material_key: str,
                                     library_path: Path = Path(__file__).parent.joinpath('materials.json')) -> Layer:
    """
    Define a layer using the materials.json library of materials properties.

    :param thickness: layer thickness in m
    :type thickness: float
    :param material_key: material key in the materials.json file
    :type material_key: str
    :param library_path: material library path, defaults to materials.json present in wit
    :type library_path: Path, optional
    :return: Layer of the selected material
    :rtype: Layer
    """

    materials_library_json = json.loads(library_path.read_bytes())

    # Check that the material is in the provided materials library
    assert material_key in materials_library_json.keys(), f"Material {material_key} is not in library {library_path}"

    # Put the material properties in a dict
    # The decoder will have converted NaN, Infinity and -Infinity from JSON to python nan, inf and -inf
    # The entries in the materials library can contain additional fields (comment, reference) that are not
    # needed for the Layer object creation.
    material_properties_dict = materials_library_json[material_key]

    layer = layer_from_dict(thickness=thickness, material_dict=material_properties_dict)

    return layer


# Resistivity rho at B=0 vs. temperature and RRR
def rho_vs_T_Hust_Lankford(T: float, rho273K: float, RRR: float,
                           P: Tuple[float, float, float, float, float, float, float],
                           rhoc: Callable[[float], float] = lambda T: 0) -> float:
    """
    Define a resistivity versus temperature and RRR law as found in Hust & Lankford, "Thermal
    conductivity of aluminum, copper, iron, and tungsten for temperatures from 1K to the melting point", National Bureau
    of Standards, 1984, Eqs. (1.2.3) to (1.2.6) p. 8.
    Note that typically P4 given there has the wrong sign - for Cu and W at least one would otherwise
    not get the right behaviour at high temperature - most probably there is a typo in the reference.

    :param T: temperature in K
    :type T: float
    :param rho273K: resistivity at 273 K in Ohm.m
    :type rho273K: float
    :param RRR: residual resistivity ratio, i.e. rho(273K)/rho(0K) at B=0
    :type RRR: float
    :param P: tuple of the fitting coeeficient. The coefficient can be found in the reference above for copper (p. 22),
              aluminum (p. 92), iron (p. 145) and tungsten (p. 204)
    :type P: tuple
    :param rhoc: function of temperature accounting for the residual deviation from the law in Ohm.m, defaults to 0.
                 This can be used with iron and tungsten for which the residual function is provided in the reference
    :type rhoc: float
    :return: the resistivity value, in Ohm.m
    :rtype: float
    """

    # To follow the notation used in the book (coeficients from P1 to P7), we unpack the P tuple to new variables
    P1 = P[0]
    P2 = P[1]
    P3 = P[2]
    P4 = P[3]
    P5 = P[4]
    P6 = P[5]
    P7 = P[6]
    rho0 = rho273K/RRR
    rhoi = P1*T**P2 / (1. + P1*P3*T**(P2+P4)*np.exp(-(P5/T)**P6)) + rhoc(T)
    rhoi0 = P7*rhoi*rho0 / (rhoi+rho0)

    return (rho0+rhoi+rhoi0)


# magnetoresistance law (Kohler-like) drho/rho = f(B*rho_273K(B=0)/rho_T(B=0))
def magnetoresistance_Kohler(B_times_Sratio: float, P: Tuple[float, ...]) -> float:
    """
    Define a magnetoresistance law in the form drho/rho = f(B*rho_273K(B=0)/rho_T(B=0))

    :param B_times_Sratio: product of magnetic field B and Sratio = rho_273K(B=0)/rho_T(B=0) at a given temperature T, in Tesla
    :type B_times_Sratio: float
    :param P: tuple of the Kohler curve fitting coefficients. Kohler curve are represented in log-log scales: if x and y are read on the
              curve, the P coefficients come from the fitting of log10(x) and log10(y)
    :type P: tuple of floats
    :return: drho/rho the resistivity variation at a given magnetic field
    :rtype: float
    """

    if B_times_Sratio == 0. or P == (0,):
        return 0.
    else:
        return 10.**np.polyval(P, np.log10(B_times_Sratio))


def copper_at_temperature(thickness: float, T: float = 300, RRR: float = 70, B: float = 0) -> Layer:
    """
    Define a layer of pure copper material at any temperature, any B field and any RRR.
    We use a magnetoresistance law fitted from the UPPER curve of the plot in NIST, "Properties of copper and copper
    alloys at cryogenic temperatures", by Simon, Crexler and Reed, 1992 (p. 8-27, Fig. 8-14).
    The upper curve was chosen, as it is in relative agreement with C. Rathjen measurements and actual LHC beam screens
    (CERN EDMS document Nr. 329882).
    The resistivity vs. temperature and RRR is found from Hust & Lankford (see above).
    The law vs. temperature was found in good agreement with the NIST reference above (p. 8-5, Fig. 8-2).
    
    :param thickness: material thickness in m
    :type thickness: float
    :param T: temperature in K, defaults to 300
    :type T: float, optional
    :param RRR: residual resistivity ratio, i.e. rho(273K)/rho(0K) at B=0, defaults to 70
    :type RRR: float, optional
    :param B: magnetic field in T, defaults to 0
    :type B: float, optional
    :return: a Layer object
    :rtype: Layer
    """

    rho273K = 15.5*1e-9  # resistivity at 273K, in Ohm.m

    # resistivity vs temperature law coefficients, found in p. 22 of Hust and Lankford
    # Here P4 = -1.14 (instead of +1.14 in the book) to get the proper law behavior
    P = (1.171e-17, 4.49, 3.841e10, -1.14, 50., 6.428, 0.4531)

    # Coefficients for the magnetoresistance law, from Simon, Crexler and Reed (p. 8-27)
    kohler_P = (0.029497104404715, 0.905633738689341, -2.361415783729567)

    rhoDC_B0 = rho_vs_T_Hust_Lankford(T, rho273K, RRR, P)  # resistivity for B=0
    Sratio = rho_vs_T_Hust_Lankford(273, rho273K, RRR, P) / rhoDC_B0
    dc_resistivity = round_sigfigs(rhoDC_B0 * (1.+magnetoresistance_Kohler(B*Sratio, kohler_P)),3) # we round it to 3 significant digits

    # tauAC formula from Ascroft-Mermin (Z=1 for copper), Drude model (used also for other
    # materials defined above) with parameters from CRC - Handbook of Chem. and Phys.
    me = 9.10938e-31  # electron mass
    e = 1.60218e-19  # electron elementary charge
    rho_m = 9.0  # Cu volumic mass in g/cm3 (it is actually 9.02 at 4K, 8.93 at 273K,
    # see https://www.copper.org/resources/properties/cryogenic/ )
    A = 63.546  # Cu atomic mass in g/mol
    Z = 1  # number of valence electrons
    n = 6.022e23*Z*rho_m*1e6 / A
    tauAC = round_sigfigs(me / (n*dc_resistivity*e**2),3)  # relaxation time (s) (3 significant digits)

    return Layer(thickness=thickness,
                 dc_resistivity=dc_resistivity,
                 resistivity_relaxation_time=tauAC,
                 re_dielectric_constant=1,
                 magnetic_susceptibility=0,
                 permeability_relaxation_frequency=np.inf)


def tungsten_at_temperature(thickness: float, T: float = 300, RRR: float = 70, B: float = 0) -> Layer:
    """
    Define a layer of tungsten at any temperature, any B field and any RRR.
    The resistivity vs. temperature and RRR is found from Hust & Lankford (see above).
    The magnetoresistance effect is not included yet.

    :param thickness: material thickness in m
    :type thickness: float
    :param T: temperature in K, defaults to 300
    :type T: float, optional
    :param RRR: residual resistivity ratio, i.e. rho(273K)/rho(0K) at B=0, defaults to 70
    :type RRR: float, optional
    :param B: magnetic field in T, defaults to 0
    :type B: float, optional
    :return: a Layer object
    :rtype: Layer
    """
    
    rho273K = 48.4*1e-9 # resistivity at 273K, in Ohm.m

    # resistivity vs temperature law coefficients, found in p. 204 of Hust and Lankford
    # Here P4 = -1.22 (instead of +1.22 in the book) to get the proper law behavior
    P = (4.801e-16, 3.839, 1.88e10, -1.22, 55.63, 2.391, 0.0)

    # Residual equation for tungsten is found in p. 204
    # The correction is small and affects only the second or third decimal of the resistivity.
    # We must therefore increase the number of decimal to see its effect
    rhoc = lambda T: 0.7e-8 * np.log(T/560) * np.exp(-(np.log(T/1000)/0.6)**2)

    # Coefficients for the magnetoresistance law. Put to zero for now
    kohler_P = (0,)

    rhoDC_B0 = rho_vs_T_Hust_Lankford(T, rho273K, RRR, P, rhoc)  # resistivity for B=0
    Sratio = rho_vs_T_Hust_Lankford(273, rho273K, RRR, P, rhoc) / rhoDC_B0
    dc_resistivity = round_sigfigs(rhoDC_B0 * (1.+magnetoresistance_Kohler(B*Sratio, kohler_P)),3)  # 3 significant digits

    # tauAC formula from Ascroft-Mermin (Z=2 for tungsten), Drude model (used also for other
    # materials defined above) with parameters from CRC - Handbook of Chem. and Phys.
    me = 9.10938e-31 # electron mass
    e = 1.60218e-19 # electron elementary charge
    rhom =  19.3 # W volumic mass in g/cm3
    A =  183.84 # W atomic mass in g/mol
    Z = 2 # number of valence electrons
    n = 6.022e23 * Z * rhom * 1e6 / A
    tauAC = round_sigfigs(me/(n*dc_resistivity*e**2),3) # relaxation time (s) (3 significant digits)

    return Layer(thickness=thickness,
                 dc_resistivity=dc_resistivity,
                 resistivity_relaxation_time=tauAC,
                 re_dielectric_constant=1,
                 magnetic_susceptibility=0,
                 permeability_relaxation_frequency=np.inf)

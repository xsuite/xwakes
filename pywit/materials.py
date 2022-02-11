from pywit.interface import Layer


from pywit.component import Component
from pywit.element import Element

import numpy as np

def vacuum(thickness: float = np.inf) -> Layer:
    '''
    Define a layer of vacuum.
    
    :param thickness: material thickness in m
    
    :return: a Layer object
    '''
    return Layer(thickness=thickness,
        dc_resistivity=np.inf,
        resistivity_relaxation_time=0,
        re_dielectric_constant=1,
        magnetic_susceptibility=1,
        permeability_relaxation_frequency=np.inf)


def stainless_steel_316LN(thickness: float = np.inf) -> Layer:
    '''
    Define a layer of stainless steel material (316LN) as for the PS chamber.
    For material reference see also Phys. Rev. Accel. Beams 19, 041001).
    
    :param thickness: material thickness in m
    
    :return: a Layer object
    '''
    return Layer(thickness=thickness,
        dc_resistivity=75e-8,
        resistivity_relaxation_time=0,
        re_dielectric_constant=1,
        magnetic_susceptibility=1,
        permeability_relaxation_frequency=np.inf)


def copper_at_temperature(thickness: float = 50e-6, T: float = 20, RRR: float = 70, B: float = 7000/(2803.9*0.299792458)):
    '''
    Define a layer of pure copper material at any temperature, any B field
    and any RRR.
    
    We use a magnetoresistance law fitted from the UPPER curve of the plot in 
    NIST, "Properties of copper and copper alloys at cryogenic temperatures",
    by Simon, Crexler and Reed, 1992 (p. 8-27, Fig. 8-14).
    The upper curve was chosen, as it is in relative agreement with C. Rathjen
    measurements and actual LHC beam screens (CERN EDMS document Nr. 329882).
    
    The resistivity vs. temperature and RRR is found from Hust & Lankford, 
    "Thermal conductivity of aluminum, copper, iron, and tungsten fortemperatures
    from 1K to the melting point", National Bureau of Standards, 1984, 
    from Eqs. (1.2.3) to (1.2.6) p. 8, with parameters from p. 22, EXCEPT P4 
    for which we took the opposite value (to get the right behaviour at high 
    temperature - most probably there is a typo in the reference).
    
    The law vs. temperature was found in good agreement with the NIST reference
    above (p. 8-5, Fig. 8-2).
    
    :param thickness: material thickness in m
    :param T: temperature in K
    :param RRR: residual resistivity ratio, i.e. rho(273K)/rho(0K) at B=0
    :param B: magnetic field in T
    
    :return: a Layer object
    '''

    # magnetoresistance law (Kohler-like) drho/rho = f(B*rho_273K(B=0)/rho_T(B=0))
    def Kohler(BS):
        p = [0.029497104404715, 0.905633738689341, -2.361415783729567]
        if BS==0.:
            return 0.
        else:
            return 10.**np.polyval(p,np.log10(BS))
    
    # Resistivity rho at B=0 vs. temperature and RRR
    def rho_vs_T(T,RRR):
        rhoc = 0. # residual (deviation from the law - unknown, we neglect it here)
        rho273K = 15.5*1e-9 # resistivity at 273K, in Ohm.m
        P1 = 1.171e-17
        P2 = 4.49
        P3 = 3.841e10
        P4 = -1.14
        P5 = 50.
        P6 = 6.428
        P7 = 0.4531
        rho0 = rho273K/RRR
        rhoi = P1 * T**P2 / (1. + P1 * P3 * T**(P2+P4) * np.exp(-(P5/T)**P6)) + rhoc
        rhoi0 = P7 * rhoi * rho0 / (rhoi + rho0)
        return (rho0 + rhoi + rhoi0)        
    
    rhoDC_B0 = rho_vs_T(T,RRR) # resistivity for B=0
    Sratio = rho_vs_T(273.,RRR) / rhoDC_B0
    rhoDC = float('{:.3g}'.format(rhoDC_B0 * (1. + Kohler(B*Sratio)))) # we round it to 3 significant digits

    # tauAC formula from Ascroft-Mermin (Z=1 for copper), Drude model (used also for other
    # materials defined above) with parameters from CRC - Handbook of Chem. and Phys.
    me = 9.10938e-31 # electron mass
    e = 1.60218e-19 # electron elementary charge
    rhom = 9.0 # Cu volumic mass in g/cm3 (it is actually 9.02 at 4K, 8.93 at 273K,
    # see https://www.copper.org/resources/properties/cryogenic/ )
    A = 63.546 # Cu atomic mass in g/mol
    Z = 1 # number of valence electrons
    n = 6.022e23 * Z * rhom * 1e6 / A
    tauAC = float('{:.3g}'.format(me/(n*rhoDC*e**2))) # relaxation time (s) (3 significant digits)

    return Layer(thickness=thickness,
        dc_resistivity=rhoDC,
        resistivity_relaxation_time=tauAC,
        re_dielectric_constant=1,
        magnetic_susceptibility=1,
        permeability_relaxation_frequency=np.inf)
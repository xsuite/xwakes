from pywit.devices import create_tesla_cavity_component, shunt_impedance_flat_taper_stupakov_formula

import numpy as np
from pytest import mark

@mark.parametrize("plane, exponents, expected_z",
                  [ ['z', (0, 0, 0, 0), 21382.03077218+1j*26226.15240245],
                    ['x', (1, 0, 0, 0), 84292624.28074735+1j*1.03389207e+08],
                    ['y', (0, 1, 0, 0), 84292624.28074735+1j*1.03389207e+08],
                  ])
def test_create_tesla_cavity_component(plane, exponents, expected_z):
    # using the parameters from SLAC-PUB-9663, "Short-range dipole wakefields in accelerating structures for the NLC"
    period_length=8.75e-3
    a=4.92e-3
    g=6.89e-3
    freq = 1e9
    comp = create_tesla_cavity_component(period_length=period_length, a=a, g=g, plane=plane, exponents=exponents)
    assert np.isclose(np.real(expected_z), np.real(comp.impedance(freq)))
    assert np.isclose(np.imag(expected_z), np.imag(comp.impedance(freq)))


@mark.parametrize("component_id, approximate_integrals, expected_r_shunt",
                  [ ['zlong', False, 171.06327842],
                    ['zlong', True, 171.0632785],
                    ['zxdip', False, 89.89264387],
                    ['zxdip', True, 89.8927562],
                    ['zydip', False, 3047.07197441],
                    ['zydip', True, 3136.96523752],
                    ['zxqua', False, -89.89264417],
                    ['zxqua', True, -89.8927562],
                    ['zyqua', False, 89.89264417],
                    ['zyqua', True, 89.8927562],
                    ['zxcst', False, 0.],
                    ['zycst', True, 0.],
                  ])
def test_shunt_impedance_flat_taper_stupakov_formula(component_id, approximate_integrals, expected_r_shunt):
    length = 97e-3
    delta = 17.6e-3
    half_width = 350e-3
    half_gap = 25e-3

    r_shunt = shunt_impedance_flat_taper_stupakov_formula(half_gap_small=half_gap, half_gap_big=half_gap+delta,
                                                          half_width=half_width, taper_slope=delta / length,
                                                          cutoff_frequency=50e9, component_id=component_id,
                                                          approximate_integrals=approximate_integrals)
    assert np.isclose(r_shunt, expected_r_shunt)

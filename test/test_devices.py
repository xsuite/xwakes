from pywit.devices import create_tesla_cavity_component, shunt_impedance_flat_taper_stupakov_formula

import numpy as np


def test_create_tesla_cavity_component():
    ref_1ghz_zlong_re = np.array([21382.03077218])
    ref_1ghz_zlong_im = np.array([26226.15240245])
    # using the parameters from SLAC-PUB-9663, "Short-range dipole wakefields in accelerating structures for the NLC"
    zlong_comp = create_tesla_cavity_component(L=8.75e-3, a=4.92e-3, g=6.89e-3, plane='z', exponents=(0, 0, 0, 0))
    freq = np.array([1e9])
    assert np.isclose(ref_1ghz_zlong_re[0], np.real(zlong_comp.impedance(freq)[0]))
    assert np.isclose(ref_1ghz_zlong_im[0], np.imag(zlong_comp.impedance(freq)[0]))

    ref_1ghz_zxdip_re = np.array([84292624.28074735])
    ref_1ghz_zxdip_im = np.array([1.03389207e+08])
    zxdip_comp = create_tesla_cavity_component(L=8.75e-3, a=4.92e-3, g=6.89e-3, plane='x', exponents=(1, 0, 0, 0))
    freq = np.array([1e9])
    assert np.isclose(ref_1ghz_zxdip_re[0], np.real(zxdip_comp.impedance(freq)[0]))
    assert np.isclose(ref_1ghz_zxdip_im[0], np.imag(zxdip_comp.impedance(freq)[0]))

    ref_1ghz_zydip_re = np.array([84292624.28074735])
    ref_1ghz_zydip_im = np.array([1.03389207e+08])
    zydip_comp = create_tesla_cavity_component(L=8.75e-3, a=4.92e-3, g=6.89e-3, plane='y', exponents=(0, 1, 0, 0))
    freq = np.array([1e9])
    assert np.isclose(ref_1ghz_zydip_re[0], np.real(zydip_comp.impedance(freq)[0]))
    assert np.isclose(ref_1ghz_zydip_im[0], np.imag(zydip_comp.impedance(freq)[0]))


def test_shunt_impedance_flat_taper_stupakov_formula():
    length = 97e-3
    delta = 17.6e-3
    half_width = 350e-3
    half_gap = 25e-3
    r_shunt_zlong_ref = 171.06327842
    r_shunt_zlong = shunt_impedance_flat_taper_stupakov_formula(half_gap_small=half_gap, half_gap_big=half_gap+delta,
                                                                half_width=half_width, taper_slope=delta / length,
                                                                cutoff_frequency=50e9, component_id='zlong',
                                                                approximate_integrals=False)
    assert np.isclose(r_shunt_zlong, r_shunt_zlong_ref)
    r_shunt_zlong_ref = 171.0632785
    r_shunt_zlong = shunt_impedance_flat_taper_stupakov_formula(half_gap_small=half_gap, half_gap_big=half_gap+delta,
                                                                half_width=half_width, taper_slope=delta / length,
                                                                cutoff_frequency=50e9, component_id='zlong',
                                                                approximate_integrals=True)
    assert np.isclose(r_shunt_zlong, r_shunt_zlong_ref)

    r_shunt_zxdip_ref = 89.89264387
    r_shunt_zxdip = shunt_impedance_flat_taper_stupakov_formula(half_gap_small=half_gap, half_gap_big=half_gap+delta,
                                                                half_width=half_width, taper_slope=delta / length,
                                                                cutoff_frequency=50e9, component_id='zxdip',
                                                                approximate_integrals=False)
    assert np.isclose(r_shunt_zxdip, r_shunt_zxdip_ref)
    r_shunt_zxdip_ref = 89.8927562
    r_shunt_zxdip = shunt_impedance_flat_taper_stupakov_formula(half_gap_small=half_gap, half_gap_big=half_gap+delta,
                                                                half_width=half_width, taper_slope=delta / length,
                                                                cutoff_frequency=50e9, component_id='zxdip',
                                                                approximate_integrals=True)
    assert np.isclose(r_shunt_zxdip, r_shunt_zxdip_ref)

    r_shunt_zydip_ref = 3047.07197441
    r_shunt_zydip = shunt_impedance_flat_taper_stupakov_formula(half_gap_small=half_gap, half_gap_big=half_gap+delta,
                                                                half_width=half_width, taper_slope=delta / length,
                                                                cutoff_frequency=50e9, component_id='zydip',
                                                                approximate_integrals=False)
    assert np.isclose(r_shunt_zydip, r_shunt_zydip_ref)
    r_shunt_zydip_ref = 3136.96523752
    r_shunt_zydip = shunt_impedance_flat_taper_stupakov_formula(half_gap_small=half_gap, half_gap_big=half_gap+delta,
                                                                half_width=half_width, taper_slope=delta / length,
                                                                cutoff_frequency=50e9, component_id='zydip',
                                                                approximate_integrals=True)
    assert np.isclose(r_shunt_zydip, r_shunt_zydip_ref)

    r_shunt_zxquad_ref = -89.89264417
    r_shunt_zxquad = shunt_impedance_flat_taper_stupakov_formula(half_gap_small=half_gap, half_gap_big=half_gap+delta,
                                                                 half_width=half_width, taper_slope=delta / length,
                                                                 cutoff_frequency=50e9, component_id='zxqua',
                                                                 approximate_integrals=False)
    assert np.isclose(r_shunt_zxquad, r_shunt_zxquad_ref)
    r_shunt_zxquad_ref = -89.8927562
    r_shunt_zxquad = shunt_impedance_flat_taper_stupakov_formula(half_gap_small=half_gap, half_gap_big=half_gap+delta,
                                                                 half_width=half_width, taper_slope=delta / length,
                                                                 cutoff_frequency=50e9, component_id='zxqua',
                                                                 approximate_integrals=True)
    assert np.isclose(r_shunt_zxquad, r_shunt_zxquad_ref)

    r_shunt_zyquad_ref = 89.89264417
    r_shunt_zyquad = shunt_impedance_flat_taper_stupakov_formula(half_gap_small=half_gap, half_gap_big=half_gap+delta,
                                                                 half_width=half_width, taper_slope=delta / length,
                                                                 cutoff_frequency=50e9, component_id='zyqua',
                                                                 approximate_integrals=False)
    assert np.isclose(r_shunt_zyquad, r_shunt_zyquad_ref)
    r_shunt_zyquad_ref = 89.8927562
    r_shunt_zyquad = shunt_impedance_flat_taper_stupakov_formula(half_gap_small=half_gap, half_gap_big=half_gap+delta,
                                                                 half_width=half_width, taper_slope=delta / length,
                                                                 cutoff_frequency=50e9, component_id='zyqua',
                                                                 approximate_integrals=True)
    assert np.isclose(r_shunt_zyquad, r_shunt_zyquad_ref)

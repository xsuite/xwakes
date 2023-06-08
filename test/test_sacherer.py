import numpy as np
from scipy import constants as cst

from pywit.sacherer_formula import sacherer_formula
from pywit.utilities import create_resonator_component

# values obtained with the old impedance model
# (https://gitlab.cern.ch/IRIS/IW2D/-/blob/master/PYTHON_codes_and_scripts/Impedance_lib_Python/Impedance.py)
tune_shift_nx_ref = np.array([[-1.89467139e-03 + 2.00708263e-06j, -3.59042292e-05 + 4.25385266e-05j,
                               1.83556168e-03 + 2.00708264e-06j],
                              [-1.89467139e-03 + 2.00708463e-06j, -3.59042230e-05 + 4.25385269e-05j,
                               1.83556167e-03 + 2.00708463e-06j],
                              [-1.89467139e-03 + 2.00708662e-06j, -3.59042167e-05 + 4.25385272e-05j,
                               1.83556167e-03 + 2.00708663e-06j],
                              [-1.89467139e-03 + 2.00708863e-06j, -3.59042104e-05 + 4.25385275e-05j,
                               1.83556167e-03 + 2.00708863e-06j],
                              [-1.89467140e-03 + 2.00709063e-06j, -3.59042042e-05 + 4.25385278e-05j,
                               1.83556167e-03 + 2.00709064e-06j]])

tune_shift_m0_ref = (-3.5904229237583215e-05 + 4.253852659650811e-05j)

z_eff_ref = np.array([[673415.90132594 + 9916238.18255225j,
                       7136258.30545439 + 6023289.34727782j,
                       673415.90381543 + 9916238.18553716j],
                      [673416.56964372 + 9916238.98384426j,
                       7136258.35621533 + 6023288.2941135j,
                       673416.57214035 + 9916238.98683727j],
                      [673417.23974633 + 9916239.78730782j,
                       7136258.40689439 + 6023287.24166201j,
                       673417.24224903 + 9916239.79030961j],
                      [673417.91165383 + 9916240.59313587j,
                       7136258.45756425 + 6023286.19008073j,
                       673417.91416421 + 9916240.59614618j],
                      [673418.58579267 + 9916241.40106015j,
                       7136258.50840891 + 6023285.1389662j,
                       673418.58831139 + 9916241.40407765j]])


def test_sacherer_formula_function():
    impedance_func = create_resonator_component(plane='x', exponents=(1, 0, 0, 0), f_r=0.5e9, q=1, r=1e7).impedance

    def impedance_func_correct(x):
        if np.isscalar(x):
            x = np.array([x])
        ind_p = x >= 0
        ind_n = x < 0
        result = np.zeros_like(x, dtype=complex)
        if len(x[ind_p]) > 0:
            result[ind_p] = impedance_func(x[ind_p])
        if len(x[ind_n]) > 0:
            result[ind_n] = -impedance_func(np.abs(x[ind_n])).conjugate()

        return result

    bunch_intensity = 1.4E11
    bunch_length = 3E-9
    omega_rev = 2.0 * np.pi * cst.c / 26658.8832
    tune = 62.28
    gamma = 6.8E6 / cst.value('proton mass energy equivalent in MeV')
    eta = 0.0003475406157 - 1 / gamma ** 2
    rf_cavities_voltage = 12E6
    beta = np.sqrt(1. - 1. / (gamma ** 2))
    p0 = cst.m_p * beta * gamma * cst.c
    h = 35640
    q_s = np.sqrt(cst.e * rf_cavities_voltage * eta * h / (2 * np.pi * beta * cst.c * p0))
    m_max = 1
    n_bunches = 5
    nx_array = np.linspace(0, n_bunches - 1, n_bunches, dtype=int)

    tune_shift_nx, tune_shift_m0, z_eff = sacherer_formula(qp=10, nx_array=nx_array,
                                                           bunch_intensity=bunch_intensity,
                                                           omegas=q_s * omega_rev, n_bunches=n_bunches,
                                                           omega_rev=omega_rev, tune=tune, gamma=gamma, eta=eta,
                                                           bunch_length_seconds=bunch_length,
                                                           m_max=m_max, impedance_function=impedance_func_correct,
                                                           mode_type='sinusoidal')

    assert np.allclose(tune_shift_nx, tune_shift_nx_ref)
    assert np.isclose(tune_shift_m0, tune_shift_m0_ref)
    assert np.allclose(z_eff, z_eff_ref)


def test_sacherer_formula_table():
    resonator_component = create_resonator_component(plane='x', exponents=(1, 0, 0, 0), f_r=0.5e9, q=1, r=1e7)
    freq_impedance_table, impedance_table = resonator_component.impedance_to_array(10 ** 5, 1e-8, 1e15)
    # breakpoint()
    bunch_intensity = 1.4E11
    bunch_length = 3E-9
    omega_rev = 2.0 * np.pi * cst.c / 26658.8832
    tune = 62.28
    gamma = 6.8E6 / cst.value('proton mass energy equivalent in MeV')
    eta = 0.0003475406157 - 1 / gamma ** 2
    rf_cavities_voltage = 12E6
    beta = np.sqrt(1. - 1. / (gamma ** 2))
    p0 = cst.m_p * beta * gamma * cst.c
    h = 35640
    q_s = np.sqrt(cst.e * rf_cavities_voltage * eta * h / (2 * np.pi * beta * cst.c * p0))
    m_max = 1
    n_bunches = 5
    nx_array = np.linspace(0, n_bunches - 1, n_bunches, dtype=int)

    tune_shift_nx, tune_shift_m0, z_eff = sacherer_formula(qp=10, nx_array=nx_array,
                                                           bunch_intensity=bunch_intensity,
                                                           omegas=q_s * omega_rev, n_bunches=n_bunches,
                                                           omega_rev=omega_rev, tune=tune, gamma=gamma, eta=eta,
                                                           bunch_length_seconds=bunch_length,
                                                           m_max=m_max, impedance_table=impedance_table,
                                                           freq_impedance_table=freq_impedance_table,
                                                           mode_type='sinusoidal')

    assert np.allclose(tune_shift_nx, tune_shift_nx_ref)
    assert np.isclose(tune_shift_m0, tune_shift_m0_ref)
    assert np.allclose(z_eff, z_eff_ref)

import numpy
import numpy as np
from scipy import constants as cst

from pywit.sacherer_formula import sacherer_formula


def test_sacherer_formula_table():
    impedance_data = np.loadtxt('test/test_data/lhc_flattop_impedance/Zx_dip_reference.dat', skiprows=1, dtype=float)

    bunch_intensity = 1.4E11
    bunch_length = 1.1E-9
    omega_rev = 2.0 * np.pi * cst.c / 26658.8832
    tune = 62.28
    gamma = 6.8E6 / cst.value('proton mass energy equivalent in MeV')
    eta = 0.0003475406157 - 1 / gamma ** 2
    V = 12E6
    beta = np.sqrt(1. - 1. / (gamma ** 2))
    p0 = cst.m_p * beta * gamma * cst.c
    h = 35640
    q_s = np.sqrt(cst.e * V * eta * h / (2 * np.pi * beta * cst.c * p0))
    m_max = 1
    n_bunches = 5
    nx_array = np.linspace(0, n_bunches - 1, n_bunches, dtype=int)
    complex_data = impedance_data[:, 1] + 1j * impedance_data[:, 2]

    tune_shift_nx, tune_shift_m0, z_eff = sacherer_formula(qp=10, nx_array=nx_array,
                                                           bunch_intensity=bunch_intensity,
                                                           omegas=q_s * omega_rev, n_bunches=n_bunches,
                                                           omega_rev=omega_rev, tune=tune, gamma=gamma, eta=eta,
                                                           bunch_length_seconds=bunch_length,
                                                           m_max=m_max, impedance_table=complex_data,
                                                           freq_impedance_table=impedance_data[:, 0],
                                                           mode_type='sinusoidal', flag_trapz=None)

    tune_shift_nx_ref = np.array([[-0.00201982 - 1.17593889e-05j,
                                   -0.00040042 + 8.92895664e-05j,
                                   0.00171041 - 1.17597337e-05j],
                                  [-0.00201981 - 1.17806341e-05j,
                                   -0.00040036 + 8.91650444e-05j,
                                   0.00171042 - 1.17806668e-05j],
                                  [-0.00201981 - 1.17878214e-05j,
                                   -0.00040035 + 8.91226650e-05j,
                                   0.00171043 - 1.17878453e-05j],
                                  [-0.00201981 - 1.17943871e-05j,
                                   -0.00040035 + 8.90838416e-05j,
                                   0.00171043 - 1.17944148e-05j],
                                  [-0.00201981 - 1.18039769e-05j,
                                   -0.00040038 + 8.90270672e-05j,
                                   0.00171042 - 1.18040287e-05j]])

    tune_shift_m0_ref = (-0.0004003813037264127 + 8.902706715083235e-05j)
    z_eff_ref = np.array([[-1446686.05890183 + 19032104.51982384j,
                           5492376.02563156 + 24630660.07462981j,
                           -1446728.4784696 + 19032089.95301624j],
                          [-1449299.72813343 + 19030859.78223774j,
                           5484716.43386444 + 24627004.36953009j,
                           -1449303.75267617 + 19030857.99329085j],
                          [-1450183.93835416 + 19030596.99730265j,
                           5482109.59394675 + 24626231.77510942j,
                           -1450186.88299486 + 19030596.9306514j],
                          [-1450991.6851084 + 19030710.94121455j,
                           5479721.49042725 + 24626565.66937676j,
                           -1450995.08259959 + 19030711.94470993j],
                          [-1452171.45336082 + 19031275.98435284j,
                           5476229.18178966 + 24628237.78771698j,
                           -1452177.83002988 + 19031279.61551182j]])

    assert np.allclose(tune_shift_nx, tune_shift_nx_ref)
    assert np.isclose(tune_shift_m0, tune_shift_m0_ref)
    assert np.allclose(z_eff, z_eff_ref)


def test_sacherer_formula_function():
    impedance_data = np.loadtxt('test/test_data/lhc_flattop_impedance/Zx_dip_reference.dat', skiprows=1, dtype=float)

    bunch_intensity = 1.4E11
    bunch_length = 1.1E-9
    omega_rev = 2.0 * np.pi * cst.c / 26658.8832
    tune = 62.28
    gamma = 6.8E6 / cst.value('proton mass energy equivalent in MeV')
    eta = 0.0003475406157 - 1 / gamma ** 2
    V = 12E6
    beta = np.sqrt(1. - 1. / (gamma ** 2))
    p0 = cst.m_p * beta * gamma * cst.c
    h = 35640
    q_s = np.sqrt(cst.e * V * eta * h / (2 * np.pi * beta * cst.c * p0))
    m_max = 1
    n_bunches = 5
    nx_array = np.linspace(0, n_bunches - 1, n_bunches)
    complex_data = impedance_data[:, 1] + 1j * impedance_data[:, 2]

    def impedance_function(x):
        if np.isscalar(x):
            x = np.array([x])
        ind_p = x >= 0
        ind_n = x < 0
        result = np.zeros_like(x, dtype=complex)
        result[ind_p] = numpy.interp(x[ind_p], impedance_data[:, 0], complex_data)
        result[ind_n] = -numpy.interp(np.abs(x[ind_n]), impedance_data[:, 0], complex_data).conjugate()

        return result

    tune_shift_nx, tune_shift_m0, z_eff = sacherer_formula(qp=10, nx_array=nx_array,
                                                           bunch_intensity=bunch_intensity,
                                                           omegas=q_s * omega_rev, n_bunches=n_bunches,
                                                           omega_rev=omega_rev, tune=tune, gamma=gamma, eta=eta,
                                                           bunch_length_seconds=bunch_length,
                                                           m_max=m_max, impedance_function=impedance_function,
                                                           mode_type='sinusoidal', flag_trapz=None)

    tune_shift_nx_ref = np.array([[-0.00201982 - 1.17593889e-05j,
                                   -0.00040042 + 8.92895664e-05j,
                                   0.00171041 - 1.17597337e-05j],
                                  [-0.00201981 - 1.17806341e-05j,
                                   -0.00040036 + 8.91650444e-05j,
                                   0.00171042 - 1.17806668e-05j],
                                  [-0.00201981 - 1.17878214e-05j,
                                   -0.00040035 + 8.91226650e-05j,
                                   0.00171043 - 1.17878453e-05j],
                                  [-0.00201981 - 1.17943871e-05j,
                                   -0.00040035 + 8.90838416e-05j,
                                   0.00171043 - 1.17944148e-05j],
                                  [-0.00201981 - 1.18039769e-05j,
                                   -0.00040038 + 8.90270672e-05j,
                                   0.00171042 - 1.18040287e-05j]])

    tune_shift_m0_ref = (-0.0004003813037264127 + 8.902706715083235e-05j)
    z_eff_ref = np.array([[-1446686.05890183 + 19032104.51982384j,
                           5492376.02563156 + 24630660.07462981j,
                           -1446728.4784696 + 19032089.95301624j],
                          [-1449299.72813343 + 19030859.78223774j,
                           5484716.43386444 + 24627004.36953009j,
                           -1449303.75267617 + 19030857.99329085j],
                          [-1450183.93835416 + 19030596.99730265j,
                           5482109.59394675 + 24626231.77510942j,
                           -1450186.88299486 + 19030596.9306514j],
                          [-1450991.6851084 + 19030710.94121455j,
                           5479721.49042725 + 24626565.66937676j,
                           -1450995.08259959 + 19030711.94470993j],
                          [-1452171.45336082 + 19031275.98435284j,
                           5476229.18178966 + 24628237.78771698j,
                           -1452177.83002988 + 19031279.61551182j]])

    assert np.allclose(tune_shift_nx, tune_shift_nx_ref)
    assert np.isclose(tune_shift_m0, tune_shift_m0_ref)
    assert np.allclose(z_eff, z_eff_ref)

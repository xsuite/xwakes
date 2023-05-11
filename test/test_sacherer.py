import numpy as np
from scipy import constants as cst

from pywit.sacherer_formula import sacherer_formula


def test_sacherer_formula():

    impedance_data = np.loadtxt('test/test_data/Zx_dip_reference.dat', skiprows=1, dtype=float)
    
    bunch_intensity = 1.4E11
    bunch_length = 1.1E-9
    omega_rev = 2.0*np.pi*cst.c/26658.8832
    tune = 62.28
    gamma = 6.8E6/cst.value('proton mass energy equivalent in MeV')
    eta = 0.0003475406157-1/gamma**2
    V = 12E6
    beta = np.sqrt(1.-1./(gamma**2))
    p0 = cst.m_p*beta*gamma*cst.c
    h = 35640
    q_s = np.sqrt(cst.e*V*eta*h/(2*np.pi*beta*cst.c*p0))
    m_max = 0
    
    tune_shift_nx, tune_shift_m0, z_eff_new = sacherer_formula(qp=10, nx_array=np.array([0]),
                                                               bunch_intensity=bunch_intensity,
                                                               omegas=q_s*omega_rev, n_bunches=1, omega_rev=omega_rev,
                                                               tune=tune, gamma=gamma, eta=eta,
                                                               bunch_length_seconds=bunch_length,
                                                               m_max=m_max, impedance_table=(impedance_data[:, 1] +
                                                                                             1j*impedance_data[:, 2]),
                                                               freq_impedance_table=impedance_data[:, 0],
                                                               mode_type='sinusoidal', flag_trapz=None)

    tune_shift_nx_ref = [[-0.00040105+8.9222004e-05j]]
    tune_shift_m0_ref = (-0.0004010520866319997+8.922200404510191e-05j)
    z_eff_ref = [[5488220.13182791+24669498.95737255j]]

    assert np.allclose(tune_shift_nx, tune_shift_nx_ref)
    assert np.isclose(tune_shift_m0, tune_shift_m0_ref)
    assert np.allclose(z_eff_new, z_eff_ref)

test_sacherer_formula()
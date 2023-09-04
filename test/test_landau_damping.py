import pytest
import numpy as np

from pywit.landau_damping import dispersion_integral_2d, find_detuning_coeffs_threshold
from pywit.landau_damping import find_detuning_coeffs_threshold_many_tune_shifts


def test_dispersion_integral_2d():

    distribution = 'gaussian'
    tune_shift = -7.3622423693e-05-3.0188372754e-06j
    b_direct = 7.888357197059519e-05
    b_cross = -5.632222163778799e-05

    # reference value obtained with DELPHI
    expected_value = 7153.859519171599-2722.966574677259j

    assert np.isclose(np.real(expected_value), np.real(dispersion_integral_2d(tune_shift=tune_shift,
                                                                              b_direct=b_direct,
                                                                              b_cross=b_cross,
                                                                              distribution=distribution)))
    assert np.isclose(np.imag(expected_value), np.imag(dispersion_integral_2d(tune_shift=tune_shift,
                                                                              b_direct=b_direct,
                                                                              b_cross=b_cross,
                                                                              distribution=distribution)))

    distribution = 'parabolic'

    # reference value obtained with DELPHI
    expected_value = 6649.001778623455-3168.4257879737443j

    assert np.isclose(np.real(expected_value), np.real(dispersion_integral_2d(tune_shift=tune_shift,
                                                                              b_direct=b_direct,
                                                                              b_cross=b_cross,
                                                                              distribution=distribution)))
    assert np.isclose(np.imag(expected_value), np.imag(dispersion_integral_2d(tune_shift=tune_shift,
                                                                              b_direct=b_direct,
                                                                              b_cross=b_cross,
                                                                              distribution=distribution)))


def test_find_detuning_coeffs_threshold():
    # reference values obtained with the old impedance wake model https://gitlab.cern.ch/IRIS/HLLHC_IW_model
    # test positive octupole polarity
    tune_shift = -7.3622423693e-05 - 3.0188372754e-06j
    q_s = 2e-3
    b_direct = 1.980315192200037e-05
    b_cross = -1.4139287608495406e-05
    assert np.isclose(b_direct, find_detuning_coeffs_threshold(tune_shift=tune_shift, q_s=q_s, b_direct_ref=b_direct,
                                                               b_cross_ref=b_cross)[0])
    assert np.isclose(b_cross, find_detuning_coeffs_threshold(tune_shift=tune_shift, q_s=q_s, b_direct_ref=b_direct,
                                                              b_cross_ref=b_cross)[1])
    # test negative octupole polarity
    b_direct = -1.2718161244917965e-05
    b_cross = 9.08066253298482e-06

    assert np.isclose(b_direct, find_detuning_coeffs_threshold(tune_shift=tune_shift, q_s=q_s, b_direct_ref=b_direct,
                                                               b_cross_ref=b_cross)[0])
    assert np.isclose(b_cross, find_detuning_coeffs_threshold(tune_shift=tune_shift, q_s=q_s, b_direct_ref=b_direct,
                                                              b_cross_ref=b_cross)[1])


def test_find_detuning_coeffs_threshold_many_tune_shifts():
    # reference values obtained with the old impedance wake model https://gitlab.cern.ch/IRIS/HLLHC_IW_model
    tune_shift = -7.3622423693e-05 - 3.0188372754e-06j
    q_s = 2e-3
    tune_shifts = [np.nan, tune_shift, 2*tune_shift]

    # test positive octupole polarity
    b_direct = 3.960630384598084e-05
    b_cross = -2.8278575218404585e-05
    assert np.isclose(b_direct, find_detuning_coeffs_threshold_many_tune_shifts(tune_shifts=tune_shifts, q_s=q_s,
                                                                                b_direct_ref=b_direct,
                                                                                b_cross_ref=b_cross)[0])
    assert np.isclose(b_cross, find_detuning_coeffs_threshold_many_tune_shifts(tune_shifts=tune_shifts, q_s=q_s,
                                                                               b_direct_ref=b_direct,
                                                                               b_cross_ref=b_cross)[1])
    # test negative octupole polarity
    b_direct = -2.5436322491034757e-05
    b_cross = 1.816132506682559e-05
    assert np.isclose(b_direct, find_detuning_coeffs_threshold_many_tune_shifts(tune_shifts=tune_shifts, q_s=q_s,
                                                                                b_direct_ref=b_direct,
                                                                                b_cross_ref=b_cross)[0])
    assert np.isclose(b_cross, find_detuning_coeffs_threshold_many_tune_shifts(tune_shifts=tune_shifts, q_s=q_s,
                                                                               b_direct_ref=b_direct,
                                                                               b_cross_ref=b_cross)[1])

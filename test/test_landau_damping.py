import pytest
import numpy as np

from pywit.landau_damping import dispersion_integral_2d, find_octupole_threshold, find_max_i_oct_from_tune_shifts

@pytest.fixture
def tune_shift():
    return -7.3622423693e-05-3.0188372754e-06j


@pytest.fixture
def b_direct_ref():
    return 7.888357197059519e-05


@pytest.fixture
def b_cross_ref():
    return -5.632222163778799e-05


@pytest.fixture
def i_ref():
    return 550


@pytest.fixture
def q_s():
    return 2e-3


@pytest.fixture
def polarity():
    return 1


def test_dispersion_integral_2d(tune_shift, b_direct_ref, b_cross_ref):

    distribution = 'gaussian'

    # reference value obtained with DELPHI
    ref_value = 7153.859519171599-2722.966574677259j

    assert np.isclose(np.real(ref_value), np.real(dispersion_integral_2d(tune_shift, b_direct_ref, b_cross_ref,
                                                                         distribution)))
    assert np.isclose(np.imag(ref_value), np.imag(dispersion_integral_2d(tune_shift, b_direct_ref, b_cross_ref,
                                                                         distribution)))

    distribution = 'parabolic'

    # reference value obtained with DELPHI
    ref_value = 6649.001778623455-3168.4257879737443j

    assert np.isclose(np.real(ref_value), np.real(dispersion_integral_2d(tune_shift, b_direct_ref, b_cross_ref,
                                                                         distribution)))
    assert np.isclose(np.imag(ref_value), np.imag(dispersion_integral_2d(tune_shift, b_direct_ref, b_cross_ref,
                                                                         distribution)))


def test_find_octupole_threshold(tune_shift, b_direct_ref, b_cross_ref, i_ref, q_s):
    # reference value obtained with the old impedance wake model
    ref_value = 138.07353400731185

    assert np.isclose(ref_value, find_octupole_threshold(tune_shift, q_s, b_direct_ref, b_cross_ref, i_ref, polarity=1))


def test_find_max_i_oct_from_tune_shifts(tune_shift, b_direct_ref, b_cross_ref, i_ref, q_s):
    # reference value obtained with the old impedance wake model
    ref_value = 276.147068028413

    tune_shifts = [tune_shift, 2*tune_shift, np.nan]

    assert np.isclose(ref_value, find_max_i_oct_from_tune_shifts(tune_shifts, q_s, b_direct_ref, b_cross_ref, i_ref,
                                                                 polarity=1))

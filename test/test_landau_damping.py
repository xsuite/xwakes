import pytest
import numpy as np

from pywit.landau_damping import dispersion_integral_2d, find_octupole_threshold
from pywit.landau_damping import find_octupole_threshold_many_tune_shifts

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

    assert np.isclose(np.real(ref_value), np.real(dispersion_integral_2d(tune_shift=tune_shift, b_direct=b_direct_ref,
                                                                         b_cross=b_cross_ref,
                                                                         distribution=distribution)))
    assert np.isclose(np.imag(ref_value), np.imag(dispersion_integral_2d(tune_shift=tune_shift, b_direct=b_direct_ref,
                                                                         b_cross=b_cross_ref,
                                                                         distribution=distribution)))

    distribution = 'parabolic'

    # reference value obtained with DELPHI
    ref_value = 6649.001778623455-3168.4257879737443j

    assert np.isclose(np.real(ref_value), np.real(dispersion_integral_2d(tune_shift=tune_shift, b_direct=b_direct_ref,
                                                                         b_cross=b_cross_ref,
                                                                         distribution=distribution)))
    assert np.isclose(np.imag(ref_value), np.imag(dispersion_integral_2d(tune_shift=tune_shift, b_direct=b_direct_ref,
                                                                         b_cross=b_cross_ref,
                                                                         distribution=distribution)))


def test_find_octupole_threshold(tune_shift, b_direct_ref, b_cross_ref, i_ref, q_s):
    # reference value obtained with the old impedance wake model
    b_direct_expected = 1.980315192200037e-05
    b_cross_expected = -1.4139287608495406e-05
    assert np.isclose(b_direct_expected, find_octupole_threshold(tune_shift=tune_shift, q_s=q_s,
                                                                 b_direct_ref=b_direct_ref,
                                                                 b_cross_ref=b_cross_ref)[0])
    assert np.isclose(b_cross_expected, find_octupole_threshold(tune_shift=tune_shift, q_s=q_s,
                                                                b_direct_ref=b_direct_ref,
                                                                b_cross_ref=b_cross_ref)[1])
    # reference value obtained with the old impedance wake model
    b_direct_expected = -1.2718161244917965e-05
    b_cross_expected = 9.08066253298482e-06

    assert np.isclose(b_direct_expected, find_octupole_threshold(tune_shift=tune_shift, q_s=q_s,
                                                                 b_direct_ref=-b_direct_ref,
                                                                 b_cross_ref=-b_cross_ref)[0])
    assert np.isclose(b_cross_expected, find_octupole_threshold(tune_shift=tune_shift, q_s=q_s,
                                                                b_direct_ref=-b_direct_ref,
                                                                b_cross_ref=-b_cross_ref)[1])


def test_find_octupole_threshold_many_tune_shifts(tune_shift, b_direct_ref, b_cross_ref, i_ref, q_s):
    # reference value obtained with the old impedance wake model
    b_direct_expected = 3.960630384598084e-05
    b_cross_expected = -2.8278575218404585e-05

    tune_shifts = [tune_shift, 2*tune_shift, np.nan]

    assert np.isclose(b_direct_expected, find_octupole_threshold_many_tune_shifts(tune_shifts=tune_shifts, q_s=q_s,
                                                                                  b_direct_ref=b_direct_ref,
                                                                                  b_cross_ref=b_cross_ref)[0])
    assert np.isclose(b_cross_expected, find_octupole_threshold_many_tune_shifts(tune_shifts=tune_shifts, q_s=q_s,
                                                                                 b_direct_ref=b_direct_ref,
                                                                                 b_cross_ref=b_cross_ref)[1])
    # reference value obtained with the old impedance wake model
    b_direct_expected = -2.5436322491034757e-05
    b_cross_expected = 1.816132506682559e-05
    assert np.isclose(b_direct_expected, find_octupole_threshold_many_tune_shifts(tune_shifts=tune_shifts, q_s=q_s,
                                                                                  b_direct_ref=-b_direct_ref,
                                                                                  b_cross_ref=-b_cross_ref)[0])
    assert np.isclose(b_cross_expected, find_octupole_threshold_many_tune_shifts(tune_shifts=tune_shifts, q_s=q_s,
                                                                                 b_direct_ref=-b_direct_ref,
                                                                                 b_cross_ref=-b_cross_ref)[1])

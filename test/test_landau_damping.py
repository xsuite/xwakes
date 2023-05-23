import pytest
import numpy as np

from pywit.landau_damping import dispersion_integral_2d

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
def distribution():
    return 'gaussian'


def test_dispersion_integral_2d(tune_shift, b_direct_ref, b_cross_ref, distribution):
    ref_value = 7153.859519171599-2722.966574677259j
    assert np.isclose(np.real(ref_value), np.real(dispersion_integral_2d(tune_shift, b_direct_ref, b_cross_ref,
                                                                         distribution)))
    assert np.isclose(np.imag(ref_value), np.imag(dispersion_integral_2d(tune_shift, b_direct_ref, b_cross_ref,
                                                                         distribution)))
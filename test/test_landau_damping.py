import pytest
from pytest import mark, raises
import numpy as np

from pywit.landau_damping import dispersion_integral_2d, find_detuning_coeffs_threshold
from pywit.landau_damping import find_detuning_coeffs_threshold_many_tune_shifts


@mark.parametrize('distribution, expected_value',
                  [['gaussian', 7153.859519171599 - 2722.966574677259j],
                   ['parabolic', 6649.001778623455 - 3168.4257879737443j],
                   ])
def test_dispersion_integral_2d(distribution: str, expected_value: complex):
    # reference values obtained with DELPHI
    tune_shift = -7.3622423693e-05 - 3.0188372754e-06j
    b_direct = 7.888357197059519e-05
    b_cross = -5.632222163778799e-05

    test_value = dispersion_integral_2d(tune_shift=tune_shift, b_direct=b_direct, b_cross=b_cross,
                                        distribution=distribution)

    assert np.isclose(np.real(expected_value), np.real(test_value))
    assert np.isclose(np.imag(expected_value), np.imag(test_value))


def test_wrong_algo_in_find_detuning_coeffs_threshold():
    with raises(ValueError, match="algorithm must be either 'newton' or 'bisect'"):
        _ = find_detuning_coeffs_threshold(tune_shift=1e-5-1j*1e-5, q_s=2e-3,
                                           reference_b_direct=1e-15,
                                           reference_b_cross=1e-15,
                                           algorithm='newto')


@mark.parametrize('algorithm',['newton','bisect'])
@mark.parametrize('b_direct, b_cross, added_b_direct, added_b_cross',
                  [[1.980315192200037e-05, -1.4139287608495406e-05, 0, 0],
                   [-1.2718161244917965e-05, 9.08066253298482e-06, 0, 0],
                   [1.980315192200037e-05, -1.4139287608495406e-05, 1.980315192200037e-05 / 3,
                    -1.4139287608495406e-05 / 3],
                   [-1.2718161244917965e-05, 9.08066253298482e-06, -1.2718161244917965e-05 / 3,
                    9.08066253298482e-06 / 3],
                   ])
def test_find_detuning_coeffs_threshold(algorithm: str, b_direct: float,
            b_cross: float, added_b_direct: float, added_b_cross: float):
    # reference values obtained with the old impedance wake model https://gitlab.cern.ch/IRIS/HLLHC_IW_model
    # test positive octupole polarity
    tune_shift = -7.3622423693e-05 - 3.0188372754e-06j
    q_s = 2e-3

    b_direct_test, b_cross_test = find_detuning_coeffs_threshold(tune_shift=tune_shift, q_s=q_s,
                                                                 reference_b_direct=b_direct,
                                                                 reference_b_cross=b_cross,
                                                                 added_b_direct=added_b_direct,
                                                                 added_b_cross=added_b_cross,
                                                                 algorithm=algorithm)

    assert np.isclose(b_direct - added_b_direct, b_direct_test)
    assert np.isclose(b_cross - added_b_cross, b_cross_test)


@mark.parametrize('algorithm',['newton','bisect'])
@mark.parametrize('b_direct, b_cross',
                  [[3.960630384598084e-05, -2.8278575218404585e-05],
                   [-2.5436322491034757e-05, 1.816132506682559e-05],
                   ])
def test_find_detuning_coeffs_threshold_many_tune_shifts(algorithm: str, b_direct: float, b_cross: float):
    # reference values obtained with the old impedance wake model https://gitlab.cern.ch/IRIS/HLLHC_IW_model
    tune_shift = -7.3622423693e-05 - 3.0188372754e-06j
    q_s = 2e-3
    tune_shifts = [np.nan, tune_shift, 2 * tune_shift]

    b_direct_test, b_cross_test = find_detuning_coeffs_threshold_many_tune_shifts(tune_shifts=tune_shifts, q_s=q_s,
                                                                                  reference_b_direct=b_direct,
                                                                                  reference_b_cross=b_cross,
                                                                                  algorithm=algorithm)
    assert np.isclose(b_direct, b_direct_test)
    assert np.isclose(b_cross, b_cross_test)



import numpy as np
import xwakes as xw
import xobjects as xo
import pytest

from scipy.constants import c as clight

@pytest.mark.parametrize('wake_type', ['dipolar', 'quadrupolar'])
@pytest.mark.parametrize('plane', ['x', 'y'])
def test_impedances_vs_analytic_resonator_transverse(wake_type, plane):

    beta0 = 0.1

    wake = xw.WakeResonator(r=1e8, q=10, f_r=1e9,
        kind=f'{wake_type}_{plane}')

    assert len(wake.components) == 1
    assert wake.components[0].plane == plane
    assert wake.components[0].source_exponents == {
        'dipolar_x': (1, 0), 'dipolar_y': (0, 1),
        'quadrupolar_x': (0, 0), 'quadrupolar_y': (0, 0)}[f'{wake_type}_{plane}']
    assert wake.components[0].test_exponents == {
        'dipolar_x': (0, 0), 'dipolar_y': (0, 0),
        'quadrupolar_x': (1, 0), 'quadrupolar_y': (0, 1)}[f'{wake_type}_{plane}']

    z = np.linspace(-20, 20, 100000)
    t = np.linspace(-20/clight, 20/clight, 100000)

    w_vs_zeta = wake.components[0].function_vs_zeta(z, beta0=beta0, dzeta=1e-20)
    w_vs_t = wake.components[0].function_vs_t(t, beta0=beta0, dt=1e-20)

    # Assert that the function is positive at close to zero from the right
    assert wake.components[0].function_vs_t(1e-10, beta0=beta0, dt=1e-20) > 0
    assert wake.components[0].function_vs_t(-1e-10, beta0=beta0, dt=1e-20) == 0

    # Zeta has opposite sign compared to t
    assert wake.components[0].function_vs_zeta(-1e-3, beta0=beta0, dzeta=1e-20) > 0
    assert wake.components[0].function_vs_zeta(+1e-3, beta0=beta0, dzeta=1e-20) == 0

    omega = np.linspace(-10e9, 10e9, 500)

    Z_from_zeta = omega * (1 + 1j)
    Z_from_t = omega * (1 + 1j)
    dz = z[1] - z[0]
    dt = t[1] - t[0]
    for ii, oo in enumerate(omega):
        print(ii, end='\r', flush=True)
        Z_from_zeta[ii] = 1j/beta0/clight * np.sum(w_vs_zeta
                    * np.exp(1j * oo * z / beta0 / clight)) * dz
        Z_from_t[ii] = 1j * np.sum(
                    w_vs_t * np.exp(-1j * oo * t)) * dt

    Z_analytical = wake.components[0].impedance(omega/2/np.pi)

    f_test_sign = 1e9
    xo.assert_allclose(wake.components[0].impedance(-f_test_sign).real,
                    -wake.components[0].impedance(f_test_sign).real,
                    atol=0, rtol=1e-8)
    xo.assert_allclose(wake.components[0].impedance(-f_test_sign).imag,
                    wake.components[0].impedance(f_test_sign).imag,
                    atol=0, rtol=1e-8)

    xo.assert_allclose(
        Z_from_zeta, Z_analytical, rtol=0, atol=1e-4 * np.max(np.abs(Z_analytical)))
    xo.assert_allclose(
        Z_from_t, Z_analytical, rtol=0, atol=1e-4 * np.max(np.abs(Z_analytical)))


def test_impedances_vs_analytic_resonator_longitudinal():

    beta0 = 0.1

    wake = xw.WakeResonator(r=1e8, q=10, f_r=1e9,
        kind='longitudinal')

    assert len(wake.components) == 1
    assert wake.components[0].plane == 'z'
    assert wake.components[0].source_exponents == (0, 0)
    assert wake.components[0].test_exponents == (0, 0)

    # Assert that the function is positive at close to zero from the right
    assert wake.components[0].function_vs_t(1e-10, beta0=beta0, dt=1e-20) > 0
    assert wake.components[0].function_vs_t(-1e-10, beta0=beta0, dt=1e-20) == 0

    # Zeta has opposite sign compared to t
    assert wake.components[0].function_vs_zeta(-1e-3, beta0=beta0, dzeta=1e-20) > 0
    assert wake.components[0].function_vs_zeta(+1e-3, beta0=beta0, dzeta=1e-20) == 0

    z = np.linspace(-20, 20, 100000)
    t = np.linspace(-20/clight, 20/clight, 100000)

    w_vs_zeta = wake.components[0].function_vs_zeta(z, beta0=beta0, dzeta=1e-20)
    w_vs_t = wake.components[0].function_vs_t(t, beta0=beta0, dt=1e-20)

    omega = np.linspace(-10e9, 10e9, 500)

    Z_from_zeta = omega * (1 + 1j)
    Z_from_t = omega * (1 + 1j)
    dz = z[1] - z[0]
    dt = t[1] - t[0]
    for ii, oo in enumerate(omega):
        print(ii, end='\r', flush=True)
        Z_from_zeta[ii] = 1/beta0/clight * np.sum(w_vs_zeta
                    * np.exp(1j * oo * z / beta0 / clight)) * dz
        Z_from_t[ii] = np.sum(
                    w_vs_t * np.exp(-1j * oo * t)) * dt

    Z_analytical = wake.components[0].impedance(omega/2/np.pi)

    f_test_sign = 1e9
    xo.assert_allclose(wake.components[0].impedance(-f_test_sign).real,
                    wake.components[0].impedance(f_test_sign).real,
                    atol=0, rtol=1e-8)
    xo.assert_allclose(wake.components[0].impedance(-f_test_sign).imag,
                    -wake.components[0].impedance(f_test_sign).imag,
                    atol=0, rtol=1e-8)

    xo.assert_allclose(
        Z_from_zeta, Z_analytical, rtol=0, atol=1e-4 * np.max(np.abs(Z_analytical)))
    xo.assert_allclose(
        Z_from_t, Z_analytical, rtol=0, atol=1e-4 * np.max(np.abs(Z_analytical)))

    xo.assert_allclose(
        Z_from_zeta, Z_analytical, rtol=0, atol=1e-4 * np.max(np.abs(Z_analytical)))
    xo.assert_allclose(
        Z_from_t, Z_analytical, rtol=0, atol=1e-4 * np.max(np.abs(Z_analytical)))


@pytest.mark.parametrize('wake_type', ['dipolar'])
@pytest.mark.parametrize('plane', ['x', 'y'])
def test_impedances_vs_analytic_thick_res_wall_transverse(wake_type, plane):

    beta0 = 0.1

    wake = xw.WakeThickResistiveWall(
        kind=f'{wake_type}_{plane}',
        resistivity=1e-7,
        radius=1e-2
        )

    assert len(wake.components) == 1
    assert wake.components[0].plane == plane
    assert wake.components[0].source_exponents == {
        'dipolar_x': (1, 0), 'dipolar_y': (0, 1),
        'quadrupolar_x': (0, 0), 'quadrupolar_y': (0, 0)}[f'{wake_type}_{plane}']
    assert wake.components[0].test_exponents == {
        'dipolar_x': (0, 0), 'dipolar_y': (0, 0),
        'quadrupolar_x': (1, 0), 'quadrupolar_y': (0, 1)}[f'{wake_type}_{plane}']

    z = np.linspace(-50, 50, 1000000)
    t = z / beta0 / clight

    w_vs_zeta = wake.components[0].function_vs_zeta(z, beta0=beta0, dzeta=1e-4)

    w_vs_t = wake.components[0].function_vs_t(t, beta0=beta0, dt=1e-4/beta0/clight)

    # Assert that the function is positive at close to zero from the right
    assert wake.components[0].function_vs_t(1e-10, beta0=beta0, dt=1e-20) > 0
    assert wake.components[0].function_vs_t(-1e-10, beta0=beta0, dt=1e-20) == 0

    # Zeta has opposite sign compared to t
    assert wake.components[0].function_vs_zeta(-1e-3, beta0=beta0, dzeta=1e-20) > 0
    assert wake.components[0].function_vs_zeta(+1e-3, beta0=beta0, dzeta=1e-20) == 0

    omega = np.linspace(-10e9, 10e9, 80)

    Z_from_zeta = omega * (1 + 1j)
    Z_from_t = omega * (1 + 1j)
    for ii, oo in enumerate(omega):
        print(ii, end='\r', flush=True)
        Z_from_zeta[ii] = 1j/beta0/clight * np.trapezoid(w_vs_zeta
                    * np.exp(1j * oo * z / beta0 / clight), z)
        Z_from_t[ii] = 1j * np.trapezoid(
                    w_vs_t * np.exp(-1j * oo * t), t)

    Z_analytical = wake.components[0].impedance(omega/2/np.pi)

    f_test_sign = 1e9
    xo.assert_allclose(wake.components[0].impedance(-f_test_sign).real,
                    -wake.components[0].impedance(f_test_sign).real,
                    atol=0, rtol=1e-8)
    xo.assert_allclose(wake.components[0].impedance(-f_test_sign).imag,
                    wake.components[0].impedance(f_test_sign).imag,
                    atol=0, rtol=1e-8)

    xo.assert_allclose(
        Z_from_zeta, Z_analytical, rtol=0, atol=0.1 * np.max(np.abs(Z_analytical)))
    xo.assert_allclose(
        Z_from_t, Z_analytical, rtol=0, atol=0.1 * np.max(np.abs(Z_analytical)))
# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

import pytest
import pathlib

import numpy as np
import pandas as pd

import xwakes as xw
import xobjects as xo

from scipy.constants import c as clight
from scipy.interpolate import interp1d

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

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
        radius=1e-2,
        length=2.
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


    # Check that it scales with the length
    wake_len_default = xw.WakeThickResistiveWall(
        kind=f'{wake_type}_{plane}',
        resistivity=1e-7,
        radius=1e-2,
        )

    xo.assert_allclose(wake.components[0].wake(10.),
                       2 * wake_len_default.components[0].wake(10.),
                       rtol=0, atol=1e-10)


def test_impedances_vs_analytic_thick_res_wall_longitudinal():

    beta0 = 0.1

    wake = xw.WakeThickResistiveWall(
        kind='longitudinal',
        resistivity=1e-7,
        radius=1e-2,
        length=2.
        )

    assert len(wake.components) == 1
    assert wake.components[0].plane == 'z'
    assert wake.components[0].source_exponents == (0, 0)
    assert wake.components[0].test_exponents == (0, 0)

    # Assert that the function is positive at close to zero from the right
    assert wake.components[0].function_vs_t(1e-10, beta0=beta0, dt=1e-20) < 0 # unphysical, coming from asymptotic expansion
    assert wake.components[0].function_vs_t(-1e-10, beta0=beta0, dt=1e-20) == 0

    # Zeta has opposite sign compared to t
    assert wake.components[0].function_vs_zeta(-1e-3, beta0=beta0, dzeta=1e-20) < 0
    assert wake.components[0].function_vs_zeta(+1e-3, beta0=beta0, dzeta=1e-20) == 0

    z = np.linspace(-500, 500, 100000)
    t = np.linspace(-500/beta0/clight, 500/beta0/clight, 100000)

    w_vs_zeta = wake.components[0].function_vs_zeta(z, beta0=beta0, dzeta=1e-4)
    w_vs_t = wake.components[0].function_vs_t(t, beta0=beta0, dt=1e-4/beta0/clight)

    omega = np.linspace(-1e9, 1e9, 50)

    Z_from_zeta = omega * (1 + 1j)
    Z_from_t = omega * (1 + 1j)
    for ii, oo in enumerate(omega):
        print(ii, end='\r', flush=True)
        Z_from_zeta[ii] = 1/beta0/clight * np.trapezoid(w_vs_zeta
                    * np.exp(1j * oo * z / beta0 / clight), z)
        Z_from_t[ii] = np.trapezoid(
                    w_vs_t * np.exp(-1j * oo * t), t)

    Z_analytical = wake.components[0].impedance(omega/2/np.pi)

    Z_from_zeta -= (Z_from_zeta.mean() - Z_analytical.mean())
    Z_from_t -= (Z_from_t.mean() - Z_analytical.mean())

    f_test_sign = 1e9
    xo.assert_allclose(wake.components[0].impedance(-f_test_sign).real,
                    wake.components[0].impedance(f_test_sign).real,
                    atol=0, rtol=1e-8)
    xo.assert_allclose(wake.components[0].impedance(-f_test_sign).imag,
                    -wake.components[0].impedance(f_test_sign).imag,
                    atol=0, rtol=1e-8)

    xo.assert_allclose(
        Z_from_zeta, Z_analytical, rtol=0, atol=0.1 * np.max(np.abs(Z_analytical)))
    xo.assert_allclose(
        Z_from_t, Z_analytical, rtol=0, atol=0.1 * np.max(np.abs(Z_analytical)))

    xo.assert_allclose(
        Z_from_zeta, Z_analytical, rtol=0, atol=0.1 * np.max(np.abs(Z_analytical)))
    xo.assert_allclose(
        Z_from_t, Z_analytical, rtol=0, atol=0.1 * np.max(np.abs(Z_analytical)))

    # check that it scales with the length
    wake_len_default = xw.WakeThickResistiveWall(
        kind='longitudinal',
        resistivity=1e-7,
        radius=1e-2,
        )

    xo.assert_allclose(wake.components[0].wake(10.),
                          2 * wake_len_default.components[0].wake(10.),
                          rtol=0, atol=1e-10)

def test_impedances_vs_iw2d_vertical():

    beta0 = 0.1
    plane = 'y'
    wake_type = 'dipolar'

    wake_data = pd.read_csv(
        test_data_folder / 'iw2d_thick_wall/WydipWLHC_1layers10.00mm_precise.dat',
        skiprows=1, names=['z', 'wydip'], sep=' ')
    imp_data = pd.read_csv(
        test_data_folder / 'iw2d_thick_wall/ZydipWLHC_1layers10.00mm_precise.dat',
        skiprows=1, names=['f', 'ReZydip', 'ImZydip'], sep=' '
    )

    Z_first_quadrant = interp1d(imp_data['f'], imp_data['ReZydip'] + 1j * imp_data['ImZydip'])
    def Z_function(omega):
        isscalar = np.isscalar(omega)
        if isscalar:
            omega = np.array([omega])
        mask_zero = np.abs(omega) < imp_data['f'][0]
        if mask_zero.any():
            raise ValueError('Frequency too low')
        mask_positive = omega > 0
        mask_negative = omega < 0
        out = np.zeros_like(omega, dtype=complex)
        out[mask_positive] = Z_first_quadrant(omega[mask_positive])
        out[mask_negative] = Z_first_quadrant(-omega[mask_negative])
        out[mask_negative] = -np.conj(out[mask_negative])
        return out[0] if isscalar else out

    table = pd.DataFrame(
        {'time': wake_data['z'].values / clight, 'dipolar_y': wake_data['wydip'].values}
    )

    wake = xw.WakeFromTable(table=table)

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

    w_vs_zeta = wake.components[0].function_vs_zeta(z, beta0=beta0, dzeta=1e-20)
    w_vs_t = wake.components[0].function_vs_t(t, beta0=beta0, dt=1e-20)

    # Assert that the function is positive at close to zero from the right
    assert wake.components[0].function_vs_t(1e-10, beta0=beta0, dt=1e-20) > 0
    assert wake.components[0].function_vs_t(-1e-10, beta0=beta0, dt=1e-20) == 0

    # Zeta has opposite sign compared to t
    assert wake.components[0].function_vs_zeta(-1e-3, beta0=beta0, dzeta=1e-20) > 0
    assert wake.components[0].function_vs_zeta(+1e-3, beta0=beta0, dzeta=1e-20) == 0

    omega = np.linspace(-10e9, 10e9, 50)

    Z_from_zeta = omega * (1 + 1j)
    Z_from_t = omega * (1 + 1j)
    for ii, oo in enumerate(omega):
        print(ii, end='\r', flush=True)
        Z_from_zeta[ii] = 1j/beta0/clight * np.trapezoid(w_vs_zeta
                    * np.exp(1j * oo * z / beta0 / clight), z)
        Z_from_t[ii] = 1j * np.trapezoid(
                    w_vs_t * np.exp(-1j * oo * t), t)

    Z_analytical = Z_function(omega/2/np.pi)


    f_test_sign = 1e9
    xo.assert_allclose(wake.components[0].impedance(-f_test_sign).real,
                    -wake.components[0].impedance(f_test_sign).real,
                    atol=0, rtol=1e-8)
    xo.assert_allclose(wake.components[0].impedance(-f_test_sign).imag,
                    wake.components[0].impedance(f_test_sign).imag,
                    atol=0, rtol=1e-8)

    xo.assert_allclose(
        Z_from_zeta, Z_analytical, rtol=0, atol=0.05 * np.max(np.abs(Z_analytical)))
    xo.assert_allclose(
        Z_from_t, Z_analytical, rtol=0, atol=0.05 * np.max(np.abs(Z_analytical)))

def test_impedance_vs_iw2d_longitudinal():

    beta0 = 1.

    wake_data = pd.read_csv(
        test_data_folder / 'iw2d_thick_wall/WlongWLHC_1layers10.00mm_precise.dat',
        skiprows=1, names=['z', 'wlong'], sep=' ')
    imp_data = pd.read_csv(
        test_data_folder / 'iw2d_thick_wall/ZlongWLHC_1layers10.00mm_precise.dat',
        skiprows=1, names=['f', 'ReZlong', 'ImZlong'], sep=' '
    )

    Z_first_quadrant = interp1d(imp_data['f'], imp_data['ReZlong'] + 1j * imp_data['ImZlong'])
    def Z_function(omega):
        isscalar = np.isscalar(omega)
        if isscalar:
            omega = np.array([omega])
        mask_zero = np.abs(omega) < imp_data['f'][0]
        if mask_zero.any():
            raise ValueError('Frequency too low')
        mask_positive = omega >= 0
        mask_negative = omega < 0
        out = np.zeros_like(omega, dtype=complex)
        out[mask_positive] = Z_first_quadrant(omega[mask_positive])
        out[mask_negative] = Z_first_quadrant(-omega[mask_negative])
        out[mask_negative] = np.conj(out[mask_negative])
        return out[0] if isscalar else out

    table = pd.DataFrame(
        {'time': np.concatenate([[0], wake_data['z'].values / clight]),
        'longitudinal': np.concatenate([[wake_data['wlong'].values[0]], wake_data['wlong'].values])
        }
    )

    wake = xw.WakeFromTable(table=table)

    wake_thick = xw.WakeThickResistiveWall(
        kind='longitudinal',
        resistivity=1.7e-8, # Copper at room temperature
        radius=1e-2
        )

    assert len(wake.components) == 1
    assert wake.components[0].plane == 'z'
    assert wake.components[0].source_exponents == (0, 0)
    assert wake.components[0].test_exponents == (0, 0)

    # Assert that the function is positive at close to zero from the right
    assert wake.components[0].function_vs_t(1e-15, beta0=beta0, dt=1e-20) > 0
    assert wake.components[0].function_vs_t(-1e-15, beta0=beta0, dt=1e-20) == 0

    # Zeta has opposite sign compared to t
    assert wake.components[0].function_vs_zeta(-1e-5, beta0=beta0, dzeta=1e-20) > 0
    assert wake.components[0].function_vs_zeta(+1e-5, beta0=beta0, dzeta=1e-20) == 0

    z_positive = wake_data['z'].values
    z_negative = -wake_data['z'].values[::-1]
    z = np.concatenate((z_negative, z_positive))
    t = -z[::-1] / clight

    w_vs_zeta = wake.components[0].function_vs_zeta(z, beta0=beta0, dzeta=1e-200)
    w_vs_t = wake.components[0].function_vs_t(t, beta0=beta0, dt=1e-200/beta0/clight)
    w_thick_vs_t = wake_thick.components[0].function_vs_t(t, beta0=beta0, dt=1e-200/beta0/clight)

    omega = np.linspace(-10e9, 10e9, 500)

    Z_from_zeta = omega * (1 + 1j)
    Z_from_t = omega * (1 + 1j)
    Z_from_zeta_thick = omega * (1 + 1j)
    Z_from_t_thick = omega * (1 + 1j)
    dz = z[1] - z[0]
    dt = t[1] - t[0]
    for ii, oo in enumerate(omega):
        print(ii, end='\r', flush=True)
        Z_from_zeta[ii] = 1/beta0/clight * np.trapezoid(w_vs_zeta
                    * np.exp(1j * oo * (z / beta0 / clight)) , z)
        Z_from_t[ii] = np.trapezoid(
                    w_vs_t * np.exp(-1j * oo * t), t)
        Z_from_zeta_thick[ii] = 1/beta0/clight * np.trapezoid(w_vs_zeta
                    * np.exp(1j * oo * (z / beta0 / clight)) , z)
        Z_from_t_thick[ii] = np.trapezoid(
                    w_thick_vs_t * np.exp(-1j * oo * t), t)
    Z_analytical = Z_function(omega/2/np.pi)

    Z_thick_wall = wake_thick.components[0].impedance(omega/2/np.pi)

    Z_from_zeta -= (Z_from_zeta.mean() - Z_analytical.mean())
    Z_from_t -= (Z_from_t.mean() - Z_analytical.mean())
    Z_from_zeta_thick -= (Z_from_zeta_thick.mean() - Z_thick_wall.mean())
    Z_from_t_thick -= (Z_from_t_thick.mean() - Z_thick_wall.mean())

    f_test_sign = 1e9
    xo.assert_allclose(wake.components[0].impedance(-f_test_sign).real,
                    wake.components[0].impedance(f_test_sign).real,
                    atol=0, rtol=1e-8)
    xo.assert_allclose(wake.components[0].impedance(-f_test_sign).imag,
                    -wake.components[0].impedance(f_test_sign).imag,
                    atol=0, rtol=1e-8)

    xo.assert_allclose(
        Z_from_zeta, Z_analytical, rtol=0, atol=0.1 * np.max(np.abs(Z_analytical)))
    xo.assert_allclose(
        Z_from_t, Z_analytical, rtol=0, atol=0.1 * np.max(np.abs(Z_analytical)))

    mask_check_thick = (t > 1e-12) & (t < 2e-11)
    xo.assert_allclose(
        w_vs_t[mask_check_thick], w_thick_vs_t[mask_check_thick], rtol=0.05, atol=0)

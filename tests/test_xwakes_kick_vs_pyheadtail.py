# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

import pathlib

import numpy as np
import pandas as pd
from scipy.constants import c as clight

import xwakes as xw
import xtrack as xt
import xobjects as xo

test_data_folder = pathlib.Path(__file__).parent.joinpath(
    '../test_data').absolute()

def test_xwakes_kick_vs_pyheadtail_table_dipolar():

    from xpart.pyheadtail_interface.pyhtxtparticles import PyHtXtParticles

    p = xt.Particles(p0c=7e12, zeta=np.linspace(-1, 1, 100000))
    p.x[p.zeta > 0] += 1e-3
    p.y[p.zeta > 0] += 1e-3
    p_table = p.copy()
    p_ref = p.copy()
    p_ref = PyHtXtParticles.from_dict(p_ref.to_dict())

    # Build equivalent WakeFromTable
    table = xw.read_headtail_file(
        test_data_folder / 'headtail_format_table_hllhc/HLLHC_wake_flattop_nocrab.dat',
        wake_file_columns=[
                        'time', 'longitudinal', 'dipolar_x', 'dipolar_y',
                        'quadrupolar_x', 'quadrupolar_y', 'dipolar_xy',
                        'quadrupolar_xy', 'dipolar_yx', 'quadrupolar_yx',
                        'constant_x', 'constant_y'])
    wake_from_table = xw.WakeFromTable(table, columns=['dipolar_x', 'dipolar_y'])
    wake_from_table.configure_for_tracking(zeta_range=(-1, 1), num_slices=100)

    # Zotter convention
    assert table['dipolar_x'].values[1] > 0
    assert table['dipolar_y'].values[1] > 0

    assert len(wake_from_table.components) == 2
    assert wake_from_table.components[0].plane == 'x'
    assert wake_from_table.components[0].source_exponents == (1, 0)
    assert wake_from_table.components[0].test_exponents == (0, 0)
    assert wake_from_table.components[1].plane == 'y'
    assert wake_from_table.components[1].source_exponents == (0, 1)
    assert wake_from_table.components[1].test_exponents == (0, 0)

    # Assert that the function is positive at close to zero from the right
    assert wake_from_table.components[0].function_vs_t(1e-10, beta0=1, dt=1e-20) > 0
    assert wake_from_table.components[0].function_vs_t(-1e-10, beta0=1, dt=1e-20) == 0
    assert wake_from_table.components[1].function_vs_t(1e-10, beta0=1, dt=1e-20) > 0
    assert wake_from_table.components[1].function_vs_t(-1e-10, beta0=1, dt=1e-20) == 0

    # Zeta has opposite sign compared to t
    assert wake_from_table.components[0].function_vs_zeta(-1e-3, beta0=1, dzeta=1e-20) > 0
    assert wake_from_table.components[0].function_vs_zeta(+1e-3, beta0=1, dzeta=1e-20) == 0
    assert wake_from_table.components[1].function_vs_zeta(-1e-3, beta0=1, dzeta=1e-20) > 0
    assert wake_from_table.components[1].function_vs_zeta(+1e-3, beta0=1, dzeta=1e-20) == 0

    from PyHEADTAIL.impedances.wakes import WakeTable, WakeField
    from PyHEADTAIL.particles.slicing import UniformBinSlicer
    slicer = UniformBinSlicer(n_slices=None,
                            z_sample_points=wake_from_table._wake_tracker.slicer.zeta_centers)
    waketable = WakeTable(
        test_data_folder / 'headtail_format_table_hllhc/HLLHC_wake_dip.dat',
        ['time', 'dipole_x', 'dipole_y'])
    wake_pyht = WakeField(slicer, waketable)

    wake_from_table.track(p_table)
    wake_pyht.track(p_ref)

    xo.assert_allclose(p_table.px, p_ref.px, atol=1e-30, rtol=2e-3)
    xo.assert_allclose(p_table.py, p_ref.py, atol=1e-30, rtol=2e-3)

def test_xwakes_kick_vs_pyheadtail_table_quadrupolar():

    from xpart.pyheadtail_interface.pyhtxtparticles import PyHtXtParticles

    p = xt.Particles(p0c=7e12, zeta=np.linspace(-1, 1, 100000))
    p.x[p.zeta > 0] += 1e-3
    p.y[p.zeta > 0] += 1e-3
    p_table = p.copy()
    p_ref = p.copy()
    p_ref = PyHtXtParticles.from_dict(p_ref.to_dict())

    # Build equivalent WakeFromTable
    table = xw.read_headtail_file(
        test_data_folder / 'headtail_format_table_hllhc/HLLHC_wake_flattop_nocrab.dat',
        wake_file_columns=[
                        'time', 'longitudinal', 'dipolar_x', 'dipolar_y',
                        'quadrupolar_x', 'quadrupolar_y', 'dipolar_xy',
                        'quadrupolar_xy', 'dipolar_yx', 'quadrupolar_yx',
                        'constant_x', 'constant_y'])

    wake_from_table = xw.WakeFromTable(table, columns=['quadrupolar_x', 'quadrupolar_y'])
    wake_from_table.configure_for_tracking(zeta_range=(-1, 1), num_slices=100)

    # This is specific of this table
    assert table['quadrupolar_x'].values[1] < 0
    assert table['quadrupolar_y'].values[1] < 0

    assert len(wake_from_table.components) == 2
    assert wake_from_table.components[0].plane == 'x'
    assert wake_from_table.components[0].source_exponents == (0, 0)
    assert wake_from_table.components[0].test_exponents == (1, 0)
    assert wake_from_table.components[1].plane == 'y'
    assert wake_from_table.components[1].source_exponents == (0, 0)
    assert wake_from_table.components[1].test_exponents == (0, 1)

    # Assert that the function is consistent with table
    assert wake_from_table.components[0].function_vs_t(1e-10, beta0=1, dt=1e-20) < 0
    assert wake_from_table.components[0].function_vs_t(-1e-10, beta0=1, dt=1e-20) == 0
    assert wake_from_table.components[1].function_vs_t(1e-10, beta0=1, dt=1e-20) < 0
    assert wake_from_table.components[1].function_vs_t(-1e-10, beta0=1, dt=1e-20) == 0

    # Zeta has opposite sign compared to t
    assert wake_from_table.components[0].function_vs_zeta(-1e-3, beta0=1, dzeta=1e-20) < 0
    assert wake_from_table.components[0].function_vs_zeta(+1e-3, beta0=1, dzeta=1e-20) == 0
    assert wake_from_table.components[1].function_vs_zeta(-1e-3, beta0=1, dzeta=1e-20) < 0
    assert wake_from_table.components[1].function_vs_zeta(+1e-3, beta0=1, dzeta=1e-20) == 0


    from PyHEADTAIL.impedances.wakes import WakeTable, WakeField
    from PyHEADTAIL.particles.slicing import UniformBinSlicer
    slicer = UniformBinSlicer(n_slices=None,
                            z_sample_points=wake_from_table._wake_tracker.slicer.zeta_centers)
    waketable = WakeTable(
        test_data_folder / 'headtail_format_table_hllhc/HLLHC_wake_quad.dat',
        ['time', 'quadrupole_x', 'quadrupole_y'])
    wake_pyht = WakeField(slicer, waketable)

    wake_from_table.track(p_table)
    wake_pyht.track(p_ref)

    xo.assert_allclose(p_table.px, p_ref.px, atol=1e-30, rtol=2e-3)
    xo.assert_allclose(p_table.py, p_ref.py, atol=1e-30, rtol=2e-3)


def test_xwakes_kick_vs_pyheadtail_table_longitudinal():

    from xpart.pyheadtail_interface.pyhtxtparticles import PyHtXtParticles

    p = xt.Particles.merge([
        xt.Particles(p0c=7e12, zeta=np.linspace(-1e-3, 1e-3, 100000)),
        xt.Particles(p0c=7e12, zeta=1e-6+np.zeros(100000))
    ])

    p_table = p.copy()
    p_ref = p.copy()
    p_ref = PyHtXtParticles.from_dict(p_ref.to_dict())

    # Build equivalent WakeFromTable
    table = xw.read_headtail_file(
        test_data_folder / 'headtail_format_table_hllhc/HLLHC_wake_flattop_nocrab.dat',
        wake_file_columns=[
                        'time', 'longitudinal', 'dipolar_x', 'dipolar_y',
                        'quadrupolar_x', 'quadrupolar_y', 'dipolar_xy',
                        'quadrupolar_xy', 'dipolar_yx', 'quadrupolar_yx',
                        'constant_x', 'constant_y'])
    wake_from_table = xw.WakeFromTable(table, columns=['time', 'longitudinal'])
    wake_from_table.configure_for_tracking(zeta_range=(-2e-3, 2e-3), num_slices=1000)

    assert len(wake_from_table.components) == 1
    assert wake_from_table.components[0].plane == 'z'
    assert wake_from_table.components[0].source_exponents == (0, 0)
    assert wake_from_table.components[0].test_exponents == (0, 0)

    # Assert that the function is positive at close to zero from the right
    # (this wake starts with very-high frequency oscillations)
    assert wake_from_table.components[0].function_vs_t(2e-14, beta0=1, dt=1e-20) > 0
    assert wake_from_table.components[0].function_vs_t(-2e-14, beta0=1, dt=1e-20) == 0

    # Zeta has opposite sign compared to t
    assert wake_from_table.components[0].function_vs_zeta(-1e-5, beta0=1, dzeta=1e-20) > 0
    assert wake_from_table.components[0].function_vs_zeta(+1e-5, beta0=1, dzeta=1e-20) == 0

    assert table['longitudinal'].values[1] > 0

    from PyHEADTAIL.impedances.wakes import WakeTable, WakeField
    from PyHEADTAIL.particles.slicing import UniformBinSlicer
    slicer = UniformBinSlicer(n_slices=None,
                            z_sample_points=wake_from_table._wake_tracker.slicer.zeta_centers)
    waketable = WakeTable(
        test_data_folder / 'headtail_format_table_hllhc/HLLHC_wake_long.dat',
        ['time', 'longitudinal'])
    wake_pyht = WakeField(slicer, waketable)

    wake_from_table.track(p_table)
    wake_pyht.track(p_ref)

    assert np.max(p_ref.delta) > 1e-12
    xo.assert_allclose(p_table.delta, p_ref.delta, atol=1e-14, rtol=0)

def test_xwakes_kick_vs_pyheadtail_resonator_dipolar():

    from xpart.pyheadtail_interface.pyhtxtparticles import PyHtXtParticles

    p = xt.Particles(p0c=7e12, zeta=np.linspace(-1, 1, 100000),
                    weight=1e14)
    p.x[p.zeta > 0] += 1e-3
    p.y[p.zeta > 0] += 1e-3
    p_table = p.copy()
    p_ref = p.copy()
    p_ref = PyHtXtParticles.from_dict(p_ref.to_dict())

    wake = xw.WakeResonator(
        r=1e8, q=1e7, f_r=1e9,
        kind=xw.Yokoya('circular'), # equivalent to: kind=['dipolar_x', 'dipolar_y'],
    )
    wake.configure_for_tracking(zeta_range=(-1, 1), num_slices=50)

    assert len(wake.components) == 2
    assert wake.components[0].plane == 'x'
    assert wake.components[0].source_exponents == (1, 0)
    assert wake.components[0].test_exponents == (0, 0)
    assert wake.components[1].plane == 'y'
    assert wake.components[1].source_exponents == (0, 1)
    assert wake.components[1].test_exponents == (0, 0)

    # Assert that the function is positive at close to zero from the right
    assert wake.components[0].function_vs_t(1e-10, beta0=1, dt=1e-20) > 0
    assert wake.components[0].function_vs_t(-1e-10, beta0=1, dt=1e-20) == 0
    assert wake.components[1].function_vs_t(1e-10, beta0=1, dt=1e-20) > 0
    assert wake.components[1].function_vs_t(-1e-10, beta0=1, dt=1e-20) == 0

    # Zeta has opposite sign compared to t
    assert wake.components[0].function_vs_zeta(-1e-3, beta0=1, dzeta=1e-20) > 0
    assert wake.components[0].function_vs_zeta(+1e-3, beta0=1, dzeta=1e-20) == 0
    assert wake.components[1].function_vs_zeta(-1e-3, beta0=1, dzeta=1e-20) > 0
    assert wake.components[1].function_vs_zeta(+1e-3, beta0=1, dzeta=1e-20) == 0


    # Build equivalent WakeFromTable
    t_samples = np.linspace(-10/clight, 10/clight, 100000)
    w_dipole_x_samples = wake.components[0].function_vs_t(t_samples, beta0=1., dt=1e-20)
    w_dipole_y_samples = wake.components[1].function_vs_t(t_samples, beta0=1., dt=1e-20)
    table = pd.DataFrame({'time': t_samples, 'dipolar_x': w_dipole_x_samples,
                            'dipolar_y': w_dipole_y_samples})
    wake_from_table = xw.WakeFromTable(table)
    wake_from_table.configure_for_tracking(zeta_range=(-1, 1), num_slices=50)

    assert len(wake_from_table.components) == 2
    assert wake_from_table.components[0].plane == 'x'
    assert wake_from_table.components[0].source_exponents == (1, 0)
    assert wake_from_table.components[0].test_exponents == (0, 0)
    assert wake_from_table.components[1].plane == 'y'
    assert wake_from_table.components[1].source_exponents == (0, 1)
    assert wake_from_table.components[1].test_exponents == (0, 0)

    # Assert that the function is positive at close to zero from the right
    assert wake_from_table.components[0].function_vs_t(1e-10, beta0=1, dt=1e-20) > 0
    assert wake_from_table.components[0].function_vs_t(-1e-10, beta0=1, dt=1e-20) == 0
    assert wake_from_table.components[1].function_vs_t(1e-10, beta0=1, dt=1e-20) > 0
    assert wake_from_table.components[1].function_vs_t(-1e-10, beta0=1, dt=1e-20) == 0

    # Zeta has opposite sign compared to t
    assert wake_from_table.components[0].function_vs_zeta(-1e-3, beta0=1, dzeta=1e-20) > 0
    assert wake_from_table.components[0].function_vs_zeta(+1e-3, beta0=1, dzeta=1e-20) == 0
    assert wake_from_table.components[1].function_vs_zeta(-1e-3, beta0=1, dzeta=1e-20) > 0
    assert wake_from_table.components[1].function_vs_zeta(+1e-3, beta0=1, dzeta=1e-20) == 0

    from PyHEADTAIL.impedances.wakes import CircularResonator, WakeField
    from PyHEADTAIL.particles.slicing import UniformBinSlicer
    res_pyht = CircularResonator(R_shunt=1e8, Q=1e7, frequency=1e9)
    slicer = UniformBinSlicer(n_slices=None, z_sample_points=wake._wake_tracker.slicer.zeta_centers)
    wake_pyht = WakeField(slicer, res_pyht)

    import xpart as xp

    wake.track(p)
    wake_from_table.track(p_table)
    wake_pyht.track(p_ref)

    assert np.max(np.abs(p_ref.px)) > 0
    xo.assert_allclose(p.px, p_ref.px, rtol=0, atol=0.5e-3*np.max(np.abs(p_ref.px)))
    xo.assert_allclose(p.py, p_ref.py, rtol=0, atol=0.5e-3*np.max(np.abs(p_ref.py)))
    xo.assert_allclose(p_table.px, p_ref.px, rtol=0, atol=2e-3*np.max(np.abs(p_ref.px)))
    xo.assert_allclose(p_table.py, p_ref.py, rtol=0, atol=2e-3*np.max(np.abs(p_ref.py)))

def test_xwakes_kick_vs_pyheadtail_resonator_quadrupolar():

    from xpart.pyheadtail_interface.pyhtxtparticles import PyHtXtParticles

    p = xt.Particles(p0c=7e12, zeta=np.linspace(-1, 1, 100000),
                    weight=1e14)
    p.x[p.zeta > 0] += 1e-3
    p.y[p.zeta > 0] += 1e-3
    p_table = p.copy()
    p_ref = p.copy()
    p_ref = PyHtXtParticles.from_dict(p_ref.to_dict())

    wake = xw.WakeResonator(
        r=1e8, q=1e7, f_r=1e9,
        kind=['quadrupolar_x', 'quadrupolar_y'],
    )
    wake.configure_for_tracking(zeta_range=(-1, 1), num_slices=50)

    assert len(wake.components) == 2
    assert wake.components[0].plane == 'x'
    assert wake.components[0].source_exponents == (0, 0)
    assert wake.components[0].test_exponents == (1, 0)
    assert wake.components[1].plane == 'y'
    assert wake.components[1].source_exponents == (0, 0)
    assert wake.components[1].test_exponents == (0, 1)

    # Assert that the function is positive at close to zero from the right
    assert wake.components[0].function_vs_t(1e-10, beta0=1, dt=1e-20) > 0
    assert wake.components[0].function_vs_t(-1e-10, beta0=1, dt=1e-20) == 0
    assert wake.components[1].function_vs_t(1e-10, beta0=1, dt=1e-20) > 0
    assert wake.components[1].function_vs_t(-1e-10, beta0=1, dt=1e-20) == 0

    # Zeta has opposite sign compared to t
    assert wake.components[0].function_vs_zeta(-1e-3, beta0=1, dzeta=1e-20) > 0
    assert wake.components[0].function_vs_zeta(+1e-3, beta0=1, dzeta=1e-20) == 0
    assert wake.components[1].function_vs_zeta(-1e-3, beta0=1, dzeta=1e-20) > 0
    assert wake.components[1].function_vs_zeta(+1e-3, beta0=1, dzeta=1e-20) == 0

    # Build equivalent WakeFromTable
    t_samples = np.linspace(-10/clight, 10/clight, 100000)
    w_quadrupole_x_samples = wake.components[0].function_vs_t(t_samples, beta0=1., dt=1e-20)
    w_quadrupole_y_samples = wake.components[1].function_vs_t(t_samples, beta0=1., dt=1e-20)
    table = pd.DataFrame({'time': t_samples, 'quadrupolar_x': w_quadrupole_x_samples,
                            'quadrupolar_y': w_quadrupole_y_samples})
    wake_from_table = xw.WakeFromTable(table)
    wake_from_table.configure_for_tracking(zeta_range=(-1, 1), num_slices=50)

    assert len(wake_from_table.components) == 2
    assert wake_from_table.components[0].plane == 'x'
    assert wake_from_table.components[0].source_exponents == (0, 0)
    assert wake_from_table.components[0].test_exponents == (1, 0)
    assert wake_from_table.components[1].plane == 'y'
    assert wake_from_table.components[1].source_exponents == (0, 0)
    assert wake_from_table.components[1].test_exponents == (0, 1)

    # Assert that the function is positive at close to zero from the right
    assert wake_from_table.components[0].function_vs_t(1e-10, beta0=1, dt=1e-20) > 0
    assert wake_from_table.components[0].function_vs_t(-1e-10, beta0=1, dt=1e-20) == 0
    assert wake_from_table.components[1].function_vs_t(1e-10, beta0=1, dt=1e-20) > 0
    assert wake_from_table.components[1].function_vs_t(-1e-10, beta0=1, dt=1e-20) == 0

    # Zeta has opposite sign compared to t
    assert wake_from_table.components[0].function_vs_zeta(-1e-3, beta0=1, dzeta=1e-20) > 0
    assert wake_from_table.components[0].function_vs_zeta(+1e-3, beta0=1, dzeta=1e-20) == 0
    assert wake_from_table.components[1].function_vs_zeta(-1e-3, beta0=1, dzeta=1e-20) > 0
    assert wake_from_table.components[1].function_vs_zeta(+1e-3, beta0=1, dzeta=1e-20) == 0

    from PyHEADTAIL.impedances.wakes import Resonator, WakeField
    from PyHEADTAIL.particles.slicing import UniformBinSlicer
    res_pyht = Resonator(R_shunt=1e8, Q=1e7, frequency=1e9,
                        Yokoya_X1=0, Yokoya_Y1=0,
                        Yokoya_X2=1, Yokoya_Y2=1,
                        switch_Z=False)
    slicer = UniformBinSlicer(n_slices=None, z_sample_points=wake._wake_tracker.slicer.zeta_centers)
    wake_pyht = WakeField(slicer, res_pyht)

    import xpart as xp

    wake.track(p)
    wake_from_table.track(p_table)
    wake_pyht.track(p_ref)

    assert np.max(np.abs(p_ref.px)) > 0
    xo.assert_allclose(p.px, p_ref.px, rtol=0, atol=0.5e-3*np.max(np.abs(p_ref.px)))
    xo.assert_allclose(p.py, p_ref.py, rtol=0, atol=0.5e-3*np.max(np.abs(p_ref.py)))
    xo.assert_allclose(p_table.px, p_ref.px, rtol=0, atol=2e-3*np.max(np.abs(p_ref.px)))
    xo.assert_allclose(p_table.py, p_ref.py, rtol=0, atol=2e-3*np.max(np.abs(p_ref.py)))

def test_xwakes_kick_vs_pyheadtail_resonator_longitudinal():

    from xpart.pyheadtail_interface.pyhtxtparticles import PyHtXtParticles

    p = xt.Particles(p0c=7e12, zeta=np.linspace(-1, 1, 100000),
                    weight=1e14)
    p.x[p.zeta > 0] += 1e-3
    p.y[p.zeta > 0] += 1e-3
    p_table = p.copy()
    p_ref = p.copy()
    p_ref = PyHtXtParticles.from_dict(p_ref.to_dict())

    wake = xw.WakeResonator(
        r=1e8, q=1e7, f_r=1e9,
        kind='longitudinal'
    )
    wake.configure_for_tracking(zeta_range=(-1.01, 1.01), num_slices=50)

    assert len(wake.components) == 1
    assert wake.components[0].plane == 'z'
    assert wake.components[0].source_exponents == (0, 0)
    assert wake.components[0].test_exponents == (0, 0)

    # Assert that the function is positive at close to zero from the right
    assert wake.components[0].function_vs_t(1e-10, beta0=1, dt=1e-20) > 0
    assert wake.components[0].function_vs_t(-1e-10, beta0=1, dt=1e-20) == 0

    # Zeta has opposite sign compared to t
    assert wake.components[0].function_vs_zeta(-1e-3, beta0=1, dzeta=1e-20) > 0
    assert wake.components[0].function_vs_zeta(+1e-3, beta0=1, dzeta=1e-20) == 0

    # Build equivalent WakeFromTable
    t_samples = np.linspace(0, 10/clight, 100000)
    w_longitudinal_x_samples = wake.components[0].function_vs_t(t_samples, beta0=1., dt=1e-20)
    w_longitudinal_x_samples[0] *= 2 # Undo sampling weight
    table = pd.DataFrame({'time': t_samples, 'longitudinal': w_longitudinal_x_samples})
    wake_from_table = xw.WakeFromTable(table)
    wake_from_table.configure_for_tracking(zeta_range=(-1.01, 1.01), num_slices=50)

    assert len(wake_from_table.components) == 1
    assert wake_from_table.components[0].plane == 'z'
    assert wake_from_table.components[0].source_exponents == (0, 0)
    assert wake_from_table.components[0].test_exponents == (0, 0)

    # Assert that the function is positive at close to zero from the right
    assert wake_from_table.components[0].function_vs_t(1e-10, beta0=1, dt=1e-20) > 0
    assert wake_from_table.components[0].function_vs_t(-1e-10, beta0=1, dt=1e-20) == 0

    # Zeta has opposite sign compared to t
    assert wake_from_table.components[0].function_vs_zeta(-1e-3, beta0=1, dzeta=1e-20) > 0
    assert wake_from_table.components[0].function_vs_zeta(+1e-3, beta0=1, dzeta=1e-20) == 0

    from PyHEADTAIL.impedances.wakes import Resonator, WakeField
    from PyHEADTAIL.particles.slicing import UniformBinSlicer
    res_pyht = Resonator(R_shunt=1e8, Q=1e7, frequency=1e9,
                        Yokoya_X1=0, Yokoya_Y1=0,
                        Yokoya_X2=0, Yokoya_Y2=0,
                        switch_Z=True)
    slicer = UniformBinSlicer(n_slices=None, z_sample_points=wake._wake_tracker.slicer.zeta_centers)
    wake_pyht = WakeField(slicer, res_pyht)

    wake.track(p)
    wake_from_table.track(p_table)
    wake_pyht.track(p_ref)

    assert np.max(np.abs(p_ref.delta)) > 0
    xo.assert_allclose(p.delta, p_ref.delta, rtol=0, atol=0.5e-3*np.max(np.abs(p_ref.delta)))
    xo.assert_allclose(p_table.delta, p_ref.delta, rtol=0, atol=2e-3*np.max(np.abs(p_ref.delta)))

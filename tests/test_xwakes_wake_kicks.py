# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

import xtrack as xt
import xpart as xp
import xwakes as xw
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts

import pytest
from scipy.constants import e as qe, c as c_light
import pandas as pd
import numpy as np
import pathlib
from xwakes.wit.component import KIND_DEFINITIONS as kind_to_parameters

test_data_folder = pathlib.Path(__file__).parent.joinpath(
    '../test_data').absolute()

exclude_contexts = ['ContextPyopencl', 'ContextCupy']


@for_all_test_contexts(excluding=exclude_contexts)
@pytest.mark.parametrize('wake_type', ['WakeResonator', 'WakeTable'],
                         ids=lambda x: x)
@pytest.mark.parametrize('kind', [
    ['longitudinal'],
    ['constant_x', 'constant_y'],
    ['dipolar_x','dipolar_y'],
    ['dipolar_xy','dipolar_yx'],
    ['quadrupolar_x','quadrupolar_y'],
    ['quadrupolar_xy','quadrupolar_yx'],
    ], ids=lambda x: '_'.join(x))
def test_wake_kick_single_bunch(test_context, kind, wake_type):
    # test single-bunch wake kick and check that the resonator wake definition
    # is consistent with the table wake definition.
    # in this test we place one particle in each slice, we offset some of them
    # and check that the wake kicks are consistent with the definitions
    p0c = 7000e9
    h_bunch = 600
    circumference=26658.883
    bucket_length = circumference/h_bunch/10
    zeta_range = (-0.5*bucket_length, 0.5*bucket_length)
    num_slices = 100

    particle_dummy = xp.Particles(p0c=p0c)

    if wake_type == 'WakeResonator':
        wf = xw.WakeResonator(
            kind=kind,
            r=1e8, q=1e5, f_r=1e3)
    else:
        wf_analytic = xw.WakeResonator(
                    kind=kind,
                    r=1e8, q=1e5, f_r=1e3)

        # would need beta_0 below here but we just need a larger value than the
        # revolution period
        t_table = np.linspace(0, circumference/c_light/particle_dummy.beta0[0], 1000)
        table = pd.DataFrame()
        table['time'] = t_table
        for i_comp, comp in enumerate(wf_analytic.components):
            print(kind[i_comp])
            table[kind[i_comp]] = comp.function_vs_t(t_table, particle_dummy.beta0, 1e-20)

        wf = xw.WakeFromTable(table)

    wf.configure_for_tracking(
        zeta_range=zeta_range,
        num_slices=num_slices,
        bunch_spacing_zeta=circumference/h_bunch,
        circumference=circumference
        )

    line = xt.Line(elements=[wf])
    line.build_tracker()
    line.particle_ref=xp.Particles(p0c=p0c)

    particles = xp.Particles(p0c=p0c,
                        zeta=wf.slicer.zeta_centers.flatten(),
                        weight=1e12,
                        _context=test_context)

    i_source_x = 5
    i_source_y = 10

    displace_x = 2e-3
    displace_y = 3e-3

    # we displace after a certain particle index so that we check also that
    # in the dipolar and quadrupolar cases there is no wake in front
    particles.x[i_source_x:] += displace_x
    particles.y[i_source_y:] += displace_y

    dict_p_bef = {}

    for kk in kind:
        if kk == 'longitudinal':
            dict_p_bef[kk] = ('delta', particles.delta.copy())
        elif kk.split('_')[1] == 'x' or kk.split('_')[1] == 'xy':
            dict_p_bef[kk] = ('px', particles.px.copy())
        elif kk.split('_')[1] == 'y' or kk.split('_')[1] == 'yx':
            dict_p_bef[kk] = ('py', particles.py.copy())
        else:
            raise ValueError('Invalid kind')

    line.track(particles)

    assert len(wf.components) == len(kind)

    for comp, kk in zip(wf.components, kind):
        if comp.plane == 'z':
            scale = -particles.q0**2 * qe**2 / (
                particles.p0c[0] * particles.beta0[0]* qe) * particles.weight[0]
        else:
            scale = particles.q0**2 * qe**2 / (
                particles.p0c[0] * particles.beta0[0]* qe) * particles.weight[0]
        assert comp.plane == kind_to_parameters[kk]['plane']
        assert comp.source_exponents == kind_to_parameters[kk]['source_exponents']
        assert comp.test_exponents == kind_to_parameters[kk]['test_exponents']

        expected = np.zeros_like(particles.zeta)

        for i_test, z_test in enumerate(particles.zeta):
            expected[i_test] += (particles.x[i_test]**comp.test_exponents[0] *
                                particles.y[i_test]**comp.test_exponents[1] *
                                np.dot(particles.x**comp.source_exponents[0] *
                                        particles.y**comp.source_exponents[1],
                                        comp.function_vs_zeta(z_test - particles.zeta,
                                                            beta0=particles.beta0[0],
                                                            dzeta=1e-12)) * scale)

        xo.assert_allclose(getattr(particles, dict_p_bef[kk][0]) - dict_p_bef[kk][1],
                           expected, rtol=1e-4, atol=1e-20)


@for_all_test_contexts(excluding=exclude_contexts)
@pytest.mark.parametrize('kind', [
    ['longitudinal'],
    ['constant_x', 'constant_y'],
    ['dipolar_x','dipolar_y'],
    ['dipolar_xy','dipolar_yx'],
    ['quadrupolar_x','quadrupolar_y'],
    ['quadrupolar_xy','quadrupolar_yx'],
    ], ids=lambda x: '_'.join(x))
def test_wake_kick_multi_bunch(test_context, kind):
    # test multi bunch wake kick.
    # in this test we place one particle in each slice (for two bunches),
    # we offset some of them and check that the wake kicks are consistent with 
    # the definitions
    p0c = 7000e9
    h_bunch = 600
    circumference=26658.883
    bucket_length = circumference/h_bunch/10
    zeta_range = (-0.5*bucket_length, 0.5*bucket_length)
    num_slices = 100

    filling_scheme = np.zeros(h_bunch)
    filling_scheme[0] = 1
    filling_scheme[1] = 1
    bunch_selection = [0, 1]

    wf = xw.WakeResonator(
        kind=kind,
        r=1e8, q=1e5, f_r=1e3)

    wf.configure_for_tracking(
        zeta_range=zeta_range,
        num_slices=num_slices,
        filling_scheme=filling_scheme,
        bunch_spacing_zeta=circumference/h_bunch,
        bunch_selection=bunch_selection,
        circumference=circumference
        )

    line = xt.Line(elements=[wf])
    line.build_tracker()
    line.particle_ref=xp.Particles(p0c=p0c)

    particles = xp.Particles(p0c=p0c,
                            zeta=wf.slicer.zeta_centers.flatten(),
                            weight=1e12,
                            _context=test_context)

    i_source_x = 5
    i_source_y = 10

    displace_x = 2e-3
    displace_y = 3e-3

    # we displace after a certain particle index so that we check also that
    # in the dipolar and quadrupolar cases there is no wake in front
    particles.x[i_source_x:] += displace_x
    particles.y[i_source_y:] += displace_y

    dict_p_bef = {}

    for kk in kind:
        if kk == 'longitudinal':
            dict_p_bef[kk] = ('delta', particles.delta.copy())
        elif kk.split('_')[1] == 'x' or kk.split('_')[1] == 'xy':
            dict_p_bef[kk] = ('px', particles.px.copy())
        elif kk.split('_')[1] == 'y' or kk.split('_')[1] == 'yx':
            dict_p_bef[kk] = ('py', particles.py.copy())
        else:
            raise ValueError('Invalid kind')

    line.track(particles)

    assert len(wf.components) == len(kind)

    for comp, kk in zip(wf.components, kind):
        if comp.plane == 'z':
            scale = -particles.q0**2 * qe**2 / (
                particles.p0c[0] * particles.beta0[0]* qe) * particles.weight[0]
        else:
            scale = particles.q0**2 * qe**2 / (
                particles.p0c[0] * particles.beta0[0]* qe) * particles.weight[0]
        assert comp.plane == kind_to_parameters[kk]['plane']
        assert comp.source_exponents == kind_to_parameters[kk]['source_exponents']
        assert comp.test_exponents == kind_to_parameters[kk]['test_exponents']

        expected = np.zeros_like(particles.zeta)

        for i_test, z_test in enumerate(particles.zeta):
            expected[i_test] += (particles.x[i_test]**comp.test_exponents[0] *
                                particles.y[i_test]**comp.test_exponents[1] *
                                np.dot(particles.x**comp.source_exponents[0] *
                                        particles.y**comp.source_exponents[1],
                                        comp.function_vs_zeta(z_test - particles.zeta,
                                                            beta0=particles.beta0[0],
                                                            dzeta=1e-12)) * scale)

        xo.assert_allclose(getattr(particles, dict_p_bef[kk][0]) - dict_p_bef[kk][1],
                           expected, rtol=1e-4, atol=1e-20)


@for_all_test_contexts(excluding=exclude_contexts)
def test_beam_loading_theorem(test_context):
    # this test is less complete than the previous ones but it tests explicitly
    # the beam loading theorem for the longitudinal wake on two particles
    p0c = 1.2e9
    h_RF = 600
    n_slices = 100
    circumference = 26658.883
    bucket_length = circumference/h_RF
    zeta_range = (-0.5*bucket_length, 0.5*bucket_length)
    dz = (zeta_range[1] - zeta_range[0])/n_slices
    zeta = np.linspace(zeta_range[0] + dz/2, zeta_range[1]-dz/2,
                       n_slices)

    i_source = -10
    i_test = 10

    particles = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        p0c=p0c,
        zeta=[zeta[i_test], zeta[i_source]],
        delta=[1e-3, 2e-3],
        weight=1e12,
        _context=test_context
    )

    delta_bef = particles.delta.copy()

    wf = xw.WakeResonator(kind='longitudinal', r=1e8, q=1e7, f_r=1e3)
    wf.configure_for_tracking(zeta_range=zeta_range, num_slices=n_slices)

    line = xt.Line(elements=[wf],
                   element_names=['wf'])
    line.build_tracker()
    line.track(particles, num_turns=1)

    scale = -particles.q0**2 * qe**2 / (qe * particles.p0c[0] * particles.beta0[0]
                                       ) * particles.weight[0]

    assert len(wf.components) == 1
    comp = wf.components[0]
    w_zero_plus = comp.function_vs_zeta(-1e-12, beta0=1, dzeta=1e-20)
    xo.assert_allclose(w_zero_plus, 62831.85307, atol=0, rtol=1e-6) # Regresssion test

    assert particles.zeta[1] > particles.zeta[0]
    xo.assert_allclose(particles.delta[1] - delta_bef[1], scale * w_zero_plus / 2, # beam loading theorem
                          rtol=1e-4, atol=0)

    # Resonator frequency chosen to have practically constant wake
    xo.assert_allclose(particles.delta[0] - delta_bef[0], scale * w_zero_plus * 3 / 2,  # 1 from particle in front
                        rtol=1e-4, atol=0)                               # 1/2 from particle itself


@for_all_test_contexts(excluding=exclude_contexts)
@pytest.mark.parametrize('kind', [
    ['longitudinal'],
    ['constant_x', 'constant_y'],
    ['dipolar_x','dipolar_y'],
    ['dipolar_xy','dipolar_yx'],
    ['quadrupolar_x','quadrupolar_y'],
    ['quadrupolar_xy','quadrupolar_yx'],
    ], ids=lambda x: '_'.join(x))
def test_wake_kick_multiturn(test_context, kind):
    # test multi-turn wake kick.
    # in this test we place one particle in each slice, we offset some of them
    # and check that the wake kicks are consistent with the definitions when
    # tracking with multi-turn wakes
    num_turns_wake = 2
    p0c = 1.2e9
    h_RF = 600
    n_slices = 100
    circumference = 26658.883
    bucket_length = circumference/h_RF
    zeta_range = (-0.5*bucket_length, 0.5*bucket_length)

    wf = xw.WakeResonator(kind=kind,
                          r=1e8, q=1e7, f_r=1e3)
    wf.configure_for_tracking(zeta_range=zeta_range, num_slices=n_slices,
                              num_turns=num_turns_wake, circumference=circumference)


    i_source = -10

    particles = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        p0c=p0c,
        zeta=wf.slicer.zeta_centers.flatten(),
        weight=1e12,
        _context=test_context
    )

    displace_x = 2e-3
    displace_y = 3e-3

    particles.x[i_source] += displace_x
    particles.y[i_source] += displace_y

    dict_p_bef = {}

    for kk in kind:
        if kk == 'longitudinal':
            dict_p_bef[kk] = ('delta', particles.delta.copy())
        elif kk.split('_')[1] == 'x' or kk.split('_')[1] == 'xy':
            dict_p_bef[kk] = ('px', particles.px.copy())
        elif kk.split('_')[1] == 'y' or kk.split('_')[1] == 'yx':
            dict_p_bef[kk] = ('py', particles.py.copy())
        else:
            raise ValueError('Invalid kind')

    line = xt.Line(elements=[wf],
                   element_names=['wf'])
    line.build_tracker()
    line.track(particles, num_turns=3)

    scale = particles.q0**2 * qe**2 / (
        particles.p0c[0] * particles.beta0[0]* qe) * particles.weight[0]

    wake_kwargs = {
        'beta0': particles.beta0[0],
        'dzeta': 1e-20,
    }

    for comp, kk in zip(wf.components, kind):
        if comp.plane == 'z':
            scale = -particles.q0**2 * qe**2 / (
                particles.p0c[0] * particles.beta0[0]* qe) * particles.weight[0]
        else:
            scale = particles.q0**2 * qe**2 / (
                particles.p0c[0] * particles.beta0[0]* qe) * particles.weight[0]
        assert comp.plane == kind_to_parameters[kk]['plane']
        assert comp.source_exponents == kind_to_parameters[kk]['source_exponents']
        assert comp.test_exponents == kind_to_parameters[kk]['test_exponents']

        expected = np.zeros_like(particles.zeta)

        for i_test, z_test in enumerate(particles.zeta):
            expected[i_test] += (
                particles.x[i_test]**comp.test_exponents[0] *
                particles.y[i_test]**comp.test_exponents[1] *
                np.dot(particles.x**comp.source_exponents[0] *
                particles.y**comp.source_exponents[1],
                comp.function_vs_zeta(z_test - particles.zeta,
                                      **wake_kwargs)) *
                scale * (num_turns_wake + 1))

            expected[i_test] += (
                particles.x[i_test]**comp.test_exponents[0] *
                particles.y[i_test]**comp.test_exponents[1] *
                np.dot(particles.x**comp.source_exponents[0] *
                       particles.y**comp.source_exponents[1],
                       comp.function_vs_zeta(z_test - particles.zeta -
                                             circumference, **wake_kwargs)) *
                scale * num_turns_wake)

        xo.assert_allclose(getattr(particles, dict_p_bef[kk][0]) - dict_p_bef[kk][1],
                           expected, rtol=1e-4, atol=1e-20)


@for_all_test_contexts(excluding=exclude_contexts)
@pytest.mark.parametrize('kind', [
    ['longitudinal'],
    ['constant_x', 'constant_y'],
    ['dipolar_x','dipolar_y'],
    ['dipolar_xy','dipolar_yx'],
    ['quadrupolar_x','quadrupolar_y'],
    ['quadrupolar_xy','quadrupolar_yx'],
    ], ids=lambda x: '_'.join(x))
def test_wake_kick_multiturn_reset(test_context, kind):
    # a more explicit test on multi-turn wake kick reset.
    # we track for n_turns_wake, set the weights to zero and track for
    # n_turns_wake-1 after which we track for another turn checking that no
    # kick is applied to the particles
    num_turns_wake = 4
    p0c = 1.2e9
    h_RF = 600
    n_slices = 100
    circumference = 26658.883
    bucket_length = circumference/h_RF
    zeta_range = (-0.5*bucket_length, 0.5*bucket_length)

    wf = xw.WakeResonator(kind=kind,
                          r=1e8, q=1e7, f_r=1e3)
    wf.configure_for_tracking(zeta_range=zeta_range, num_slices=n_slices,
                              num_turns=num_turns_wake, circumference=circumference)


    i_source = -10

    particles = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        p0c=p0c,
        zeta=wf.slicer.zeta_centers.flatten(),
        weight=1e12,
        _context=test_context
    )

    displace_x = 2e-3
    displace_y = 3e-3

    particles.x[i_source] += displace_x
    particles.y[i_source] += displace_y

    line = xt.Line(elements=[wf],
                   element_names=['wf'])
    line.build_tracker()
    line.track(particles, num_turns=num_turns_wake)

    particles.weight *= 0
    line.track(particles, num_turns=num_turns_wake-1)

    dict_p_bef = {}
    for kk in kind:
        if kk == 'longitudinal':
            dict_p_bef[kk] = ('delta', particles.delta.copy())
        elif kk.split('_')[1] == 'x' or kk.split('_')[1] == 'xy':
            dict_p_bef[kk] = ('px', particles.px.copy())
        elif kk.split('_')[1] == 'y' or kk.split('_')[1] == 'yx':
            dict_p_bef[kk] = ('py', particles.py.copy())
        else:
            raise ValueError('Invalid kind')

    line.track(particles, num_turns=1)

    for kk in kind:
        xo.assert_allclose(getattr(particles, dict_p_bef[kk][0]),
                           dict_p_bef[kk][1], rtol=1e-4, atol=1e-20)

@for_all_test_contexts(excluding=exclude_contexts)
@pytest.mark.parametrize('kind', [['longitudinal'],
                                  ['constant_x', 'constant_y'],
                                  ['dipolar_x','dipolar_y'],
                                  ['dipolar_xy','dipolar_yx'],
                                  ['quadrupolar_x','quadrupolar_y'],
                                  ['quadrupolar_xy','quadrupolar_yx']],
                                  ids=lambda x: '_'.join(x))
def test_wake_kick_multibunch_pipeline(test_context, kind):
    # test multi-bunch wake kick with pipeline
    # in this test we place one particle in each slice for a random number of
    # bunches on two communicating pipeline branches, we offset them
    # and check that the wake kicks are consistent with the definitions
    p0c = 1e9

    # Filling scheme
    n_slots = 100
    filling_scheme = np.array(np.floor(np.random.rand(n_slots)+0.1), dtype=int)
    while np.sum(filling_scheme) < 2:
        # if less than 2 slots are filled, try again
        filling_scheme = np.array(np.floor(np.random.rand(n_slots)+0.1), dtype=int)
    filling_scheme[0] = 1
    filled_slots = np.nonzero(filling_scheme)[0]
    n_bunches = len(filled_slots)
    n_bunches_0 = int(np.floor(n_bunches/2))
    bunch_selection_0 = np.arange(n_bunches_0, dtype=int)
    bunch_selection_1 = np.arange(n_bunches_0, n_bunches, dtype=int)

    print('initialising pipeline')
    comm = xt.pipeline.core.PipelineCommunicator()
    pipeline_manager = xt.PipelineManager(comm)
    pipeline_manager.add_particles(f'b0', 0)
    pipeline_manager.add_particles(f'b1', 0)
    pipeline_manager.add_element('wake')

    bunch_spacing = 25E-9*c_light
    sigma_zeta = bunch_spacing/20
    zeta_range = (-1.1*sigma_zeta, 1.1*sigma_zeta)
    num_slices = 1001
    dzeta = (zeta_range[1]-zeta_range[0])/num_slices
    zeta_slice_edges = np.linspace(zeta_range[0], zeta_range[1], num_slices+1)
    zeta_centers = zeta_slice_edges[:-1]+dzeta/2

    zeta_0 = []
    for bunch_number in bunch_selection_0:
        zeta_0.append(zeta_centers-filled_slots[bunch_number]*bunch_spacing)
    zeta_0 = np.hstack(zeta_0)

    print('Initialising particles')
    particles_0 = xt.Particles(p0c=p0c, zeta=zeta_0,
                                x=np.random.randn(len(zeta_0)),
                                y=np.random.randn(len(zeta_0)),
                                _context=test_context)
    particles_0.init_pipeline('b0')

    zeta_1 = []
    for bunch_number in bunch_selection_1:
        zeta_1.append(zeta_centers-filled_slots[bunch_number]*bunch_spacing)
    zeta_1 = np.hstack(zeta_1)
    particles_1 = xt.Particles(p0c=p0c, zeta=zeta_1,
                                x=np.random.randn(len(zeta_1)),
                                y=np.random.randn(len(zeta_1)),
                                _context=test_context)
    particles_1.init_pipeline('b1')

    i_source_x = 0
    i_source_y = 0

    displace_x_0 = 2e-3
    displace_y_0 = 3e-3
    displace_x_1 = 4e-3
    displace_y_1 = 5e-3

    # we displace after a certain particle index so that we check also that
    # in the dipolar and quadrupolar cases there is no wake in front
    particles_0.x[i_source_x:] += displace_x_0
    particles_0.y[i_source_y:] += displace_y_0

    particles_1.x[i_source_x:] += displace_x_1
    particles_1.y[i_source_y:] += displace_y_1

    print('Initialising wake')
    n_turns_wake = 1
    circumference = n_slots * bunch_spacing
    wake_table_name = (test_data_folder /
                    'headtail_format_table_hllhc/HLLHC_wake_flattop_nocrab.dat')

    wake_file_columns = ['time', 'longitudinal', 'dipolar_x', 'dipolar_y',
                            'quadrupolar_x', 'quadrupolar_y', 'dipolar_xy',
                            'quadrupolar_xy', 'dipolar_yx', 'quadrupolar_yx',
                            'constant_x', 'constant_y']
    components = kind
    wake_df = xw.read_headtail_file(wake_table_name,
                                    wake_file_columns)
    wf_0 = xw.WakeFromTable(wake_df, columns=components)
    wf_0.configure_for_tracking(zeta_range=zeta_range,
                                num_slices=num_slices,
                                bunch_spacing_zeta=bunch_spacing,
                                filling_scheme=filling_scheme,
                                bunch_selection=bunch_selection_0,
                                num_turns=n_turns_wake,
                                circumference=circumference
                                )
    wf_0._wake_tracker.init_pipeline(pipeline_manager=pipeline_manager,
                                        element_name='wake',
                                        partner_names=['b1'])

    wf_1 = xw.WakeFromTable(wake_df, columns=components)
    wf_1.configure_for_tracking(zeta_range=zeta_range,
                                num_slices=num_slices,
                                bunch_spacing_zeta=bunch_spacing,
                                filling_scheme=filling_scheme,
                                bunch_selection=bunch_selection_1,
                                num_turns=n_turns_wake,
                                circumference=circumference
                                )
    wf_1._wake_tracker.init_pipeline(pipeline_manager=pipeline_manager,
                                        element_name='wake',
                                        partner_names=['b0'])

    dict_p_bef_0 = {}
    dict_p_bef_1 = {}

    for kk in kind:
        if kk == 'longitudinal':
            dict_p_bef_0[kk] = ('delta', particles_0.delta.copy())
            dict_p_bef_1[kk] = ('delta', particles_1.delta.copy())
        elif kk.split('_')[1] == 'x' or kk.split('_')[1] == 'xy':
            dict_p_bef_0[kk] = ('px', particles_0.px.copy())
            dict_p_bef_1[kk] = ('px', particles_1.px.copy())
        elif kk.split('_')[1] == 'y' or kk.split('_')[1] == 'yx':
            dict_p_bef_0[kk] = ('py', particles_0.py.copy())
            dict_p_bef_1[kk] = ('py', particles_1.py.copy())
        else:
            raise ValueError('Invalid kind')

    print('Initialising lines')
    line_0 = xt.Line(elements=[wf_0])
    line_1 = xt.Line(elements=[wf_1])
    print('Initialising multitracker')
    line_0.build_tracker()
    line_1.build_tracker()
    multitracker = xt.PipelineMultiTracker(
        branches=[xt.PipelineBranch(line=line_0, particles=particles_0),
                  xt.PipelineBranch(line=line_1, particles=particles_1)])
    print('Tracking')
    pipeline_manager.verbose = True
    multitracker.track(num_turns=1)
    print('loading test data')
    particles_tot = xt.Particles.merge([particles_0, particles_1])

    for particles, dict_p_bef, wf in [(particles_0, dict_p_bef_0, wf_0),
                                      (particles_1, dict_p_bef_1, wf_1)]:

        assert len(wf.components) == len(kind)

        for comp, kk in zip(wf.components, kind):
            if comp.plane == 'z':
                scale = -particles.q0**2 * qe**2 / (
                    particles.p0c[0] * particles.beta0[0]* qe) * particles.weight[0]
            else:
                scale = particles.q0**2 * qe**2 / (
                    particles.p0c[0] * particles.beta0[0]* qe) * particles.weight[0]
            assert comp.plane == kind_to_parameters[kk]['plane']
            assert comp.source_exponents == kind_to_parameters[kk]['source_exponents']
            assert comp.test_exponents == kind_to_parameters[kk]['test_exponents']

            expected = np.zeros_like(particles.zeta)

            for i_test, z_test in enumerate(particles.zeta):
                expected[i_test] += (particles.x[i_test]**comp.test_exponents[0] *
                                        particles.y[i_test]**comp.test_exponents[1] *
                                        np.dot(particles_tot.x**comp.source_exponents[0] *
                                            particles_tot.y**comp.source_exponents[1],
                                            comp.function_vs_zeta(z_test - particles_tot.zeta,
                                                                    beta0=particles_tot.beta0[0],
                                                                    dzeta=1e-12)) * scale)

            xo.assert_allclose(getattr(particles, dict_p_bef[kk][0]) - dict_p_bef[kk][1],
                    expected, rtol=1e-4, atol=1e-20)


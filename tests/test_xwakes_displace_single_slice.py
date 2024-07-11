import numpy as np
from scipy.constants import e as qe
from xobjects.test_helpers import for_all_test_contexts
import xtrack as xt
import xwakes as xw
import xobjects as xo
import pytest

exclude_contexts = ['ContextPyopencl', 'ContextCupy']

@for_all_test_contexts(excluding=exclude_contexts)
def test_longitudinal_wake_kick(test_context):

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
def test_constant_wake_kick(test_context):

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
        zeta=np.array([zeta[i_test], zeta[i_source]]),
        px=[1e-4, 2e-4],
        py=[-3e-4, -4e-4],
        weight=1e12,
        _context=test_context
    )

    px_bef = particles.px.copy()
    py_bef = particles.py.copy()

    wf = xw.WakeResonator(kind=['constant_x', 'constant_y'],
                          r=1e8, q=1e7, f_r=1e3)
    wf.configure_for_tracking(zeta_range=zeta_range, num_slices=n_slices)

    line = xt.Line(elements=[wf],
                   element_names=['wf'])
    line.build_tracker()
    line.track(particles, num_turns=1)

    scale = particles.q0**2 * qe**2 / (
        particles.p0c[0] * particles.beta0[0]* qe) * particles.weight[0]

    assert len(wf.components) == 2
    comp_x = wf.components[0]
    comp_y = wf.components[1]
    assert comp_x.plane == 'x'
    assert comp_y.plane == 'y'
    assert comp_x.source_exponents == (0, 0)
    assert comp_x.test_exponents == (0, 0)
    assert comp_y.source_exponents == (0, 0)
    assert comp_y.test_exponents == (0, 0)
    xo.assert_allclose((particles.px - px_bef)[0],
            scale * comp_x.function_vs_zeta(-particles.zeta[1] +
                particles.zeta[0], beta0=particles.beta0[0], dzeta=1e-20),
            rtol=1e-4, atol=0)
    xo.assert_allclose((particles.py - py_bef)[0],
            scale *comp_y.function_vs_zeta(-particles.zeta[1] +
                particles.zeta[0], beta0=particles.beta0[0], dzeta=1e-20),
            rtol=1e-4, atol=0)


@for_all_test_contexts(excluding=exclude_contexts)
def test_dipolar_wake_kick(test_context):

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

    particles = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        p0c=p0c,
        zeta=zeta,
        weight=1e12,
        _context=test_context
    )

    displace_x = 2e-3
    displace_y = 3e-3

    particles.x[i_source] += displace_x
    particles.y[i_source] += displace_y

    px_bef = particles.px.copy()
    py_bef = particles.py.copy()

    wf = xw.WakeResonator(kind=['dipolar_x', 'dipolar_y'],
                          r=1e8, q=1e7, f_r=1e3)
    wf.configure_for_tracking(zeta_range=zeta_range, num_slices=n_slices)

    line = xt.Line(elements=[wf],
                   element_names=['wf'])
    line.build_tracker()
    line.track(particles, num_turns=1)

    scale = particles.q0**2 * qe**2 / (
        particles.p0c[0] * particles.beta0[0]* qe) * particles.weight[0]

    assert len(wf.components) == 2
    comp_x = wf.components[0]
    comp_y = wf.components[1]
    assert comp_x.plane == 'x'
    assert comp_y.plane == 'y'
    assert comp_x.source_exponents == (1, 0)
    assert comp_x.test_exponents == (0, 0)
    assert comp_y.source_exponents == (0, 1)
    assert comp_y.test_exponents == (0, 0)
    expected_x = comp_x.function_vs_zeta(-particles.zeta[i_source] +
                              particles.zeta,
                              beta0=particles.beta0[0],
                              dzeta=1e-20) * scale
    expected_y = comp_y.function_vs_zeta(-particles.zeta[i_source] +
                                particles.zeta,
                                beta0=particles.beta0[0],
                                dzeta=1e-20) * scale
    assert expected_x.max() > 1e-14
    assert expected_y.max() > 1e-14
    xo.assert_allclose((particles.px - px_bef)/displace_x, expected_x,
                       rtol=1e-4, atol=1e-20)
    xo.assert_allclose((particles.py - py_bef)/displace_y, expected_y,
                        rtol=1e-4, atol=1e-20)


@for_all_test_contexts(excluding=exclude_contexts)
def test_cross_dipolar_wake_kick(test_context):

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

    particles = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        p0c=p0c,
        zeta=zeta,
        weight=1e12,
        _context=test_context
    )

    displace_x = 2e-3
    displace_y = 3e-3

    particles.x[i_source] += displace_x
    particles.y[i_source] += displace_y

    px_bef = particles.px.copy()
    py_bef = particles.py.copy()

    wf = xw.WakeResonator(kind=['dipolar_xy', 'dipolar_yx'],
                          r=1e8, q=1e7, f_r=1e3)
    wf.configure_for_tracking(zeta_range=zeta_range, num_slices=n_slices)

    line = xt.Line(elements=[wf],
                   element_names=['wf'])
    line.build_tracker()
    line.track(particles, num_turns=1)

    scale = particles.q0**2 * qe**2 / (
        particles.p0c[0] * particles.beta0[0]* qe) * particles.weight[0]

    assert len(wf.components) == 2
    comp_x = wf.components[0]
    comp_y = wf.components[1]
    assert comp_x.plane == 'x'
    assert comp_y.plane == 'y'
    assert comp_x.source_exponents == (0, 1)
    assert comp_x.test_exponents == (0, 0)
    assert comp_y.source_exponents == (1, 0)
    assert comp_y.test_exponents == (0, 0)
    expected_x = comp_x.function_vs_zeta(-particles.zeta[i_source] +
                              particles.zeta,
                              beta0=particles.beta0[0],
                              dzeta=1e-20) * scale
    expected_y = comp_y.function_vs_zeta(-particles.zeta[i_source] +
                                particles.zeta,
                                beta0=particles.beta0[0],
                                dzeta=1e-20) * scale
    assert expected_x.max() > 1e-14
    assert expected_y.max() > 1e-14
    xo.assert_allclose((particles.px - px_bef)/displace_y, expected_x,
                       rtol=1e-4, atol=1e-20)
    xo.assert_allclose((particles.py - py_bef)/displace_x, expected_y,
                        rtol=1e-4, atol=1e-20)

@for_all_test_contexts(excluding=exclude_contexts)
def test_quadrupolar_wake_kick(test_context):

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
        zeta=zeta,
        weight=1e12,
        _context=test_context
    )

    displace_x_test = 2e-3
    displace_y_test = 3e-3
    displace_x_source = 4e-3
    displace_y_source = 5e-3

    particles.x[i_source] += displace_x_source
    particles.y[i_source] += displace_y_source

    particles.x[i_test] += displace_x_test
    particles.y[i_test] += displace_y_test

    px_bef = particles.px.copy()
    py_bef = particles.py.copy()

    wf = xw.WakeResonator(kind=['quadrupolar_x', 'quadrupolar_y'],
                          r=1e8, q=1e7, f_r=1e3)
    wf.configure_for_tracking(zeta_range=zeta_range, num_slices=n_slices)

    line = xt.Line(elements=[wf],
                   element_names=['wf'])
    line.build_tracker()
    line.track(particles, num_turns=1)

    scale = particles.q0**2 * qe**2 / (
        particles.p0c[0] * particles.beta0[0]* qe) * particles.weight[0]

    assert len(wf.components) == 2
    comp_x = wf.components[0]
    comp_y = wf.components[1]
    assert comp_x.plane == 'x'
    assert comp_y.plane == 'y'
    assert comp_x.source_exponents == (0, 0)
    assert comp_x.test_exponents == (1, 0)
    assert comp_y.source_exponents == (0, 0)
    assert comp_y.test_exponents == (0, 1)
    expected_x = np.sum(comp_x.function_vs_zeta(particles.zeta[i_test] - particles.zeta,
                        beta0=particles.beta0[0], dzeta=1e-20))*scale
    expected_y = np.sum(comp_y.function_vs_zeta(particles.zeta[i_test] - particles.zeta,
                        beta0=particles.beta0[0], dzeta=1e-20))*scale
    assert expected_x.max() > 1e-14
    assert expected_y.max() > 1e-14
    xo.assert_allclose((particles.px - px_bef)[i_test]/displace_x_test, expected_x,
                       rtol=1e-4, atol=1e-20)
    xo.assert_allclose((particles.py - py_bef)[i_test]/displace_y_test, expected_y,
                        rtol=1e-4, atol=1e-20)


@for_all_test_contexts(excluding=exclude_contexts)
def test_cross_quadrupolar_wake_kick(test_context):

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
        zeta=zeta,
        weight=1e12,
        _context=test_context
    )

    displace_x_test = 2e-3
    displace_y_test = 3e-3
    displace_x_source = 4e-3
    displace_y_source = 5e-3

    particles.x[i_source] += displace_x_source
    particles.y[i_source] += displace_y_source

    particles.x[i_test] += displace_x_test
    particles.y[i_test] += displace_y_test

    px_bef = particles.px.copy()
    py_bef = particles.py.copy()

    wf = xw.WakeResonator(kind=['quadrupolar_xy', 'quadrupolar_yx'],
                          r=1e8, q=1e7, f_r=1e3)
    wf.configure_for_tracking(zeta_range=zeta_range, num_slices=n_slices)

    line = xt.Line(elements=[wf],
                   element_names=['wf'])
    line.build_tracker()
    line.track(particles, num_turns=1)

    scale = particles.q0**2 * qe**2 / (
        particles.p0c[0] * particles.beta0[0]* qe) * particles.weight[0]

    assert len(wf.components) == 2
    comp_x = wf.components[0]
    comp_y = wf.components[1]
    assert comp_x.plane == 'x'
    assert comp_y.plane == 'y'
    assert comp_x.source_exponents == (0, 0)
    assert comp_x.test_exponents == (0, 1)
    assert comp_y.source_exponents == (0, 0)
    assert comp_y.test_exponents == (1, 0)
    expected_x = np.sum(comp_x.function_vs_zeta(particles.zeta[i_test] - particles.zeta,
                        beta0=particles.beta0[0], dzeta=1e-20))*scale
    expected_y = np.sum(comp_y.function_vs_zeta(particles.zeta[i_test] - particles.zeta,
                        beta0=particles.beta0[0], dzeta=1e-20))*scale
    assert expected_x.max() > 1e-14
    assert expected_y.max() > 1e-14
    xo.assert_allclose((particles.px - px_bef)[i_test]/displace_y_test, expected_x,
                       rtol=1e-4, atol=1e-20)
    xo.assert_allclose((particles.py - py_bef)[i_test]/displace_x_test, expected_y,
                        rtol=1e-4, atol=1e-20)

@for_all_test_contexts(excluding=exclude_contexts)
def test_dipolar_wake_kick_multiturn(test_context):
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

    particles = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        p0c=p0c,
        zeta=zeta,
        weight=1e12,
        _context=test_context
    )

    displace_x = 2e-3
    displace_y = 3e-3

    particles.x[i_source] += displace_x
    particles.y[i_source] += displace_y

    px_bef = particles.px.copy()
    py_bef = particles.py.copy()

    wf = xw.WakeResonator(kind=['dipolar_x', 'dipolar_y'],
                          r=1e8, q=1e7, f_r=1e3)
    wf.configure_for_tracking(zeta_range=zeta_range, num_slices=n_slices,
                              num_turns=2, circumference=circumference)

    line = xt.Line(elements=[wf],
                   element_names=['wf'])
    line.build_tracker()
    line.track(particles, num_turns=2)

    scale = particles.q0**2 * qe**2 / (
        particles.p0c[0] * particles.beta0[0]* qe) * particles.weight[0]

    assert len(wf.components) == 2
    comp_x = wf.components[0]
    comp_y = wf.components[1]
    assert comp_x.plane == 'x'
    assert comp_y.plane == 'y'
    assert comp_x.source_exponents == (1, 0)
    assert comp_x.test_exponents == (0, 0)
    assert comp_y.source_exponents == (0, 1)
    assert comp_y.test_exponents == (0, 0)
    wake_kwargs = {
        'beta0': particles.beta0[0],
        'dzeta': 1e-20,
    }
    expected_x = (
        comp_x.function_vs_zeta(-particles.zeta[i_source] +
                        particles.zeta - circumference, **wake_kwargs)
        + comp_x.function_vs_zeta(-particles.zeta[i_source] +
                        particles.zeta, **wake_kwargs)
        + comp_x.function_vs_zeta(-particles.zeta[i_source] +
                        particles.zeta, **wake_kwargs)) * scale
    expected_y = (
        comp_y.function_vs_zeta(-particles.zeta[i_source] +
                        particles.zeta - circumference, **wake_kwargs)
        + comp_y.function_vs_zeta(-particles.zeta[i_source] +
                        particles.zeta, **wake_kwargs)
        + comp_y.function_vs_zeta(-particles.zeta[i_source] +
                        particles.zeta, **wake_kwargs)) * scale

    assert expected_x.max() > 1e-14
    assert expected_y.max() > 1e-14
    xo.assert_allclose((particles.px - px_bef)/displace_x, expected_x,
                       rtol=1e-4, atol=1e-20)
    xo.assert_allclose((particles.py - py_bef)/displace_y, expected_y,
                        rtol=1e-4, atol=1e-20)


@for_all_test_contexts(excluding=exclude_contexts)
@pytest.mark.parametrize('kind', [['longitudinal'],
                                  ['constant_x', 'constant_y'],
                                  ['dipolar_x', 'dipolar_y'],
                                  ['dipolar_xy', 'dipolar_yx'],
                                  ['quadrupolar_x', 'quadrupolar_y'],
                                  ['quadrupolar_xy', 'quadrupolar_yx']
                                  ])
def test_wake_kick(test_context, kind):
    p0c = 1.2e9
    h_RF = 600
    n_slices = 100
    circumference = 26658.883
    bucket_length = circumference/h_RF
    zeta_range = (-0.5*bucket_length, 0.5*bucket_length)
    dz = (zeta_range[1] - zeta_range[0])/n_slices
    zz = np.linspace(zeta_range[0] + dz/2, zeta_range[1]-dz/2,
                       n_slices)

    i_source = -10
    i_test = 10

    if 'longitudinal' in kind:
        zeta = [zz[i_test], zz[i_source]]
    else:
        zeta = zz

    particles = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        p0c=p0c,
        zeta=zeta,
        weight=1e12,
        _context=test_context
    )

    delta_bef = particles.delta.copy()
    px_bef = particles.px.copy()
    py_bef = particles.py.copy()

    # is this really necessary?
    if 'dipolar_x' in kind or 'dipolar_xy' in kind:
        displace_x_source = 2e-3
    else:
        displace_x_source = 0

    if 'dipolar_y' in kind or 'dipolar_yx' in kind:
        displace_y_source = 3e-3
    else:
        displace_y_source = 0

    if 'quadrupolar_x' in kind or 'quadrupolar_xy' in kind:
        displace_x_test = 2e-3
    else:
        displace_x_test = 0

    if 'quadrupolar_y' in kind or 'quadrupolar_yx' in kind:
        displace_y_test = 3e-3
    else:
        displace_y_test = 0

    if displace_x_source != 0:
        particles.x[i_source] += displace_x_source
    if displace_y_source != 0:
        particles.y[i_source] += displace_y_source
    if displace_x_test != 0:
        particles.x[i_test] += displace_x_test
    if displace_y_test != 0:
        particles.y[i_test] += displace_y_test

    wf = xw.WakeResonator(kind=kind,
                          r=1e8, q=1e7, f_r=1e3)
    wf.configure_for_tracking(zeta_range=zeta_range, num_slices=n_slices)

    line = xt.Line(elements=[wf],
                   element_names=['wf'])
    line.build_tracker()
    line.track(particles, num_turns=1)

    assert len(wf.components) == len(kind)

    if len(kind) == 1:
        comp = wf.components[0]
        assert comp.source_exponents == (0, 0)
        assert comp.test_exponents == (0, 0)
        assert comp.plane == 'z'
        w_zero_plus = comp.function_vs_zeta(-1e-12, beta0=1, dzeta=1e-20)
        xo.assert_allclose(w_zero_plus, 62831.85307, atol=0, rtol=1e-6)
        assert particles.zeta[1] > particles.zeta[0]
        scale = -particles.q0**2 * qe**2 / (qe * particles.p0c[0] * particles.beta0[0]
                                    ) * particles.weight[0]
        xo.assert_allclose(particles.delta[1] - delta_bef[1], scale * w_zero_plus / 2, # beam loading theorem
                        rtol=1e-4, atol=0)

        # Resonator frequency chosen to have practically constant wake
        xo.assert_allclose(particles.delta[0] - delta_bef[0], scale * w_zero_plus * 3 / 2,  # 1 from particle in front
                        rtol=1e-4, atol=0)
    else:
        comp_x = wf.components[0]
        comp_y = wf.components[1]
        assert comp_x.plane == 'x'
        assert comp_y.plane == 'y'
        if kind[0] is 'constant_x':
            assert comp_x.source_exponents == (0, 0)
            assert comp_x.test_exponents == (0, 0)
        if kind[0] is 'dipolar_x':
            assert comp_x.source_exponents == (1, 0)
            assert comp_x.test_exponents == (0, 0)
        if kind[0] is 'quadrupolar_x':
            assert comp_x.source_exponents == (0, 0)
            assert comp_x.test_exponents == (1, 0)
        if kind[0] is 'dipolar_xy':
            assert comp_x.source_exponents == (0, 1)
            assert comp_x.test_exponents == (0, 0)
        if kind[0] is 'quadrupolar_xy':
            assert comp_x.source_exponents == (0, 0)
            assert comp_x.test_exponents == (0, 1)
        if kind[1] is 'constant_y':
            assert comp_y.source_exponents == (0, 0)
            assert comp_y.test_exponents == (0, 0)
        if kind[1] is 'dipolar_y':
            assert comp_y.source_exponents == (0, 1)
            assert comp_y.test_exponents == (0, 0)
        if kind[1] is 'quadrupolar_y':
            assert comp_y.source_exponents == (0, 0)
            assert comp_y.test_exponents == (0, 1)
        if kind[1] is 'dipolar_yx':
            assert comp_y.source_exponents == (1, 0)
            assert comp_y.test_exponents == (0, 0)
        if kind[1] is 'quadrupolar_yx':
            assert comp_y.source_exponents == (0, 0)
            assert comp_y.test_exponents == (1, 0)
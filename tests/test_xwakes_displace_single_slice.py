import numpy as np
from scipy.constants import e as qe
from xobjects.test_helpers import for_all_test_contexts
import xtrack as xt
import xwakes as xw
import xobjects as xo

exclude_contexts = ['ContextPyopencl', 'ContextCupy']

@for_all_test_contexts(excluding=exclude_contexts)
def test_longitudinal_wake_kick(test_context):

    p0c = 7000e9
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
    xo.assert_allclose(particles.delta[1], scale * w_zero_plus / 2, # beam loading theorem
                          rtol=1e-4, atol=0)

    # Resonator frequency chosen to have practically constant wake
    xo.assert_allclose(particles.delta[0], scale * w_zero_plus * 3 / 2,  # 1 from particle in front
                        rtol=1e-4, atol=0)                               # 1/2 from particle itself

import xtrack as xt
import xpart as xp
import xwakes as xw
import xobjects as xo
import xfields as xf
from xobjects.test_helpers import for_all_test_contexts

import pytest
from scipy.constants import e as qe, c as c_light
import pandas as pd
import numpy as np

exclude_contexts = ['ContextPyopencl', 'ContextCupy']

kind_to_parameters = {
    'longitudinal': {'plane': 'z', 'source_exponents': (0, 0), 'test_exponents': (0, 0)},
    'constant_x': {'plane': 'x', 'source_exponents': (0, 0), 'test_exponents': (0, 0)},
    'constant_y': {'plane': 'y', 'source_exponents': (0, 0), 'test_exponents': (0, 0)},
    'dipolar_x': {'plane': 'x', 'source_exponents': (1, 0), 'test_exponents': (0, 0)},
    'dipolar_xy': {'plane': 'x', 'source_exponents': (0, 1), 'test_exponents': (0, 0)},
    'dipolar_yx': {'plane': 'y', 'source_exponents': (1, 0), 'test_exponents': (0, 0)},
    'dipolar_y': {'plane': 'y', 'source_exponents': (0, 1), 'test_exponents': (0, 0)},
    'quadrupolar_x': {'plane': 'x', 'source_exponents': (0, 0), 'test_exponents': (1, 0)},
    'quadrupolar_y': {'plane': 'y', 'source_exponents': (0, 0), 'test_exponents': (0, 1)},
    'quadrupolar_xy': {'plane': 'x', 'source_exponents': (0, 0), 'test_exponents': (0, 1)},
    'quadrupolar_yx': {'plane': 'y', 'source_exponents': (0, 0), 'test_exponents': (1, 0)},
}


#@for_all_test_contexts(excluding=exclude_contexts)
#@pytest.mark.parametrize('wake_type', ['WakeResonator', 'WakeTable'])
#@pytest.mark.parametrize('kind', [
#    ['longitudinal'],
#    ['constant_x', 'constant_y'],
#    ['dipolar_x','dipolar_y'],
#    ['dipolar_xy','dipolar_yx'],
#    ['quadrupolar_x','quadrupolar_y'],
#    ['quadrupolar_xy','quadrupolar_yx'],
#    ])
#def test_wake_kick_single_bunch(test_context, kind, wake_type):

p0c = 7e12
h_bunch = 600
circumference=26658.883
bucket_length = circumference/h_bunch/10
zeta_range = (-0.5*bucket_length, 0.5*bucket_length)
num_slices = 100

particle_dummy = xp.Particles(p0c=p0c)

kind = ['dipolar_x']

wf_analytic = xw.WakeResonator(
            kind=kind,
            r=1e8, q=1e5, f_r=1e3)

# would need beta_0 below here but we just need a larger value than the
# revolution period

t_table = np.linspace(-bucket_length/20/c_light/particle_dummy.beta0[0], circumference/c_light/particle_dummy.beta0[0], 1000)
table = pd.DataFrame()
table['time'] = t_table
for i_comp, comp in enumerate(wf_analytic.components):
    print(kind[i_comp])
    #table[kind[i_comp]] = comp.function_vs_t(t_table, particle_dummy.beta0, 1e-20)
    table[kind[i_comp]] = t_table

    wf = xw.WakeFromTable(table)

'''
wf = xw.WakeThickResistiveWall(
    kind=kind,
    resistivity=1e-7,
    radius=1e-2
    )
'''

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
                    #_context=test_context
)

i_source_x = 50
i_source_y = 50

displace_x = 2e-3
displace_y = 3e-3

# we displace after a certain particle index so that we check also that
# in the dipolar and quadrupolar cases there is no wake in front
#particles.x[i_source_x:] += displace_x
#particles.y[i_source_y:] += displace_y

particles.x[-i_source_x] += displace_x
particles.y[-i_source_y] += displace_y

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

import matplotlib.pyplot as plt
plt.plot(particles.zeta, particles.px, label='px')
plt.show()


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
                                                        dzeta=wf._wake_tracker.moments_data.dz)) * scale)

    xo.assert_allclose(getattr(particles, dict_p_bef[kk][0]) - dict_p_bef[kk][1],
                        expected, rtol=1e-4, atol=1e-20)

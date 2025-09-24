# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

import numpy as np
import pandas as pd

import xwakes as xw
import xtrack as xt
import xobjects as xo

from scipy.constants import c as clight

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

import matplotlib.pyplot as plt
plt.close('all')
plt.plot(p.zeta, p.delta, label='xwakes')
plt.plot(p_table.zeta, p_table.delta, '--', label='xwakes from table')
plt.plot(p_ref.zeta, p_ref.delta, '-.', label='pyht')

plt.legend()

plt.show()
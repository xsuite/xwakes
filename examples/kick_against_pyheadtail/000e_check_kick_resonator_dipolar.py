# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

import numpy as np
import pandas as pd

import xwakes as xw
import xobjects as xo
import xtrack as xt

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

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
sp1 = plt.subplot(211)
plt.plot(p_table.zeta, p.px, '-', label='xwakes resonator')
plt.plot(p_table.zeta, p_table.px, '--', label='xwakes from table')
plt.plot(p_ref.zeta, p_ref.px, '-.', label='pyht')
plt.legend()
plt.subplot(212, sharex=sp1)
plt.plot(p_table.zeta, p.py, '-', label='xwakes resonator')
plt.plot(p_table.zeta, p_table.py, '--', label='xwakes from table')
plt.plot(p_ref.zeta, p_ref.py, '-.', label='pyht')
plt.legend()
plt.legend()

plt.show()
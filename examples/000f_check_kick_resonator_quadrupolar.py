import numpy as np
import pandas as pd

import xwakes as xw

from scipy.constants import c as clight

import xtrack as xt

from xpart.pyheadtail_interface.pyhtxtparticles import PyHtXtParticles

p = xt.Particles(p0c=7e12, zeta=np.linspace(-1, 1, 100000))
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

# Build equivalent WakeFromTable
t_samples = np.linspace(-10/clight, 10/clight, 100000)
w_dipole_x_samples = wake.components[0].function_vs_t(t_samples, beta0=1.)
w_dipole_y_samples = wake.components[1].function_vs_t(t_samples, beta0=1.)
table = pd.DataFrame({'time': t_samples, 'dipolar_x': w_dipole_x_samples,
                        'dipolar_y': w_dipole_y_samples})
wake_from_table = xw.WakeFromTable(table)
wake_from_table.configure_for_tracking(zeta_range=(-1, 1), num_slices=50)

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

import matplotlib.pyplot as plt
plt.close('all')
plt.plot(p.zeta, p.px, label='xwakes')
plt.plot(p_table.zeta, p_table.px, '--', label='xwakes from table')
plt.plot(p_ref.zeta, p_ref.px, '-.', label='pyht')

plt.legend()

plt.show()
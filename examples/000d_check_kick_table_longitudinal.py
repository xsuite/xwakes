import numpy as np

import xwakes as xw
import xobjects as xo

from scipy.constants import c as clight

import xtrack as xt

from xpart.pyheadtail_interface.pyhtxtparticles import PyHtXtParticles

p = xt.Particles.merge([
    xt.Particles(p0c=7e12, zeta=np.linspace(-1e-3, 1e-3, 100000)),
    xt.Particles(p0c=7e12, zeta=1e-6+np.zeros(100000))
])

p_table = p.copy()
p_ref = p.copy()
p_ref = PyHtXtParticles.from_dict(p_ref.to_dict())

# Build equivalent WakeFromTable
table = xw.read_headtail_file('HLLHC_wake.dat', wake_file_columns=[
                     'time', 'longitudinal', 'dipole_x', 'dipole_y',
                     'quadrupole_x', 'quadrupole_y', 'dipole_xy',
                     'quadrupole_xy', 'dipole_yx', 'quadrupole_yx',
                     'constant_x', 'constant_y'])
wake_from_table = xw.WakeFromTable(table, columns=['time', 'longitudinal'])
wake_from_table.configure_for_tracking(zeta_range=(-2e-3, 2e-3), num_slices=1000)

from PyHEADTAIL.impedances.wakes import WakeTable, WakeField
from PyHEADTAIL.particles.slicing import UniformBinSlicer
slicer = UniformBinSlicer(n_slices=None,
                          z_sample_points=wake_from_table._wake_tracker.slicer.zeta_centers)
waketable = WakeTable('HLLHC_wake_long.dat', [
                     'time', 'longitudinal'])
wake_pyht = WakeField(slicer, waketable)

wake_from_table.track(p_table)
wake_pyht.track(p_ref)

assert np.max(p_ref.delta) > 1e-12
xo.assert_allclose(p_table.delta, p_ref.delta, atol=1e-16, rtol=0)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)

plt.plot(p_table.zeta, p_table.delta, '--', label='xwakes from table')
plt.plot(p_ref.zeta, p_ref.delta, '-.', label='pyht')

plt.legend()

plt.show()
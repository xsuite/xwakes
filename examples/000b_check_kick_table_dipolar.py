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

# Build equivalent WakeFromTable
table = xw.read_headtail_file('HLLHC_wake.dat', wake_file_columns=[
                     'time', 'longitudinal', 'dipole_x', 'dipole_y',
                     'quadrupole_x', 'quadrupole_y', 'dipole_xy',
                     'quadrupole_xy', 'dipole_yx', 'quadrupole_yx',
                     'constant_x', 'constant_y'])
wake_from_table = xw.WakeFromTable(table, columns=['dipole_x', 'dipole_y'])
wake_from_table.configure_for_tracking(zeta_range=(-1, 1), num_slices=100)

from PyHEADTAIL.impedances.wakes import WakeTable, WakeField
from PyHEADTAIL.particles.slicing import UniformBinSlicer
slicer = UniformBinSlicer(n_slices=None,
                          z_sample_points=wake_from_table._wake_tracker.slicer.zeta_centers)
waketable = WakeTable('HLLHC_wake_dip.dat', [
                     'time', 'dipole_x', 'dipole_y'])
wake_pyht = WakeField(slicer, waketable)

wake_from_table.track(p_table)
wake_pyht.track(p_ref)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
sp1 = plt.subplot(211)
plt.plot(p_table.zeta, p_table.px, '--', label='xwakes from table')
plt.plot(p_ref.zeta, p_ref.px, '-.', label='pyht')
plt.legend()
plt.subplot(212, sharex=sp1)
plt.plot(p_table.zeta, p_table.py, '--', label='xwakes from table')
plt.plot(p_ref.zeta, p_ref.py, '-.', label='pyht')
plt.legend()

plt.show()
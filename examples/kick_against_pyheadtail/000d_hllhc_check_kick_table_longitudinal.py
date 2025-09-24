# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

import numpy as np

import xwakes as xw
import xobjects as xo

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
table = xw.read_headtail_file(
    '../../test_data/headtail_format_table_hllhc/HLLHC_wake_flattop_nocrab.dat',
    wake_file_columns=[
                     'time', 'longitudinal', 'dipole_x', 'dipole_y',
                     'quadrupole_x', 'quadrupole_y', 'dipole_xy',
                     'quadrupole_xy', 'dipole_yx', 'quadrupole_yx',
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
    '../../test_data/headtail_format_table_hllhc/HLLHC_wake_long.dat',
    ['time', 'longitudinal'])
wake_pyht = WakeField(slicer, waketable)

wake_from_table.track(p_table)
wake_pyht.track(p_ref)

assert np.max(p_ref.delta) > 1e-12
assert (np.abs(p_ref.delta - p_table.delta) > 1e-16).sum() <= 1 # 1 particle can be different
                                                                # PyHEADTAIL has a strange discontinuity

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)

plt.plot(p_table.zeta, p_table.delta, '--', label='xwakes from table')
plt.plot(p_ref.zeta, p_ref.delta, '-.', label='pyht')

plt.legend()

plt.show()

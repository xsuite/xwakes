# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

import numpy as np
import pandas as pd

import xwakes as xw
import xtrack as xt
import xobjects as xo

from xpart.pyheadtail_interface.pyhtxtparticles import PyHtXtParticles

p = xt.Particles(p0c=7e12, zeta=np.linspace(-1, 1, 100000))
p.x[p.zeta > 0] += 1e-3
p.y[p.zeta > 0] += 1e-3
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
wake_from_table = xw.WakeFromTable(table, columns=['dipole_x', 'dipole_y'])
wake_from_table.configure_for_tracking(zeta_range=(-1, 1), num_slices=100)

# Zotter convention
assert table['dipole_x'].values[1] > 0
assert table['dipole_y'].values[1] > 0

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

from PyHEADTAIL.impedances.wakes import WakeTable, WakeField
from PyHEADTAIL.particles.slicing import UniformBinSlicer
slicer = UniformBinSlicer(n_slices=None,
                          z_sample_points=wake_from_table._wake_tracker.slicer.zeta_centers)
waketable = WakeTable(
    '../../test_data/headtail_format_table_hllhc/HLLHC_wake_dip.dat',
    ['time', 'dipole_x', 'dipole_y'])
wake_pyht = WakeField(slicer, waketable)

wake_from_table.track(p_table)
wake_pyht.track(p_ref)

xo.assert_allclose(p_table.px, p_ref.px, atol=1e-30, rtol=2e-3)
xo.assert_allclose(p_table.py, p_ref.py, atol=1e-30, rtol=2e-3)

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

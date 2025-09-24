# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
This file generates a headtail formatted table with dipolar and quadrupolar
wakefields only
"""

import numpy as np

data = np.loadtxt('HLLHC_wake_flattop_nocrab.dat')

out = np.zeros((data.shape[0], 5), dtype=float)
out[:, 0] = data[:, 0]
out[:, 1] = data[:, 2]
out[:, 2] = data[:, 3]
out[:, 3] = data[:, 4]
out[:, 4] = data[:, 5]

np.savetxt('HLLHC_wake_dip_quad.dat', out, delimiter='\t')
np.savetxt('HLLHC_wake_dip.dat', out[:, :3], delimiter='\t')
np.savetxt('HLLHC_wake_quad.dat', out[:, [0, 3, 4]], delimiter='\t')

out_long = np.zeros((data.shape[0], 2), dtype=float)
out_long[:, 0] = data[:, 0]
out_long[:, 1] = data[:, 1]
np.savetxt('HLLHC_wake_long.dat', out_long, delimiter='\t')
# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

import pathlib

import numpy as np
from scipy.constants import c as clight

import xwakes as xw
import xtrack as xt
import xobjects as xo
import xpart as xp

test_data_folder = pathlib.Path(__file__).parent.joinpath(
    'test_data').absolute()

def test_xwakes_kick_vs_pyheadtail_table():

    p = xt.Particles(mass0 = xp.PROTON_MASS_EV, gamma0 = 3.14, zeta=np.linspace(-1, 1, 100000))
    p.x[p.zeta > 0] += 1e-3
    p.y[p.zeta > 0] += 1e-3
    p_table = p.copy()

    # Build equivalent WakeFromTable
    table = xw.read_headtail_file(
        test_data_folder / 'PS_wall_impedance_Ekin_2.0GeV.wake',
        wake_file_columns=['time', 'dipolar_x', 'dipolar_y','quadrupolar_x', 'quadrupolar_y'])
    
    wake_from_table = xw.WakeFromTable(table, columns=['time', 'dipolar_x', 'dipolar_y','quadrupolar_x', 'quadrupolar_y'], 
                                       method="Integrated")
    wake_from_table.configure_for_tracking(zeta_range=(-1, 1), num_slices=100)

    # Zotter convention
    assert table['dipolar_x'].values[1] > 0
    assert table['dipolar_y'].values[1] > 0
    assert table['quadrupolar_x'].values[1] < 0
    assert table['quadrupolar_y'].values[1] < 0

    
    wake_from_table.track(p_table)
    p_ref = np.load(test_data_folder / 'particles_pyht.npy')

    xo.assert_allclose(p_table.px, p_ref[3], atol=1e-30, rtol=2e-3)
    xo.assert_allclose(p_table.py, p_ref[4], atol=1e-30, rtol=2e-3)
# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

import pathlib

import numpy as np
from scipy.constants import c as clight
from scipy.constants import e ,m_p

import xwakes as xw
import xtrack as xt
import xobjects as xo
import xpart as xp

from PyHEADTAIL.impedances.wakes import WakeTable, WakeField
from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.particles.particles import Particles
from PyHEADTAIL.machines.synchrotron import *

test_data_folder = pathlib.Path(__file__).parent.joinpath(
    'test_data').absolute()

def test_xwakes_kick_vs_pyheadtail_table():

    gamma = 3.14
    radious = 100
    zmin = -1
    zmax = 1
    Npart = 100000
    kick = 1e-3
    n_slices = 1000
    epsilon_range = 0.1

    ### Xwakes
    p = xt.Particles(mass0 = xp.PROTON_MASS_EV, gamma0 = gamma, zeta=np.linspace(zmin, zmax, Npart))
    p.x[p.zeta > 0] += kick
    p.y[p.zeta > 0] += kick
    p_xwakes = p.copy()

    table = xw.read_headtail_file(
        test_data_folder / 'integration_method/PS_wall_impedance_Ekin_2.0GeV.wake',
        wake_file_columns=['time', 'dipolar_x', 'dipolar_y','quadrupolar_x', 'quadrupolar_y'])

    wake_from_table = xw.WakeFromTable(table, columns=['dipolar_x', 'dipolar_y','quadrupolar_x', 'quadrupolar_y'],
                                       method="Integrated")
    wake_from_table.configure_for_tracking(zeta_range=(zmin-epsilon_range, zmax+epsilon_range), num_slices=n_slices)


    ### PyHEADTAIL
    x_pt = np.zeros(Npart)
    y_pt = np.zeros(Npart)
    z_pt = np.linspace(zmin, zmax, Npart)
    xp_pt = np.zeros(Npart)
    yp_pt = np.zeros(Npart)
    dp_pt = np.zeros(Npart)

    coords_n_momenta_dict = {
        'x':x_pt, 'y': y_pt, 'z': z_pt,
        'xp': xp_pt, 'yp': yp_pt, 'dp': dp_pt
    }

    p_pyht = Particles(Npart, 1, e, m_p, 2*np.pi*radious, gamma, coords_n_momenta_dict)
    p_pyht.x[p_pyht.z > 0] += kick
    p_pyht.y[p_pyht.z > 0] += kick

    slicer = UniformBinSlicer(n_slices=n_slices, z_sample_points=wake_from_table._wake_tracker.slicer.zeta_centers)
    waketable = WakeTable(test_data_folder / 'integration_method/PS_wall_impedance_Ekin_2.0GeV.wake',
        ['time', 'dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y'], method='Integrated')
    wake_pyht = WakeField(slicer, waketable)


    ### Comparison
    wake_from_table.track(p_xwakes)
    wake_pyht.track(p_pyht)

    xo.assert_allclose(p_xwakes.px, p_pyht.xp, atol=1e-30, rtol=2e-2)
    xo.assert_allclose(p_xwakes.py, p_pyht.yp, atol=1e-30, rtol=2e-2)

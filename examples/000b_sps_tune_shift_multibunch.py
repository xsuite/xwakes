# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

import numpy as np
import pathlib
import h5py

import xwakes as xw
import xtrack as xt
import xpart as xp
import nafflib

# a few machine parameters
p0c = 26e9
circumference = 6911.5662

h_RF = [4620, 4*4620]
V_RF = [4.5e6, 0.45e6]
dphi_RF = [180, 0]
f_rev = 43347.25733575443
f_RF = [f_rev*h for h in h_RF]

bucket_length = circumference / h_RF[0]
bunch_spacing_buckets = 5

Q_x = 20.13
Q_y = 20.18
xi = 0.05

# prepare the filling scheme
filling_scheme = np.zeros(int(h_RF[0]/bunch_spacing_buckets))
n_bunches = 12
filling_scheme[0:n_bunches] = 1

# initialize the one turn map
one_turn_map = xt.LineSegmentMap(
    length=circumference, betx=54.65, bety=54.51,
    dnqx=np.array([Q_x, xi*Q_x]), dnqy=np.array([Q_y, xi*Q_y]),
    longitudinal_mode='nonlinear',
    voltage_rf=V_RF, frequency_rf=f_RF,
    lag_rf=dphi_RF, momentum_compaction_factor=0.0031
)

# initialize the wakes
test_data_folder = pathlib.Path(__file__).parent.joinpath('../test_data').absolute()
wake_table_name_kickers = test_data_folder.joinpath('sps_wake_2022/MKP_wake.dat')
wake_file_columns = ['time', 'dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y']
components = ['dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y']
wake_df_kickers = xw.read_headtail_file(wake_table_name_kickers,
                                        wake_file_columns)
wf_kickers = xw.WakeFromTable(table=wake_df_kickers, columns=components)

wake_table_name_wall = test_data_folder.joinpath('sps_wake_2022/wall_wake.dat')
wake_df_wall = xw.read_headtail_file(wake_table_name_wall,
                                     wake_file_columns)
wf_wall = xw.WakeFromTable(table=wake_df_wall, columns=components)

# sum the wakes and prepare for tracking
wf_sps = wf_kickers + wf_wall
wf_sps.configure_for_tracking(
    zeta_range=(-0.5*bucket_length, 0.5*bucket_length),
    num_slices=100,
    bunch_spacing_zeta=5*bucket_length,
    num_turns=37,
    circumference=circumference,
    filling_scheme=filling_scheme
)

# initialize a damper with 100 turns gain
transverse_damper = xw.TransverseDamper(
    gain_x=2/100, gain_y=2/100,
    zeta_range=(-0.5*bucket_length, 0.5*bucket_length),
    num_slices=100,
    bunch_spacing_zeta=5*bucket_length,
    circumference=circumference,
    filling_scheme=filling_scheme
)

# initialize a monitor for the average transverse positions
monitor = xw.CollectiveMonitor(
    base_file_name=f'sps_tune_shift',
    monitor_bunches=True,
    monitor_slices=False,
    monitor_particles=False,
    flush_data_every=100,
    stats_to_store=['mean_x', 'mean_y'],
    backend='hdf5',
    zeta_range=(-0.5*bucket_length, 0.5*bucket_length),
    num_slices=100,
    bunch_spacing_zeta=5*bucket_length,
    filling_scheme=filling_scheme
)

elements = [one_turn_map, wf_sps, transverse_damper, monitor]
element_names = ['one_turn_map', 'wake', 'transverse_damper', 'monitor']
line = xt.Line(elements, element_names=element_names)

line.particle_ref = xt.Particles(p0c=26e9)
line.build_tracker()

# initialize a matched gaussian bunch
particles = xp.generate_matched_gaussian_multibunch_beam(
    bunch_num_particles=100_000,
    bunch_intensity_particles=2.3e11,
    nemitt_x=2e-6, nemitt_y=2e-6,
    sigma_z=0.21734953205,
    filling_scheme=filling_scheme,
    bucket_length=bucket_length,
    bunch_spacing_buckets=bunch_spacing_buckets,
    circumference=circumference,
    line=line,
)

# apply a kick to the particles
particles.px += 1e-3
particles.py += 1e-3

# track the particles
line.track(particles, num_turns=101, with_progress=1)

qx_bunch = []
qy_bunch = []

# read mean positions from the monitor file
with h5py.File(monitor.base_file_name + '_bunches.h5', 'r') as h5file:
    for bunch in h5file.keys():
        mean_x = h5file[bunch]['mean_x'][:]
        mean_y = h5file[bunch]['mean_y'][:]

        qx_bunch.append(nafflib.tune(mean_x))
        qy_bunch.append(nafflib.tune(mean_y))

import matplotlib.pyplot as plt
plt.close('all')
plt.plot(qx_bunch, 'rx', label='Q_x')
plt.plot(qy_bunch, 'bo', label='Q_y')
plt.xlabel('Bunch number')
plt.ylabel('Tune')
plt.legend()
plt.show()

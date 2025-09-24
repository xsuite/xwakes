# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

import xtrack as xt
import xpart as xp
import xfields as xf
import xwakes as xw
import xobjects as xo
from scipy.constants import e as qe, c as c_light
from scipy.signal import hilbert
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n_turns = 10000
p0c = 7000e9

# Filling scheme
filling_scheme = np.zeros(3564, dtype=int)
filling_scheme[0] = 1
filling_scheme[1] = 1

n_turns_wake = 1
circumference = 26658.8832
bucket_length_m = circumference / 35640
num_slices = 100

wake_table_name = xf.general._pkg_root.joinpath(
    '../test_data/HLLHC_wake.dat')
wake_file_columns = ['time', 'longitudinal', 'dipolar_x', 'dipolar_y',
                    'quadrupolar_x', 'quadrupolar_y', 'dipolar_xy',
                    'quadrupolar_xy', 'dipolar_yx', 'quadrupolar_yx',
                    'constant_x', 'constant_y']
wake_df = xw.read_headtail_file(wake_table_name,
                                wake_file_columns)
wf= xw.WakeFromTable(wake_df, columns=['dipolar_x', 'dipolar_y'],
)
wf.configure_for_tracking(zeta_range=(-0.5*bucket_length_m, 0.5*bucket_length_m),
                        num_slices=num_slices,
                        bunch_spacing_zeta=circumference/3564,
                        filling_scheme=filling_scheme,
                        num_turns=n_turns_wake,
                        circumference=circumference
                        )

part_aux = xt.Particles(p0c=p0c)

f_rev = part_aux.beta0[0]*c_light/circumference
f_rf = f_rev*35640

one_turn_map = xt.LineSegmentMap(
    length=circumference, betx=70., bety=80.,
    qx=62.31, qy=60.32,
    longitudinal_mode='nonlinear',
    dqx=-10., dqy=-10.,  # <-- to see fast mode-0 instability
    voltage_rf=16e6, frequency_rf=f_rf,
    lag_rf=180, momentum_compaction_factor=53.86**-2
)

line = xt.Line(elements=[one_turn_map, wf],
            element_names=['one_turn_map', 'wf'])
line.particle_ref = xt.Particles(p0c=p0c)
line.build_tracker()

# Only train of identical gaussian bunches for now...
# Need to develop a way of assembling more complex train structures and
# handling parallel simulation in that case
particles = xp.generate_matched_gaussian_multibunch_beam(
            line=line,
            filling_scheme=filling_scheme,
            bunch_num_particles=100_000, # This needs to be renamed
            bunch_intensity_particles=2.3e11, # This needs to be renamed
            nemitt_x=2e-6, nemitt_y=2e-6, sigma_z=0.075,
            bunch_spacing_buckets=10,
            bucket_length=bucket_length_m,
)

particles.x += 1e-3
particles.y += 1e-3

mean_x_xt = np.zeros(n_turns)
mean_y_xt = np.zeros(n_turns)

plt.ion()

fig1 = plt.figure(figsize=(6.4*1.7, 4.8))
ax_x = fig1.add_subplot(121)
line1_x, = ax_x.plot(mean_x_xt, 'r-', label='average x-position')
line2_x, = ax_x.plot(mean_x_xt, 'm-', label='exponential fit')
ax_x.set_ylim(-3.5, -1)
ax_x.set_xlim(0, n_turns)
ax_y = fig1.add_subplot(122, sharex=ax_x)
line1_y, = ax_y.plot(mean_y_xt, 'b-', label='average y-position')
line2_y, = ax_y.plot(mean_y_xt, 'c-', label='exponential fit')
ax_y.set_ylim(-3.5, -1)
ax_y.set_xlim(0, n_turns)

plt.xlabel('turn')
plt.ylabel('log10(average x-position)')
plt.legend()


turns = np.linspace(0, n_turns - 1, n_turns)

for i_turn in range(n_turns):
    line.track(particles, num_turns=1)

    prof_num_part = wf._wake_tracker.moments_data.get_moment_profile(moment_name='num_particles', i_turn=0)
    mask_nonzero = prof_num_part[1] > 0
    prof_x = wf._wake_tracker.moments_data.get_moment_profile(moment_name='x', i_turn=0)[1]
    prof_y = wf._wake_tracker.moments_data.get_moment_profile(moment_name='y', i_turn=0)[1]
    mean_x_xt[i_turn] = np.average(prof_x[mask_nonzero], weights=prof_num_part[1][mask_nonzero])
    mean_y_xt[i_turn] = np.average(prof_y[mask_nonzero], weights=prof_num_part[1][mask_nonzero])

    if i_turn % 50 == 0:
        print(f'Turn: {i_turn}')

    if i_turn % 50 == 0 and i_turn > 1:
        i_fit_end = i_turn #np.argmax(mean_x_xt)
        i_fit_start = int(i_fit_end * 0.9)

        # compute x instability growth rate
        ampls_x_xt = np.abs(hilbert(mean_x_xt))
        fit_x_xt = linregress(turns[i_fit_start: i_fit_end],
                            np.log(ampls_x_xt[i_fit_start: i_fit_end]))

        # compute y instability growth rate

        ampls_y_xt = np.abs(hilbert(mean_y_xt))
        fit_y_xt = linregress(turns[i_fit_start: i_fit_end],
                            np.log(ampls_y_xt[i_fit_start: i_fit_end]))

        line1_x.set_xdata(turns[:i_turn])
        line1_x.set_ydata(np.log10(np.abs(mean_x_xt[:i_turn])))
        line2_x.set_xdata(turns[:i_turn])
        line2_x.set_ydata(np.log10(np.exp(fit_x_xt.intercept +
                                        fit_x_xt.slope*turns[:i_turn])))

        line1_y.set_xdata(turns[:i_turn])
        line1_y.set_ydata(np.log10(np.abs(mean_y_xt[:i_turn])))
        line2_y.set_xdata(turns[:i_turn])
        line2_y.set_ydata(np.log10(np.exp(fit_y_xt.intercept +
                                        fit_y_xt.slope*turns[:i_turn])))
        print(f'xtrack h growth rate: {fit_x_xt.slope}')
        print(f'xtrack v growth rate: {fit_y_xt.slope}')

        fig1.canvas.draw()
        fig1.canvas.flush_events()

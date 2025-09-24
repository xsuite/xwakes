# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

import xtrack as xt
import xpart as xp
import xwakes as xw
import time

import numpy as np
import matplotlib.pyplot as plt

n_turns = 10

one_turn_map = xt.LineSegmentMap(
    length=26000, betx=50., bety=40.,
    qx=62.28, qy=62.31,
    longitudinal_mode='linear_fixed_qs',
    qs=1e-3, bets=100
)

filling_scheme = np.zeros(3564)
filling_scheme[0] = 1
filling_scheme[1] = 1
bunch_selection = [0, 1]

wake = xw.WakeResonator(
    kind='dipolar_x',
    r=1e8, q=1e5, f_r=600e6)

wake.configure_for_tracking(zeta_range=(-20e-2, 20e-2), num_slices=500,
    filling_scheme=filling_scheme,
    bunch_spacing_zeta=26000/3564,
    bunch_selection=bunch_selection
    )

line = xt.Line(elements=[one_turn_map, wake])
line.build_tracker()
line.particle_ref=xp.Particles(p0c=7000e9)

particles = xp.generate_matched_gaussian_multibunch_beam(
    particle_ref=line.particle_ref,
    filling_scheme=filling_scheme,
    bunch_num_particles=100000,
    bunch_intensity_particles=1e11,
    nemitt_x=1e-6, nemitt_y=2e-6,
    sigma_z=0.08,
    line=line,
    bunch_spacing_buckets=10,
    bucket_length=26000/3564/10,
    bunch_selection=bunch_selection
)

mask_shift = (particles.zeta > 1.5e-3) & (particles.zeta < 2.5e-3)
particles.x = 0
particles.x[mask_shift] += 1e-3
particles.px = 0
particles_megabunch = particles.copy()

time_0 = time.time()
for i_turn in range(n_turns):
    wake.track(particles)
time_1 = time.time()

print(f'Time for {n_turns} turns: {time_1 - time_0} s')

wake_megabunch = xw.WakeResonator(
    kind='dipolar_x',
    r=1e8, q=1e5, f_r=600e6)
dzeta = wake.slicer.dzeta
zeta_range_megabunch = (-20e-2 - 26000/3654*3, 20e-2)
num_slices_megabunch = int(
    np.round((zeta_range_megabunch[1] - zeta_range_megabunch[0]) / dzeta))
wake_megabunch.configure_for_tracking(zeta_range=zeta_range_megabunch,
                                      num_slices=num_slices_megabunch
    )

line_megabunch = xt.Line(elements=[one_turn_map, wake_megabunch])
line_megabunch.build_tracker()
line_megabunch.particle_ref=xp.Particles(p0c=7000e9)

time_0 = time.time()
for i_turn in range(n_turns):
    wake_megabunch.track(particles_megabunch)
time_1 = time.time()

print(f'Time for {n_turns} turns with megabunch: {time_1 - time_0} s')

plt.close('all')
plt.plot(particles_megabunch.zeta, particles_megabunch.px, 'rx')
plt.plot(particles.zeta, particles.px, 'b.')
plt.show()
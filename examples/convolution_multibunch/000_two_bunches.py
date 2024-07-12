import xtrack as xt
import xpart as xp
import xwakes as xw
import time

import numpy as np

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

wake = xw.WakeResonator(
    kind='dipolar_x',
    r=1e4, q=1, f_r=600e6)
wake.configure_for_tracking(zeta_range=(-20e-2, 20e-2), num_slices=500,
    filling_scheme=filling_scheme,
    bunch_spacing_zeta=26000/3564,
    bunch_numbers=[0, 1]
    )


line = xt.Line(elements=[one_turn_map, wake])
line.build_tracker()
line.particle_ref=xp.Particles(p0c=7000e9)

particles = xp.generate_matched_gaussian_multibunch_beam(
    particle_ref=line.particle_ref,
    filling_scheme=filling_scheme,
    num_particles=100000,
    total_intensity_particles=1e11,
    nemitt_x=1e-6, nemitt_y=2e-6,
    sigma_z=0.08,
    line=line,
    bunch_spacing_buckets=10,
    bucket_length=26000/3564/10,
    bunch_numbers=[0, 1]
)

time_0 = time.time()
for i_turn in range(n_turns):
    line.track(particles)

time_1 = time.time()
print(f'Time for {n_turns} turns: {time_1 - time_0} s')

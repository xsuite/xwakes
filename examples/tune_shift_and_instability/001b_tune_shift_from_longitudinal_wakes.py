# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

import pandas as pd
import numpy as np
from scipy.constants import c as clight

import xtrack as xt
import xpart as xp
import xwakes as xw
import xobjects as xo

table = pd.DataFrame({'time': [0, 10],
                      'longitudinal': [1e13, 1e13]})

wake = xw.WakeFromTable(table)
wake.configure_for_tracking(zeta_range=(-20e-2, 20e-2), num_slices=100)

assert len(wake.components) == 1
assert wake.components[0].plane == 'z'
assert wake.components[0].source_exponents == (0, 0)
assert wake.components[0].test_exponents == (0, 0)

beta0_check = 0.3
xo.assert_allclose(wake.components[0].function_vs_t(0.1, beta0=beta0_check, dt=1e-20), 1e13, rtol=1e-10, atol=0)
xo.assert_allclose(wake.components[0].function_vs_t(0, beta0=beta0_check, dt=1e-20), 1e13/2, rtol=1e-10, atol=0)
xo.assert_allclose(wake.components[0].function_vs_t(-0.1, beta0=beta0_check, dt=1e-20), 0, rtol=1e-10, atol=0)
xo.assert_allclose(wake.components[0].function_vs_t(10 - 0.1, beta0=beta0_check, dt=1e-20), 1e13, rtol=1e-10, atol=0)
xo.assert_allclose(wake.components[0].function_vs_t(10 + 0.1, beta0=beta0_check, dt=1e-20), 0, rtol=1e-10, atol=0)

xo.assert_allclose(wake.components[0].function_vs_zeta(-0.1, beta0=beta0_check, dzeta=1e-20), 1e13, rtol=1e-10, atol=0)
xo.assert_allclose(wake.components[0].function_vs_zeta(0, beta0=beta0_check, dzeta=1e-20), 1e13/2, rtol=1e-10, atol=0)
xo.assert_allclose(wake.components[0].function_vs_zeta(0.1, beta0=beta0_check, dzeta=1e-20), 0, rtol=1e-10, atol=0)
xo.assert_allclose(
    wake.components[0].function_vs_zeta(-beta0_check * clight * 10 + 0.1, beta0=beta0_check, dzeta=1e-20),
    1e13, rtol=1e-10, atol=0)
xo.assert_allclose(
    wake.components[0].function_vs_zeta(-beta0_check * clight * 10 - 0.1, beta0=beta0_check, dzeta=1e-20),
    0, rtol=1e-10, atol=0)

one_turn_map = xt.LineSegmentMap(length=1, qx=0.28, qy=0.31, qs=5e-3, bets=100)

line_no_wake = xt.Line(elements=[one_turn_map])
line_with_wake = xt.Line(elements=[one_turn_map, wake])

line_no_wake.particle_ref = xt.Particles(p0c=2e9)
line_with_wake.particle_ref = xt.Particles(p0c=2e9)

line_no_wake.build_tracker()
line_with_wake.build_tracker()

p = xp.generate_matched_gaussian_bunch(line=line_no_wake,
            num_particles=1000, nemitt_x=1e-6, nemitt_y=1e-6, sigma_z=0.07,
            total_intensity_particles=1e11)
p.zeta += 5e-3

p0 = p.copy()

mylog = xt.Log(mean_zeta=lambda line, part: part.zeta.mean(),
               mean_delta=lambda line, part: part.delta.mean())

line_no_wake.track(p0, num_turns=3000,
                   log=mylog,
                   with_progress=10)
log_no_wake = line_no_wake.log_last_track

line_with_wake.track(p, num_turns=3000,
                        log=mylog,
                        with_progress=10)
log_with_wake = line_with_wake.log_last_track

import nafflib as nl

tune_no_wake = nl.get_tune(log_no_wake[f'mean_zeta'])
tune_with_wake = nl.get_tune(log_with_wake[f'mean_zeta']-np.mean(log_with_wake[f'mean_zeta']))

print(f'Tune without wake: {tune_no_wake}')
print(f'Tune with wake: {tune_with_wake}')
print(f'Tune shift: {tune_with_wake - tune_no_wake}')

xo.assert_allclose(tune_no_wake, 5e-3, atol=1e-6, rtol=0)
xo.assert_allclose(tune_with_wake, tune_no_wake + 1.15e-3, atol=1e-4, rtol=0)

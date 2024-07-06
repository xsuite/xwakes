import pandas as pd

import xtrack as xt
import xpart as xp
import xwakes as xw

table = pd.DataFrame({'time': [0, 10],
                      'dipole_x': [1e16, 1e16]})

wake = xw.WakeFromTable(table)
wake.configure_for_tracking(zeta_range=(-10e-2, 10e-2), num_slices=1000)

ot_map = xt.LineSegmentMap(length=1, qx=0.28, qy=0.31, qs=1e-3, bets=100)

line_no_wake = xt.Line(elements=[ot_map])
line_with_wake = xt.Line(elements=[ot_map, wake])

line_no_wake.particle_ref = xt.Particles(p0c=2e9)
line_with_wake.particle_ref = xt.Particles(p0c=2e9)

line_no_wake.build_tracker()
line_with_wake.build_tracker()

p = xp.generate_matched_gaussian_bunch(line=line_no_wake,
            num_particles=1000, nemitt_x=1e-6, nemitt_y=1e-6, sigma_z=0.07,
            total_intensity_particles=1e11)

p0 = p.copy()

line_no_wake.track(p0, num_turns=1000,
                   log=xt.Log(mean_x=lambda line, part: part.x.mean()),
                   with_progress=10)
log_no_wake = line_no_wake.log_last_track

line_with_wake.track(p, num_turns=1000,
                        log=xt.Log(mean_x=lambda line, part: part.x.mean()),
                        with_progress=10)
log_with_wake = line_with_wake.log_last_track

import nafflib as nl

qx_no_wake = nl.get_tune(log_no_wake['mean_x'])
qx_with_wake = nl.get_tune(log_with_wake['mean_x'])

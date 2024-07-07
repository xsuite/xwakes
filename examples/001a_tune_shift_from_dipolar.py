import pandas as pd

import xtrack as xt
import xpart as xp
import xwakes as xw
import xobjects as xo

plane = 'y'
wake_type = 'quadrupolar'

table = pd.DataFrame({'time': [0, 10],
                      f'{wake_type}_{plane}': [1e16, 1e16]})

wake = xw.WakeFromTable(table)
wake.configure_for_tracking(zeta_range=(-10e-2, 10e-2), num_slices=1000)

assert len(wake.components) == 1
assert wake.components[0].plane == plane
assert wake.components[0].source_exponents == {
    'dipolar_x': (1, 0), 'dipolar_y': (0, 1),
    'quadrupolar_x': (0, 0), 'quadrupolar_y': (0, 0)}[f'{wake_type}_{plane}']
assert wake.components[0].test_exponents == {
    'dipolar_x': (0, 0), 'dipolar_y': (0, 0),
    'quadrupolar_x': (1, 0), 'quadrupolar_y': (0, 1)}[f'{wake_type}_{plane}']

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
p.x += 1e-3
p.y += 1e-3

p0 = p.copy()

mylog = xt.Log(mean_x=lambda line, part: part.x.mean(),
               mean_y=lambda line, part: part.y.mean())

line_no_wake.track(p0, num_turns=1000,
                   log=mylog,
                   with_progress=10)
log_no_wake = line_no_wake.log_last_track

line_with_wake.track(p, num_turns=1000,
                        log=mylog,
                        with_progress=10)
log_with_wake = line_with_wake.log_last_track

import nafflib as nl

tune_no_wake = nl.get_tune(log_no_wake[f'mean_{plane}'])
tune_with_wake = nl.get_tune(log_with_wake[f'mean_{plane}'])

print(f'Tune without wake: {tune_no_wake}')
print(f'Tune with wake: {tune_with_wake}')

# Expect negative tune shift for positive dipolar wake
xo.assert_allclose(tune_no_wake, {'x': 0.28, 'y': 0.31}[plane], atol=1e-6, rtol=0)
xo.assert_allclose(tune_with_wake, tune_no_wake - 6e-3, atol=1.5e-3, rtol=0)

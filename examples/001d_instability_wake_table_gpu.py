import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import linregress

import xtrack as xt
import xpart as xp
import xwakes as xw
import xfields as xf
import xobjects as xo

context = xo.ContextCupy(device=3)
#context = xo.ContextCpu()

# Simulation settings
n_turns = 1000

wake_table_filename = xf.general._pkg_root.joinpath(
    '../test_data/HLLHC_wake.dat')
wake_file_columns = ['time', 'longitudinal', 'dipole_x', 'dipole_y',
                     'quadrupole_x', 'quadrupole_y', 'dipole_xy',
                     'quadrupole_xy', 'dipole_yx', 'quadrupole_yx',
                     'constant_x', 'constant_y']
mytable = xw.read_headtail_file(
    wake_file=wake_table_filename,
    wake_file_columns=wake_file_columns
)
wf = xw.WakeFromTable(
    table=mytable,
    columns=['dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y'],
)
wf.configure_for_tracking(
    zeta_range=(-0.375, 0.375),
    num_slices=100,
    num_turns=1,
    _context=context
)

one_turn_map = xt.LineSegmentMap(
    length=27e3, betx=70., bety=80.,
    qx=62.31, qy=60.32,
    longitudinal_mode='linear_fixed_qs',
    dqx=-10., dqy=-10.,  # <-- to see fast mode-0 instability
    qs=2e-3, bets=731.27,
    _context=context
)

# Generate line
line = xt.Line(elements=[one_turn_map, wf],
               element_names=['one_turn_map', 'wf'])


line.particle_ref = xt.Particles(p0c=7e12, _context=context)
line.build_tracker(_context=context)

# Generate particles
particles = xp.generate_matched_gaussian_bunch(line=line,
                    num_particles=40_000_000, total_intensity_particles=2.3e11,
                    nemitt_x=2e-6, nemitt_y=2e-6, sigma_z=0.075,
                    _context=context)

# Apply a distortion to the bunch to trigger an instability
amplitude = 1e-3
particles.x += amplitude
particles.y += amplitude

flag_plot = True

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

import time

line.track(particles, num_turns=1)

start = time.time()

for i_turn in range(n_turns):
    line.track(particles, num_turns=1)

    if i_turn % 50 == 0:
        print(f'Turn: {i_turn}')

    '''
    mean_x_xt[i_turn] = np.mean(particles.x)
    mean_y_xt[i_turn] = np.mean(particles.y)

    if i_turn % 50 == 0 and i_turn > 1:
        i_fit_end = np.argmax(mean_x_xt)  # i_turn
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

        out_folder = '.'
        np.save(f'{out_folder}/mean_x.npy', mean_x_xt)
        np.save(f'{out_folder}/mean_y.npy', mean_y_xt)
    '''

end = time.time()
print(f'Time: {end - start}')
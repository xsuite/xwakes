import xfields as xf
import xtrack as xt
import numpy as np

slicer = xf.UniformBinSlicer(
    zeta_range=(-1, 1),
    num_slices=10,
    filling_scheme=[1, 0, 1, 1],
    bunch_spacing_zeta=5)

slicer2 = xf.UniformBinSlicer(
    zeta_range=(-1, 1),
    num_slices=10,
    filling_scheme=[1, 0, 1, 1],
    bunch_spacing_zeta=5,
    bunch_numbers=[1, 2])

assert (slicer.filled_slots == np.array([0, 2, 3])).all()
assert (slicer2.filled_slots == np.array([0, 2, 3])).all()
assert (slicer.bunch_numbers == np.array([0, 1, 2])).all()
assert (slicer2.bunch_numbers == np.array([1, 2])).all()
assert slicer.num_bunches == 3
assert slicer2.num_bunches == 2




particles = xt.Particles(p0c=7000e9,
                         zeta=np.linspace(-20, 20, 1000000))

slicer.slice(particles)

slicer2.slice(particles)

import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1)
ax1 = plt.subplot(311)
ax1.plot(slicer.zeta_centers.T, slicer.num_particles.T, '.-')

ax2 = plt.subplot(312, sharex=ax1)
ax2.plot(slicer2.zeta_centers.T, slicer2.num_particles.T, '.-')


plt.show()


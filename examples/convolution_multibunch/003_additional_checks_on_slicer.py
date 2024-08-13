import xfields as xf
import xtrack as xt
import numpy as np
import xobjects as xo

buffer_round_trip = True
from xfields.beam_elements.element_with_slicer import ElementWithSlicer

ele = ElementWithSlicer(
    zeta_range=(-1, 1),
    num_slices=10,
    filling_scheme=[1, 0, 1, 1],
    bunch_spacing_zeta=5,
    with_compressed_profile=True)
slicer = ele.slicer

ele1 = ElementWithSlicer(
    zeta_range=(-1, 1),
    num_slices=10,
    filling_scheme=[1, 0, 1, 1],
    bunch_spacing_zeta=5,
    bunch_numbers=[1, 2],
    with_compressed_profile=True)
slicer1 = ele1.slicer

ele2 = ElementWithSlicer(
    zeta_range=(-1, 1),
    num_slices=10,
    filling_scheme=[1, 0, 1, 1],
    bunch_spacing_zeta=5,
    bunch_numbers=[0],
    with_compressed_profile=True)
slicer2 = ele2.slicer

if buffer_round_trip:
    slicer = xf.UniformBinSlicer._from_npbuffer(slicer._to_npbuffer())
    slicer1 = xf.UniformBinSlicer._from_npbuffer(slicer1._to_npbuffer())
    slicer2 = xf.UniformBinSlicer._from_npbuffer(slicer2._to_npbuffer())

particles = xt.Particles(p0c=7000e9,
                         zeta=np.linspace(-20, 20, 1000000))

ele.track(particles)
ele1.track(particles)
ele2.track(particles)

assert (slicer.filled_slots == np.array([0, 2, 3])).all()
assert (slicer1.filled_slots == np.array([0, 2, 3])).all()
assert (slicer2.filled_slots == np.array([0, 2, 3])).all()
assert (slicer.bunch_numbers == np.array([0, 1, 2])).all()
assert (slicer1.bunch_numbers == np.array([1, 2])).all()
assert (slicer2.bunch_numbers == np.array([0])).all()
assert slicer.num_bunches == 3
assert slicer1.num_bunches == 2
assert slicer2.num_bunches == 1

xo.assert_allclose(slicer.zeta_centers,
    np.array([[ -0.9,  -0.7,  -0.5,  -0.3,  -0.1,   0.1,   0.3,   0.5,   0.7,   0.9],
              [-10.9, -10.7, -10.5, -10.3, -10.1,  -9.9,  -9.7,  -9.5,  -9.3,  -9.1],
              [-15.9, -15.7, -15.5, -15.3, -15.1, -14.9, -14.7, -14.5, -14.3, -14.1]]),
    rtol=0, atol=1e-12)

xo.assert_allclose(slicer1.zeta_centers,
    np.array([[-10.9, -10.7, -10.5, -10.3, -10.1,  -9.9,  -9.7,  -9.5,  -9.3,  -9.1],
              [-15.9, -15.7, -15.5, -15.3, -15.1, -14.9, -14.7, -14.5, -14.3, -14.1]]),
    rtol=0, atol=1e-12)

xo.assert_allclose(slicer2.zeta_centers,
    np.array([[-0.9,  -0.7,  -0.5,  -0.3,  -0.1,   0.1,   0.3,   0.5,   0.7,   0.9]]),
    rtol=0, atol=1e-12)

xo.assert_allclose(slicer.zeta_range[0], -1, rtol=0, atol=1e-12)
xo.assert_allclose(slicer1.zeta_range[0], -1, rtol=0, atol=1e-12)
xo.assert_allclose(slicer2.zeta_range[0], -1, rtol=0, atol=1e-12)
xo.assert_allclose(slicer.zeta_range[1], 1, rtol=0, atol=1e-12)
xo.assert_allclose(slicer1.zeta_range[1], 1, rtol=0, atol=1e-12)
xo.assert_allclose(slicer2.zeta_range[1], 1, rtol=0, atol=1e-12)

import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1, figsize=(6.4, 4.8*1.4))
ax1 = plt.subplot(311)
ax1.plot(*ele.moments_data.get_moment_profile('num_particles', i_turn=0), 'x',
         label='compressed profile')
ax1.plot(slicer.zeta_centers.T, slicer.num_particles.T, '.-', label='slicer')
plt.legend(loc='lower left')

ax2 = plt.subplot(312, sharex=ax1)
ax2.plot(*ele1.moments_data.get_moment_profile('num_particles', i_turn=0), 'x')
ax2.plot(slicer1.zeta_centers.T, slicer1.num_particles.T, '.-')

ax3 = plt.subplot(313, sharex=ax1)
ax3.plot(*ele2.moments_data.get_moment_profile('num_particles', i_turn=0), 'x')
ax3.plot(slicer2.zeta_centers.T, slicer2.num_particles.T, '.-')

plt.show()

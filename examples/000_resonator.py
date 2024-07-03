import numpy as np

import xwakes as xw

from scipy.constants import c as clight

import xtrack as xt

from xpart.pyheadtail_interface.pyhtxtparticles import PyHtXtParticles

p = xt.Particles(p0c=7e12, zeta=np.linspace(-1, 1, 100000))
p.x[p.zeta > 0] += 1e-3
p.y[p.zeta > 0] += 1e-3
p_ref = p.copy()
p_ref = PyHtXtParticles.from_dict(p_ref.to_dict())

wake = xw.WakeResonator(
    r=1e8, q=1e7, f_r=1e9,
    # kind=['dipolar_x', 'dipolar_y'],
    kind=xw.Yokoya('circular'),
)
wake.configure_for_tracking(zeta_range=(-1, 1), num_slices=50)

# import xfields as xf
# xfwake = xf.Wakefield(components=[
#         xf.ResonatorWake(
#             r_shunt=1e8, q_factor=1e7, frequency=1e9,
#             source_exponents=(1, 0), test_exponents=(0, 0),
#             kick='px')],
#         zeta_range=(-1, 1), num_slices=10)

from PyHEADTAIL.impedances.wakes import CircularResonator, WakeField
from PyHEADTAIL.particles.slicing import UniformBinSlicer
res_pyht = CircularResonator(R_shunt=1e8, Q=1e7, frequency=1e9)
slicer = UniformBinSlicer(n_slices=None, z_sample_points=wake._xfields_wf.slicer.zeta_centers)
wake_pyht = WakeField(slicer, res_pyht)

import xpart as xp

wake.track(p)
wake_pyht.track(p_ref)

import matplotlib.pyplot as plt
plt.close('all')
plt.plot(p.zeta, p.px, label='xwakes')
plt.plot(p_ref.zeta, p_ref.px, '--', label='pyht')

plt.legend()

plt.show()
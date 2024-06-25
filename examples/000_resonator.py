import numpy as np

import xwakes.wit as wit

res = wit.ComponentResonator(
    r=1e8, q=1e7, f_r=1e9,
    source_exponents=(1, 0),
    test_exponents=(0, 0),
    plane='x'
)

import xfields as xf

xfcomponent = xf.WakeComponent(
    source_exponents=res.source_exponents,
    test_exponents=res.test_exponents,
    kick={'x': 'px', 'y': 'py', 'z': 'delta'}[res.plane],
    function=res.wake
)

wake = xf.Wakefield(components=[xfcomponent], zeta_range=(-1, 1),
                    num_slices=100)

xfwake = xf.Wakefield(components=[
        xf.ResonatorWake(
            r_shunt=1e8, q_factor=1e7, frequency=1e9,
            source_exponents=(1, 0), test_exponents=(0, 0),
            kick='px')],
        zeta_range=(-1, 1), num_slices=100)

import xtrack as xt
p = xt.Particles(p0c=7e12, zeta=np.linspace(-1, 1, 1000))
p.x[p.zeta > 0] += 1e-3
p_ref = p.copy()

wake.track(p)
xfwake.track(p_ref)
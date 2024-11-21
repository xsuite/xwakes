# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

import numpy as np

import xwakes as xw
import xtrack as xt


w1 = xw.WakeResonator(r=1e8, q=1e7, f_r=1e8,
    kind='dipolar_x')

w2 = xw.WakeResonator(r=1e8, q=1e7, f_r=1e8,
    kind=['dipolar_x', 'dipolar_y'])

w3 = xw.WakeResonator(r=1e8, q=1e7, f_r=1e8,
    kind=xw.Yokoya('flat_horizontal'))

w4 = xw.WakeResonator(r=1e8, q=1e7, f_r=1e8,
                 plane='y', source_exponents=(1, 0),
                 test_exponents=(0, 1))

w = w1 + w2 + w3 + w4.components[0]

w.configure_for_tracking(zeta_range=(-1, 1), num_slices=100)


p = xt.Particles(p0c=7e12, zeta=np.linspace(-1, 1, 1000))

p.x += 1e-3
p.y += 1e-3

w.track(p)

import matplotlib.pyplot as plt

plt.figure(0)
plt.plot(p.zeta, p.px, 'x')
plt.xlabel('zeta')
plt.ylabel('px')

plt.figure(1)
plt.plot(p.zeta, p.py, 'x')
plt.xlabel('zeta')
plt.ylabel('py')

plt.show()
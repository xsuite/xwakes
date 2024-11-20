# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

import numpy as np
import xwakes as xw
import xobjects as xo

from scipy.constants import c as clight

beta0 = 0.1

wake = xw.WakeThickResistiveWall(
    kind='longitudinal',
    resistivity=1e-7,
    radius=1e-2,
    length=2.
    )

assert len(wake.components) == 1
assert wake.components[0].plane == 'z'
assert wake.components[0].source_exponents == (0, 0)
assert wake.components[0].test_exponents == (0, 0)

# Assert that the function is positive at close to zero from the right
assert wake.components[0].function_vs_t(1e-10, beta0=beta0, dt=1e-20) < 0 # unphysical, coming from asymptotic expansion
assert wake.components[0].function_vs_t(-1e-10, beta0=beta0, dt=1e-20) == 0

# Zeta has opposite sign compared to t
assert wake.components[0].function_vs_zeta(-1e-3, beta0=beta0, dzeta=1e-20) < 0
assert wake.components[0].function_vs_zeta(+1e-3, beta0=beta0, dzeta=1e-20) == 0

z = np.linspace(-500, 500, 100000)
t = np.linspace(-500/beta0/clight, 500/beta0/clight, 100000)

w_vs_zeta = wake.components[0].function_vs_zeta(z, beta0=beta0, dzeta=1e-4)
w_vs_t = wake.components[0].function_vs_t(t, beta0=beta0, dt=1e-4/beta0/clight)

omega = np.linspace(-1e9, 1e9, 50)

Z_from_zeta = omega * (1 + 1j)
Z_from_t = omega * (1 + 1j)
for ii, oo in enumerate(omega):
    print(ii, end='\r', flush=True)
    Z_from_zeta[ii] = 1/beta0/clight * np.trapezoid(w_vs_zeta
                * np.exp(1j * oo * z / beta0 / clight), z)
    Z_from_t[ii] = np.trapezoid(
                w_vs_t * np.exp(-1j * oo * t), t)

Z_analytical = wake.components[0].impedance(omega/2/np.pi)

Z_from_zeta -= (Z_from_zeta.mean() - Z_analytical.mean())
Z_from_t -= (Z_from_t.mean() - Z_analytical.mean())

f_test_sign = 1e9
xo.assert_allclose(wake.components[0].impedance(-f_test_sign).real,
                   wake.components[0].impedance(f_test_sign).real,
                   atol=0, rtol=1e-8)
xo.assert_allclose(wake.components[0].impedance(-f_test_sign).imag,
                   -wake.components[0].impedance(f_test_sign).imag,
                   atol=0, rtol=1e-8)

xo.assert_allclose(
    Z_from_zeta, Z_analytical, rtol=0, atol=0.1 * np.max(np.abs(Z_analytical)))
xo.assert_allclose(
    Z_from_t, Z_analytical, rtol=0, atol=0.1 * np.max(np.abs(Z_analytical)))

xo.assert_allclose(
    Z_from_zeta, Z_analytical, rtol=0, atol=0.1 * np.max(np.abs(Z_analytical)))
xo.assert_allclose(
    Z_from_t, Z_analytical, rtol=0, atol=0.1 * np.max(np.abs(Z_analytical)))

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
spre = plt.subplot(211)
plt.plot(omega, Z_from_zeta.real, label='Re Zx from zeta')
plt.plot(omega, Z_from_t.real, '--', label='Re Zx from t')
plt.plot(omega, Z_analytical.real, '-.', label='Re Zx from xwakes')
plt.legend()

spim = plt.subplot(212, sharex=spre)
plt.plot(omega, Z_from_zeta.imag, label='Im Zx from zeta')
plt.plot(omega, Z_from_t.imag, '--', label='Im Zx from t')
plt.plot(omega, Z_analytical.imag, '-.', label='Im Zx from xwakes')

plt.figure(2)
plt.plot(z, w_vs_zeta)
plt.xlabel('z [m]')
plt.ylabel('W(z)')

plt.figure(3)
plt.plot(t, w_vs_t)
plt.xlabel('t [s]')
plt.ylabel('W(t)')



plt.show()




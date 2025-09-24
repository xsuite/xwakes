# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

import numpy as np
import xwakes as xw
import xobjects as xo

from scipy.constants import c as clight

beta0 = 0.1

wake = xw.WakeResonator(r=1e8, q=10, f_r=1e9,
    kind='longitudinal')

assert len(wake.components) == 1
assert wake.components[0].plane == 'z'
assert wake.components[0].source_exponents == (0, 0)
assert wake.components[0].test_exponents == (0, 0)

# Assert that the function is positive at close to zero from the right
assert wake.components[0].function_vs_t(1e-10, beta0=beta0, dt=1e-20) > 0
assert wake.components[0].function_vs_t(-1e-10, beta0=beta0, dt=1e-20) == 0

# Zeta has opposite sign compared to t
assert wake.components[0].function_vs_zeta(-1e-3, beta0=beta0, dzeta=1e-20) > 0
assert wake.components[0].function_vs_zeta(+1e-3, beta0=beta0, dzeta=1e-20) == 0

z = np.linspace(-20, 20, 100000)
t = np.linspace(-20/clight, 20/clight, 100000)

w_vs_zeta = wake.components[0].function_vs_zeta(z, beta0=beta0, dzeta=1e-20)
w_vs_t = wake.components[0].function_vs_t(t, beta0=beta0, dt=1e-20)

omega = np.linspace(-10e9, 10e9, 500)

Z_from_zeta = omega * (1 + 1j)
Z_from_t = omega * (1 + 1j)
dz = z[1] - z[0]
dt = t[1] - t[0]
for ii, oo in enumerate(omega):
    print(ii, end='\r', flush=True)
    Z_from_zeta[ii] = 1/beta0/clight * np.sum(w_vs_zeta
                * np.exp(1j * oo * z / beta0 / clight)) * dz
    Z_from_t[ii] = np.sum(
                w_vs_t * np.exp(-1j * oo * t)) * dt

Z_analytical = wake.components[0].impedance(omega/2/np.pi)

f_test_sign = 1e9
xo.assert_allclose(wake.components[0].impedance(-f_test_sign).real,
                   wake.components[0].impedance(f_test_sign).real,
                   atol=0, rtol=1e-8)
xo.assert_allclose(wake.components[0].impedance(-f_test_sign).imag,
                   -wake.components[0].impedance(f_test_sign).imag,
                   atol=0, rtol=1e-8)

xo.assert_allclose(
    Z_from_zeta, Z_analytical, rtol=0, atol=1e-4 * np.max(np.abs(Z_analytical)))
xo.assert_allclose(
    Z_from_t, Z_analytical, rtol=0, atol=1e-4 * np.max(np.abs(Z_analytical)))

xo.assert_allclose(
    Z_from_zeta, Z_analytical, rtol=0, atol=1e-4 * np.max(np.abs(Z_analytical)))
xo.assert_allclose(
    Z_from_t, Z_analytical, rtol=0, atol=1e-4 * np.max(np.abs(Z_analytical)))

import matplotlib.pyplot as plt
plt.close('all')
fig1 = plt.figure(1, figsize=(6.4 * 1.2, 4.8 * 1.2))
spre = plt.subplot(211)
plt.plot(omega / (2 * np.pi), Z_from_zeta.real, label='Numerical FT of W(z)')
plt.plot(omega / (2 * np.pi), Z_from_t.real, '--', label='Numerical FT of W(t)')
plt.plot(omega / (2 * np.pi), Z_analytical.real, '-.', label='Analytical impedance')
plt.ylabel(r'Re$\{ Z\}$')
plt.legend()

spim = plt.subplot(212, sharex=spre)
plt.plot(omega / (2 * np.pi), Z_from_zeta.imag)
plt.plot(omega / (2 * np.pi), Z_from_t.imag)
plt.plot(omega / (2 * np.pi), Z_analytical.imag)
plt.xlabel('f [Hz]')
plt.ylabel(r'Im$\{ Z\}$')
plt.subplots_adjust(hspace=0.3, bottom=0.09, top=0.9)

plt.figure(2)
plt.plot(z, w_vs_zeta)
plt.xlabel('z [m]')
plt.ylabel('Wx(z)')

fig3 = plt.figure(3)
plt.plot(t, w_vs_t)
plt.xlabel('t [s]')
plt.ylabel('W(t)')
plt.xlim(-0.2e-8, 1e-8)



plt.show()




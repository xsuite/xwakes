import numpy as np
import xwakes as xw
import xobjects as xo

from scipy.constants import c as clight

beta0 = 0.1

comp  = xw.wit.ComponentClassicThickWall(
    kind='longitudinal',
    resistivity=1e-7,
    radius=1e-2
    )

class Dummy:
    pass

wake = Dummy()
wake.components = [comp]

assert len(wake.components) == 1
assert wake.components[0].plane == 'z'
assert wake.components[0].source_exponents == (0, 0)
assert wake.components[0].test_exponents == (0, 0)

# # Assert that the function is positive at close to zero from the right
# assert wake.components[0].function_vs_t(1e-10, beta0=beta0, dt=1e-20) > 0
# assert wake.components[0].function_vs_t(-1e-10, beta0=beta0, dt=1e-20) == 0

# # Zeta has opposite sign compared to t
# assert wake.components[0].function_vs_zeta(-1e-3, beta0=beta0, dzeta=1e-20) > 0
# assert wake.components[0].function_vs_zeta(+1e-3, beta0=beta0, dzeta=1e-20) == 0

z = np.linspace(-50, 50, 1000000)
t = np.linspace(-50/beta0/clight, 50/beta0/clight, 1000000)

w_vs_zeta = wake.components[0].function_vs_zeta(z, beta0=beta0, dzeta=1e-4)
w_vs_t = wake.components[0].function_vs_t(t, beta0=beta0, dt=1e-4/beta0/clight)

omega = np.linspace(-10e9, 10e9, 200)

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

# f_test_sign = 1e9
# xo.assert_allclose(wake.components[0].impedance(-f_test_sign).real,
#                    wake.components[0].impedance(f_test_sign).real,
#                    atol=0, rtol=1e-8)
# xo.assert_allclose(wake.components[0].impedance(-f_test_sign).imag,
#                    -wake.components[0].impedance(f_test_sign).imag,
#                    atol=0, rtol=1e-8)

# xo.assert_allclose(
#     Z_from_zeta, Z_analytical, rtol=0, atol=1e-4 * np.max(np.abs(Z_analytical)))
# xo.assert_allclose(
#     Z_from_t, Z_analytical, rtol=0, atol=1e-4 * np.max(np.abs(Z_analytical)))

# xo.assert_allclose(
#     Z_from_zeta, Z_analytical, rtol=0, atol=1e-4 * np.max(np.abs(Z_analytical)))
# xo.assert_allclose(
#     Z_from_t, Z_analytical, rtol=0, atol=1e-4 * np.max(np.abs(Z_analytical)))

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




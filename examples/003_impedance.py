import numpy as np
import xwakes as xw

from scipy.constants import c as clight


res = xw.WakeResonator(r=1e8, q=10, f_r=1e9,
    kind='dipolar_x')


z = np.linspace(-20, 20, 100000)
t = np.linspace(-20/clight, 20/clight, 100000)

wx_vs_zeta = res.components[0].function_vs_zeta(z, beta0=1)
wx_vs_t = res.components[0].function_vs_t(t, beta0=1)


omega = np.linspace(-10e9, 10e9, 1000)

Zx_from_zeta = omega * (1 + 1j)
Zx_from_t = omega * (1 + 1j)
dz = z[1] - z[0]
dt = t[1] - t[0]
for ii, oo in enumerate(omega):
    print(ii, end='\r', flush=True)
    Zx_from_zeta[ii] = 1j/clight * np.sum(wx_vs_zeta
                * np.exp(1j * oo * z / clight)) * dz
    Zx_from_t[ii] = 1j * np.sum(
                wx_vs_t * np.exp(-1j * oo * t)) * dt

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
spre = plt.subplot(211)
plt.plot(omega, Zx_from_zeta.real, label='Re Zx from zeta')
plt.plot(omega, Zx_from_t.real, '--', label='Re Zx from t')
plt.plot(omega, res.components[0].impedance(omega/2/np.pi).real, '-.', label='Re Zx from xwakes')
plt.legend()

spim = plt.subplot(212, sharex=spre)
plt.plot(omega, Zx_from_zeta.imag, label='Im Zx from zeta')
plt.plot(omega, Zx_from_t.imag, '--', label='Im Zx from t')
plt.plot(omega, res.components[0].impedance(omega/2/np.pi).imag, '-.', label='Im Zx from xwakes')

plt.show()




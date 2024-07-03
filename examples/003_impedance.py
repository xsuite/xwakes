import numpy as np
import xwakes as xw

from scipy.constants import c as clight


res = xw.WakeResonator(r=1e8, q=10, f_r=1e9,
    kind='dipolar_x')


z = np.linspace(-2, 2, 1000)
t = -z / clight

wx_vs_zeta = res.components[0].function_vs_zeta(z, beta0=1)
wx_vs_t = res.components[0].function_vs_t(t, beta0=1)


omega = np.linspace(-10e9, 10e9, 1000)

Zx_from_zeta = omega * (1 + 1j)
Zx_from_t = omega * (1 + 1j)
for ii, oo in enumerate(omega):
    print(ii)
    Zx_from_zeta[ii] = 1j * np.sum(wx_vs_zeta
                * np.exp(-1j * oo * z / clight)) * (z[1] - z[0])/clight
    Zx_from_t[ii] = 1j * np.sum(
                wx_vs_t * np.exp(1j * oo * t)) * (t[1] - t[0])





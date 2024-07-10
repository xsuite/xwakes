import numpy as np
import pandas as pd
import xwakes as xw
import xobjects as xo

from scipy.constants import c as clight
from scipy.interpolate import interp1d

beta0 = 1.

wake_data = pd.read_csv(
    './comparison_rw_for_xwakes/WlongWLHC_1layers10.00mm_precise.dat',
    skiprows=1, names=['z', 'wlong'], sep=' ')
imp_data = pd.read_csv(
    './comparison_rw_for_xwakes/ZlongWLHC_1layers10.00mm_precise.dat',
    skiprows=1, names=['f', 'ReZlong', 'ImZlong'], sep=' '
)


Z_first_quadrant = interp1d(imp_data['f'], imp_data['ReZlong'] + 1j * imp_data['ImZlong'])
def Z_function(omega):
    isscalar = np.isscalar(omega)
    if isscalar:
        omega = np.array([omega])
    mask_zero = np.abs(omega) < imp_data['f'][0]
    if mask_zero.any():
        raise ValueError('Frequency too low')
    mask_positive = omega >= 0
    mask_negative = omega < 0
    out = np.zeros_like(omega, dtype=complex)
    out[mask_positive] = Z_first_quadrant(omega[mask_positive])
    out[mask_negative] = Z_first_quadrant(-omega[mask_negative])
    out[mask_negative] = np.conj(out[mask_negative])
    return out[0] if isscalar else out

table = pd.DataFrame(
    {'time': np.concatenate([[0], wake_data['z'].values / clight]),
     'longitudinal': np.concatenate([[wake_data['wlong'].values[0]], wake_data['wlong'].values])
     }
)

wake = xw.WakeFromTable(table=table)

wake_thick = xw.WakeThickResistiveWall(
    kind='longitudinal',
    resistivity=1.7e-8, # Copper at room temperature
    radius=1e-2
    )

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

z_positive = wake_data['z'].values
z_negative = -wake_data['z'].values[::-1]
z = np.concatenate((z_negative, z_positive))
t = -z[::-1] / clight

w_vs_zeta = wake.components[0].function_vs_zeta(z, beta0=beta0, dzeta=1e-200)
w_vs_t = wake.components[0].function_vs_t(t, beta0=beta0, dt=1e-200/beta0/clight)
w_thick_vs_t = wake_thick.components[0].function_vs_t(t, beta0=beta0, dt=1e-200/beta0/clight)

omega = np.linspace(-10e9, 10e9, 500)

Z_from_zeta = omega * (1 + 1j)
Z_from_t = omega * (1 + 1j)
dz = z[1] - z[0]
dt = t[1] - t[0]
for ii, oo in enumerate(omega):
    print(ii, end='\r', flush=True)
    Z_from_zeta[ii] = 1/beta0/clight * np.trapezoid(w_vs_zeta
                * np.exp(1j * oo * (z / beta0 / clight)) , z)
    Z_from_t[ii] = np.trapezoid(
                w_vs_t * np.exp(-1j * oo * t), t)
Z_analytical = Z_function(omega/2/np.pi)

Z_thick_wall = wake_thick.components[0].impedance(omega/2/np.pi)

# Z_from_zeta -= (Z_from_zeta.mean() - Z_analytical.mean())
# Z_from_t -= (Z_from_t.mean() - Z_analytical.mean())



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
plt.plot(omega, Z_thick_wall.real, ':', label='Re Zx from thick wall')
plt.legend()

spim = plt.subplot(212, sharex=spre)
plt.plot(omega, Z_from_zeta.imag, label='Im Zx from zeta')
plt.plot(omega, Z_from_t.imag, '--', label='Im Zx from t')
plt.plot(omega, Z_analytical.imag, '-.', label='Im Zx from xwakes')
plt.plot(omega, Z_thick_wall.imag, ':', label='Im Zx from thick wall')

plt.figure(2)
plt.plot(z, w_vs_zeta)
plt.xlabel('z [m]')
plt.ylabel('W(z)')

plt.figure(3)
plt.plot(t, w_vs_t)
plt.xlabel('t [s]')
plt.ylabel('W(t)')
plt.xlim(-0.5e-12, 2e-12)


mask_positive_t = t > 0
plt.figure(4)
plt.plot(t[mask_positive_t], w_vs_t[mask_positive_t])
plt.plot(t[mask_positive_t], w_thick_vs_t[mask_positive_t])

plt.show()




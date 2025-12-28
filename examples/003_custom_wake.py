import numpy as np
import xwakes as xw

a = 1.0e9   # frequency term
b = 0.1e9   # damping rate
c = 2.0   # amplitude

def wake_vs_t(t):
    t = np.atleast_1d(t)
    out = c * np.sin(a * t) * np.exp(-b * t)
    out[t <= 0] = 0.0
    return out

comp = xw.Component(
    wake=wake_vs_t,
    plane='y',
    source_exponents=(2, 0),
    test_exponents=(1, 1),
    name="Example damped sine wake"
)

wake = xw.Wake(
    components=[comp]
)

# Plot the wake as a function of zeta
zeta_test = np.linspace(-10, 10, 500)
wake_values = comp.function_vs_zeta(zeta_test, beta0=0.7)


import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(zeta_test * 1e9, wake_values)
plt.xlabel('Longitudinal distance zeta (m))')
plt.ylabel('Wake function value')

plt.show()
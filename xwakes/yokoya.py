import numpy as np

PI = np.pi

# Yokoya factors (taken from PyHEADTAIL)
YOKOYA_FACTORS= {
    'circular': {'dipolar_x': 1.0, 'dipolar_y': 1.0,
                 'quadrupolar_x': 0.0,'quadrupolar_y': 0.0},
    'flat_horizontal': {'dipolar_x': PI**2 / 24, 'dipolar_y': PI**2 / 12,
                        'quadrupolar_x': -PI**2 / 24, 'quadrupolar_y': PI**2 / 12},
    'flat_vertical': {'dipolar_x': PI**2 / 12, 'dipolar_y': PI**2 / 24,
                      'quadrupolar_x': PI**2 / 12, 'quadrupolar_y': -PI**2 / 24}
}
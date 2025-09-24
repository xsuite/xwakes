# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

from typing import Tuple, List
import numpy as np

PI = np.pi

# Yokoya factors (taken from PyHEADTAIL)
_YOKOYA_FACTORS= {
    'circular': {'dipolar_x': 1.0, 'dipolar_y': 1.0,
                 'quadrupolar_x': 0.0,'quadrupolar_y': 0.0},
    'flat_horizontal': {'dipolar_x': PI**2 / 24, 'dipolar_y': PI**2 / 12,
                        'quadrupolar_x': -PI**2 / 24, 'quadrupolar_y': PI**2 / 24},
    'flat_vertical': {'dipolar_x': PI**2 / 12, 'dipolar_y': PI**2 / 24,
                      'quadrupolar_x': PI**2 / 24, 'quadrupolar_y': -PI**2 / 24}
}

class Yokoya:
    def __init__(self, shape: str | Tuple | List):
        self.shape = shape

        if isinstance(shape, str):
            assert shape in _YOKOYA_FACTORS
            self.dipolar_x = _YOKOYA_FACTORS[shape]['dipolar_x']
            self.dipolar_y = _YOKOYA_FACTORS[shape]['dipolar_y']
            self.quadrupolar_x = _YOKOYA_FACTORS[shape]['quadrupolar_x']
            self.quadrupolar_y = _YOKOYA_FACTORS[shape]['quadrupolar_y']
        else:
            assert len(shape) == 4
            self.dipolar_x = shape[0]
            self.dipolar_y = shape[1]
            self.quadrupolar_x = shape[2]
            self.quadrupolar_y = shape[3]

    def __getitem__(self, key):
        return getattr(self, key)

    def keys(self):
        return ['dipolar_x', 'dipolar_y', 'quadrupolar_x', 'quadrupolar_y']

    def __repr__(self):
        return f'Yokoya({self.shape})'

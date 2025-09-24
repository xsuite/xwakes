# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

import xwakes as xw


xw.WakeResonator(r=1e8, q=1e7, f_r=1e9,
    kind='dipolar_x')

xw.WakeResonator(r=1e8, q=1e7, f_r=1e9,
    kind=['dipolar_x', 'dipolar_y'])

xw.WakeResonator(r=1e8, q=1e7, f_r=1e9,
    kind={'dipolar_x': 1.0, 'dipolar_y': 2.0})

xw.WakeResonator(r=1e8, q=1e7, f_r=1e9,
    kind=xw.Yokoya('flat_horizontal'))

xw.WakeResonator(r=1e8, q=1e7, f_r=1e9,
                 plane='y', source_exponents=(3, 0),
                 test_exponents=(0, 2))
# is y_test^2 integral(x_source(z')^3 W(z-z') dz')


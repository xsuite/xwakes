import xwakes as xw


w1 = xw.WakeResonator(r=1e8, q=1e7, f_r=1e9,
    kind='dipolar_x')

w2 = xw.WakeResonator(r=1e8, q=1e7, f_r=1e9,
    kind=['dipolar_x', 'dipolar_y'])

w3 = xw.WakeResonator(r=1e8, q=1e7, f_r=1e9,
    kind=xw.Yokoya('flat_horizontal'))

w4 = xw.WakeResonator(r=1e8, q=1e7, f_r=1e9,
                 plane='y', source_exponents=(3, 0),
                 test_exponents=(0, 2))
# is y_test^2 integral(x_source(z')^3 W(z-z') dz')

w = w1 + w2 + w3 + w4


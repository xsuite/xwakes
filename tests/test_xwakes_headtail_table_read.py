import xfields as xf
import xobjects as xo
import xwakes as xw

import numpy as np

def test_headail_table_read():
    wake_table_name = xf.general._pkg_root.joinpath(
        '../test_data/HLLHC_wake.dat')
    wake_file_columns = ['time', 'longitudinal', 'dipolar_x', 'dipolar_y',
                        'quadrupolar_x', 'quadrupolar_y', 'dipolar_xy',
                        'quadrupolar_xy', 'dipolar_yx', 'quadrupolar_yx',
                        'constant_x', 'constant_y']
    components = wake_file_columns[1:]
    wake_df = xw.read_headtail_file(wake_table_name,
                                    wake_file_columns)

    wake_data = np.loadtxt(wake_table_name)

    conversion_factor_time = -1E-9
    itime = wake_file_columns.index('time')

    dict_components = {}

    dict_components['time'] = conversion_factor_time * wake_data[:, itime]

    for i_component, component in enumerate(wake_file_columns):
        if component != 'time':
            if component == 'longitudinal':
                conversion_factor = 1E12
            else:
                conversion_factor = 1E15

            dict_components[component] = (wake_data[:, i_component] *
                                        conversion_factor)

    for component in components:
        xo.assert_allclose(dict_components[component],
                            wake_df[component].values)

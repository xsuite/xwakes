# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

import pandas as pd
import numpy as np

def read_headtail_file(wake_file, wake_file_columns):
    """
    Read legacy HEADTAIL/PyHEADTAIL wake tables into a pandas DataFrame.

    Parameters
    ----------
    wake_file : str | path-like
        Path to the ASCII wake table file.
    wake_file_columns : list[str]
        Ordered list of column names in the file. Must include ``'time'`` and
        can contain any of the valid wake components (e.g. ``'longitudinal'``,
        ``'dipolar_x'``, ``'quadrupolar_y'``). Length must match the number of
        columns in the file.

    Returns
    -------
    pandas.DataFrame
        Columns are converted to SI units (time in seconds, wakes scaled to
        V/C/m^n according to component).

    Examples
    --------
    .. code-block:: python

        import xwakes as xw

        cols = ['time', 'longitudinal', 'dipolar_x', 'quadrupolar_x']
        table = xw.read_headtail_file('HLLHC_wake.dat', cols)
        wf = xw.WakeFromTable(table, columns=['dipolar_x'])
        wf.configure_for_tracking(zeta_range=(-0.4, 0.4), num_slices=100)
    """
    valid_wake_components = ['constant_x', 'constant_y', 'dipolar_x',
                             'dipolar_y', 'dipolar_xy', 'dipolar_yx',
                             'dipole_x', 'dipole_y', 'dipole_xy',
                             'dipole_yx', 'quadrupolar_x', 'quadrupolar_y',
                             'quadrupolar_xy', 'quadrupolar_yx',
                             'quadrupole_x', 'quadrupole_y',
                             'quadrupole_xy', 'quadrupole_yx',
                             'longitudinal']

    wake_data = np.loadtxt(wake_file)
    if len(wake_file_columns) != wake_data.shape[1]:
        raise ValueError("Length of wake_file_columns list does not" +
                            " correspond to the number of columns in the" +
                            " specified wake_file. \n")
    if 'time' not in wake_file_columns:
        raise ValueError("No wake_file_column with name 'time' has" +
                            " been specified. \n")

    dict_components = {}

    conversion_factor_time = 1E-9

    itime = wake_file_columns.index('time')
    dict_components['time'] = conversion_factor_time * wake_data[:, itime]

    for i_component, component in enumerate(wake_file_columns):
        if component != 'time':
            assert component in valid_wake_components
            if component == 'longitudinal':
                conversion_factor = 1E12
            else:
                conversion_factor = 1E15

            dict_components[component] = (wake_data[:, i_component] *
                                            conversion_factor)

    df = pd.DataFrame(data=dict_components)

    return df

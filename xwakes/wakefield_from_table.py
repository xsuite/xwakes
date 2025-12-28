# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

import pandas as pd
from .basewake import Wake
from .wit import ComponentFromArrays
from .wit.component import KIND_DEFINITIONS


class WakeFromTable(Wake):
    """
    Build wake components from sampled tables (e.g. HEADTAIL format).

    Parameters
    ----------
    table : pandas.DataFrame | dict
        Must contain a ``'time'`` column plus one or more wake columns matching
        ``KIND_DEFINITIONS`` keys (e.g. ``'longitudinal'``, ``'dipolar_x'``).
        Dict input is converted to a DataFrame.
    columns : iterable[str], optional
        Subset of columns to use. Defaults to all columns except ``'time'``.

    Examples
    --------
    .. code-block:: python

        import pandas as pd
        import numpy as np
        import xwakes as xw

        # Minimal example table: time plus dipolar and quadrupolar components
        table = pd.DataFrame({
            'time': np.linspace(0, 1e-9, 5),
            'dipolar_x': [0.0, 1.0, 0.5, -0.2, 0.0],
            'quadrupolar_x': [0.0, -0.5, -0.25, 0.1, 0.0],
        })

        wf = xw.WakeFromTable(table, columns=['dipolar_x', 'quadrupolar_x'])
        wf.configure_for_tracking(zeta_range=(-0.4, 0.4), num_slices=100)

    You can also read legacy HEADTAIL/PyHEADTAIL tables directly:

    .. code-block:: python

        import xwakes as xw

        cols = ['time', 'dipolar_x', 'dipolar_y', 'quadrupolar_x', 'quadrupolar_y']
        table = xw.read_headtail_file('HLLHC_wake.dat', cols)
        wf = xw.WakeFromTable(table, columns=['dipolar_x', 'dipolar_y'])
        wf.configure_for_tracking(zeta_range=(-0.4, 0.4), num_slices=100)

    Notes
    -----
    Additional information on the definition of element properties and the
    implemented physics and models can be found in the "Wakefields and
    impedances" section of the Xsuite physics guide:
    https://xsuite.readthedocs.io/en/latest/physicsguide.html
    """

    def __init__(self, table, columns=None):

        if isinstance(table, dict):
            table = pd.DataFrame(table)

        self.table = table

        assert 'time' in table.columns

        columns = columns or table.columns
        columns = list(columns)
        if 'time' in columns:
            columns.remove('time')

        components = []
        for cc in columns:
            assert cc in table.columns, f'Column {cc} not in table'
            assert cc in KIND_DEFINITIONS, f'Invalid component {cc}'

            component = ComponentFromArrays(
                interpolation_times=table['time'].values,
                wake_samples=table[cc].values,
                kind=cc)
            components.append(component)

        self.columns = columns

        super().__init__(components=components)

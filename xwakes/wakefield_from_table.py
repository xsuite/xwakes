# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

import pandas as pd
from .basewake import BaseWake
from .wit import ComponentFromArrays
from .wit.component import KIND_DEFINITIONS


class WakeFromTable(BaseWake):

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

        self.components = components
        self.columns = columns


import pandas
import numpy as np
from scipy.constants import c as clight
from scipy.interpolate import interp1d

from .basewake import BaseWake, _handle_kind
from .wit import ComponentInterpolated
from .wit.component import KIND_DEFINITIONS


class WakeFromTable(BaseWake):

    def __init__(self, table, columns=None):

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

            cc = ComponentInterpolated(
                interpolation_times=table['time'].values,
                wake_input=table[cc].values,
                kind=cc)
            components.append(cc)

        self.components = components
        self.columns = columns


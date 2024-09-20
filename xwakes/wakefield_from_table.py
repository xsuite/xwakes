import pandas as pd
from .basewake import BaseWake
from .wit import ComponentInterpolated
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

            cc = ComponentInterpolated(
                interpolation_times=table['time'].values,
                wake_input=table[cc].values,
                kind=cc)
            components.append(cc)

        self.components = components
        self.columns = columns


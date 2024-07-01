from typing import Tuple

from .basewake import BaseWake
from .wit import ComponentResonator


class WakeResonator(BaseWake):

    def __init__(self, kind: str = None,
                plane: str = None,
                source_exponents: Tuple[int, int] | None = None,
                test_exponents: Tuple[int, int] | None = None,
                r: float = None, q: float = None, f_r: float = None,
                f_roi_level: float = 0.5,
                factor=1.):

        if kind is not None:
            if isinstance(kind, str):
                kind = [kind]
            if not hasattr(factor, '__iter__'):
                factor = [factor]*len(kind)

            components = []
            for kk, ff in zip(kind, factor):
                cc = ComponentResonator(r=r, q=q, f_r=f_r,
                                        kind=kk, f_roi_level=f_roi_level,
                                        factor=ff)
                components.append(cc)
        else:
            assert not hasattr(factor, '__iter__') or len(factor) == 1
            if hasattr(factor, '__iter__'):
                factor = factor[0]
            cc = ComponentResonator(r=r, q=q, f_r=f_r, f_roi_level=f_roi_level,
                                    plane=plane, source_exponents=source_exponents,
                                    test_exponents=test_exponents,
                                    factor=factor)
            components = [cc]

        self.components = components



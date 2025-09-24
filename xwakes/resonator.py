# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

from typing import Tuple

from .basewake import BaseWake, _handle_kind
from .wit import ComponentResonator


class WakeResonator(BaseWake):

    def __init__(self, kind: str = None,
                plane: str = None,
                source_exponents: Tuple[int, int] | None = None,
                test_exponents: Tuple[int, int] | None = None,
                r: float = None, q: float = None, f_r: float = None,
                f_roi_level: float = 0.5):

        if kind is not None:

            kind = _handle_kind(kind)

            components = []
            for kk in kind.keys():
                ff = kind[kk]
                if ff == 0:
                    continue
                cc = ComponentResonator(r=r, q=q, f_r=f_r,
                                        kind=kk, f_roi_level=f_roi_level,
                                        factor=ff, plane=plane,)
                components.append(cc)
        else:
            cc = ComponentResonator(r=r, q=q, f_r=f_r, f_roi_level=f_roi_level,
                                    plane=plane, source_exponents=source_exponents,
                                    test_exponents=test_exponents)
            components = [cc]

        self.components = components
        self.kind = kind

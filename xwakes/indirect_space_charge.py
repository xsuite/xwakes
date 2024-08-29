from typing import Tuple

from .basewake import BaseWake, _handle_kind
from .wit import ComponentIndirectSpaceCharge


class WakeIndirectSpaceCharge(BaseWake):

    def __init__(self, kind: str = None,
                 plane: str = None,
                 source_exponents: Tuple[int, int] | None = None,
                 test_exponents: Tuple[int, int] | None = None,
                 gamma: float = None,
                 length: float = None,
                 radius: float = None,
                 stop_crit: float = 1e-14,
                 min_zeta: float = 1e-4,
                 max_iter: int = 10_000,
                 n_step: int = 10,
                 factor: float = 1.0):

        if kind is not None:

            kind = _handle_kind(kind)

            components = []
            for kk in kind.keys():
                ff = kind[kk]
                if ff == 0: # if the factor is zero we skip the component
                    continue
                cc = ComponentIndirectSpaceCharge(
                        kind=kk,
                        factor=ff, plane=plane,
                        gamma=gamma,
                        length=length,
                        radius=radius,
                        stop_crit=stop_crit,
                        min_zeta=min_zeta,
                        max_iter=max_iter,
                        n_step=n_step)

                components.append(cc)
        else:
            cc = ComponentIndirectSpaceCharge(
                        plane=plane, source_exponents=source_exponents,
                        test_exponents=test_exponents,
                        factor=ff,
                        gamma=gamma,
                        length=length,
                        radius=radius,
                        stop_crit=stop_crit,
                        min_zeta=min_zeta,
                        max_iter=max_iter,
                        n_step=n_step)
            components = [cc]

        self.components = components
        self.kind = kind

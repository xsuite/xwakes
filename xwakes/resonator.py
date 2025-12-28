# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

from typing import Tuple

from .basewake import Wake, _handle_kind
from .wit import ComponentResonator


class WakeResonator(Wake):
    """
    Analytic resonator wake builder.

    Parameters
    ----------
    kind : str | list[str] | tuple[str] | dict[str, float] | xwakes.Yokoya, optional
        Predefined wake kind(s). A dict scales each kind by its value. If
        None, a custom polynomial term must be provided via `plane` and
        exponent arguments.
    plane : {'x','y','z'}, optional
        Plane used only when `kind` is None to define a the plane in which the
        wake acts.
    source_exponents : tuple[int, int], optional
        Exponents (x^a y^b) on the source coordinates when `kind` is None.
    test_exponents : tuple[int, int], optional
        Exponents (x^c y^d) on the test coordinates when `kind` is None.
    r : float
        Shunt impedance (units depend on kind, e.g. Ohm/m for dipolar).
    q : float
        Quality factor.
    f_r : float
        Resonant frequency [Hz].
    f_roi_level : float, default 0.5
        Fractional cutoff used to build ROI meshes for impedance/wake sampling.

    Examples
    --------
    Single component:
        .. code-block:: python

            xw.WakeResonator(
                kind='dipolar_x',
                r=1e8,
                q=1e5,
                f_r=1e9,
            )

    Multiple components:
        .. code-block:: python

            xw.WakeResonator(
                kind=['dipolar_x', 'dipolar_y'],
                r=1e8,
                q=1e5,
                f_r=1e9,
            )

    Weighted components:
        .. code-block:: python

            xw.WakeResonator(
                kind={'dipolar_x': 2.0, 'dipolar_y': 1.0},
                r=1e8,
                q=1e5,
                f_r=1e9,
            )

    Custom polynomial term:
        .. code-block:: python

            xw.WakeResonator(
                plane='y',
                source_exponents=(1, 0),
                test_exponents=(0, 2),
                r=1e8,
                q=1e5,
                f_r=1e9,
            )

    Yokoya factors:
        .. code-block:: python

            xw.WakeResonator(
                kind=xw.Yokoya('flat_horizontal'),
                r=1e8,
                q=1e5,
                f_r=1e9,
            )

    Notes
    -----
    Additional information on the definition of element properties and the
    implemented physics and models can be found in the "Wakefields and
    impedances" section of the Xsuite physics guide:
    https://xsuite.readthedocs.io/en/latest/physicsguide.html
    """

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

        self.kind = kind

        super().__init__(components=components)

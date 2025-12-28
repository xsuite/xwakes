# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

from typing import Tuple

from .basewake import Wake, _handle_kind
from .wit import ComponentClassicThickWall


class WakeThickResistiveWall(Wake):
    """
    Classic thick-wall resistive wake builder.

    Parameters
    ----------
    kind : str | list[str] | tuple[str] | dict[str, float], optional
        Predefined wake kind(s). A dict scales each kind by its value. If
        None, a custom term must be defined via `plane` and exponent arguments.
    plane : {'x','y','z'}, optional
        Plane used only when `kind` is None to define a the plane in which the
        wake acts.
    source_exponents : tuple[int, int], optional
        Exponents (x^a y^b) on the source coordinates when `kind` is None.
    test_exponents : tuple[int, int], optional
        Exponents (x^c y^d) on the test coordinates when `kind` is None.
    radius : float
        Beam pipe radius [m].
    length : float, default 1.0
        Effective length of the wall segment [m].
    resistivity : float
        Material resistivity [Ohm*m].

    Examples
    --------
    Single component:
        .. code-block:: python

            xw.WakeThickResistiveWall(
                kind='longitudinal',
                radius=0.02,
                length=1.0,
                resistivity=1.7e-8,
            )

    Multiple components:
        .. code-block:: python

            xw.WakeThickResistiveWall(
                kind=['dipolar_x', 'dipolar_y'],
                radius=0.02,
                length=1.0,
                resistivity=1.7e-8,
            )

    Weighted components:
        .. code-block:: python

            xw.WakeThickResistiveWall(
                kind={'dipolar_x': 2.0, 'dipolar_y': 1.0},
                radius=0.02,
                length=1.0,
                resistivity=1.7e-8,
            )

    Custom polynomial term:
        .. code-block:: python

            xw.WakeThickResistiveWall(
                plane='z',
                source_exponents=(0, 0),
                test_exponents=(0, 0),
                radius=0.02,
                length=1.0,
                resistivity=1.7e-8,
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
                radius: float = None,
                length: float = 1.0,
                resistivity: float = None):

        if kind is not None:

            kind = _handle_kind(kind)

            components = []
            for kk in kind.keys():
                ff = kind[kk]
                if ff == 0: # if the factor is zero we skip the component
                    continue
                cc = ComponentClassicThickWall(
                        radius=radius, resistivity=resistivity,
                        kind=kk,
                        factor=ff, plane=plane, length=length)
                components.append(cc)
        else:
            cc = ComponentClassicThickWall(
                        radius=radius, resistivity=resistivity,
                        plane=plane, source_exponents=source_exponents,
                        test_exponents=test_exponents,
                        factor=ff, length=length)
            components = [cc]

        self.kind = kind

        super().__init__(components=components)

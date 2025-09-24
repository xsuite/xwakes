# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

@dataclass(frozen=True, eq=True)
class Sampling:
    start: float
    stop: float
    # 0 = logarithmic, 1 = linear, 2 = both
    scan_type: int
    added: Tuple[float]
    sampling_exponent: Optional[float] = None
    points_per_decade: Optional[float] = None
    min_refine: Optional[float] = None
    max_refine: Optional[float] = None
    n_refine: Optional[float] = None


@dataclass(frozen=True, eq=True)
class Layer:
    # The distance in mm of the inner surface of the layer from the reference orbit
    thickness: float
    dc_resistivity: float
    resistivity_relaxation_time: float = 0.0
    re_dielectric_constant: float = 1.0
    magnetic_susceptibility: float = 0.0
    permeability_relaxation_frequency: float = np.inf


# Define several dataclasses for IW2D input elements. We must split mandatory
# and optional arguments into private dataclasses to respect the resolution
# order. The public classes RoundIW2DInput and FlatIW2D input inherit from
# from the private classes.
# https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses

@dataclass(frozen=True, eq=True)
class _IW2DInputBase:
    machine: str
    length: float
    relativistic_gamma: float
    calculate_wake: bool
    f_params: Sampling


@dataclass(frozen=True, eq=True)
class _IW2DInputOptional:
    z_params: Optional[Sampling] = None
    long_factor: Optional[float] = None
    wake_tol: Optional[float] = None
    freq_lin_bisect: Optional[float] = None
    comment: Optional[str] = None


@dataclass(frozen=True, eq=True)
class IW2DInput(_IW2DInputOptional, _IW2DInputBase):
    pass


@dataclass(frozen=True, eq=True)
class _RoundIW2DInputBase(_IW2DInputBase):
    layers: Tuple[Layer]
    inner_layer_radius: float
    # (long, xdip, ydip, xquad, yquad)
    yokoya_factors: Tuple[float, float, float, float, float]


@dataclass(frozen=True, eq=True)
class _RoundIW2DInputOptional(_IW2DInputOptional):
    pass


@dataclass(frozen=True, eq=True)
class RoundIW2DInput(_RoundIW2DInputOptional, _RoundIW2DInputBase):
    pass


@dataclass(frozen=True, eq=True)
class _FlatIW2DInputBase(_IW2DInputBase):
    top_bottom_symmetry: bool
    top_layers: Tuple[Layer]
    top_half_gap: float


@dataclass(frozen=True, eq=True)
class _FlatIW2DInputOptional(_IW2DInputOptional):
    bottom_layers: Optional[Tuple[Layer]] = None
    bottom_half_gap: Optional[float] = None


@dataclass(frozen=True, eq=True)
class FlatIW2DInput(_FlatIW2DInputOptional, _FlatIW2DInputBase):
    pass
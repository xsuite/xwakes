# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

import numpy as np
import pathlib

from xwakes.yokoya import Yokoya

data_folder = pathlib.Path(__file__).parent.joinpath('../test_data').absolute()

def test_yokoya_circular():
    yokoya_table = np.loadtxt(data_folder / 'yokoya_factors_elliptic.dat',
                              skiprows=1)[:, 1:]
    yokoya = Yokoya(shape='circular')

    for comp in yokoya.keys():
        assert np.isclose(yokoya[comp],
                          yokoya_table[0, yokoya.keys().index(comp)],
                          atol=1e-2)

def test_yokoya_flat_vertical():
    yokoya_table = np.loadtxt(data_folder / 'yokoya_factors_elliptic.dat',
                              skiprows=1)[:, 1:]
    yokoya = Yokoya(shape='flat_vertical')

    for comp in yokoya.keys():
        assert np.isclose(yokoya[comp],
                          yokoya_table[-1, yokoya.keys().index(comp)],
                          atol=1e-2)

def test_yokoya_flat_horizontal():
    yokoya_v = Yokoya(shape='flat_vertical')
    yokoya_h = Yokoya(shape='flat_horizontal')

    assert yokoya_v['dipolar_x'] == yokoya_h['dipolar_y']
    assert yokoya_v['dipolar_y'] == yokoya_h['dipolar_x']

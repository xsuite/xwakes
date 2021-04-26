from interface import import_data_iw2d, create_component_from_data
from parameters import *

from pytest import raises
from pathlib import Path

import numpy as np


def test_duplicate_component_iw2d_import():
    with raises(AssertionError) as error_message:
        import_data_iw2d(directory=Path("test/test_data/iw2d/duplicate_components").resolve(),
                         common_string="WLHC_2layersup_0layersdown6.50mm")

    assert error_message.value.args[0] == "The wake files 'WlongWLHC_2layersup_0layersdown6.50mm.dat' and " \
                                          "'Wlong2WLHC_2layersup_0layersdown6.50mm.dat' both correspond to the " \
                                          "z-plane with exponents (0, 0, 0, 0)."


def test_no_matching_filename_iw2d_import():
    with raises(AssertionError) as error_message:
        import_data_iw2d(directory=Path("test/test_data/iw2d/valid_directory").resolve(),
                         common_string="this_string_matches_no_file")

    expected_error_message = f"No files in " \
                             f"'{Path('test/test_data/iw2d/valid_directory').resolve()}'" \
                             f" matched the common string 'this_string_matches_no_file'."

    assert error_message.value.args[0] == expected_error_message


def test_valid_iw2d_component_import():
    recipes = import_data_iw2d(directory=Path("test/test_data/iw2d/valid_directory").resolve(),
                               common_string="precise")
    for recipe in recipes:
        component = create_component_from_data(*recipe)
        data = recipe[-1]
        x = data[:, 0]
        y_actual = data[:, 1] + (1j * data[:, 2] if data.shape[1] == 3 else 0)
        if recipe[0]:
            np.testing.assert_allclose(y_actual, component.impedance(x), rtol=REL_TOL, atol=ABS_TOL)
        else:
            np.testing.assert_allclose(y_actual, component.wake(x), rtol=REL_TOL, atol=ABS_TOL)

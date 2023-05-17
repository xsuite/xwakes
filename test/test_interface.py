from pywit.interface import import_data_iw2d, create_component_from_data,get_component_name
from pywit.parameters import *

from pathlib import Path

from pytest import raises,mark
import numpy as np


@mark.parametrize('is_impedance, plane, exponents, expected_comp_name',
                  [ [True,  'z', (0, 0, 0, 0), 'zlong'],
                    [False, 'y', (0, 1, 0, 0), 'wydip'],
                    [True,  'x', (0, 0, 1, 0), 'zxqua'],
                    [False, 'y', (0, 0, 0, 0), 'wycst'],
                  ])
def test_get_component_name(is_impedance, plane, exponents, expected_comp_name):
    assert get_component_name(is_impedance, plane, exponents) == expected_comp_name


@mark.parametrize('is_impedance, plane, exponents',
                  [ [None,  'l', (0, 0, 0, 0)],
                    [True,  'l', (0, 1, 0, 0)],
                    [False, 'x', ('a', 0, 1, 0)],
                  ])
def test_get_component_name_raise(is_impedance, plane, exponents):
    with raises(ValueError) as error_message:
        get_component_name(is_impedance, plane, exponents)

    assert error_message.value.args[0] == f"({is_impedance},{plane},{exponents}) cannot be found in" \
                                           " the values of component_names dictionary"


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
    # Normally, the relativistic gamma would be an attribute of a required IW2DInput object, but here it has been
    # hard-coded instead
    relativstic_gamma = 479.605064966
    recipes = import_data_iw2d(directory=Path("test/test_data/iw2d/valid_directory").resolve(),
                               common_string="precise")
    for recipe in recipes:
        component = create_component_from_data(*recipe, relativistic_gamma=relativstic_gamma)
        data = recipe[-1]
        x = data[:, 0]
        y_actual = data[:, 1] + (1j * data[:, 2] if data.shape[1] == 3 else 0)
        if recipe[0]:
            np.testing.assert_allclose(y_actual, component.impedance(x), rtol=REL_TOL, atol=ABS_TOL)
        else:
            np.testing.assert_allclose(y_actual, component.wake(x), rtol=REL_TOL, atol=ABS_TOL)

import os

import pytest

from pywit.interface import import_data_iw2d, create_component_from_data, Sampling
from pywit.interface import check_valid_working_directory, get_component_name
from pywit.interface import check_already_computed, get_iw2d_config_value, RoundIW2DInput, add_iw2d_input_to_database
from pywit.parameters import *
from pywit.materials import layer_from_json_material_library

from pathlib import Path
import glob
from hashlib import sha256
from pytest import raises, mark

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

    assert error_message.value.args[0] in ["The wake files 'WlongWLHC_2layersup_0layersdown6.50mm.dat' and " \
                                           "'Wlong2WLHC_2layersup_0layersdown6.50mm.dat' both correspond to the " \
                                           "z-plane with exponents (0, 0, 0, 0).",
                                           "The wake files 'Wlong2WLHC_2layersup_0layersdown6.50mm.dat' and " \
                                           "'WlongWLHC_2layersup_0layersdown6.50mm.dat' both correspond to the " \
                                           "z-plane with exponents (0, 0, 0, 0).",
                                           ]


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


@pytest.fixture
def round_tung_layer_iw2d_input():
    f_params = Sampling(start=1, stop=1e9, scan_type=0, added=(1e2,))
    layers_tung = (layer_from_json_material_library(thickness=np.inf, material_key='W'),)
    return RoundIW2DInput(machine='test', length=1, relativistic_gamma=7000,
                          calculate_wake=False, f_params=f_params, comment='test',
                          layers=layers_tung, inner_layer_radius=5e-2, yokoya_factors=(1, 1, 1, 1, 1))


def test_check_already_computed(round_tung_layer_iw2d_input):
    name = 'test_hash'

    # create the expected directories for the dummy input
    projects_path = Path(get_iw2d_config_value('project_directory'))
    input_hash = sha256(round_tung_layer_iw2d_input.__str__().encode()).hexdigest()
    directory_level_1 = projects_path.joinpath(input_hash[0:2])
    directory_level_2 = directory_level_1.joinpath(input_hash[2:4])
    working_directory = directory_level_2.joinpath(input_hash[4:])

    if not os.path.exists(directory_level_1):
        os.mkdir(directory_level_1)

    if not os.path.exists(directory_level_2):
        os.mkdir(directory_level_2)

    # check if the directory already existed, otherwise remove it with the content
    if os.path.exists(working_directory):
        [os.remove(old_file) for old_file in glob.glob(f'{working_directory}/*')]
    else:
        os.mkdir(working_directory)

    dummy_string = 'dummy_string'
    for comp in ['Zlong', 'Zxdip','Zydip','Zxquad','Zyquad']:
        with open(f'{working_directory}/{comp}_test.txt', 'w') as f:
            f.write(dummy_string)

    # check that the input is detected in the hashmap
    already_computed, input_hash, working_directory = check_already_computed(round_tung_layer_iw2d_input, name)
    assert already_computed

    # now we remove the folder and check that check_already_computed gives false
    os.system(f'rm -r {working_directory}')
    already_computed, input_hash, working_directory = check_already_computed(round_tung_layer_iw2d_input, name)

    assert not already_computed

    # check_already_computed creates working_directory again so we clean it up
    os.system(f'rm -r {working_directory}')


def test_add_iw2d_input_to_database(round_tung_layer_iw2d_input):
    projects_path = Path(get_iw2d_config_value('project_directory'))
    input_hash = sha256(round_tung_layer_iw2d_input.__str__().encode()).hexdigest()
    directory_level_1 = projects_path.joinpath(input_hash[0:2])
    directory_level_2 = directory_level_1.joinpath(input_hash[2:4])
    working_directory = directory_level_2.joinpath(input_hash[4:])

    if not check_valid_working_directory(working_directory):
        raise ValueError("working directory is not in the right format. The right format is "
                         "`<project_directory>/hash[0:2]/hash[2:4]/hash[4:]`")

    add_iw2d_input_to_database(round_tung_layer_iw2d_input, input_hash, working_directory)

    assert os.path.exists(f"{working_directory}/input.txt")


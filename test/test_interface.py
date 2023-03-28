from pywit.interface import import_data_iw2d, create_component_from_data, IW2DInput, Sampling, check_already_computed, \
    get_iw2d_config_value, add_elements_to_hashmap
from pywit.parameters import *

from pathlib import Path
from typing import Dict
import pickle
from hashlib import sha256
from pytest import raises

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


def test_check_already_computed():
    # create dummy iw2d input
    f_params = Sampling(start=1, stop=1e9, scan_type=0, added=(1e2,))
    iw2d_input = IW2DInput(machine='test', length=1, relativistic_gamma=100, calculate_wake=True, f_params=f_params)
    name = 'test_hash'

    projects_path = Path(get_iw2d_config_value('project_directory'))
    # add the dummy input the the hashmap
    with open(projects_path.joinpath('hashmap.pickle'), 'rb') as pickle_file:
        hashmap: Dict[str, str] = pickle.load(pickle_file)

    input_hash = sha256(iw2d_input.__str__().encode()).hexdigest()

    working_directory = Path.joinpath(projects_path, input_hash)

    if Path.exists(working_directory):
        Path.rmdir(working_directory)

    Path.mkdir(Path.joinpath(projects_path, input_hash))

    hashmap[input_hash] = name

    with open(projects_path.joinpath('hashmap.pickle'), 'wb') as handle:
        pickle.dump(hashmap, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # check that the input is detected in the hashmap
    read_ready, input_hash = check_already_computed(iw2d_input, name)

    # remove the dummy input from the hashmap
    hashmap.pop(input_hash)
    with open(projects_path.joinpath('hashmap.pickle'), 'wb') as handle:
        pickle.dump(hashmap, handle, protocol=pickle.HIGHEST_PROTOCOL)
    Path.rmdir(Path.joinpath(projects_path, input_hash))

    assert read_ready


def test_add_elements_to_hashmap():
    # create dummy iw2d input
    f_params = Sampling(start=1, stop=1e9, scan_type=0, added=(1e2,))
    iw2d_input = IW2DInput(machine='test', length=1, relativistic_gamma=100, calculate_wake=True, f_params=f_params)
    name = 'test_hash'
    input_hash = sha256(iw2d_input.__str__().encode()).hexdigest()

    # add the input to the hashmap
    add_elements_to_hashmap(name, input_hash)

    # check that the input is in the hashmap
    projects_path = Path(get_iw2d_config_value('project_directory'))

    with open(projects_path.joinpath('hashmap.pickle'), 'rb') as pickle_file:
        hashmap: Dict[str, str] = pickle.load(pickle_file)

    hashmap_keys = list(hashmap.keys())

    # remove the dummy input from the hashmap
    hashmap.pop(input_hash)
    with open(projects_path.joinpath('hashmap.pickle'), 'wb') as handle:
        pickle.dump(hashmap, handle, protocol=pickle.HIGHEST_PROTOCOL)

    assert input_hash in hashmap_keys

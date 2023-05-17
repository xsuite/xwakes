from pywit.component import Component
from pywit.element import Element

import pickle
import subprocess
from typing import Tuple, List, Optional, Dict, Any, Union
from os import listdir, makedirs
from dataclasses import dataclass
from pathlib import Path
from hashlib import sha256

import numpy as np
from yaml import load, BaseLoader, dump
from joblib import Parallel, delayed
from scipy.interpolate import interp1d

# A dictionary mapping the datafile-prefixes (as used in IW2D) to (is_impedance, plane, (a, b, c, d))
# Where is impedance is True if the component in question is an impedance component, and False if it is a
# wake component, and a, b, c and d are the source and test exponents of the component
component_names = {'wlong': (False, 'z', (0, 0, 0, 0)),
                   'wxdip': (False, 'x', (1, 0, 0, 0)),
                   'wydip': (False, 'y', (0, 1, 0, 0)),
                   'wxqua': (False, 'x', (0, 0, 1, 0)),
                   'wyqua': (False, 'y', (0, 0, 0, 1)),
                   'wxcst': (False, 'x', (0, 0, 0, 0)),
                   'wycst': (False, 'y', (0, 0, 0, 0)),
                   'zlong': (True, 'z', (0, 0, 0, 0)),
                   'zxdip': (True, 'x', (1, 0, 0, 0)),
                   'zydip': (True, 'y', (0, 1, 0, 0)),
                   'zxqua': (True, 'x', (0, 0, 1, 0)),
                   'zyqua': (True, 'y', (0, 0, 0, 1)),
                   'zxcst': (True, 'x', (0, 0, 0, 0)),
                   'zycst': (True, 'y', (0, 0, 0, 0))}

# The parent directory of this file
IW2D_SETTINGS_PATH = Path.home().joinpath('pywit').joinpath('config').joinpath('iw2d_settings.yaml')

def get_component_name(is_impedance,plane,exponents):
    """
    Get the component name from is_impedance, plane and exponents (doing the
    reverse operation of the dictionary in component_names)
    :param is_impedance: True for impedance component, False for wake
    :param plane: plane ('x', 'y' or 'z')
    :param exponents: four integers corresponding to (source_x, source_y, test_x, test_y) aka (a, b, c, d)
    :return: str with component name (e.g. 'zydip' or 'wxqua')
    """
    comp_list = [comp_name for comp_name,v in component_names.items()
                 if v==(is_impedance,plane,exponents)]
    if len(comp_list)!=1:
        raise ValueError(f"({is_impedance},{plane},{exponents}) cannot be found in"
                          " the values of component_names dictionary")

    return comp_list[0]


def get_iw2d_config_value(key: str) -> Any:
    with open(IW2D_SETTINGS_PATH, 'r') as file:
        config = load(file, Loader=BaseLoader)

    return config[key]


def import_data_iw2d(directory: Union[str, Path],
                     common_string: str) -> List[Tuple[bool, str, Tuple[int, int, int, int], np.ndarray]]:
    """
    Imports data on the format generated by the IW2D library and prepares it for construction of Components and
    Elements in PyWIT
    :param directory: The directory where the .dat files are located. All .dat files must be in the root of this
    directory
    :param common_string: A string preceding ".dat" in the filenames of all files to be imported
    :return: A list of tuples, one for each imported file, on the form (is_impedance, plane, (a, b, c, d), data),
    where data is a numpy array with 2 or 3 columns, one for each column of the imported datafile.
    """
    # The tuples are iteratively appended to this array
    component_recipes = []

    # Keeps track of what combinations of (is_impedance, plane, exponents) have been imported to avoid duplicates
    seen_configs = []

    # A list of all of the filenames in the user-specified directory
    filenames = listdir(directory)
    for i, filename in enumerate(filenames):
        # If the string preceding ".dat" in the filename does not match common_string, or if the first 5 letters
        # of the filename are not recognized as a type of impedance/wake, the file is skipped
        if filename[-4 - len(common_string):-4] != common_string or filename[:5].lower() not in component_names:
            continue

        # The values of is_impedance, plane and exponents are deduced from the first 5 letters of the filename using
        # the component_names-dictionary
        is_impedance, plane, exponents = component_names[filename[:5].lower()]

        # Validates that the combination of (is_impedance, plane, exponents) is unique
        assert (is_impedance, plane, exponents) not in seen_configs, \
            f"The {'impedance' if is_impedance else 'wake'} files " \
            f"'{filename}' and '{filenames[seen_configs.index((is_impedance, plane, exponents))]}' " \
            f"both correspond to the {plane}-plane with exponents {exponents}."
        seen_configs.append((is_impedance, plane, exponents))

        # Loads the data from the file as a numpy array
        data = np.loadtxt(f"{directory}/{filename}", delimiter=" ", skiprows=1)

        # Appends the constructed tuple to component_recipes
        component_recipes.append((is_impedance, plane, exponents, data))

    # Validates that at least one file in the directory matched the user-specified common_string
    assert component_recipes, f"No files in '{directory}' matched the common string '{common_string}'."
    return component_recipes


def create_component_from_data(is_impedance: bool, plane: str, exponents: Tuple[int, int, int, int],
                               data: np.ndarray, relativistic_gamma: float) -> Component:
    """
    Creates a Component from a component recipe, e.g. as generated by import_data_iw2d
    :param is_impedance: a bool which is True if the component to be generated is an impedance component, and False
    if it is a wake component
    :param plane: the plane of the component
    :param exponents: the exponents of the component on the form (a, b, c, d)
    :param data: a numpy-array with 2 or 3 columns corresponding to (frequency, Re[impedance], Im[impedance]) or
    (position, Re[wake], Im[wake]), where the imaginary column is optional
    :param relativistic_gamma: The relativistic gamma used in the computation of the data files. Necessary for
    converting the position-data of IW2D into time-data for PyWIT
    :return: A Component object as specified by the input
    """
    # Extracts the position/frequency column of the data array
    x = data[:, 0]

    if not is_impedance:
        # Converts position-data to time-data using Lorentz factor
        x /= 299792458 * np.sqrt(1 - (1 / relativistic_gamma ** 2))

    # Extracts the wake/values from the data array
    y = data[:, 1] + (1j * data[:, 2] if data.shape[1] == 3 else 0)

    # Creates a callable impedance/wake function from the data array
    func = interp1d(x=x, y=y, kind='linear', assume_sorted=True, bounds_error=False, fill_value=(0., 0.))

    # Initializes and returns a component based on the parameters provided
    return Component(impedance=(func if is_impedance else None),
                     wake=(None if is_impedance else func),
                     plane=plane,
                     source_exponents=exponents[:2],
                     test_exponents=exponents[2:],)


@dataclass(frozen=True, eq=True)
class Layer:
    # The distance in mm of the inner surface of the layer from the reference orbit
    thickness: float
    dc_resistivity: float
    resistivity_relaxation_time: float
    re_dielectric_constant: float
    magnetic_susceptibility: float
    permeability_relaxation_frequency: float


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


def _iw2d_format_layer(layer: Layer, n: int) -> str:
    """
    Formats the information describing a single layer into a string in accordance with IW2D standards.
    Intended only as a helper-function for create_iw2d_input_file.
    :param layer: A Layer object
    :param n: The 1-indexed index of the layer
    :param thickness: The thickness of the given layer
    :return: A string on the correct format for IW2D
    """
    return (f"Layer {n} DC resistivity (Ohm.m):\t{layer.dc_resistivity}\n"
            f"Layer {n} relaxation time for resistivity (ps):\t{layer.resistivity_relaxation_time * 1e12}\n"
            f"Layer {n} real part of dielectric constant:\t{layer.re_dielectric_constant}\n"
            f"Layer {n} magnetic susceptibility:\t{layer.magnetic_susceptibility}\n"
            f"Layer {n} relaxation frequency of permeability (MHz):\t{layer.permeability_relaxation_frequency / 1e6}\n"
            f"Layer {n} thickness in mm:\t{layer.thickness * 1e3}\n")


def _iw2d_format_freq_params(params: Sampling) -> str:
    """
    Formats the frequency-parameters of an IW2DInput object to a string in accordance with IW2D standards.
    Intended only as a helper-function for create_iw2d_input_file.
    :param params: Parameters specifying a frequency-sampling
    :return: A string on the correct format for IW2D
    """
    lines = [f"start frequency exponent (10^) in Hz:\t{np.log10(params.start)}",
             f"stop frequency exponent (10^) in Hz:\t{np.log10(params.stop)}",
             f"linear (1) or logarithmic (0) or both (2) frequency scan:\t{params.scan_type}"]

    if params.sampling_exponent is not None:
        lines.append(f"sampling frequency exponent (10^) in Hz (for linear):\t{np.log10(params.sampling_exponent)}")

    if params.points_per_decade is not None:
        lines.append(f"Number of points per decade (for log):\t{params.points_per_decade}")

    if params.min_refine is not None:
        lines.append(f"when both, fmin of the refinement (in THz):\t{params.min_refine / 1e12}")

    if params.max_refine is not None:
        lines.append(f"when both, fmax of the refinement (in THz):\t{params.max_refine / 1e12}")

    if params.n_refine is not None:
        lines.append(f"when both, number of points in the refinement:\t{params.n_refine}")

    lines.append(f"added frequencies [Hz]:\t{' '.join(str(f) for f in params.added)}")

    return "\n".join(lines) + "\n"


def _iw2d_format_z_params(params: Sampling) -> str:
    """
    Formats the position-parameters of an IW2DInput object to a string in accordance with IW2D standards.
    Intended only as a helper-function for create_iw2d_input_file.
    :param params: Parameters specifying a position-sampling
    :return: A string on the correct format for IW2D
    """
    lines = [f"linear (1) or logarithmic (0) or both (2) scan in z for the wake:\t{params.scan_type}"]

    if params.sampling_exponent is not None:
        lines.append(f"sampling distance in m for the linear sampling:\t{params.sampling_exponent}")

    if params.min_refine is not None:
        lines.append(f"zmin in m of the linear sampling:\t{params.min_refine}")

    if params.max_refine is not None:
        lines.append(f"zmax in m of the linear sampling:\t{params.max_refine}")

    if params.points_per_decade is not None:
        lines.append(f"Number of points per decade for the logarithmic sampling:\t{params.points_per_decade}")

    lines.append(f"exponent (10^) of zmin (in m) of the logarithmic sampling:\t{np.log10(params.start)}")
    lines.append(f"exponent (10^) of zmax (in m) of the logarithmic sampling:\t{np.log10(params.stop)}")
    lines.append(f"added z [m]:\t{' '.join(str(z) for z in params.added)}")

    return "\n".join(lines) + "\n"


def create_iw2d_input_file(iw2d_input: IW2DInput, filename: Union[str, Path]) -> None:
    """
    Writes an IW2DInput object to the specified filename using the appropriate format for interfacing with the IW2D
    software.
    :param iw2d_input: An IW2DInput object to be written
    :param filename: The filename (including path) of the file the IW2DInput object will be written to
    :return: Nothing
    """
    # Creates the input-file at the location specified by filename
    file = open(filename, 'w')

    file.write(f"Machine:\t{iw2d_input.machine}\n"
               f"Relativistic Gamma:\t{iw2d_input.relativistic_gamma}\n"
               f"Impedance Length in m:\t{iw2d_input.length}\n")

    # Just pre-defining layers to avoid potentially unbound variable later on
    layers = []
    if isinstance(iw2d_input, RoundIW2DInput):
        file.write(f"Number of layers:\t{len(iw2d_input.layers)}\n"
                   f"Layer 1 inner radius in mm:\t{iw2d_input.inner_layer_radius * 1e3}\n")
        layers = iw2d_input.layers
    elif isinstance(iw2d_input, FlatIW2DInput):
        if iw2d_input.bottom_layers:
            print("WARNING: bottom layers of IW2D input object are being ignored because the top_bottom_symmetry flag "
                  "is enabled")
        file.write(f"Number of upper layers in the chamber wall:\t{len(iw2d_input.top_layers)}\n")
        if iw2d_input.top_layers:
            file.write(f"Layer 1 inner half gap in mm:\t{iw2d_input.top_half_gap * 1e3}\n")
        layers = iw2d_input.top_layers

    for i, layer in enumerate(layers):
        file.write(_iw2d_format_layer(layer, i + 1))

    if isinstance(iw2d_input, FlatIW2DInput) and not iw2d_input.top_bottom_symmetry:
        file.write(f"Number of lower layers in the chamber wall:\t{len(iw2d_input.bottom_layers)}\n")
        if iw2d_input.bottom_layers:
            file.write(f"Layer -1 inner half gap in mm:\t{iw2d_input.bottom_half_gap * 1e3}\n")
            for i, layer in enumerate(iw2d_input.bottom_layers):
                file.write(_iw2d_format_layer(layer, -(i + 1)))

    if isinstance(iw2d_input, FlatIW2DInput):
        file.write(f"Top bottom symmetry (yes or no):\t{'yes' if iw2d_input.top_bottom_symmetry else 'no'}\n")

    file.write(_iw2d_format_freq_params(iw2d_input.f_params))
    if iw2d_input.z_params is not None:
        file.write(_iw2d_format_z_params(iw2d_input.z_params))

    if isinstance(iw2d_input, RoundIW2DInput):
        file.write(f"Yokoya factors long, xdip, ydip, xquad, yquad:\t"
                   f"{' '.join(str(n) for n in iw2d_input.yokoya_factors)}\n")

    for desc, val in zip(["factor weighting the longitudinal impedance error",
                          "tolerance (in wake units) to achieve",
                          "frequency above which the mesh bisecting is linear [Hz]",
                          "Comments for the output files names"],
                         [iw2d_input.long_factor, iw2d_input.wake_tol, iw2d_input.freq_lin_bisect, iw2d_input.comment]):
        if val is not None:
            file.write(f"{desc}:\t{val}\n")

    file.close()


def create_element_using_iw2d(iw2d_input: IW2DInput, name: str, beta_x: float, beta_y: float, tag: str = 'IW2D') -> Element:
    assert " " not in name, "Spaces are not allowed in element name"

    assert verify_iw2d_config_file(), "The binary and/or project directories specified in config/iw2d_settings.yaml " \
                                      "do not exist or do not contain the required files and directories."

    bin_path = Path(get_iw2d_config_value('binary_directory'))
    projects_path = Path(get_iw2d_config_value('project_directory'))

    input_hash = sha256(iw2d_input.__str__().encode()).hexdigest()
    delete_removed_projects()

    with open(projects_path.joinpath('hashmap.pickle'), 'rb') as pickle_file:
        hashmap: Dict[str, str] = pickle.load(pickle_file)

    read_ready = False

    if input_hash in hashmap:
        if hashmap[input_hash] == name:
            print(f"The computation of '{name}' has already been performed with the exact given parameters. "
                  f"These results will be used to generate the element.")
            read_ready = True
        else:
            print(f"Another element, '{hashmap[input_hash]}', has previously been computed with the exact same "
                  f"parameters as '{name}'. Do you wish to re-perform the computation, or construct an element from "
                  f"the already computed values?")
            choice = input("1: Re-do computation\n"
                           "2: Use old values (recommended)\n"
                           "Your choice: ")
            if choice == '2':
                name = hashmap[input_hash]
                read_ready = True

    if not read_ready:
        bin_string = ("wake_" if iw2d_input.calculate_wake else "") + \
                     ("round" if isinstance(iw2d_input, RoundIW2DInput) else "flat") + "chamber.x"
        subprocess.run(['mkdir', name], cwd=projects_path)
        working_directory = projects_path.joinpath(name)
        create_iw2d_input_file(iw2d_input, working_directory.joinpath(f"{name}_input.txt"))
        subprocess.run(f'{bin_path.joinpath(bin_string)} < {name}_input.txt',
                       shell=True, cwd=working_directory)
        hashmap[input_hash] = name
        with open(projects_path.joinpath('hashmap.pickle'), 'wb') as handle:
            pickle.dump(hashmap, handle, protocol=pickle.HIGHEST_PROTOCOL)

    component_recipes = import_data_iw2d(projects_path.joinpath(name), iw2d_input.comment)

    return Element(length=iw2d_input.length,
                   beta_x=beta_x, beta_y=beta_y,
                   components=[create_component_from_data(*recipe, relativistic_gamma=iw2d_input.relativistic_gamma)
                               for recipe in component_recipes],
                   name=name, tag=tag, description='A resistive wall element created using IW2D')


def verify_iw2d_config_file() -> bool:
    bin_path = Path(get_iw2d_config_value('binary_directory'))
    projects_path = Path(get_iw2d_config_value('project_directory'))
    if not bin_path.exists() or not projects_path.exists():
        return False

    contents = listdir(bin_path)
    for filename in ('flatchamber.x', 'roundchamber.x', 'wake_flatchamber.x', 'wake_roundchamber.x'):
        if filename not in contents:
            return False

    if 'hashmap.pickle' not in listdir(projects_path):
        return False

    return True


def delete_removed_projects() -> None:
    """
    Updates the dictionary in hashmap.pickle by deleting any references to project folders which have been deleted by
    the user.
    :return: Nothing
    """
    projects_path = Path(get_iw2d_config_value('project_directory'))
    with open(projects_path.joinpath('hashmap.pickle'), 'rb') as pickle_file:
        hashmap: Dict[str, str] = pickle.load(pickle_file)

    projects = listdir(projects_path)
    new_dict = {k: v for k, v in hashmap.items() if v in projects}

    with open(projects_path.joinpath('hashmap.pickle'), 'wb') as handle:
        pickle.dump(new_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _typecast_sampling_dict(d: Dict[str, str]) -> Dict[str, Any]:
    added = [float(f) for f in d['added'].split()] if 'added' in d else []
    added = tuple(added)
    scan_type = int(d['scan_type'])
    d.pop('added'), d.pop('scan_type')

    new_dict = {k: float(v) for k, v in d.items()}
    new_dict['added'] = added
    new_dict['scan_type'] = scan_type
    return new_dict


def _create_iw2d_input_from_dict(d: Dict[str, Any]) -> IW2DInput:
    is_round = d['is_round'].lower() in ['true', 'yes', 'y', '1']
    d.pop('is_round')
    layers, inner_layer_radius, yokoya_factors = list(), float(), tuple()
    top_layers, top_half_gap, bottom_layers, bottom_half_gap = list(), float(), None, None

    if is_round:
        inner_layer_radius = d['inner_layer_radius']
        if 'layers' in d:
            layers_dicts = [{k: float(v) for k, v in layer.items()} for layer in d['layers']]
            layers = [Layer(**kwargs) for kwargs in layers_dicts]
            d.pop('layers')
    else:
        if 'top_layers' in d:
            top_layers_dicts = [{k: float(v) for k, v in layer.items()} for layer in d['top_layers']]
            top_layers = [Layer(**kwargs) for kwargs in top_layers_dicts]
            top_half_gap = d['top_half_gap']
            d.pop('top_layers')
            if d['top_bottom_symmetry'].lower() in ['true', 'yes', 'y', '1']:
                bottom_layers = None
            else:
                bottom_layers_dicts = [{k: float(v) for k, v in layer.items()} for layer in d['bottom_layers']]
                bottom_layers = [Layer(**kwargs) for kwargs in bottom_layers_dicts]
                bottom_half_gap = d['bottom_half_gap']
                d.pop('bottom_layers')

    if 'yokoya_factors' in d:
        yokoya_factors = tuple(float(x) for x in d['yokoya_factors'].split())
        d.pop('yokoya_factors')

    f_params = Sampling(**_typecast_sampling_dict(d['f_params']))
    z_params = Sampling(**_typecast_sampling_dict(d['z_params'])) \
        if d['calculate_wake'].lower() in ['true', 'yes', 'y', '1'] else None

    d.pop('f_params')
    d.pop('z_params', None)

    transformations = {
        'machine': str,
        'length': float,
        'relativistic_gamma': float,
        'calculate_wake': lambda x: x.lower() in ['true', 'yes', 'y', '1'],
        'long_factor': float,
        'wake_tol': float,
        'freq_lin_bisect': float,
        'comment': str
    }

    new_dict = {k: transformations[k](d[k]) if k in d else None for k in transformations}

    if is_round:
        return RoundIW2DInput(
            f_params=f_params,
            z_params=z_params,
            layers=tuple(layers),
            inner_layer_radius=inner_layer_radius,
            yokoya_factors=yokoya_factors,
            **new_dict
        )
    else:
        return FlatIW2DInput(
            f_params=f_params,
            z_params=z_params,
            top_bottom_symmetry=d['top_bottom_symmetry'].lower() in ['true', 'yes', 'y', '1'],
            top_layers=tuple(top_layers),
            top_half_gap = top_half_gap,
            bottom_layers=bottom_layers,
            bottom_half_gap=bottom_half_gap,
            **new_dict
        )


def create_iw2d_input_from_yaml(name: str) -> IW2DInput:
    path = Path.home().joinpath('pywit').joinpath('config').joinpath('iw2d_inputs.yaml')
    with open(path) as file:
        inputs = load(file, Loader=BaseLoader)
        d = inputs[name]

    return _create_iw2d_input_from_dict(d)


def create_multiple_elements_using_iw2d(iw2d_inputs: List[IW2DInput], names: List[str],
                                        beta_xs: List[float], beta_ys: List[float]) -> List[Element]:
    assert len(iw2d_inputs) == len(names) == len(beta_xs) == len(beta_ys), "All input lists need to have the same" \
                                                                           "number of elements"

    for name in names:
        assert " " not in name, "Spaces are not allowed in element name"

    assert verify_iw2d_config_file(), "The binary and/or project directories specified in config/iw2d_settings.yaml " \
                                      "do not exist or do not contain the required files and directories."

    bin_path = Path(get_iw2d_config_value('binary_directory'))
    projects_path = Path(get_iw2d_config_value('project_directory'))
    delete_removed_projects()
    read_ready = [False for _ in iw2d_inputs]
    input_hashes = [sha256(iw2d_input.__str__().encode()).hexdigest() for iw2d_input in iw2d_inputs]

    with open(projects_path.joinpath('hashmap.pickle'), 'rb') as pickle_file:
        hashmap: Dict[str, str] = pickle.load(pickle_file)
        for i, ih in enumerate(input_hashes):

            if ih in hashmap:
                if hashmap[ih] == names[i]:
                    print(f"The computation of '{names[i]}' has already been performed with the exact given parameters."
                          f" These results will be used to generate the element.")
                    read_ready[i] = True
                else:
                    print(f"Another element, '{hashmap[ih]}', has previously been computed with the exact same "
                          f"parameters as '{names[i]}'. These computed values will be re-used to construct the new "
                          f"element.")
                    names[i] = hashmap[ih]
                    read_ready[i] = True

    elements = Parallel(n_jobs=-1, prefer='threads')(delayed(_generate_iw2d_element_async)(
        iw2d_input=iw2d_inputs[i],
        name=names[i],
        beta_x=beta_xs[i],
        beta_y=beta_ys[i],
        read_ready=read_ready[i],
        projects_path=projects_path,
        bin_path=bin_path
    ) for i in range(len(names)))

    with open(projects_path.joinpath('hashmap.pickle'), 'wb') as pickle_file:
        for ih, name in zip(input_hashes, names):
            hashmap[ih] = name

        pickle.dump(hashmap, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    return elements


def _generate_iw2d_element_async(iw2d_input: IW2DInput, name: str, beta_x: float, beta_y: float, read_ready: bool,
                                 projects_path: Union[str, Path], bin_path: Union[str, Path]) -> Element:
    if not read_ready:
        print(f'Running IW2D computation for {name}')
        bin_string = ("wake_" if iw2d_input.calculate_wake else "") + \
                     ("round" if isinstance(iw2d_input, RoundIW2DInput) else "flat") + "chamber.x"

        subprocess.run(['mkdir', name], cwd=projects_path)
        working_directory = Path(projects_path).joinpath(name)
        create_iw2d_input_file(iw2d_input, working_directory.joinpath(f'{name}_input.txt'))
        proc = subprocess.run(f'{Path(bin_path).joinpath(bin_string)} < {name}_input.txt',
                              shell=True, cwd=working_directory, stdout=subprocess.PIPE)
        with open(working_directory.joinpath(f'IW2D_terminal_output{iw2d_input.comment}.txt'), 'w') as file:
            file.write(proc.stdout.decode())

    component_recipes = import_data_iw2d(Path(projects_path).joinpath(name), iw2d_input.comment)
    components = [create_component_from_data(*recipe, relativistic_gamma=iw2d_input.relativistic_gamma)
                  for recipe in component_recipes]

    print(f'Element {name} completed')
    return Element(length=iw2d_input.length,
                   beta_x=beta_x, beta_y=beta_y,
                   components=components,
                   name=name, tag='IW2D', description='A resistive wall element created using IW2D')


def create_htcondor_input_file(iw2d_input: IW2DInput, name: str, directory: Union[str, Path]) -> None:
    exec_string = ""
    if iw2d_input.calculate_wake:
        exec_string += "wake_"
    exec_string += ("round" if isinstance(iw2d_input, RoundIW2DInput) else "flat") + "chamber.x"

    text = f"executable = {exec_string}\n" \
           f"input = {name}_input.txt\n" \
           f"ID = $(Cluster).$(Process)\n" \
           f"output = $(ID).out\n" \
           f"error = $(ID).err\n" \
           f"log = $(Cluster).log\n" \
           f"universe = vanilla\n" \
           f"initialdir = \n" \
           f"when_to_transfer_output = ON_EXIT\n" \
           f'+JobFlavour = "tomorrow"\n' \
           f'queue'

    with open(directory, 'w') as file:
        file.write(text)


def _verify_iw2d_binary_directory(ignore_missing_files: bool = False) -> None:
    bin_path = Path(get_iw2d_config_value('binary_directory'))
    if not ignore_missing_files:
        filenames = ('flatchamber.x', 'roundchamber.x', 'wake_flatchamber.x', 'wake_roundchamber.x')
        assert all(filename in listdir(bin_path) for filename in filenames), \
            "In order to utilize IW2D with PyWIT, the four binary files 'flatchamber.x', 'roundchamber.x', " \
            f"'wake_flatchamber.x' and 'wake_roundchamber.x' (as generated by IW2D) must be placed in the directory " \
            f"'{bin_path}'."


def _read_cst_data(filename: Union[str, Path]) -> np.ndarray:
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = []
    for l in lines:
        try:
            data.append([float(e) for e in l.strip().split()])
        except ValueError:
            pass

    return np.asarray(data)


def load_longitudinal_impedance_datafile(path: Union[str, Path]) -> Component:
    data = _read_cst_data(path)
    fs = data[:, 0]
    zs = data[:, 1] + 1j * data[:, 2]
    func = interp1d(x=fs, y=zs, kind='linear', assume_sorted=True, bounds_error=False, fill_value=(0., 0.))
    return Component(impedance=func, plane='z', source_exponents=(0, 0), test_exponents=(0, 0))


def load_transverse_impedance_datafile(path: Union[str, Path]) -> Tuple[Component, Component, Component, Component]:
    data = _read_cst_data(path)
    fs = data[:, 0]
    zs = [data[:, 2 * i + 1] + 1j * data[:, 2 * i + 2] for i in range(4)]
    components = tuple()
    for i, z in enumerate(zs):
        exponents = [int(j == i) for j in range(4)]
        func = interp1d(x=fs, y=z, kind='linear', assume_sorted=True, bounds_error=False, fill_value=(0., 0.))
        components += (Component(impedance=func,
                                 plane='x' if i % 2 == 0 else 'y',
                                 source_exponents=(exponents[0], exponents[1]),
                                 test_exponents=(exponents[2], exponents[3])),)

    return components


def load_longitudinal_wake_datafile(path: Union[str, Path]) -> Component:
    data = _read_cst_data(path)
    ts = data[:, 0]
    ws = data[:, 1] * 1e15
    func = interp1d(x=ts, y=ws, kind='linear', assume_sorted=True, bounds_error=False, fill_value=(0., 0.))
    return Component(wake=func, plane='z', source_exponents=(0, 0), test_exponents=(0, 0))


def load_transverse_wake_datafile(path: Union[str, Path]) -> Tuple[Component, Component, Component, Component]:
    data = _read_cst_data(path)
    ts = data[:, 0]
    ws = [data[:, i] * 1e15 for i in range(1, 5)]
    components = tuple()
    for i, w in enumerate(ws):
        exponents = [int(j == i) for j in range(4)]
        func = interp1d(x=ts, y=w, kind='linear', assume_sorted=True, bounds_error=False, fill_value=(0., 0.))
        components += (Component(wake=func,
                                 plane='x' if i % 2 == 0 else 'y',
                                 source_exponents=(exponents[0], exponents[1]),
                                 test_exponents=(exponents[2], exponents[3])),)

    return components


def clear_iw2d_hashmap():
    print("WARNING: This will delete all of PyWIT's information about performed IW2D computations. The impedance/wake"
          "data itself will not be touched.")
    answer = input("Clear PyWIT's IW2D cache? (y / N): ")
    if answer in ('y', 'Y', 'yes'):
        with open(Path.home().joinpath('pywit').joinpath('IW2D').joinpath('projects').joinpath('hashmap.pickle'),
                  'wb') as handle:
            pickle.dump(dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

from pywit.utilities import (create_resonator_component, create_resonator_element,
                             create_many_resonators_element,
                             create_resistive_wall_single_layer_approx_component,
                             create_resistive_wall_single_layer_approx_element,
                             create_taper_RW_approx_component,
                             create_taper_RW_approx_element)
from xwakes.wit.utilities import (_zlong_round_single_layer_approx,
                                  _zdip_round_single_layer_approx)
from test_common import relative_error
from pywit.parameters import *
from pywit.interface import (FlatIW2DInput, RoundIW2DInput, Sampling,
                             component_names)
from xwakes.wit.interface import _IW2DInputBase
from pywit.materials import layer_from_json_material_library, copper_at_temperature

from typing import Dict
from pathlib import Path

from pytest import raises, mark, fixture
import numpy as np
from numpy import testing as npt


def test_incompatible_dictionaries():
    rs = {"z0000": 2}
    qs = {"z0000": 2}
    fs = {"x0000": 2}
    with raises(AssertionError):
        create_resonator_element(1, 1, 1, rs, qs, fs)


def test_resonator_functions():
    # Assumes transverse shunt-impedance of 10^6 Ohm/m, and for simplicity, parallel shunt-impedance of 10^6 Ohm
    # Assumes resonance frequency of 5x10^9 Hz
    # Resulting values calculated using WolframAlpha
    params = [
        (1, 1e3, 1e-8,

         0.20000000000000799999999999999999999999998719999999999948800 +
         999999.9999999999999999999983999999999999360000000000000000 * 1j,

         4.000000000000159999999999999999999999999743999999999989760 * 1e-8 +
         0.1999999999999999999999999996799999999999872000000000000000 * 1j,

         2.0790568962711056134022117523654280541098216352367107167170 * 1e-52,

         -1.64047989255734044430680584824414225044330450145887282475 * 1e-52),

        (1, 2e6, 1e-9,

         400.0000639999999999983615997378560000000067108874737418239999725 +
         999999.999999974399995904000000000104857616777215999999570503202 * 1j,

         0.160000025599999999999344639895142400000002684354989496729599989 +
         399.999999999989759998361600000000041943046710886399999828201280 * 1j,

         4.78853721755932413794457670247279476645166922134243566471030 * 1e9,

         -4.6784101513043641063808617269665441974594819336165860702670 * 1e9),

        (1, 5e9, 1e-10,

         1e6,

         1e6,

         3.08108843686609735291707294749863455067119767175475504569435 * 1e15,

         -7.501299063383624459887240816560900288553129357784921049140 * 1e15),

        (1000, 1e3, 1e-8,

         2.000000000000159999920000009599987200003711998720000793599 * 1e-7 +
         1000.000000000039999960000001599995200001663999616000322559 * 1j,

         4.00000000000031999984000001919997440000742399744000158719 * 1e-14 +
         0.000200000000000007999992000000319999040000332799923200064511 * 1j,

         -1.0543650997439856725369172254858242688504101104167514427198 * 1e9,

         2.68491822708118498041028798429290190516850878075318261084845 * 1e13),

        (1000, 2e6, 1e-9,

         0.000400000127999966719965593595166724839180685593689238045309985 +
         1000.00015999986559992729600167937403126177831194885763757974 * 1j,

         1.600000511999866879862374380666899356722742374756952181239 * 1e-7 +
         0.4000000639999462399709184006717496125047113247795430550318 * 1j,

         -1.2144732628197967915943457947230343699439427175360563689881 * 1e8,

         3.09263019467959901903826466633770980448650547056418446854141 * 1e13),

        (1000, 5e9, 1e-10,

         1e6,

         1e6,

         1.23176441001988511772523795607220037036962745466914210177191 * 1e7,

         -3.136661725760914558037302917413121144031282294807848994107 * 1e13)
    ]
    for q, f, t, z_transverse, z_parallel, w_transverse, w_parallel in params:
        transverse = create_resonator_component('x', (0, 0, 0, 0), 1e6, q, 5e9)
        parallel = create_resonator_component('z', (0, 0, 0, 0), 1e6, q, 5e9)
        assert relative_error(transverse.impedance(f), z_transverse) < REL_TOL
        assert relative_error(parallel.impedance(f), z_parallel) < REL_TOL
        assert relative_error(transverse.wake(t), w_transverse) < REL_TOL
        assert relative_error(parallel.wake(t), w_parallel) < REL_TOL


def test_resonator():
    names = ['dipx', 'dipy', 'quadx', 'quady']
    data_directory = Path(__file__).parent.joinpath('test_data').joinpath('resonator')
    params: Dict = np.load(data_directory.joinpath('dict_gen.npy'),
                           allow_pickle=True).item()

    conversion = {'dipx': ('x', (1, 0, 0, 0)),
                  'dipy': ('y', (0, 1, 0, 0)),
                  'quadx': ('x', (0, 0, 1, 0)),
                  'quady': ('y', (0, 0, 0, 1))}
    separated_components = {name: [] for name in names}
    for dim, param_sets in params.items():
        plane, exponents = conversion[dim]
        for (shunt, q, f_res) in param_sets.values():
            separated_components[dim].append(create_resonator_component(plane, exponents, shunt, q, f_res))

    freq_lin = np.arange(1e7, 1e10, 5e5)
    freq_log = np.logspace(1, 12, 625 * 11)
    freq_mesh = np.sort(np.append(freq_lin, freq_log))
    freq_mesh = freq_mesh[::100]

    time_lin = np.arange(-12e-9, 205e-9, 5e-13)
    time_log = np.logspace(-10, np.log10(3e-5), 100 * 5)
    time = np.sort(np.append(time_lin, time_log))
    time = np.delete(time, np.argwhere(np.abs(time) < 1e-20))
    time_mesh = time[time > 0]
    time_mesh = time_mesh[::100]

    for k, v in separated_components.items():
        separated_components[k] = sum(v)

    correct_impedances = np.loadtxt(data_directory.joinpath('resonator_impedances.txt'),
                                    delimiter='\t', skiprows=0)
    correct_wakes = np.loadtxt(data_directory.joinpath('resonator_wakes.txt'),
                               delimiter='\t', skiprows=0)

    correct_impedances = {'dipx': correct_impedances[:, 1] + 1j * correct_impedances[:, 2],
                          'dipy': correct_impedances[:, 3] + 1j * correct_impedances[:, 4],
                          'quadx': correct_impedances[:, 5] + 1j * correct_impedances[:, 6],
                          'quady': correct_impedances[:, 7] + 1j * correct_impedances[:, 8]}
    correct_wakes = {name: data for name, data in zip(names, [correct_wakes[:, i] for i in range(1, 5)])}

    impedances = {name: separated_components[name].impedance(freq_mesh) for name in names}
    wakes = {name: separated_components[name].wake(time_mesh) / 1e15 for name in names}

    for correct, calculated in zip(correct_impedances.values(), impedances.values()):
        np.testing.assert_allclose(correct, calculated)

    for correct, calculated in zip(correct_wakes.values(), wakes.values()):
        np.testing.assert_allclose(correct, calculated)


FREQS_LIST = [10, 20, 40, 60, 80, 100]


@mark.parametrize('test_freq', FREQS_LIST)
def test_many_resonators(test_freq):
    params_dict = {
        'z0000':
            [
                {'r': 10, 'q': 10, 'f': 50},
                {'r': 40, 'q': 100, 'f': 60}
            ],
        'x1000':
            [
                {'r': 20, 'q': 10, 'f': 40},
                {'r': 50, 'q': 100, 'f': 70}
            ],
        'y0100':
            [
                {'r': 30, 'q': 10, 'f': 70},
                {'r': 60, 'q': 100, 'f': 80}
            ]
    }

    elem = create_many_resonators_element(length=1, beta_x=1, beta_y=1, params_dict=params_dict)

    elem1 = create_resonator_element(length=1, beta_x=1, beta_y=1,
                                     rs={'z0000': params_dict['z0000'][0]['r'],
                                         'x1000': params_dict['x1000'][0]['r'],
                                         'y0100': params_dict['y0100'][0]['r']},
                                     qs={'z0000': params_dict['z0000'][0]['q'],
                                         'x1000': params_dict['x1000'][0]['q'],
                                         'y0100': params_dict['y0100'][0]['q']},
                                     fs={'z0000': params_dict['z0000'][0]['f'],
                                         'x1000': params_dict['x1000'][0]['f'],
                                         'y0100': params_dict['y0100'][0]['f']}
                                     )

    elem2 = create_resonator_element(length=1, beta_x=1, beta_y=1,
                                     rs={'z0000': params_dict['z0000'][1]['r'],
                                         'x1000': params_dict['x1000'][1]['r'],
                                         'y0100': params_dict['y0100'][1]['r']},
                                     qs={'z0000': params_dict['z0000'][1]['q'],
                                         'x1000': params_dict['x1000'][1]['q'],
                                         'y0100': params_dict['y0100'][1]['q']},
                                     fs={'z0000': params_dict['z0000'][1]['f'],
                                         'x1000': params_dict['x1000'][1]['f'],
                                         'y0100': params_dict['y0100'][1]['f']}
                                     )

    sum_elem = elem1 + elem2

    for comp in ['z0000', 'x1000', 'y0100']:
        assert elem.get_component(comp).impedance(test_freq) == sum_elem.get_component(comp).impedance(test_freq)


def test_f_roi_resonator_component():
    r = 1e9
    f_roi_level = 0.3
    resonator_component_z = create_resonator_component(plane='z', exponents=(0, 0, 0, 0), r=1e9, q=1e6, f_r=400e6,
                                                       f_roi_level=f_roi_level)
    assert np.isclose(np.real(resonator_component_z.impedance(resonator_component_z.f_rois[0][0]))/r, f_roi_level)

    resonator_component_x = create_resonator_component(plane='x', exponents=(1, 0, 0, 0), r=1e9, q=1e6, f_r=400e6,
                                                       f_roi_level=f_roi_level)
    assert np.isclose(np.real(resonator_component_x.impedance(resonator_component_x.f_rois[0][0])/r), f_roi_level)


@fixture
def flat_symmetric_input():
    return FlatIW2DInput(machine='LHC', length=1.,
            relativistic_gamma=7460.5232328, calculate_wake=False,
            f_params=Sampling(1e3,1e13,0,added=(1e4,1e9),points_per_decade=0),
            top_bottom_symmetry=True,
            top_layers=[layer_from_json_material_library(thickness=np.inf,
                                                        material_key='Mo')],
            top_half_gap=0.002)


@fixture
def flat_single_plate_input():
    return FlatIW2DInput(machine='LHC', length=1.,
            relativistic_gamma=7460.5232328, calculate_wake=False,
            f_params=Sampling(1e3,1e13,0,added=(1e5,1e9),points_per_decade=0),
            top_bottom_symmetry=False,
            top_layers=[layer_from_json_material_library(thickness=np.inf,
                                                        material_key='graphite')],
            top_half_gap=0.004,
            bottom_layers=[],
            bottom_half_gap=np.inf,
            )


@fixture
def round_single_layer_input():
    return RoundIW2DInput(machine='LHC', length=0.03,
            relativistic_gamma=479.6, calculate_wake=False,
            f_params=Sampling(1e3,1e13,0,added=(1e8),points_per_decade=0),
            layers=[copper_at_temperature(thickness=np.inf,T=293)],
            inner_layer_radius=0.02,
            yokoya_factors=(1,1,1,0,0),
            )


@mark.parametrize("component_id", ['zlong', 'zxdip', 'zydip', 'zxqua', 'zyqua'])
def test_single_plate_RW_approx_error(component_id):

    with raises(NotImplementedError):
        flat_input = FlatIW2DInput(machine='LHC', length=1.,
            relativistic_gamma=7460.5232328, calculate_wake=False,
            f_params=Sampling(1e3,1e13,0,added=(1e4,1e9),points_per_decade=0),
            top_bottom_symmetry=False,
            top_layers=[layer_from_json_material_library(thickness=np.inf,
                                                         material_key='Mo')],
            top_half_gap=0.002,
            bottom_layers=[],
            bottom_half_gap=0.025,
            )
        _ , plane, exponents = component_names[component_id]
        create_resistive_wall_single_layer_approx_component(
                plane=plane,exponents=exponents,input_data=flat_input)


@mark.parametrize("component_id", ['zlong', 'zxdip', 'zydip', 'zxqua', 'zyqua'])
def test_more_than_one_layer_RW_approx_error(component_id):

    _ , plane, exponents = component_names[component_id]

    flat_input = FlatIW2DInput(
        machine='LHC', length=1.,
        relativistic_gamma=7460.5232328, calculate_wake=False,
        f_params=Sampling(1e3,1e13,0,added=(1e4,1e9),points_per_decade=0),
        top_bottom_symmetry=True,
        top_layers=[layer_from_json_material_library(thickness=0.002,
                                                     material_key='Mo'),
                    layer_from_json_material_library(thickness=np.inf,
                                                     material_key='W')],
        top_half_gap=0.002,
        )

    with raises(NotImplementedError):
        create_resistive_wall_single_layer_approx_component(
                plane=plane,exponents=exponents,input_data=flat_input)

    with raises(NotImplementedError):
        create_taper_RW_approx_component(
                plane=plane,exponents=exponents,input_data=flat_input,
                radius_small=0.001,radius_large=0.002)

    round_input = RoundIW2DInput(
        machine='LHC', length=2.,
        relativistic_gamma=479.6, calculate_wake=False,
        f_params=Sampling(1e4,1e10,0,added=(1e12,),points_per_decade=0),
        layers=[layer_from_json_material_library(thickness=0.02,
                                                     material_key='Mo'),
                    layer_from_json_material_library(thickness=np.inf,
                                                     material_key='graphite')],
        inner_layer_radius=0.004,
        yokoya_factors=(1,1,1,0,0),
        )

    with raises(NotImplementedError):
        create_resistive_wall_single_layer_approx_component(
                plane=plane,exponents=exponents,input_data=round_input)

    with raises(NotImplementedError):
        create_taper_RW_approx_component(
                plane=plane,exponents=exponents,input_data=round_input,
                radius_small=0.01,radius_large=0.02)


@mark.parametrize("component_id", ['zlong', 'zxdip', 'zydip', 'zxqua', 'zyqua'])
def test_wrong_input_RW_approx_error(component_id):

    _ , plane, exponents = component_names[component_id]

    input_data = _IW2DInputBase(
        machine='LHC', length=1.,
        relativistic_gamma=7460.5232328, calculate_wake=False,
        f_params=Sampling(1e3,1e13,0,added=(1e4,1e9),points_per_decade=0),
        )

    with raises(NotImplementedError):
        create_resistive_wall_single_layer_approx_component(
                plane=plane,exponents=exponents,input_data=input_data)

    with raises(NotImplementedError):
        create_taper_RW_approx_component(
                plane=plane,exponents=exponents,input_data=input_data,
                radius_small=0.001,radius_large=0.002)


# expected impedance values obtained from an exact flat chamber IW2D computation
@mark.parametrize("freq, component_id, component_str, expected_Z, rtol",
                  [
                    [1e4, 'z0000', 'zlong', 2.73790435e-03+1j*3.52379262e-03, 5e-2],
                    [1e4, 'x1000', 'zxdip', 1.41444672e+06+1j*2.94098193e+06, 5e-2],
                    [1e4, 'y0100', 'zydip', 2.90873693e+06+1j*5.92025573e+06, 5e-2],
                    [1e4, 'x0010', 'zxqua',-1.41444672e+06-1j*2.94098193e+06, 5e-2],
                    [1e4, 'y0001', 'zyqua', 1.41444672e+06+1j*2.94098193e+06, 5e-2],
                    [1e9, 'z0000', 'zlong', 1.15540625e+00+1j*1.15680934e+00, 2e-3],
                    [1e9, 'x1000', 'zxdip', 1.13144575e+04+1j*1.13465281e+04, 2e-3],
                    [1e9, 'y0100', 'zydip', 2.26466174e+04+1j*2.26502884e+04, 2e-3],
                    [1e9, 'x0010', 'zxqua',-1.13144575e+04-1j*1.13465281e+04, 2e-3],
                    [1e9, 'y0001', 'zyqua', 1.13144575e+04+1j*1.13465281e+04, 2e-3],
                   ]
                  )
def test_single_layer_RW_approx_flat_sym(freq, component_id, component_str,
                                         expected_Z, rtol, flat_symmetric_input):

    elem = create_resistive_wall_single_layer_approx_element(
            input_data=flat_symmetric_input, beta_x=10,beta_y=10,
            component_ids=(component_str,),
            )
    npt.assert_allclose(elem.get_component(component_id).impedance(freq),
                        expected_Z, rtol=rtol)


# expected impedance values obtained from an exact flat chamber IW2D computation
@mark.parametrize("freq, component_id, component_str, expected_Z, rtol",
                  [
                    [1e5, 'z0000', 'zlong', 5.01728296e-02+1j*8.19175077e-02, 3e-2],
                    [1e5, 'x1000', 'zxdip', 1.73918933e+05+1j*7.32919146e+05, 3e-2],
                    [1e5, 'y0100', 'zydip', 1.73918933e+05+1j*7.32919146e+05, 3e-2],
                    [1e5, 'x0010', 'zxqua',-1.73918933e+05-1j*7.32919146e+05, 3e-2],
                    [1e5, 'y0001', 'zyqua', 1.73918933e+05+1j*7.32919146e+05, 3e-2],
                    [1e9, 'z0000', 'zlong', 9.58598752e+00+1j*9.71657980e+00, 2e-3],
                    [1e9, 'x1000', 'zxdip', 1.40485975e+04+1j*1.44908306e+04, 1e-3],
                    [1e9, 'y0100', 'zydip', 1.40485975e+04+1j*1.44908306e+04, 1e-3],
                    [1e9, 'x0010', 'zxqua',-1.40485975e+04-1j*1.44908306e+04, 1e-3],
                    [1e9, 'y0001', 'zyqua', 1.40485975e+04+1j*1.44908306e+04, 1e-3],
                   ]
                  )
def test_single_layer_RW_approx_single_plate(freq, component_id, component_str,
                                             expected_Z, rtol, flat_single_plate_input):

    elem = create_resistive_wall_single_layer_approx_element(
            input_data=flat_single_plate_input, beta_x=1,beta_y=1,
            component_ids=(component_str,),
            )
    npt.assert_allclose(elem.get_component(component_id).impedance(freq),
                        expected_Z, rtol=rtol)


# expected impedance values obtained from an exact round chamber IW2D computation
# (except xquad and yquad which are set to zero)
@mark.parametrize("freq, component_id, component_str, expected_Z, rtol",
                  [
                    [1e8, 'z0000', 'zlong', 6.18359811e-04+1j*7.73533991e-04, 1e-6],
                    [1e8, 'x1000', 'zxdip', 1.47471685e+00+1j*1.49501534e+00, 1e-6],
                    [1e8, 'y0100', 'zydip', 1.47471685e+00+1j*1.49501534e+00, 1e-6],
                    [1e8, 'x0010', 'zxqua', 0, 1e-15],
                    [1e8, 'y0001', 'zyqua', 0, 1e-15],
                   ]
                  )
def test_single_layer_RW_approx_round(freq, component_id, component_str,
                                      expected_Z, rtol, round_single_layer_input):

    elem = create_resistive_wall_single_layer_approx_element(
            input_data=round_single_layer_input, beta_x=1,beta_y=1,
            component_ids=(component_str,),
            )
    npt.assert_allclose(elem.get_component(component_id).impedance(freq),
                        expected_Z, rtol=rtol)


# test algorithm for taper RW impedance (trapz vs. step-wise integral)
@mark.parametrize("component_id", ['zlong', 'zxdip'])
@mark.parametrize("freq", [1e9, np.array([1e4, 1e8])])
def test_taper_RW_algorithm(freq, component_id, round_single_layer_input):

    _, plane, exponents = component_names[component_id]
    radius_small = 0.005
    radius_large = 0.02

    comp_taper_trapz = create_taper_RW_approx_component(
                plane, exponents, round_single_layer_input,
                radius_small=radius_small, radius_large=radius_large,
                step_size=1e-5,
                )

    step_size = 1e-6
    npts = int(np.floor(abs(radius_large-radius_small)/step_size)+1)
    radii = np.linspace(radius_small,radius_large,npts)
    z_steps = 0.

    for radius in radii:
        if component_id == 'zlong':
            z_steps += _zlong_round_single_layer_approx(freq,
                            round_single_layer_input.relativistic_gamma,
                            round_single_layer_input.layers[0], radius,
                            round_single_layer_input.length/float(npts))
        else:
            z_steps += _zdip_round_single_layer_approx(freq,
                            round_single_layer_input.relativistic_gamma,
                            round_single_layer_input.layers[0], radius,
                            round_single_layer_input.length/float(npts))

    npt.assert_allclose(comp_taper_trapz.impedance(freq), z_steps, rtol=1e-3)


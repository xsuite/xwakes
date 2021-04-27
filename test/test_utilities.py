from pywib.utilities import create_resonator_component, create_resonator_element
from test.test_common import relative_error
from pywib.parameters import *

from typing import Dict
from pathlib import Path

from pytest import raises
import numpy as np


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

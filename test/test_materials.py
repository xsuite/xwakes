from pywit import materials

from numpy import testing
import numpy as np
from pytest import raises,mark


@mark.parametrize('missing_key',
                  ['dc_resistivity',
                   'resistivity_relaxation_time',
                   're_dielectric_constant',
                   'magnetic_susceptibility',
                   'permeability_relaxation_frequency',
                   ])
def test_layer_from_dict_missing_key(missing_key):
    input_dict = {'dc_resistivity': 1e-5,
                  'resistivity_relaxation_time': 1e-12,
                  're_dielectric_constant': 1,
                  'magnetic_susceptibility': 0,
                  'permeability_relaxation_frequency': np.inf,
                  }
    input_dict.pop(missing_key)
    with raises(AssertionError, match=f"{missing_key} missing from the input dictionary"):
        materials.layer_from_dict(1.,input_dict)


def test_layer_from_dict_two_missing_keys():
    input_dict = {'resistivity_relaxation_time': 1e-12,
                  'magnetic_susceptibility': 0,
                  'permeability_relaxation_frequency': np.inf,
                  }
    with raises(AssertionError, match=f"dc_resistivity, re_dielectric_constant missing from the input dictionary"):
        materials.layer_from_dict(1.,input_dict)


@mark.parametrize("T, B, rho",
                  [ [20, 0.5, 2.6e-10],
                    [20, 8.3, 7.6e-10],
                  ])
def test_copper(T,B,rho):
    RRR = 70.
    copper_layer = materials.copper_at_temperature(1., T, RRR, B)
    testing.assert_allclose(copper_layer.dc_resistivity, rho, atol=1e-11)


@mark.parametrize("material_key, material_function, T, RRR",
                  [ ["Cu", 'copper_at_temperature', 293, 70],
                    ["W",  'tungsten_at_temperature', 300, 70],
                  ])
@mark.parametrize("material_property, tolerance",
                  [ ['dc_resistivity', 2e-9],
                    ['resistivity_relaxation_time', 3e-15], 
                    ['re_dielectric_constant', 0],
                    ['magnetic_susceptibility', 0],
                    ['permeability_relaxation_frequency', 0],
                  ])
def test_materials_at_temperature(material_key, material_function, T,
                                  RRR, material_property, tolerance):
    B = 0.
    layer_at_temperature = getattr(materials,material_function)(1., T, RRR, B)
    layer_from_library = materials.layer_from_json_material_library(1.,material_key)
    
    assert layer_at_temperature.thickness == layer_from_library.thickness == 1.
    testing.assert_allclose(getattr(layer_at_temperature,material_property),
                            getattr(layer_from_library,material_property),
                            atol=tolerance)


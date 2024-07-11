import numpy as np
from scipy.constants import c, e

import xtrack as xt
import xfields as xf
import xwakes as xw
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts


exclude_contexts = ['ContextPyopencl', 'ContextCupy']

@for_all_test_contexts(excluding=exclude_contexts)
def test_multibunch_pipeline_longitudinal(test_context):
    # Filling scheme
    n_slots = 100
    filling_scheme = np.array(np.floor(np.random.rand(n_slots)+0.1), dtype=int)
    filling_scheme[0] = 1
    filled_slots = np.nonzero(filling_scheme)[0]
    n_bunches = len(filled_slots)
    n_bunches_0 = int(np.floor(n_bunches/2))
    bunch_numbers_0 = np.arange(n_bunches_0, dtype=int)
    bunch_numbers_1 = np.arange(n_bunches_0, n_bunches, dtype=int)

    print('initialising pipeline')
    comm = xt.pipeline.core.PipelineCommunicator()
    pipeline_manager = xt.PipelineManager(comm)
    pipeline_manager.add_particles(f'b0', 0)
    pipeline_manager.add_particles(f'b1', 0)
    pipeline_manager.add_element('wake')

    bunch_spacing = 25E-9*c
    sigma_zeta = bunch_spacing/20
    zeta_range = (-1.1*sigma_zeta, 1.1*sigma_zeta)
    num_slices = 1001
    dzeta = (zeta_range[1]-zeta_range[0])/num_slices
    zeta_slice_edges = np.linspace(zeta_range[0], zeta_range[1], num_slices+1)
    zeta_centers = zeta_slice_edges[:-1]+dzeta/2

    zeta_0 = []
    for bunch_number in bunch_numbers_0:
        zeta_0.append(zeta_centers-filled_slots[bunch_number]*bunch_spacing)
    zeta_0 = np.hstack(zeta_0)

    print('Initialising particles')
    particles_0 = xt.Particles(p0c=7E12, zeta=zeta_0,
                               x=np.random.randn(len(zeta_0)),
                               y=np.random.randn(len(zeta_0)),
                               _context=test_context)
    particles_0.init_pipeline('b0')

    zeta_1 = []
    for bunch_number in bunch_numbers_1:
        zeta_1.append(zeta_centers-filled_slots[bunch_number]*bunch_spacing)
    zeta_1 = np.hstack(zeta_1)
    particles_1 = xt.Particles(p0c=7E12, zeta=zeta_1,
                               x=np.random.randn(len(zeta_1)),
                               y=np.random.randn(len(zeta_1)),
                               _context=test_context)
    particles_1.init_pipeline('b1')

    print('Initialising wake')
    n_turns_wake = 1
    circumference = n_slots * bunch_spacing
    wake_table_name = xf.general._pkg_root.joinpath(
        '../test_data/HLLHC_wake.dat')
    wake_file_columns = ['time', 'longitudinal', 'dipole_x', 'dipole_y',
                         'quadrupole_x', 'quadrupole_y', 'dipole_xy',
                         'quadrupole_xy', 'dipole_yx', 'quadrupole_yx',
                         'constant_x', 'constant_y']
    components = ['longitudinal']
    wake_df = xw.read_headtail_file(wake_table_name,
                                    wake_file_columns)
    wf_0 = xw.WakeFromTable(wake_df, columns=components)
    wf_0.configure_for_tracking(zeta_range=zeta_range,
                                num_slices=num_slices,
                                bunch_spacing_zeta=bunch_spacing,
                                filling_scheme=filling_scheme,
                                bunch_numbers=bunch_numbers_0,
                                num_turns=n_turns_wake,
                                circumference=circumference
                                )
    wf_0._wake_tracker.init_pipeline(pipeline_manager=pipeline_manager,
                                     element_name='wake',
                                     partners_names=['b1'])

    wf_1 = xw.WakeFromTable(wake_df, columns=components)
    wf_1.configure_for_tracking(zeta_range=zeta_range,
                                num_slices=num_slices,
                                bunch_spacing_zeta=bunch_spacing,
                                filling_scheme=filling_scheme,
                                bunch_numbers=bunch_numbers_1,
                                num_turns=n_turns_wake,
                                circumference=circumference
                                )
    wf_1._wake_tracker.init_pipeline(pipeline_manager=pipeline_manager,
                                     element_name='wake',
                                     partners_names=['b0'])

    print('Initialising lines')
    line_0 = xt.Line(elements=[wf_0])
    line_1 = xt.Line(elements=[wf_1])
    print('Initialising multitracker')
    line_0.build_tracker()
    line_1.build_tracker()
    multitracker = xt.PipelineMultiTracker(
        branches=[xt.PipelineBranch(line=line_0, particles=particles_0),
                  xt.PipelineBranch(line=line_1, particles=particles_1)])
    print('Tracking')
    pipeline_manager.verbose = True
    multitracker.track(num_turns=1)
    print('loading test data')
    scaling_constant = particles_0.q0**2 * e**2 / (particles_0.p0c[0] * e)
    wake_data_raw = np.loadtxt(wake_table_name)
    print('Computing constant wake for comparison')
    all_zetas = np.hstack([particles_0.zeta, particles_1.zeta])
    wake_component = 'longitudinal'
    i_wake_comp = wake_file_columns.index(wake_component)
    wake_time = -1E-9*np.flip(wake_data_raw[:, 0])*c
    wake_data = -1E12*scaling_constant*np.flip(wake_data_raw[:, i_wake_comp])
    interpolated_wake_0 = np.zeros_like(particles_0.x)
    for ipart in range(len(particles_0.x)):
        zeta_wake = particles_0.zeta[ipart]-all_zetas
        zeta_wake = zeta_wake[zeta_wake < 0]
        interpolated_wake = np.interp(zeta_wake, wake_time, wake_data)
        interpolated_wake_0[ipart] = np.sum(interpolated_wake)
    interpolated_wake_1 = np.zeros_like(particles_1.x)
    for ipart in range(len(particles_1.x)):
        zeta_wake = particles_1.zeta[ipart]-all_zetas
        zeta_wake = zeta_wake[zeta_wake < 0]
        interpolated_wake = np.interp(zeta_wake, wake_time, wake_data)
        interpolated_wake_1[ipart] = np.sum(interpolated_wake)
    print('Testing longitudinal')
    kicks_0 = interpolated_wake_0
    norm = np.max(np.abs(kicks_0))
    kicks_0 /= norm
    kicks_1 = interpolated_wake_1 / norm
    xo.assert_allclose(particles_0.delta/norm, kicks_0,
                       rtol=1e-3, atol=1e-10*np.abs(kicks_0).max())
    xo.assert_allclose(particles_1.delta/norm, kicks_1,
                       rtol=1e-3, atol=1e-10*np.abs(kicks_0).max())


@for_all_test_contexts(excluding=exclude_contexts)
def test_multibunch_pipeline_transverse_constant(test_context):
    # Filling scheme
    n_slots = 100
    filling_scheme = np.array(np.floor(np.random.rand(n_slots) + 0.1),
                              dtype=int)
    filling_scheme[0] = 1
    filled_slots = np.nonzero(filling_scheme)[0]
    n_bunches = len(filled_slots)
    n_bunches_0 = int(np.floor(n_bunches / 2))
    bunch_numbers_0 = np.arange(n_bunches_0, dtype=int)
    bunch_numbers_1 = np.arange(n_bunches_0, n_bunches, dtype=int)

    print('initialising pipeline')
    comm = xt.pipeline.core.PipelineCommunicator()
    pipeline_manager = xt.PipelineManager(comm)
    pipeline_manager.add_particles(f'b0', 0)
    pipeline_manager.add_particles(f'b1', 0)
    pipeline_manager.add_element('wake')

    bunch_spacing = 25E-9 * c
    sigma_zeta = bunch_spacing / 20
    zeta_range = (-1.1 * sigma_zeta, 1.1 * sigma_zeta)
    num_slices = 1001
    dzeta = (zeta_range[1] - zeta_range[0]) / num_slices
    zeta_slice_edges = np.linspace(zeta_range[0], zeta_range[1], num_slices + 1)
    zeta_centers = zeta_slice_edges[:-1] + dzeta / 2

    zeta_0 = []
    for bunch_number in bunch_numbers_0:
        zeta_0.append(zeta_centers - filled_slots[bunch_number] * bunch_spacing)
    zeta_0 = np.hstack(zeta_0)

    print('Initialising particles')
    particles_0 = xt.Particles(p0c=7E12, zeta=zeta_0,
                               x=np.random.randn(len(zeta_0)),
                               y=np.random.randn(len(zeta_0)),
                               _context=test_context)
    particles_0.init_pipeline('b0')

    zeta_1 = []
    for bunch_number in bunch_numbers_1:
        zeta_1.append(zeta_centers - filled_slots[bunch_number] * bunch_spacing)
    zeta_1 = np.hstack(zeta_1)
    particles_1 = xt.Particles(p0c=7E12, zeta=zeta_1,
                               x=np.random.randn(len(zeta_1)),
                               y=np.random.randn(len(zeta_1)),
                               _context=test_context)
    particles_1.init_pipeline('b1')

    print('Initialising wake')
    n_turns_wake = 1
    circumference = n_slots * bunch_spacing
    wake_table_name = xf.general._pkg_root.joinpath(
        '../test_data/HLLHC_wake.dat')
    wake_file_columns = ['time', 'longitudinal', 'dipole_x', 'dipole_y',
                         'quadrupole_x', 'quadrupole_y', 'dipole_xy',
                         'quadrupole_xy', 'dipole_yx', 'quadrupole_yx',
                         'constant_x', 'constant_y']
    components = ['constant_x', 'constant_y']
    wake_df = xw.read_headtail_file(wake_table_name,
                                    wake_file_columns)
    wf_0 = xw.WakeFromTable(wake_df, columns=components)
    wf_0.configure_for_tracking(zeta_range=zeta_range,
                                num_slices=num_slices,
                                bunch_spacing_zeta=bunch_spacing,
                                filling_scheme=filling_scheme,
                                bunch_numbers=bunch_numbers_0,
                                num_turns=n_turns_wake,
                                circumference=circumference
                                )
    wf_0._wake_tracker.init_pipeline(pipeline_manager=pipeline_manager,
                                     element_name='wake',
                                     partners_names=['b1'])

    wf_1 = xw.WakeFromTable(wake_df, columns=components)
    wf_1.configure_for_tracking(zeta_range=zeta_range,
                                num_slices=num_slices,
                                bunch_spacing_zeta=bunch_spacing,
                                filling_scheme=filling_scheme,
                                bunch_numbers=bunch_numbers_1,
                                num_turns=n_turns_wake,
                                circumference=circumference
                                )
    wf_1._wake_tracker.init_pipeline(pipeline_manager=pipeline_manager,
                                     element_name='wake',
                                     partners_names=['b0'])

    print('Initialising lines')
    line_0 = xt.Line(elements=[wf_0])
    line_1 = xt.Line(elements=[wf_1])
    print('Initialising multitracker')
    line_0.build_tracker()
    line_1.build_tracker()
    multitracker = xt.PipelineMultiTracker(
        branches=[xt.PipelineBranch(line=line_0, particles=particles_0),
                  xt.PipelineBranch(line=line_1, particles=particles_1)])
    print('Tracking')
    pipeline_manager.verbose = True
    multitracker.track(num_turns=1)
    print('loading test data')
    scaling_constant = -particles_0.q0 ** 2 * e ** 2 / (particles_0.p0c[0] * e)
    wake_data_raw = np.loadtxt(wake_table_name)
    print('Computing constant wake for comparison')
    all_zetas = np.hstack([particles_0.zeta, particles_1.zeta])
    wake_component = 'constant_x'
    i_wake_comp = wake_file_columns.index(wake_component)
    wake_time = -1E-9 * np.flip(wake_data_raw[:, 0]) * c
    wake_data = -1E15 * scaling_constant * np.flip(
        wake_data_raw[:, i_wake_comp])
    interpolated_wake_0 = np.zeros_like(particles_0.x)
    for ipart in range(len(particles_0.x)):
        zeta_wake = particles_0.zeta[ipart] - all_zetas
        zeta_wake = zeta_wake[zeta_wake < 0]
        interpolated_wake = np.interp(zeta_wake, wake_time, wake_data)
        interpolated_wake_0[ipart] = np.sum(interpolated_wake)
    interpolated_wake_1 = np.zeros_like(particles_1.x)
    for ipart in range(len(particles_1.x)):
        zeta_wake = particles_1.zeta[ipart] - all_zetas
        zeta_wake = zeta_wake[zeta_wake < 0]
        interpolated_wake = np.interp(zeta_wake, wake_time, wake_data)
        interpolated_wake_1[ipart] = np.sum(interpolated_wake)
    print('Testing constant X')
    kicks_0 = interpolated_wake_0
    norm = np.max(np.abs(kicks_0))
    kicks_0 /= norm
    kicks_1 = interpolated_wake_1 / norm
    assert np.allclose(particles_0.px / norm, kicks_0)
    assert np.allclose(particles_1.px / norm, kicks_1)
    wake_component = 'constant_y'
    i_wake_comp = wake_file_columns.index(wake_component)
    wake_time = -1E-9 * np.flip(wake_data_raw[:, 0]) * c
    wake_data = -1E15 * scaling_constant * np.flip(
        wake_data_raw[:, i_wake_comp])
    interpolated_wake_0 = np.zeros_like(particles_0.x)
    for ipart in range(len(particles_0.x)):
        zeta_wake = particles_0.zeta[ipart] - all_zetas
        zeta_wake = zeta_wake[zeta_wake < 0]
        interpolated_wake = np.interp(zeta_wake, wake_time, wake_data)
        interpolated_wake_0[ipart] = np.sum(interpolated_wake)
    interpolated_wake_1 = np.zeros_like(particles_1.x)
    for ipart in range(len(particles_1.x)):
        zeta_wake = particles_1.zeta[ipart] - all_zetas
        zeta_wake = zeta_wake[zeta_wake < 0]
        interpolated_wake = np.interp(zeta_wake, wake_time, wake_data)
        interpolated_wake_1[ipart] = np.sum(interpolated_wake)
    print('Testing constant Y')
    kicks_0 = interpolated_wake_0
    norm = np.max(np.abs(kicks_0))
    kicks_0 /= norm
    kicks_1 = interpolated_wake_1 / norm
    xo.assert_allclose(particles_0.py / norm, kicks_0, rtol=1e-4, atol=5e-14)
    xo.assert_allclose(particles_1.py / norm, kicks_1, rtol=1e-4, atol=5e-14)


@for_all_test_contexts(excluding=exclude_contexts)
def test_multibunch_pipeline_transverse_dipole_direct(test_context):
    # Filling scheme
    n_slots = 100
    filling_scheme = np.array(np.floor(np.random.rand(n_slots) + 0.1),
                              dtype=int)
    filling_scheme[0] = 1
    filled_slots = np.nonzero(filling_scheme)[0]
    n_bunches = len(filled_slots)
    n_bunches_0 = int(np.floor(n_bunches / 2))
    bunch_numbers_0 = np.arange(n_bunches_0, dtype=int)
    bunch_numbers_1 = np.arange(n_bunches_0, n_bunches, dtype=int)

    print('initialising pipeline')
    comm = xt.pipeline.core.PipelineCommunicator()
    pipeline_manager = xt.PipelineManager(comm)
    pipeline_manager.add_particles(f'b0', 0)
    pipeline_manager.add_particles(f'b1', 0)
    pipeline_manager.add_element('wake')

    bunch_spacing = 25E-9 * c
    sigma_zeta = bunch_spacing / 20
    zeta_range = (-1.1 * sigma_zeta, 1.1 * sigma_zeta)
    num_slices = 1001
    dzeta = (zeta_range[1] - zeta_range[0]) / num_slices
    zeta_slice_edges = np.linspace(zeta_range[0], zeta_range[1], num_slices + 1)
    zeta_centers = zeta_slice_edges[:-1] + dzeta / 2

    zeta_0 = []
    for bunch_number in bunch_numbers_0:
        zeta_0.append(zeta_centers - filled_slots[bunch_number] * bunch_spacing)
    zeta_0 = np.hstack(zeta_0)

    ioffset = np.argmin(np.abs(zeta_0))
    print('Initialising particles')
    particles_0 = xt.Particles(_context=test_context, p0c=7E12, zeta=zeta_0)
    particles_0.init_pipeline('b0')
    particles_0.x[ioffset] += 0.8
    particles_0.y[ioffset] += 1.9

    zeta_1 = []
    for bunch_number in bunch_numbers_1:
        zeta_1.append(zeta_centers - filled_slots[bunch_number] * bunch_spacing)
    zeta_1 = np.hstack(zeta_1)
    particles_1 = xt.Particles(_context=test_context, p0c=7E12, zeta=zeta_1)
    particles_1.init_pipeline('b1')

    print('Initialising wake')
    n_turns_wake = 1
    circumference = n_slots * bunch_spacing
    wake_table_name = xf.general._pkg_root.joinpath(
        '../test_data/HLLHC_wake.dat')
    wake_file_columns = ['time', 'longitudinal', 'dipole_x', 'dipole_y',
                         'quadrupole_x', 'quadrupole_y', 'dipole_xy',
                         'quadrupole_xy', 'dipole_yx', 'quadrupole_yx',
                         'constant_x', 'constant_y']
    components = ['dipole_x', 'dipole_y']
    wake_df = xw.read_headtail_file(wake_table_name,
                                    wake_file_columns)
    wf_0 = xw.WakeFromTable(wake_df, columns=components)
    wf_0.configure_for_tracking(zeta_range=zeta_range,
                                num_slices=num_slices,
                                bunch_spacing_zeta=bunch_spacing,
                                filling_scheme=filling_scheme,
                                bunch_numbers=bunch_numbers_0,
                                num_turns=n_turns_wake,
                                circumference=circumference
                                )
    wf_0._wake_tracker.init_pipeline(pipeline_manager=pipeline_manager,
                                     element_name='wake',
                                     partners_names=['b1'])

    wf_1 = xw.WakeFromTable(wake_df, columns=components)
    wf_1.configure_for_tracking(zeta_range=zeta_range,
                                num_slices=num_slices,
                                bunch_spacing_zeta=bunch_spacing,
                                filling_scheme=filling_scheme,
                                bunch_numbers=bunch_numbers_1,
                                num_turns=n_turns_wake,
                                circumference=circumference
                                )
    wf_1._wake_tracker.init_pipeline(pipeline_manager=pipeline_manager,
                                     element_name='wake',
                                     partners_names=['b0'])

    print('Initialising lines')
    line_0 = xt.Line(elements=[wf_0])
    line_1 = xt.Line(elements=[wf_1])
    print('Initialising multitracker')
    line_0.build_tracker()
    line_1.build_tracker()
    multitracker = xt.PipelineMultiTracker(
        branches=[xt.PipelineBranch(line=line_0, particles=particles_0),
                  xt.PipelineBranch(line=line_1, particles=particles_1)])
    print('Tracking')
    pipeline_manager.verbose = True
    multitracker.track(num_turns=1)
    print('loading test data')
    scaling_constant = -particles_0.q0 ** 2 * e ** 2 / (particles_0.p0c[0] * e)
    wake_data = np.loadtxt(wake_table_name)
    print('Testing')
    wake_component = 'dipole_x'
    i_w_comp = wake_file_columns.index(wake_component)
    interpolated_wake_0 = (
            particles_0.x[ioffset] *
            np.interp(particles_0.zeta,
                      -1E-9 * np.flip(wake_data[:, 0]) * c,
                      -1E15 * scaling_constant * np.flip(
                          wake_data[:, i_w_comp])))
    interpolated_wake_1 = (
            particles_0.x[ioffset] *
            np.interp(particles_1.zeta,
                      -1E-9 * np.flip(wake_data[:, 0]) * c,
                      -1E15 * scaling_constant * np.flip(
                          wake_data[:, i_w_comp])))
    norm = np.max(np.abs(interpolated_wake_0))
    xo.assert_allclose(particles_0.px / norm, interpolated_wake_0 / norm)
    xo.assert_allclose(particles_1.px / norm, interpolated_wake_1 / norm)
    wake_component = 'dipole_y'
    i_w_comp = wake_file_columns.index(wake_component)
    interpolated_wake_0 = (
            particles_0.y[ioffset] *
            np.interp(particles_0.zeta,
                      -1E-9 * np.flip(wake_data[:, 0]) * c,
                      -1E15 * scaling_constant * np.flip(
                          wake_data[:, i_w_comp])))
    interpolated_wake_1 = (
            particles_0.y[ioffset] *
            np.interp(particles_1.zeta,
                      -1E-9 * np.flip(wake_data[:, 0]) * c,
                      -1E15 * scaling_constant * np.flip(
                          wake_data[:, i_w_comp])))
    norm = np.max(np.abs(interpolated_wake_0))
    xo.assert_allclose(particles_0.py / norm, interpolated_wake_0 / norm)
    xo.assert_allclose(particles_1.py / norm, interpolated_wake_1 / norm)


@for_all_test_contexts(excluding=exclude_contexts)
def test_multibunch_pipeline_transverse_dipole_coupling(test_context):
    # Filling scheme
    n_slots = 100
    filling_scheme = np.array(np.floor(np.random.rand(n_slots) + 0.1),
                              dtype=int)
    filling_scheme[0] = 1
    filled_slots = np.nonzero(filling_scheme)[0]
    n_bunches = len(filled_slots)
    n_bunches_0 = int(np.floor(n_bunches / 2))
    bunch_numbers_0 = np.arange(n_bunches_0, dtype=int)
    bunch_numbers_1 = np.arange(n_bunches_0, n_bunches, dtype=int)

    print('initialising pipeline')
    comm = xt.pipeline.core.PipelineCommunicator()
    pipeline_manager = xt.PipelineManager(comm)
    pipeline_manager.add_particles(f'b0', 0)
    pipeline_manager.add_particles(f'b1', 0)
    pipeline_manager.add_element('wake')

    bunch_spacing = 25E-9 * c
    sigma_zeta = bunch_spacing / 20
    zeta_range = (-1.1 * sigma_zeta, 1.1 * sigma_zeta)
    num_slices = 1001
    dzeta = (zeta_range[1] - zeta_range[0]) / num_slices
    zeta_slice_edges = np.linspace(zeta_range[0], zeta_range[1], num_slices + 1)
    zeta_centers = zeta_slice_edges[:-1] + dzeta / 2

    zeta_0 = []
    for bunch_number in bunch_numbers_0:
        zeta_0.append(zeta_centers - filled_slots[bunch_number] * bunch_spacing)
    zeta_0 = np.hstack(zeta_0)

    ioffset = np.argmin(np.abs(zeta_0))
    print('Initialising particles')
    particles_0 = xt.Particles(_context=test_context, p0c=7E12, zeta=zeta_0)
    particles_0.init_pipeline('b0')
    particles_0.x[ioffset] += 2.0
    particles_0.y[ioffset] += 0.5

    zeta_1 = []
    for bunch_number in bunch_numbers_1:
        zeta_1.append(zeta_centers - filled_slots[bunch_number] * bunch_spacing)
    zeta_1 = np.hstack(zeta_1)
    particles_1 = xt.Particles(_context=test_context, p0c=7E12, zeta=zeta_1)
    particles_1.init_pipeline('b1')

    print('Initialising wake')
    n_turns_wake = 1
    circumference = n_slots * bunch_spacing
    wake_table_name = xf.general._pkg_root.joinpath(
        '../test_data/HLLHC_wake.dat')
    wake_file_columns = ['time', 'longitudinal', 'dipole_x', 'dipole_y',
                         'quadrupole_x', 'quadrupole_y', 'dipole_xy',
                         'quadrupole_xy', 'dipole_yx', 'quadrupole_yx',
                         'constant_x', 'constant_y']
    components = ['dipole_xy', 'dipole_yx']
    wake_df = xw.read_headtail_file(wake_table_name,
                                    wake_file_columns)
    wf_0 = xw.WakeFromTable(wake_df, columns=components)
    wf_0.configure_for_tracking(zeta_range=zeta_range,
                                num_slices=num_slices,
                                bunch_spacing_zeta=bunch_spacing,
                                filling_scheme=filling_scheme,
                                bunch_numbers=bunch_numbers_0,
                                num_turns=n_turns_wake,
                                circumference=circumference
                                )
    wf_0._wake_tracker.init_pipeline(pipeline_manager=pipeline_manager,
                                     element_name='wake',
                                     partners_names=['b1'])

    wf_1 = xw.WakeFromTable(wake_df, columns=components)
    wf_1.configure_for_tracking(zeta_range=zeta_range,
                                num_slices=num_slices,
                                bunch_spacing_zeta=bunch_spacing,
                                filling_scheme=filling_scheme,
                                bunch_numbers=bunch_numbers_1,
                                num_turns=n_turns_wake,
                                circumference=circumference
                                )
    wf_1._wake_tracker.init_pipeline(pipeline_manager=pipeline_manager,
                                     element_name='wake',
                                     partners_names=['b0'])

    print('Initialising lines')
    line_0 = xt.Line(elements=[wf_0])
    line_1 = xt.Line(elements=[wf_1])
    print('Initialising multitracker')
    line_0.build_tracker()
    line_1.build_tracker()
    multitracker = xt.PipelineMultiTracker(
        branches=[xt.PipelineBranch(line=line_0, particles=particles_0),
                  xt.PipelineBranch(line=line_1, particles=particles_1)])
    print('Tracking')
    pipeline_manager.verbose = True
    multitracker.track(num_turns=1)
    print('loading test data')
    scaling_constant = -particles_0.q0 ** 2 * e ** 2 / (particles_0.p0c[0] * e)
    wake_data = np.loadtxt(wake_table_name)
    print('Testing XY')
    wake_component = 'dipole_xy'
    i_wake_comp = wake_file_columns.index(wake_component)
    interpolated_wake_0 = (particles_0.y[ioffset] *
                           np.interp(particles_0.zeta,
                                     -1E-9 * np.flip(wake_data[:, 0]) * c,
                                     -1E15 * scaling_constant *
                                     np.flip(wake_data[:, i_wake_comp])))
    interpolated_wake_1 = (particles_0.y[ioffset] *
                           np.interp(particles_1.zeta,
                                     -1E-9 * np.flip(wake_data[:, 0]) * c,
                                     -1E15 * scaling_constant *
                                     np.flip(wake_data[:, i_wake_comp])))
    norm = np.max(np.abs(interpolated_wake_0))
    assert np.allclose(particles_0.px / norm, interpolated_wake_0 / norm)
    assert np.allclose(particles_1.px / norm, interpolated_wake_1 / norm)
    print('Testing YX')
    wake_component = 'dipole_yx'
    i_wake_comp = wake_file_columns.index(wake_component)
    interpolated_wake_0 = (particles_0.x[ioffset] *
                           np.interp(particles_0.zeta,
                                     -1E-9 * np.flip(wake_data[:, 0]) * c,
                                     -1E15 * scaling_constant *
                                     np.flip(wake_data[:, i_wake_comp])))
    interpolated_wake_1 = (particles_0.x[ioffset] *
                           np.interp(particles_1.zeta,
                                     -1E-9 * np.flip(wake_data[:, 0]) * c,
                                     -1E15 * scaling_constant *
                                     np.flip(wake_data[:, i_wake_comp])))
    norm = np.max(np.abs(interpolated_wake_0))
    assert np.allclose(particles_0.py / norm, interpolated_wake_0 / norm)
    assert np.allclose(particles_1.py / norm, interpolated_wake_1 / norm)


@for_all_test_contexts(excluding=exclude_contexts)
def test_multibunch_pipeline_transverse_quadrupole_direct(test_context):
    # Filling scheme
    n_slots = 100
    filling_scheme = np.array(np.floor(np.random.rand(n_slots) + 0.1),
                              dtype=int)
    filling_scheme[0] = 1
    filled_slots = np.nonzero(filling_scheme)[0]
    n_bunches = len(filled_slots)
    n_bunches_0 = int(np.floor(n_bunches / 2))
    bunch_numbers_0 = np.arange(n_bunches_0, dtype=int)
    bunch_numbers_1 = np.arange(n_bunches_0, n_bunches, dtype=int)

    print('initialising pipeline')
    comm = xt.pipeline.core.PipelineCommunicator()
    pipeline_manager = xt.PipelineManager(comm)
    pipeline_manager.add_particles(f'b0', 0)
    pipeline_manager.add_particles(f'b1', 0)
    pipeline_manager.add_element('wake')

    bunch_spacing = 25E-9 * c
    sigma_zeta = bunch_spacing / 20
    zeta_range = (-1.1 * sigma_zeta, 1.1 * sigma_zeta)
    num_slices = 1001
    dzeta = (zeta_range[1] - zeta_range[0]) / num_slices
    zeta_slice_edges = np.linspace(zeta_range[0], zeta_range[1], num_slices + 1)
    zeta_centers = zeta_slice_edges[:-1] + dzeta / 2

    zeta_0 = []
    for bunch_number in bunch_numbers_0:
        zeta_0.append(zeta_centers - filled_slots[bunch_number] * bunch_spacing)
    zeta_0 = np.hstack(zeta_0)

    print('Initialising particles')
    particles_0 = xt.Particles(p0c=7E12, zeta=zeta_0,
                               x=np.random.randn(len(zeta_0)),
                               y=np.random.randn(len(zeta_0)),
                               _context=test_context)
    particles_0.init_pipeline('b0')

    zeta_1 = []
    for bunch_number in bunch_numbers_1:
        zeta_1.append(zeta_centers - filled_slots[bunch_number] * bunch_spacing)
    zeta_1 = np.hstack(zeta_1)
    particles_1 = xt.Particles(p0c=7E12, zeta=zeta_1,
                               x=np.random.randn(len(zeta_1)),
                               y=np.random.randn(len(zeta_1)),
                               _context=test_context)
    particles_1.init_pipeline('b1')

    print('Initialising wake')
    n_turns_wake = 1
    circumference = n_slots * bunch_spacing
    wake_table_name = xf.general._pkg_root.joinpath(
        '../test_data/HLLHC_wake.dat')
    wake_file_columns = ['time', 'longitudinal', 'dipole_x', 'dipole_y',
                         'quadrupole_x', 'quadrupole_y', 'dipole_xy',
                         'quadrupole_xy', 'dipole_yx', 'quadrupole_yx',
                         'constant_x', 'constant_y']
    components = ['quadrupole_x', 'quadrupole_y']
    wake_df = xw.read_headtail_file(wake_table_name,
                                    wake_file_columns)
    wf_0 = xw.WakeFromTable(wake_df, columns=components)
    wf_0.configure_for_tracking(zeta_range=zeta_range,
                                num_slices=num_slices,
                                bunch_spacing_zeta=bunch_spacing,
                                filling_scheme=filling_scheme,
                                bunch_numbers=bunch_numbers_0,
                                num_turns=n_turns_wake,
                                circumference=circumference
                                )
    wf_0._wake_tracker.init_pipeline(pipeline_manager=pipeline_manager,
                                     element_name='wake',
                                     partners_names=['b1'])

    wf_1 = xw.WakeFromTable(wake_df, columns=components)
    wf_1.configure_for_tracking(zeta_range=zeta_range,
                                num_slices=num_slices,
                                bunch_spacing_zeta=bunch_spacing,
                                filling_scheme=filling_scheme,
                                bunch_numbers=bunch_numbers_1,
                                num_turns=n_turns_wake,
                                circumference=circumference
                                )
    wf_1._wake_tracker.init_pipeline(pipeline_manager=pipeline_manager,
                                     element_name='wake',
                                     partners_names=['b0'])


    print('Initialising lines')
    line_0 = xt.Line(elements=[wf_0])
    line_1 = xt.Line(elements=[wf_1])
    print('Initialising multitracker')
    line_0.build_tracker()
    line_1.build_tracker()
    multitracker = xt.PipelineMultiTracker(
        branches=[xt.PipelineBranch(line=line_0, particles=particles_0),
                  xt.PipelineBranch(line=line_1, particles=particles_1)])
    print('Tracking')
    pipeline_manager.verbose = True
    multitracker.track(num_turns=1)
    print('loading test data')
    scaling_constant = -particles_0.q0 ** 2 * e ** 2 / (particles_0.p0c[0] * e)
    wake_data_raw = np.loadtxt(wake_table_name)
    print('Computing quad wake for comparison')
    all_zetas = np.hstack([particles_0.zeta, particles_1.zeta])
    wake_component = 'quadrupole_x'
    i_w_comp = wake_file_columns.index(wake_component)
    wake_time = -1E-9 * np.flip(wake_data_raw[:, 0]) * c
    wake_data = -1E15 * scaling_constant * np.flip(wake_data_raw[:, i_w_comp])
    interpolated_wake_0 = np.zeros_like(particles_0.x)
    for ipart in range(len(particles_0.x)):
        zeta_wake = particles_0.zeta[ipart] - all_zetas
        zeta_wake = zeta_wake[zeta_wake < 0]
        interpolated_wake = np.interp(zeta_wake, wake_time, wake_data)
        interpolated_wake_0[ipart] = np.sum(interpolated_wake)
    interpolated_wake_1 = np.zeros_like(particles_1.x)
    for ipart in range(len(particles_1.x)):
        zeta_wake = particles_1.zeta[ipart] - all_zetas
        zeta_wake = zeta_wake[zeta_wake < 0]
        interpolated_wake = np.interp(zeta_wake, wake_time, wake_data)
        interpolated_wake_1[ipart] = np.sum(interpolated_wake)
    print('Testing quad x')
    kicks_0 = particles_0.x * interpolated_wake_0
    norm = np.max(np.abs(kicks_0))
    kicks_0 /= norm
    kicks_1 = particles_1.x * interpolated_wake_1 / norm
    assert np.allclose(particles_0.px / norm, kicks_0)
    assert np.allclose(particles_1.px / norm, kicks_1)
    wake_component = 'quadrupole_y'
    i_w_comp = wake_file_columns.index(wake_component)
    wake_time = -1E-9 * np.flip(wake_data_raw[:, 0]) * c
    wake_data = -1E15 * scaling_constant * np.flip(wake_data_raw[:, i_w_comp])
    interpolated_wake_0 = np.zeros_like(particles_0.x)
    for ipart in range(len(particles_0.x)):
        zeta_wake = particles_0.zeta[ipart] - all_zetas
        zeta_wake = zeta_wake[zeta_wake < 0]
        interpolated_wake = np.interp(zeta_wake, wake_time, wake_data)
        interpolated_wake_0[ipart] = np.sum(interpolated_wake)
    interpolated_wake_1 = np.zeros_like(particles_1.x)
    for ipart in range(len(particles_1.x)):
        zeta_wake = particles_1.zeta[ipart] - all_zetas
        zeta_wake = zeta_wake[zeta_wake < 0]
        interpolated_wake = np.interp(zeta_wake, wake_time, wake_data)
        interpolated_wake_1[ipart] = np.sum(interpolated_wake)
    print('Testing quad y')
    kicks_0 = particles_0.y * interpolated_wake_0
    norm = np.max(np.abs(kicks_0))
    kicks_0 /= norm
    kicks_1 = particles_1.y * interpolated_wake_1 / norm
    assert np.allclose(particles_0.py / norm, kicks_0)
    assert np.allclose(particles_1.py / norm, kicks_1)


@for_all_test_contexts(excluding=exclude_contexts)
def test_multibunch_pipeline_transverse_quadrupole_coupling(test_context):
    # Filling scheme
    n_slots = 100
    filling_scheme = np.array(np.floor(np.random.rand(n_slots) + 0.1),
                              dtype=int)
    filling_scheme[0] = 1
    filled_slots = np.nonzero(filling_scheme)[0]
    n_bunches = len(filled_slots)
    n_bunches_0 = int(np.floor(n_bunches / 2))
    bunch_numbers_0 = np.arange(n_bunches_0, dtype=int)
    bunch_numbers_1 = np.arange(n_bunches_0, n_bunches, dtype=int)

    print('initialising pipeline')
    comm = xt.pipeline.core.PipelineCommunicator()
    pipeline_manager = xt.PipelineManager(comm)
    pipeline_manager.add_particles(f'b0', 0)
    pipeline_manager.add_particles(f'b1', 0)
    pipeline_manager.add_element('wake')

    bunch_spacing = 25E-9 * c
    sigma_zeta = bunch_spacing / 20
    zeta_range = (-1.1 * sigma_zeta, 1.1 * sigma_zeta)
    num_slices = 1001
    dzeta = (zeta_range[1] - zeta_range[0]) / num_slices
    zeta_slice_edges = np.linspace(zeta_range[0], zeta_range[1], num_slices + 1)
    zeta_centers = zeta_slice_edges[:-1] + dzeta / 2

    zeta_0 = []
    for bunch_number in bunch_numbers_0:
        zeta_0.append(zeta_centers - filled_slots[bunch_number] * bunch_spacing)
    zeta_0 = np.hstack(zeta_0)

    print('Initialising particles')
    particles_0 = xt.Particles(p0c=7E12, zeta=zeta_0,
                               x=np.random.randn(len(zeta_0)),
                               y=np.random.randn(len(zeta_0)),
                               _context=test_context)
    particles_0.init_pipeline('b0')

    zeta_1 = []
    for bunch_number in bunch_numbers_1:
        zeta_1.append(zeta_centers - filled_slots[bunch_number] * bunch_spacing)
    zeta_1 = np.hstack(zeta_1)
    particles_1 = xt.Particles(p0c=7E12, zeta=zeta_1,
                               x=np.random.randn(len(zeta_1)),
                               y=np.random.randn(len(zeta_1)),
                               _context=test_context)
    particles_1.init_pipeline('b1')

    print('Initialising wake')
    n_turns_wake = 1
    circumference = n_slots * bunch_spacing
    wake_table_name = xf.general._pkg_root.joinpath(
        '../test_data/HLLHC_wake.dat')
    wake_file_columns = ['time', 'longitudinal', 'dipole_x', 'dipole_y',
                         'quadrupole_x', 'quadrupole_y', 'dipole_xy',
                         'quadrupole_xy', 'dipole_yx', 'quadrupole_yx',
                         'constant_x', 'constant_y']
    components = ['quadrupole_xy', 'quadrupole_yx']
    wake_df = xw.read_headtail_file(wake_table_name,
                                    wake_file_columns)
    wf_0 = xw.WakeFromTable(wake_df, columns=components)
    wf_0.configure_for_tracking(zeta_range=zeta_range,
                                num_slices=num_slices,
                                bunch_spacing_zeta=bunch_spacing,
                                filling_scheme=filling_scheme,
                                bunch_numbers=bunch_numbers_0,
                                num_turns=n_turns_wake,
                                circumference=circumference
                                )
    wf_0._wake_tracker.init_pipeline(pipeline_manager=pipeline_manager,
                                     element_name='wake',
                                     partners_names=['b1'])

    wf_1 = xw.WakeFromTable(wake_df, columns=components)
    wf_1.configure_for_tracking(zeta_range=zeta_range,
                                num_slices=num_slices,
                                bunch_spacing_zeta=bunch_spacing,
                                filling_scheme=filling_scheme,
                                bunch_numbers=bunch_numbers_1,
                                num_turns=n_turns_wake,
                                circumference=circumference
                                )
    wf_1._wake_tracker.init_pipeline(pipeline_manager=pipeline_manager,
                                     element_name='wake',
                                     partners_names=['b0'])


    print('Initialising lines')
    line_0 = xt.Line(elements=[wf_0])
    line_1 = xt.Line(elements=[wf_1])
    print('Initialising multitracker')
    line_0.build_tracker()
    line_1.build_tracker()
    multitracker = xt.PipelineMultiTracker(
        branches=[xt.PipelineBranch(line=line_0, particles=particles_0),
                  xt.PipelineBranch(line=line_1, particles=particles_1)])
    print('Tracking')
    pipeline_manager.verbose = True
    multitracker.track(num_turns=1)
    print('loading test data')
    scaling_constant = -particles_0.q0 ** 2 * e ** 2 / (particles_0.p0c[0] * e)
    wake_data_raw = np.loadtxt(wake_table_name)
    print('Computing quad wake for comparison')
    all_zetas = np.hstack([particles_0.zeta, particles_1.zeta])
    wake_component = 'quadrupole_xy'
    i_wake_comp = wake_file_columns.index(wake_component)
    wake_time = -1E-9 * np.flip(wake_data_raw[:, 0]) * c
    wake_data = -1E15 * scaling_constant * np.flip(
        wake_data_raw[:, i_wake_comp])
    interpolated_wake_0 = np.zeros_like(particles_0.x)
    for ipart in range(len(particles_0.x)):
        zeta_wake = particles_0.zeta[ipart] - all_zetas
        zeta_wake = zeta_wake[zeta_wake < 0]
        interpolated_wake = np.interp(zeta_wake, wake_time, wake_data)
        interpolated_wake_0[ipart] = np.sum(interpolated_wake)
    interpolated_wake_1 = np.zeros_like(particles_1.x)
    for ipart in range(len(particles_1.x)):
        zeta_wake = particles_1.zeta[ipart] - all_zetas
        zeta_wake = zeta_wake[zeta_wake < 0]
        interpolated_wake = np.interp(zeta_wake, wake_time, wake_data)
        interpolated_wake_1[ipart] = np.sum(interpolated_wake)
    print('Testing quad XY')
    kicks_0 = particles_0.y * interpolated_wake_0
    norm = np.max(np.abs(kicks_0))
    kicks_0 /= norm
    kicks_1 = particles_1.y * interpolated_wake_1 / norm
    assert np.allclose(particles_0.px / norm, kicks_0)
    assert np.allclose(particles_1.px / norm, kicks_1)
    wake_component = 'quadrupole_yx'
    i_wake_comp = wake_file_columns.index(wake_component)
    wake_time = -1E-9 * np.flip(wake_data_raw[:, 0]) * c
    wake_data = -1E15 * scaling_constant * np.flip(
        wake_data_raw[:, i_wake_comp])
    interpolated_wake_0 = np.zeros_like(particles_0.x)
    for ipart in range(len(particles_0.x)):
        zeta_wake = particles_0.zeta[ipart] - all_zetas
        zeta_wake = zeta_wake[zeta_wake < 0]
        interpolated_wake = np.interp(zeta_wake, wake_time, wake_data)
        interpolated_wake_0[ipart] = np.sum(interpolated_wake)
    interpolated_wake_1 = np.zeros_like(particles_1.x)
    for ipart in range(len(particles_1.x)):
        zeta_wake = particles_1.zeta[ipart] - all_zetas
        zeta_wake = zeta_wake[zeta_wake < 0]
        interpolated_wake = np.interp(zeta_wake, wake_time, wake_data)
        interpolated_wake_1[ipart] = np.sum(interpolated_wake)
    print('Testing quad YX')
    kicks_0 = particles_0.x * interpolated_wake_0
    norm = np.max(np.abs(kicks_0))
    kicks_0 /= norm
    kicks_1 = particles_1.x * interpolated_wake_1 / norm
    assert np.allclose(particles_0.py / norm, kicks_0)
    assert np.allclose(particles_1.py / norm, kicks_1)

import xtrack as xt
import xpart as xp
import xfields as xf
import xwakes as xw
import xobjects as xo
from scipy.constants import e as qe, c as c_light
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mpi4py import MPI

kind2params = {
    'longitudinal': {'plane': 'z', 'source_exponents': (0, 0), 'test_exponents': (0, 0)},
    'constant_x': {'plane': 'x', 'source_exponents': (0, 0), 'test_exponents': (0, 0)},
    'constant_y': {'plane': 'y', 'source_exponents': (0, 0), 'test_exponents': (0, 0)},
    'dipolar_x': {'plane': 'x', 'source_exponents': (1, 0), 'test_exponents': (0, 0)},
    'dipolar_xy': {'plane': 'x', 'source_exponents': (0, 1), 'test_exponents': (0, 0)},
    'dipolar_yx': {'plane': 'y', 'source_exponents': (1, 0), 'test_exponents': (0, 0)},
    'dipolar_y': {'plane': 'y', 'source_exponents': (0, 1), 'test_exponents': (0, 0)},
    'quadrupolar_x': {'plane': 'x', 'source_exponents': (0, 0), 'test_exponents': (1, 0)},
    'quadrupolar_y': {'plane': 'y', 'source_exponents': (0, 0), 'test_exponents': (0, 1)},
    'quadrupolar_xy': {'plane': 'x', 'source_exponents': (0, 0), 'test_exponents': (0, 1)},
    'quadrupolar_yx': {'plane': 'y', 'source_exponents': (0, 0), 'test_exponents': (1, 0)},
}

kind = ['longitudinal', 'dipolar_x', 'quadrupolar_y']

p0c = 1e9

# Filling scheme
n_slots = 100

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
comm_size = comm.Get_size()

if my_rank == 0:
    filling_scheme = np.array(np.floor(np.random.rand(n_slots)+0.1), dtype=int)
    comm.Send(filling_scheme, dest=1, tag=0)
elif my_rank == 1:
    filling_scheme = np.empty(n_slots, dtype=int)
    comm.Recv(filling_scheme, source=0, tag=0)

filling_scheme[0] = 1
filled_slots = np.nonzero(filling_scheme)[0]

bunch_numbers_rank = xp.split_scheme(filling_scheme=filling_scheme,
                                     n_chunk=comm_size)
print(bunch_numbers_rank)
bunch_spacing = 25E-9*c_light
sigma_zeta = bunch_spacing/20
zeta_range = (-1.1*sigma_zeta, 1.1*sigma_zeta)
num_slices = 1001
dzeta = (zeta_range[1]-zeta_range[0])/num_slices
zeta_slice_edges = np.linspace(zeta_range[0], zeta_range[1], num_slices+1)
zeta_centers = zeta_slice_edges[:-1]+dzeta/2

print('Initialising particles')
zeta = []
for bunch_number in bunch_numbers_rank[0]:
    zeta.append(zeta_centers-filled_slots[bunch_number]*bunch_spacing)
zeta = np.hstack(zeta)

particles_0 = xt.Particles(p0c=p0c, zeta=zeta)

particles_0.x += 2e-3
particles_0.y += 3e-3

zeta = []
for bunch_number in bunch_numbers_rank[1]:
    zeta.append(zeta_centers-filled_slots[bunch_number]*bunch_spacing)
zeta = np.hstack(zeta)

particles_1 = xt.Particles(p0c=p0c, zeta=zeta)

particles_1.x += 2e-3
particles_1.y += 3e-3

print('Initialising wake')
n_turns_wake = 1
circumference = n_slots * bunch_spacing

wf = xw.WakeResonator(
        kind=kind,
        r=1e8, q=1e5, f_r=1e3)
wf.configure_for_tracking(zeta_range=zeta_range,
                        num_slices=num_slices,
                        bunch_spacing_zeta=bunch_spacing,
                        filling_scheme=filling_scheme,
                        bunch_numbers=bunch_numbers_rank[my_rank],
                        num_turns=n_turns_wake,
                        circumference=circumference
                        )

dict_p_bef_0 = {}
dict_p_bef_1 = {}

for kk in kind:
    if kk == 'longitudinal':
        dict_p_bef_0[kk] = ('delta', particles_0.delta.copy())
        dict_p_bef_1[kk] = ('delta', particles_1.delta.copy())
    elif kk.split('_')[1] == 'x' or kk.split('_')[1] == 'xy':
        dict_p_bef_0[kk] = ('px', particles_0.px.copy())
        dict_p_bef_1[kk] = ('px', particles_1.px.copy())
    elif kk.split('_')[1] == 'y' or kk.split('_')[1] == 'yx':
        dict_p_bef_0[kk] = ('py', particles_0.py.copy())
        dict_p_bef_1[kk] = ('py', particles_1.py.copy())
    else:
        raise ValueError('Invalid kind')

print('Initialising lines')
line = xt.Line(elements=[wf])
line.build_tracker()

print('Initialising pipeline and multitracker')
if my_rank == 0:
    pipeline_manager = xw.PipelineManagerForWakes(particles=particles_0,
                    line=line,
                    wakes_dict={'wake': wf},
                    communicator=comm)
if my_rank == 1:
    pipeline_manager = xw.PipelineManagerForWakes(particles=particles_1,
                    line=line,
                    wakes_dict={'wake': wf},
                    communicator=comm)

if my_rank == 0:
    multitracker = xt.PipelineMultiTracker(
        branches=[xt.PipelineBranch(line=line, particles=particles_0)])
elif my_rank == 1:
    multitracker = xt.PipelineMultiTracker(
        branches=[xt.PipelineBranch(line=line, particles=particles_1)])
print('Tracking')
pipeline_manager.verbose = False
multitracker.track(num_turns=1)
print('loading test data')

parts_tot = xt.Particles.merge([particles_0, particles_1])

if my_rank == 0:
    particles = particles_0
    dict_p_bef = dict_p_bef_0
else:
    particles = particles_1
    dict_p_bef = dict_p_bef_1

assert len(wf.components) == len(kind)

for comp, kk in zip(wf.components, kind):
    print(kk)
    if comp.plane == 'z':
        scale = -particles.q0**2 * qe**2 / (
            particles.p0c[0] * particles.beta0[0]* qe) * particles.weight[0]
    else:
        scale = particles.q0**2 * qe**2 / (
            particles.p0c[0] * particles.beta0[0]* qe) * particles.weight[0]
    assert comp.plane == kind2params[kk]['plane']
    assert comp.source_exponents == kind2params[kk]['source_exponents']
    assert comp.test_exponents == kind2params[kk]['test_exponents']

    expected = np.zeros_like(particles.zeta)

    for i_test, z_test in enumerate(particles.zeta):
        expected[i_test] += (particles.x[i_test]**comp.test_exponents[0] *
                                particles.y[i_test]**comp.test_exponents[1] *
                                np.dot(parts_tot.x**comp.source_exponents[0] *
                                    parts_tot.y**comp.source_exponents[1],
                                    comp.function_vs_zeta(z_test - parts_tot.zeta,
                                                            beta0=parts_tot.beta0[0],
                                                            dzeta=1e-12)) * scale)

    xo.assert_allclose(getattr(particles, dict_p_bef[kk][0]) - dict_p_bef[kk][1],
                expected, rtol=1e-4, atol=1e-20)

    plt.figure()
    plt.plot(particles.zeta, getattr(particles, dict_p_bef[kk][0]) - dict_p_bef[kk][1], 'rx')
    plt.plot(particles.zeta, expected, 'b.')
    plt.title(f'component={kk}, rank={my_rank}, particles={particles.name}')
    plt.show()

    # we have a barrier here just to avoid mixing up the plots
    comm.Barrier()

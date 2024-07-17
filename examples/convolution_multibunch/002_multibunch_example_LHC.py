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

p0c = 7000e9

n_procs = MPI.COMM_WORLD.Get_size()
my_rank = MPI.COMM_WORLD.Get_rank()

# Filling scheme
filling_scheme = np.zeros(3564, dtype=int)
filling_scheme[0:2] = 1
bunch_numbers_rank = xp.split_scheme(filling_scheme=filling_scheme,
                                     n_chunk=int(n_procs))

n_turns_wake = 1
circumference = 26658.8832
bucket_length_m = circumference / 35640
num_slices = 100

wake_table_name = xf.general._pkg_root.joinpath(
    '../test_data/HLLHC_wake.dat')
wake_file_columns = ['time', 'longitudinal', 'dipolar_x', 'dipolar_y',
                        'quadrupolar_x', 'quadrupolar_y', 'dipolar_xy',
                        'quadrupolar_xy', 'dipolar_yx', 'quadrupolar_yx',
                        'constant_x', 'constant_y']
wake_df = xw.read_headtail_file(wake_table_name,
                                wake_file_columns)
wf= xw.WakeFromTable(wake_df, columns=['dipolar_x', 'dipolar_y'],
)
wf.configure_for_tracking(zeta_range=(-0.5*bucket_length_m, 0.5*bucket_length_m),
                          num_slices=num_slices,
                          bunch_spacing_zeta=circumference/3564,
                          filling_scheme=filling_scheme,
                          bunch_numbers=bunch_numbers_rank[my_rank],
                          num_turns=n_turns_wake,
                          circumference=circumference
                          )

one_turn_map = xt.LineSegmentMap(
    length=circumference, betx=70., bety=80.,
    qx=62.31, qy=60.32,
    longitudinal_mode='linear_fixed_qs',
    dqx=-10., dqy=-10.,  # <-- to see fast mode-0 instability
    qs=2e-3, bets=731.27
)

line = xt.Line(elements=[one_turn_map, wf],
               element_names=['one_turn_map', 'wf'])
line.particle_ref = xt.Particles(p0c=p0c)
line.build_tracker()

particles = xp.generate_matched_gaussian_multibunch_beam(
            filling_scheme=filling_scheme,
            num_particles=100_000,
            total_intensity_particles=2.3e11,
            nemitt_x=2e-6, nemitt_y=2e-6, sigma_z=0.075,
            line=line, bunch_spacing_buckets=10,
            bunch_numbers=np.array(bunch_numbers_rank[my_rank], dtype=int),
            bucket_length=bucket_length_m,
            particle_ref=line.particle_ref
)

particles.x += 1e-3
particles.y += 1e-3

pipeline_manager, multitracker = xw.config_pipeline_manager_and_multitracker_for_wakes(
    particles=particles,
    line=line,
    wakes_dict={'wake_lhc': wf},
    communicator=MPI.COMM_WORLD)

print('Tracking')
pipeline_manager.verbose = False
multitracker.track(num_turns=1)
print('loading test data')

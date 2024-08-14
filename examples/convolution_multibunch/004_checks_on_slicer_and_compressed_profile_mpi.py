# mpiexec -n 3 xterm -e ipython
# mpiexec -n 3 /Applications/iTerm.app/Contents/MacOS/iTerm2

import xfields as xf
import xtrack as xt
import numpy as np
import xobjects as xo
import xwakes as xw

num_turns = 3

wake_ref = xw.WakeResonator(kind='dipole_x', r=1e9, q=5, f_r=20e6)
wake_mpi = xw.WakeResonator(kind='dipole_x', r=1e9, q=5, f_r=20e6)

for ww in [wake_ref, wake_mpi]:
    ww.configure_for_tracking(
        zeta_range=(-1, 1),
        num_slices=10,
        filling_scheme=[1, 0, 1, 1, 1],
        bunch_spacing_zeta=5,
        num_turns=num_turns,
        circumference=100)

zeta = np.linspace(-25, 25, 1000000)
particles_ref = xt.Particles(p0c=7000e9,
                         zeta=zeta)

# different weight for the different bunches
zeta = particles_ref.zeta
mask_bunch0 = (zeta > -1) & (zeta < 1)
particles_ref.weight[mask_bunch0] = 1
mask_bunch1 = (zeta > -6) & (zeta < -4)
particles_ref.weight[mask_bunch1] = 2
mask_bunch2 = (zeta > -11) & (zeta < -9)
particles_ref.weight[mask_bunch2] = 3
mask_bunch3 = (zeta > -16) & (zeta < -14)
particles_ref.weight[mask_bunch3] = 4
mask_bunch4 = (zeta > -21) & (zeta < -19)
particles_ref.weight[mask_bunch4] = 5

particles_mpi = particles_ref.copy()

line_mpi = xt.Line(elements=[wake_mpi])
line_mpi.build_tracker()

from mpi4py import MPI
xw.config_pipeline_for_wakes(particles=particles_mpi, line=line_mpi,
                             communicator=MPI.COMM_WORLD)


assert wake_mpi._wake_tracker.pipeline_manager is not None
assert wake_ref._wake_tracker.pipeline_manager is None

comm = wake_mpi._wake_tracker.pipeline_manager._communicator
assert comm is MPI.COMM_WORLD

n_proc = comm.Get_size()
assert n_proc == 3

my_rank = comm.Get_rank()
assert my_rank in [0, 1, 2]

expected_bunch_numbers = {
    0: [0, 1],
    1: [2],
    2: [3]
}[my_rank]

expected_partner_names = {
    0: ['particles1', 'particles2'],
    1: ['particles0', 'particles2'],
    2: ['particles0', 'particles1']
}[my_rank]

slicer_mpi = wake_mpi._wake_tracker.slicer
slice_ref = wake_ref._wake_tracker.slicer

assert (slicer_mpi.bunch_numbers
        == np.array(expected_bunch_numbers)).all()
assert (slicer_mpi.num_bunches
        == len(expected_bunch_numbers))
assert (np.array(wake_mpi._wake_tracker.partners_names)
        == expected_partner_names).all()

assert (slicer_mpi.filled_slots == [0, 2, 3, 4]).all()


# line_mpi.track(particles_mpi)
# wake_ref.track(particles_ref)

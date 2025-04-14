import numpy as np
from matplotlib import pyplot as plt
from scipy import constants as cst

import xobjects as xo
import xtrack as xt
import xfields as xf
import xwakes as xw

context = xo.ContextCpu()

energy = 7E3
qx = 62.31
qy = 60.32
momentumCompaction = 3.48e-04
basic_slot_time = 25E-9
n_bunch_max = 3000
n_bunch = 100
bunch_intensity = 5E12
emit_norm = 2.5E-6
sigma_z = 0.08
selected_bunch = n_bunch-1
r_over_q = 1806
q = 5E3
coupled_bunch_number_x = 16
coupled_bunch_number_y = 73
n_turns_wake = 1
num_slices = 11

assert n_bunch_max%n_bunch == 0
circumference = n_bunch_max * basic_slot_time * cst.c
bunch_spacing_zeta = circumference/n_bunch
beta_x = circumference/(2.0*np.pi*qx)
beta_y = circumference/(2.0*np.pi*qy)
mass0 = cst.value('proton mass energy equivalent in MeV')*1E-3
gamma = energy/mass0
RF_freq = 10.0/basic_slot_time
sigma_x = np.sqrt(emit_norm*beta_x/gamma)
sigma_y = np.sqrt(emit_norm*beta_y/gamma)
betar = np.sqrt(1 - 1 / gamma ** 2)
zeta_range=(-3*sigma_z, 3*sigma_z)
filling_scheme = np.ones(n_bunch,dtype=int)

n_part = num_slices
slice_intensity = bunch_intensity/num_slices
x0 = np.ones(n_part)
z_edges = np.linspace(zeta_range[0],zeta_range[1],num_slices+1)
z0 = z_edges[:-1]+(z_edges[1]-z_edges[0])/2
zeta0 = z0-1*(n_bunch-1)*bunch_spacing_zeta
particles = xt.Particles(
        _context = context, 
        q0       = 1,
        p0c      = energy*1E9,
        mass0    = mass0*1E9,
        x        = sigma_x*x0,
        y        = sigma_y*x0,
        zeta     = zeta0,
        weight   = bunch_intensity / n_part
        )

wfx = xw.wit.ComponentResonator(kind = 'dipole_x',
                                 r=r_over_q*q, q=q, f_r=RF_freq+3E5)
wfy = xw.wit.ComponentResonator(kind = 'dipole_y',
                                 r=r_over_q*q, q=q, f_r=RF_freq-3E5)

                                     
coupled_bunch_phase_x = 2*np.pi*coupled_bunch_number_x/n_bunch
wfx.configure_for_tracking(zeta_range=zeta_range,
                    num_slices=num_slices,
                    bunch_spacing_zeta=bunch_spacing_zeta,
                    num_turns=n_turns_wake,
                    circumference=circumference,
                    filling_scheme = np.ones(n_bunch,dtype=int),
                    bunch_selection = [selected_bunch],
                    fake_coupled_bunch_phase_x = coupled_bunch_phase_x,
                    beta_x = beta_x,
                    )
coupled_bunch_phase_y = 2*np.pi*coupled_bunch_number_y/n_bunch
wfy.configure_for_tracking(zeta_range=zeta_range,
                    num_slices=num_slices,
                    bunch_spacing_zeta=bunch_spacing_zeta,
                    num_turns=n_turns_wake,
                    circumference=circumference,
                    filling_scheme = np.ones(n_bunch,dtype=int),
                    bunch_selection = [selected_bunch],
                    fake_coupled_bunch_phase_y = coupled_bunch_phase_y,
                    beta_y = beta_y,
                    )


for turn in range(n_turns_wake):
    px0 = np.copy(particles.px)
    wfx.track(particles)
    py0 = np.copy(particles.py)
    wfy.track(particles)

moments_data = wfx._xfields_wf.moments_data
z,x = moments_data.get_moment_profile('x',0)
moments_data = wfy._xfields_wf.moments_data
z,y = moments_data.get_moment_profile('y',0)
zetas = np.array([])
positions_x = np.array([])
positions_y = np.array([])
for slot in np.arange(n_bunch):
    zetas = np.hstack([zetas,z0-bunch_spacing_zeta*slot])
    positions_x = np.hstack([positions_x,x0*np.cos(coupled_bunch_phase_x*(selected_bunch-slot))])
    positions_y = np.hstack([positions_y,x0*np.cos(coupled_bunch_phase_y*(selected_bunch-slot))])
indices = np.argsort(zetas)
zetas = zetas[indices]
positions_x = positions_x[indices]
positions_y = positions_y[indices]

assert np.allclose(z,zetas)
assert np.allclose(x/sigma_x,positions_x)
assert np.allclose(y/sigma_y,positions_y)


for i_slice in range(num_slices):
    zetas_slice = zeta0[i_slice]-zetas

    scaling_constant = particles.q0**2 * cst.e**2 / (particles.p0c[0] * particles.beta0[0] * cst.e)
    kicks = positions_x*sigma_x*scaling_constant*slice_intensity*wfx.function_vs_zeta(zetas_slice,beta0=betar,dzeta=moments_data.dz)
    kick_from_track = particles.px[i_slice]-px0[i_slice]
    assert np.isclose(np.sum(kicks),kick_from_track)
    kicks = positions_x*sigma_x*scaling_constant*slice_intensity*wfy.function_vs_zeta(zetas_slice,beta0=betar,dzeta=moments_data.dz)
    kick_from_track = particles.py[i_slice]-py0[i_slice]
    assert np.isclose(np.sum(kicks),kick_from_track)



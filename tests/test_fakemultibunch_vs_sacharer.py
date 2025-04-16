import time, h5py, os
import numpy as np
from matplotlib import pyplot as plt
from scipy import constants as cst
from scipy.signal import hilbert
from scipy.stats import linregress

import xobjects as xo
import xtrack as xt
import xfields as xf
import xwakes as xw

context = xo.ContextCpu()

energy = 7E3
qx = 62.31
qy = 60.32
momentumCompaction = 3.483575072011584e-04
basic_slot_time = 25E-9
n_bunch_max = 3000
n_bunch = 100
bunch_intensity = 5E12
RF_voltage = 16.0E6
emit_norm = 2.5E-6
sigma_z = 0.08
selected_bunch = n_bunch-1

assert n_bunch_max%n_bunch == 0
circumference = n_bunch_max * basic_slot_time * cst.c
bunch_spacing_zeta = circumference/n_bunch
beta_x = circumference/(2.0*np.pi*qx)
beta_y = circumference/(2.0*np.pi*qy)
mass0 = cst.value('proton mass energy equivalent in MeV')*1E-3
gamma = energy/mass0
RF_freq = 10.0/basic_slot_time
sigma_x = np.sqrt(emit_norm*beta_x/gamma)
sigma_px = np.sqrt(emit_norm/beta_x/gamma)
sigma_y = np.sqrt(emit_norm*beta_y/gamma)
sigma_py = np.sqrt(emit_norm/beta_y/gamma)
bucket_length = cst.c/RF_freq
bucker_center = -1*selected_bunch*bunch_spacing_zeta
betar = np.sqrt(1 - 1 / gamma ** 2)
p0 = cst.m_p * betar * gamma * cst.c
harmonic_number = circumference / bucket_length
eta = 1/gamma**2-momentumCompaction
qs = np.sqrt(cst.e * RF_voltage * np.abs(eta) * harmonic_number / (2 * np.pi * betar * cst.c * p0))
averageRadius = circumference / (2.0*np.pi)
sigma_pz = qs*sigma_z/(averageRadius*np.abs(eta))
zeta_range=(-3*sigma_z, 3*sigma_z)
filling_scheme = np.ones(n_bunch,dtype=int)
omegarev= 2.0*np.pi*cst.c/circumference

n_part = int(2E4)
n_turns_wake = 10
num_slices = 21
n_turn = int(1E4)
coupled_bunch_mode_number = 90

wf = xw.wit.ComponentClassicThickWall(kind = 'dipole_x',
                                 layer=xw.wit.Layer(thickness=None,dc_resistivity=1E-8),
                                 radius=0.02, length = circumference, zero_rel_tol = 0.0)

tune_shift_nx, tune_shift_m0, effective_impedance = xw.wit.sacherer_formula.sacherer_formula(
                    qp = 0.0,
                    nx_array = np.array([coupled_bunch_mode_number]),
                    bunch_intensity = bunch_intensity,
                    omegas = qs*omegarev,
                    n_bunches = n_bunch,
                    omega_rev = omegarev,
                    tune=qx,
                    gamma=gamma,
                    eta = eta,
                    bunch_length_seconds = sigma_z*4/cst.c,
                    m_max = 0,
                    impedance_function = wf.impedance)

particles = xt.Particles(
        _context = context, 
        q0       = 1,
        p0c      = energy*1E9,
        mass0    = mass0*1E9,
        x        = sigma_x*(np.random.randn(n_part)+2),
        px       = sigma_px*(np.random.randn(n_part)),
        y        = sigma_y*np.random.randn(n_part),
        py       = sigma_py*np.random.randn(n_part),
        zeta     = sigma_z*np.random.randn(n_part) + bucker_center,
        delta    = sigma_pz*np.random.randn(n_part),
        weight   = bunch_intensity / n_part
        )

arc = xt.LineSegmentMap(length=None, qx=qx, qy=qy,
            betx=beta_x, bety=beta_y,
            longitudinal_mode='linear_fixed_rf',
            voltage_rf = RF_voltage,
            frequency_rf = RF_freq,
            lag_rf = 180.0,
            slippage_length = circumference,
            momentum_compaction_factor = momentumCompaction)

wf.configure_for_tracking(zeta_range=zeta_range,
                        num_slices=num_slices,
                        bunch_spacing_zeta=bunch_spacing_zeta,
                        num_turns=n_turns_wake,
                        circumference=circumference,
                        filling_scheme = filling_scheme,
                        bunch_selection = [selected_bunch],
                        fake_coupled_bunch_phase_x = 2.0*np.pi*coupled_bunch_mode_number/n_bunch,
                        beta_x = beta_x,
                        )

longitudinal_acceptance = xt.LongitudinalLimitRect(min_zeta=bucker_center-bucket_length, max_zeta=bucker_center+bucket_length)

log = xt.Log(mean_x=lambda l,p:p.x.mean())
            
line = xt.Line([arc,longitudinal_acceptance,wf])
line.build_tracker(_context=context)
line.track(particles,num_turns=n_turn,log=log)

ampl = np.abs(hilbert(line.log_last_track['mean_x']))
turns = np.arange(len(ampl))
start_fit = 100
end_fit = len(ampl)-100
fit = linregress(turns[start_fit:end_fit],np.log(ampl[start_fit:end_fit]))

assert np.isclose(fit.slope,-2.0*np.pi*np.imag(tune_shift_nx[0][0]),rtol=5E-2)



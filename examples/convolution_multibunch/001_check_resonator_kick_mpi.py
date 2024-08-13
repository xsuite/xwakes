import xwakes as xw
import xtrack as xt
import xpart as xp

bunch_spacing_buckets = 2
bucket_length_m = 0.1 #What happens if not multiple?
sigma_zeta = 0.03
circumference = 1
filling_scheme=[1, 1, 0, 1]

kind = ['longitudinal', 'dipolar_x', 'quadrupolar_y']
wf = xw.WakeResonator(
        kind=kind,
        r=1e8, q=1e5, f_r=1e3)

wf.configure_for_tracking(
    zeta_range=(-3*sigma_zeta, 3*sigma_zeta),
    num_slices=30,
    bunch_spacing_zeta=bunch_spacing_buckets*bucket_length_m,
    filling_scheme=filling_scheme,
    num_turns=5,
    circumference=circumference
    )

elements = dict(
    wf=wf,
    matrix=xt.LineSegmentMap(
        length=circumference, betx=70., bety=80.,
        qx=1.1, qy=0.2, qs=0.002, bets=300.,
    )
)

line = xt.Line(elements=elements, element_names=list(elements.keys()))

line.particle_ref = xt.Particles(p0c=7000e9)
line.build_tracker()

particles = xp.generate_matched_gaussian_multibunch_beam(
            line=line,
            filling_scheme=filling_scheme,
            num_particles=1_000,
            total_intensity_particles=2.3e11, # This needs to be renamed
            nemitt_x=2e-6, nemitt_y=2e-6, sigma_z=sigma_zeta,
            bunch_spacing_buckets=bunch_spacing_buckets,
            bucket_length=bucket_length_m,
            particle_ref=line.particle_ref,
            #prepare_line_and_particles_for_mpi_wake_sim=True
)

line.discard_tracker()
line.element_names.remove('matrix')
line.build_tracker()

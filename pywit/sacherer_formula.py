import numpy as np
import sys

from scipy.constants import e as q_p, m_p, c

from typing import List, Callable, Iterable, Tuple, Union


def sacherer_formula(qp: float, nx_array: np.array, bunch_intensity: float, omegas: float, n_bunches: int,
                     omega_rev: float, tune: float, gamma: float, eta: float, bunch_length_seconds: float, m_max: int,
                     impedance_table: np.array = None, freq_impedance_table: np.array = None,
                     impedance_function: Callable[[float], float] = None, m0: float = m_p, charge: float = q_p,
                     mode_type: str = 'sinusoidal') -> Tuple[np.array, float, np.array]:

    r"""
    Computes frequency shift and effective impedance from Sacherer formula, in transverse, in the case of low
    intensity perturbations (no mode coupling), for modes of kind 'mode_type'.

    Documentation: see Elias Metral's USPAS 2009 course : Bunched beams transverse coherent
    instabilities.

    NOTE: this is NOT the original Sacherer formula, which assumes an impedance normalized by beta
    (see E. Metral, USPAS 2009 lectures, or C. Zannini,
    https://indico.cern.ch/event/766028/contributions/3179810/attachments/1737652/2811046/Z_definition.pptx)
    Here this formula is instead divided by beta (compared to Sacherer initial one),
    so is valid with our usual definition of impedance (not beta-normalized).

    :param qp: the chromaticity (defined as $\frac{\Delta q \cdot p}{\Delta p}$
    :param nx_array: a vector of coupled bunch modes for which the tune shift is computed (it must contain integers in
    the range (0, M-1))
    :param bunch_intensity: number of particles per bunch
    :param omegas: the synchrotron angular frequency (i.e. $Q_s \cdot \omega_{rev}$)
    :param n_bunches: the number of bunches
    :param omega_rev: the revolution angular frequency (i.e. $2\cdot \pi f_{rev}$)
    :param tune: machine tune in the considered plane (the TOTAL tune, including the integer part, must be passed)
    :param gamma: the relativistic gamma factor of the beam
    :param eta: the slippage factor (i.e. alpha_p - 1/gamma^2, with alpha_p the momentum compaction factor)
    :param bunch_length_seconds: the total bunch length in seconds (4 times $\sigma$ for a Gaussian bunch)
    :param m_max: specifies the range (-m_max to m_max) of the azimuthal modes to be considered
    :param impedance_table: a numpy array giving the complex impedance at a discrete set of points. It must be specified
    if impedance_function is not specified
    :param freq_impedance_table: the frequencies at which the impedance is sampled. It must be specified if
    impedance_function is not specified
    :param impedance_function: the impedance function. It must be specified if impedance_table is not specified
    :param m0: the rest mass of the considered particles
    :param charge: the charge of the considered particles
    :param mode_type: the type of modes in the effective impedance. It can be 'sinusoidal' (typically
    well-adpated for protons) or 'hermite' (typically better for leptons).

    :return tune_shift_nx: tune shifts for all multibunch modes and synchrotron modes. It is an array of dimensions
    ( len(nx_scan), (2*m_max+1) )
    :return tune_shift_m0: tune shift of the most unstable coupled-bunch mode with m=0
    :return effective_impedance: the effective impedance for all multibunch modes and synchrotron modes. It is an array
    of dimensions ( len(nx_scan), (2*m_max+1) )
    """

    def hmm(m_mode: int, omega: Union[float, np.ndarray]):
        """
        Compute hmm power spectrum of Sacherer formula, for azimuthal mode number m,
        at angular frequency 'omega' (rad/s) (can be an array), for total bunch length
        'bunch_length_seconds' (s), and for a kind of mode specified by 'mode_type'
        (which can be 'hermite' or 'sinusoidal')
        :param m_mode: the azimuthal mode number
        :param omega: the angular frequency at which hmm is computed
        """

        if mode_type.lower().startswith('sinus'):
            # best for protons
            hmm_val = (((bunch_length_seconds * (np.abs(m_mode) + 1.)) ** 2 / (2. * np.pi ** 4)) *
                       (1. + (-1) ** m_mode * np.cos(omega * bunch_length_seconds)) /
                       (((omega * bunch_length_seconds / np.pi) ** 2 - (np.abs(m_mode) + 1.) ** 2) ** 2))

        elif mode_type.lower() == 'hermite':
            # best for leptons
            hmm_val = (omega * bunch_length_seconds / 4) ** (2 * m_mode) * np.exp(
                -(omega * bunch_length_seconds / 4.) ** 2)

        else:
            raise ValueError("mode_type can only be 'sinusoidal' or 'hermite'")

        return hmm_val

    def hmm_weighted_sum(m_mode: int, nx_mode: int, weight_function: Callable[[float], complex] = None):
        """
        Compute sum of hmm functions in the Sacherer formula, optionally
        weighted by weight_function.
        Note: In the end the sum runs over k with hmm taken at the angular frequencies
        (k_offset+k*n_bunches)*omega0+m*omegas-omegaksi but the impedance is taken at
        (k_offset+k*n_bunches)*omega0+m*omegas
        :param m_mode: the azimuthal mode number
        :param nx_mode: the coupled-bunch mode number
        :param weight_function: function of frequency (NOT angular) giving
        the sum weights (typically, it is the impedance) (optional)
        :return: the (possibly weigthed) sum of hmm functions
        """
        eps = 1.e-5  # relative precision of the summations
        k_max = 20
        k_offset = nx_mode + fractional_tune
        # sum initialization
        omega_k = k_offset * omega_rev + m_mode * omegas
        hmm_k = hmm(m_mode, omega_k - omega_ksi)

        omega = np.arange(-100.01 / bunch_length_seconds, 100.01 / bunch_length_seconds,
                          0.01 / bunch_length_seconds)

        if weight_function is not None:
            z_pk = weight_function(omega_k / (2 * np.pi))
        else:
            z_pk = np.ones_like(omega_k)

        sum1_inner = z_pk * hmm_k

        k = np.arange(1, k_max + 1)
        old_sum1 = 10. * sum1_inner

        while ((np.abs(np.real(sum1_inner - old_sum1))) > eps * np.abs(np.real(sum1_inner))) or (
                (np.abs(np.imag(sum1_inner - old_sum1))) > eps * np.abs(np.imag(sum1_inner))):
            old_sum1 = sum1_inner
            # omega_k^x and omega_-k^x in Elias's slides:
            omega_k = (k_offset + k * n_bunches) * omega_rev + m_mode * omegas
            omega_mk = (k_offset - k * n_bunches) * omega_rev + m_mode * omegas
            # power spectrum function h(m,m) for k and -k:
            hmm_k = hmm(m_mode, omega_k - omega_ksi)
            hmm_mk = hmm(m_mode, omega_mk - omega_ksi)

            if weight_function is not None:
                z_pk = weight_function(omega_k / (2 * np.pi))
                z_pmk = weight_function(omega_mk / (2 * np.pi))
            else:
                z_pk = np.ones_like(omega_k)
                z_pmk = np.ones_like(omega_mk)

            # sum
            sum1_inner = sum1_inner + np.sum(z_pk * hmm_k) + np.sum(z_pmk * hmm_mk)

            k = k + k_max

        sum1_inner = np.squeeze(sum1_inner) # return a scalar if only one element

        return sum1_inner

    if impedance_function is not None and impedance_table is not None:
        raise ValueError('Only one between impedance_function and impedance_table can be specified')

    if impedance_table is not None and freq_impedance_table is None:
        raise ValueError('When impedance_table is specified, also the corresponding frequencies must be specified in'
                         'omega_impedance_table')

    # some parameters
    beta = np.sqrt(1. - 1. / (gamma ** 2))  # relativistic velocity factor
    f0 = omega_rev / (2. * np.pi)  # revolution angular frequency
    single_bunch_current = charge * bunch_intensity * f0  # single-bunch current
    fractional_tune = tune - np.floor(tune)  # fractional part of the tune
    bunch_length_seconds_meters = bunch_length_seconds * beta * c  # full bunch length (in meters)

    if impedance_table is not None:
        def impedance_function(x):
            if np.isscalar(x):
                x = np.array([x])
            ind_p = x >= 0
            ind_n = x < 0
            result = np.zeros_like(x, dtype=complex)
            result[ind_p] = np.interp(x[ind_p], freq_impedance_table, impedance_table)
            result[ind_n] = -np.interp(np.abs(x[ind_n]), freq_impedance_table, impedance_table).conjugate()

            return result

    tune_shift_nx = np.zeros((len(nx_array), 2 * m_max + 1), dtype=complex)
    tune_shift_m0 = complex(0)
    effective_impedance = np.zeros((len(nx_array), 2 * m_max + 1), dtype=complex)

    omega_ksi = qp * omega_rev / eta

    for inx, nx in enumerate(nx_array):  # coupled-bunch modes

        for im, m in enumerate(range(-m_max, m_max + 1)):
            # consider each synchrotron mode individually
            # sum power spectrum functions and computes effective impedance

            # sum power functions
            # BE CAREFUL: maybe for this "normalization sum" the sum should run
            # on all single-bunch harmonics instead of only coupled-bunch
            # harmonics (and then the frequency shift should be multiplied by
            # n_bunches). This has to be checked.
            sum1 = hmm_weighted_sum(m, nx)

            # effective impedance
            sum2 = hmm_weighted_sum(m, nx, weight_function=impedance_function)

            effective_impedance[inx, im] = sum2 / sum1
            freq_shift = 1j * charge * single_bunch_current / (2 * (np.abs(m) + 1.) * m0 * gamma * tune * omega_rev *
                                                               bunch_length_seconds_meters) * sum2 / sum1

            tune_shift = (freq_shift / omega_rev + m * omegas / omega_rev)

            tune_shift_nx[inx, im] = tune_shift

    # find the most unstable coupled-bunch mode for m=0
    inx = np.argmin(np.imag(tune_shift_nx[:, m_max]))
    tune_shift_m0 = tune_shift_nx[inx, m_max]

    return tune_shift_nx, tune_shift_m0, effective_impedance

import numpy as np
import sys

from scipy.constants import e as q_p, m_p, c

from typing import List, Callable, Iterable, Tuple


def hmm(m: int, omega: float, bunch_length: float, mode_type: str = 'sinusoidal'):
    """
    compute hmm power spectrum of Sacherer formula, for azimuthal mode number m,
    at angular frequency 'omega' (rad/s) (can be an arrray), for total bunch length
    'taub' (s), and for a kind of mode specified by 'mode_type'
    (which can be 'Hermite' - leptons -  or 'sinusoidal' - protons).
    """

    if mode_type.lower().startswith('sinus'):
        # best for protons
        hmm_val = (((bunch_length * (np.abs(m) + 1.)) ** 2 / (2. * np.pi ** 4)) *
                   (1. + (-1) ** m * np.cos(omega * bunch_length)) /
                   (((omega * bunch_length / np.pi) ** 2 - (np.abs(m) + 1.) ** 2) ** 2))

    elif mode_type.lower() == 'hermite':
        # best for leptons
        hmm_val = (omega * bunch_length / 4) ** (2 * m) * np.exp(-(omega * bunch_length / 4.) ** 2)

    else:
        print("Pb in hmm: : kind of mode not recognized!")
        sys.exit()

    return hmm_val


def hmm_sum(m: int, omega0: float, n_bunches: int, k_offset: int, bunch_length: float, omega_ksi: float,
            eps: float = 1e-5, omegas: float = 0, k_max: int = 20, mode_type: str = 'sinusoidal',
            impedance_function: Callable = None, impedance_table: List[float] = None,
            omega_impedance_table: Iterable[float] = None, flag_trapz: bool = False):
    """
    compute sum of hmm functions (defined above), weighted or not by the impedance Z
    (table of complex impedances in Ohm given from negative to positive angular frequencies
    omegaZ in rad/s] If these are None, then only sum the hmm.
    Use the trapz integration method if flag_trapz==True.

     - m: azimuthal mode number,
     - omega0: angular revolution frequency in rad/s,
     - n_bunches: number of bunches,
     - offk: offset in k (typically nx+[tune] where nx is the coupled-bunch mode and [tune]
     the fractional part of the tune),
     - taub: total bunch length in s,
     - omegaksi: chromatic angular frequency,
     - eps: relative precision of the sum,
     - omegas: synchrotron frequency,
     - kmax: step in k between sums,
     - mode_type: kind of mode for the hmm power spectrum ('sinusoidal', 'Hermite').

    In the end the sum runs over k with hmm taken at the angular frequencies
    (offk+k*n_bunches)*omega0+m*omegas-omegaksi
    but the impedance is taken at (offk+k*n_bunches)*omega0+m*omegas
    """
    if impedance_function is not None and impedance_table is not None:
        raise ValueError('Only one between impedance_function and impedance_table can be specified')

    if impedance_table is not None and omega_impedance_table is None:
        raise ValueError('When impedance_table is specified, also the corresponding frequencies must be specified in'
                         'omega_impedance_table')

    # omega shouldn't be needed
    if omega_impedance_table is None:
        omega = np.arange(-100.01/bunch_length, 100.01/bunch_length, 0.01/bunch_length)
    else:
        omega = omega_impedance_table

    # sum initialization
    omega_k = k_offset * omega0 + m * omegas
    hmm_k = hmm(m, omega_k - omega_ksi, bunch_length, mode_type=mode_type)

    if flag_trapz:
        # initialization of correcting term sum_i (with an integral instead of discrete sum)
        ind_i = np.where(np.sign(omega - omega_k - n_bunches * omega0) * np.sign(omega - 1e15) == -1)
        ind_mi = np.where(np.sign(omega - omega_k + n_bunches * omega0) * np.sign(omega + 1e15) == -1)
        omega_i = omega[ind_i]
        omega_mi = omega[ind_mi]
        hmm_i = hmm(m, omega_i - omega_ksi, bunch_length, mode_type=mode_type)
        hmm_mi = hmm(m, omega_mi - omega_ksi, bunch_length, mode_type=mode_type)
        if impedance_function is not None:
            z_i = impedance_function(omega_i/(2*np.pi))
            z_mi = impedance_function(omega_i/(2*np.pi))
        elif impedance_table is not None:
            z_i = impedance_table[ind_i]
            z_mi = impedance_table[ind_mi]
        else:
            z_i = np.ones_like(ind_i)
            z_mi = np.ones_like(ind_mi)

        sum_i = (np.trapz(z_i * hmm_i, omega_i) + np.trapz(z_mi * hmm_mi, omega_mi)) / (n_bunches * omega0)
    else:
        sum_i = 0.

    if impedance_function is not None:
        z_pk = impedance_function(omega_k/(2*np.pi))
    elif impedance_table is not None:
        z_pk = (np.interp(omega_k, omega, np.real(impedance_table)) +
                1j * np.interp(omega_k, omega, np.imag(impedance_table)))
    else:
        z_pk = np.ones_like(omega_k)

    sum1 = z_pk * hmm_k + sum_i

    k = np.arange(1, k_max + 1)
    old_sum1 = 10. * sum1

    while ((np.abs(np.real(sum1 - old_sum1))) > eps * np.abs(np.real(sum1))) or (
            (np.abs(np.imag(sum1 - old_sum1))) > eps * np.abs(np.imag(sum1))):
        old_sum1 = sum1
        # omega_k^x and omega_-k^x in Elias's slides:
        omega_k = (k_offset + k * n_bunches) * omega0 + m * omegas
        omega_mk = (k_offset - k * n_bunches) * omega0 + m * omegas
        # power spectrum function h(m,m) for k and -k:
        hmm_k = hmm(m, omega_k - omega_ksi, bunch_length, mode_type=mode_type)
        hmm_mk = hmm(m, omega_mk - omega_ksi, bunch_length, mode_type=mode_type)

        if flag_trapz:
            # subtract correction (rest of the sum considered as integral -> should suppress redundant terms)
            ind_i = np.where(np.sign(omega - omega_k[0]) * np.sign(omega - omega_k[-1] - n_bunches * omega0) == -1)
            ind_mi = np.where(np.sign(omega - omega_mk[0]) * np.sign(omega - omega_mk[-1] + n_bunches * omega0) == -1)
            omega_i = omega[ind_i]
            omega_mi = omega[ind_mi]
            hmm_i = hmm(m, omega_i - omega_ksi, bunch_length, mode_type=mode_type)
            hmm_mi = hmm(m, omega_mi - omega_ksi, bunch_length, mode_type=mode_type)

            if impedance_function is not None:
                z_i = impedance_function(omega_i/(2*np.pi))
                z_mi = impedance_function(omega_mi/(2*np.pi))
            elif impedance_table is not None:
                z_i = impedance_table[ind_i]
                z_mi = impedance_table[ind_mi]
            else:
                z_i = np.ones_like(ind_i)
                z_mi = np.ones_like(ind_mi)

            sum_i = (np.trapz(z_i * hmm_i, omega_i) + np.trapz(z_mi * hmm_mi, omega_mi)) / (n_bunches * omega0)

        else:
            sum_i = 0.

        if impedance_function is not None:
            z_pk = impedance_function(omega_k/(2*np.pi))
            z_pmk = impedance_function(omega_mk/(2*np.pi))
        elif impedance_table is not None:
            # impedances at omega_k and omega_mk
            z_pk = (np.interp(omega_k, omega, np.real(impedance_table)) +
                    1j * np.interp(omega_k, omega, np.imag(impedance_table)))
            z_pmk = (np.interp(omega_mk, omega, np.real(impedance_table)) +
                     1j * np.interp(omega_mk, omega, np.imag(impedance_table)))
        else:
            z_pk = np.ones_like(omega_k)
            z_pmk = np.ones_like(omega_mk)

        # sum
        sum1 = sum1 + np.sum(z_pk * hmm_k) + np.sum(z_pmk * hmm_mk) - sum_i

        k = k + k_max

    return sum1


def sacherer_formula(qp: float, nx_array: np.array, bunch_intensity: float, omegas: float, n_bunches: int,
                     omega_rev: float, tune: float, gamma: float, eta: float, bunch_length_seconds: float, m_max: int,
                     impedance_table: List[float] = None, freq_impedance_table: List[float] = None,
                     impedance_function: callable(float) = None, m0: float = m_p, charge: float = q_p,
                     mode_type: str = 'sinusoidal', flag_trapz: bool = None) -> Tuple[np.array, float, np.array]:
    """
    Computes frequency shift and effective impedance from Sacherer formula, in transverse, in the case of low
    intensity perturbations (no mode coupling), for modes of kind 'mode_type'.
    :param qp: the chromaticity (defined as $\frac{\Delta q \cdot p}{\Delta p}$
    :param nx_array: a vector of coupled bunch modes for which the tune shift is computed (it must contain integers in
    the range (0, M-1))

    Documentation: see Elias Metral's USPAS 2009 course : Bunched beams transverse coherent
    instabilities.

    NOTE: this is NOT the original Sacherer formula, which assumes an impedance normalized by beta
    (see E. Metral, USPAS 2009 lectures, or C. Zannini,
    https://indico.cern.ch/event/766028/contributions/3179810/attachments/1737652/2811046/Z_definition.pptx)
    Here this formula is instead divided by beta (compared to Sacherer initial one),
    so is valid with our usual definition of impedance (not beta-normalized).
    This was corrected on April 15th, 2019. NM

    :param bunch_intensity: number of particles per bunch
    :param omegas: the synchrotron frequency (i.e. $Q_s \cdot \omega_{rev}$)
    :param n_bunches: the number of bunches
    :param omega_rev: the revolution angular frequency (i.e. $2\cdot \pi f_{rev}$)
    :param tune: machine tune in the considered plane (the total tune comprehensive of the integer part must be passed)
    :param gamma: the relativistic gamma factor of the beam
    :param eta: the slip factor
    :param bunch_length_seconds: the bunch length in seconds
    :param m_max: specifies the the range (-m_max to m_max) of the azymuthal modes to be considered
    :param impedance_table: a numpy array giving the complex impedance at a discrete set of points. It must be specified
    if impedance_function is not specified
    :param freq_impedance_table: the frequencies at which the impedance is sampled. It must be specified if
    impedance_function is not specified
    :param impedance_function: the impedance function. It must be specified if impedance_table is not specified
    :param m0: the rest mass of the considered particles
    :param charge: the charge of the considered particles
    :param mode_type: the tipe of modes in the effective impedance. It can be 'sinusoidal' or 'hermite'

    :return tune_shift_nx: tune shifts for all multibunch modes and synchrotron modes. It is and array of dimensions
    len(nx_scan)*(2*m_max+1)
    :return tune_shift_m0: tune shifts of the most unstable mode with m=0
    :return effective_impedance: the effective impedance for all multibunch modes and synchrotron modes. It is and array
    of dimensions len(nx_scan)*(2*m_max+1)
    """
    # some parameters
    beta = np.sqrt(1. - 1. / (gamma ** 2))  # relativistic velocity factor
    f0 = omega_rev / (2. * np.pi)  # revolution angular frequency
    single_bunch_current = charge * bunch_intensity * f0  # single-bunch current
    fractional_tune = tune - np.floor(tune)  # fractional part of the tune
    bunch_length_seconds_meters = bunch_length_seconds * beta * c  # full bunch length (in meters)

    if freq_impedance_table is not None:
        omega_impedance_table_p = 2*np.pi*freq_impedance_table
        omega_impedance_table_m = -omega_impedance_table_p[::-1]
        # compute complex impedance and 'symmetrize' it for negative frequencies
        impedance_table_p = np.copy(impedance_table)
        impedance_table_m = -impedance_table[::-1].conjugate()
        omega_impedance_table = np.concatenate((omega_impedance_table_m, omega_impedance_table_p))
        impedance_table = np.concatenate((impedance_table_m, impedance_table_p))
    else:
        omega_impedance_table = None

    eps = 1.e-5  # relative precision of the summations
    tune_shift_nx = np.zeros((len(nx_array), 2 * m_max + 1), dtype=complex)
    tune_shift_m0 = complex(0)
    effective_impedance = np.zeros((len(nx_array), 2 * m_max + 1), dtype=complex)

    omega_ksi = qp * omega_rev / eta
    if flag_trapz is None:
        flag_trapz = np.ceil(100 * (4*np.pi / bunch_length_seconds + abs(omega_ksi)) / (omega_rev * n_bunches)) > 1e9

    for inx, nx in enumerate(nx_array):  # coupled-bunch modes

        for im, m in enumerate(range(-m_max, m_max + 1)):
            # consider each synchrotron mode individually
            # sum power spectrum functions and computes effective impedance

            # sum power functions
            # BE CAREFUL: maybe for this "normalization sum" the sum should run
            # on all single-bunch harmonics instead of only coupled-bunch
            # harmonics (and then the frequency shift should be multiplied by
            # n_bunches). This has to be checked.
            sum1 = hmm_sum(m, omega_rev, n_bunches, nx + fractional_tune, bunch_length_seconds, omega_ksi,
                           impedance_function=None, impedance_table=None,
                           omega_impedance_table=None, eps=eps, omegas=omegas, k_max=20,
                           mode_type=mode_type, flag_trapz=flag_trapz)

            # effective impedance
            sum2 = hmm_sum(m, omega_rev, n_bunches, nx + fractional_tune, bunch_length_seconds, omega_ksi, eps=eps,
                           omegas=omegas, k_max=20, mode_type=mode_type, impedance_function=impedance_function,
                           impedance_table=impedance_table, omega_impedance_table=omega_impedance_table,
                           flag_trapz=flag_trapz)

            effective_impedance[inx, im] = sum2 / sum1
            # 15/04/2019 NM: beta suppressed (was for a "beta-normalized" definition of impedance)
            freq_shift = 1j*charge*single_bunch_current/(2 * (np.abs(m) + 1.) * m0 * gamma * tune * omega_rev *
                                                         bunch_length_seconds_meters) * sum2 / sum1

            tune_shift = (freq_shift / omega_rev + m * omegas / omega_rev)

            tune_shift_nx[inx, im] = tune_shift

    # find the most unstable coupled-bunch mode for m=0
    inx = np.argmin(np.imag(tune_shift_nx[:, m_max]))
    tune_shift_m0 = tune_shift_nx[inx, m_max]

    return tune_shift_nx, tune_shift_m0, effective_impedance

from pywit.utilities import string_to_params

import numpy as np
import sys

from scipy.constants import e, m_p, epsilon_0, mu_0, c
import scipy.constants

from typing import Union, Callable


def hmm(m: int, omega: float, bunch_length: float, mode_type: str = 'sinusoidal'):
    """
    compute hmm power spectrum of Sacherer formula, for azimuthal mode number m,
    at angular frequency 'omega' (rad/s) (can be an arrray), for total bunch length
    'taub' (s), and for a kind of mode specified by 'modetype'
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


def hmmsum(m: int, omega0: float, n_bunches: int, k_offset: int, bunch_length: float, omega_ksi: float,
           eps: float = 1e-5, omegas: float = 0, k_max: int = 20, mode_type: str = 'sinusoidal',
           impedance: Union[Callable[[float], complex], np.array] = None, omega_impedance_table: np.array = None,
           flag_trapz: bool = False):
    """
    compute sum of hmm functions (defined above), weighted or not by the impedance Z
    (table of complex impedances in Ohm given from negative to positive angular frequencies
    omegaZ in rad/s] If these are None, then only sum the hmm.
    Use the trapz integration method if flagtrapz==True.

     - m: azimuthal mode number,
     - omega0: angular revolution frequency in rad/s,
     - M: number of bunches,
     - offk: offset in k (typically nx+[Q] where nx is the coupled-bunch mode and [Q]
     the fractional part of the tune),
     - taub: total bunch length in s,
     - omegaksi: chromatic angular frequency,
     - eps: relative precision of the sum,
     - omegas: synchrotron frequency,
     - kmax: step in k between sums,
     - modetype: kind of mode for the hmm power spectrum ('sinusoidal', 'Hermite').

    In the end the sum runs over k with hmm taken at the angular frequencies
    (offk+k*M)*omega0+m*omegas-omegaksi
    but the impedance is taken at (offk+k*M)*omega0+m*omegas
    """
    if impedance is not None:
        func_input = np.shape(impedance) == 0

    # omega shouldn't be needed
    # if (np.any(omegaZ)==None):
    #    omega=np.arange(-100.01/taub,100.01/taub,0.01/taub)
    #    #pylab.plot(omega,hmm(m,omega,taub,modetype=modetype))
    #    #pylab.show();
    #    #pylab.loglog(omega,hmm(m,omega,taub,modetype=modetype))
    #    #pylab.show()
    # else:
    #    omega=omegaZ

    # sum initialization
    omegak = k_offset * omega0 + m * omegas
    # omegak=Qfrac*omega0+m*omegas
    hmm_k = hmm(m, omegak - omega_ksi, bunch_length, mode_type=mode_type)

    if flag_trapz:
        omega = np.arange(-100.01 / bunch_length, 100.01 / bunch_length, 0.01 / bunch_length)
        # initialization of correcting term sum_i (with an integral instead of discrete sum)
        ind_i = np.where(np.sign(omega - omegak - n_bunches * omega0) * np.sign(omega - 1e15) == -1)
        ind_mi = np.where(np.sign(omega - omegak + n_bunches * omega0) * np.sign(omega + 1e15) == -1)
        omega_i = omega[ind_i]
        omega_mi = omega[ind_mi]
        hmm_i = hmm(m, omega_i - omega_ksi, bunch_length, mode_type=mode_type)
        hmm_mi = hmm(m, omega_mi - omega_ksi, bunch_length, mode_type=mode_type)
        if impedance is not None:
            Z_i = impedance[ind_i]
            Z_mi = impedance[ind_mi]
            sum_i = (np.trapz(Z_i * hmm_i, omega_i) + np.trapz(Z_mi * hmm_mi, omega_mi)) / (n_bunches * omega0)
        else:
            sum_i = (np.trapz(hmm_i, omega_i) + np.trapz(hmm_mi, omega_mi)) / (n_bunches * omega0)
    else:
        sum_i = 0.

    if impedance is not None:
        if np.shape(impedance) == 0:
            Zpk = impedance(omegak)
        else:
            Zpk = np.interp(omegak, omegaZ, np.real(impedance)) + 1j * np.interp(omegak, omegaZ, np.imag(impedance))

        sum1 = Zpk * hmm_k + sum_i
    else:
        sum1 = hmm_k + sum_i

    k = np.arange(1, k_max + 1)
    oldsum1 = 10. * sum1

    while ((np.abs(np.real(sum1 - oldsum1))) > eps * np.abs(np.real(sum1))) or (
            (np.abs(np.imag(sum1 - oldsum1))) > eps * np.abs(np.imag(sum1))):
        oldsum1 = sum1
        # omega_k^x and omega_-k^x in Elias's slides:
        omegak = (k_offset + k * n_bunches) * omega0 + m * omegas
        omegamk = (k_offset - k * n_bunches) * omega0 + m * omegas
        # power spectrum function h(m,m) for k and -k:
        hmm_k = hmm(m, omegak - omega_ksi, bunch_length, mode_type=mode_type)
        hmm_mk = hmm(m, omegamk - omega_ksi, bunch_length, mode_type=mode_type)

        if flag_trapz:
            # subtract correction (rest of the sum considered as integral -> should suppress redundant terms)
            ind_i = np.where(np.sign(omega - omegak[0]) * np.sign(omega - omegak[-1] - n_bunches * omega0) == -1)
            ind_mi = np.where(np.sign(omega - omegamk[0]) * np.sign(omega - omegamk[-1] + n_bunches * omega0) == -1)
            omega_i = omega[ind_i]
            omega_mi = omega[ind_mi]
            hmm_i = hmm(m, omega_i - omega_ksi, bunch_length, mode_type=mode_type)
            hmm_mi = hmm(m, omega_mi - omega_ksi, bunch_length, mode_type=mode_type)
            if impedance is not None:
                Z_i = impedance[ind_i]
                Z_mi = impedance[ind_mi]
                sum_i = (np.trapz(Z_i * hmm_i, omega_i) + np.trapz(Z_mi * hmm_mi, omega_mi)) / (n_bunches * omega0)
            else:
                sum_i = (np.trapz(hmm_i, omega_i) + np.trapz(hmm_mi, omega_mi)) / (n_bunches * omega0)
        else:
            sum_i = 0.

        if np.any(impedance) is not None:
            if func_input:
                Zpk = impedance(omegak)
                Zpmk = impedance(omegamk)
            else:
                # impedances at omegak and omegamk
                Zpk = np.interp(omegak, omega_impedance_table, np.real(impedance)) + 1j * np.interp(omegak, omega_impedance_table, np.imag(impedance))
                Zpmk = np.interp(omegamk, omega_impedance_table, np.real(impedance)) + 1j * np.interp(omegamk, omega_impedance_table, np.imag(impedance))
            # sum
            sum1 = sum1 + np.sum(Zpk * hmm_k) + np.sum(Zpmk * hmm_mk) - sum_i
        else:
            # sum
            sum1 = sum1 + np.sum(hmm_k) + np.sum(hmm_mk) - sum_i
        k = k + k_max

    # print k[-1],kmax,omegak[-1],omegaksi,m*omegas

    return sum1


def sacherer(imp_mod, Qpscan, nxscan, Nbscan, omegasscan, M, omega0, Q, gamma, eta, taub, mmax,
             particle='proton', modetype='sinusoidal', compname='x1000', flagtrapz=None):
    """
    omputes frequency shift and effective impedance from Sacherer formula, in transverse, in the case of low
    intensity perturbations (no mode coupling), for modes of kind 'modetype'.
    It gives in output:
     - tuneshift_most: tune shifts for the most unstable multibunch mode and synchrotron modes
    sorted by ascending imaginary parts (most unstable synchrotron mode first).
    Array of dimensions len(Qpscan)*len(Nbscan)*len(omegasscan)*(2*mmax+1)
     - tuneshiftnx: tune shifts for all multibunch modes and synchrotron modes m.
    Array of dimensions len(Qpscan)*len(nxscan)*len(Nbscan)*len(omegasscan)*(2*mmax+1)
     - tuneshiftm0: tune shifts for the most unstable multibunch mode and synchrotron mode m=0.
    Array of dimensions len(Qpscan)*len(Nbscan)*len(omegasscan)
     - Zeff: effective impedance for different multibunch modes and synchrotron modes m.
    Array of dimensions len(Qpscan)*len(nxscan)*len(omegasscan)*(2*mmax+1)

    Input parameters are similar to DELPHI's ones:
     - imp_mod: impedance model (list of impedance-wake objects),
     - Qpscan: scan in Q' (DeltaQ*p/Deltap),
     - nxscan: scan in multibunch modes (from 0 to M-1),
     - Nbscan: scan in number of particles per bunch,
     - omegasscan: scan in synchrotron angular frequency (Qs*omega0),
     - M: number of bunches,
     - omega0: angular revolution frequency
     - Q: transverse betatron tune (integer part + fractional part),
     - gamma: relativistic mass factor,
     - eta: slip factor (Elias's convention, i.e. oppostie to Joel Le Duff),
     - taub: total bunch length in seconds,
     - mmax: azimuthal modes considered are from -mmax to mmax,
     - particle: 'proton' or 'electron',
     - modetype: 'sinusoidal' or 'Hermite': kind of modes in effective impedance,'
     - compname: component to extract from impedance model

     see Elias Metral's USPAS 2009 course : Bunched beams transverse coherent
     instabilities.

     NOTE: this is NOT the original Sacherer formula, which assumes an impedance normalized by beta
     (see E. Metral, USPAS 2009 lectures, or C. Zannini,
     https://indico.cern.ch/event/766028/contributions/3179810/attachments/1737652/2811046/Z_definition.pptx)
     Here this formula is instead divided by beta (compared to Sacherer initial one),
     so is valid with our usual definition of impedance (not beta-normalized).
     This was corrected on April 15th, 2019. NM
     """
    if particle == 'proton':
        m0 = m_p
    else:
        raise ValueError('Works only for protons for now. To use other particles we need to'
                         'adjust the mass in the line above')
    E0 = scipy.constants.physical_constants['proton mass energy equivalent in MeV'][0] * 1e6

    # some parameters
    Z0 = np.sqrt(mu_0 / epsilon_0)  # free space impedance: here mu0 c (SI unit - Ohm) or 4 pi/c (c.g.s)
    beta = np.sqrt(1. - 1. / (gamma ** 2))  # relativistic velocity factor
    f0 = omega0 / (2. * np.pi)  # revolution angular frequency
    Ibscan = e * Nbscan * f0  # single-bunch intensity
    Qfrac = Q - np.floor(Q)  # fractional part of the tune
    Lb = taub * beta * c  # full bunch length (in meters)

    Zcomp = imp_mod.get_component(compname).impedance  # real part of impedance

    # omegap = 2. * np.pi * freq
    # omegam = -omegap[::-1]

    #########SHOULD NOT BE NEEDED BUT CHECK################################
    # compute complex impedance and 'symmetrize' it for negative frequencies
    # Zpcomp=Zreal+1j*Zimag;
    # Zmcomp=-Zpcomp[::-1].conjugate();

    #######NOW THIS WOULD BE Z_func#######
    # omega=np.concatenate((omegam,omegap))
    # Zcomp=np.concatenate((Zmcomp,Zpcomp))

    # first guess of the maximum k, and step for k (compute sums on the array
    # [0 kmax-1]+[Integer]*kmax)

    eps = 1.e-5  # relative precision of the summations
    tuneshiftnx = np.zeros((len(Qpscan), len(nxscan), len(Nbscan), len(omegasscan), 2 * mmax + 1), dtype=complex)
    tuneshift_most = np.zeros((len(Qpscan), len(Nbscan), len(omegasscan), 2 * mmax + 1), dtype=complex)
    tuneshiftm0 = np.zeros((len(Qpscan), len(Nbscan), len(omegasscan)), dtype=complex)
    Zeff = np.zeros((len(Qpscan), len(nxscan), len(omegasscan), 2 * mmax + 1), dtype=complex)

    for iQp, Qp in enumerate(Qpscan):

        omegaksi = Qp * omega0 / eta
        if flagtrapz is None:
            flagtrapz = (np.ceil(100. * (4. * np.pi / taub + abs(omegaksi)) / omega0 / M) > 1e9)
        # print np.ceil(100.*(4.*np.pi/taub+abs(omegaksi))/omega0/M),"flagtrapz=",flagtrapz

        for inx, nx in enumerate(nxscan):  # coupled-bunch modes

            for iomegas, omegas in enumerate(omegasscan):

                for im, m in enumerate(range(-mmax, mmax + 1)):
                    # consider each sychrotron mode individually
                    # sum power spectrum functions and computes effective impedance

                    # sum power functions
                    # BE CAREFUL: maybe for this "normalization sum" the sum should run
                    # on all single-bunch harmonics instead of only coupled-bunch
                    # harmonics (and then the frequency shift should be multiplied by
                    # M). This has to be checked.
                    sum1 = hmmsum(m, omega0, M, nx + Qfrac, taub, omegaksi, eps=eps, omegas=omegas, k_max=20,
                                  mode_type=modetype, flag_trapz=flagtrapz)

                    # effective impedance
                    omega = None  # fixxxxx
                    sum2 = hmmsum(m, omega0, M, nx + Qfrac, taub, omegaksi, eps=eps, omegas=omegas, k_max=20,
                                  mode_type=modetype, impedance=Zcomp, omega_impedance_table=omega, flag_trapz=flagtrapz)

                    Zeff[iQp, inx, iomegas, im] = sum2 / sum1
                    freqshift = (1j * e * Ibscan / (2. * (np.abs(
                        m) + 1.) * m0 * gamma * Q * omega0 * Lb)) * sum2 / sum1;  # 15/04/2019 NM: beta suppressed (was for a "beta-normalized" definition of impedance)
                    tuneshiftnx[iQp, inx, :, iomegas, im] = freqshift / omega0 + m * omegas / omega0

        # find the most unstable coupled-bunch mode
        for iomegas, omegas in enumerate(omegasscan):

            for im, m in enumerate(range(-mmax, mmax + 1)):

                inx = np.argmin(np.imag(tuneshiftnx[iQp, :, -1, iomegas, im]))  # check one intensity (last one) is enough

                # print "Sacherer: Qp=",Qp,", M=",M,", omegas=",omegas,", m=",m,", Most unstable coupled-bunch mode: ",nxscan[inx];
                tuneshift_most[iQp, :, iomegas, im] = tuneshiftnx[iQp, inx, :, iomegas, im]
                if (m == 0): tuneshiftm0[iQp, :, iomegas] = tuneshiftnx[iQp, inx, :, iomegas, im]

            # sort tuneshift_most (most unstable modes first) (only one instensity - last one - is enough)
            ind = np.argmin(np.imag(tuneshift_most[iQp, -1, iomegas, :]))
            for iNb, Nb in enumerate(Nbscan): tuneshift_most[iQp, iNb, iomegas, :] = tuneshift_most[
                iQp, iNb, iomegas, ind]

    return tuneshift_most, tuneshiftnx, tuneshiftm0, Zeff

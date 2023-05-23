import numpy as np
from scipy.special import exp1
from scipy.optimize import newton

from typing import Sequence


def dispersion_integral_2d(tune_shift: np.ndarray, b_direct: float, b_cross: float, distribution: str = 'gaussian'):
    """
    Compute the dispersion integral in 2D from q complex tuneshift, given the detuning coefficients (multiplied by
    sigma). This is the integral vs Jx and Jy of Jx*dphi/dJx/(Q-bx*Jx-bxy*Jy-i0) (with phi the distribution function)
    The transverse distribution can be 'gaussian' or 'parabolic'
    NOTE: for stability diagrams, use -1/dispersion_integral, and usually the convention is to plot
    -Im[Q] vs Re[Q].
    :param tune_shift: the complex tune shift
    :param b_direct: the direct detuning coefficient multiplied by sigma (i.e. $\alpha_x \sigma_x$ if working in
    the x plane or $\alpha_y \sigma_y$ if working in the y plane)
    :param b_cross: the cross detuning coefficient multiplied by sigma (i.e. $\alpha_{xy} \sigma_y$ if working in
    the x plane or $\alpha_{yx} \sigma_x$ if working in the y plane)
    :param distribution: the transverse distribution of the beam. It can be 'gaussian' or 'parabolic'
    :return: the dispersion integral
    """
    if np.imag(tune_shift) == 0:
        tune_shift = tune_shift - 1e-15 * 1j

    c = b_cross / b_direct
    q = -tune_shift / b_direct

    if distribution == 'gaussian':

        i1 = (1 - c - (q + c - c * q) * np.exp(q) * exp1(q) + c * np.exp(q / c) * exp1(q / c)) / ((1 - c) ** 2)

        if np.isnan(i1):
            i1 = 1. / q - (c + 2) / q ** 2  # asymptotic form for large q (assuming c is of order 1)

    elif distribution == 'parabolic':

        xi = q / 5.

        if np.abs(xi) > 100.:
            # asymptotic form for large q (assuming c is of order 1) (obtained thanks to Mathematica - actually the same
            # as for Gaussian)
            i1 = 1. / q - (c + 2) / q ** 2
        else:
            i1 = (((c + xi) ** 3 * np.log((1 + xi) / (c + xi)) +
                   (-1 + c) * (c * (c + 2 * c * xi + (-1 + 2 * c) * xi ** 2) +
                               (-1 + c) * xi ** 2 * (3 * c + xi + 2 * c * xi) * np.log(xi / (1 + xi)))) /
                  ((-1 + c) ** 2 * c ** 2))
            i1 = -i1 * 4. / 5.
            # this is the same as in Scott Berg-Ruggiero CERN SL-AP-96-71 (AP)

    else:
        raise ValueError("distribution can only be 'gaussian' or 'parabolic'")

    i = -i1 / b_direct

    # additional minus sign because for DELPHI we want the integral with dphi/dJx (derivative of distribution) on the
    # numerator, so -[the one of Berg-Ruggiero]
    return -i


def find_octupole_threshold(tune_shift: float, q_s: float, b_direct_ref: float, b_cross_ref: float, i_ref: float,
                            polarity: int = 1, fraction_of_qs_allowed_on_positive_side: float = 0.05,
                            use_newton: float = True, distribution: str = 'gaussian'):
    """
    Find the octupole threshod from a complex tuneshift, given the detuning coefficients (multiplied by sigma).
    Returns 0 if the mode is stable, and 'not found' if the threshold cannot be found (failure of Newton's algorithm).
    It assumes that the focusing and defocusing octupole currents have the same absolute value.
    :param tune_shift: the tune shift for which the octupole threshold is computed
    :param q_s: the synchrotron tune
    :param b_direct_ref: the direct detuning coefficient multiplied by sigma (i.e. $\alpha_x \sigma_x$ if working in
    the x plane or $\alpha_y \sigma_y$ if working in the y plane)
    :param b_cross_ref: the cross detuning coefficient multiplied by sigma (i.e. $\alpha_{xy} \sigma_y$ if working in
    the x plane or $\alpha_{yx} \sigma_x$ if working in the y plane)
    :param i_ref: the octupole current for which the detuning coefficients have been computed
    :param polarity: 1 if using positive octupole polarity, -1 if using negative octupole polarity,
    :param use_newton: if True, solve for the octupole current using Newton's algorithm (default), if False, do a simple
    estimate
    :param distribution: the transverse distribution of the beam. It can be 'gaussian' or 'parabolic'
    :param fraction_of_qs_allowed_on_positive_side: to determine azimuthal mode number l_mode (around which is drawn the
    stability diagram), one can consider positive tuneshift up to this fraction of q_s (default=5%)
    :return: the octupole threshold if the corresponding mode is unstable, 0 if the corresponding mode is stable or
     'not found' if the threshold cannot be found (failure of Newton's algorithm).
    """
    # evaluate azimuthal mode number
    l_mode = np.int(np.ceil(np.real(tune_shift) / q_s))
    if (l_mode - np.real(tune_shift) / q_s) > 1 - fraction_of_qs_allowed_on_positive_side:
        l_mode -= 1
    # take away the shift from azimuthal mode number
    tune_shift -= q_s * l_mode

    if tune_shift.imag < 0.:

        # function to solve (distance in imag. part w.r.t stab. diagram, as a function of oct. current)
        def f(i):
            b_direct_i = b_direct_ref * i / i_ref
            b_cross_i = b_cross_ref * i / i_ref
            stab = [-1. / dispersion_integral_2d(t_s, b_direct_i, b_cross_i, distribution=distribution) for e in (-1, 1)
                    for t_s in b_direct_i * e * 10. ** np.arange(-3, 2, 0.01)[::e]]
            # note: one has to reverse the table to get the interpolation right,
            # for negative polarity
            # (np.interp always wants monotonically increasing abscissae)
            return tune_shift.imag - np.interp(tune_shift.real, np.real(stab)[::polarity], np.imag(stab)[::polarity])

        # first estimate (using a interpolation of complex argument and a linear scaling)
        bx1 = b_direct_ref * 1 / i_ref
        bxy1 = b_cross_ref * 1 / i_ref
        stab1 = [-1. / dispersion_integral_2d(t_s, bx1, bxy1, distribution=distribution) for e in (-1, 1)
                 for t_s in bx1 * e * 10. ** np.arange(-3, 2, 0.01)[::e]]
        # note: one has to reverse the table to get the interpolation right,
        # for negative polarity (see above)
        i_oct_0 = np.abs(tune_shift) / np.interp(np.angle(tune_shift), np.angle(stab1)[::polarity],
                                                 np.abs(stab1)[::polarity])

        if use_newton:
            # Newton root finding
            try:
                i_oct = newton(f, i_oct_0, tol=1e-10)
            except RuntimeError:
                i_oct = 'not found'
            else:
                if np.abs(f(i_oct)) > 1e-10:
                    i_oct = 'not found'
        else:
            i_oct = i_oct_0
    else:
        i_oct = 0.

    return i_oct


def find_max_i_oct_from_tune_shifts(tune_shifts: Sequence[float], q_s: float, b_direct_ref: float, b_cross_ref: float,
                                    i_ref: float, polarity: int = 1, use_newton: bool = True,
                                    distribution: str = 'gaussian',
                                    fraction_of_qs_allowed_on_positive_side: float = 0.05):
    """
    Compute the maximum octupole threshold for a sequence of complex tune shifts. It assumes that the focusing and
    defocusing octupole currents have the same absolute value.
    :param tune_shifts: the sequence of complex tune shifts
    :param q_s: the synchrotron tune
    :param b_direct_ref: the direct detuning coefficient multiplied by sigma (i.e. $\alpha_x \sigma_x$ if working in
    the x plane or $\alpha_y \sigma_y$ if working in the y plane)
    :param b_cross_ref: the cross detuning coefficient multiplied by sigma (i.e. $\alpha_{xy} \sigma_y$ if working in
    the x plane or $\alpha_{yx} \sigma_x$ if working in the y plane)
    :param i_ref: the octupole current for which the detuning coefficients have been computed
    :param polarity: 1 if using positive octupole polarity, -1 if using negative octupole polarity,
    :param use_newton: if True, solve for the octupole current using Newton's algorithm (default), if False, do a simple
    estimate
    :param distribution: the transverse distribution of the beam. It can be 'gaussian' or 'parabolic'
    :param fraction_of_qs_allowed_on_positive_side: to determine azimuthal mode number l_mode (around which is drawn the
    stability diagram), one can consider positive tuneshift up to this fraction of q_s (default=5%)
    """
    # find max octupole current required from a list of modes, given their tuneshifts
    i_oct_all = [find_octupole_threshold(tune_shift=tune_shift, q_s=q_s, b_direct_ref=b_direct_ref, b_cross_ref=b_cross_ref, i_ref=i_ref,
                                         polarity=polarity, use_newton=use_newton, distribution=distribution,
                                         fraction_of_qs_allowed_on_positive_side=fraction_of_qs_allowed_on_positive_side
                                         )
                 for tune_shift in tune_shifts if not np.isnan(tune_shift)]

    i_oct_max = np.max([i_oct for i_oct in i_oct_all if i_oct != 'not found'])

    return i_oct_max

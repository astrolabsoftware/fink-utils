# Copyright 2022-2023 AstroLab Software
# Author: Julien Peloton
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy import linalg

from fink_utils.test.tester import regular_unit_tests

def func_hg(ph, h, g):
    """ Return f(H, G) part of the lightcurve in mag space

    Parameters
    ----------
    ph: array-like
        Phase angle in radians
    h: float
        Absolute magnitude in mag
    G: float
        G parameter (no unit)
    """
    from sbpy.photometry import HG

    # Standard G part
    func1 = (1 - g) * HG._hgphi(ph, 1) + g * HG._hgphi(ph, 2)
    func1 = -2.5 * np.log10(func1)

    return h + func1

def func_hg12(ph, h, g12):
    """ Return f(H, G) part of the lightcurve in mag space

    Parameters
    ----------
    ph: array-like
        Phase angle in radians
    h: float
        Absolute magnitude in mag
    G: float
        G parameter (no unit)
    """
    from sbpy.photometry import HG1G2, HG12

    # Standard G1G2 part
    g1 = HG12._G12_to_G1(g12)
    g2 = HG12._G12_to_G2(g12)
    func1 = g1 * HG1G2._phi1(ph) + g2 * HG1G2._phi2(ph) + (1 - g1 - g2) * HG1G2._phi3(ph)
    func1 = -2.5 * np.log10(func1)

    return h + func1

def func_hg1g2(ph, h, g1, g2):
    """ Return f(H, G1, G2) part of the lightcurve in mag space

    Parameters
    ----------
    ph: array-like
        Phase angle in radians
    h: float
        Absolute magnitude in mag
    G1: float
        G1 parameter (no unit)
    G2: float
        G2 parameter (no unit)
    """
    from sbpy.photometry import HG1G2

    # Standard G1G2 part
    func1 = g1 * HG1G2._phi1(ph) + g2 * HG1G2._phi2(ph) + (1 - g1 - g2) * HG1G2._phi3(ph)
    func1 = -2.5 * np.log10(func1)

    return h + func1

def func_hg1g2_with_spin(pha, h, g1, g2, R, lambda0, beta0):
    """ Return f(H, G1, G2, R, lambda0, beta0) part of the lightcurve in mag space

    Parameters
    ----------
    pha: array-like [3, N]
        List containing [phase angle in radians, RA in radians, Dec in radians]
    h: float
        Absolute magnitude in mag
    G1: float
        G1 parameter (no unit)
    G2: float
        G2 parameter (no unit)
    R: float
        Oblateness (no units)
    lambda0: float
        RA of the spin (radian)
    beta0: float
        Dec of the spin (radian)
    """
    ph = pha[0]
    ra = pha[1]
    dec = pha[2]

    # Standard HG1G2 part: h + f(alpha, G1, G2)
    func1 = func_hg1g2(ph, h, g1, g2)

    # Spin part
    geo = np.sin(dec) * np.sin(beta0) + np.cos(dec) * np.cos(beta0) * np.cos(ra - lambda0)
    func2 = 1 - (1 - R) * np.abs(geo)
    func2 = -2.5 * np.log10(func2)

    return func1 + func2

def add_ztf_color_correction(pdf, combined=False):
    """ Add a new column with ZTF color correction.

    The factor is color-dependent, and assumed to be:
    - V_minus_g = -0.32
    - V_minus_r = 0.13
    g --> g + (V - g)
    r --> r + (V - r) - (V - g)

    Parameters
    ----------
    pdf: pd.DataFrame
        Pandas DataFrame with Fink ZTF data
    combined: bool
        If True, normalised using g

    Returns
    ----------
    out: pd.DataFrame
        Input Pandas DataFrame with a new column `color_corr`
    """
    filts = np.unique(pdf['i:fid'].values)
    color_sso = np.ones_like(pdf['i:magpsf'])
    for i, filt in enumerate(filts):
        # SSO Color index
        V_minus_g = -0.32
        V_minus_r = 0.13

        cond = pdf['i:fid'] == filt

        # Color conversion
        if filt == 1:
            color_sso[cond] = V_minus_g
        else:
            if combined:
                color_sso[cond] = V_minus_r - V_minus_g
            else:
                color_sso[cond] = V_minus_r

    pdf['color_corr'] = color_sso

    return pdf

def estimate_sso_params(
        pdf,
        func,
        bounds=([0, 0, 0, 1e-6, 0, -np.pi / 2], [30, 1, 1, 1, 2 * np.pi, np.pi / 2])):
    """ Fit for phase curve parameters

    Parameters
    ----------
    pdf: pd.DataFrame
        Contain Fink data + ephemerides
    func: function
        Parametric function. Currently supported:
            - func_hg1g2_with_spin
            - func_hg1g2
            - func_hg12
            - func_hg
    bounds: tuple of lists
        Parameters boundaries for `func` ([all_mins], [all_maxs]).
        Defaults are given for `func_hg1g2_with_spin`: (H, G1, G2, R, lambda0, beta0).

    Returns
    ----------
    popt: list
        Estimated parameters for `func`
    perr: list
        Error estimates on popt elements
    chi2_red: float
        Reduced chi2

    Example
    ---------
    >>> import io
    >>> import requests
    >>> import pandas as pd

    >>> r = requests.post(
    ...     'https://fink-portal.org/api/v1/sso',
    ...     json={
    ...         'n_or_d': '1465',
    ...         'withEphem': True,
    ...         'output-format': 'json'
    ...     }
    ... )

    # Extract relevant information
    >>> pdf = pd.read_json(io.BytesIO(r.content))

    # Add color correction in the DataFrame
    >>> pdf = add_ztf_color_correction(pdf, combined=True)

    >>> bounds = ([0, 0, 0, 1e-1, 0, -np.pi/2], [30, 1, 1, 1, 2*np.pi, np.pi/2])
    >>> popt, perr, chisq_red = estimate_sso_params(pdf, func=func_hg1g2_with_spin, bounds=bounds)
    >>> assert popt is not None, "Fit failed!"

    >>> bounds = ([0, 0, 0], [30, 1, 1])
    >>> popt, perr, chisq_red = estimate_sso_params(pdf, func=func_hg1g2, bounds=bounds)
    >>> assert popt is not None, "Fit failed!"

    >>> bounds = ([0, 0], [30, 1])
    >>> popt, perr, chisq_red = estimate_sso_params(pdf, func=func_hg12, bounds=bounds)
    >>> assert popt is not None, "Fit failed!"

    >>> bounds = ([0, 0], [30, 1])
    >>> popt, perr, chisq_red = estimate_sso_params(pdf, func=func_hg, bounds=bounds)
    >>> assert popt is not None, "Fit failed!"
    """
    ydata = pdf['i:magpsf_red'] + pdf['color_corr']

    # Fink returns degrees -- convert in radians
    alpha = np.deg2rad(pdf['Phase'].values)
    ra = np.deg2rad(pdf['i:ra'].values)
    dec = np.deg2rad(pdf['i:dec'].values)
    pha = np.transpose([[i, j, k] for i, j, k in zip(alpha, ra, dec)])

    if func.__name__ == 'func_hg1g2_with_spin':
        nparams = 6
        assert len(bounds[0]) == nparams, "You need to specify bounds on all (H, G1, G2, R, alpha, beta) parameters"
        x = pha
    elif func.__name__ == 'func_hg1g2':
        nparams = 3
        assert len(bounds[0]) == nparams, "You need to specify bounds on all (H, G1, G2) parameters"
        x = alpha
    elif func.__name__ == 'func_hg12':
        nparams = 2
        assert len(bounds[0]) == nparams, "You need to specify bounds on all (H, G12) parameters"
        x = alpha
    elif func.__name__ == 'func_hg':
        nparams = 2
        assert len(bounds[0]) == nparams, "You need to specify bounds on all (H, G) parameters"
        x = alpha

    if not np.alltrue([i == i for i in ydata.values]):
        popt = [None] * nparams
        perr = [None] * nparams
        chisq_red = None
        return popt, perr, chisq_red

    try:
        # TODO: incorporate jacobian and error estimates
        popt, pcov = curve_fit(
            func,
            x,
            ydata.values,
            # sigma=pdf['i:sigmapsf'],
            bounds=bounds,
            # jac=Dfunc_hg1g2_with_spin
        )

        perr = np.sqrt(np.diag(pcov))

        r = ydata.values - func(x, *popt)
        chisq = np.sum((r / pdf['i:sigmapsf'])**2)
        chisq_red = 1. / (len(ydata.values) - 1 - nparams) * chisq

    except RuntimeError as e:
        print(e)
        popt = [None] * nparams
        perr = [None] * nparams
        chisq_red = None

    return popt, perr, chisq_red

def build_eqs_for_spins(x, filters=[], ph=[], ra=[], dec=[], rhs=[]):
    """ Build the system of equations to solve using the HG1G2 + spin model

    Parameters
    ----------
    x: list
        List of parameters to fit for
    filters: np.array
        Array of size N containing the filtername for each measurement
    ph: np.array
        Array of size N containing phase angles
    ra: np.array
        Array of size N containing the RA (radian)
    dec: np.array
        Array of size N containing the Dec (radian)
    rhs: np.array
        Array of size N containing the actual measurements (magnitude)

    Returns
    ----------
    out: np.array
        Array of size N containing (model - y)

    Notes
    ----------
    the input `x` should start with filter independent variables,
    that is (R, alpha, beta), followed by filter dependent variables,
    that is (H, G1, G2). For example with two bands g & r:

    ```
    x = [
        R, alpha, beta,
        h_g, g_1_g, g_2_g,
        h_r, g_1_r, g_2_r
    ]
    ```

    """
    R, alpha, beta = x[0:3]
    filternames = np.unique(filters)

    params = x[3:]
    nparams = len(params) / len(filternames)
    assert int(nparams) == nparams, "You need to input all parameters for all bands"

    params_per_band = np.reshape(params, (len(filternames), int(nparams)))
    eqs = []
    for index, filtername in enumerate(filternames):
        mask = filters == filtername

        myfunc = func_hg1g2_with_spin(
            np.vstack([ph[mask].tolist(), ra[mask].tolist(), dec[mask].tolist()]),
            params_per_band[index][0], params_per_band[index][1], params_per_band[index][2],
            R, alpha, beta
        ) - rhs[mask]

        eqs = np.concatenate((eqs, myfunc))

    return np.ravel(eqs)

def estimate_hybrid_sso_params(
        magpsf_red, sigmapsf, phase, ra, dec, filters,
        p0=[15.0, 0.0, 0.0, 0.5, np.pi, 0.0],
        bounds=([0, 0, 0, 1e-6, 0, -np.pi / 2], [30, 1, 1, 1, 2 * np.pi, np.pi / 2])):
    """ Fit for phase curve parameters (R, alpha, delta, H^b, G_1^b, G_2^b)

    Code for quality `fit`:
    0: success
    1: bad_vals
    2: MiriadeFail
    3: RunTimError
    4: LinalgError

    Code for quality `status` (least square convergence):
    -2: failure
    -1 : improper input parameters status returned from MINPACK.
    0 : the maximum number of function evaluations is exceeded.
    1 : gtol termination condition is satisfied.
    2 : ftol termination condition is satisfied.
    3 : xtol termination condition is satisfied.
    4 : Both ftol and xtol termination conditions are satisfied.

    Typically, one would only trust status = 2 and 4.


    Parameters
    ----------
    magpsf_red: array
        Reduced magnitude, that is m_obs - 5 * np.log10('Dobs' * 'Dhelio')
    sigmapsf: array
        Error estimates on magpsf_red
    phase: array
        Phase angle [rad]
    ra: array
        Right ascension [rad]
    dec: array
        Declination [rad]
    filters: array
        Filter name for each measurement
    p0: list
        Initial guess for [H, G1, G2, R, alpha, beta]. Note that even if
        there is several bands `b`, we take the same initial guess for all (H^b, G1^b, G2^b).
    bounds: tuple of lists
        Parameters boundaries for `func_hg1g2_with_spin` ([all_mins], [all_maxs]).
        Lists should be ordered as: (H, G1, G2, R, alpha, beta). Note that even if
        there is several bands `b`, we take the same bounds for all (H^b, G1^b, G2^b).

    Returns
    ----------
    outdic: dict
        Dictionary containing reduced chi2, and estimated parameters and
        error on each parameters.

    Example
    ---------
    >>> import io
    >>> import requests
    >>> import pandas as pd

    >>> r = requests.post(
    ...     'https://fink-portal.org/api/v1/sso',
    ...     json={
    ...         'n_or_d': '1465',
    ...         'withEphem': True,
    ...         'output-format': 'json'
    ...     }
    ... )

    # Extract relevant information
    >>> pdf = pd.read_json(io.BytesIO(r.content))
    >>> magpsf_red = pdf['i:magpsf_red'].values
    >>> sigmapsf = pdf['i:sigmapsf'].values
    >>> phase = np.deg2rad(pdf['Phase'].values)
    >>> ra = np.deg2rad(pdf['i:ra'].values)
    >>> dec = np.deg2rad(pdf['i:dec'].values)
    >>> filters = pdf['i:fid'].values

    >>> outdic = estimate_hybrid_sso_params(magpsf_red, sigmapsf, phase, ra, dec, filters)
    >>> assert outdic['fit'] == 0, "Fit failed with code {}!".format(outdic['fit'])

    >>> assert outdic['status'] == 2, "Convergence failed with code {}!".format(outdic['status'])
    """
    ufilters = np.unique(filters)

    params = ['R', 'alpha0', 'delta0']
    spin_params = ['H', 'G1', 'G2']
    for filt in ufilters:
        spin_params_with_filt = [i + '_{}'.format(str(filt)) for i in spin_params]
        params = np.concatenate((params, spin_params_with_filt))

    initial_guess = p0[3:]
    for filt in ufilters:
        initial_guess = np.concatenate((initial_guess, p0[:3]))

    lower_bounds = bounds[0][3:]
    upper_bounds = bounds[1][3:]
    for filt in ufilters:
        lower_bounds = np.concatenate((lower_bounds, bounds[0][:3]))
        upper_bounds = np.concatenate((upper_bounds, bounds[1][:3]))

    if not np.alltrue([i == i for i in magpsf_red]):
        outdic = {'fit': 1, 'status': -2}
        return outdic

    try:
        res_lsq = least_squares(
            build_eqs_for_spins,
            x0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            jac='2-point',
            loss='linear',
            args=(filters, phase, ra, dec, magpsf_red)
        )

    except RuntimeError:
        outdic = {'fit': 3, 'status': -2}
        return outdic

    popt = res_lsq.x

    # estimate covariance matrix using the jacobian
    try:
        cov = linalg.inv(res_lsq.jac.T @ res_lsq.jac)
        chi2dof = np.sum(res_lsq.fun**2) / (res_lsq.fun.size - res_lsq.x.size)
        cov *= chi2dof

        # 1sigma uncertainty on fitted parameters
        perr = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        # raised if jacobian is degenerated
        outdic = {'fit': 4, 'status': res_lsq.status}
        return outdic

    rms = np.sqrt(np.mean(res_lsq.fun**2))

    # For the chi2, we use the error estimate from the data directly
    chisq = np.sum((res_lsq.fun / sigmapsf)**2)
    chisq_red = chisq / (res_lsq.fun.size - res_lsq.x.size - 1)

    outdic = {
        'chi2red': chisq_red,
        'rms': rms,
        'minphase': np.min(phase),
        'maxphase': np.max(phase),
        'status': res_lsq.status,
        'fit': 0
    }
    for i in range(len(params)):
        outdic[params[i]] = popt[i]
        outdic['err' + params[i]] = perr[i]

    return outdic


if __name__ == "__main__":
    """Execute the unit test suite"""

    # Run the Spark test suite
    regular_unit_tests(globals())

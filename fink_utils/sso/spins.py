# Copyright 2022-2025 AstroLab Software
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
import pandas as pd

from scipy.optimize import least_squares
from scipy import linalg

from astropy.coordinates import SkyCoord
import astropy.units as u

from fink_utils.sso.utils import estimate_axes_ratio
from fink_utils.sso.utils import get_opposition, split_dataframe_per_apparition
from fink_utils.tester import regular_unit_tests


def sort_quantity_by_filter(filter, quantity):
    """Sort a vector (quantity) by its corresponding filter under which it was measured

    Parameters
    ----------
    filter: array-like (1xN)
        filters used to measure quantity (1,2,3,4...)
    quantity: array-like (1xN)
        quantity to be sorted according to the filters

    Returns
    -------
    sorted_quantity: np.array of floats (1xN)
        quantity sorted by the filter
    """
    _, sorted_quantity = zip(*sorted(zip(filter, quantity)))
    return np.array(sorted_quantity)


def split_quantity_by_filter(list_of_filters, ordered_vector):
    """
    Split an ordered (by filter) vector at each filter

    Parameters
    ----------
    list_of_filters: array-like (1xN)
        filters used to measure quantity (1,2,3,4...)
    ordered_vector: array-like (1xN)
        quantity to be split according to the filters

    Returns
    -------
    Sub-arrays containing the parts of the ordered_vector at the order of the list_of_filters
    """
    ufilters = np.unique(list_of_filters)
    split_at = []
    for filt in ufilters:
        mask = list_of_filters == filt
        split_at.append(list_of_filters[mask].size)
    return np.array_split(ordered_vector, np.cumsum(split_at))


def func_hg(ph, h, g):
    """Return f(H, G) part of the lightcurve in mag space

    Parameters
    ----------
    ph: array-like
        Phase angle in radians
    h: float
        Absolute magnitude in mag
    G: float
        G parameter (no unit)

    Returns
    -------
    out: array of floats
        H - 2.5 log(f(G))
    """
    from sbpy.photometry import HG

    # Standard G part
    func1 = (1 - g) * HG._hgphi(ph, 1) + g * HG._hgphi(ph, 2)
    func1 = -2.5 * np.log10(func1)

    return h + func1


def func_hg12(ph, h, g12):
    """Return f(H, G) part of the lightcurve in mag space

    Parameters
    ----------
    ph: array-like
        Phase angle in radians
    h: float
        Absolute magnitude in mag
    G: float
        G parameter (no unit)

    Returns
    -------
    out: array of floats
        H - 2.5 log(f(G12))
    """
    from sbpy.photometry import HG1G2, HG12

    # Standard G1G2 part
    g1 = HG12._G12_to_G1(g12)
    g2 = HG12._G12_to_G2(g12)
    func1 = (
        g1 * HG1G2._phi1(ph) + g2 * HG1G2._phi2(ph) + (1 - g1 - g2) * HG1G2._phi3(ph)
    )
    func1 = -2.5 * np.log10(func1)

    return h + func1


def func_hg1g2(ph, h, g1, g2):
    """Return f(H, G1, G2) part of the lightcurve in mag space

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

    Returns
    -------
    out: array of floats
        H - 2.5 log(f(G1G2))
    """
    from sbpy.photometry import HG1G2

    # Standard G1G2 part
    func1 = (
        g1 * HG1G2._phi1(ph) + g2 * HG1G2._phi2(ph) + (1 - g1 - g2) * HG1G2._phi3(ph)
    )
    func1 = -2.5 * np.log10(func1)

    return h + func1


def func_hg1g2_with_spin(pha, h, g1, g2, R, alpha0, delta0):
    """Return f(H, G1, G2, R, alpha0, delta0) part of the lightcurve in mag space

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
    alpha0: float
        RA of the spin (radian)
    delta0: float
        Dec of the spin (radian)

    Returns
    -------
    out: array of floats
        H - 2.5 log(f(G1G2)) - 2.5 log(f(R, spin))
    """
    ph = pha[0]
    ra = pha[1]
    dec = pha[2]

    # Standard HG1G2 part: h + f(alpha, G1, G2)
    func1 = func_hg1g2(ph, h, g1, g2)

    # Spin part
    geo = cos_aspect_angle(ra, dec, alpha0, delta0)
    func2 = 1 - (1 - R) * np.abs(geo)
    func2 = 2.5 * np.log10(func2)

    return func1 + func2


def cos_aspect_angle(ra, dec, ra0, dec0):
    """Compute the cosine of the aspect angle

    This angle is computed from the coordinates of the target and
    the coordinates of its pole.
    See Eq 12.4 "Introduction to Ephemerides and Astronomical Phenomena", IMCCE

    Parameters
    ----------
    ra: float
        Right ascension of the target in radians.
    dec: float
        Declination of the target in radians.
    ra0: float
        Right ascension of the pole in radians.
    dec0: float
        Declination of the pole in radians.

    Returns
    -------
    float: The cosine of the aspect angle
    """
    return np.sin(dec) * np.sin(dec0) + np.cos(dec) * np.cos(dec0) * np.cos(ra - ra0)


def rotation_phase(t, W0, W1, t0):
    """Compute the rotational phase

    This angle is computed from the location of the prime meridian at
    at reference epoch (W0, t0), and an angular velocity (W1)
    See Eq 12.1 "Introduction to Ephemerides and Astronomical Phenomena", IMCCE

    Parameters
    ----------
    t: float
        Time (JD)
    W0: float
        Location of the prime meridian at reference epoch (radian)
    W1: float
        Angular velocity of the target in radians/day.
    t0: float
        Reference epoch (JD)

    Returns
    -------
    float: The rotational phase W (radian)
    """
    return W0 + W1 * (t - t0)


def subobserver_longitude(ra, dec, ra0, dec0, W):
    """Compute the subobserver longitude (radian)

    This angle is computed from the coordinates of the target,
    the coordinates of its pole, and its rotation phase
    See Eq 12.4 "Introduction to Ephemerides and Astronomical Phenomena", IMCCE

    Parameters
    ----------
    ra: float
        Right ascension of the target in radians.
    dec: float
        Declination of the target in radians.
    ra0: float
        Right ascension of the pole in radians.
    dec0: float
        Declination of the pole in radians.
    W: float
        Rotation phase of the target in radians.

    Returns
    -------
    float: The subobserver longitude in radians.
    """
    x = -np.cos(dec0) * np.sin(dec) + np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)
    y = -(np.cos(dec) * np.sin(ra - ra0))
    return W - np.arctan2(x, y)


def func_sshg1g2(pha, h, g1, g2, alpha0, delta0, period, a_b, a_c, phi0):
    """Return f(H, G1, G2, alpha0, delta0, period, a_b, a_c, phi0) part of the lightcurve in mag space

    Parameters
    ----------
    pha: array-like [4, N]
        List containing [phase angle in radians, RA in radians, Dec in radians, time (jd)]
    h: float
        Absolute magnitude in mag
    G1: float
        G1 parameter (no unit)
    G2: float
        G2 parameter (no unit)
    alpha0: float
        RA of the spin (radian)
    delta0: float
        Dec of the spin (radian)
    period: float
        Sidereal rotation period (days)
    a_b: float
        Equatorial axes ratio
    a_c: float
        Polar axes ratio
    phi0: float
        Initial rotation phase at reference time t0 (radian)
    t0: float
        Reference time (jd)

    Notes
    -----
    Input times must be corrected from the light travel time,
    that is jd_lt = jd - d_obs / c_speed

    Returns
    -------
    out: array of floats
        H - 2.5 log(f(G1G2)) - 2.5 log(f(spin, shape))
    """
    ph = pha[0]
    ra = pha[1]
    dec = pha[2]
    ep = pha[3]

    # TBD: For the time being, we fix the reference time
    # Time( '2022-01-01T00:00:00', format='isot', scale='utc').jd
    # Kinda middle of ZTF
    # TODO: take the middle jd?
    t0 = 2459580.5

    # Standard HG1G2 part: h + f(alpha, G1, G2)
    func1 = func_hg1g2(ph, h, g1, g2)

    # Spin part
    cos_aspect = cos_aspect_angle(ra, dec, alpha0, delta0)
    cos_aspect_2 = cos_aspect**2
    sin_aspect_2 = 1 - cos_aspect_2

    # Sidereal
    W = rotation_phase(ep, phi0, 2 * np.pi / period, t0)
    rot_phase = subobserver_longitude(ra, dec, alpha0, delta0, W)

    # https://ui.adsabs.harvard.edu/abs/1985A%26A...149..186P/abstract
    func2 = np.sqrt(
        sin_aspect_2 * (np.cos(rot_phase) ** 2 + (a_b**2) * np.sin(rot_phase) ** 2)
        + cos_aspect_2 * a_c**2
    )
    func2 = -2.5 * np.log10(func2)

    return func1 + func2


def func_sfhg1g2(phas, g1, g2, *args):
    """HG1G2 model in the case of simultaneous fit

    Parameters
    ----------
    phas: np.array
        (N, M_o) array of phase angles. N is the number
        of opposition, M_o is the number of observations
        per opposition. Phase is radians.
    g1: float
        G1 parameter (no unit)
    g2: float
        G2 parameter (no unit)
    args: float
        List of Hs, one per opposition.

    Returns
    -------
    out: np.array
        Magnitude as predicted by `func_hg1g2`.
    """
    fl = []
    for alpha, h in zip(phas, args[0][:]):
        fl.append(func_hg1g2(alpha, h, g1, g2))
    return np.concatenate(fl)


def sfhg1g2_error_fun(params, phas, mags):
    """Difference between sfHG1G2 predictions and observations

    Parameters
    ----------
    params: list
        [G1, G2, *H_i], where H_i are the Hs, one per opposition
    phas: np.array
        (N, M_o) array of phase angles. N is the number
        of opposition, M_o is the number of observations
        per opposition. Must be sorted by time. Phase is radians.
    mags: np.array
        Reduced magnitude, that is m_obs - 5 * np.log10('Dobs' * 'Dhelio')
        Sorted by time.
    """
    return func_sfhg1g2(phas, params[0], params[1], params[2:]) - mags


def func_sshg1g2_terminator(pha, h, g1, g2, alpha0, delta0, period, a_b, a_c, phi0):
    """Extension of the ssHG1G2 model with correction for the non-illuminated part

    Notes
    -----
    Absolute magnitude is computed according to Ostro & Connelly (1984)

    Parameters
    ----------
    pha: array-like [6, N]
        List containing [phase angle in radians, RA in radians, Dec in radians, time (jd), RA sun in radians, DEC sun in radians ]
    h: float
        Absolute magnitude in mag
    G1: float
        G1 parameter (no unit)
    G2: float
        G2 parameter (no unit)
    alpha0: float
        RA of the spin (radian)
    delta0: float
        Dec of the spin (radian)
    period: float
        Sidereal rotation period (days)
    a_b: float
        Equatorial axes ratio
    a_c: float
        Polar axes ratio
    phi0: float
        Initial rotation phase at reference time t0 (radian)
    t0: float
        Reference time (jd)

    Notes
    -----
    Input times must be corrected from the light travel time,
    that is jd_lt = jd - d_obs / c_speed

    Returns
    -------
    out: array of floats
        H - 2.5 log(f(G1G2)) - 2.5 log(f(spin, shape))
        Similar to the ssHG1G2 model, but including the correction for the non-illuminated part of the asteroid
    """
    ph = pha[0]
    ra = pha[1]
    dec = pha[2]
    ep = pha[3]
    ra_s = pha[4]
    dec_s = pha[5]

    # TBD: For the time being, we fix the reference time
    # Time( '2022-01-01T00:00:00', format='isot', scale='utc').jd
    # Kinda middle of ZTF
    # TODO: take the middle jd?
    t0 = 2459580.5

    # Standard HG1G2 part: h + f(alpha, G1, G2)
    func1 = func_hg1g2(ph, h, g1, g2)

    # Rotation
    W = rotation_phase(ep, phi0, 2 * np.pi / period, t0)

    # Sub-Earth (e.TQe):
    # Spin part
    cos_aspect = cos_aspect_angle(ra, dec, alpha0, delta0)
    cos_aspect_2 = cos_aspect**2
    sin_aspect_2 = 1 - cos_aspect_2

    # Sidereal
    rot_phase = subobserver_longitude(ra, dec, alpha0, delta0, W)

    # https://www.sciencedirect.com/science/article/pii/0019103588901261
    eQe = (
        sin_aspect_2 * (np.cos(rot_phase) ** 2 + (a_b**2) * np.sin(rot_phase) ** 2)
        + cos_aspect_2 * a_c**2
    )

    # Sub-Solar (s.TQs):

    cos_aspect_s = cos_aspect_angle(ra_s, dec_s, alpha0, delta0)
    cos_aspect_s_2 = cos_aspect_s**2
    sin_aspect_s_2 = 1 - cos_aspect_s_2

    # Sidereal
    rot_phase_s = subobserver_longitude(ra_s, dec_s, alpha0, delta0, W)

    sQs = (
        sin_aspect_s_2
        * (np.cos(rot_phase_s) ** 2 + (a_b**2) * np.sin(rot_phase_s) ** 2)
        + cos_aspect_s_2 * a_c**2
    )

    # Cross-term (e.TQs):

    # sin(Lamda), sin(Lamda_sun):
    sin_aspect = np.sqrt(sin_aspect_2)
    sin_aspect_s = np.sqrt(sin_aspect_s_2)

    eQs = (
        sin_aspect * np.cos(rot_phase) * sin_aspect_s * np.cos(rot_phase_s)
        + sin_aspect * np.sin(rot_phase) * sin_aspect_s * np.sin(rot_phase_s) * (a_b**2)
        + cos_aspect * cos_aspect_s * a_c**2
    )

    # Full spin-shape term:
    I_tot = (np.sqrt(eQe) + eQs / np.sqrt(sQs)) / 2
    I_tot = -2.5 * np.log10(I_tot)

    return func1 + I_tot


def illum_and_shadowed_areas(coords, alpha0, delta0, period, a_b, a_c, phi0):
    """Return the full projected asteroid area and the terminator-defined ellipse area

    Parameters
    ----------
    coords: array-like [5]
        List containing [SEP_RA  in radians, SEP_Dec in radians, time (jd),
                         SSP_RA sun in radians, SSP_DEC sun in radians
    alpha0: float
        RA of the spin (radian)
    delta0: float
        Dec of the spin (radian)
    period: float
        Sidereal rotation period (days)
    a_b: float
        Equatorial axes ratio
    a_c: float
        Polar axes ratio
    phi0: float
        Initial rotation phase at reference time t0 (radian)

    Notes
    -----
    Input times must be corrected from the light travel time,
    that is jd_lt = jd - d_obs / c_speed

    Returns
    -------
    E1: array of floats
    E2: array of floats
    """
    ra = coords[0]
    dec = coords[1]
    ep = coords[2]
    ra_s = coords[3]
    dec_s = coords[4]

    # TBD: For the time being, we fix the reference time
    # Time( '2022-01-01T00:00:00', format='isot', scale='utc').jd
    # Kinda middle of ZTF
    # TODO: take the middle jd?
    t0 = 2459580.5

    # Rotation
    W = rotation_phase(ep, phi0, 2 * np.pi / period, t0)

    # Sub-Earth (e.TQe):
    # Spin part
    cos_aspect = cos_aspect_angle(ra, dec, alpha0, delta0)
    cos_aspect_2 = cos_aspect**2
    sin_aspect_2 = 1 - cos_aspect_2

    # Sidereal
    rot_phase = subobserver_longitude(ra, dec, alpha0, delta0, W)

    # https://www.sciencedirect.com/science/article/pii/0019103588901261
    eQe = (
        sin_aspect_2 * (np.cos(rot_phase) ** 2 + (a_b**2) * np.sin(rot_phase) ** 2)
        + cos_aspect_2 * a_c**2
    )

    # Sub-Solar (s.TQs):
    cos_aspect_s = cos_aspect_angle(ra_s, dec_s, alpha0, delta0)
    cos_aspect_s_2 = cos_aspect_s**2
    sin_aspect_s_2 = 1 - cos_aspect_s_2

    # Sidereal
    rot_phase_s = subobserver_longitude(ra_s, dec_s, alpha0, delta0, W)

    sQs = (
        sin_aspect_s_2
        * (np.cos(rot_phase_s) ** 2 + (a_b**2) * np.sin(rot_phase_s) ** 2)
        + cos_aspect_s_2 * a_c**2
    )

    # Cross-term (e.TQs):
    # sin(Lamda), sin(Lamda_sun):
    sin_aspect = np.sqrt(sin_aspect_2)
    sin_aspect_s = np.sqrt(sin_aspect_s_2)

    eQs = (
        sin_aspect * np.cos(rot_phase) * sin_aspect_s * np.cos(rot_phase_s)
        + sin_aspect * np.sin(rot_phase) * sin_aspect_s * np.sin(rot_phase_s) * (a_b**2)
        + cos_aspect * cos_aspect_s * a_c**2
    )

    E1 = eQe
    E2 = eQs / np.sqrt(sQs)

    return E1, E2


def color_correction_to_V():  # noqa: N802
    """Color correction from band to V

    Available:
        - 1: ZTF-g
        - 2: ZTF-r
        - 3: ATLAS-o
        - 4: ATLAS-c

    Returns
    -------
    out: dict
        Dictionary with color correction to V
    """
    dic = {1: -0.2833, 2: 0.1777, 3: 0.4388, 4: -0.0986}

    return dic


def compute_color_correction(filters: np.array) -> np.array:
    """Return the color correction `V - band` for each measurement.

    band --> band + (V - band)

    Parameters
    ----------
    filters: np.array
        Array with the filter code for each measurement

    Returns
    -------
    out: pd.DataFrame
        Array containing the color correction for each measurement

    Example
    ---------
    >>> import io
    >>> import requests
    >>> import pandas as pd

    >>> r = requests.post(
    ...     'https://api.fink-portal.org/api/v1/sso',
    ...     json={
    ...         'n_or_d': '1465',
    ...         'output-format': 'json'
    ...     }
    ... )

    # Extract relevant information
    >>> pdf = pd.read_json(io.BytesIO(r.content))

    # Compute color correction
    >>> color_to_V = compute_color_correction(pdf['i:fid'].values)
    >>> assert len(np.unique(color_to_V)) == len(np.unique(pdf['i:fid'].values)), "Filters and colors have different shape!"

    >>> assert 0.0 not in color_to_V, "Some filters have no color correction!"
    """
    filts = np.unique(filters)
    color_sso = np.zeros_like(filters, dtype=float)
    conversion = color_correction_to_V()
    for filt in filts:
        if filt == 3:
            continue
        cond = filters == filt
        color_sso[cond] = conversion[filt]

    return color_sso


def build_eqs(x, filters, ph, rhs, func=None):
    """Build the system of equations to solve using the HG, HG12, or HG1G2 model

    Parameters
    ----------
    x: list
        List of parameters to fit for
    filters: np.array
        Array of size N containing the filtername for each measurement
    ph: np.array
        Array of size N containing phase angles (rad)
    rhs: np.array
        Array of size N containing the actual measurements (magnitude)
    func: callable
        Model function to use (e.g. `func_hg1g2`)

    Returns
    -------
    out: np.array
        Array of size N containing (model - y)

    Notes
    -----
    the input `x` contains filter dependent variables,
    that is (Hs and Gs). For example with two bands g & r with the HG1G2 model:

    ```
    x = [
        h_g, g_1_g, g_2_g,
        h_r, g_1_r, g_2_r
    ]
    ```

    """
    filternames = np.unique(filters)

    params = x
    nparams = len(params) / len(filternames)
    assert int(nparams) == nparams, "You need to input all parameters for all bands"

    params_per_band = np.reshape(params, (len(filternames), int(nparams)))
    eqs = []
    for index, filtername in enumerate(filternames):
        mask = filters == filtername

        myfunc = (
            func(
                ph[mask],
                *params_per_band[index],
            )
            - rhs[mask]
        )

        eqs = np.concatenate((eqs, myfunc))

    return np.ravel(eqs)


def build_eqs_for_spins(x, filters, ph, ra, dec, rhs):
    """Build the system of equations to solve using the HG1G2 + spin model

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
    -------
    out: np.array
        Array of size N containing (model - y)

    Notes
    -----
    the input `x` should start with filter independent variables,
    that is (R, alpha, delta), followed by filter dependent variables,
    that is (H, G1, G2). For example with two bands g & r:

    ```
    x = [
        R, alpha, delta,
        h_g, g_1_g, g_2_g,
        h_r, g_1_r, g_2_r
    ]
    ```

    """
    R, alpha, delta = x[0:3]
    filternames = np.unique(filters)

    params = x[3:]
    nparams = len(params) / len(filternames)
    assert int(nparams) == nparams, "You need to input all parameters for all bands"

    params_per_band = np.reshape(params, (len(filternames), int(nparams)))
    eqs = []
    for index, filtername in enumerate(filternames):
        mask = filters == filtername

        myfunc = (
            func_hg1g2_with_spin(
                np.vstack([ph[mask].tolist(), ra[mask].tolist(), dec[mask].tolist()]),
                params_per_band[index][0],
                params_per_band[index][1],
                params_per_band[index][2],
                R,
                alpha,
                delta,
            )
            - rhs[mask]
        )

        eqs = np.concatenate((eqs, myfunc))

    return np.ravel(eqs)


def build_eqs_for_spin_shape(
    x,
    filters,
    ph,
    ra,
    dec,
    jd,
    rhs,
    terminator=False,
    ra_s=None,
    dec_s=None,
):
    """Build the system of equations to solve using the HG1G2 + spin model

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
    jd: np.array
        Array of size N containing the (time travel corrected) time of the measurements (jd)
    rhs: np.array
        Array of size N containing the actual measurements (magnitude)
    terminator: bool
        Include correction for the non-illuminated part
    ra_s: optional, np.array
        Array of size N containing the solar RA (radian), required if terminator=True
    dec_s: optional, np.array
        Array of size N containing the solar DEC (radian), required if terminator=True

    Returns
    -------
    out: np.array
        Array of size N containing (model - y)

    Notes
    -----
    the input `x` should start with filter independent variables,
    that is (alpha, delta, period, a/b, a/c, phi0), followed by filter dependent variables,
    that is (H, G1, G2). For example with two bands g & r:

    ```
    x = [
        alpha, delta, period, a_b, a_c, phi0,
        h_g, g_1_g, g_2_g,
        h_r, g_1_r, g_2_r
    ]
    ```
    """
    alpha, delta, period, a_b, a_c, phi0 = x[0:6]
    filternames = np.unique(filters)

    params = x[6:]
    nparams = len(params) / len(filternames)
    assert int(nparams) == nparams, "You need to input all parameters for all bands"

    params_per_band = np.reshape(params, (len(filternames), int(nparams)))
    eqs = []
    if not terminator:
        for index, filtername in enumerate(filternames):
            mask = filters == filtername

            myfunc = (
                func_sshg1g2(
                    np.vstack([
                        ph[mask].tolist(),
                        ra[mask].tolist(),
                        dec[mask].tolist(),
                        jd[mask].tolist(),
                    ]),
                    params_per_band[index][0],
                    params_per_band[index][1],
                    params_per_band[index][2],
                    alpha,
                    delta,
                    period,
                    a_b,
                    a_c,
                    phi0,
                )
                - rhs[mask]
            )

            eqs = np.concatenate((eqs, myfunc))
    else:
        for index, filtername in enumerate(filternames):
            mask = filters == filtername

            myfunc = (
                func_sshg1g2_terminator(
                    np.vstack([
                        ph[mask].tolist(),
                        ra[mask].tolist(),
                        dec[mask].tolist(),
                        jd[mask].tolist(),
                        ra_s[mask].tolist(),
                        dec_s[mask].tolist(),
                    ]),
                    params_per_band[index][0],
                    params_per_band[index][1],
                    params_per_band[index][2],
                    alpha,
                    delta,
                    period,
                    a_b,
                    a_c,
                    phi0,
                )
                - rhs[mask]
            )

            eqs = np.concatenate((eqs, myfunc))

    return np.ravel(eqs)


def estimate_sso_params(
    magpsf_red,
    sigmapsf,
    phase,
    filters,
    ra=None,
    dec=None,
    jd=None,
    model="SHG1G2",
    normalise_to_V=False,
    p0=None,
    bounds=None,
    ssnamenr=None,
    terminator=False,
    ra_s=None,
    dec_s=None,
):
    """Fit for phase curve parameters

    Under the hood, it uses a `least_square`. Along with the fitted parameters,
    we also provide flag to assess the quality of the fit:

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
    filters: array
        Filter name for each measurement.
    ra: optional, array
        Right ascension [rad]. Required for SHG1G2 model.
    dec: optional, array
        Declination [rad]. Required for SHG1G2 model.
    jd: options, array
        Observing time (JD), corrected for the light travel. Required for SSHG1G2 model.
    model: str
        Parametric function. Currently supported:
            - SSHG1G2
            - SHG1G2 (default)
            - HG1G2
            - HG12
            - HG
    normalise_to_V: optional, bool
        If True, bring all bands to V. Default is False.
    p0: list
        Initial guess for input parameters. If there are several bands,
        we replicate the bounds for all. Note that in the case of SHG1G2,
        the spin parameters are fitted globally (and not per band).
    bounds: tuple of lists
        Parameters boundaries ([all_mins], [all_maxs]).
        Lists should be ordered as:
            - SSHG1G2: (H, G1, G2, alpha, delta, period, a_b, a_c, phi0)
            - SHG1G2 (default): (H, G1, G2, R, alpha, delta)
            - HG1G2: (H, G1, G2)
            - HG12: (H, G12)
            - HG: (H, G)
        Note that even if there is several bands `b`, we take the same
        bounds for all H's and G's.
    ssnamenr: str, optional
        SSO name/number. Only required for sfHG1G2 model, when
        querying Horizons.
    terminator: bool
        Include correction for the non-illuminated part
    ra_s: optional, np.array
        Array of size N containing the solar RA (radian), required if terminator=True
    dec_s: optional, np.array
        Array of size N containing the solar DEC (radian), required if terminator=True

    Returns
    -------
    outdic: dict
        Dictionary containing reduced chi2, and estimated parameters and
        error on each parameters.

    Examples
    --------
    >>> import io
    >>> import requests
    >>> import pandas as pd

    >>> r = requests.post(
    ...     'https://api.fink-portal.org/api/v1/sso',
    ...     json={
    ...         'n_or_d': '223',
    ...         'withEphem': True,
    ...         'output-format': 'json'
    ...     }
    ... )

    # Extract relevant information
    >>> pdf = pd.read_json(io.BytesIO(r.content))

    >>> hg = estimate_sso_params(
    ...    pdf['i:magpsf_red'].values,
    ...    pdf['i:sigmapsf'].values,
    ...    np.deg2rad(pdf['Phase'].values),
    ...    pdf['i:fid'].values,
    ...    p0=[15.0, 0.15],
    ...    bounds=([-3, 0], [30, 1]),
    ...    model='HG',
    ...    normalise_to_V=False)
    >>> assert len(hg) == 26, "Found {} parameters: {}".format(len(hg), hg)

    >>> hg12 = estimate_sso_params(
    ...    pdf['i:magpsf_red'].values,
    ...    pdf['i:sigmapsf'].values,
    ...    np.deg2rad(pdf['Phase'].values),
    ...    pdf['i:fid'].values,
    ...    p0=[15.0, 0.15],
    ...    bounds=([-3, 0], [30, 1]),
    ...    model='HG12',
    ...    normalise_to_V=False)
    >>> assert len(hg12) == 26, "Found {} parameters: {}".format(len(hg12), hg12)

    >>> hg1g2 = estimate_sso_params(
    ...    pdf['i:magpsf_red'].values,
    ...    pdf['i:sigmapsf'].values,
    ...    np.deg2rad(pdf['Phase'].values),
    ...    pdf['i:fid'].values,
    ...    p0=[15.0, 0.15, 0.15],
    ...    bounds=([-3, 0, 0], [30, 1, 1]),
    ...    model='HG1G2',
    ...    normalise_to_V=False)
    >>> assert len(hg1g2) == 30, "Found {} parameters: {}".format(len(hg1g2), hg1g2)

    >>> shg1g2 = estimate_sso_params(
    ...    pdf['i:magpsf_red'].values,
    ...    pdf['i:sigmapsf'].values,
    ...    np.deg2rad(pdf['Phase'].values),
    ...    pdf['i:fid'].values,
    ...    np.deg2rad(pdf['i:ra'].values),
    ...    np.deg2rad(pdf['i:dec'].values),
    ...    model='SHG1G2',
    ...    normalise_to_V=False)
    >>> assert len(shg1g2) == 41, "Found {} parameters: {}".format(len(shg1g2), shg1g2)

    >>> sshg1g2 = estimate_sso_params(
    ...    pdf['i:magpsf_red'].values,
    ...    pdf['i:sigmapsf'].values,
    ...    np.deg2rad(pdf['Phase'].values),
    ...    pdf['i:fid'].values,
    ...    np.deg2rad(pdf['i:ra'].values),
    ...    np.deg2rad(pdf['i:dec'].values),
    ...    pdf['i:jd'].values,
    ...    model='SSHG1G2',
    ...    normalise_to_V=False)
    >>> assert len(sshg1g2) == 45, "Found {} parameters: {}".format(len(sshg1g2), sshg1g2)

    # You can also combine data into single V band
    >>> shg1g2 = estimate_sso_params(
    ...    pdf['i:magpsf_red'].values,
    ...    pdf['i:sigmapsf'].values,
    ...    np.deg2rad(pdf['Phase'].values),
    ...    pdf['i:fid'].values,
    ...    np.deg2rad(pdf['i:ra'].values),
    ...    np.deg2rad(pdf['i:dec'].values),
    ...    model='SHG1G2',
    ...    normalise_to_V=True)
    >>> assert len(shg1g2) == 30, "Found {} parameters: {}".format(len(shg1g2), shg1g2)

    # If you enter a wrong model name, raise an error
    >>> wrong = estimate_sso_params(
    ...    pdf['i:magpsf_red'].values,
    ...    pdf['i:sigmapsf'].values,
    ...    np.deg2rad(pdf['Phase'].values),
    ...    pdf['i:fid'].values,
    ...    np.deg2rad(pdf['i:ra'].values),
    ...    np.deg2rad(pdf['i:dec'].values),
    ...    model='toto',
    ...    normalise_to_V=True) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    AssertionError: model toto is not understood. Please choose among: SSHG1G2, SHG1G2, HG1G2, HG12, HG
    """
    if normalise_to_V:
        color = compute_color_correction(filters)
        ydata = magpsf_red + color
        filters = np.array(["V"] * len(filters))
    else:
        ydata = magpsf_red

    if model in ["SSHG1G2", "SHG1G2"]:
        outdic = fit_spin(
            ydata,
            sigmapsf,
            phase,
            ra,
            dec,
            filters,
            jd=jd,
            p0=p0,
            bounds=bounds,
            model=model,
            terminator=terminator,
            ra_s=ra_s,
            dec_s=dec_s,
        )
    elif model in ["HG", "HG12", "HG1G2"]:
        outdic = fit_legacy_models(
            ydata, sigmapsf, phase, filters, model, p0=p0, bounds=bounds
        )
    elif model == "sfHG1G2":
        outdic = fit_sfhg1g2(
            ssnamenr,
            ydata,
            sigmapsf,
            jd,
            phase,
            filters,
        )
    else:
        raise AssertionError(
            "model {} is not understood. Please choose among: SSHG1G2, SHG1G2, sfHG1G2, HG1G2, HG12, HG".format(
                model
            )
        )

    return outdic


def fit_legacy_models(
    magpsf_red,
    sigmapsf,
    phase,
    filters,
    model,
    p0=None,
    bounds=None,
):
    """Fit for phase curve parameters

    Parameters
    ----------
    magpsf_red: array
        Reduced magnitude, that is m_obs - 5 * np.log10('Dobs' * 'Dhelio')
    sigmapsf: array
        Error estimates on magpsf_red
    phase: array
        Phase angle [rad]
    filters: array
        Filter name for each measurement
    model: function
        Parametric function. Currently supported:
            - func_hg1g2
            - func_hg12
            - func_hg
    bounds: tuple of lists
        Parameters boundaries for `func` ([all_mins], [all_maxs]).
        Defaults are given for `func_hg1g2_with_spin`: (H, G1, G2, R, alpha0, delta0).

    Returns
    -------
    outdic: dict
    """
    if p0 is None:
        p0 = [15, 0.15, 0.15]
    if bounds is None:
        bounds = ([-3, 0, 0], [30, 1, 1])

    if model == "HG1G2":
        func = func_hg1g2
        nparams = 3
        params_ = ["H", "G1", "G2"]
        assert len(bounds[0]) == nparams, (
            "You need to specify bounds on all (H, G1, G2) parameters"
        )
    elif model == "HG12":
        func = func_hg12
        nparams = 2
        params_ = ["H", "G12"]
        assert len(bounds[0]) == nparams, (
            "You need to specify bounds on all (H, G12) parameters"
        )
    elif model == "HG":
        func = func_hg
        nparams = 2
        params_ = ["H", "G"]
        assert len(bounds[0]) == nparams, (
            "You need to specify bounds on all (H, G) parameters"
        )

    ufilters = np.unique(filters)

    params = []
    for filt in ufilters:
        tmp = [i + "_{}".format(str(filt)) for i in params_]
        params = np.concatenate((params, tmp))

    initial_guess = []
    for _ in ufilters:
        initial_guess = np.concatenate((initial_guess, p0))

    lower_bounds = []
    upper_bounds = []
    for _ in ufilters:
        lower_bounds = np.concatenate((lower_bounds, bounds[0]))
        upper_bounds = np.concatenate((upper_bounds, bounds[1]))

    if not np.all([i == i for i in magpsf_red]):
        outdic = {"fit": 1, "status": -2}
        return outdic

    try:
        res_lsq = least_squares(
            build_eqs,
            x0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            jac="2-point",
            loss="soft_l1",
            args=(filters, phase, magpsf_red, func),
        )

    except RuntimeError:
        outdic = {"fit": 3, "status": -2}
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
        outdic = {"fit": 4, "status": res_lsq.status}
        return outdic
    # For the chi2, we use the error estimate from the data directly

    sorted_sigmapsf = sort_quantity_by_filter(filters, sigmapsf)

    chisq = np.sum((res_lsq.fun / sorted_sigmapsf) ** 2)
    chisq_red = chisq / (res_lsq.fun.size - res_lsq.x.size - 1)

    outdic = {"chi2red": chisq_red, "status": res_lsq.status, "fit": 0}

    # Total RMS, and per-band
    rms = np.sqrt(np.mean(res_lsq.fun**2))
    outdic["rms"] = rms

    res_lsq_byfilter = split_quantity_by_filter(filters, res_lsq.fun)

    for i, filt in enumerate(ufilters):
        outdic["rms_{}".format(filt)] = np.sqrt(np.mean(res_lsq_byfilter[i] ** 2))

    median_error_phot = np.median(sigmapsf)
    outdic["median_error_phot"] = median_error_phot
    for filt in ufilters:
        mask = filters == filt
        outdic["median_error_phot_{}".format(filt)] = np.median(sigmapsf[mask])

    outdic["n_obs"] = len(phase)
    for filt in ufilters:
        mask = filters == filt
        outdic["n_obs_{}".format(filt)] = len(phase[mask])

    # in degree
    outdic["min_phase"] = np.degrees(np.min(phase))
    for filt in ufilters:
        mask = filters == filt
        outdic["min_phase_{}".format(filt)] = np.degrees(np.min(phase[mask]))

    outdic["max_phase"] = np.degrees(np.max(phase))
    for filt in ufilters:
        mask = filters == filt
        outdic["max_phase_{}".format(filt)] = np.degrees(np.max(phase[mask]))

    for i in range(len(params)):
        outdic[params[i]] = popt[i]
        outdic["err_" + params[i]] = perr[i]

    return outdic


def fit_sfhg1g2(
    ssnamenr,
    magpsf_red,
    sigmapsf,
    jds,
    phase,
    filters,
):
    """Fit for phase curve parameters for sfHG1G2

    Notes
    -----
    Unlike other models, it returns less information, and
    only per-band.

    Parameters
    ----------
    ssnamenr: str
        SSO name/number
    magpsf_red: array
        Reduced magnitude, that is m_obs - 5 * np.log10('Dobs' * 'Dhelio')
    sigmapsf: array
        Error estimates on magpsf_red
    jds: array
        Julian Dates
    phase: array
        Phase angle [rad]
    filters: array
        Filter name for each measurement

    Returns
    -------
    outdic: dict
        Dictionary containing reduced chi2, and estimated parameters and
        error on each parameters.
    """
    # exit if NaN values
    if not np.all([i == i for i in magpsf_red]):
        outdic = {"fit": 1, "status": -2}
        return outdic

    pdf = pd.DataFrame({
        "i:magpsf_red": magpsf_red,
        "i:sigmapsf": sigmapsf,
        "Phase": phase,
        "i:jd": jds,
        "i:fid": filters,
    })
    pdf = pdf.sort_values("i:jd")

    # Get oppositions
    pdf[["elong", "elongFlag"]] = get_opposition(pdf["i:jd"].to_numpy(), ssnamenr)

    # loop over filters
    ufilters = np.unique(pdf["i:fid"].to_numpy())
    outdics = {}
    for filt in ufilters:
        outdic = {}

        # Select data for a filter
        sub = pdf[pdf["i:fid"] == filt].copy()

        # Compute apparitions
        splitted = split_dataframe_per_apparition(sub, "elongFlag", "i:jd")
        napparition = len(splitted)

        # H for all apparitions plus G1, G2
        params_ = ["G1", "G2", *["H{}".format(i) for i in range(napparition)]]
        params = [i + "_{}".format(str(filt)) for i in params_]

        # Split phase
        phase_list = [df["Phase"].to_numpy().tolist() for df in splitted]

        # Fit
        res_lsq = least_squares(
            sfhg1g2_error_fun,
            x0=[0.15, 0.15] + [15] * napparition,
            bounds=(
                [0, 0] + [-3] * napparition,
                [1, 1] + [30] * napparition,
            ),
            loss="huber",
            method="trf",
            args=[
                phase_list,
                sub["i:magpsf_red"].to_numpy().tolist(),
            ],
            xtol=1e-20,
            gtol=1e-17,
        )

        # Result
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
            outdic = {"fit": 4, "status": res_lsq.status}
            return outdic

        chisq = np.sum((res_lsq.fun / sub["i:sigmapsf"]) ** 2)
        chisq_red = chisq / (res_lsq.fun.size - res_lsq.x.size - 1)
        outdic["chi2red_{}".format(filt)] = chisq_red
        outdic["rms_{}".format(filt)] = np.sqrt(np.mean(res_lsq.fun**2))
        outdic["n_obs_{}".format(filt)] = len(sub)
        outdic["n_app_{}".format(filt)] = napparition

        for i in range(len(params)):
            outdic[params[i]] = popt[i]
            outdic["err_" + params[i]] = perr[i]

        outdics.update(outdic)

    # only if both bands converged
    outdics["fit"] = 0

    return outdics


def fit_spin(
    magpsf_red,
    sigmapsf,
    phase,
    ra,
    dec,
    filters,
    terminator=False,
    ra_s=None,
    dec_s=None,
    jd=None,
    p0=None,
    bounds=None,
    model="SHG1G2",
):
    """Fit for phase curve parameters

    SHG1G2: (H^b, G_1^b, G_2^b, alpha, delta, R)
    SSHG1G2: (H^b, G_1^b, G_2^b, alpha, delta, period, a_b, a_c, phi0, t0)

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
    terminator: bool
        Include correction for the non-illuminated part
    ra_s: optional, np.array
        Array of size N containing the solar RA (radian), required if terminator=True
    dec_s: optional, np.array
        Array of size N containing the solar DEC (radian), required if terminator=True
    jd: optional, array
        Observing time (JD), corrected for the light travel. Required for SSHG1G2 model.
    p0: list
        Initial guess for parameters. Note that even if
        there is several bands `b`, we take the same initial guess for all (H^b, G1^b, G2^b).
    bounds: tuple of lists
        Parameters boundaries for `func_hg1g2_with_spin` ([all_mins], [all_maxs]).
        Lists should be ordered as: (H, G1, G2, R, alpha, delta). Note that even if
        there is several bands `b`, we take the same bounds for all (H^b, G1^b, G2^b).

    Returns
    -------
    outdic: dict
        Dictionary containing reduced chi2, and estimated parameters and
        error on each parameters.
    """
    assert model in ["SHG1G2", "SSHG1G2"], model

    if p0 is None:
        if model == "SHG1G2":
            p0 = [15.0, 0.15, 0.15, 0.8, np.pi, 0.0]
        elif model == "SSHG1G2":
            p0 = [15.0, 0.15, 0.15, np.pi, 0.0, 1, 1.05, 1.05, 0.0]

    if bounds is None:
        if model == "SHG1G2":
            bounds = (
                [-3, 0, 0, 3e-1, 0, -np.pi / 2],
                [30, 1, 1, 1, 2 * np.pi, np.pi / 2],
            )
        elif model == "SSHG1G2":
            bounds = (
                [-3, 0, 0, 0, -np.pi / 2, 2.2 / 24.0, 1, 1, -np.pi / 2],
                [30, 1, 1, 2 * np.pi, np.pi / 2, 1000, 5, 5, np.pi / 2],
            )

    ufilters = np.unique(filters)
    if model == "SHG1G2":
        params = ["R", "alpha0", "delta0"]
    elif model == "SSHG1G2":
        params = ["alpha0", "delta0", "period", "a_b", "a_c", "phi0"]

    phase_params = ["H", "G1", "G2"]

    for filt in ufilters:
        phase_params_with_filt = [i + "_{}".format(str(filt)) for i in phase_params]
        params = np.concatenate((params, phase_params_with_filt))

    initial_guess = p0[3:]
    for _ in ufilters:
        initial_guess = np.concatenate((initial_guess, p0[:3]))

    lower_bounds = bounds[0][3:]
    upper_bounds = bounds[1][3:]
    for _ in ufilters:
        lower_bounds = np.concatenate((lower_bounds, bounds[0][:3]))
        upper_bounds = np.concatenate((upper_bounds, bounds[1][:3]))

    if not np.all([i == i for i in magpsf_red]):
        outdic = {"fit": 1, "status": -2}
        return outdic

    try:
        if model == "SHG1G2":
            func = build_eqs_for_spins
            args = (filters, phase, ra, dec, magpsf_red)
        elif model == "SSHG1G2":
            func = build_eqs_for_spin_shape
            if not terminator:
                args = (filters, phase, ra, dec, jd, magpsf_red)
            else:
                args = (
                    filters,
                    phase,
                    ra,
                    dec,
                    jd,
                    magpsf_red,
                    terminator,
                    ra_s,
                    dec_s,
                )

        res_lsq = least_squares(
            func,
            x0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            jac="2-point",
            loss="soft_l1",
            args=args,
        )

    except (RuntimeError, ValueError):
        outdic = {"fit": 3, "status": -2}
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
        outdic = {"fit": 4, "status": res_lsq.status}
        return outdic

    # For the chi2, we use the error estimate from the data directly
    sorted_sigmapsf = sort_quantity_by_filter(filters, sigmapsf)

    chisq = np.sum((res_lsq.fun / sorted_sigmapsf) ** 2)
    chisq_red = chisq / (res_lsq.fun.size - res_lsq.x.size - 1)

    geo = cos_aspect_angle(
        ra,
        dec,
        popt[params.tolist().index("alpha0")],
        popt[params.tolist().index("delta0")],
    )
    outdic = {
        "chi2red": chisq_red,
        "min_cos_lambda": np.min(np.abs(geo)),
        "mean_cos_lambda": np.mean(np.abs(geo)),
        "max_cos_lambda": np.max(np.abs(geo)),
        "status": res_lsq.status,
        "fit": 0,
    }

    # Total RMS, and per-band
    rms = np.sqrt(np.mean(res_lsq.fun**2))
    outdic["rms"] = rms

    res_lsq_byfilter = split_quantity_by_filter(filters, res_lsq.fun)

    for i, filt in enumerate(ufilters):
        outdic["rms_{}".format(filt)] = np.sqrt(np.mean(res_lsq_byfilter[i] ** 2))

    median_error_phot = np.median(sigmapsf)
    outdic["median_error_phot"] = median_error_phot
    for filt in ufilters:
        mask = filters == filt
        outdic["median_error_phot_{}".format(filt)] = np.median(sigmapsf[mask])

    outdic["n_obs"] = len(phase)
    for filt in ufilters:
        mask = filters == filt
        outdic["n_obs_{}".format(filt)] = len(phase[mask])

    # in degrees
    outdic["min_phase"] = np.degrees(np.min(phase))
    for filt in ufilters:
        mask = filters == filt
        outdic["min_phase_{}".format(filt)] = np.degrees(np.min(phase[mask]))

    outdic["max_phase"] = np.degrees(np.max(phase))
    for filt in ufilters:
        mask = filters == filt
        outdic["max_phase_{}".format(filt)] = np.degrees(np.max(phase[mask]))

    for i, name in enumerate(params):
        if name in ["alpha0", "delta0"]:
            # convert in degrees
            outdic[params[i]] = np.degrees(popt[i])
            outdic["err_" + params[i]] = np.degrees(perr[i])
        else:
            outdic[params[i]] = popt[i]
            outdic["err_" + params[i]] = perr[i]

    if "R" in outdic:
        # SHG1G2
        a_b, a_c = estimate_axes_ratio(res_lsq.fun, outdic["R"])
        outdic["a_b"] = a_b
        outdic["a_c"] = a_c

    return outdic


def extract_obliquity(sso_name, alpha0, delta0):
    """Extract obliquity using spin values, and the BFT information

    Parameters
    ----------
    sso_name: np.array or pd.Series of str
        SSO names according to quaero (see `rockify`)
    alpha0: np.array or pd.Series of double
        RA of the pole [degree]
    delta0: np.array or pd.Series of double
        DEC of the pole [degree]

    Returns
    -------
    obliquity: np.array of double
        Obliquity for each object [degree]
    """
    import rocks

    cols = [
        "sso_name",
        "orbital_elements.node_longitude.value",
        "orbital_elements.inclination.value",
    ]
    pdf_bft = rocks.load_bft(columns=cols)

    sub = pdf_bft[cols]

    pdf = pd.DataFrame({"sso_name": sso_name, "alpha0": alpha0, "delta0": delta0})

    pdf = pdf.merge(sub[cols], left_on="sso_name", right_on="sso_name", how="left")

    # Orbit
    lon_orbit = (pdf["orbital_elements.node_longitude.value"] - 90).to_numpy()
    lat_orbit = (90.0 - pdf["orbital_elements.inclination.value"]).to_numpy()

    # Spin -- convert to EC
    ra = np.nan_to_num(pdf.alpha0.to_numpy()) * u.degree
    dec = np.nan_to_num(pdf.delta0.to_numpy()) * u.degree

    # Trick to put the object "far enough"
    coords_spin = SkyCoord(ra=ra, dec=dec, distance=200 * u.parsec, frame="hcrs")

    # in radian
    lon_spin = coords_spin.heliocentricmeanecliptic.lon.value
    lat_spin = coords_spin.heliocentricmeanecliptic.lat.value

    obliquity = np.degrees(
        angular_separation(
            np.radians(lon_spin),
            np.radians(lat_spin),
            np.radians(lon_orbit),
            np.radians(lat_orbit),
        )
    )

    return obliquity


def angular_separation(lon1, lat1, lon2, lat2):
    """Angular separation between two points on a sphere.

    Notes
    -----
    Stolen from astropy -- for version <5

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : `~astropy.coordinates.Angle`, `~astropy.units.Quantity` or float
        Longitude and latitude of the two points. Quantities should be in
        angular units; floats in radians.

    Returns
    -------
    angular separation : `~astropy.units.Quantity` ['angle'] or float
        Type depends on input; ``Quantity`` in angular units, or float in
        radians.

    Notes
    -----
    The angular separation is calculated using the Vincenty formula [1]_,
    which is slightly more complex and computationally expensive than
    some alternatives, but is stable at at all distances, including the
    poles and antipodes.

    .. [1] https://en.wikipedia.org/wiki/Great-circle_distance
    """
    sdlon = np.sin(lon2 - lon1)
    cdlon = np.cos(lon2 - lon1)
    slat1 = np.sin(lat1)
    slat2 = np.sin(lat2)
    clat1 = np.cos(lat1)
    clat2 = np.cos(lat2)

    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon
    denominator = slat1 * slat2 + clat1 * clat2 * cdlon

    return np.arctan2(np.hypot(num1, num2), denominator)


def angle_between_vectors(v1, v2):
    """Compute the angle between two 3D vectors.

    Parameters
    ----------
    v1 : list or np.ndarray
        The first 3D vector.
    v2 : list or np.ndarray
        The second 3D vector.

    Returns
    -------
    float
        The angle between the two vectors in radians.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    cos_theta = dot_product / (norm_v1 * norm_v2)

    # Clip to handle numerical issues
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return angle


if __name__ == "__main__":
    """Execute the unit test suite"""

    # Run the Spark test suite
    regular_unit_tests(globals())

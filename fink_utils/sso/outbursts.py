# Copyright 2022 AstroLab Software
# Authors: Julien Peloton, Roman Le Montagner, Benoit Carry
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
import rocks
import requests
import pandas as pd
import numpy as np
import io

from sbpy.photometry import HG1G2
from sbpy.data import Obs

from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

from scipy.optimize import curve_fit


def get_sso_fink(ssname: str, withEphem: bool = True, withComplement=True):
    """ Fetch data for `ssname` from the Fink API.

    Parameters
    ----------
    ssname: str | int
        MPC designation. Taken from `ssnamenr` in the alert packet.
    withEphem: bool, optional
        If True, query Miriade for ephemerides. Default is True.
    withComplement: bool, optional
        If True, query for extra fields that are not available from
        the SSO table. Adds processing time delay. Default is False.

    Returns
    ----------
    pdf_sso: pd.DataFrame
        Pandas DataFrame containing data for all observations in Fink
    """
    r = requests.post(
        "https://fink-portal.org/api/v1/sso",
        json={
            "n_or_d": ssname,
            "withEphem": withEphem,
            "columns": "i:magpsf,i:sigmapsf,i:fid,i:jd,i:ssnamenr,i:ra,i:dec",
            "output-format": "json",
        },
    )

    # Format output in a DataFrame
    pdf_sso = pd.read_json(io.BytesIO(r.content))

    if withComplement:
        l1 = []
        l2 = []
        for index, oid in enumerate(pdf_sso["i:objectId"].values):

            r = requests.post(
                "https://fink-portal.org/api/v1/objects",
                json={"objectId": oid, "columns": "i:bimagerat,i:aimagerat",},
            )

            tmp = pd.read_json(io.BytesIO(r.content))
            l1.append(tmp["i:aimagerat"].values[0])
            l2.append(tmp["i:bimagerat"].values[0])

        pdf_sso["i:aimagerat"] = l1
        pdf_sso["i:bimagerat"] = l2

    return pdf_sso


def get_H_from_ephem(ssname: str):
    """ Retrieve H from rocks for a given `ssname`
    """
    sso = rocks.Rock(ssname)
    H_ephem = sso.parameters.physical.absolute_magnitude.value

    return H_ephem


def get_scale(H_obs: list, H_ephem: float, pos: int):
    """ Get rescaling by H of the lightcurve `H_obs - H_ephem`

    Parameters
    ----------
    H_obs: list
        Absolute magnitude in each observed filter band
    H_ephem: float
        Absolute magnitude returned by the ephemeride service
    pos: int
        Index for the filter band
    """
    return H_obs[pos] - H_ephem


def get_gault_activation(filename):
    """ Read data from `Multiple Outbursts of Asteroid (6478) Gault`
    Original data in https://github.com/Yeqzids/activation_of_6478_gault

    Parameters
    ----------
    filename: str
        Path to modified CSV

    Returns
    ----------
    pdf_hist_gault: pd.DataFrame
        Data as Pandas DataFrame
    """
    pdf_hist_gault = pd.read_csv(filename, sep="\s+", header=0)
    pdf_hist_gault["jd"] = Time(
        [
            "{} {}".format(i, j)
            for i, j in zip(pdf_hist_gault["night"], pdf_hist_gault["date"])
        ]
    ).jd
    return pdf_hist_gault


def get_full_gault(pdf_fink, pdf_activation):
    """ Concatenate Gault data from Fink & others

    Parameters
    ----------
    pdf_fink: pd.DataFrame
        Pandas DataFrame with Fink alert data
    pdf_activation: pd.DataFrame
        Pandas DataFrame with Gault data from 2018/2019

    Returns
    ----------
    pdf_obs: pd.DataFrame
        Pandas DataFrame with concatenated data
    pdf_ephem_sub: pd.DataFrame
        Pandas DataFrame with ephemerides corresponding to `pdf_obs`.
    """
    conv = {"g": 1, "r": 2}
    mag_obs = np.concatenate((pdf_activation["m"], pdf_fink["i:magpsf"]))
    jds = np.concatenate((pdf_activation["jd"], pdf_fink["i:jd"]))
    filters_ = np.concatenate(
        (pdf_activation["filter"].apply(lambda x: conv[x]), pdf_fink["i:fid"])
    )

    pdf_obs = pd.DataFrame(
        {
            "i:magpsf": mag_obs,
            "i:fid": filters_,
            "i:jd": jds,
            "i:ssnamenr": pdf_fink["i:ssnamenr"].values[0],
        }
    )

    pdf_ephem_sub = get_miriade_data(pdf_obs)

    return pdf_obs, pdf_ephem_sub


def compute_residual_hg(
    pdf, gaussfit="", compute_ephem=False, observer="I41", rplane="1", tcoor=5
):
    """ Compute the residual between observations and ephemerides using (H, G) model

    Parameters
    ----------
    pdf: pd.DataFrame
        Pandas DataFrame with Fink data. Optionaly contains ephemerides data.
    gaussfit: str, optional
        Choose between '1comp', '2comp' (not available), or '' (no fit)
    compute_ephem: bool, optional
        If True, compute ephemerides. Default is False, namely ephemerides are
        already contained in `pdf`.
    observer: str, optional
        IAU Obs code - default to ZTF
        https://minorplanetcenter.net//iau/lists/ObsCodesF.html
    rplane: str, optional
        Reference plane: equator ('1'), ecliptic ('2').
        If rplane = '2', then tcoor is automatically set to 1 (spherical)
    tcoor: int, optional
        See https://ssp.imcce.fr/webservices/miriade/api/ephemcc/
        Default is 5 (dedicated to observation)

    Returns
    ---------
    residual: array-like
        Difference between observations and ephemerides.
    amplitude: float
        If gaussfit is not False, returns the `amplitude` of the fitted Gaussian.
        Otherwise None.
    mu: float
        If gaussfit is not False, returns the `mu` of the fitted Gaussian.
        Otherwise None.
    sigma: float
        If gaussfit is not False, returns the `sigma` of the fitted Gaussian.
        Otherwise None.
    """
    if compute_ephem is True:
        # Compute ephemerides using Miriade
        pdf_ephem = query_miriade(
            str(pdf["i:ssnamenr"].values[0]),
            pdf["i:jd"].values,
            observer=observer,
            rplane=rplane,
            tcoor=tcoor,
        )
    else:
        pdf_ephem = pdf

    residual = (pdf["i:magpsf"] + pdf["color_corr"] - pdf_ephem["VMag"]).values
    amplitude, mu, sigma = fit_residual(residual, gaussfit=gaussfit)

    return residual, amplitude, mu, sigma


def compute_residual_hg1g2(
    pdf, gaussfit="", compute_ephem=False, observer="I41", rplane="1", tcoor=5
):
    """ Compute the residual between observations and the (H, G1, G2) model

    Parameters
    ----------
    pdf: pd.DataFrame
        Pandas DataFrame with Fink data. Optionaly contains ephemerides data.
    gaussfit: str, optional
        Choose between '1comp', '2comp' (not available), or '' (default, no fit)
    compute_ephem: bool, optional
        If True, compute ephemerides. Default is False, namely ephemerides are
        already contained in `pdf`.
    observer: str, optional
        IAU Obs code - default to ZTF
        https://minorplanetcenter.net//iau/lists/ObsCodesF.html
    rplane: str, optional
        Reference plane: equator ('1'), ecliptic ('2').
        If rplane = '2', then tcoor is automatically set to 1 (spherical)
    tcoor: int, optional
        See https://ssp.imcce.fr/webservices/miriade/api/ephemcc/
        Default is 5 (dedicated to observation)

    Returns
    ---------
    residual: array-like
        Difference between observations and ephemerides.
    amplitude: float
        If gaussfit is not False, returns the `amplitude` of the fitted Gaussian.
        Otherwise None.
    mu: float
        If gaussfit is not False, returns the `mu` of the fitted Gaussian.
        Otherwise None.
    sigma: float
        If gaussfit is not False, returns the `sigma` of the fitted Gaussian.
        Otherwise None.
    """
    if compute_ephem is True:
        # Compute ephemerides using Miriade
        pdf_ephem = query_miriade(
            str(pdf["i:ssnamenr"].values[0]),
            pdf["i:jd"].values,
            observer=observer,
            rplane=rplane,
            tcoor=tcoor,
        )
    else:
        pdf_ephem = pdf

    ydata = pdf_ephem["i:magpsf_red"] + pdf["color_corr"]

    # Values in radians
    alpha = np.deg2rad(pdf_ephem["Phase"].values)

    try:
        popt, pcov = curve_fit(
            func_hg1g2,
            alpha,
            ydata.values,
            sigma=pdf["i:sigmapsf"],
            bounds=([0, 0, 0], [30, 1, 1]),
        )

        residual = (
            ydata.values
            - HG1G2(popt[0] * u.mag, popt[1], popt[2]).to_mag(alpha * u.rad).value
        )
        amplitude, mu, sigma = fit_residual(residual, gaussfit=gaussfit)

    except RuntimeError as e:
        residual, amplitude, mu, sigma = None, None, None, None

    return residual, amplitude, mu, sigma


def compute_residual_hg1g2re(
    pdf,
    bounds=([0, 0, 0, 1e-2, 0, -np.pi / 2], [30, 1, 1, 1, 2 * np.pi, np.pi / 2]),
    gaussfit="",
    compute_ephem=False,
    observer="I41",
    rplane="1",
    tcoor=5,
):
    """ Compute the residual between observations and the (H, G1, G2, R, l0, b0) model

    Parameters
    ----------
    pdf: pd.DataFrame
        Pandas DataFrame with Fink data. Optionaly contains ephemerides data.
    gaussfit: str, optional
        Choose between '1comp', '2comp' (not available), or '' (default, no fit)
    compute_ephem: bool, optional
        If True, compute ephemerides. Default is False, namely ephemerides are
        already contained in `pdf`.
    observer: str, optional
        IAU Obs code - default to ZTF
        https://minorplanetcenter.net//iau/lists/ObsCodesF.html
    rplane: str, optional
        Reference plane: equator ('1'), ecliptic ('2').
        If rplane = '2', then tcoor is automatically set to 1 (spherical)
    tcoor: int, optional
        See https://ssp.imcce.fr/webservices/miriade/api/ephemcc/
        Default is 5 (dedicated to observation)

    Returns
    ---------
    residual: array-like
        Difference between observations and ephemerides.
    amplitude: float
        If gaussfit is not False, returns the `amplitude` of the fitted Gaussian.
        Otherwise None.
    mu: float
        If gaussfit is not False, returns the `mu` of the fitted Gaussian.
        Otherwise None.
    sigma: float
        If gaussfit is not False, returns the `sigma` of the fitted Gaussian.
        Otherwise None.
    """
    if compute_ephem is True:
        # Compute ephemerides using Miriade
        pdf_ephem = query_miriade(
            str(pdf["i:ssnamenr"].values[0]),
            pdf["i:jd"].values,
            observer=observer,
            rplane=rplane,
            tcoor=tcoor,
        )
    else:
        pdf_ephem = pdf

    ydata = pdf_ephem["i:magpsf_red"] + pdf["color_corr"]

    # Values in radians
    alpha = np.deg2rad(pdf_ephem["Phase"].values)
    ra = np.deg2rad(pdf["i:ra"].values)
    dec = np.deg2rad(pdf["i:dec"].values)
    pha = np.transpose([[i, j, k] for i, j, k in zip(alpha, ra, dec)])

    try:
        popt, pcov = curve_fit(
            func_hg1g2_with_spin,
            pha,
            ydata.values,
            sigma=pdf["i:sigmapsf"],
            bounds=bounds,
            jac=Dfunc_hg1g2_with_spin,
        )

        residual = ydata.values - func_hg1g2_with_spin(pha, *popt)
        amplitude, mu, sigma = fit_residual(residual, gaussfit=gaussfit)

    except RuntimeError as e:
        print(e)
        residual, amplitude, mu, sigma = None, None, None, None

    return residual, amplitude, mu, sigma


def gaussian(x, amplitude, mu, sigma):
    """ 1D Gaussian function

    Parameters
    ----------
    x: array-like
    """
    arg = (x - mu) / sigma
    return amplitude * np.exp(-0.5 * arg ** 2)


def fit_residual(residual, gaussfit="", bins=20):
    """ Adjust 1D Gaussian to residuals

    Parameters
    ----------
    residual: array-like
        Residuals
    gaussfit: str, optional
        Choose between '1comp', '2comp' (not available), or '' (default, no fit)

    Returns
    ----------
    amplitude: float
    mu: float
    sigma: float
    """
    if gaussfit == "1comp":
        # fit for 1D gaussian
        hist, bin_borders = np.histogram(residual, bins=bins)
        x = bin_borders[:-1] + np.diff(bin_borders) / 2
        try:
            popt, pcov = curve_fit(
                gaussian, x, hist, bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf])
            )
            amplitude, mu, sigma = popt
        except RuntimeError as e:
            print("Gauss fit failed", e)
            amplitude, mu, sigma = None, None, None
    elif gaussfit == "2comp":
        # fit for 2 1D gaussians
        pass
    else:
        amplitude, mu, sigma = None, None, None

    return amplitude, mu, sigma


def compute_spin_distance(spin1, spin2):
    """ Compute the angular distance between two spins

    Parameters
    ----------
    spin1: list
        [RA, Dec] in degrees
    spin2: list
        [RA, Dec] in degrees

    Returns
    ----------
    distance: float
        Distance on the sphere in degrees between the two spins
    """
    distance = SkyCoord(spin1[0], spin1[1], unit="deg").separation(
        SkyCoord(spin2[0], spin2[1], unit="deg")
    )
    return distance


def plot_mwd(
    RA,
    Dec,
    err_ra,
    err_dec,
    color,
    scatter_color,
    ax,
    fig,
    withcb=False,
    cb_title="",
    cmap="viridis",
    alpha=0.5,
    org=0,
    label="",
    projection="mollweide",
):
    """ Project data on the 2D sphere

    Parameters
    ----------
    RA: array-like
        RA in degrees between [0, 360).
    Dec: array-like
        Dec in degrees between [-90, 90]
    err_ra, err_dec: array-like
        errors on RA, Dec

    org is the origin of the plot, 0 or a multiple of 30 degrees in [0,360).
    projection is the kind of projection: 'mollweide', 'aitoff', 'hammer', 'lambert'
    """
    # shift RA values
    x = np.remainder(RA + 360 - org, 360)

    # scale conversion to [-180, 180]
    ind = x > 180
    x[ind] -= 360

    # reverse the scale: East to the left
    x = -x

    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels + 360 + org, 360)

    # Plot error bars first
    scatter_kwargs = {"zorder": 100}
    error_kwargs = {"lw": 0.5, "zorder": 0}
    ax.errorbar(
        np.radians(x),
        np.radians(Dec),
        xerr=np.radians(err_ra),
        yerr=np.radians(err_dec),
        color=color,
        alpha=alpha,
        marker="o",
        ls="",
        **error_kwargs
    )

    # Spins
    cm = ax.scatter(
        np.radians(x),
        np.radians(Dec),
        c=scatter_color,
        alpha=alpha,
        marker="o",
        cmap=cmap,
        label=label,
        **scatter_kwargs
    )

    cb = fig.colorbar(cm)
    cb.set_label(cb_title)
    if not withcb:
        cb.remove()

    tick_labels = np.array(["", "120", "", "60", "", "0", "", "300", "", "240", ""])
    ax.set_xticklabels(tick_labels)
    ax.title.set_fontsize(15)
    ax.set_xlabel("RA")
    ax.xaxis.label.set_fontsize(12)
    ax.set_ylabel("Dec")
    ax.yaxis.label.set_fontsize(12)
    ax.grid(True)

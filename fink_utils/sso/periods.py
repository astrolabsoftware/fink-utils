# Copyright 2024 AstroLab Software
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
"""Utils for fitting asteroid synodic period of rotation"""

from fink_utils.sso.spins import (
    estimate_sso_params,
    func_sshg1g2,
    func_hg1g2_with_spin,
    func_hg1g2,
    func_hg12,
    func_hg,
)
from fink_utils.sso.utils import compute_light_travel_correction
from astropy.timeseries import LombScargleMultiband

import requests
import numpy as np
import io
import pandas as pd
from line_profiler import profile
import nifty_ls  # noqa: F401

import logging

from fink_utils.test.tester import regular_unit_tests

_LOG = logging.getLogger(__name__)


def extract_physical_parameters(pdf, flavor):
    """Fit for phase curve parameters

    Parameters
    ----------
    pdf: pandas DataFrame
        DataFrame with SSO data from Fink REST API
    flavor: str
        Model flavor: SHG1G2, HG1G2, HG12, or HG

    Returns
    -------
    outdic: dict
        Dictionary containing reduced chi2, and estimated parameters and
        error on each parameters.
    """
    if flavor == "HG1G2":
        bounds = (
            [-3, 0, 0],
            [30, 1, 1],
        )
        p0 = [15.0, 0.15, 0.15]
    elif flavor == "HG12":
        bounds = (
            [-3, 0],
            [30, 1],
        )
        p0 = [15.0, 0.15]
    elif flavor == "HG":
        bounds = (
            [-3, 0],
            [30, 1],
        )
        p0 = [15.0, 0.15]
    elif flavor == "SHG1G2":
        bounds = (
            [-3, 0, 0, 3e-1, 0, -np.pi / 2],
            [30, 1, 1, 1, 2 * np.pi, np.pi / 2],
        )
        p0 = [15.0, 0.15, 0.15, 0.8, np.pi, 0.0]

    outdic = estimate_sso_params(
        magpsf_red=pdf["i:magpsf_red"].to_numpy(),
        sigmapsf=pdf["i:sigmapsf"].to_numpy(),
        phase=np.deg2rad(pdf["Phase"].to_numpy()),
        filters=pdf["i:fid"].to_numpy(),
        ra=np.deg2rad(pdf["i:ra"].to_numpy()),
        dec=np.deg2rad(pdf["i:dec"].to_numpy()),
        p0=p0,
        bounds=bounds,
        model=flavor,
        normalise_to_V=False,
    )

    return outdic


def compute_residuals(pdf, flavor, phyparam):
    """Compute residuals between data and predictions

    Parameters
    ----------
    pdf: pandas DataFrame
        DataFrame with SSO data from Fink REST API
    flavor: str
        Model flavor: SHG1G2, HG1G2, HG12, or HG
    phyparam: dict
        Dictionary containing reduced chi2, and estimated parameters and
        error on each parameters.

    Returns
    -------
    pd.Series
        Series containing `observation - model` in magnitude
    """
    pdf["preds"] = 0.0
    for filtnum in pdf["i:fid"].unique():
        if filtnum == 3:
            continue
        cond = pdf["i:fid"] == filtnum

        if flavor == "SSHG1G2":
            pha = [
                np.deg2rad(pdf["Phase"][cond]),
                np.deg2rad(pdf["i:ra"][cond]),
                np.deg2rad(pdf["i:dec"][cond]),
            ]
            preds = func_sshg1g2(
                pha,
                phyparam["H_{}".format(filtnum)],
                phyparam["G1_{}".format(filtnum)],
                phyparam["G2_{}".format(filtnum)],
                np.deg2rad(phyparam["alpha0"]),
                np.deg2rad(phyparam["delta0"]),
                phyparam["period"],
                phyparam["a_b"],
                phyparam["a_c"],
                phyparam["phi0"],
            )
        elif flavor == "SHG1G2":
            pha = [
                np.deg2rad(pdf["Phase"][cond]),
                np.deg2rad(pdf["i:ra"][cond]),
                np.deg2rad(pdf["i:dec"][cond]),
            ]
            preds = func_hg1g2_with_spin(
                pha,
                phyparam["H_{}".format(filtnum)],
                phyparam["G1_{}".format(filtnum)],
                phyparam["G2_{}".format(filtnum)],
                phyparam["R"],
                np.deg2rad(phyparam["alpha0"]),
                np.deg2rad(phyparam["delta0"]),
            )
        elif flavor == "HG":
            preds = func_hg(
                np.deg2rad(pdf["Phase"][cond]),
                phyparam["H_{}".format(filtnum)],
                phyparam["G_{}".format(filtnum)],
            )
        elif flavor == "HG12":
            preds = func_hg12(
                np.deg2rad(pdf["Phase"][cond]),
                phyparam["H_{}".format(filtnum)],
                phyparam["G12_{}".format(filtnum)],
            )
        elif flavor == "HG1G2":
            preds = func_hg1g2(
                np.deg2rad(pdf["Phase"][cond]),
                phyparam["H_{}".format(filtnum)],
                phyparam["G1_{}".format(filtnum)],
                phyparam["G2_{}".format(filtnum)],
            )
        pdf.loc[cond, "preds"] = preds

    return pdf["i:magpsf_red"] - pdf["preds"]


@profile
def estimate_synodic_period(
    ssnamenr: str = None,
    pdf=None,
    phyparam=None,
    flavor="SHG1G2",
    Nterms_base=1,
    Nterms_band=1,
    period_range=(0.05, 1.2),
    sb_method="auto",
    return_extra_info=False,
    lt_correction=True,
):
    """Estimate the synodic period of a Solar System object seen by Fink

    Parameters
    ----------
    ssnamenr: str
        SSO number (we do not resolve name yet)
    pdf: pandas DataFrame, optional
        Pandas DataFrame with Fink SSO data for one object.
        If not specified, data will be downloaded from Fink servers
        using `ssnamenr`.
    phyparam: dict, optional
        Dictionary containing physical properties (phase curve, etc.)
        of the object. If not specified, they will be recomputed
        from the data.
    flavor: str, optional
        Model flavor: SHG1G2 (default), HG1G2, HG12, or HG
    Nterms_base: int, optional
        Number of frequency terms to use for the
        base model common to all bands. Default is 1.
    Nterms_band: int, optional
        Number of frequency terms to use for the
        residuals between the base model and
        each individual band. Default is 1.
    period_range: tupe of float, optional
        (min_period, max_period) for the search, in days.
        Default is (0.05, 1.2), that is between
        1.2 hours and 28.8 hours.
    sb_method: str, optional
        Specify the single-band lomb scargle implementation to use.
        See https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargleMultiband.html#astropy.timeseries.LombScargleMultiband.autopower
        If nifty-ls is installed, one can also specify fastnifty. Although
        in this case it does not work yet for Nterms_* higher than 1.
    return_extra_info: bool, optional
        If True, returns also the fitted model, and the original
        SSO data used for the fit. Default is False.
    lt_correction: bool, optional
        Apply light travel correction. Default is True.

    Returns
    -------
    best_period: float
        Best period found, in hour
    reduced_chi2: float
        Reduced chi2 (chi2 per ddof)
    model: object, optional
        If `return_extra_info`, the fitted model is returned
    pdf: pandas DataFrame, optional
        If `return_extra_info`, the original data is returned.
        Note that the extra field "residuals" contains the
        `observation - model` data, in magnitude.

    Examples
    --------
    >>> ssnamenr = 2363
    >>> P, chi2 = estimate_synodic_period(ssnamenr, flavor="SHG1G2", Nterms_base=2)
    >>> assert int(P) == 20, P

    >>> P_HG, chi2_HG = estimate_synodic_period(ssnamenr, flavor="HG", Nterms_base=2)
    >>> assert chi2 < chi2_HG, (chi2, chi2_HG)

    # by default we apply the light travel correction. Disable it.
    >>> P_no_lt, chi2_no_lt = estimate_synodic_period(ssnamenr, flavor="SHG1G2", Nterms_base=2, lt_correction=False)
    >>> assert chi2 < chi2_no_lt, (chi2, chi2_no_lt)


    One can also use the nifty-ls implementation (faster and more accurate)
    # TODO: check alias between astropy and nifty-ls...
    >>> P_nifty, _ = estimate_synodic_period(ssnamenr, flavor="SHG1G2", sb_method="fastnifty")
    >>> p1 = np.isclose(P, P_nifty, rtol=1e-1)
    >>> p2 = np.isclose(P, 2 * P_nifty, rtol=1e-1)
    >>> assert p1 or p2, (P, P_nifty)

    One can also directly specify the Pandas dataframe with Fink data:
    # TODO: check alias between astropy and nifty-ls...
    >>> r = requests.post("https://api.fink-portal.org/api/v1/sso", json={"n_or_d": ssnamenr, "withEphem": True, "output-format": "json"})
    >>> pdf = pd.read_json(io.BytesIO(r.content))
    >>> P_from_pdf, _ = estimate_synodic_period(pdf=pdf, flavor="SHG1G2")
    >>> p1 = np.isclose(P, P_from_pdf, rtol=1e-1)
    >>> p2 = np.isclose(P, 2 * P_from_pdf, rtol=1e-1)
    >>> assert p1 or p2, (P, P_from_pdf)
    """
    if pdf is None:
        if ssnamenr is not None:
            # TODO: use quaero
            r = requests.post(
                "https://api.fink-portal.org/api/v1/sso",
                json={"n_or_d": ssnamenr, "withEphem": True, "output-format": "json"},
            )
        else:
            _LOG.error("You need to specify either `ssnamenr` or `pdf`.")

        pdf = pd.read_json(io.BytesIO(r.content))

    if phyparam is None:
        # get the physical parameters with the latest data
        phyparam = extract_physical_parameters(pdf, flavor)

    # Compute the residuals (obs - model)
    residuals = compute_residuals(pdf, flavor, phyparam)

    if lt_correction:
        # Speed of light in AU/day
        time = compute_light_travel_correction(pdf["i:jd"], pdf["Dobs"])
    else:
        time = pdf["i:jd"]

    model = LombScargleMultiband(
        time,
        residuals,
        pdf["i:fid"],
        pdf["i:sigmapsf"],
        nterms_base=Nterms_base,
        nterms_band=Nterms_band,
    )

    frequency, power = model.autopower(
        method="fast",
        sb_method=sb_method,
        minimum_frequency=1 / period_range[1],
        maximum_frequency=1 / period_range[0],
    )
    freq_maxpower = frequency[np.argmax(power)]
    # Rotation in days (2* LS value: double-peaked lightcurve)
    # TODO: I need to be convinced...
    best_period = 2 / freq_maxpower

    out = model.model(time.to_numpy(), freq_maxpower)
    prediction = np.zeros_like(residuals)
    for index, filt in enumerate(pdf["i:fid"].unique()):
        if filt == 3:
            continue
        cond = pdf["i:fid"] == filt
        prediction[cond] = out[index][cond]

    chi2 = np.sum(((residuals - prediction) / pdf["i:sigmapsf"].to_numpy()) ** 2)
    reduced_chi2 = chi2 / len(residuals - 1)

    if return_extra_info:
        pdf["residuals"] = residuals
        return best_period * 24, reduced_chi2, frequency, power, model, pdf
    else:
        return best_period * 24, reduced_chi2


if __name__ == "__main__":
    """Execute the unit test suite"""

    # Run the Spark test suite
    regular_unit_tests(globals())

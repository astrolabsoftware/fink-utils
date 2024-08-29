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
    func_hg1g2_with_spin,
    func_hg1g2,
    func_hg12,
    func_hg,
)
from gatspy import periodic
import requests
import numpy as np
import io
import pandas as pd
from line_profiler import profile

from fink_utils.test.tester import regular_unit_tests


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
            [0, 0, 0],
            [30, 1, 1],
        )
        p0 = [15.0, 0.15, 0.15]
    elif flavor == "HG12":
        bounds = (
            [0, 0],
            [30, 1],
        )
        p0 = [15.0, 0.15]
    elif flavor == "HG":
        bounds = (
            [0, 0],
            [30, 1],
        )
        p0 = [15.0, 0.15]
    elif flavor == "SHG1G2":
        bounds = (
            [0, 0, 0, 3e-1, 0, -np.pi / 2],
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
        cond = pdf["i:fid"] == filtnum

        if flavor == "SHG1G2":
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
def extract_period_from_number(
    ssnamenr: str,
    flavor="SHG1G2",
    Nterms_base=1,
    Nterms_band=1,
    period_range=(0.05, 1.2),
    return_extra_info=False,
):
    """Extract the period of a Solar System objeect seen by Fink

    Parameters
    ----------
    ssnamenr: str
        SSO number (we do not resolve name yet)
    flavor: str
        Model flavor: SHG1G2, HG1G2, HG12, or HG
    Nterms_base: int
        Number of frequency terms to use for the
        base model common to all bands. Default is 1.
    Nterms_band: int
        Number of frequency terms to use for the
        residuals between the base model and
        each individual band. Default is 1.
    period_range: tupe of float
        (min_period, max_period) for the search, in days.
        Default is (0.05, 1.2), that is between
        1.2 hours and 28.8 hours.
    return_extra_info: bool
        If True, returns also the fitted model, and the original
        SSO data used for the fit. Default is False.

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
    >>> P, chi2 = extract_period_from_number(ssnamenr, flavor="SHG1G2", Nterms_base=2)
    >>> assert int(P) == 20, P

    >>> P_HG, chi2_HG = extract_period_from_number(ssnamenr, flavor="HG", Nterms_base=2)
    >>> assert chi2 < chi2_HG, (chi2, chi2_HG)
    """
    # TODO: use quaero
    r = requests.post(
        "https://fink-portal.org/api/v1/sso",
        json={"n_or_d": ssnamenr, "withEphem": True, "output-format": "json"},
    )

    pdf = pd.read_json(io.BytesIO(r.content))

    # get the physical parameters with the latest data
    phyparam = extract_physical_parameters(pdf, flavor)

    # Compute the residuals (obs - model)
    residuals = compute_residuals(pdf, flavor, phyparam)

    # fit model
    model = fit_model(
        jd=pdf["i:jd"],
        residuals=residuals,
        sigmapsf=pdf["i:sigmapsf"],
        fid=pdf["i:fid"],
        Nterms_base=Nterms_base,
        Nterms_band=Nterms_band,
        period_range=period_range,
    )

    prediction = model.predict(
        pdf["i:jd"].to_numpy(), period=model.best_period, filts=pdf["i:fid"].to_numpy()
    )
    chi2 = np.sum(((residuals - prediction) / pdf["i:sigmapsf"].to_numpy()) ** 2)
    reduced_chi2 = chi2 / len(residuals - 1)

    if return_extra_info:
        pdf["residuals"] = residuals
        return model.best_period * 24, reduced_chi2, model, pdf
    else:
        return model.best_period * 24, reduced_chi2


def fit_model(
    jd, residuals, sigmapsf, fid, Nterms_base=1, Nterms_band=1, period_range=(0.05, 1.2)
):
    """Fit the Multiband Periodogram model to the data.

    Parameters
    ----------
    jd: array-like of float
        Times
    residuals: array-like of float
        Difference observation - model
    fid: array-like of int
        Filter ID for each measurement
    sigmapsf: array-like of float
        Error estimates on `residuals`
    Nterms_base: int
        Number of frequency terms to use for the
        base model common to all bands. Default is 1.
    Nterms_band: int
        Number of frequency terms to use for the
        residuals between the base model and
        each individual band. Default is 1.
    period_range: tupe of float
        (min_period, max_period) for the search, in days.
        Default is (0.05, 1.2), that is between
        1.2 hours and 28.8 hours.

    Returns
    -------
    model: object
        LombScargleMultiband model
    """
    model = periodic.LombScargleMultiband(
        Nterms_base=Nterms_base,
        Nterms_band=Nterms_band,
        fit_period=True,
    )

    # Not sure about that...
    model.optimizer.period_range = period_range
    model.optimizer.quiet = True

    model.fit(
        jd,
        residuals,
        sigmapsf,
        fid,
    )

    return model


if __name__ == "__main__":
    """Execute the unit test suite"""

    # Run the Spark test suite
    regular_unit_tests(globals())

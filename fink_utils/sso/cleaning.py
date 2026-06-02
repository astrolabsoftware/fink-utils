# Copyright 2026 AstroLab Software
# Author: Odysseas Xenos
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
"""Utilities for cleaning data before calling SOCCA"""

import numpy as np
from scipy.stats import gaussian_kde

from fink_utils.sso.spins import estimate_sso_params, func_shg1g2

from fink_utils.tester import regular_unit_tests


def dxy_cleaning(data, dxy, mag_red, threshold=0.95):
    """
    Filter observations based on their density in (dxy, reduced magnitude) space using a Gaussian KDE.

    Notes
    -----
    A 2D KDE is computed over the (dxy, mag_red) plane.
    Points are retained if they fall within the 95% (default) highest-density region(s).

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing the observations.
    dxy : array-like
        Values of the positional residuals of between the prediction and observed position [pix]
        computed from sqrt(dx^2 + dy^2)
    mag_red : array-like
        Reduced magnitudes corresponding to each observation.

    Returns
    -------
    pandas.DataFrame
        Subset of `data` containing only the retained points.

    Examples
    --------
    >>> import pandas as pd
    >>> pdf = pd.read_parquet('fink_utils/test_data/atlas-sscat.v3.0_x_ztf.202512_M22_with_ephems.parquet')
    >>> data = pd.DataFrame.from_dict(pdf.head(1).to_dict(orient='records')[0])

    # Dummy values
    >>> data["dx"] = np.random.normal(0, 1, size=len(data))
    >>> data["dy"] = np.random.normal(0, 1, size=len(data))
    >>> data["dxy"] = np.sqrt(data["dx"] ** 2 + data["dy"] ** 2)

    # Dummy values
    >>> data["mred"] = data["cmagpsf"]
    >>> data_xy = dxy_cleaning(data, data["dxy"], data["mred"])
    """
    x = dxy
    y = mag_red

    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    Z = kde(positions).reshape(X.shape)

    Z_flat = Z.ravel()
    idx = np.argsort(Z_flat)[::-1]
    Z_sorted = Z_flat[idx]

    Z_cumsum = np.cumsum(Z_sorted)
    Z_cumsum /= Z_cumsum[-1]

    threshold_index = np.searchsorted(Z_cumsum, threshold)
    level = Z_sorted[threshold_index]

    cond_kde = kde(xy) >= level

    data_kde = data[cond_kde]

    return data_kde


def iterative_cleaning(
    data, mag_red, sigma, phase_angle, filters, ra, dec, verbose=False
):
    """
    Iteratively filter observations based on residuals from an sHG1G2 photometric model fit.

    An sHG1G2 model is fitted to the data, and residuals between the model and the
    observed reduced magnitudes are computed. At each iteration, observations with
    residuals exceeding 3-sigma are removed. The process is repeated until convergence
    (no further points are rejected) or a maximum of 10 iterations is reached.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing the observations.
    mag_red : array-like
        Reduced magnitudes corresponding to each observation.
    sigma : array-like
        Uncertainties on the reduced magnitudes.
    phase_angle : array-like
        Phase angles of the observations [deg].
    filters : array-like
        Photometric filter identifiers for each observation.
    ra : array-like
        Right ascension values of the observations [deg].
    dec : array-like
        Declination values of the observations [deg].
    verbose: bool
        If True, print useful debugging messages

    Returns
    -------
    pandas.DataFrame
        Subset of `data` containing only the retained points after iterative cleaning.

    Examples
    --------
    >>> import pandas as pd
    >>> pdf = pd.read_parquet('fink_utils/test_data/atlas-sscat.v3.0_x_ztf.202512_M22_with_ephems.parquet')
    >>> data = pd.DataFrame.from_dict(pdf.head(1).to_dict(orient='records')[0])
    >>> data_it = iterative_cleaning(
    ...     data,
    ...     data["cmagpsf"].values,
    ...     data["csigmapsf"].values,
    ...     data["Phase"].values,
    ...     data["cfid"].values,
    ...     data["cra"].values,
    ...     data["cdec"].values,
    ... )
    """
    data_inl, mag_red_inl, sigma_inl, phase_angle_inl, filters_inl, ra_inl, dec_inl = (
        data.copy(),
        mag_red.copy(),
        sigma.copy(),
        phase_angle.copy(),
        filters.copy(),
        ra.copy(),
        dec.copy(),
    )
    for k in range(11):
        shgg_params = estimate_sso_params(
            mag_red_inl,
            sigma_inl,
            np.radians(phase_angle_inl),
            filters_inl,
            np.radians(ra_inl),
            np.radians(dec_inl),
            model="SHG1G2",
        )
        fw_model = np.zeros(len(data_inl))

        for ff in np.unique(filters_inl):
            mask = filters_inl == ff

            pts = func_shg1g2(
                [
                    np.radians(phase_angle_inl[mask]),
                    np.radians(ra_inl[mask]),
                    np.radians(dec_inl[mask]),
                ],
                shgg_params[f"H_{ff}"],
                shgg_params[f"G1_{ff}"],
                shgg_params[f"G2_{ff}"],
                shgg_params["R"],
                np.radians(shgg_params["alpha0"]),
                np.radians(shgg_params["delta0"]),
            )

            fw_model[mask] = pts

        residuals = fw_model - mag_red_inl

        threshold = 3 * np.std(residuals)
        cutoff = np.abs(residuals) <= threshold

        prev_len = len(data_inl)

        data_inl = data_inl[cutoff]
        mag_red_inl = mag_red_inl[cutoff]
        sigma_inl = sigma_inl[cutoff]
        phase_angle_inl = phase_angle_inl[cutoff]
        filters_inl = filters_inl[cutoff]
        ra_inl = ra_inl[cutoff]
        dec_inl = dec_inl[cutoff]

        new_len = len(data_inl)
        if (prev_len == new_len) and verbose:
            print("Number of sHG1G2 cleaning iterations:", k)
            break

    return data_inl


if __name__ == "__main__":
    """Execute the unit test suite"""

    # Run the test suite
    regular_unit_tests(globals())

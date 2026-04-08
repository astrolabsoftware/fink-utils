import numpy as np
from scipy.stats import gaussian_kde
from fink_utils.sso.spins import estimate_sso_params, func_shg1g2


def dxy_cleaning(data, dxy, mag_red):
    """
    Filter observations based on their density in (dxy, reduced magnitude) space using a Gaussian KDE.

    A 2D KDE is computed over the (dxy, mag_red) plane.
    Points are retained if they fall within the 95% highest-density region(s).

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

    threshold_index = np.searchsorted(Z_cumsum, 0.95)
    level = Z_sorted[threshold_index]

    cond_kde = kde(xy) >= level

    data_kde = data[cond_kde]

    return data_kde


def iterative_cleaning(data, mag_red, sigma, phase_angle, filters, ra, dec):
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

    Returns
    -------
    pandas.DataFrame
        Subset of `data` containing only the retained points after iterative cleaning.

    Examples
    --------
    >>> data_it = iterative_cleaning(
    ...     data_xy,
    ...     data_xy["mred"].values,
    ...     data_xy["dm"].values,
    ...     data_xy["SOE"].values,
    ...     data_xy["filt"].values,
    ...     data_xy["ra"].values,
    ...     data_xy["dec"].values,
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
        if prev_len == new_len:
            print("Number of sHG1G2 cleaning iterations:", k)
            break

    return data_inl

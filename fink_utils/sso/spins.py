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
import requests
import pandas as pd
import numpy as np
import io

from astropy.coordinates import SkyCoord
import astropy.units as u

from sbpy.photometry import HG1G2

from scipy.optimize import curve_fit

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

    # Standard G1G2 part
    func1 = (
        g1 * HG1G2._phi1(ph) + g2 * HG1G2._phi2(ph) + (1 - g1 - g2) * HG1G2._phi3(ph)
    )
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
    geo = np.sin(dec) * np.sin(beta0) + np.cos(dec) * np.cos(beta0) * np.cos(
        ra - lambda0
    )
    func2 = 1 - (1 - R) * np.abs(geo)
    func2 = -2.5 * np.log10(func2)

    return func1 + func2


def Dfunc_hg1g2_with_spin(pha, h, g1, g2, R, lambda0, beta0):
    """ Return partial derivatives of f(H, G1, G2, R, lambda0, beta0)

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

    Returns
    ----------
    out: array-like transpose([6, N])
        Vector whose elements are partial derivatives at each phase angle.

    """
    ph = pha[0]
    ra = pha[1]
    dec = pha[2]

    # H
    ddh = np.ones(len(ph))

    # G1, G2
    phi1 = HG1G2._phi1(ph)
    phi2 = HG1G2._phi2(ph)
    phi3 = HG1G2._phi3(ph)
    dom = g1 * phi1 + g2 * phi2 + (1 - g1 - g2) * phi3

    ddg1 = 1.085736205 * (phi3 - phi1) / dom
    ddg2 = 1.085736205 * (phi3 - phi2) / dom

    # R
    geo = np.sin(dec) * np.sin(beta0) + np.cos(dec) * np.cos(beta0) * np.cos(
        ra - lambda0
    )
    F2 = 1 - (1 - R) * np.abs(geo)

    ddR = -2.5 * np.abs(geo) / F2

    fact = 2.5 * (1 - R) / F2 * geo / np.abs(geo)
    # lambda0
    ddlambda0 = fact * np.sin(ra - lambda0) * np.cos(dec) * np.cos(beta0)

    # beta0
    ddbeta0 = fact * (np.sin(dec) * np.cos(beta0) - np.cos(dec) * np.cos(ra - lambda0) * np.sin(beta0))

    return np.transpose([ddh, ddg1, ddg2, ddR, ddlambda0, ddbeta0])

def estimate_hg1g2re(
    pdf, bounds=([0, 0, 0, 1e-2, 0, -np.pi / 2], [30, 1, 1, 1, 2 * np.pi, np.pi / 2])
):
    """ Fit for (H, G1, G2, R, lambda0, beta0) using all colors.

    Parameters
    ----------
    pdf: pd.DataFrame
        Contain Fink data + ephemerides
    bounds: tuple of lists
        Parameters boundaries (all_mins, all_maxs)

    Returns
    ----------
    popt: [H, G1, G2, R, lambda0, beta0]
    perr: error on popt
    chi2_red: reduced chi2

    """
    ydata = pdf["i:magpsf_red"] + pdf["color_corr"]

    if not np.alltrue([i == i for i in ydata.values]):
        popt = [None] * 6
        perr = [None] * 6
        chisq_red = None
        return popt, perr, chisq_red

    # Values in radians
    alpha = np.deg2rad(pdf["Phase"].values)
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

        perr = np.sqrt(np.diag(pcov))

        r = ydata.values - func_hg1g2_with_spin(pha, *popt)
        chisq = np.sum((r / pdf["i:sigmapsf"]) ** 2)
        chisq_red = 1.0 / len(ydata.values - 1 - 6) * chisq

    except RuntimeError as e:
        print(e)
        popt = [None] * 6
        perr = [None] * 6
        chisq_red = None

    return popt, perr, chisq_red

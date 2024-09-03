# Copyright 2019-2024 AstroLab Software
# Author: Roman Le Montagner
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
from typing import Tuple


def vect_mag2fluxcal_snana(
    magpsf: "np.array[np.float]", sigmapsf: "np.array[np.float]"
) -> Tuple["np.array[np.float]", "np.array[np.float]"]:
    """Conversion from magnitude to Fluxcal from SNANA manual.

    vectorized version

    Parameters
    ----------
    magpsf: float
        PSF-fit magnitude from ZTF
    sigmapsf: float

    Returns
    -------
    fluxcal: float
        Flux cal as used by SNANA
    fluxcal_err: float
        Absolute error on fluxcal (the derivative has a minus sign)

    """
    magpsf = np.where(np.equal(magpsf, None), np.nan, magpsf)
    sigmapsf = np.where(np.equal(sigmapsf, None), np.nan, sigmapsf)

    fluxcal = 10 ** (-0.4 * magpsf) * 10 ** (11)
    fluxcal_err = 9.21034 * 10**10 * np.exp(-0.921034 * magpsf) * sigmapsf

    return fluxcal, fluxcal_err


def vect_apparent_flux(
    magpsf: "np.array[np.float]",
    sigmapsf: "np.array[np.float]",
    magnr: "np.array[np.float]",
    sigmagnr: "np.array[np.float]",
    isdiffpos: "np.array[str]",
    jansky: bool = True,
) -> Tuple["np.array[np.float]", "np.array[np.float]"]:
    """Compute apparent flux from difference magnitude supplied by ZTF

    Implemented according to p.107 of the ZTF Science Data System Explanatory Supplement
    https://irsa.ipac.caltech.edu/data/ZTF/docs/ztf_explanatory_supplement.pdf
    vectorized version

    Parameters
    ----------
    magpsf,sigmapsf; floats
        magnitude from PSF-fit photometry, and 1-sigma error
    magnr,sigmagnr: floats
        magnitude of nearest source in reference image PSF-catalog
        within 30 arcsec and 1-sigma error
    isdiffpos: str
        t or 1 => candidate is from positive (sci minus ref) subtraction;
        f or 0 => candidate is from negative (ref minus sci) subtraction
    jansky: bool
        If True, normalise units to Jansky. Default is True.

    Returns
    -------
    dc_flux: float
        Apparent flux
    dc_sigflux: float
        Error on apparent flux
    """
    magpsf = np.where(np.equal(magpsf, None), np.nan, magpsf)
    magnr = np.where(magnr < 0.0, np.nan, magnr)

    difference_flux = 10 ** (-0.4 * magpsf)
    difference_sigflux = (sigmapsf / 1.0857) * difference_flux

    ref_flux = 10 ** (-0.4 * magnr)
    ref_sigflux = (sigmagnr / 1.0857) * ref_flux

    # add or subract difference flux based on isdiffpos
    dc_flux = np.where(
        (isdiffpos == "t") | (isdiffpos == "1"),
        ref_flux + difference_flux,
        ref_flux - difference_flux,
    )

    # assumes errors are independent. Maybe too conservative.
    dc_sigflux = np.sqrt(difference_sigflux**2 + ref_sigflux**2)

    if jansky:
        dc_flux *= 3631
        dc_sigflux *= 3631

    return dc_flux, dc_sigflux


def vect_dc_mag(
    magpsf: "np.array[np.float]",
    sigmapsf: "np.array[np.float]",
    magnr: "np.array[np.float]",
    sigmagnr: "np.array[np.float]",
    isdiffpos: "np.array[str]",
) -> Tuple["np.array[np.float]", "np.array[np.float]"]:
    """Compute apparent magnitude from difference magnitude supplied by ZTF

    Implemented according to p.107 of the ZTF Science Data System Explanatory Supplement
    https://irsa.ipac.caltech.edu/data/ZTF/docs/ztf_explanatory_supplement.pdf
    vectorized version

    Parameters
    ----------
    magpsf,sigmapsf
        magnitude from PSF-fit photometry, and 1-sigma error
    magnr,sigmagnr
        magnitude of nearest source in reference image PSF-catalog
        within 30 arcsec and 1-sigma error
    isdiffpos
        t or 1 => candidate is from positive (sci minus ref) subtraction
        f or 0 => candidate is from negative (ref minus sci) subtraction

    Returns
    -------
    dc_mag: float
        Apparent magnitude
    dc_sigmag: float
        Error on apparent magnitude
    """
    dc_flux, dc_sigflux = vect_apparent_flux(
        magpsf, sigmapsf, magnr, sigmagnr, isdiffpos, jansky=False
    )

    # apparent mag and its error from fluxes
    dc_mag = -2.5 * np.log10(dc_flux)

    dc_sigmag = dc_sigflux / dc_flux * 1.0857

    return dc_mag, dc_sigmag

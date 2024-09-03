# Copyright 2019-2024 AstroLab Software
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
from typing import Tuple


def mag2fluxcal_snana(magpsf: float, sigmapsf: float) -> Tuple[float, float]:
    """Conversion from magnitude to Fluxcal from SNANA manual

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
    if magpsf is None:
        return None, None
    fluxcal = 10 ** (-0.4 * magpsf) * 10 ** (11)
    fluxcal_err = 9.21034 * 10**10 * np.exp(-0.921034 * magpsf) * sigmapsf

    return fluxcal, fluxcal_err


def apparent_flux(
    magpsf: float,
    sigmapsf: float,
    magnr: float,
    sigmagnr: float,
    isdiffpos: int,
    jansky: bool = True,
) -> Tuple[float, float]:
    """Compute apparent flux from ZTF difference magnitude

    Implementation according to p.107 of the ZTF Science Data System Explanatory Supplement
    https://irsa.ipac.caltech.edu/data/ZTF/docs/ztf_explanatory_supplement.pdf

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
    if magpsf is None or magnr < 0:
        return float("Nan"), float("Nan")

    difference_flux = 10 ** (-0.4 * magpsf)
    difference_sigflux = (sigmapsf / 1.0857) * difference_flux

    ref_flux = 10 ** (-0.4 * magnr)
    ref_sigflux = (sigmagnr / 1.0857) * ref_flux

    # add or subract difference flux based on isdiffpos
    if (isdiffpos == "t") or (isdiffpos == "1"):
        dc_flux = ref_flux + difference_flux
    else:
        dc_flux = ref_flux - difference_flux

    # assumes errors are independent. Maybe too conservative.
    dc_sigflux = np.sqrt(difference_sigflux**2 + ref_sigflux**2)

    if jansky:
        dc_flux *= 3631
        dc_sigflux *= 3631

    return dc_flux, dc_sigflux


def dc_mag(
    magpsf: float,
    sigmapsf: float,
    magnr: float,
    sigmagnr: float,
    isdiffpos: int,
) -> Tuple[float, float]:
    """Compute apparent magnitude from ZTF difference magnitude

    Implementation according to p.107 of the ZTF Science Data System Explanatory Supplement
    https://irsa.ipac.caltech.edu/data/ZTF/docs/ztf_explanatory_supplement.pdf

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
    dc_flux, dc_sigflux = apparent_flux(
        magpsf, sigmapsf, magnr, sigmagnr, isdiffpos, jansky=False
    )

    # apparent mag and its error from fluxes
    dc_mag = -2.5 * np.log10(dc_flux)
    dc_sigmag = dc_sigflux / dc_flux * 1.0857

    return dc_mag, dc_sigmag

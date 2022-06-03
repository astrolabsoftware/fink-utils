# Copyright 2019-2022 AstroLab Software
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
    fid: "np.array[np.int]",
    magpsf: "np.array[np.float]",
    sigmapsf: "np.array[np.float]",
    magnr: "np.array[np.float]",
    sigmagnr: "np.array[np.float]",
    magzpsci: "np.array[np.float]",
    isdiffpos: "np.array[str]",
) -> Tuple["np.array[np.float]", "np.array[np.float]"]:
    """Compute apparent flux from difference magnitude supplied by ZTF
    This was heavily influenced by the computation provided by Lasair:
    https://github.com/lsst-uk/lasair/blob/master/src/alert_stream_ztf/common/mag.py
    vectorized version

    Parameters
    ---------
    fid
        filter, 1 for green and 2 for red
    magpsf,sigmapsf; floats
        magnitude from PSF-fit photometry, and 1-sigma error
    magnr,sigmagnr: floats
        magnitude of nearest source in reference image PSF-catalog
        within 30 arcsec and 1-sigma error
    magzpsci: float
        Magnitude zero point for photometry estimates
    isdiffpos: str
        t or 1 => candidate is from positive (sci minus ref) subtraction;
        f or 0 => candidate is from negative (ref minus sci) subtraction

    Returns
    -------
    dc_flux: float
        Apparent flux
    dc_sigflux: float
        Error on apparent flux
    """

    magpsf = np.where(np.equal(magpsf, None), np.nan, magpsf)
    magnr = np.where(magnr < 0.0, np.nan, magnr)

    # zero points. Looks like they are fixed.
    magzpref = np.select([fid == 1, fid == 2, fid == 3], [26.325, 26.275, 25.660])

    # reference flux and its error
    magdiff = magzpref - magnr

    ref_flux = 10 ** (0.4 * magdiff)
    ref_sigflux = (sigmagnr / 1.0857) * ref_flux

    # difference flux and its error
    magzpsci = np.where(magzpsci == 0.0, magzpref, magzpsci)

    magdiff = magzpsci - magpsf
    difference_flux = 10 ** (0.4 * magdiff)
    difference_sigflux = (sigmapsf / 1.0857) * difference_flux

    # add or subract difference flux based on isdiffpos
    dc_flux = np.where(
        isdiffpos == "t", ref_flux + difference_flux, ref_flux - difference_flux
    )

    # assumes errors are independent. Maybe too conservative.
    dc_sigflux = np.sqrt(difference_sigflux**2 + ref_sigflux**2)

    return dc_flux, dc_sigflux


def vect_dc_mag(
    fid: "np.array[np.int]",
    magpsf: "np.array[np.float]",
    sigmapsf: "np.array[np.float]",
    magnr: "np.array[np.float]",
    sigmagnr: "np.array[np.float]",
    magzpsci: "np.array[np.float]",
    isdiffpos: "np.array[str]",
) -> Tuple["np.array[np.float]", "np.array[np.float]"]:
    """Compute apparent magnitude from difference magnitude supplied by ZTF
    Parameters
    Stolen from Lasair.
    vectorized version

    Parameters
    ----------
    fid
        filter, 1 for green and 2 for red
    magpsf,sigmapsf
        magnitude from PSF-fit photometry, and 1-sigma error
    magnr,sigmagnr
        magnitude of nearest source in reference image PSF-catalog
        within 30 arcsec and 1-sigma error
    magzpsci
        Magnitude zero point for photometry estimates
    isdiffpos
        t or 1 => candidate is from positive (sci minus ref) subtraction
        f or 0 => candidate is from negative (ref minus sci) subtraction

    Returns
    --------
    dc_mag: float
        Apparent magnitude
    dc_sigmag: float
        Error on apparent magnitude
    """
    # zero points. Looks like they are fixed.
    magzpref = np.select([fid == 1, fid == 2, fid == 3], [26.325, 26.275, 25.660])

    # difference flux and its error
    magzpsci = np.where(np.equal(magzpsci, None), magzpref, magzpsci)

    dc_flux, dc_sigflux = vect_apparent_flux(
        fid, magpsf, sigmapsf, magnr, sigmagnr, magzpsci, isdiffpos
    )

    # apparent mag and its error from fluxes
    test_mag = np.logical_and(np.equal(dc_flux, dc_flux), dc_flux > 0.0)

    dc_mag = np.where(test_mag, magzpsci - 2.5 * np.log10(dc_flux), magzpsci)

    dc_sigmag = np.where(test_mag, dc_sigflux / dc_flux * 1.0857, sigmapsf)

    return dc_mag, dc_sigmag

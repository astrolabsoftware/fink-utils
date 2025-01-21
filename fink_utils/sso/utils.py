# Copyright 2022-2024 AstroLab Software
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
import requests
import subprocess
import json
import os

import pandas as pd
import numpy as np

from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.constants as const
import astropy.units as u

from scipy import signal

from fink_utils.test.tester import regular_unit_tests


def query_miriade(
    ident, jd, observer="I41", rplane="1", tcoor=5, shift=15.0, timeout=30
):
    """Gets asteroid or comet ephemerides from IMCCE Miriade for a suite of JD for a single SSO

    Original function by M. Mahlke, adapted for Fink usage.

    Limitations:
        - Color ephemerides are returned only for asteroids
        - Temporary designations (C/... or YYYY...) do not have ephemerides available

    Parameters
    ----------
    ident: int, float, str
        asteroid or comet identifier
    jd: array
        dates to query
    observer: str
        IAU Obs code - default to ZTF: https://minorplanetcenter.net//iau/lists/ObsCodesF.html
    rplane: str
        Reference plane: equator ('1'), ecliptic ('2').
        If rplane = '2', then tcoor is automatically set to 1 (spherical)
    tcoor: int
        See https://ssp.imcce.fr/webservices/miriade/api/ephemcc/
        Default is 5 (dedicated to observation)
    shift: float
        Time shift to center exposure times, in second.
        Default is 15 seconds which is half of the exposure time for ZTF.
    timeout: int
        Timeout in seconds. Default is 30.

    Returns
    -------
    pd.DataFrame
        Input dataframe with ephemerides columns
        appended False if query failed somehow
    """
    # Miriade URL
    url = "https://ssp.imcce.fr/webservices/miriade/api/ephemcc.php"

    if rplane == "2":
        tcoor = "1"

    # Query parameters
    if ident.endswith("P") or ident.startswith("C/"):
        otype = "c"
    else:
        otype = "a"
    params = {
        "-name": f"{otype}:{ident}",
        "-mime": "json",
        "-rplane": rplane,
        "-tcoor": tcoor,
        "-output": "--jd,--colors(SDSS:r,SDSS:g)",
        "-observer": observer,
        "-tscale": "UTC",
    }

    # Pass sorted list of epochs to speed up query
    shift_hour = shift / 24.0 / 3600.0
    files = {
        "epochs": (
            "epochs",
            "\n".join(["{:.6f}".format(epoch + shift_hour) for epoch in jd]),
        )
    }

    # Execute query
    try:
        r = requests.post(url, params=params, files=files, timeout=timeout)
    except requests.exceptions.ReadTimeout:
        return pd.DataFrame()

    j = r.json()

    # Read JSON response
    try:
        ephem = pd.DataFrame.from_dict(j["data"])
    except KeyError:
        return pd.DataFrame()

    return ephem


def query_miriade_epehemcc(
    ident,
    jd,
    observer="I41",
    rplane="1",
    tcoor=5,
    shift=15.0,
    parameters=None,
    uid=None,
):
    """Gets asteroid or comet ephemerides from IMCCE Miriade for a suite of JD for a single SSO

    This uses local installation of ephemcc instead of the REST API.

    Limitations:
        - Color ephemerides are returned only for asteroids
        - Temporary designations (C/... or YYYY...) do not have ephemerides available

    Parameters
    ----------
    ident: int, float, str
        asteroid or comet identifier
    jd: array
        dates to query
    observer: str
        IAU Obs code - default to ZTF: https://minorplanetcenter.net//iau/lists/ObsCodesF.html
    rplane: str
        Reference plane: equator ('1'), ecliptic ('2').
        If rplane = '2', then tcoor is automatically set to 1 (spherical)
    tcoor: int
        See https://ssp.imcce.fr/webservices/miriade/api/ephemcc/
        Default is 5 (dedicated to observation)
    shift: float
        Time shift to center exposure times, in second.
        Default is 15 seconds which is half of the exposure time for ZTF.
    parameters: dict
        runner_path, userconf, iofile, outdir
    uid: int, optional
        If specified, ID used to write files on disk. Must be unique for each object.
        Default is None, i.e. randomly sampled from U(0, 1e7)

    Returns
    -------
    pd.DataFrame
        Input dataframe with ephemerides columns
        appended False if query failed somehow

    """
    # write tmp files on disk
    if uid is None:
        uid = np.random.randint(0, 1e7)
    date_path = "{}/dates_{}.txt".format(parameters["outdir"], uid)
    ephem_path = "{}/ephem_{}.json".format(parameters["outdir"], uid)

    pdf = pd.DataFrame(jd)

    shift_hour = shift / 24.0 / 3600.0
    pdf.apply(lambda epoch: Time(epoch + shift_hour, format="jd").iso).to_csv(
        date_path, index=False, header=False
    )

    # launch the processing
    cmd = [
        parameters["runner_path"],
        str(ident),
        str(rplane),
        str(tcoor),
        observer,
        "UTC",
        parameters["userconf"],
        parameters["iofile"],
        parameters["outdir"],
        str(uid),
    ]

    # subprocess.run(cmd, capture_output=True)
    out = subprocess.run(cmd)

    if out.returncode != 0:
        print("Error code {}".format(out.returncode))

        # clean date file
        os.remove(date_path)

        return pd.DataFrame()

    # read the data from disk and return
    with open(ephem_path, "r") as f:
        data = json.load(f)
    ephem = pd.DataFrame(data["data"], columns=data["datacol"].keys())

    # clean tmp files
    os.remove(ephem_path)
    os.remove(date_path)

    return ephem


def get_miriade_data(
    pdf,
    sso_colname="i:ssnamenr",
    observer="I41",
    rplane="1",
    tcoor=5,
    withecl=True,
    method="rest",
    parameters=None,
    timeout=30,
    uid=None,
):
    """Add ephemerides information from Miriade to a Pandas DataFrame with SSO lightcurve

    Parameters
    ----------
    pdf: pd.DataFrame
        Pandas DataFrame containing Fink alert data for a (or several) SSO
    sso_colname: str, optional
        Name of the column containing the SSO name to use when querying
        the miriade service. Beware that `ssnamenr` from ZTF is not very
        accurate, and it is better to use quaero before. Default is `i:ssnamenr`
    observer: str
        IAU Obs code - default to ZTF
        https://minorplanetcenter.net//iau/lists/ObsCodesF.html
    rplane: str
        Reference plane: equator ('1', default), ecliptic ('2').
        If rplane = '2', then tcoor is automatically set to 1 (spherical)
    tcoor: int
        See https://ssp.imcce.fr/webservices/miriade/api/ephemcc/
        Default is 5 (dedicated to observation)
    withecl: bool
        If True, query for also for ecliptic Longitude & Latitude (extra call to miriade).
        Default is True.
    method: str
        Use the REST API (`rest`), or a local installation of miriade (`ephemcc`)
    parameters: dict, optional
        If method == `ephemcc`, specify the mapping of extra parameters here. Default is {}.
    timeout: int, optional
        Timeout in seconds when using the REST API. Default is 30.
    uid: int, optional
        If specified, ID used to write files on disk. Must be unique for each object.
        Default is None, i.e. randomly sampled from U(0, 1e7). Only used for method == `ephemcc`.

    Returns
    -------
    out: pd.DataFrame
        DataFrame of the same length, but with new columns from the ephemerides service.
    """
    if parameters is None:
        parameters = {}

    ssnamenrs = np.unique(pdf[sso_colname].values)

    infos = []
    for ssnamenr in ssnamenrs:
        mask = pdf[sso_colname] == ssnamenr
        pdf_sub = pdf[mask]

        if method == "rest":
            eph = query_miriade(
                str(ssnamenr),
                pdf_sub["i:jd"],
                observer=observer,
                rplane=rplane,
                tcoor=tcoor,
                timeout=timeout,
            )
        elif method == "ephemcc":
            eph = query_miriade_epehemcc(
                str(ssnamenr),
                pdf_sub["i:jd"],
                observer=observer,
                rplane=rplane,
                tcoor=tcoor,
                parameters=parameters,
                uid=uid,
            )
        else:
            raise AssertionError(
                "Method must be `rest` or `ephemcc`. {} not supported".format(method)
            )

        if not eph.empty:
            sc = SkyCoord(eph["RA"], eph["DEC"], unit=(u.deg, u.deg))

            eph = eph.drop(columns=["RA", "DEC"])
            eph["RA"] = sc.ra.value * 15
            eph["Dec"] = sc.dec.value

            if withecl:
                # Add Ecliptic coordinates
                if method == "rest":
                    eph_ec = query_miriade(
                        str(ssnamenr),
                        pdf_sub["i:jd"],
                        observer=observer,
                        rplane="2",
                        timeout=timeout,
                    )
                elif method == "ephemcc":
                    eph_ec = query_miriade_epehemcc(
                        str(ssnamenr),
                        pdf_sub["i:jd"],
                        observer=observer,
                        rplane="2",
                        parameters=parameters,
                        uid=uid,
                    )
                else:
                    raise AssertionError(
                        "Method must be `rest` or `ephemcc`. {} not supported".format(
                            method
                        )
                    )

                sc = SkyCoord(
                    eph_ec["Longitude"], eph_ec["Latitude"], unit=(u.deg, u.deg)
                )
                eph["Longitude"] = sc.ra.value
                eph["Latitude"] = sc.dec.value

            # Merge fink & Eph
            info = pd.concat([eph.reset_index(), pdf_sub.reset_index()], axis=1)

            # index has been duplicated obviously
            info = info.loc[:, ~info.columns.duplicated()]

            # Compute magnitude reduced to unit distance
            info["i:magpsf_red"] = info["i:magpsf"] - 5 * np.log10(
                info["Dobs"] * info["Dhelio"]
            )
            infos.append(info)
        else:
            infos.append(pdf_sub)

    if len(infos) > 1:
        info_out = pd.concat(infos)
    else:
        info_out = infos[0]

    return info_out


def is_peak(x, y, xpeak, band=50):
    """Estimate if `xpeak` corresponds to a true extremum for a periodic signal `y`

    Assuming `y` a sparse signal along `x`, we would first estimate
    the period of the signal assuming a sine wave. We would then generate
    predictions, and locate the extrema of the sine.

    But this first step would generate false positives:
    1. As `y` is sparse, some extrema will not coincide with measurements
    2. As the signal is not a perfect sine, the fitted signal might shift
        from the real signal after several periods.

    This function is an extremely quick and dirty attempt to reduce false
    positives by looking at the data around a fitted peak, and
    estimating if the peak is real:
    1. take a band around the peak, and look if data is present
    2. if data is present, check the data is above the mean

    Parameters
    ----------
    xpeak: int
        Candidate peak position
    x: array
        Array of times
    y: array
        Array of elongation
    band: optional, int
        Bandwidth in units of x

    Returns
    -------
    out: bool
        True if `xpeak` corresponds to the location of a peak.
        False otherwise.
    """
    xband = np.where((x > xpeak - band) & (x < xpeak + band))[0]
    if (len(xband) >= 10) and (np.mean(y[xband]) > np.mean(y)):
        return True
    return False


def get_num_opposition(elong, width=4):
    """Estimate the number of opposition according to the solar elongation

    Under the hood, it assumes `elong` is peroidic, and uses a periodogram.

    Parameters
    ----------
    elong: array
        array of solar elongation corresponding to jd
    width: optional, int
        width of peaks in samples.

    Returns
    -------
    nopposition: int
        Number of oppositions estimate

    Examples
    --------
    >>> import io
    >>> import requests
    >>> import pandas as pd

    >>> r = requests.post(
    ...     'https://api.fink-portal.org/api/v1/sso',
    ...     json={
    ...         'n_or_d': '8467',
    ...         'withEphem': True,
    ...         'output-format': 'json'
    ...     }
    ... )
    >>> pdf = pd.read_json(io.BytesIO(r.content))

    # estimate number of oppositions
    >>> noppositions = get_num_opposition(
    ...     pdf['Elong.'].values,
    ...     width=4
    ... )
    >>> assert noppositions == 3, "Found {} oppositions for 8467 instead of 3!".format(noppositions)
    """
    peaks, _ = signal.find_peaks(elong, width=4)
    return len(peaks)


def compute_light_travel_correction(jd, d_obs):
    """Compute the time with light travel corrected

    Parameters
    ----------
    jd: np.array
        Array of times (JD), in day
    d_obs: np.array
        Array of distance to the observer, in AU
    """
    c_speed = const.c.to("au/day").value
    jd_lt = jd - d_obs / c_speed
    return jd_lt


def estimate_axes_ratio(residuals, R):
    """Estimate the axes ratio of a SSO from the residuals of the sHG1G2 model and its oblateness R.

    Parameters
    ----------
    residuals: np.array
        Residuals (observed - model) of the SSO with sHG1G2 model
    R: float
        Oblateness parameter of the sHG1G2 model

    Returns
    -------
    a_b, a_c: float
        a/b and a/c axes ratio
    """
    # Estimate the amplitude of the lightcurve from residuals
    # Taken at 2 sigma
    amplitude = np.std(residuals) * 2.0

    # Estimate the a/b ratio
    a_b = 10 ** (0.4 * (amplitude * 2))

    # Estimate the a/c ratio (and force c<b)
    a_c = (a_b + 1) / (2 * R)
    if a_c < a_b:
        a_c = a_b

    return a_b, a_c


if __name__ == "__main__":
    """Execute the unit test suite"""

    # Run the Spark test suite
    regular_unit_tests(globals())

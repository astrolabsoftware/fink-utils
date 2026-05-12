# Copyright 2025 AstroLab Software
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
"""Utilities for Miriade/eproc"""

import logging
import requests
import subprocess
import json
import os
import io

from astropy.time import Time

import pandas as pd
import numpy as np

from line_profiler import profile

from fink_utils.tester import regular_unit_tests

_LOG = logging.getLogger(__name__)


def get_sso_data(ssnamenr):
    """Wrapper for tests

    Parameters
    ----------
    ssnamenr: str
        SSO name
    """
    r = requests.post(
        "https://api.ztf.fink-portal.org/api/v1/sso",
        json={"n_or_d": ssnamenr, "withEphem": True, "output-format": "json"},
    )

    if r.status_code != 200:
        raise AssertionError((r.content, ssnamenr))
    pdf = pd.read_json(io.BytesIO(r.content))
    return pdf


@profile
def query_miriade(
    ident,
    jd,
    observer="I41",
    rplane="1",
    tcoor=5,
    shift=15.0,
    timeout=30,
    return_json=False,
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
    return_json: bool
        If True, return the JSON payload. Otherwise, returns
        a pandas DataFrame. Default is False.

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
        "-output": "--jd,--colors(SDSS:r,SDSS:g),--iofile(default-ephemcc-observation.xml)",
        "-observer": observer,
        "-tscale": "UTC",
        "-from": "fink",
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
    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
        if return_json:
            return {}
        return pd.DataFrame()

    j = r.json()

    if return_json:
        return j

    # Read JSON response
    try:
        ephem = pd.DataFrame.from_dict(j["data"])
    except KeyError:
        return pd.DataFrame()

    return ephem


@profile
def query_miriade_ephemcc(
    ident,
    jd,
    observer="I41",
    rplane="1",
    tcoor=5,
    shift=15.0,
    parameters=None,
    uid=None,
    return_json=False,
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
    return_json: bool
        If True, return the JSON payload. Otherwise, returns
        a pandas DataFrame. Default is False.

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
        if return_json:
            return {}
        return pd.DataFrame()

    # read the data from disk and return
    with open(ephem_path, "r") as f:
        data = json.load(f)

    # clean tmp files
    os.remove(ephem_path)
    os.remove(date_path)

    if return_json:
        return data
    else:
        ephem = pd.DataFrame(data["data"], columns=data["datacol"].keys())
        return ephem


@profile
def get_miriade_data(
    pdf,
    survey="ztf",
    observer="I41",
    rplane="1",
    tcoor=5,
    method="rest",
    parameters=None,
    timeout=30,
    shift=15.0,
    uid=None,
):
    """Add ephemerides information from Miriade to a Pandas DataFrame with SSO lightcurve

    Parameters
    ----------
    pdf: pd.DataFrame
        Pandas DataFrame containing Fink alert data for a (or several) SSO.
        Mandatory columns depend on the survey:
        - ztf: i:ssnamenr, i:jd, i:magpsf
        - lsst: r:packed_primary_provisional_designation, r:midpointMjdTai, r:psfFlux
    survey: str
        Survey name among ztf | lsst.
    observer: str
        IAU Obs code - default to ZTF
        https://minorplanetcenter.net//iau/lists/ObsCodesF.html
    rplane: str
        Reference plane: equator ('1', default), ecliptic ('2').
        If rplane = '2', then tcoor is automatically set to 1 (spherical)
    tcoor: int
        See https://ssp.imcce.fr/webservices/miriade/api/ephemcc/
        Default is 5 (dedicated to observation)
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

    Examples
    --------
    >>> ssnamenrs = ["33803"]
    >>> for ssnamenr in ssnamenrs:
    ...     pdf = get_sso_data(ssnamenr)
    ...     pdfEphem = get_miriade_data(pdf, survey="ztf", observer="I41", shift=15.0)
    ...     assert "Phase" in pdfEphem.columns, (ssnamenr)
    ...     pdfEphem = get_miriade_data(pdf, survey="lsst", observer="X05", shift=0.0)
    ...     assert "Phase" in pdfEphem.columns, (ssnamenr)
    """
    COLDEF = {
        "ztf": {
            "name": "i:ssnamenr",
            "time": "i:jd",
            "mag": "i:magpsf",
            "scale": "utc",
            "unittime": "jd",
            "unitphot": "mag",
        },
        "lsst": {
            "name": "r:packed_primary_provisional_designation",
            "time": "r:midpointMjdTai",
            "mag": "r:psfFlux",
            "scale": "tai",
            "unittime": "mjd",
            "unitphot": "flux",
        },
    }
    for colname in COLDEF[survey].values():
        if colname not in pdf.columns:
            _LOG.warning(f"You must have {colname} in your DataFrame!")
            return pdf

    if parameters is None:
        parameters = {}

    if (COLDEF[survey]["unittime"] != "jd") and (COLDEF[survey]["scale"] != "utc"):
        time = Time(
            pdf[COLDEF[survey]["time"]].to_numpy(),
            format=COLDEF[survey]["unittime"],
            scale=COLDEF[survey]["scale"],
        ).utc.jd
    else:
        time = pdf[COLDEF[survey]["time"]].to_numpy()

    # FIXME: why looping? despite being different names, they should be
    # the same target (i.e. resolved the same way by quaero).
    ssnamenrs = np.unique(pdf[COLDEF[survey]["name"]].values)

    infos = []
    for ssnamenr in ssnamenrs:
        mask = pdf[COLDEF[survey]["name"]] == ssnamenr
        pdf_sub = pdf[mask]

        if method == "rest":
            eph = query_miriade(
                str(ssnamenr),
                time[mask],
                observer=observer,
                rplane=rplane,
                tcoor=tcoor,
                shift=shift,
                timeout=timeout,
            )
        elif method == "ephemcc":
            eph = query_miriade_ephemcc(
                str(ssnamenr),
                time[mask],
                observer=observer,
                rplane=rplane,
                tcoor=tcoor,
                shift=shift,
                parameters=parameters,
                uid=uid,
            )
        else:
            raise AssertionError(
                "Method must be `rest` or `ephemcc`. {} not supported".format(method)
            )

        if not eph.empty:
            # Merge fink & Eph
            info = pd.concat([eph.reset_index(), pdf_sub.reset_index()], axis=1)

            # index has been duplicated obviously
            info = info.loc[:, ~info.columns.duplicated()]

            if COLDEF[survey]["unitphot"] == "flux":
                # FIXME: assume LSST zero point...
                mag = 31.4 - 2.5 * np.log10(info[COLDEF[survey]["mag"]])
            else:
                mag = info[COLDEF[survey]["mag"]].to_numpy()

            # Compute magnitude reduced to unit distance
            info["i:magpsf_red"] = mag - 5 * np.log10(info["Dobs"] * info["Dhelio"])
            infos.append(info)
        else:
            infos.append(pdf_sub)

    if len(infos) > 1:
        info_out = pd.concat(infos)
    else:
        info_out = infos[0]

    return info_out


if __name__ == "__main__":
    """Execute the unit test suite"""

    # Run the Spark test suite
    regular_unit_tests(globals())

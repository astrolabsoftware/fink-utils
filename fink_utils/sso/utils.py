# Copyright 2022-2023 AstroLab Software
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
import astropy.units as u

def query_miriade(ident, jd, observer='I41', rplane='1', tcoor=5, shift=15.):
    """ Gets asteroid or comet ephemerides from IMCCE Miriade for a suite of JD for a single SSO

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

    Returns
    ----------
    pd.DataFrame
        Input dataframe with ephemerides columns
        appended False if query failed somehow
    """
    # Miriade URL
    url = 'https://ssp.imcce.fr/webservices/miriade/api/ephemcc.php'

    if rplane == '2':
        tcoor = '1'

    # Query parameters
    if ident.endswith('P') or ident.startswith('C/'):
        otype = 'c'
    else:
        otype = 'a'
    params = {
        '-name': f'{otype}:{ident}',
        '-mime': 'json',
        '-rplane': rplane,
        '-tcoor': tcoor,
        '-output': '--jd,--colors(SDSS:r,SDSS:g)',
        '-observer': observer,
        '-tscale': 'UTC'
    }

    # Pass sorted list of epochs to speed up query
    shift_hour = shift / 24.0 / 3600.0
    files = {
        'epochs': ('epochs', '\n'.join(['{:.6f}'.format(epoch + shift_hour) for epoch in jd]))
    }

    # Execute query
    try:
        r = requests.post(url, params=params, files=files, timeout=10)
    except requests.exceptions.ReadTimeout:
        return pd.DataFrame()

    j = r.json()

    # Read JSON response
    try:
        ephem = pd.DataFrame.from_dict(j['data'])
    except KeyError:
        return pd.DataFrame()

    return ephem

def query_miriade_epehemcc(ident, jd, observer='I41', rplane='1', tcoor=5, shift=15., parameters={}):
    """ Gets asteroid or comet ephemerides from IMCCE Miriade for a suite of JD for a single SSO

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

    Returns
    ----------
    pd.DataFrame
        Input dataframe with ephemerides columns
        appended False if query failed somehow

    """
    # write tmp files on disk
    date_path = '{}/dates_{}.txt'.format(parameters['outdir'], ident)
    ephem_path = '{}/ephem_{}.json'.format(parameters['outdir'], ident)

    pdf = pd.DataFrame(jd)

    shift_hour = shift / 24.0 / 3600.0
    pdf\
        .apply(lambda epoch: Time(epoch + shift_hour, format='jd').iso)\
        .to_csv(date_path, index=False, header=False)

    # launch the processing
    cmd = [
        parameters['runner_path'],
        str(ident),
        str(rplane),
        str(tcoor),
        observer,
        "UTC",
        parameters['userconf'],
        parameters['iofile'],
        parameters['outdir']
    ]

    # subprocess.run(cmd, capture_output=True)
    out = subprocess.run(cmd)

    if out.returncode != 0:
        print("Error code {}".format(out.returncode))

        # clean date file
        os.remove(date_path)

        return pd.DataFrame()

    # read the data from disk and return
    with open(ephem_path, 'r') as f:
        data = json.load(f)
    ephem = pd.DataFrame(data['data'], columns=data['datacol'].keys())

    # clean tmp files
    os.remove(ephem_path)
    os.remove(date_path)

    return ephem

def get_miriade_data(pdf, observer='I41', rplane='1', tcoor=5, withecl=True, method='rest', parameters={}):
    """ Add ephemerides information from Miriade to a Pandas DataFrame with SSO lightcurve

    Parameters
    ----------
    pdf: pd.DataFrame
        Pandas DataFrame containing Fink alert data for a (or several) SSO
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
    parameters: dict
        If method == `ephemcc`, specify the mapping of extra parameters here.

    Returns
    ----------
    out: pd.DataFrame
        DataFrame of the same length, but with new columns from the ephemerides service.
    """
    ssnamenrs = np.unique(pdf['i:ssnamenr'].values)

    infos = []
    for ssnamenr in ssnamenrs:
        mask = pdf['i:ssnamenr'] == ssnamenr
        pdf_sub = pdf[mask]

        if method == 'rest':
            eph = query_miriade(
                str(ssnamenr),
                pdf_sub['i:jd'],
                observer=observer,
                rplane=rplane,
                tcoor=tcoor
            )
        elif method == 'ephemcc':
            eph = query_miriade_epehemcc(
                str(ssnamenr),
                pdf_sub['i:jd'],
                observer=observer,
                rplane=rplane,
                tcoor=tcoor,
                parameters=parameters
            )
        else:
            raise AssertionError('Method must be `rest` or `ephemcc`. {} not supported'.format(method))

        if not eph.empty:
            sc = SkyCoord(eph['RA'], eph['DEC'], unit=(u.deg, u.deg))

            eph = eph.drop(columns=['RA', 'DEC'])
            eph['RA'] = sc.ra.value * 15
            eph['Dec'] = sc.dec.value

            if withecl:
                # Add Ecliptic coordinates
                if method == 'rest':
                    eph_ec = query_miriade(
                        str(ssnamenr),
                        pdf_sub['i:jd'],
                        observer=observer,
                        rplane='2'
                    )
                elif method == 'ephemcc':
                    eph_ec = query_miriade_epehemcc(
                        str(ssnamenr),
                        pdf_sub['i:jd'],
                        observer=observer,
                        rplane='2',
                        parameters=parameters
                    )
                else:
                    raise AssertionError('Method must be `rest` or `ephemcc`. {} not supported'.format(method))

                sc = SkyCoord(eph_ec['Longitude'], eph_ec['Latitude'], unit=(u.deg, u.deg))
                eph['Longitude'] = sc.ra.value
                eph['Latitude'] = sc.dec.value

            # Merge fink & Eph
            info = pd.concat([eph.reset_index(), pdf_sub.reset_index()], axis=1)

            # index has been duplicated obviously
            info = info.loc[:, ~info.columns.duplicated()]

            # Compute magnitude reduced to unit distance
            info['i:magpsf_red'] = info['i:magpsf'] - 5 * np.log10(info['Dobs'] * info['Dhelio'])
            infos.append(info)
        else:
            infos.append(pdf_sub)

    if len(infos) > 1:
        info_out = pd.concat(infos)
    else:
        info_out = infos[0]

    return info_out

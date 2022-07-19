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

def get_sso_fink(ssname: str, withEphem: bool = True, withComplement=True):
    """ Fetch data for `ssname` from the Fink API.

    Parameters
    ----------
    ssname: str | int
        MPC designation. Taken from `ssnamenr` in the alert packet.
    withEphem: bool, optional
        If True, query Miriade for ephemerides. Default is True.
    withComplement: bool, optional
        If True, query for extra fields that are not available from
        the SSO table. Adds processing time delay. Default is False.

    Returns
    ----------
    pdf_sso: pd.DataFrame
        Pandas DataFrame containing data for all observations in Fink
    """
    cols = 'i:magpsf,i:sigmapsf,i:fid,i:jd,i:ssnamenr,i:ra,i:dec'

    if withComplement:
        cols += ',i:objectId'

    r = requests.post(
      'https://fink-portal.org/api/v1/sso',
      json={
        'n_or_d': ssname,
        'withEphem': withEphem,
        'columns': cols,
        'output-format': 'json'
      }
    )

    # Format output in a DataFrame
    pdf_sso = pd.read_json(io.BytesIO(r.content))
    
    if withComplement:
        l1 = []
        l2 = []
        for index, oid in enumerate(pdf_sso['i:objectId'].values):

            r = requests.post(
                'https://fink-portal.org/api/v1/objects',
                json={
                    'objectId': oid,
                    'columns': 'i:bimagerat,i:aimagerat',
                }
            )

            tmp = pd.read_json(io.BytesIO(r.content))
            l1.append(tmp['i:aimagerat'].values[0])
            l2.append(tmp['i:bimagerat'].values[0])

        pdf_sso['i:aimagerat'] = l1
        pdf_sso['i:bimagerat'] = l2

    return pdf_sso

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
    func1 = g1*HG1G2._phi1(ph)+g2*HG1G2._phi2(ph)+(1-g1-g2)*HG1G2._phi3(ph)
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
    geo = np.sin(dec) * np.sin(beta0) + np.cos(dec) * np.cos(beta0) * np.cos(ra - lambda0)
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
    dom = (g1*phi1+g2*phi2+(1-g1-g2)*phi3)

    ddg1 = 1.085736205*(phi3-phi1)/dom
    ddg2 = 1.085736205*(phi3-phi2)/dom

    # R
    geo = np.sin(dec) * np.sin(beta0) + np.cos(dec) * np.cos(beta0) * np.cos(ra - lambda0)
    F2 = 1 - (1 - R) * np.abs(geo)

    ddR = -2.5 * np.abs(geo) / F2

    # lambda0
    ddlambda0 = 2.5 * (1 - R) / F2 * geo / np.abs(geo) * np.sin(ra - lambda0) * np.cos(dec) * np.cos(beta0)

    # beta0
    ddbeta0 = 2.5 * (1 - R) / F2 * geo / np.abs(geo) * (np.sin(dec)*np.cos(beta0) - np.cos(dec)*np.cos(ra-lambda0)*np.sin(beta0))


    return np.transpose([ddh, ddg1, ddg2, ddR, ddlambda0, ddbeta0])

def query_miriade(ident, jd, observer='I41', rplane='1', tcoor=5):
    """ Gets asteroid or comet ephemerides from IMCCE Miriade for a suite of JD for a single SSO
    
    Original function by M. Mahlke

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
    files = {
        'epochs': ('epochs', '\n'.join(['{:.6f}'.format(epoch + 15./24/3600) for epoch in jd]))
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

def get_miriade_data(pdf, add_ecl=False, observer='I41', rplane='1', tcoor=5):
    """ Concatenate ephemerides data to the ZTF data

    Parameters
    ----------
    pdf: pd.DataFrame
        Pandas DataFrame containing ZTF alert data for a SSO
    observer: str
        IAU Obs code - default to ZTF
        https://minorplanetcenter.net//iau/lists/ObsCodesF.html
    rplane: str
        Reference plane: equator ('1'), ecliptic ('2').
        If rplane = '2', then tcoor is automatically set to 1 (spherical)
    tcoor: int
        See https://ssp.imcce.fr/webservices/miriade/api/ephemcc/
        Default is 5 (dedicated to observation)

    Returns
    ---------
    out: pd.DataFrame
        Pandas Dataframe with the same length as the input but
        more columns from the Miriade service.
    """
    ssnamenrs = np.unique(pdf['i:ssnamenr'].values)

    infos = []
    for ssnamenr in ssnamenrs:
        mask = pdf['i:ssnamenr'] == ssnamenr
        pdf_sub = pdf[mask]

        eph = query_miriade(
            str(ssnamenr), 
            pdf_sub['i:jd'], 
            observer=observer, 
            rplane=rplane, 
            tcoor=tcoor
        )

        if not eph.empty:
            sc = SkyCoord(eph['RA'], eph['DEC'], unit=(u.deg, u.deg))

            eph = eph.drop(columns=['RA', 'DEC'])
            eph['RA'] = sc.ra.value * 15
            eph['Dec'] = sc.dec.value

            if add_ecl:
                # Add Ecliptic coordinates
                eph_ec = query_miriade(
                    str(ssnamenr), 
                    pdf_sub['i:jd'], 
                    observer=observer, 
                    rplane='2'
                )

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


def add_fdist(pdf):
    """ Add a new column containing the distance part of the lightcuve

    f(dist) = -5 * log10(Dhelio * Dobs)

    Parameters
    ----------
    pdf: pd.DataFrame
        Pandas DataFrame containing ephemerides

    Returns
    ----------
    out: pd.DataFrame
        Input Pandas DataFrame with a new column `fdist`
    """
    pdf['fdist'] = 5 * np.log10(pdf['Dhelio'] * pdf['Dobs'])

    return pdf
    

def add_ztf_color_correction(pdf):
    """ Add a new column with ZTF color correction.

    The factor is color-dependent, and assumed to be:
    - V_minus_g = -0.32
    - V_minus_r = 0.13

    g --> g + (V - g)
    r --> r + (V - r) - (V - g)

    Parameters
    ----------
    pdf: pd.DataFrame
        Pandas DataFrame with Fink ZTF data

    Returns
    ----------
    out: pd.DataFrame
        Input Pandas DataFrame with a new column `color_corr`
    """
    filts = np.unique(pdf['i:fid'].values)
    color_sso = np.ones_like(pdf['i:magpsf'])
    for i, filt in enumerate(filts):
        # SSO Color index
        V_minus_g = -0.32
        V_minus_r = 0.13

        cond = pdf['i:fid'] == filt

        # Color conversion
        if filt == 1:
            color_sso[cond] = V_minus_g
        else:
            color_sso[cond] = V_minus_r - V_minus_g
            
    pdf['color_corr'] = color_sso
    
    return pdf

def estimate_hg1g2re(pdf, bounds=([0, 0, 0, 1e-2, 0, -np.pi/2], [30, 1, 1, 1, 2*np.pi, np.pi/2])):
    """
    """
    ydata = pdf['i:magpsf_red'] + pdf['color_corr']
    
    if not np.alltrue([i==i for i in ydata.values]):
        popt = [None] * 6
        perr = [None] * 6
        chisq_red = None
        return popt, perr, chisq_red
        

    # Values in radians
    alpha = np.deg2rad(pdf['Phase'].values)
    ra = np.deg2rad(pdf['i:ra'].values)
    dec = np.deg2rad(pdf['i:dec'].values)
    pha = np.transpose([[i, j, k] for i, j, k in zip(alpha, ra, dec)])

    try:
        popt, pcov = curve_fit(
            func_hg1g2_with_spin, 
            pha,
            ydata.values, 
            sigma=pdf['i:sigmapsf'],
            bounds=bounds,
            jac=Dfunc_hg1g2_with_spin
        )

        perr = np.sqrt(np.diag(pcov))

        r = ydata.values - func_hg1g2_with_spin(pha, *popt)
        chisq = np.sum((r / pdf['i:sigmapsf'])**2)
        chisq_red = 1. / len(ydata.values - 1 - 6) * chisq
        
    except RuntimeError as e:
        print(e)
        popt = [None] * 6
        perr = [None] * 6
        chisq_red = None
        
    return popt, perr, chisq_red
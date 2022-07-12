import requests
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import json

def query_miriade(ident, jd, observer='I41', rplane='1', tcoor=5):
    """ Gets asteroid or comet ephemerides from IMCCE Miriade for a suite of JD for a single SSO
    Original function by M. Mahlke
    Limitations:
        - Color ephemerides are returned only for asteroids
        - Temporary designations (C/... or YYYY...) do not have ephemerides available
    Parameters
    ----------
    ident: str
        asteroid or comet identifier
    jd: float array
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
        'epochs': ('epochs', '\n'.join(['%.6f' % epoch for epoch in jd]))
    }

    # Execute query
    try:
        r = requests.post(url, params=params, files=files, timeout=2000)
    except requests.exceptions.ReadTimeout:
        return False

    if len(r.text) > 0:

        j = json.loads(r.text)

        # Read JSON response
        try:
            ephem = pd.DataFrame.from_dict(j['data'])
        except KeyError:
            return pd.DataFrame()

        return ephem
    else:
        return pd.DataFrame()


def get_miriade_data(pdf):
    """
    Query miriade and concatenate the ephemeries result with pdf.

    Parameters
    ----------
    pdf : pd.DataFrame
        Observation of one or multiple solar system objects. 
            must contains at least the following columns: i:ssnamenr, i:jd
    
    Return
    ------
    info_out : pd.DataFrame
        concatenation of the observations with the ephemeries
    """
    pdf["i:ssnamenr"] = pdf["i:ssnamenr"].astype(str)
    ssnamenrs = pdf['i:ssnamenr'].unique()
    ztf_code = 'I41'

    infos = []
    for ssnamenr in ssnamenrs:
        mask = pdf['i:ssnamenr'] == ssnamenr
        pdf_sub = pdf[mask]

        eph = query_miriade(ssnamenr, pdf_sub['i:jd'], observer=ztf_code)

        if not eph.empty:
            sc = SkyCoord(eph['RA'], eph['DEC'], unit=(u.deg, u.deg))

            eph = eph.drop(columns=['RA', 'DEC'])
            eph['RA'] = sc.ra.value * 15
            eph['Dec'] = sc.dec.value

            # Add Ecliptic coordinates
            eph_ec = query_miriade(ssnamenr, pdf_sub['i:jd'], rplane='2')

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
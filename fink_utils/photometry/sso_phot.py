import pandas as pd
import numpy as np

from astropy.modeling.fitting import LevMarLSQFitter
from sbpy.photometry import HG1G2, HG12, HG
import astropy.units as u
from sbpy.data import Obs

def fit_phase_function(pdf, switch_func: str):
    """
    Compute the absolute magnitude model for one solar system objects

    Parameters
    ----------
    pdf : pd.DataFrame
        observations of one solar system objects
    switch_func : string
        select the phase function between : [HG1G2, HG12, HG]
    
    Returns
    -------
    df_table : pd.DataFrame
        the output of the fitted function, depends of switch_func
    """
    pdf = pdf.sort_values("Phase")

    pdf['i:fid'] = pdf['i:fid'].astype(int)
    filters = {1: 'g', 2: 'R', 3: 'i'}

    # instanciate the fitter
    fitter = LevMarLSQFitter(calc_uncertainties=True)

    # select the phase function
    if switch_func == 'HG1G2':
        fitfunc = HG1G2
        params = ['H', 'G1', 'G2']
    elif switch_func == 'HG12':
        fitfunc = HG12
        params = ['H', 'G12']
    elif switch_func == 'HG':
        fitfunc = HG
        params = ['H', 'G']

    filts = pdf['i:fid'].unique()

    dd = {'': [filters[f] + ' band' for f in filts]}
    dd.update({i: [''] * len(filts) for i in params})
    df_table = pd.DataFrame(
        dd,
        index=[filters[f] for f in filts]
    )

    for i, f in enumerate(filts):


        cond = pdf['i:fid'] == f
        ydata = pdf.loc[cond, 'i:magpsf_red']  # + color_sso

        try:
            obs = Obs.from_dict(
                {
                    'alpha': pdf.loc[cond, 'Phase'].values * u.deg,
                    'mag': ydata.values * u.mag
                }
            )


            model_func = fitfunc.from_obs(
                obs,
                fitter,
                'mag',
                weights = 1 / pdf.loc[cond, 'i:sigmapsf'] * u.mag
            )

        except RuntimeError as e:
            return print("The fitting procedure could not converge.")

        
        if model_func.cov_matrix is not None:
            perr = np.sqrt(np.diag(model_func.cov_matrix.cov_matrix))
        else:
            perr = [np.nan] * len(params)

        for pindex, param in enumerate(params):
                df_table[param][df_table[param].index == filters[f]] = '{:.2f} plus_minus {:.2f}'.format(model_func.parameters[pindex], perr[pindex])

    return df_table


if __name__ == "__main__":

    from fink_utils.requests.miriade import get_miriade_data
    import requests
    import io

    import astropy

    r = requests.post(
        'https://fink-portal.org/api/v1/sso',
        json={
            'n_or_d': '265',
            'output-format': 'json'
        }
    )

    pdf = pd.read_json(io.BytesIO(r.content))

    pdf_ephem = get_miriade_data(pdf)

    df = fit_phase_function(pdf_ephem, "HG1G2")

    print(df)
import pandas as pd
import numpy as np
from fink_utils.sso.spins import estimate_sso_params, func_shg1g2
import getopt
import sys
import pyarrow.dataset as ds
from scipy.stats import gaussian_kde
import rocks
from astropy.time import Time
import requests
from asteroid_spinprops.ssolib import modelfit
import os

def usage():
    print("""
SOCCA - ATLAS
--------------------------------------------------------------------------
Description:
    This script reads and cleans ATLAS v3 SSO photometric data and fits SOCCA to them.

Usage:
    python SOCCA-atlas.py -d <path_data> -t <output_dir> -n <sso_name>

Required arguments:
    -d, --path_data     Path to the input parquet dataset directory
    -t, --target        Output directory for the resulting SOCCA parameters (CSV file)
    -n, --target        Asteroid packed designation (SSO identifier)

Optional:
    -h, --help          Show this help message and exit

Example for (45) Eugenia:
    python SOCCA-atlas.py -d ./dataset -t ./results/ -n 00045

Notes:
    - The dataset must contain fields:
        kast, dx, dy, m, R, delta, dm, SOE, filt, ra, dec, MJD_lc
    - Output will be written as:
        <output_dir>/<sso_name>.csv
""")


def query(name, epochs):
    """Gets asteoid ephemerides from VOSSP Miriade.

    Parameters
    ----------
    name : str
        Name or designation of asteroid.
    epochs : list
        List of observation epochs in JD format.

    Returns
    -------
    pd.DataFrame - Input dataframe with ephemerides columns appended
                False - If query failed somehow
    """
    # Pass sorted list of epochs to speed up query
    # Have to convert them to JD
    epochs = [Time(str(e), format="jd").jd for e in epochs]
    files = {"epochs": ("epochs", "\n".join(["%.6f" % epoch for epoch in epochs]))}

    # ------
    # Query Miriade for phase angles
    url = "https://ssp.imcce.fr/webservices/miriade/api/ephemcc.php?"

    params = {
        "-name": f"{name}",
        "-mime": "json",
        "-rplane": "1",
        "-tcoor": 5,
        "-output": "--jd",
        "-observer": "500",
        "-tscale": "UTC",
    }
    params["-output"] += ",--iofile(ephemcc-photom.xml)"

    # Execute query
    try:
        r = requests.post(url, params=params, files=files, timeout=50)
    except requests.exceptions.ReadTimeout:
        return False
    j = r.json()

    # Read JSON response
    try:
        ephem = pd.DataFrame.from_dict(j["data"])
    except KeyError:
        return False
    return ephem


def main(argv):
    opts, _ = getopt.getopt(
        argv,
        "d:t:n:h",
        [
            "path_data=",
            "target=",
            "sso_name=",
            "help",
        ],
    )
    args = dict(opts)

    if "--help" in args or "-h" in args:
        usage()
        sys.exit()

    output_dir = args.get("-t") or args.get("--target")
    path_data = args.get("-d") or args.get("--path_data")
    ssnamenr = args.get("-n") or args.get("--sso_name")

    dataset = ds.dataset(path_data)

    table = dataset.to_table(filter=ds.field("kast") == ssnamenr)
    data = table.to_pandas()

    data["dxy"] = np.sqrt(data["dx"] ** 2 + data["dy"] ** 2)
    data["mred"] = data["m"] - 5 * np.log10(data["R"] * data["delta"])

    x = data["dxy"].to_numpy()
    y = data["mred"].to_numpy()

    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    Z = kde(positions).reshape(X.shape)

    Z_flat = Z.ravel()
    idx = np.argsort(Z_flat)[::-1]
    Z_sorted = Z_flat[idx]

    Z_cumsum = np.cumsum(Z_sorted)
    Z_cumsum /= Z_cumsum[-1]

    threshold_index = np.searchsorted(Z_cumsum, 0.95)
    level = Z_sorted[threshold_index]

    cond_kde = kde(xy) >= level

    data_kde = data[cond_kde]
    data_inl = data_kde.copy()

    for k in range(11):
        shgg_params = estimate_sso_params(
            data_inl["mred"],
            data_inl["dm"],
            np.radians(data_inl["SOE"]),
            data_inl["filt"],
            np.radians(data_inl["ra"]),
            np.radians(data_inl["dec"]),
            model="SHG1G2",
        )
        fw_model = np.zeros(len(data_inl))

        for ff in np.unique(data_inl["filt"]):
            mask = data_inl["filt"] == ff

            pts = func_shg1g2(
                [
                    np.radians(data_inl.loc[mask, "SOE"]),
                    np.radians(data_inl.loc[mask, "ra"]),
                    np.radians(data_inl.loc[mask, "dec"]),
                ],
                shgg_params[f"H_{ff}"],
                shgg_params[f"G1_{ff}"],
                shgg_params[f"G2_{ff}"],
                shgg_params["R"],
                np.radians(shgg_params["alpha0"]),
                np.radians(shgg_params["delta0"]),
            )
            fw_model[mask] = pts

        residuals = fw_model - data_inl["mred"]

        threshold = 3 * np.std(residuals)
        cutoff = np.abs(residuals) <= threshold

        prev_len = len(data_inl)
        data_inl = data_inl[cutoff]
        new_len = len(data_inl)

        if prev_len == new_len:
            print("Number of sHG1G2 cleaning iterations:", k)
            break

    data_inl = data_inl[cutoff]

    rockid = str(rocks.Rock(ssnamenr).number)

    data_inl["JD_lc"] = data_inl["MJD_lc"] + 2400000.5

    print("Querying ephemerides via IMCCE Miriade..")
    ephem = query(rockid, data_inl["JD_lc"])

    ra_s = ephem["RA_h"].to_numpy()
    dec_s = ephem["DEC_h"].to_numpy()

    pdf = data_inl.reset_index(drop=True)

    # Rename columns to match Fink format
    pdf.rename(
        columns={
            "SOE": "Phase",
            "dm": "csigmapsf",
            "filt": "cfid",
            "mred": "cmred",
            "R": "Dhelio",
        },
        # inplace=True,
    )

    # Add missing columns

    pdf["residuals"] = 0.0
    pdf["ra_s"] = ra_s
    pdf["dec_s"] = dec_s

    # LT correction
    pdf["cjd"] = pdf["MJD_lc"] + 2400000.5  # MJD to JD

    cfid_map = {
        "o": 1,
        "c": 2,
    }
    pdf["cfid"] = pdf["cfid"].map(cfid_map)

    # Make it readable by asteroid_spinprops
    pdf_s = pd.DataFrame({col: [np.array(pdf[col])] for col in pdf.columns})

    base_kwargs = dict(
        use_angles=True,
        use_filter_dependent=True,
        use_phase=True,
        use_shape=True,
    )

    current_kwargs = base_kwargs.copy()

    SOCCA_params = modelfit.get_fit_params(
        data=pdf_s,
        flavor="SOCCA",
        shg1g2_constrained=True,
        period_blind=True,
        pole_blind=False,
        period_in=None,
        period_quality_flag=True,
        terminator=True,
        time_me=True,
        remap=True,
        remap_kwargs=current_kwargs,
    )

    SOCCA_out = pd.DataFrame.from_dict([SOCCA_params])

    SOCCA_out.to_csv(os.path.join(output_dir, ssnamenr + ".csv"), index=None)


if __name__ == "__main__":
    main(sys.argv[1:])

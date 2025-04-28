# Copyright 2024-2025 AstroLab Software
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
"""Contains definition and functionalities for the SSO Fink Table"""

import datetime
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from astropy.time import Time

from fink_utils.sso.utils import retrieve_last_date_of_previous_month
from fink_utils.tester import spark_unit_tests

COLUMNS = {
    "ssnamenr": {
        "type": "str",
        "description": "Designation (name or number) of the object from MPC archive as given by ZTF",
    },
    "sso_name": {
        "type": "str",
        "description": "Official name or provisional designation of the SSO",
    },
    "sso_number": {"type": "int", "description": "IAU number of the SSO"},
    "last_jd": {
        "type": "double",
        "description": "Julian Date for the last detection in Fink, in UTC",
    },
    "H_1": {
        "type": "double",
        "description": "Absolute magnitude for the ZTF filter band g",
    },
    "H_2": {
        "type": "double",
        "description": "Absolute magnitude for the ZTF filter band r",
    },
    "err_H_1": {
        "type": "double",
        "description": "Uncertainty on the absolute magnitude for the ZTF filter band g",
    },
    "err_H_2": {
        "type": "double",
        "description": "Uncertainty on the absolute magnitude for the ZTF filter band r",
    },
    "min_phase": {
        "type": "double",
        "description": "Minimum phase angle of the observations used to compute the phase function, in degree",
    },
    "min_phase_1": {
        "type": "double",
        "description": "Minimum phase angle of the observations used to compute the phase function for the ZTF filter band g, in degree",
    },
    "min_phase_2": {
        "type": "double",
        "description": "Minimum phase angle of the observations used to compute the phase function for the ZTF filter band r, in degree",
    },
    "max_phase": {
        "type": "double",
        "description": "Maximum phase angle of the observations used to compute the phase function, in degree",
    },
    "max_phase_1": {
        "type": "double",
        "description": "Maximum phase angle of the observations used to compute the phase function for the ZTF filter band g, in degree",
    },
    "max_phase_2": {
        "type": "double",
        "description": "Maximum phase angle of the observations used to compute the phase function for the ZTF filter band r, in degree",
    },
    "chi2red": {"type": "double", "description": "Reduced chi-square of the fit"},
    "rms": {"type": "double", "description": "RMS of the fit, in magnitude"},
    "rms_1": {
        "type": "double",
        "description": "RMS of the fit for the filter band g, in magnitude",
    },
    "rms_2": {
        "type": "double",
        "description": "RMS of the fit for the filter band r, in magnitude",
    },
    "median_error_phot": {
        "type": "double",
        "description": "Median of the 1-sigma uncertainty on the measurements, in magnitude",
    },
    "median_error_phot_1": {
        "type": "double",
        "description": "Median of the 1-sigma uncertainty on the measurements for the filter band g, in magnitude",
    },
    "median_error_phot_2": {
        "type": "double",
        "description": "Median of the 1-sigma uncertainty on the measurements for the filter band r, in magnitude",
    },
    "mean_astrometry": {
        "type": "double",
        "description": "Astrometry: mean of the angular separation between observations and ephemerides, in arcsecond",
    },
    "std_astrometry": {
        "type": "double",
        "description": "Astrometry: standard deviation of the angular separation between observations and ephemerides, in arcsecond",
    },
    "skew_astrometry": {
        "type": "double",
        "description": "Astrometry: skewness of the angular separation between observations and ephemerides",
    },
    "kurt_astrometry": {
        "type": "double",
        "description": "Astrometry: kurtosis of the angular separation between observations and ephemerides",
    },
    "period": {
        "type": "double",
        "description": "Sidereal period estimated, in hour. Available only from 2024.10",
    },
    "period_chi2red": {
        "type": "double",
        "description": "Reduced chi-square for the period estimation. Available only from 2024.10",
    },
    "n_obs": {"type": "int", "description": "Number of observations in Fink"},
    "n_obs_1": {
        "type": "int",
        "description": "Number of observations for the ZTF filter band g in Fink",
    },
    "n_obs_2": {
        "type": "int",
        "description": "Number of observations for the ZTF filter band r in Fink",
    },
    "n_days": {
        "type": "int",
        "description": "Number of days between the first and the last observations in Fink",
    },
    "n_days_1": {
        "type": "int",
        "description": "Number of days between the first and the last observations in Fink, for the ZTF filter band g",
    },
    "n_days_2": {
        "type": "int",
        "description": "Number of days between the first and the last observations in Fink, for the ZTF filter band r",
    },
    "fit": {
        "type": "int",
        "description": "Code to assess the quality of the fit: 0: success, 1: bad_vals, 2: MiriadeFail, 3: RunTimError, 4: LinalgError",
    },
    "status": {
        "type": "int",
        "description": "Code for quality `status` (least square convergence): -2: failure, -1 : improper input parameters status returned from MINPACK, 0 : the maximum number of function evaluations is exceeded, 1 : gtol termination condition is satisfied, 2 : ftol termination condition is satisfied, 3 : xtol termination condition is satisfied, 4 : Both ftol and xtol termination conditions are satisfied.",
    },
    "flag": {"type": "int", "description": "TBD"},
    "version": {"type": "str", "description": "Version of the SSOFT YYYY.MM"},
}

COLUMNS_SSHG1G2 = {
    "G1_1": {
        "type": "double",
        "description": "G1 phase parameter for the ZTF filter band g",
    },
    "G1_2": {
        "type": "double",
        "description": "G1 phase parameter for the ZTF filter band r",
    },
    "G2_1": {
        "type": "double",
        "description": "G2 phase parameter for the ZTF filter band g",
    },
    "G2_2": {
        "type": "double",
        "description": "G2 phase parameter for the ZTF filter band r",
    },
    "a_b": {"type": "double", "description": "a/b ratio of the ellipsoid (a>=b>=c)."},
    "a_c": {"type": "double", "description": "a/c ratio of the ellipsoid (a>=b>=c)."},
    "phi0": {
        "type": "double",
        "description": "Initial rotation phase at reference time t0, in radian",
    },
    "alpha0": {
        "type": "double",
        "description": "Right ascension of the spin axis (EQJ2000), in degree",
    },
    "delta0": {
        "type": "double",
        "description": "Declination of the spin axis (EQJ2000), in degree",
    },
    "alpha0_alt": {
        "type": "double",
        "description": "Flipped `alpha0`: (`alpha0` + 180) modulo 360, in degree",
    },
    "delta0_alt": {
        "type": "double",
        "description": "Flipped `delta0`: -`delta0`, in degree",
    },
    "obliquity": {
        "type": "double",
        "description": "Obliquity of the spin axis, in degree",
    },
    "err_G1_1": {
        "type": "double",
        "description": "Uncertainty on the G1 phase parameter for the ZTF filter band g",
    },
    "err_G1_2": {
        "type": "double",
        "description": "Uncertainty on the G1 phase parameter for the ZTF filter band r",
    },
    "err_G2_1": {
        "type": "double",
        "description": "Uncertainty on the G2 phase parameter for the ZTF filter band g",
    },
    "err_G2_2": {
        "type": "double",
        "description": "Uncertainty on the G2 phase parameter for the ZTF filter band r",
    },
    "err_a_b": {"type": "double", "description": "Uncertainty on a/b"},
    "err_a_c": {"type": "double", "description": "Uncertainty on a/c"},
    "err_phi0": {
        "type": "double",
        "description": "Uncertainty on the initial rotation phase, in radian",
    },
    "err_alpha0": {
        "type": "double",
        "description": "Uncertainty on the right ascension of the spin axis (EQJ2000), in degree",
    },
    "err_delta0": {
        "type": "double",
        "description": "Uncertainty on the declination of the spin axis (EQJ2000), in degree",
    },
    "err_period": {
        "type": "double",
        "description": "Uncertainty on the sidereal period, in hour. Available only from 2024.10",
    },
    "max_cos_lambda": {
        "type": "double",
        "description": "Maximum of the absolute value of the cosine for the aspect angle",
    },
    "mean_cos_lambda": {
        "type": "double",
        "description": "Mean of the absolute value of the cosine for the aspect angle",
    },
    "min_cos_lambda": {
        "type": "double",
        "description": "Minimum of the absolute value of the cosine for the aspect angle",
    },
}

COLUMNS_SHG1G2 = {
    "G1_1": {
        "type": "double",
        "description": "G1 phase parameter for the ZTF filter band g",
    },
    "G1_2": {
        "type": "double",
        "description": "G1 phase parameter for the ZTF filter band r",
    },
    "G2_1": {
        "type": "double",
        "description": "G2 phase parameter for the ZTF filter band g",
    },
    "G2_2": {
        "type": "double",
        "description": "G2 phase parameter for the ZTF filter band r",
    },
    "R": {"type": "double", "description": "Oblateness of the object"},
    "a_b": {
        "type": "double",
        "description": "a/b ratio of the ellipsoid (a>=b>=c). Estimation based on the fit residuals and the oblateness.",
    },
    "a_c": {
        "type": "double",
        "description": "a/c ratio of the ellipsoid (a>=b>=c). Estimation based on the fit residuals and the oblateness.",
    },
    "alpha0": {
        "type": "double",
        "description": "Right ascension of the spin axis (EQJ2000), in degree",
    },
    "delta0": {
        "type": "double",
        "description": "Declination of the spin axis (EQJ2000), in degree",
    },
    "alpha0_alt": {
        "type": "double",
        "description": "Flipped `alpha0`: (`alpha0` + 180) modulo 360, in degree",
    },
    "delta0_alt": {
        "type": "double",
        "description": "Flipped `delta0`: -`delta0`, in degree",
    },
    "obliquity": {
        "type": "double",
        "description": "Obliquity of the spin axis, in degree",
    },
    "err_G1_1": {
        "type": "double",
        "description": "Uncertainty on the G1 phase parameter for the ZTF filter band g",
    },
    "err_G1_2": {
        "type": "double",
        "description": "Uncertainty on the G1 phase parameter for the ZTF filter band r",
    },
    "err_G2_1": {
        "type": "double",
        "description": "Uncertainty on the G2 phase parameter for the ZTF filter band g",
    },
    "err_G2_2": {
        "type": "double",
        "description": "Uncertainty on the G2 phase parameter for the ZTF filter band r",
    },
    "err_R": {"type": "double", "description": "Uncertainty on the oblateness"},
    "err_alpha0": {
        "type": "double",
        "description": "Uncertainty on the right ascension of the spin axis (EQJ2000), in degree",
    },
    "err_delta0": {
        "type": "double",
        "description": "Uncertainty on the declination of the spin axis (EQJ2000), in degree",
    },
    "max_cos_lambda": {
        "type": "double",
        "description": "Maximum of the absolute value of the cosine for the aspect angle",
    },
    "mean_cos_lambda": {
        "type": "double",
        "description": "Mean of the absolute value of the cosine for the aspect angle",
    },
    "min_cos_lambda": {
        "type": "double",
        "description": "Minimum of the absolute value of the cosine for the aspect angle",
    },
}

COLUMNS_HG1G2 = {
    "G1_1": {
        "type": "double",
        "description": "G1 phase parameter for the ZTF filter band g",
    },
    "G1_2": {
        "type": "double",
        "description": "G1 phase parameter for the ZTF filter band r",
    },
    "G2_1": {
        "type": "double",
        "description": "G2 phase parameter for the ZTF filter band g",
    },
    "G2_2": {
        "type": "double",
        "description": "G2 phase parameter for the ZTF filter band r",
    },
    "err_G1_1": {
        "type": "double",
        "description": "Uncertainty on the G1 phase parameter for the ZTF filter band g",
    },
    "err_G1_2": {
        "type": "double",
        "description": "Uncertainty on the G1 phase parameter for the ZTF filter band r",
    },
    "err_G2_1": {
        "type": "double",
        "description": "Uncertainty on the G2 phase parameter for the ZTF filter band g",
    },
    "err_G2_2": {
        "type": "double",
        "description": "Uncertainty on the G2 phase parameter for the ZTF filter band r",
    },
}

COLUMNS_HG = {
    "G_1": {
        "type": "double",
        "description": "G phase parameter for the ZTF filter band g",
    },
    "G_2": {
        "type": "double",
        "description": "G phase parameter for the ZTF filter band r",
    },
    "err_G_1": {
        "type": "double",
        "description": "Uncertainty on the G phase parameter for the ZTF filter band g",
    },
    "err_G_2": {
        "type": "double",
        "description": "Uncertainty on the G phase parameter for the ZTF filter band r",
    },
}


def join_aggregated_sso_data(df_prev, df_new, on="ssnamenr", output_filename=None):
    """Join two DataFrame containing arrays

    Notes
    -----
    We perform an outer join

    Parameters
    ----------
    df_prev: Spark DataFrame
        DataFrame containing previous lightcurves
    df_new: Spark DataFrame
        DataFrame containing new lightcurve portions to be added
    on: str
        Column name for the join. Must exist in both DataFrames
    output_filename: str, optional
        If given, save data on HDFS. Cannot overwrite. Default is None.

    Returns
    -------
    out: Spark DataFrame
        Concatenated DataFrame with full lightcurves

    Examples
    --------
    Dummy example
    >>> import pandas as pd
    >>> df1 = spark.createDataFrame(pd.DataFrame({"a": [1, 2, 3], "b": [[1,2], [3,4], [5, 6]]}))
    >>> df2 = spark.createDataFrame(pd.DataFrame({"a": [1, 3, 4], "b": [[10,20], [30,40], [50, 60]]}))
    >>> df_join = join_aggregated_sso_data(df1, df2, on="a")
    >>> df_join.show()
    +---+--------------+
    |  a|             b|
    +---+--------------+
    |  1|[1, 2, 10, 20]|
    |  2|        [3, 4]|
    |  3|[5, 6, 30, 40]|
    |  4|      [50, 60]|
    +---+--------------+
    <BLANKLINE>

    SSO example
    >>> path = "fink_utils/test_data/benoit_julien_2025/science"
    >>> df_new = aggregate_ztf_sso_data(year=2025, month=1, prefix_path=path)
    >>> path = "fink_utils/test_data/agg_benoit_julien_2024"
    >>> df_prev = spark.read.format("parquet").load(path)

    >>> df_join = join_aggregated_sso_data(df_prev, df_new, on="ssnamenr")
    >>> assert df_join.count() == 2

    >>> inp = df_prev.filter(df_prev["ssnamenr"] == "8467").collect()
    >>> len_inp = len(inp[0]["cfid"])
    >>> out = df_join.filter(df_prev["ssnamenr"] == "8467").collect()
    >>> len_out = len(out[0]["cfid"])
    >>> assert len_out == len_inp + 3, (len_out, len_inp)

    >>> inp = df_prev.filter(df_prev["ssnamenr"] == "33803").collect()
    >>> len_inp = len(inp[0]["cfid"])
    >>> out = df_join.filter(df_prev["ssnamenr"] == "33803").collect()
    >>> len_out = len(out[0]["cfid"])
    >>> assert len_out == len_inp, (len_out, len_inp)
    """
    assert len([i for i in df_new.columns if i not in df_prev.columns]) == 0, (
        df_prev.columns,
        df_new.columns,
    )

    # join
    df_join = df_prev.join(
        df_new.withColumnsRenamed({
            col: col + "_r" for col in df_new.columns if col != on
        }),
        on=on,
        how="outer",
    )

    # concatenate
    df_concatenated = df_join.withColumns({
        col: F.when(F.col(col + "_r").isNull(), F.col(col)).otherwise(
            F.when(F.col(col).isNull(), F.col(col + "_r")).otherwise(
                F.concat(F.col(col), F.col(col + "_r"))
            )
        )
        for col in df_new.columns
        if col != on
    })

    return df_concatenated.select(df_new.columns)


def aggregate_ztf_sso_data(
    year,
    month=None,
    stop_previous_month=False,
    prefix_path="archive/science",
    output_filename=None,
):
    """Aggregate ZTF SSO data in Fink

    Parameters
    ----------
    year: str
        Year date in format YYYY.
    month: str, optional
        Month date in format MM. Default is None, in
        which case `year` only will be considered.
    stop_previous_month: bool, optional
        If True, load data only until previous month.
        To use only with month=None, to reconstruct
        data from the current year.
    prefix_path: str, optional
        Prefix path on HDFS. Default is archive/science
    output_filename: str, optional
        If given, save data on HDFS. Cannot overwrite. Default is None.

    Returns
    -------
    df_grouped: Spark DataFrame
        Spark DataFrame with aggregated SSO data.

    Examples
    --------
    >>> path = "fink_utils/test_data/benoit_julien_2025/science"

    Check monthly aggregation
    >>> df_agg = aggregate_ztf_sso_data(year=2025, month=1, prefix_path=path)
    >>> assert df_agg.count() == 1, df_agg.count()

    >>> out = df_agg.collect()
    >>> assert len(out[0]["cfid"]) == 3, len(out[0]["cfid"])

    Check yearly aggregation
    >>> df_agg = aggregate_ztf_sso_data(year=2025, prefix_path=path)
    >>> assert df_agg.count() == 1, df_agg.count()

    >>> out = df_agg.collect()
    >>> assert len(out[0]["cfid"]) == 3, len(out[0]["cfid"])
    """
    spark = SparkSession.builder.getOrCreate()
    cols0 = ["candidate.ssnamenr"]
    cols = [
        "candidate.ra",
        "candidate.dec",
        "candidate.magpsf",
        "candidate.sigmapsf",
        "candidate.fid",
        "candidate.jd",
    ]

    if month is not None:
        path = "{}/year={}/month={}".format(prefix_path, year, month)
    else:
        path = "{}/year={}".format(prefix_path, year)

    df = spark.read.format("parquet").option("basePath", prefix_path).load(path)

    if month is None and stop_previous_month:
        prevdate = retrieve_last_date_of_previous_month(datetime.datetime.today())
        # take the last hour of the last day
        prevdate = prevdate.replace(hour=23)
        jd0 = Time(prevdate, format="datetime").jd
        df = df.filter(df["candidate.jd"] <= jd0)

    df_agg = (
        df.select(cols0 + cols)
        .filter(F.col("roid") == 3)
        .groupBy("ssnamenr")
        .agg(*[
            F.collect_list(col.split(".")[1]).alias("c" + col.split(".")[1])
            for col in cols
        ])
    )

    if output_filename is not None:
        df_agg.write.parquet(output_filename)

    return df_agg


if __name__ == "__main__":
    """Execute the unit test suite"""

    # Run the Spark test suite
    spark_unit_tests(globals())

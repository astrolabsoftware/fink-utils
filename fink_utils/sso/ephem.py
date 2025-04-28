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
"""Get ephemerides at scales"""

import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import MapType, StringType, FloatType, ArrayType

from fink_utils.sso.miriade import query_miriade_ephemcc
from fink_utils.sso.miriade import query_miriade

from fink_utils.tester import spark_unit_tests


COLUMNS = ["Dobs", "Dhelio", "Phase", "Elong.", "RA", "DEC"]


def sanitize_name(col):
    """Remove trailing '.' from names"""
    return col.replace(".", "")


def expand_columns(df, col_to_expand="ephem"):
    """Expand a MapType column into individual columns

    Notes
    -----
    The operation will transform a dataframe with columns
    ["toto", "container.col1", container.col2] to a dataframe with columns
    ["toto", "col1", "col2"]
    Note that `col_to_expand` is dropped.

    Parameters
    ----------
    df: Spark DataFrame
        Spark DataFrame with the map column
    col_to_expand: str
        Name of the column to expand

    Returns
    -------
    out: Spark DataFrame
        The expanded input DataFrame

    Examples
    --------
    >>> pdf = pd.DataFrame({"a": [{"Dobs": 1, "Elong": 2}, {"Dobs": 10, "Elong": 20}]})
    >>> df = spark.createDataFrame(pdf)
    >>> assert "a" in df.columns, df.columns
    >>> assert "Dobs" not in df.columns, df.columns
    >>> df = expand_columns(df, col_to_expand="a")
    >>> assert "Dobs" in df.columns, df.columns
    >>> assert "a" not in df.columns, df.columns
    """
    if col_to_expand not in df.columns:
        print(
            "{} not found in the DataFrame columns. Have you computed ephemerides?".format(
                col_to_expand
            )
        )
        return df
    for col in COLUMNS:
        df = df.withColumn(
            sanitize_name(col), df["{}.{}".format(col_to_expand, sanitize_name(col))]
        )
    df = df.drop(col_to_expand)
    return df


@pandas_udf(MapType(StringType(), ArrayType(FloatType())), PandasUDFType.SCALAR)
def extract_ztf_ephemerides_from_miriade(ssnamenr, cjd, uid, method):
    """Extract ephemerides for ZTF from Miriade

    Parameters
    ----------
    ssnamenr: pd.Series of str
        ZTF ssnamenr
    cjd: pd.Series of list of floats
        List of JD values
    uid: pd.Series of int
        Unique ID for each object
    method: pd.Series of str
        Method to compute ephemerides: `ephemcc` or `rest`.
        Use only the former on the Spark Cluster (local installation of ephemcc),
        otherwise use `rest` to call the ssodnet web service.

    Returns
    -------
    out: pd.Series of dictionaries of lists

    Examples
    --------
    >>> import pyspark.sql.functions as F

    Basic ephemerides computation
    >>> path = "fink_utils/test_data/agg_benoit_julien_2024"
    >>> df_prev = spark.read.format("parquet").load(path)

    >>> df_prev_ephem = df_prev.withColumn(
    ...     "ephem",
    ...     extract_ztf_ephemerides_from_miriade(
    ...         "ssnamenr",
    ...         "cjd",
    ...         F.expr("uuid()"),
    ...         F.lit("rest")))

    >>> df_prev_ephem = expand_columns(df_prev_ephem)
    >>> out = df_prev_ephem.select(["cjd", "Dobs"]).collect()
    >>> assert len(out[0]["cjd"]) == len(out[0]["Dobs"])
    >>> assert len(out[1]["cjd"]) == len(out[1]["Dobs"])

    Aggregation of ephemerides
    >>> from fink_utils.sso.ssoft import aggregate_ztf_sso_data
    >>> path = "fink_utils/test_data/benoit_julien_2025/science"
    >>> df_new = aggregate_ztf_sso_data(year=2025, month=1, prefix_path=path)

    >>> df_new_ephem = df_new.withColumn(
    ...     "ephem",
    ...     extract_ztf_ephemerides_from_miriade(
    ...         "ssnamenr",
    ...         "cjd",
    ...         F.expr("uuid()"),
    ...         F.lit("rest")))
    >>> df_new_ephem = expand_columns(df_new_ephem)
    >>> out = df_new_ephem.select(["cjd", "RA"]).collect()
    >>> assert len(out[0]["cjd"]) == len(out[0]["RA"])

    Checking rolling add
    >>> from fink_utils.sso.ssoft import join_aggregated_sso_data
    >>> df_join = join_aggregated_sso_data(df_prev, df_new, on="ssnamenr")
    >>> df_join_ephem = df_join.withColumn(
    ...     "ephem",
    ...     extract_ztf_ephemerides_from_miriade(
    ...         "ssnamenr",
    ...         "cjd",
    ...         F.expr("uuid()"),
    ...         F.lit("rest")))
    >>> df_join_ephem = expand_columns(df_join_ephem)

    >>> df_join_ephem_bis = join_aggregated_sso_data(df_prev_ephem, df_new_ephem, on="ssnamenr")
    >>> out_1 = df_join_ephem.select(["Elong"]).collect()
    >>> out_2 = df_join_ephem_bis.select(["Elong"]).collect()
    >>> assert out_1 == out_2, (out_1, out_2)
    """
    method_ = method.to_numpy()[0]
    out = []
    for index, ssname in enumerate(ssnamenr.to_numpy()):
        if method_ == "ephemcc":
            # Hardcoded!
            parameters = {
                "outdir": "/tmp/ramdisk/spins",
                "runner_path": "/tmp/fink_run_ephemcc4.sh",
                "userconf": "/tmp/.eproc-4.3",
                "iofile": "/tmp/default-ephemcc-observation.xml",
            }
            ephems = query_miriade_ephemcc(
                ssname,
                cjd.to_numpy()[index],
                observer="I41",
                rplane="1",
                tcoor=5,
                shift=15.0,
                parameters=parameters,
                uid=uid.to_numpy()[index],
                return_json=True,
            )
        else:
            ephems = query_miriade(
                ssname,
                cjd.to_numpy()[index],
                observer="I41",
                rplane="1",
                tcoor=5,
                shift=15.0,
                timeout=30,
                return_json=True,
            )
        if ephems.get("data", None) is not None:
            # Remove any "." in name
            ephems_corr = {
                sanitize_name(k): [dic[k] for dic in ephems["data"]] for k in COLUMNS
            }

            # In-place transformation of RA/DEC coordinates
            sc = SkyCoord(ephems_corr["RA"], ephems_corr["DEC"], unit=(u.deg, u.deg))
            ephems_corr["RA"] = sc.ra.value * 15
            ephems_corr["DEC"] = sc.dec.value

            out.append(ephems_corr)
        else:
            # Not sure about that
            out.append({})

    return pd.Series(out)


if __name__ == "__main__":
    """Execute the unit test suite"""

    # Run the Spark test suite
    spark_unit_tests(globals())

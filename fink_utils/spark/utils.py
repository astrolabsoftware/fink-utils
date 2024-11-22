# Copyright 2020-2024 AstroLab Software
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
import numpy as np
import pandas as pd

import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    StructType,
    MapType,
    StringType,
    FloatType,
)
from pyspark.sql import DataFrame

import importlib
import logging

_LOG = logging.getLogger(__name__)


def concat_col(
    df,
    colname: str,
    prefix: str = "c",
    current: str = "candidate",
    history: str = "prv_candidates",
):
    """Concatenate historical and current measurements

    Add new column to the DataFrame named `prefix`+`colname`, containing
    the concatenation of historical and current measurements.

    Parameters
    ----------
    df: DataFrame
        Pyspark DataFrame containing alert data
    colname: str
        Name of the column to add (without the prefix)
    prefix: str
        Additional prefix to add to the column name. Default is 'c'.
    current: str
        Name of the field containing current `colname` measurement, to extract
        `current.colname`. Usually a struct type field. Default is `candidate`
        from ZTF schema.
    history: str
        Name of the field containing history for `colname` measurements,
        to extract `history.colname`. Usually a list of struct type field.
        Default is `prv_candidates` from ZTF schema.

    Returns
    -------
    df: DataFrame
        Dataframe with new column containing the concatenation of
        historical and current measurements.
    """
    return df.withColumn(
        prefix + colname,
        F.when(
            df["{}.{}".format(history, colname)].isNotNull(),
            F.concat(
                df["{}.{}".format(history, colname)],
                F.array(df["{}.{}".format(current, colname)]),
            ),
        ).otherwise(F.array(df["{}.{}".format(current, colname)])),
    )


def extract_values(cmagpsf, cdiffmaglim, onlyfainterlimits=False):
    """Extract the first upper values before measurements start

    Parameters
    ----------
    cmagpsf: np.array
        Array of magpsf (history + current)
    cdiffmaglim: np.array
        Array of diffmaglim (history + current)
    onlyfainterlimits: bool, optional
        If True, only consider upper limits that are fainter than the
        first measurement. Default False

    Returns
    -------
    val: float
        Last diffmaglim values before measurements start

    Examples
    --------
    >>> cmagpsf = [np.nan, np.nan, 1.0, 2.0, np.nan, 3.0]
    >>> cdiffmaglim = [0.3, 0.2, 0.4, 0.7, 0.4, 0.2]
    >>> val = extract_values(cmagpsf, cdiffmaglim)
    >>> assert val == 0.2
    """
    if len(cmagpsf) == 0:
        return np.nan

    if not np.isnan(cmagpsf[0]):
        # young transient must start with NaN
        return np.nan

    if np.alltrue(np.isnan(cmagpsf)):
        # all NaNs
        return np.nan

    # 0..N_alert
    array_indices = np.arange(len(cmagpsf))

    # Position of the last upper values
    first_mag = np.nan
    pos = 0
    while np.isnan(first_mag) and pos < len(cmagpsf):
        first_mag = cmagpsf[pos]
        pos += 1

    if (pos == len(cmagpsf)) and np.isnan(first_mag):
        return np.nan

    # Arrays with only upper values
    diffmaglim = cdiffmaglim[array_indices <= pos - 2]

    # Selecting only fainter limits if required
    if onlyfainterlimits:
        diffmaglim = diffmaglim[diffmaglim > first_mag]

    if len(diffmaglim) == 0:
        return np.nan
    else:
        return diffmaglim[-1]


@pandas_udf(MapType(StringType(), ArrayType(FloatType())), PandasUDFType.SCALAR)
def extend_lc_with_upper_limits(
    cmagpsf, csigmapsf, cfid, cdiffmaglim, onlyfainterlimits=False
):
    """Extend valid measurements with the last upper limit for each band

    Notes
    -----
    The extended time-series can have:
    1. new measurements in r & g
    2. new measurements in r or g
    3. no new measurements

    In the case of (2) or (3), missing addition means the history
    already starts with a valid measurement, and there is nothing to
    be further added.

    Parameters
    ----------
    cmagpsf: pd.Series of np.array
        Series of arrays of magpsf (history + current)
    csigmapsf: pd.Series of np.array
        Series of arrays of sigmapsf (history + current)
    cfid: pd.Series of np.array
        Series of arrays of fid (history + current)
    cdiffmaglim: pd.Series of np.array
        Series of arrays of diffmaglim (history + current)
    onlyfainterlimits: bool, optional
        If True, only consider upper limits that are fainter than the
        first measurement. Default False

    Returns
    -------
    pd.Series
        new column with dictionaries containing extended
        time-series (`cmagpsf_ext` & `csigmapsf_ext`). Note that
        added values for sigmapsf is always 0.2 mag.

    Examples
    --------
    >>> to_expand = ["jd", "fid", "magpsf", "sigmapsf", "diffmaglim", "sigmapsf"]
    >>> prefix = "c"
    >>> for colname in to_expand:
    ...     df = concat_col(df, colname, prefix=prefix)

    >>> df = spark.read.format("parquet").load("datatest")
    >>> df = df.withColumn("ext", extend_lc_with_upper_limits(
    ...     "cmagpsf", "csigmapsf", "cfid", "cdiffmaglim"))
    >>> df = df.withColumn("cmagpsf_ext", df["ext"].getItem("cmagpsf_ext"))
    >>> df = df.withColumn("csigmapsf_ext", df["ext"].getItem("csigmapsf_ext"))

    >>> pdf = df.select(["cmagpsf", "cmagpsf_ext"]).toPandas()

    >>> cmagpsf, cmagpsf_ext = pdf.to_numpy()[0]
    >>> assert len(cmagpsf) == len(cmagpsf_ext) # they have the same size

    >>> assert len(cmagpsf[~np.isnan(cmagpsf)]) <= len(cmagpsf_ext[~np.isnan(cmagpsf_ext)])
    """
    if not isinstance(onlyfainterlimits, bool):
        # Series to boolean
        onlyfainterlimits = True
    out = []
    for index in range(len(cmagpsf)):
        row = {"cmagpsf_ext": cmagpsf[index], "csigmapsf_ext": csigmapsf[index]}
        for fid in [1, 2]:
            mask = cfid[index] == fid
            val = extract_values(
                cmagpsf[index][mask],
                cdiffmaglim[index][mask],
                onlyfainterlimits=onlyfainterlimits,
            )
            if not np.isnan(val):
                offset = np.where(cdiffmaglim[index] == val)[0][0]
                row["cmagpsf_ext"][offset] = val
                row["csigmapsf_ext"][offset] = 0.2

        out.append(row)

    return pd.Series(out)


def return_flatten_names(
    df: DataFrame, pref: str = "", flatten_schema: list = None
) -> list:
    """Retrieve full path from a nested schema

    Example, if one has a nested structure such as:
    root
     |-- timestamp: timestamp (nullable = true)
     |-- decoded: struct (nullable = true)
     |    |-- schemavsn: string (nullable = true)
     |    |-- publisher: string (nullable = true)
     |    |-- objectId: string (nullable = true)
     |    |-- candid: long (nullable = true)
     |    |-- candidate: struct (nullable = true)

    It will return a list like
        ["timestamp", "decoded" ,"decoded.schemavsn", "decoded.publisher", ...]

    Parameters
    ----------
    df : DataFrame
        Alert DataFrame
    pref : str, optional
        Internal variable to keep track of the structure, initially sets to "".
    flatten_schema: list, optional
        List containing the names of the flatten schema names.
        Initially sets to [].

    Returns
    -------
    flatten_frame: list
        List containing the names of the flatten schema names.

    Examples
    --------
    >>> df = spark.read.format("parquet").load("datatest")
    >>> flatten_schema = return_flatten_names(df)
    >>> assert("candidate.candid" in flatten_schema)
    """
    if (flatten_schema == []) or (flatten_schema is None):
        for colname in df.columns:
            flatten_schema.append(colname)  # noqa: PERF402

    # If the entry is not top level, it is then hidden inside a nested structure
    l_struct_names = [i.name for i in df.schema if isinstance(i.dataType, StructType)]

    for l_struct_name in l_struct_names:
        colnames = df.select("{}.*".format(l_struct_name)).columns
        for colname in colnames:
            if pref == "":
                flatten_schema.append(".".join([l_struct_name, colname]))
            else:
                flatten_schema.append(".".join([pref, l_struct_name, colname]))

        # Check if there are other levels nested
        flatten_schema = return_flatten_names(
            df.select("{}.*".format(l_struct_name)),
            pref=l_struct_name,
            flatten_schema=flatten_schema,
        )

    return flatten_schema


def apply_user_defined_filter(df: DataFrame, toapply: str, logger=None) -> DataFrame:
    """Apply a user filter to keep only wanted alerts.

    Parameters
    ----------
    df: DataFrame
        Spark DataFrame with alert data
    toapply: string
        Filter name to be applied. It should be in the form
        module.module.routine (see example below).

    Returns
    -------
    df: DataFrame
        Spark DataFrame with filtered alert data

    Examples
    --------
    >>> from pyspark.sql.functions import struct
    >>> colnames = ["cdsxmatch", "rb", "magdiff"]
    >>> df = spark.sparkContext.parallelize(zip(
    ...   ['RRLyr', 'Unknown', 'Star', 'SN1a'],
    ...   [0.01, 0.02, 0.6, 0.01],
    ...   [0.02, 0.05, 0.1, 0.01])).toDF(colnames)
    >>> df.show() # doctest: +NORMALIZE_WHITESPACE
    +---------+----+-------+
    |cdsxmatch|  rb|magdiff|
    +---------+----+-------+
    |    RRLyr|0.01|   0.02|
    |  Unknown|0.02|   0.05|
    |     Star| 0.6|    0.1|
    |     SN1a|0.01|   0.01|
    +---------+----+-------+
    <BLANKLINE>


    # Nest the DataFrame as for alerts
    >>> df = df.select(struct(df.columns).alias("candidate"))\
        .select(struct("candidate").alias("decoded"))

    # Apply quality cuts for example (level one)
    >>> toapply = 'fink_filters.filter_rrlyr.filter.rrlyr'
    >>> df = apply_user_defined_filter(df, toapply)
    >>> df.select("decoded.candidate.*").show() # doctest: +NORMALIZE_WHITESPACE
    +---------+----+-------+
    |cdsxmatch|  rb|magdiff|
    +---------+----+-------+
    |    RRLyr|0.01|   0.02|
    +---------+----+-------+
    <BLANKLINE>

    # Using a wrong filter name will lead to an error
    >>> df = apply_user_defined_filter(
    ...   df, "unknownfunc") # doctest: +SKIP
    """
    flatten_schema = return_flatten_names(df, pref="", flatten_schema=[])

    # Load the filter
    filter_name = toapply.split(".")[-1]
    module_name = toapply.split("." + filter_name)[0]
    module = importlib.import_module(module_name)
    filter_func = getattr(module, filter_name, None)

    # Note: to access input argument, we need f.func and not just f.
    # This is because f has a decorator on it.
    ninput = filter_func.func.__code__.co_argcount

    # Note: This works only with `struct` fields - not `array`
    argnames = filter_func.func.__code__.co_varnames[:ninput]
    colnames = []
    for argname in argnames:
        colname = [F.col(i) for i in flatten_schema if i.endswith("{}".format(argname))]
        if len(colname) == 0:
            raise AssertionError(
                """
                Column name {} is not a valid column of the DataFrame.
                """.format(argname)
            )
        colnames.append(colname[0])

    if logger is not None:
        logger.info(
            "new filter/topic registered: {} from {}".format(filter_name, module_name)
        )

    return (
        df.withColumn("toKeep", filter_func(*colnames))
        .filter("toKeep == true")
        .drop("toKeep")
    )


@pandas_udf(ArrayType(BooleanType()), PandasUDFType.SCALAR)
def apply_quality_flags_on_history(rb, nbad):
    """Apply quality flags for the history vector

    Parameters
    ----------
    rb: Series
        Pandas Series of list of float containing
        `rb` values from `prv_candidates`
    nbad: Series
        Pandas Series of list of int containing
        `nbad` values from `prv_candidates`

    Returns
    -------
    out: Series
        Pandas Series of list of boolean. True if good candidate
        False otherwise.

    Examples
    --------
    >>> df = spark.read.format("parquet").load("fink_utils/test_data/online")
    >>> df = df.withColumn(
    ...     'history_flag',
    ...     apply_quality_flags_on_history(
    ...         F.col("prv_candidates.rb"),
    ...         F.col("prv_candidates.nbad")))
    """
    rbf = rb.apply(lambda x: [i >= 0.55 for i in x])
    nbadf = nbad.apply(lambda x: [i == 0 for i in x])

    f = [np.array(i) * np.array(j) for i, j in zip(rbf.to_list(), nbadf.to_list())]
    return pd.Series(f)


def check_status_last_prv_candidates(df, status="valid"):
    """Check the `status` of the last alert in the history

    Parameters
    ----------
    df: Spark DataFrame
        Alert DataFrame (full schema)
    status: str
        Status can be:
            - valid: alert was processed by Fink
            - uppervalid: alert was not processed by Fink of quality cuts*
            - upper: upper limit from ZTF

    Returns
    -------
    out: Spark DataFrame
        Input DataFrame with an extra column (boolean) named `status`.

    Examples
    --------
    >>> df = spark.read.format("parquet").load("fink_utils/test_data/online")
    >>> print(df.count())
    11

    >>> df = check_status_last_prv_candidates(df, 'valid')
    >>> print(df.filter(F.col("valid")).count())
    11

    >>> df = check_status_last_prv_candidates(df, 'uppervalid')
    >>> print(df.filter(F.col("uppervalid")).count())
    0

    >>> df = check_status_last_prv_candidates(df, 'upper')
    >>> print(df.filter(F.col("upper")).count())
    0
    """
    # measurements are ordered from the most ancient to the newest
    # we want to check the last one
    rb = F.element_at("prv_candidates.rb", -1)
    nbad = F.element_at("prv_candidates.nbad", -1)
    magpsf = F.element_at("prv_candidates.magpsf", -1)
    if status == "valid":
        f1 = (rb >= 0.55) & (nbad == 0)
        cond = f1 & magpsf.isNotNull()
    elif status == "uppervalid":
        f1 = (rb >= 0.55) & (nbad == 0)
        cond = ~f1 & magpsf.isNotNull()
    elif status == "upper":
        cond = ~magpsf.isNotNull()

    return df.withColumn(status, cond)

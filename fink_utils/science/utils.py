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
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StringType, LongType

import pandas as pd
import numpy as np
import healpy as hp


def dec2theta(dec: float) -> float:
    """Convert Dec (deg) to theta (rad)"""
    return np.pi / 2.0 - np.pi / 180.0 * dec


def ra2phi(ra: float) -> float:
    """Convert RA (deg) to phi (rad)"""
    return np.pi / 180.0 * ra


@pandas_udf(LongType(), PandasUDFType.SCALAR)
def ang2pix(ra: pd.Series, dec: pd.Series, nside: pd.Series) -> pd.Series:
    """Compute pixel number at given nside

    Parameters
    ----------
    ra: float
        Spark column containing RA (float)
    dec: float
        Spark column containing RA (float)
    nside: int
        Spark column containing nside

    Returns
    -------
    out: long
        Spark column containing pixel number

    Examples
    --------
    >>> from fink_broker.sparkUtils import load_parquet_files
    >>> df = load_parquet_files(ztf_alert_sample)
    >>> df_index = df.withColumn(
    ...     'p',
    ...     ang2pix(df['candidate.ra'], df['candidate.dec'], F.lit(256))
    ... )
    >>> df_index.select('p').take(1)[0][0] > 0
    True
    """
    return pd.Series(
        hp.ang2pix(
            nside.to_numpy()[0], dec2theta(dec.to_numpy()), ra2phi(ra.to_numpy())
        )
    )


@pandas_udf(StringType(), PandasUDFType.SCALAR)
def ang2pix_array(ra: pd.Series, dec: pd.Series, nside: pd.Series) -> pd.Series:
    """Return a col string with the pixel numbers corresponding to the nsides pix@nside[0]_pix@nside[1]_...etc

    Parameters
    ----------
    ra: float
        Spark column containing RA (float)
    dec: float
        Spark column containing RA (float)
    nside: list
        Spark column containing list of nside

    Returns
    -------
    out: str
        Spark column containing _ separated pixel values

    Examples
    --------
    >>> from fink_broker.sparkUtils import load_parquet_files
    >>> df = load_parquet_files(ztf_alert_sample)
    >>> nsides = F.array([F.lit(256), F.lit(4096), F.lit(131072)])
    >>> df_index = df.withColumn(
    ...     'p',
    ...     ang2pix_array(df['candidate.ra'], df['candidate.dec'], nsides)
    ... )
    >>> l = len(df_index.select('p').take(1)[0][0].split('_'))
    >>> print(l)
    3
    """
    pixs = [
        hp.ang2pix(int(nside_), dec2theta(dec.to_numpy()), ra2phi(ra.to_numpy()))
        for nside_ in nside.to_numpy()[0]
    ]

    to_return = ["_".join(list(np.array(i, dtype=str))) for i in np.transpose(pixs)]

    return pd.Series(to_return)


# if __name__ == "__main__":
#     """ Execute the test suite with SparkSession initialised """

#     globs = globals()
#     root = os.environ['FINK_HOME']
#     globs["ztf_alert_sample"] = os.path.join(
#         root, "online/raw")

#     globs['elasticc_alert_sample'] = os.path.join(
#         root, "elasticc_parquet")

#     # Run the Spark test suite
#     spark_unit_tests(globs)

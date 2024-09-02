# Copyright 2022 AstroLab Software
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
import json

from pyspark.sql import functions as F
from pyspark.sql import DataFrame


def load_hbase_catalog_as_dict(path_to_catalog: str) -> dict:
    """Load HBase table catalog from disk as dictionary

    Parameters
    ----------
    path_to_catalog: str
        Path to the json file containing the HBase table catalog

    Returns
    -------
    dcat: dict
        Dictionary containing the table catalog
    rowkey: str
        Name of the rowkey
    """
    with open(path_to_catalog) as f:
        catalog = json.load(f)

    dcat = json.loads(catalog)
    rowkey = [i["col"] for i in dcat["columns"].values() if i["cf"] == "rowkey"][0]

    return dcat, rowkey


def select_columns_in_catalog(catalog: dict, cols: list) -> (dict, str):
    """Select only `cols` in the catalog

    Parameters
    ----------
    catalog: dict
        Dictionary containing the HBase table catalog
    cols: list of str
        List of column names to keep

    Returns
    -------
    dcat_small: dict
        Same as input, but with fewer columns
    catalog_small: str
        json str view of the above
    """
    dcat_small = {}

    for k in catalog.keys():
        if k != "columns":
            dcat_small[k] = catalog[k]
        else:
            dcat_small[k] = {k_: v_ for k_, v_ in catalog[k].items() if k_ in cols}

    catalog_small = json.dumps(dcat_small)

    return dcat_small, catalog_small


def group_by_key(df: DataFrame, key: str, position: int, sep="_") -> DataFrame:
    """Group by the input `df` by split(`key`, '_')[`position`]

    Parameters
    ----------
    df: DataFrame
        Input dataframe
    key: str
        Row to groupby against. Better performance if rowkey.
    position: int
        Nth element to return from the list after the split of `key` is performed
        Zero-based index: First is 0. -1 means last.
    sep: str
        Separator to use for splitting `key`. Default is `_`.

    Returns
    -------
    df_grouped: DataFrame
        Result of the group by with two columns `id`, `count`.
        `id` = split(`key`, '_')[`position`]
    """
    if position >= 0:
        # The position is not zero based internally, but 1 based index.
        position += 1
    # Groupby key
    df_grouped = (
        df.select(F.element_at(F.split(df[key], "_"), position).alias("id"))
        .groupby("id")
        .count()
    )

    return df_grouped

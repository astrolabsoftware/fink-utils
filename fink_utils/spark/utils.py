# Copyright 2020-2022 AstroLab Software
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
from pyspark.sql import functions as F

def concat_col(
        df, colname: str, prefix: str = 'c',
        current: str = 'candidate', history: str = 'prv_candidates'):
    """ Add new column to the DataFrame named `prefix`+`colname`, containing
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
    ----------
    df: DataFrame
        Dataframe with new column containing the concatenation of
        historical and current measurements.
    """
    return df.withColumn(
        prefix + colname,
        F.when(
            df['{}.{}'.format(history, colname)].isNotNull(),
            F.concat(
                df['{}.{}'.format(history, colname)],
                F.array(df['{}.{}'.format(current, colname)])
            )
        ).otherwise(F.array(df['{}.{}'.format(current, colname)]))
    )

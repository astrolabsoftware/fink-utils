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
"""Contains functionalities to work with HDFS"""

from pyspark.sql import SparkSession


def path_exist(path: str) -> bool:
    """Check if a path exists on Spark shared filesystem (HDFS or S3)

    Parameters
    ----------
    path : str
        Path to check

    Returns
    -------
    bool
        True if the path exists, False otherwise
    """
    spark = SparkSession.builder.getOrCreate()

    jvm = spark._jvm
    jsc = spark._jsc

    conf = jsc.hadoopConfiguration()
    uri = jvm.java.net.URI(path)

    fs = jvm.org.apache.hadoop.fs.FileSystem.get(uri, conf)

    path_glob = jvm.org.apache.hadoop.fs.Path(path)
    status_list = fs.globStatus(path_glob)

    if status_list is None:
        return False

    # Not clear what is the type of status_list
    # in general as it is a Java object
    if len(list(status_list)) > 0:
        return True
    else:
        return False

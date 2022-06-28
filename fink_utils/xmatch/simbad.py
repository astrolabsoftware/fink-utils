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
import pandas as pd

def return_list_of_eg_host():
    """ Return potential SN host names

    This includes:
    - List of object names in SIMBAD that would correspond to extra-galactic object
    - Unknown objects
    - objects with failed crossmatch

    In practice, this exclude galactic objects from SIMBAD.

    """
    list_simbad_galaxies = [
        "galaxy",
        "Galaxy",
        "EmG",
        "Seyfert",
        "Seyfert_1",
        "Seyfert_2",
        "BlueCompG",
        "StarburstG",
        "LSB_G",
        "HII_G",
        "High_z_G",
        "GinPair",
        "GinGroup",
        "BClG",
        "GinCl",
        "PartofG",
    ]

    keep_cds = \
        ["Unknown", "Candidate_SN*", "SN", "Transient", "Fail"] + \
        list_simbad_galaxies

    return keep_cds

def read_conversion_dic(path: str) -> pd.DataFrame:
    """ Read the file containing the mapping between old and new otypes

    Parameters
    ----------
    path: str
        Path to the file. Can be an URL:
        https://simbad.cds.unistra.fr/guide/otypes.labels.txt

    Returns
    ----------
    pdf: pd.DataFrame
        Data formatted in a pandas DataFrame: otype, old_label, new_label

    Examples
    ----------
    >>> path = 'https://simbad.cds.unistra.fr/guide/otypes.labels.txt'
    >>> pdf = read_conversion_dic(path)
    >>> print(len(pdf))
    199
    """
    pdf = pd.read_csv(
        path,
        sep='|',
        skiprows=[0, 1, 3],
        skipfooter=2,
        dtype='str',
        header=0
    )

    pdf = pdf.rename(columns={i: i.strip() for i in pdf.columns})

    pdf = pdf[['otype', 'old_label', 'new_label']]

    pdf = pdf.applymap(lambda x: x.strip())

    return pdf


if __name__ == "__main__":
    """ Execute the test suite """
    import sys
    import doctest

    sys.exit(doctest.testmod()[0])

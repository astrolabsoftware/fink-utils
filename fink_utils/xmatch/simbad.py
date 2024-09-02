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
import os
import pandas as pd
import numpy as np

from urllib.error import URLError

from fink_utils import __file__


def return_list_of_eg_host(full_simbad_conversion=False) -> list:
    """Return potential SN host names

    This includes:
    - List of object names in SIMBAD that would correspond to extra-galactic object
    - Unknown objects
    - objects with failed crossmatch

    In practice, this exclude galactic objects from SIMBAD.

    Parameters
    ----------
    full_simbad_conversion: bool
        If True, download the file containing the taxonomy change, and
        include old and new taxonomy in the output. If False, we only add
        manually the new labels we know have changed. In principle, we make sure
        that we do not miss ones, but if the taxonomy changed, the output could
        be incomplete. The former is in principle the most robust technics, but
        it is much slower (x1000) than the latter. Default is False.

    Returns
    -------
    out: list
        List of labels

    Examples
    --------
    >>> gals = return_list_of_eg_host(full_simbad_conversion=True)
    >>> print(len(gals))
    42

    >>> gals = return_list_of_eg_host(full_simbad_conversion=False)
    >>> print(len(gals))
    30
    """
    # old taxonomy
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
        "Compact_Gr_G",
        "IG",
        "PairG",
        "GroupG",
        "ClG",
        "SuperClG",
        "Void",
    ]

    cds = [
        "Unknown",
        "Candidate_SN*",
        "SN",
        "Transient",
        "Fail",
        "Fail 504",
    ] + list_simbad_galaxies

    if full_simbad_conversion:
        conv = get_conversion_dic()
        cds_with_new_taxonomy = [old2new(conv, i) for i in cds]
    else:
        # fields that really changed
        cds_with_new_taxonomy = ["SN*_Candidate"]

    out = np.concatenate((cds, cds_with_new_taxonomy))

    return np.unique(out)


def get_conversion_dic(path: str = None, remove_unknown: bool = True) -> pd.DataFrame:
    """Read the file containing the mapping between old and new otypes

    Parameters
    ----------
    path: str
        Path to the file. Can be an URL:
        https://simbad.cds.unistra.fr/guide/otypes.labels.txt
    remove_unknown: bool
        If True, remove the row containing the class `Unknown` from
        the DataFrame. Default is True.

    Returns
    -------
    pdf: pd.DataFrame
        Data formatted in a pandas DataFrame: otype, old_label, new_label

    Examples
    --------
    >>> pdf = get_conversion_dic(remove_unknown=False)
    >>> print(len(pdf))
    199

    >>> pdf = get_conversion_dic()
    >>> print(len(pdf))
    198
    """
    if path is None:
        path = "https://simbad.cds.unistra.fr/guide/otypes.labels.txt"

    try:
        pdf = pd.read_csv(
            path,
            sep="|",
            skiprows=[0, 1, 3],
            skipfooter=2,
            dtype="str",
            header=0,
            engine="python",
        )
    except URLError:
        curdir = os.path.dirname(os.path.abspath(__file__))
        fn = curdir + "/xmatch/otypes.txt"
        pdf = pd.read_csv(
            fn,
            sep="|",
            skiprows=[0, 1, 3],
            skipfooter=2,
            dtype="str",
            header=0,
            engine="python",
        )

    pdf = pdf.rename(columns={i: i.strip() for i in pdf.columns})

    pdf = pdf[["otype", "old_label", "new_label"]]

    pdf = pdf.applymap(lambda x: x.strip())

    if remove_unknown:
        pdf = pdf[pdf["old_label"] != "Unknown"]

    return pdf


def old2new(conv, old_label=""):
    """Return the new label name in SIMBAD from an old label

    Parameters
    ----------
    conv: pd.DataFrame
        DataFrame containing all the labels (see `get_conversion_dic`)
    old_label: str
        Old label in SIMBAD

    Returns
    -------
    new_label: str
        New label in SIMBAD corresponding to the old label

    Examples
    --------
    >>> path = 'https://simbad.cds.unistra.fr/guide/otypes.labels.txt'
    >>> pdf = get_conversion_dic(path)
    >>> new_label = old2new(pdf, 'BlueCompG')
    >>> print(new_label)
    BlueCompactG
    """
    out = conv[conv["old_label"] == old_label]["new_label"].to_numpy()

    if len(out) == 1:
        return out[0]
    elif len(out) == 0:
        return ""

    raise ValueError("Label conversion is ambiguous: {} --> {}".format(old_label, out))


def new2old(conv, new_label=""):
    """Return the old label name in SIMBAD from a new label

    Parameters
    ----------
    conv: pd.DataFrame
        DataFrame containing all the labels (see `get_conversion_dic`)
    new_label: str
        New label in SIMBAD

    Returns
    -------
    old_label: str
        Old label in SIMBAD corresponding to the new label

    Examples
    --------
    >>> path = 'https://simbad.cds.unistra.fr/guide/otypes.labels.txt'
    >>> pdf = get_conversion_dic(path)
    >>> old_label = new2old(pdf, 'GtowardsGroup')
    >>> print(old_label)
    GinGroup
    """
    out = conv[conv["new_label"] == new_label]["old_label"].to_numpy()

    if len(out) == 1:
        return out[0]
    elif len(out) == 0:
        return ""

    raise ValueError("Label conversion is ambiguous: {} --> {}".format(new_label, out))


def get_simbad_labels(which: str, remove_unknown=True):
    """Get list of labels in SIMBAD.

    Parameters
    ----------
    which: str
        Choose between: `old`, `new`, `old_and_new`, `otype`.
        `old_and_new` is the unique concatenation of old and new labels.
    remove_unknown: bool
        If True, remove the row containing the class `Unknown` from
        the DataFrame. Default is True.

    Returns
    -------
    out: list
        List of labels (str)


    Examples
    --------
    >>> labels = get_simbad_labels(which='old')
    >>> print(len(labels))
    198

    >>> assert 'Candidate_YSO' in labels

    >>> labels = get_simbad_labels(which='new')
    >>> print(len(labels))
    198

    >>> assert 'YSO_Candidate' in labels

    >>> labels = get_simbad_labels(which='old_and_new')
    >>> print(len(labels))
    315

    >>> assert 'Candidate_YSO' in labels
    >>> assert 'YSO_Candidate' in labels

    >>> labels = get_simbad_labels(which='otype')
    >>> print(len(labels))
    198

    >>> assert 'Y*?' in labels
    """
    pdf = get_conversion_dic(remove_unknown=remove_unknown)
    if which == "old":
        return pdf["old_label"].to_numpy()
    elif which == "new":
        return pdf["new_label"].to_numpy()
    elif which == "old_and_new":
        old = pdf["old_label"].to_numpy()
        new = pdf["new_label"].to_numpy()
        out = np.concatenate((old, new))
        return np.unique(out)
    elif which == "otype":
        return pdf["otype"].to_numpy()


if __name__ == "__main__":
    """Execute the test suite"""
    import sys
    import doctest

    sys.exit(doctest.testmod()[0])

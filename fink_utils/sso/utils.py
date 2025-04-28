# Copyright 2022-2025 AstroLab Software
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
"""Collection of utilities for SSO in Fink"""

import datetime
import re

import pandas as pd
import numpy as np

import astropy.constants as const

from scipy import signal

from fink_utils.tester import regular_unit_tests


def is_peak(x, y, xpeak, band=50):
    """Estimate if `xpeak` corresponds to a true extremum for a periodic signal `y`

    Assuming `y` a sparse signal along `x`, we would first estimate
    the period of the signal assuming a sine wave. We would then generate
    predictions, and locate the extrema of the sine.

    But this first step would generate false positives:
    1. As `y` is sparse, some extrema will not coincide with measurements
    2. As the signal is not a perfect sine, the fitted signal might shift
        from the real signal after several periods.

    This function is an extremely quick and dirty attempt to reduce false
    positives by looking at the data around a fitted peak, and
    estimating if the peak is real:
    1. take a band around the peak, and look if data is present
    2. if data is present, check the data is above the mean

    Parameters
    ----------
    xpeak: int
        Candidate peak position
    x: array
        Array of times
    y: array
        Array of elongation
    band: optional, int
        Bandwidth in units of x

    Returns
    -------
    out: bool
        True if `xpeak` corresponds to the location of a peak.
        False otherwise.
    """
    xband = np.where((x > xpeak - band) & (x < xpeak + band))[0]
    if (len(xband) >= 10) and (np.mean(y[xband]) > np.mean(y)):
        return True
    return False


def get_num_opposition(elong, width=4):
    """Estimate the number of opposition according to the solar elongation

    Under the hood, it assumes `elong` is peroidic, and uses a periodogram.

    Parameters
    ----------
    elong: array
        array of solar elongation corresponding to jd
    width: optional, int
        width of peaks in samples.

    Returns
    -------
    nopposition: int
        Number of oppositions estimate

    Examples
    --------
    >>> import io
    >>> import requests
    >>> import pandas as pd

    >>> r = requests.post(
    ...     'https://api.fink-portal.org/api/v1/sso',
    ...     json={
    ...         'n_or_d': '8467',
    ...         'withEphem': True,
    ...         'output-format': 'json'
    ...     }
    ... )
    >>> pdf = pd.read_json(io.BytesIO(r.content))

    # estimate number of oppositions
    >>> noppositions = get_num_opposition(
    ...     pdf['Elong.'].values,
    ...     width=4
    ... )
    >>> assert noppositions == 3, "Found {} oppositions for 8467 instead of 3!".format(noppositions)
    """
    peaks, _ = signal.find_peaks(elong, width=4)
    return len(peaks)


def compute_light_travel_correction(jd, d_obs):
    """Compute the time with light travel corrected

    Parameters
    ----------
    jd: np.array
        Array of times (JD), in day
    d_obs: np.array
        Array of distance to the observer, in AU
    """
    c_speed = const.c.to("au/day").value
    jd_lt = jd - d_obs / c_speed
    return jd_lt


def estimate_axes_ratio(residuals, R):
    """Estimate the axes ratio of a SSO from the residuals of the sHG1G2 model and its oblateness R.

    Parameters
    ----------
    residuals: np.array
        Residuals (observed - model) of the SSO with sHG1G2 model
    R: float
        Oblateness parameter of the sHG1G2 model

    Returns
    -------
    a_b, a_c: float
        a/b and a/c axes ratio
    """
    # Estimate the amplitude of the lightcurve from residuals
    # Taken at 2 sigma
    amplitude = np.std(residuals) * 2.0

    # Estimate the a/b ratio
    a_b = 10 ** (0.4 * (amplitude * 2))

    # Estimate the a/c ratio (and force c<b)
    a_c = (a_b + 1) / (2 * R)
    if a_c < a_b:
        a_c = a_b

    return a_b, a_c


def remove_leading_zeros(val):
    """Iteratively remove leading zeros from a string

    Parameters
    ----------
    val: str
        A string

    Returns
    -------
    The input string with leading zeros removed

    Examples
    --------
    >>> string = '0abcd'
    >>> remove_leading_zeros(string)
    'abcd'

    >>> string = '000000a0bcd'
    >>> remove_leading_zeros(string)
    'a0bcd'

    >>> string = 'toto'
    >>> remove_leading_zeros(string)
    'toto'
    """
    if val.startswith("0"):
        return remove_leading_zeros(val[1:])
    else:
        return val


def process_regex(regex, data):
    """Extract parameters from a regex given the data

    Parameters
    ----------
    regex: str
        Regular expression to use
    data: str
        Data entered by the user

    Returns
    -------
    parameters: dict or None
        Parameters (key: value) extracted from the data
    """
    template = re.compile(regex)
    m = template.match(data)
    if m is None:
        return None

    parameters = m.groupdict()
    return parameters


def extract_array_from_series(col, index, elementtype):
    """Extract array element from a series of arrays.

    Parameters
    ----------
    col: pd.Series
        Pandas Series
    index: int
        Index of the element in the Series
    elementtype: type
        Type of the elements in the array

    Returns
    -------
    out: np.array
        Array with elements `elementtype`.
    """
    return col.to_numpy()[index].astype(elementtype)


def correct_ztf_mpc_names(ssnamenr):
    """Remove trailing 0 at the end of SSO names from ZTF

    e.g. 2010XY03 should read 2010XY3

    Parameters
    ----------
    ssnamenr: np.array
        Array with SSO names from ZTF

    Returns
    -------
    out: np.array
        Array with corrected names from ZTF

    Examples
    --------
    >>> ssnamenr = np.array(['2010XY03', '2023AB0', '2023XY00', '345', '2023UY12'])
    >>> ssnamenr_alt = correct_ztf_mpc_names(ssnamenr)

    # the first ones changed
    >>> assert ssnamenr_alt[0] == '2010XY3'
    >>> assert ssnamenr_alt[1] == '2023AB'
    >>> assert ssnamenr_alt[2] == '2023XY'

    >>> assert np.all(ssnamenr_alt[3:] == ssnamenr[3:])
    """
    # remove numbered
    regex = r"^\d+$"  # noqa: W605
    template = re.compile(regex)
    unnumbered = np.array([template.findall(str(x)) == [] for x in ssnamenr])

    # Extract names
    regex = r"(?P<year>\d{4})(?P<letter>\w{2})(?P<end>\d+)$"  # noqa: W605
    processed = [process_regex(regex, x) for x in ssnamenr[unnumbered]]

    def f(x, y):
        """Correct for trailing 0 in SSO names

        Parameters
        ----------
        x: dict, or None
            Data extracted from the regex
        y: str
            Corresponding ssnamenr

        Returns
        -------
        out: str
            Name corrected for trailing 0 at the end (e.g. 2010XY03 should read 2010XY3)
        """
        if x is None:
            return y
        else:
            return "{}{}{}".format(
                x["year"], x["letter"], remove_leading_zeros(x["end"])
            )

    corrected = np.array([f(x, y) for x, y in zip(processed, ssnamenr[unnumbered])])

    ssnamenr[unnumbered] = corrected

    return ssnamenr


def rockify(ssnamenr: pd.Series):
    """Extract names and numbers from ssnamenr

    Parameters
    ----------
    ssnamenr: pd.Series of str
        SSO names as given in ZTF alert packets

    Returns
    -------
    sso_name: np.array of str
        SSO names according to quaero
    sso_number: np.array of int
        SSO numbers according to quaero
    """
    import rocks

    # prune names
    ssnamenr_alt = correct_ztf_mpc_names(ssnamenr.to_numpy())

    # rockify
    names_numbers = rocks.identify(ssnamenr_alt)

    sso_name = np.transpose(names_numbers)[0]
    sso_number = np.transpose(names_numbers)[1]

    return sso_name, sso_number


def retrieve_last_date_of_previous_month(mydate):
    """Given a date, retrieve the last date from last month

    Parameters
    ----------
    mydate: datetime
        Input date

    Returns
    -------
    out: datetime
        Last date from previous month according to `mydate`

    Examples
    --------
    >>> mydate = datetime.date(year=2025, month=4, day=5)
    >>> out = retrieve_last_date_of_previous_month(mydate)
    >>> assert out.strftime("%m") == "03"
    >>> assert out.day == 31

    >>> mydate = datetime.date(year=2025, month=1, day=14)
    >>> out = retrieve_last_date_of_previous_month(mydate)
    >>> assert out.month == 12
    >>> assert out.year == 2024
    """
    first = mydate.replace(day=1)
    last_month = first - datetime.timedelta(days=1)
    return last_month


def retrieve_first_date_of_next_month(mydate):
    """Given a date, retrieve the first date from next month

    Parameters
    ----------
    mydate: datetime
        Input date

    Returns
    -------
    out: datetime
        Last date from previous month according to `mydate`

    Examples
    --------
    >>> mydate = datetime.date(year=2025, month=4, day=5)
    >>> out = retrieve_first_date_of_next_month(mydate)
    >>> assert out.strftime("%m") == "05"
    >>> assert out.day == 1

    >>> mydate = datetime.date(year=2025, month=12, day=14)
    >>> out = retrieve_first_date_of_next_month(mydate)
    >>> assert out.month == 1
    >>> assert out.year == 2026
    """
    return (mydate.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)


if __name__ == "__main__":
    """Execute the unit test suite"""

    # Run the Spark test suite
    regular_unit_tests(globals())

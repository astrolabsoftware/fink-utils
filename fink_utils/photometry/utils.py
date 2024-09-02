# Copyright 2023 AstroLab Software
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

from fink_utils.test.tester import regular_unit_tests


def is_source_behind(distnr: float, chinr: float = None, sharpnr: float = None) -> bool:
    """Check if the alert is behind a source

    Parameters
    ----------
    distnr: float
        Distance to nearest source in reference image PSF-catalog within 30 arcsec [pixels]
    chinr: float, optional
        DAOPhot chi parameter of nearest source in reference image PSF-catalog within 30 arcsec
    sharpnr: float, optional
        DAOPhot sharp parameter of nearest source in reference image PSF-catalog within 30 arcsec

    Returns
    -------
    out: bool
        True if there is a source behind. False otherwise.

    Examples
    --------
    >>> is_source_behind(distnr=0.5)
    True

    >>> is_source_behind(distnr=3.5)
    False

    >>> is_source_behind(distnr=0.5, chinr=1.0)
    True

    >>> is_source_behind(distnr=0.5, chinr=0.0)
    True

    >>> is_source_behind(distnr=0.5, sharpnr=0.0)
    True

    >>> is_source_behind(distnr=0.5, chinr=2.0, sharpnr=0.0)
    True

    >>> is_source_behind(distnr=0.5, chinr=2.0, sharpnr=1.0)
    False

    >>> import pandas as pd
    >>> distnr = pd.Series([1.5, 2.5, 0.0, 1.0])
    >>> chinr = pd.Series([0.0, 0.0, 3.0, 0.0])
    >>> sharpnr = pd.Series([0.0, 0.0, 0.0, 3.0])
    >>> out = is_source_behind(distnr=distnr, chinr=chinr, sharpnr=sharpnr)
    >>> out.values
    array([ True, False,  True, False], dtype=bool)
    """
    cond1 = (distnr >= 0) & (distnr <= 1.5)
    cond2 = cond3 = True

    if chinr is not None:
        cond2 = (chinr >= 0.5) & (chinr <= 1.5)
    if sharpnr is not None:
        cond3 = np.abs(sharpnr) <= 0.5

    return cond1 & (cond2 | cond3)


if __name__ == "__main__":
    """Execute the unit test suite"""

    # Run the Spark test suite
    regular_unit_tests(globals())

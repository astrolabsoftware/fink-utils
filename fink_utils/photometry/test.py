# Copyright 2019-2022 AstroLab Software
# Author: Roman Le Montagner
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
import pytest
import requests
import pandas as pd
import numpy as np
import time as t

from fink_utils.photometry.conversion import dc_mag
from fink_utils.photometry.conversion import mag2fluxcal_snana

from fink_utils.photometry.vect_conversion import vect_mag2fluxcal_snana
from fink_utils.photometry.vect_conversion import vect_dc_mag


@pytest.fixture
def requests_alerts():
    r = requests.post(
        "https://api.fink-portal.org/api/v1/latests",
        json={"class": "Seyfert_2", "n": "500", "columns": "i:objectId"},
    )

    # Format output in a DataFrame
    pdf = pd.DataFrame.from_dict(r.json())

    request_object = ",".join(list(pdf["i:objectId"]))
    request_object += ",ZTF21abeyuqh,ZTF21abeyutr,ZTF21abeyuua"
    request_columns = (
        "i:objectId,i:candid,i:magpsf,i:sigmapsf,i:magnr,i:sigmagnr,i:isdiffpos"
    )

    r = requests.post(
        "https://api.fink-portal.org/api/v1/objects",
        json={
            "objectId": request_object,
            "output-format": "json",
            "columns": request_columns,
        },
    )

    # Format output in a DataFrame
    pdf = pd.DataFrame.from_dict(r.json())

    return pdf


def test_dc_mag(requests_alerts):
    t_before = t.time()
    res_dc_mag, res_sig_mag = vect_dc_mag(
        requests_alerts["i:magpsf"].values,
        requests_alerts["i:sigmapsf"].values,
        requests_alerts["i:magnr"].values,
        requests_alerts["i:sigmagnr"].values,
        requests_alerts["i:isdiffpos"].values,
    )
    t_vect = t.time() - t_before

    t_before = t.time()
    _dc_mag = requests_alerts.apply(
        lambda x: dc_mag(
            x["i:magpsf"],
            x["i:sigmapsf"],
            x["i:magnr"],
            x["i:sigmagnr"],
            x["i:isdiffpos"],
        ),
        axis=1,
        result_type="expand",
    )
    t_loop = t.time() - t_before

    t1 = np.allclose(res_dc_mag, _dc_mag[0])

    t2 = np.allclose(res_sig_mag, _dc_mag[1])

    assert np.all(np.allclose(t1, t2))
    assert t_vect < t_loop


def test_flux_snana(requests_alerts):
    t_before = t.time()
    vect_flux, vect_sigflux = vect_mag2fluxcal_snana(
        requests_alerts["i:magpsf"].values, requests_alerts["i:sigmapsf"].values
    )
    t_vect = t.time() - t_before

    t_before = t.time()
    _flux_snana = requests_alerts.apply(
        lambda x: mag2fluxcal_snana(x["i:magpsf"], x["i:sigmapsf"]),
        axis=1,
        result_type="expand",
    )
    t_loop = t.time() - t_before

    t1 = np.allclose(vect_flux, _flux_snana[0])

    t2 = np.allclose(vect_sigflux, _flux_snana[1])

    assert np.all(np.allclose(t1, t2))
    assert t_vect < t_loop

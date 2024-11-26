# Copyright 2023-2024 AstroLab Software
# Author: Тимофей Пшеничный, Julien Peloton
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
"""Part of these functionalities were taken from the Anomaly detection module"""

import os
import io
import time
import requests
import pandas as pd
import numpy as np
import gzip
from astropy.io import fits

import matplotlib.pyplot as plt


from fink_utils.data.utils import extract_field

COLORS_ZTF = {1: "#15284F", 2: "#F5622E"}


def status_check(res, header, sleep=8, timeout=25):
    """Checks whether the request was successful.

    In case of an error, sends information about the error to the @fink_test telegram channel

    Parameters
    ----------
    res : Response object

    Returns
    -------
        result : bool
            True : The request was successful
            False: The request was executed with an error
    """
    if res.status_code != 200:
        msg = """
        {}
        Content: {}
        """.format(header, res.content)
        url = "https://api.telegram.org/bot"
        url += os.environ["FINK_TG_TOKEN"]
        method = url + "/sendMessage"
        time.sleep(sleep)
        requests.post(
            method, data={"chat_id": "@fink_bot_error", "text": msg}, timeout=timeout
        )
        return False
    return True


def msg_handler_tg(tg_data, channel_id, init_msg, timeout=25):
    """Send `tg_data` to a telegram channel

    Notes
    -----
    The function sends notifications to the "channel_id" channel of Telegram.

    Parameters
    ----------
    tg_data: list
        List of tuples. Each item is a separate notification.
        Content of the tuple:
            text_data : str
                Notification text
            cutout : BytesIO stream
                cutout image in png format
            curve : BytesIO stream
                light curve picture
    channel_id: string
        Channel id in Telegram
    init_msg: str
        Initial message

    Returns
    -------
        None
    """
    url = "https://api.telegram.org/bot"
    url += os.environ["FINK_TG_TOKEN"]
    method = url + "/sendMediaGroup"

    if init_msg != "":
        res = requests.post(
            url + "/sendMessage",
            data={"chat_id": channel_id, "text": init_msg, "parse_mode": "markdown"},
            timeout=timeout,
        )
        status_check(res, header=channel_id)
    for text_data, cutout, curve in tg_data:
        files = {"first": curve}
        media = [
            {
                "type": "photo",
                "media": "attach://first",
                "caption": text_data,
                "parse_mode": "markdown",
            }
        ]
        if cutout is not None:
            files.update({"second": cutout})
            media.append({"type": "photo", "media": "attach://second"})
        res = requests.post(
            method,
            params={"chat_id": channel_id, "media": str(media).replace("'", '"')},
            files=files,
            timeout=timeout,
        )
        status_check(res, header=channel_id)
        time.sleep(10)


def msg_handler_tg_cutouts(tg_data, channel_id, init_msg, timeout=25, sleep_seconds=10):
    """Multi-cutout version of `msg_handler_tg`

    Notes
    -----
    The function sends notifications to the "channel_id" channel of Telegram.

    Parameters
    ----------
    tg_data: list
        List of tuples. Each item is a separate notification.
        Content of the tuple:
            text_data : str
                Notification text
            curve : BytesIO stream
                light curve picture
            cutouts : list of BytesIO stream
                List of cutout images in png format (1, 2, or 3 cutouts)
    channel_id: string
        Channel id in Telegram
    init_msg: str
        Initial message
    timeout: int
        Timeout when sending message. Default is 25 seconds.
    sleep_seconds: int
        How many seconds to sleep between two messages to avoid
        code 429 from the Telegram API. Default is 10 seconds.

    Returns
    -------
        None
    """
    url = "https://api.telegram.org/bot"
    url += os.environ["FINK_TG_TOKEN"]
    method = url + "/sendMediaGroup"

    if init_msg != "":
        res = requests.post(
            url + "/sendMessage",
            data={"chat_id": channel_id, "text": init_msg, "parse_mode": "markdown"},
            timeout=timeout,
        )
        status_check(res, header=channel_id)
    for text_data, curve, cutouts in tg_data:
        files = {"first": curve}
        media = [
            {
                "type": "photo",
                "media": "attach://first",
                "caption": text_data,
                "parse_mode": "markdown",
            }
        ]
        if isinstance(cutouts, list):
            names = ["second", "thrid", "fourth"]
            for cutout, name in zip(cutouts, names):
                files.update({name: cutout})
                media.append({"type": "photo", "media": "attach://{}".format(name)})
        res = requests.post(
            method,
            params={"chat_id": channel_id, "media": str(media).replace("'", '"')},
            files=files,
            timeout=timeout,
        )
        status_check(res, header=channel_id)
        time.sleep(sleep_seconds)


def get_cutout(cutout=None, ztf_id=None, kind="Difference", origin="alert"):
    """Loads cutout image from alert packet or via Fink API

    Parameters
    ----------
    cutout: bytes, optional
        Gzipped FITS file from alert packet. Only
        used for origin=alert.
    ztf_id : str, optional
        unique identifier for this object. Only
        used for origin=API.
    origin: str, optional
        Choose between `alert`[default], or API.

    Returns
    -------
    out : BytesIO stream
        cutout image in png format
    """
    if origin == "API":
        assert ztf_id is not None
        r = requests.post(
            "https://fink-portal.org/api/v1/cutouts",
            json={"objectId": ztf_id, "kind": kind, "output-format": "array"},
            timeout=25,
        )
        if not status_check(r, header=ztf_id):
            return io.BytesIO()
        data = np.log(np.array(r.json()["b:cutoutScience_stampData"], dtype=float))
        plt.axis("off")
        plt.imshow(data, cmap="PuBu_r")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        plt.close()
    elif origin == "alert":
        assert cutout is not None

        # Unzip
        with gzip.open(io.BytesIO(cutout), "rb") as fits_file:
            with fits.open(
                io.BytesIO(fits_file.read()), ignore_missing_simple=True
            ) as hdul:
                img = hdul[0].data[::-1]

        data = np.log(img)
        plt.axis("off")
        plt.imshow(data, cmap="PuBu_r")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        plt.close()
    else:
        buf = io.BytesIO()

    return buf


def get_curve(
    alert=None,
    jd=None,
    magpsf=None,
    sigmapsf=None,
    diffmaglim=None,
    fid=None,
    objectId=None,
    origin="API",
):
    """Generate PNG lightcurve

    Notes
    -----
    Based on `origin`, the other arguments are mandatory:
    - origin=API: objectId
    - origin=alert: alert
    - origin=fields: objectId, jd, magpsf, sigmapsf, diffmaglim, fid

    Parameters
    ----------
    alert: dict, optional
        alert packet containing `objectId`, `candidate`, `prv_candidates`
    jd: np.array, optional
        Vector of times.
    magpsf: np.array, optional
        Vector of difference magnitudes.
    sigmapsf: np.array, optional
        Vector of error on difference magnitudes.
    diffmaglim: np.array, optional
        Vector of upper limits on magnitude.
    fid: np.array, optional
        Vector of filter ID.
    objectId : str
        unique identifier for this object
    origin: str, optional
        Choose between `alert`, `API`[default], or `fields`.

    Returns
    -------
    out : BytesIO stream
        light curve picture
    """
    filter_dict = {1: "g band", 2: "r band"}
    if origin == "API":
        assert objectId is not None

        r = requests.post(
            "https://fink-portal.org/api/v1/objects",
            json={
                "objectId": objectId,
                "columns": "i:jd,i:fid,i:magpsf,i:sigmapsf,d:tag",
                "withupperlim": "True",
            },
        )
        if not status_check(r, header=objectId):
            return None

        # Format output in a DataFrame
        pdf = pd.read_json(io.BytesIO(r.content))

        plt.figure(figsize=(12, 4))

        for filt in pdf["i:fid"].unique():
            if filt == 3:
                continue
            maskFilt = pdf["i:fid"] == filt

            # The column `d:tag` is used to check data type
            maskValid = pdf["d:tag"] == "valid"
            plt.errorbar(
                pdf[maskValid & maskFilt]["i:jd"].apply(lambda x: x - 2400000.5),
                pdf[maskValid & maskFilt]["i:magpsf"],
                pdf[maskValid & maskFilt]["i:sigmapsf"],
                ls="",
                marker="o",
                color=COLORS_ZTF[filt],
                label=filter_dict[filt],
            )

            # see fink-utils#78 & fink-broker#872
            if "i:diffmaglim" in pdf.columns:
                maskUpper = pdf["d:tag"] == "upperlim"
                plt.plot(
                    pdf[maskUpper & maskFilt]["i:jd"].apply(lambda x: x - 2400000.5),
                    pdf[maskUpper & maskFilt]["i:diffmaglim"],
                    ls="",
                    marker="^",
                    color=COLORS_ZTF[filt],
                    markerfacecolor="none",
                )

            maskBadquality = pdf["d:tag"] == "badquality"
            plt.errorbar(
                pdf[maskBadquality & maskFilt]["i:jd"].apply(lambda x: x - 2400000.5),
                pdf[maskBadquality & maskFilt]["i:magpsf"],
                pdf[maskBadquality & maskFilt]["i:sigmapsf"],
                ls="",
                marker="v",
                color=COLORS_ZTF[filt],
            )

        plt.gca().invert_yaxis()
        plt.legend()
        plt.xlabel("Modified Julian Date")
        plt.ylabel("Difference magnitude")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

    elif origin in ["alert", "fields"]:
        if origin == "alert":
            # extract current and historical data as one vector
            magpsf = extract_field(alert, "magpsf")
            sigmapsf = extract_field(alert, "sigmapsf")
            diffmaglim = extract_field(alert, "diffmaglim")
            fid = extract_field(alert, "fid")
            jd = extract_field(alert, "jd")
            objectId = alert["objectId"]

        # Rescale dates
        dates = np.array([i - jd[-1] for i in jd])

        # loop over filters
        plt.figure(num=1, figsize=(12, 4))

        # Loop over each filter
        for filt in COLORS_ZTF.keys():
            mask = np.where(fid == filt)[0]

            # Skip if no data
            if len(mask) == 0:
                continue

            # y data -- assume NaN (Spark style) and None (Pandas style) for missing values
            maskNotNone = np.array([
                ~np.isnan(i) and (i is not None) for i in magpsf[mask]
            ])
            plt.errorbar(
                dates[mask][maskNotNone],
                magpsf[mask][maskNotNone],
                yerr=sigmapsf[mask][maskNotNone],
                color=COLORS_ZTF[filt],
                marker="o",
                ls="",
                label=filter_dict[filt],
                mew=4,
            )
            # Upper limits
            plt.plot(
                dates[mask][~maskNotNone],
                diffmaglim[mask][~maskNotNone],
                color=COLORS_ZTF[filt],
                marker="v",
                ls="",
                mew=4,
                alpha=0.5,
            )
            plt.title(objectId)
        plt.legend()
        plt.gca().invert_yaxis()
        plt.xlabel("Days to candidates")
        plt.ylabel("Difference magnitude")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

    return buf

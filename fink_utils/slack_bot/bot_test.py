from PIL import Image
import pandas as pd

import io
import requests
import numpy as np
import matplotlib.pyplot as plt

import fink_utils.slack_bot.bot as slack_bot

from fink_science.image_classification.utils import img_normalizer
from fink_utils.logging.logs import init_logging


def unzip_img(stamp: bytes, title_name: str) -> io.BytesIO:
    """
    Unzip and prepare an image to be send on slack

    Parameters
    ----------
    stamp : bytes
        raw alert cutout
    title_name : str
        image title

    Returns
    -------
    io.BytesIO
        image within a byte class
    """
    min_img = 0
    max_img = 255
    img = img_normalizer(stamp, min_img, max_img)
    img = Image.fromarray(img).convert("L")

    _ = plt.figure(figsize=(15, 6))
    plt.imshow(img, cmap="gray", vmin=min_img, vmax=max_img)
    plt.title(title_name)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf


def get_imgs():
    def request(kind):
        r = requests.post(
            "https://fink-portal.org/api/v1/cutouts",
            json={
                "objectId": "ZTF23abjzkmx",
                "kind": kind,
                "output-format": "array",
            },
        )
        return r

    def decode_array(content):
        pdf = pd.read_json(io.BytesIO(content))
        return np.array(pdf[pdf.columns[0]].values[0])

    return (
        decode_array(request("Science").content),
        decode_array(request("Template").content),
        decode_array(request("Difference").content),
    )


if __name__ == "__main__":
    alert_id = 10
    raw_science, raw_template, raw_difference = get_imgs()

    prep_science = unzip_img(raw_science, "Science")
    prep_template = unzip_img(raw_template, "Template")
    prep_difference = unzip_img(raw_difference, "Difference")

    logger = init_logging("test_slack_bot")
    slack_client = slack_bot.init_slackbot(logger)

    results_post = slack_bot.post_files_on_slack(
        slack_client,
        [prep_science, prep_template, prep_difference],
        ["scienceImage", "templateImage", "differenceImage"],
        sleep_delay=3,
    )

    slack_bot.post_msg_on_slack(
        slack_client,
        "#bot_fink_utils_test",
        [
            f"""
        Test slack_bot fink_utils
        {results_post[0]}

        {results_post[1]}

        {results_post[2]}
        """
        ],
        logger=logger,
        verbose=True,
    )

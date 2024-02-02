from PIL import Image
import pandas as pd

import io
import requests
import numpy as np
import matplotlib.pyplot as plt

import fink_utils.slack_bot.bot as slack_bot

from fink_science.image_classification.utils import img_normalizer
from fink_utils.logging.logs import init_logging
from fink_utils.slack_bot.msg_builder import Message, Divider, Header
from fink_utils.slack_bot.section import Section, Type_text
import fink_utils.slack_bot.image as img


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
        sleep_delay=1,
    )

    msg = Message()
    msg.add_header(Type_text.PLAIN_TXT, "Fink Slack Test")
    msg.add_divider()

    first_section = Section(
        Type_text.MARKDOWN, "this is a test message from the fink-utils CI"
    )

    first_section.add_textfield(Type_text.MARKDOWN, "first field :stars:", True)
    first_section.add_textfield(
        Type_text.MARKDOWN, "second field :ringed_planet:", True
    )

    science_section = Section(Type_text.MARKDOWN, "Science")
    science_section.add_slack_image(results_post[0], "Science")

    template_section = Section(Type_text.MARKDOWN, "Template")
    template_section.add_slack_image(results_post[1], "Template")

    difference_section = Section(Type_text.MARKDOWN, "Difference")
    difference_section.add_slack_image(results_post[2], "Difference")

    msg.add_elements(first_section)
    msg.add_elements(science_section)
    msg.add_elements(template_section)
    msg.add_elements(difference_section)

    fink_section = Section(Type_text.MARKDOWN, "fink-science-portal")
    fink_section.add_urlbutton(
        Type_text.PLAIN_TXT,
        "fink-science-portal",
        "View on Fink :fink:",
        "https://fink-portal.org/ZTF23abjzkmx",
        True,
    )
    msg.add_elements(fink_section)

    img_sci = img.Image(results_post[0], "scienceImage")
    img_temp = img.Image(results_post[1], "scienceTemplate")
    img_diff = img.Image(results_post[2], "scienceDifference")
    header_img = Header(Type_text.PLAIN_TXT, "Image Test")
    msg.add_elements([header_img, img_sci, Divider(), img_temp, Divider(), img_diff])

    slack_bot.post_msg_on_slack(
        slack_client,
        "#bot_fink_grb_bronze_test",
        [msg],
        logger=logger,
        verbose=True,
    )

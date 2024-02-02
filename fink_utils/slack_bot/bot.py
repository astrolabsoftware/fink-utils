import os
import time
import io

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from typing import List

from fink_utils.logging.logs import init_logging, LoggerNewLine
from fink_utils.slack_bot.msg_builder import Message

import json


def init_slackbot(logger: LoggerNewLine = None) -> WebClient:
    """
    Initialize a slack bot

    Returns
    -------
    WebClient
        the web client used by the bot
    """
    try:
        token_slack = os.environ["FINK_SLACK_TOKEN"]
    except KeyError as e:
        if logger is None:
            logger = init_logging()
        logger.error("FINK_SLACK_TOKEN environement variable not found !!", exc_info=1)
        raise e
    client = WebClient(token=token_slack)
    return client


def post_files_on_slack(
    slack_client: WebClient,
    file_list: List[io.BytesIO],
    title_list: List[str],
    sleep_delay: int = 1,
) -> List[str]:
    """
    Post the files in bytes format on slack with the associated title.

    Parameters
    ----------
    slack_client : WebClient
        slack bot client
    file_list : list[io.BytesIO]
        files to upload on slack in bytes format
    title_list : list[str]
        title to give to the files, should be in the same order than the file_list
    sleep_delay : int, optional
        delay to wait before executing the instruction after the upload, by default 1

    Returns
    -------
    list[str]
        list of string containing the link to refers to the files upload on slack in a msg
    """
    results = slack_client.files_upload_v2(
        file_uploads=[
            {
                "file": file,
                "title": title,
            }
            for file, title in zip(file_list, title_list)
        ]
    )
    time.sleep(sleep_delay)
    return [
        f"https://files.slack.com/files-tmb/{el['user_team']}-{el['id']}/"
        for el in results["files"]
    ]


def post_msg_on_slack(
    webclient: WebClient,
    channel: str,
    msg: List[Message],
    sleep_delay: int = 1,
    logger: LoggerNewLine = None,
    verbose: bool = False,
):
    """
    Send a msg on the specified slack channel

    Parameters
    ----------
    webclient : WebClient
        slack bot client
    channel : str
        the channel where will be posted the message
    msg : list
        list of message to post on slack, each string in the list will be a single post.
    sleep_delay : int, optional
        delay to wait between the message, by default 1
    logger : _type_, optional
        logger used to print logs, by default None
    verbose : bool, optional
        if true, print logs between the message, by default False

    * Notes:

    Before sending message on slack, check that the Fink bot have been added to the targeted channel.

    Examples
    --------
    see bot_test.py
    """
    if verbose and logger is None:
        logger = init_logging()
    try:
        for tmp_msg in msg:
            json_p = json.dumps(tmp_msg.blocks["blocks"])
            webclient.chat_postMessage(
                channel=channel, text="error with msg blocks", blocks=json_p
            )
            if verbose:
                logger.debug("Post msg on slack successfull")
            time.sleep(sleep_delay)
    except SlackApiError as e:
        if e.response["ok"] is False:
            logger.error("Post slack msg error", exc_info=1)

import logging
import datetime
import pytz

import fink_utils


class CustomTZFormatter(logging.Formatter):  # pragma: no cover
    """override logging.Formatter to use an aware datetime object"""

    def converter(self, timestamp):
        dt = datetime.datetime.fromtimestamp(timestamp)
        tzinfo = pytz.timezone("Europe/Paris")
        return tzinfo.localize(dt)

    def formatTime(self, record, datefmt=None):  # noqa: N802
        dt = self.converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            try:
                s = dt.isoformat(timespec="milliseconds")
            except TypeError:
                s = dt.isoformat()
        return s


class LoggerNewLine(logging.Logger):
    """
    A custom logger class adding only a method to print a newline.

    Examples
    --------
    logger.newline()
    """

    def __init__(self, name: str, level: int = 0) -> None:
        super().__init__(name, level)
        ch = logging.StreamHandler()

        self.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = CustomTZFormatter(
            "%(asctime)s - %(name)s - %(levelname)s \n\t message: %(message)s"
        )

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        self.addHandler(ch)

        blank_handler = logging.StreamHandler()
        blank_handler.setLevel(logging.DEBUG)
        blank_handler.setFormatter(logging.Formatter(fmt=""))
        self.console_handler = ch
        self.blank_handler = blank_handler

    def newline(self, how_many_lines=1):
        """
        Print blank line using the logger class

        Parameters
        ----------
        how_many_lines : int, optional
            how many blank line to print, by default 1
        """
        # Switch handler, output a blank line
        self.removeHandler(self.console_handler)
        self.addHandler(self.blank_handler)
        for _ in range(how_many_lines):
            self.info("\n")

        # Switch back
        self.removeHandler(self.blank_handler)
        self.addHandler(self.console_handler)


def init_logging(logger_name=fink_utils.__name__) -> LoggerNewLine:
    """
    Initialise a logger for the gcn stream

    Parameters
    ----------
    None

    Returns
    -------
    logger : Logger object
        A logger object for the logging management.

    Examples
    --------
    >>> l = init_logging()
    >>> type(l)
    <class 'logging.Logger'>
    """
    # create logger

    logging.setLoggerClass(LoggerNewLine)
    logger = logging.getLogger(logger_name)
    return logger

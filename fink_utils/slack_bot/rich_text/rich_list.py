from fink_utils.slack_bot.rich_text.rich_text import RichElement
from fink_utils.slack_bot.rich_text.rich_section import SectionElement
from enum import Enum
from typing import List, Union
from typing_extensions import Self


class RichListStyle(Enum):
    BULLET = "bullet"
    ORDERED = "ordered"


class RichList(RichElement):
    def __init__(
        self,
        style: RichListStyle = RichListStyle.BULLET,
        indent: int = None,
        offset: int = None,
        border: int = None,
    ) -> None:
        """
        A class representing a list

        Parameters
        ----------
        style : RichListStyle, optional
            style of the list, by default RichListStyle.BULLET
        indent : int, optional
            list indent in pixels, by default None
        offset : int, optional
            number of pixels to offset the list, by default None
        border : int, optional
            border thickness in pixels, by default None
        """
        super().__init__()
        self.rich_list = {
            "type": "rich_text_list",
            "style": style.value,
            "elements": [],
        }
        if indent is not None:
            self.rich_list["indent"] = indent

        if offset is not None:
            self.rich_list["offset"] = offset

        if border is not None:
            self.rich_list["border"] = border

    def get_element(self):
        return self.rich_list

    def add_elements(
        self, elements: Union[SectionElement, List[SectionElement]]
    ) -> Self:
        if type(elements) is list:
            self.rich_list["elements"] += [el.get_element() for el in elements]
        else:
            self.rich_list["elements"].append(elements.get_element())
        return self

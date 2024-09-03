from fink_utils.slack_bot.section import Section, TypeText
from fink_utils.slack_bot.image import Image
from typing import List, Union
from typing_extensions import Self
from fink_utils.slack_bot.rich_text.rich_text import RichText


class Message:
    def __init__(self):
        self.blocks = {"blocks": []}

    def add_elements(
        self,
        elements: Union[
            Section, Image, RichText, List[Union[Image, Section, RichText]]
        ],
    ) -> Self:
        """
        Add elements in the message

        Parameters
        ----------
        elements : Union[Section, Image, List[Union[Image, Section]]]
            elements to add in the message.
            Can be a single elements or a list of elements

        Returns
        -------
        Self
            return instance of the current object to chain add operation
        """
        if type(elements) is list:
            self.blocks["blocks"] += [el.get_element() for el in elements]
        else:
            self.blocks["blocks"].append(elements.get_element())
        return self

    def add_divider(self) -> Self:
        """
        Add a divider in the message
        """
        self.blocks["blocks"].append({"type": "divider"})
        return self

    def add_header(self, header_txt: str, allow_emoji: bool = False) -> Self:
        """
        Add a header in the message

        Parameters
        ----------
        type_txt : TypeText
            type of the header text
        header_txt : str
            header text
        allow_emoji : bool, optional
            if True allow emoji in the header, by default False
        """
        self.blocks["blocks"].append({
            "type": "header",
            "text": {
                "type": TypeText.PLAIN_TXT.value,
                "text": header_txt,
                "emoji": allow_emoji,
            },
        })
        return self


class Divider:
    def __init__(self) -> None:
        """
        Instanciate a divider
        """
        self.divider = {"type": "divider"}

    def get_element(self):
        return self.divider


class Header:
    def __init__(self, header_txt: str, allow_emoji: bool = False) -> None:
        self.header = {
            "type": "header",
            "text": {
                "type": TypeText.PLAIN_TXT.value,
                "text": header_txt,
                "emoji": allow_emoji,
            },
        }

    def get_element(self):
        return self.header

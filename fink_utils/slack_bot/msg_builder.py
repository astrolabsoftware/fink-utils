from fink_utils.slack_bot.section import Section, Type_text
from fink_utils.slack_bot.image import Image
from typing import List, Union


class Message:
    def __init__(self):
        self.blocks = {"blocks": []}

    def add_elements(
        self, elements: Union[Section, Image, List[Union[Image, Section]]]
    ):
        """
        Add elements in the message

        Parameters
        ----------
        elements : Union[Section, Image, List[Union[Image, Section]]]
            elements to add in the message.
            Can be a single elements or a list of elements
        """
        if type(elements) == list:
            self.blocks["blocks"] += [el.get_element() for el in elements]
        else:
            self.blocks["blocks"].append(elements.get_element())

    def add_divider(self):
        """
        Add a divider in the message
        """
        self.blocks["blocks"].append({"type": "divider"})

    def add_header(
        self, type_txt: Type_text, header_txt: str, allow_emoji: bool = False
    ):
        """
        Add a header in the message

        Parameters
        ----------
        type_txt : Type_text
            type of the header text
        header_txt : str
            header text
        allow_emoji : bool, optional
            if True allow emoji in the header, by default False
        """
        self.blocks["blocks"].append(
            {
                "type": "header",
                "text": {
                    "type": type_txt.value,
                    "text": header_txt,
                    "emoji": allow_emoji,
                },
            }
        )


class Divider:
    def __init__(self) -> None:
        """
        Instanciate a divider
        """
        self.divider = {"type": "divider"}

    def get_element(self):
        return self.divider


class Header:
    def __init__(
        self, type_txt: Type_text, header_txt: str, allow_emoji: bool = False
    ) -> None:
        self.header = {
            "type": "header",
            "text": {
                "type": type_txt.value,
                "text": header_txt,
                "emoji": allow_emoji,
            },
        }

    def get_element(self):
        return self.header

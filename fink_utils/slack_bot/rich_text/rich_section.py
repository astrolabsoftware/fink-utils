from fink_utils.slack_bot.rich_text.rich_text import RichElement
from fink_utils.slack_bot.rich_text.rich_text_element import RichTextElement
from typing import List, Union
from typing_extensions import Self


class SectionElement(RichElement):
    def __init__(self) -> None:
        """
        A class representing a section of a rich text
        """
        super().__init__()
        self.section = {"type": "rich_text_section", "elements": []}

    def add_elements(
        self, elements: Union[RichTextElement, List[RichTextElement]]
    ) -> Self:
        if type(elements) is list:
            self.section["elements"] += [el.get_element() for el in elements]
        else:
            self.section["elements"].append(elements.get_element())
        return self

    def get_element(self) -> dict:
        return self.section

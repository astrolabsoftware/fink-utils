from fink_utils.slack_bot.rich_text.rich_text import RichElement
from typing import List, Union
from typing_extensions import Self
from fink_utils.slack_bot.rich_text.rich_text_element import RichTextElement


class RichPreformatted(RichElement):
    def __init__(self, border: int = None) -> None:
        """
        A class representing a block surrounded by a border

        Parameters
        ----------
        border : _type_, optional
            border thickness in pixels, by default None
        """
        super().__init__()
        self.preform = {"type": "rich_text_preformatted", "elements": []}
        if border is not None:
            self.preform["border"] = border

    def get_element(self):
        return self.preform

    def add_elements(
        self, elements: Union[RichTextElement, List[RichTextElement]]
    ) -> Self:
        if type(elements) is list:
            self.preform["elements"] += [el.get_element() for el in elements]
        else:
            self.preform["elements"].append(elements.get_element())
        return self

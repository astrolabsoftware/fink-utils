from fink_utils.slack_bot.rich_text.rich_text import RichElement
from typing import List, Union
from typing_extensions import Self
from fink_utils.slack_bot.rich_text.rich_text_element import RichTextElement


class RichQuote(RichElement):
    def __init__(self, border: int = None) -> None:
        """
        A class representing a quote

        Parameters
        ----------
        border : int, optional
            border thickness in pixels, by default None
        """
        super().__init__()
        self.quote = {"type": "rich_text_quote", "elements": []}
        if border is not None:
            self.quote["border"] = border

    def get_element(self):
        return self.quote

    def add_elements(
        self, elements: Union[RichTextElement, List[RichTextElement]]
    ) -> Self:
        if type(elements) is list:
            self.quote["elements"] += [el.get_element() for el in elements]
        else:
            self.quote["elements"].append(elements.get_element())
        return self

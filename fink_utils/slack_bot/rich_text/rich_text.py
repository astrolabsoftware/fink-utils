from abc import ABC, abstractmethod
from typing import List, Union
from typing_extensions import Self


class RichElement(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_element(self):
        pass


class RichText:
    def __init__(self) -> None:
        """
        A class representing a rich text block
        """
        self.rich_text = {"type": "rich_text", "elements": []}

    def add_elements(self, elements: Union[RichElement, List[RichElement]]) -> Self:
        if type(elements) is list:
            self.rich_text["elements"] += [el.get_element() for el in elements]
        else:
            self.rich_text["elements"].append(elements.get_element())
        return self

    def get_element(self):
        return self.rich_text

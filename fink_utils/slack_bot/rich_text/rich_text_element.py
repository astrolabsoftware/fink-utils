from abc import ABC, abstractmethod
from enum import Enum


class RichTextStyle(Enum):
    BOLD = "bold"
    ITALIC = "italic"
    STRIKE = "strike"
    CODE = "code"


class RichTextElement(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_element(self):
        pass


class Emoji(RichTextElement):
    def __init__(self, emoji_name) -> None:
        super().__init__()
        self.emoji = {"type": "emoji", "name": emoji_name}

    def get_element(self):
        return self.emoji


class Link(RichTextElement):
    def __init__(
        self,
        url: str,
        text: str = None,
        unsafe: bool = None,
        style: RichTextStyle = None,
    ) -> None:
        """
        A link element

        Parameters
        ----------
        url : str
            url
        text : str, optional
            text to show instead of the url, if not provided, show the url, by default None
        unsafe : bool, optional
            indicates whether the link is safe, by default None
        style : RichTextStyle, optional
            link style, by default None
        """
        super().__init__()
        self.link = {"type": "link", "url": url}
        if text is not None:
            self.link["text"] = text
        if unsafe is not None:
            self.link["unsafe"] = unsafe
        if style is not None:
            self.link["style"] = style.value

    def get_element(self):
        return self.link


class Text(RichTextElement):
    def __init__(self, text: str, style: RichTextStyle = None) -> None:
        """
        A text element

        Parameters
        ----------
        text : str
            text
        style : RichTextStyle, optional
            text style, by default None
        """
        super().__init__()
        self.text = {"type": "text", "text": text}
        if style is not None:
            self.link["style"] = style.value

    def get_element(self):
        return self.text

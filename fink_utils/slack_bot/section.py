from enum import Enum
from typing_extensions import Self


class TypeText(Enum):
    PLAIN_TXT = "plain_text"
    MARKDOWN = "mrkdwn"


class Section:
    def __init__(
        self, type_text: TypeText, section_text: str, allow_emoji: bool = False
    ) -> None:
        """Instanciate a section.

        A section has always a text and cannot be empty.

        Parameters
        ----------
        type_text : TypeText
            type of the text, either markdown or plain_text
        section_text : str
            text of this section
        allow_emoji : bool, optional
            if True allow emoji in the text, by default False
        """
        assert len(section_text) > 0, "error, section_text cannot be empty"
        self.section = {"type": "section"}
        self.add_text(type_text, section_text, allow_emoji)

    def add_text(self, type_txt: TypeText, txt: str, allow_emoji: bool = False):
        """Override the text set by the constructor

        Parameters
        ----------
        type_txt : TypeText
            Type of the text
        txt : str
            text to put in the section
        allow_emoji : bool, optional
            if True allow the emoji in the text, by default False
        """
        self.section["text"] = {"type": type_txt.value, "text": txt}
        if type_txt == TypeText.PLAIN_TXT:
            self.section["text"]["emoji"] = allow_emoji

    def add_textfield(
        self, type_txt: TypeText, txt: str, allow_emoji: bool = False
    ) -> Self:
        """Add a text field in the section.

        The first call create the field.
        The next calls add more field into the textfield.

        Parameters
        ----------
        type_txt : TypeText
            type of the text in the field
        txt : str
            text of the field
        allow_emoji : bool, optional
            if True allow the emoji in the text, by default False

        Returns
        -------
        Self
            return instance of the current object to chain add operation
        """
        field_txt = {"type": type_txt.value, "text": txt}
        if type_txt == TypeText.PLAIN_TXT:
            field_txt = {"emoji": allow_emoji}

        if "fields" in self.section:
            self.section["fields"].append(field_txt)
        else:
            self.section["fields"] = [field_txt]
        return self

    def add_slack_image(self, image_slack_url: str, image_txt: str):
        """Add an image in the section with a text.

        WARNING: Call multiple times this method in the same section will override the previous call.
        Create multiple sections to add multiple images.

        Parameters
        ----------
        image_slack_url : str
            url of the slack image
        image_txt : str
            text to add with the image
        """
        self.section["accessory"] = {
            "type": "image",
            "slack_file": {"url": image_slack_url},
            "alt_text": image_txt,
        }

    def add_url_image(self, image_url: str, image_txt: str):
        """Add an image from the url in the section.

        Parameters
        ----------
        image_url : str
            url of the image
        image_txt : str
            text to add with the image
        """
        self.section["accessory"] = {
            "type": "image",
            "image_url": image_url,
            "alt_text": image_txt,
        }

    def add_urlbutton(
        self,
        type_txt: TypeText,
        button_value: str,
        button_txt: str,
        url: str,
        allow_emoji: bool = False,
    ):
        """Add a button with a url, click on it will redirect to the url.

        Parameters
        ----------
        type_txt : TypeText
            type of the text
        button_value : str
            value of the button
        button_txt : str
            text in the button
        url : str
            redirection url
        allow_emoji : bool, optional
            if True allow the emoji in the text, by default False
        """
        self.section["accessory"] = {
            "type": "button",
            "text": {"type": type_txt.value, "text": button_txt, "emoji": allow_emoji},
            "value": button_value,
            "url": url,
            "action_id": "button-action",
        }

    def get_element(self):
        return self.section

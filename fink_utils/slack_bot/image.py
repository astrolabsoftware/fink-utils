from fink_utils.slack_bot.section import TypeText


class Image:
    def __init__(self, image_url: str, alt_text: str) -> None:
        """
        Instanciate an image to upload on slack

        Parameters
        ----------
        image_url : str
            url of the image
        alt_text : str
            text of the image
        """
        self.image = {"type": "image", "image_url": image_url, "alt_text": alt_text}

    def add_title(self, type_text: TypeText, title: str, allow_emoji: bool = False):
        """
        Add a title to the image

        Parameters
        ----------
        type_text : TypeText
            type of the title
        title : str
            text of the title
        allow_emoji : bool, optional
            if True allow the emoji in the text, by default False
        """
        self.image["title"] = {
            "type": type_text.value,
            "text": title,
            "emoji": allow_emoji,
        }

    def get_element(self):
        return self.image

class Note(object):
    """docstring for Note."""

    def __init__(self, text: str):
        self.text = text

    def __str__(self):
        return self.text

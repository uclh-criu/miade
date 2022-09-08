from typing import Optional

from .concept import Category


class NoteSection(object):
    def __init__(self, heading: str, body: str, category: Optional[Category]):
        self.heading = heading
        self.body = body
        self.category = category

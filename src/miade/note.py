import re

from typing import List, Optional

from .notesection import NoteSection
from .concept import Category


class Note(object):
    """docstring for Note."""

    def __init__(self, text: str):
        self.text = text
        self.sections: Optional[List[NoteSection]] = []

    def get_sections(self) -> None:
        sections = re.split(r"\n(?=[A-Za-z\s]+:)", self.text)
        for text in sections:
            heading, body = text.split(":")
            category = None
            if re.match(r"allerg(y|ies)", heading.lower()):
                category = Category.ALLERGY
            elif re.match(r"medications?|mx", heading.lower()):
                category = Category.MEDICATION
            elif re.match(r"problems?|diagnos(is|ses)", heading.lower()):
                category = Category.PROBLEM

            self.sections.append(
                NoteSection(heading=heading, body=body, category=category)
            )

    def __str__(self):
        return self.text

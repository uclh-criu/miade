import re

from typing import List, Optional

from .paragraph import Paragraph, ParagraphType


class Note(object):
    """docstring for Note."""

    def __init__(self, text: str):
        self.text = text
        self.paragraphs: Optional[List[Paragraph]] = []
        self.get_paragraphs()

    def get_paragraphs(self) -> None:
        paragraphs = re.split(r"\n\n+", self.text)
        start = 0

        for text in paragraphs:
            heading, body = text.split(":")
            paragraph_type = ParagraphType.prose

            end = start + len(text)
            paragraph = Paragraph(heading=heading, body=body, type=paragraph_type, start=start, end=end)
            start = end + 2  # Account for the two newline characters

            # TODO: insert loading of regex config
            if re.search(r"allerg(y|ies)", heading.lower()):
                paragraph.type = ParagraphType.allergy
            elif re.search(r"medications?|meds", heading.lower()):
                paragraph.type = ParagraphType.med
            elif re.search(r"problems?|diagnos(is|ses)", heading.lower()):
                paragraph.type = ParagraphType.prob
            # print(paragraph.__dict__)

            self.paragraphs.append(paragraph)

    def __str__(self):
        return self.text

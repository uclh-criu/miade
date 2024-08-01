from enum import Enum
from typing import List


class ParagraphType(Enum):
    prob = "prob"
    pmh = "pmh"
    imp = "imp"
    med = "med"
    allergy = "allergy"
    history = "history"
    exam = "exam"
    ddx = "ddx"
    plan = "plan"
    prose = "prose"


class Paragraph(object):
    """
    Represents a paragraph in a document.

    Attributes:
        heading (str): The heading of the paragraph.
        body (str): The body text of the paragraph.
        type (ParagraphType): The type of the paragraph.
        start (int): The starting position of the paragraph.
        end (int): The ending position of the paragraph.
    """

    def __init__(self, heading: str, body: str, type: ParagraphType, start: int, end: int):
        self.heading: str = heading
        self.body: str = body
        self.type: ParagraphType = type
        self.start: int = start
        self.end: int = end

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.type == other.type and self.start == other.start and self.end == other.end


class ListItem(object):
    """
    Represents an item in a NumberedList

    Attributes:
        content (str): The content of the list item.
        start (int): The starting index of the list item.
        end (int): The ending index of the list item.
    """

    def __init__(self, content: str, start: int, end: int) -> None:
        self.content: str = content
        self.start: int = start
        self.end: int = end

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end


class NumberedList(object):
    """
    Represents a numbered list.

    Attributes:
        items (List[ListItem]): The list of items in the numbered list.
        list_start (int): The starting number of the list.
        list_end (int): The ending number of the list.
    """

    def __init__(self, items: List[ListItem], list_start: int, list_end: int) -> None:
        self.list_start: int = list_start
        self.list_end: int = list_end
        self.items: List[ListItem] = items

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.list_start == other.list_start and self.list_end == other.list_end

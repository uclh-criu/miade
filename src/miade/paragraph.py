from typing import Optional
from enum import Enum


class ParagraphType(Enum):
    prob = 1
    pmh = 2
    imp = 3
    med = 4
    allergy = 5
    history = 6
    exam = 7
    ddx = 8
    plan = 9
    prose = 0


class Paragraph(object):
    def __init__(self, heading: str, body: str, type: ParagraphType, start: int, end: int):
        self.heading = heading
        self.body = body
        self.type = type
        self.start = start
        self.end = end

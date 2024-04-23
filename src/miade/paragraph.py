from enum import Enum


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

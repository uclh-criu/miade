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
    def __init__(
        self, heading: str, body: str, type: ParagraphType, start: int, end: int
    ):
        self.heading = heading
        self.body = body
        self.type = type
        self.start = start
        self.end = end

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return (
            self.type == other.type
            and self.start == other.start
            and self.end == other.end
        )

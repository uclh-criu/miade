from enum import Enum
from typing import Optional


class Category(Enum):
    PROBLEM = 1
    MEDICATION = 2
    ALLERGY = 3


class Concept(object):
    """docstring for Concept."""

    def __init__(
        self,
        id: str,
        name: str,
        category: Category,
        start: Optional[int] = None,
        end: Optional[int] = None,
        dosage: Optional = None,  # sub in classes later
        meta: Optional = None
    ):

        self.name: str = name
        self.id: str = id
        self.category: Category = category
        self.start: int = start
        self.end: int = end
        self.dosage = dosage
        self.meta = meta

    def __str__(self):
        return f"{{name: {self.name}, id: {self.id}, type: {self.category.name}, start: {self.start}, end: {self.end}," \
               f" dosage: {self.dosage}, meta: {self.meta}}} "

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

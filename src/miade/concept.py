from enum import Enum


class Category(Enum):
    DIAGNOSIS = 1
    MEDICATION = 2
    ALLERGY = 3


class Concept(object):
    """docstring for Concept."""

    def __init__(self, id: str, name: str, category: Category):
        self.name: str = name
        self.id: str = id
        self.category: Category = category

    def __str__(self):
        return f"{{name: {self.name}, id: {self.id}, type: {self.category.name}}}"

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

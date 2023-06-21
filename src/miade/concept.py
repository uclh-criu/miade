from __future__ import annotations
from enum import Enum
from typing import Optional, Dict, List

from .dosage import Dosage
from .metaannotations import MetaAnnotations


class Category(Enum):
    PROBLEM = 1
    MEDICATION = 2
    ALLERGY = 3
    REACTION = 4


class Concept(object):
    """docstring for Concept."""

    def __init__(
        self,
        id: str,
        name: str,
        category: Optional[Category] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        dosage: Optional[Dosage] = None,
        linked_concept: Optional[Concept] = None,
        negex: Optional[bool] = False,
        meta_anns: Optional[List[MetaAnnotations]] = None,
        debug_dict: Optional[Dict] = None,
    ):

        self.name = name
        self.id = id
        self.category = category
        self.start = start
        self.end = end
        self.dosage = dosage
        self.linked_concept = linked_concept
        self.negex = negex
        self.meta = meta_anns
        self.debug = debug_dict


    @classmethod
    def from_entity(cls, entity: [Dict]):

        meta_anns = None
        if entity["meta_anns"]:
            meta_anns = [MetaAnnotations(**value) for value in entity["meta_anns"].values()]

        return Concept(
            id=entity["cui"],
            name=entity["pretty_name"],
            category=None,
            start=entity["start"],
            end=entity["end"],
            negex=entity["negex"] if "negex" in entity else False,
            meta_anns=meta_anns,
        )

    def __str__(self):
        return (
            f"{{name: {self.name}, id: {self.id}, category: {self.category}, start: {self.start}, end: {self.end},"
            f" dosage: {self.dosage}, linked_concept: {self.linked_concept}, negex: {self.negex}, meta: {self.meta}}} "
        )

    def __hash__(self):
        return hash((self.id, self.name, self.category))

    def __eq__(self, other):
        return (
            self.id == other.id
            and self.name == other.name
            and self.category == other.category
        )

    def __lt__(self, other):
        return int(self.id) < int(other.id)

    def __gt__(self, other):
        return int(self.id) > int(other.id)

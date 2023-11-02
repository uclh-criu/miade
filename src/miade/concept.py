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
    SEVERITY = 5
    ALLERGY_TYPE = 6


class Concept(object):
    """docstring for Concept."""

    def __init__(
        self,
        id: str,
        name: str,
        category: Optional[Enum] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        dosage: Optional[Dosage] = None,
        linked_concepts: Optional[List[Concept]] = None,
        negex: Optional[bool] = None,
        meta_anns: Optional[List[MetaAnnotations]] = None,
        debug_dict: Optional[Dict] = None,
    ):

        self.name = name
        self.id = id
        self.category = category
        self.start = start
        self.end = end
        self.dosage = dosage
        self.linked_concepts = linked_concepts
        self.negex = negex
        self.meta = meta_anns
        self.debug = debug_dict

        if linked_concepts is None:
            self.linked_concepts = []

    @classmethod
    def from_entity(cls, entity: [Dict]):
        meta_anns = None
        if entity["meta_anns"]:
            meta_anns = [MetaAnnotations(**value) for value in entity["meta_anns"].values()]

        return Concept(
            id=entity["cui"],
            name=entity["source_value"],  # can also use detected_name which is spell checked but delimited by ~ e.g. liver~failure
            category=None,
            start=entity["start"],
            end=entity["end"],
            negex=entity["negex"] if "negex" in entity else None,
            meta_anns=meta_anns,
        )

    def __str__(self):
        return (
            f"{{name: {self.name}, id: {self.id}, category: {self.category}, start: {self.start}, end: {self.end},"
            f" dosage: {self.dosage}, linked_concepts: {self.linked_concepts}, negex: {self.negex}, meta: {self.meta}}} "
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

from enum import Enum
from typing import Optional, Dict

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
        category: [Category],
        start: Optional[int] = None,
        end: Optional[int] = None,
        dosage: Optional[Dosage] = None,
        negex: Optional[bool] = False,
        meta_anns: Optional[MetaAnnotations] = None,
        debug_dict: Optional[Dict] = None,
    ):

        self.name = name
        self.id = id
        self.category = category
        self.start = start
        self.end = end
        self.dosage = dosage
        self.negex = negex
        self.meta = meta_anns
        self.debug = debug_dict

    @property
    def dosage(self):
        return self._dosage

    @dosage.setter
    def dosage(self, dosage: [Dosage]):
        if dosage is not None:
            if self.category is not Category.MEDICATION:
                raise ValueError(f"Dosage can only be assigned to Medication, not {self.category}.")
        self._dosage = dosage

    @property
    def meta(self):
        return self._meta_annotations

    @meta.setter
    def meta(self, meta_anns: [MetaAnnotations]):
        if meta_anns is not None:
            if not isinstance(meta_anns, MetaAnnotations):
                raise TypeError(f"Type should be MetaAnnotations, not {type(meta_anns)}")
            if self.category is Category.PROBLEM:
                if not (meta_anns.presence or meta_anns.relevance or meta_anns.laterality):
                    raise ValueError("Problems meta-annotations does not have one of presence, relevance or laterality.")

        self._meta_annotations = meta_anns

    @classmethod
    def from_entity(cls, entity: [Dict]):
        meta_anns = MetaAnnotations.from_dict(entity["meta_anns"]) if entity["meta_anns"] else None

        if entity["ontologies"] == ["SNO"] or entity["ontologies"] == ["SNOMED-CT"]:
            category = Category.PROBLEM
        else:
            raise ValueError(f"Entity ontology {entity['ontologies']} not recognised.")

        return Concept(
                id=entity["cui"],
                name=entity["pretty_name"],
                category=category,
                start=entity["start"],
                end=entity["end"],
                negex=entity["negex"] if entity["negex"] else False,
                meta_anns=meta_anns,
            )

    def __str__(self):
        return (
            f"{{name: {self.name}, id: {self.id}, type: {self.category.name}, start: {self.start}, end: {self.end},"
            f" dosage: {self.dosage}, negex: {self.negex},"
            f" meta: {None if not self.meta else self.meta.__dict__}}} "
        )

    def __hash__(self):
        return hash((self.id, self.name, self.category))

    def __eq__(self, other):
        return self.id == other.id and self.name == other.name and self.category == other.category

    def __lt__(self, other):
        return int(self.id) < int(other.id)

    def __gt__(self, other):
        return int(self.id) > int(other.id)

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
        meta_anns: Optional[MetaAnnotations] = None,
        debug_deprecated: Optional[Dict] = None,
    ):

        self.name: str = name
        self.id: str = id
        self.category: Category = category
        self.start: int = start
        self.end: int = end
        self.dosage: Dosage = dosage
        self.meta_annotations = meta_anns
        self.debug_deprecated = debug_deprecated

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
    def meta_annotations(self):
        return self._meta_annotations

    @meta_annotations.setter
    def meta_annotations(self, meta_anns: [MetaAnnotations]):
        if meta_anns is not None:
            if isinstance(meta_anns, MetaAnnotations):
                raise TypeError(f"Type should be MetaAnnotations, not {type(meta_anns)}")
            if self.category == Category.PROBLEM:
                if not (meta_anns.presence and meta_anns.relevance and meta_anns.laterality):
                    raise ValueError("Problem meta-annotations missing presence, relevance, or laterality.")
            if self.category == Category.MEDICATION or self.category == Category.ALLERGY:
                if not (meta_anns.substance and meta_anns.reaction and meta_anns.severity):
                    raise ValueError("Medications or Allergy meta-annotations missing substance, reaction, or severity.")

        self._meta_annotations = meta_anns

    @classmethod
    def from_entity(cls, entity: [Dict]):
        # TODO: assign meds or allergen from meta-annotations here (earlier in the pipeline)
        meta_anns = MetaAnnotations.from_dict(entity["meta_anns"]) if entity["meta_anns"] else None

        if entity["ontologies"] == ["FDB"]:
            category = Category.MEDICATION
        elif entity["ontologies"] == ["SNO"] or entity["ontologies"] == ["SNOMED-CT"]:
            category = Category.PROBLEM
        elif entity["ontologies"] == ["ELG"]:
            category = Category.ALLERGY
        else:
            raise ValueError(f"Entity ontology {entity['ontologies']} not recognised.")

        return Concept(
                id=entity["cui"],
                name=entity["pretty_name"],
                category=category,
                start=entity["start"],
                end=entity["end"],
                meta_anns=meta_anns,
            )

    def __str__(self):
        return (
            f"{{name: {self.name}, id: {self.id}, type: {self.category.name}, start: {self.start}, end: {self.end},"
            f" dosage: {self.dosage}, meta: {self.meta_annotations}}} "
        )

    def __hash__(self):
        return hash((self.id, self.name, self.category))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __lt__(self, other):
        return int(self.id) < int(other.id)

    def __gt__(self, other):
        return int(self.id) > int(other.id)

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel

from .dosage import Dosage
from .metaannotations import MetaAnnotations


class Category(Enum):
    PROBLEM = 1
    MEDICATION = 2
    ALLERGY = 3
    REACTION = 4
    SEVERITY = 5
    ALLERGY_TYPE = 6


class Concept(BaseModel):
    """Represents a concept in the system.

    Attributes:
        id (str): The unique identifier of the concept.
        name (str): The name of the concept.
        category (Optional[Enum]): The category of the concept (optional).
        start (Optional[int]): The start position of the concept (optional).
        end (Optional[int]): The end position of the concept (optional).
        dosage (Optional[Dosage]): The dosage of the concept (optional).
        linked_concepts (Optional[List[Concept]]): The linked concepts of the concept (optional).
        negex (Optional[bool]): The negex value of the concept (optional).
        meta_anns (Optional[List[MetaAnnotations]]): The meta annotations of the concept (optional).
        debug_dict (Optional[Dict]): The debug dictionary of the concept (optional).
    """

    id: str
    name: str
    category: Optional[Enum] = None
    start: Optional[int] = None
    end: Optional[int] = None
    dosage: Optional[Dosage] = None
    linked_concepts: Optional[List[Concept]] = None
    negex: Optional[bool] = None
    meta_anns: Optional[List[MetaAnnotations]] = None
    debug_dict: Optional[Dict] = None

    @classmethod
    def from_entity(cls, entity: Dict) -> Concept:
        """
        Converts an entity dictionary into a Concept object.

        Args:
            entity (Dict): The entity dictionary containing the necessary information.

        Returns:
            The Concept object created from the entity dictionary.
        """
        meta_anns = None
        if entity["meta_anns"]:
            meta_anns = [MetaAnnotations(**value) for value in entity["meta_anns"].values()]

        return Concept(
            id=entity["cui"],
            name=entity[
                "source_value"
            ],  # can also use detected_name which is spell checked but delimited by ~ e.g. liver~failure
            category=None,
            start=entity["start"],
            end=entity["end"],
            negex=entity["negex"] if "negex" in entity else None,
            meta_anns=meta_anns,
        )

    def __str__(self):
        return (
            f"{{name: {self.name}, id: {self.id}, category: {self.category}, start: {self.start}, end: {self.end},"
            f" dosage: {self.dosage}, linked_concepts: {self.linked_concepts}, negex: {self.negex}, meta: {self.meta_anns}}} "
        )

    def __hash__(self):
        return hash((self.id, self.name, self.category))

    def __eq__(self, other):
        return self.id == other.id and self.name == other.name and self.category == other.category

    def __lt__(self, other):
        return int(self.id) < int(other.id)

    def __gt__(self, other):
        return int(self.id) > int(other.id)

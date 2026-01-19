from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Dict, List, Optional

from pydantic import BaseModel

from miade.utils.metaannotationstypes import (
    AllergyType,
    Laterality,
    Presence,
    ReactionPos,
    Relevance,
    Severity,
    SubstanceCategory,
)

from .dosage import Dosage


class Category(StrEnum):
    PROBLEM = "PROBLEM"
    MEDICATION = "MEDICATION"
    ALLERGY = "ALLERGY"
    REACTION = "REACTION"
    SEVERITY = "SEVERITY"
    ALLERGY_TYPE = "ALLERGY_TYPE"


class Concept(BaseModel):
    id: Optional[str] = None
    name: str
    category: Optional[Category] = None
    start: Optional[int] = None
    end: Optional[int] = None
    dosage: Optional[Dosage] = None
    bodysite: List[Concept] = []
    presence: Optional[Presence] = None
    relevance: Optional[Relevance] = None
    substance_category: Optional[SubstanceCategory] = None
    allergy_type: Optional[AllergyType] = None
    laterality: Optional[Laterality] = None
    reaction_position: Optional[ReactionPos] = None
    severity: Optional[Severity] = None
    onset: Optional[datetime] = None
    note: Optional[str] = None
    linked_concepts: List[Concept] = []
    debug_dict: Optional[Dict] = None

    @classmethod
    def from_entity(cls, entity: Dict) -> Concept:
        """
        Converts a medcat entity dictionary into a Concept object.

        Args:
            entity (Dict): The entity dictionary containing the necessary information.

        Returns:
            The Concept object created from the entity dictionary.
        """

        concept = Concept(
            id=entity["cui"],
            name=entity[
                "source_value"
            ],  # can also use detected_name which is spell checked but delimited by ~ e.g. liver~failure
            category=None,
            start=entity["start"],
            end=entity["end"],
        )

        if meta_annotations := entity.get("meta_anns"):
            if presence := meta_annotations.get("presence"):
                if presence.get("value") in [member.value for member in Presence]:
                    concept.presence = Presence(presence["value"])

            if relevance := meta_annotations.get("relevance"):
                if relevance.get("value") in [member.value for member in Relevance]:
                    concept.relevance = Relevance(relevance["value"])

            if substance_category := meta_annotations.get("substance_category"):
                if substance_category.get("value") in [member.value for member in SubstanceCategory]:
                    concept.substance_category = SubstanceCategory(substance_category["value"])

            if allergy_type := meta_annotations.get("allergy_type"):
                if allergy_type.get("value") in [member.value for member in AllergyType]:
                    concept.allergy_type = AllergyType(allergy_type["value"])

            if reaction_position := meta_annotations.get("reaction_pos"):
                if reaction_position.get("value") in [member.value for member in ReactionPos]:
                    concept.reaction_position = ReactionPos(reaction_position["value"])

            if severity := meta_annotations.get("severity"):
                if severity.get("vaule") in [member.value for member in Severity]:
                    concept.severity = Severity(severity["value"])

        if entity.get("negex") is True:
            concept.presence = Presence.NEGATED

        return concept

    def __hash__(self):
        return hash((self.id, self.name, self.category))

    def __eq__(self, other):
        return self.id == other.id and self.name == other.name and self.category == other.category

    def __lt__(self, other):
        return int(self.id) < int(other.id)

    def __gt__(self, other):
        return int(self.id) > int(other.id)

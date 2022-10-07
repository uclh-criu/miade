from typing import List
from .concept import Concept, Category


class ConceptFilter:
    """
    Post-processing for extracted concepts: deduplication, meta-annotations
    """
    def __init__(self, record_concepts: List[Concept], extracted_concepts: List[Concept]):
        self.record_concepts = record_concepts
        self.extracted_concepts = extracted_concepts

    def deduplicate(self) -> List[Concept]:
        self.extracted_concepts = sorted(set(self.extracted_concepts))
        return [
            concept for concept in self.extracted_concepts
            if concept.id not in [
                concept.id for concept in self.record_concepts
            ] and concept.name not in [
                concept.name for concept in self.record_concepts if concept.category == Category.ALLERGY
            ]
        ]

    def disambiguate_meds_allergen(self) -> List[Concept]:
        pass

    def filter_meta(self) -> List[Concept]:
        pass

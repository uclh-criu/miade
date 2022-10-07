from typing import List, Optional
from .concept import Concept, Category


class ConceptFilter:
    """
    Post-processing for extracted concepts: deduplication, meta-annotations
    """

    def __init__(self, extracted_concepts: List[Concept], record_concepts: Optional[List[Concept]] = None):
        self.extracted_concepts = extracted_concepts
        self.record_concepts = record_concepts

    def deduplicate(self) -> List[Concept]:
        self.extracted_concepts = sorted(set(self.extracted_concepts))
        if self.record_concepts is not None:
            return [
                concept for concept in self.extracted_concepts
                if concept.id not in [
                    concept.id for concept in self.record_concepts
                ] and concept.name not in [
                       concept.name for concept in self.record_concepts if concept.category == Category.ALLERGY
                   ]
            ]
        else:
            return self.extracted_concepts

    def disambiguate_meds_allergen(self) -> List[Concept]:
        pass

    def filter_meta(self) -> List[Concept]:
        pass

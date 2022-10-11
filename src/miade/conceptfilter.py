import logging

from typing import List, Optional
from .concept import Concept, Category


log = logging.getLogger(__name__)


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
            filtered_concepts = []
            for extracted_concept in self.extracted_concepts:
                if extracted_concept.id in [record_concept.id for record_concept in self.record_concepts]:
                    # medication code is snomed though - need lookup?
                    log.debug(f"Filtered duplicate problem/medication {extracted_concept}")
                    continue
                if extracted_concept.category == Category.ALLERGY and extracted_concept.name in [
                    record_concept.name for record_concept in self.record_concepts
                    if record_concept.category == Category.ALLERGY
                ]:
                    log.debug(f"Filtered duplicate allergy {extracted_concept}")
                    continue

                filtered_concepts.append(extracted_concept)
            self.extracted_concepts = filtered_concepts

        return self.extracted_concepts

    def disambiguate_meds_allergen(self) -> List[Concept]:
        pass

    def filter_meta(self) -> List[Concept]:
        pass

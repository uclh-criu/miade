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
                    # assume medication code is snomed
                    log.debug(f"Filtered duplicate problem/medication {extracted_concept}")
                    continue
                if extracted_concept.category == Category.ALLERGY and extracted_concept.name in [
                    record_concept.name for record_concept in self.record_concepts
                    if record_concept.category == Category.ALLERGY
                ]:
                    # by text match as epic does not return code
                    log.debug(f"Filtered duplicate allergy {extracted_concept}")
                    continue

                filtered_concepts.append(extracted_concept)
            self.extracted_concepts = filtered_concepts

        return self.extracted_concepts

    def find_overlapping_med_allergen(self) -> List[Concept]:
        """just returns overlapping concepts, to be completed"""
        concept_spans = {}
        for concept in self.extracted_concepts:
            if concept.start is not None and concept.end is not None:
                if (concept.start, concept.end) in concept_spans.keys():
                    concept_spans[(concept.start, concept.end)].append(concept)
                else:
                    concept_spans[(concept.start, concept.end)] = [concept]

        overlapping_concepts = [span[1] for span in concept_spans.items() if len(span[1]) > 1]
        if len(overlapping_concepts) != 0:
            overlapping_concepts = overlapping_concepts[0]

            assert Category.ALLERGY and Category.MEDICATION in [
                concept.category for concept in overlapping_concepts
            ], "Overlapping concepts that are not Allergy or Medication"

        return overlapping_concepts

    def process_meta_annotations(self) -> List[Concept]:
        # TODO
        pass

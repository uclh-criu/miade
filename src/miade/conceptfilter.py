import logging

from typing import List, Optional

from .concept import Concept, Category

log = logging.getLogger(__name__)

default_pipeline_config = [
    "meta-annotations",
    "deduplicate",
]


def deduplicate(extracted_concepts: List[Concept], record_concepts: Optional[List[Concept]]) -> List[Concept]:
    extracted_concepts = sorted(set(extracted_concepts))
    if record_concepts is not None:
        filtered_concepts = []
        for extracted_concept in extracted_concepts:
            if extracted_concept.id in [record_concept.id for record_concept in record_concepts]:
                # assume medication code is snomed
                log.debug(f"Filtered duplicate problem/medication {extracted_concept}")
                continue
            if extracted_concept.category == Category.ALLERGY and extracted_concept.name in [
                record_concept.name for record_concept in record_concepts
                if record_concept.category == Category.ALLERGY
            ]:
                # by text match as epic does not return code
                log.debug(f"Filtered duplicate allergy {extracted_concept}")
                continue

            filtered_concepts.append(extracted_concept)
        extracted_concepts = filtered_concepts

    return extracted_concepts


def find_overlapping_med_allergen(extracted_concepts: List[Concept]) -> List[Concept]:
    """just returns overlapping concepts, to be completed"""
    concept_spans = {}
    for concept in extracted_concepts:
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


class ConceptFilter(object):
    """
    Post-processing for extracted concepts: deduplication, meta-annotations
    """

    def __init__(self, config: [List] = None):
        if config is None:
            self.config = default_pipeline_config

        self.suspected_lookup = None
        self.negated_lookup = None
        self.historic_lookup = None

    def filter(self, extracted_concepts: List[Concept], record_concepts: Optional[List[Concept]]) -> List[Concept]:
        for concept in extracted_concepts:
            if concept.category == Category.PROBLEM:
                self.handle_problem_meta(concept)
            if concept.category == Category.ALLERGY or concept.category == Category.MEDICATION:
                self.handle_meds_allergen_meta(concept)

        # deduplicate after meta-annotations are handled, concepts may be able to be updated
        concepts = deduplicate(extracted_concepts, record_concepts)

        return concepts

    def handle_problem_meta(self, concept: [Concept]) -> Concept:
        # add, convert, or ignore concepts
        pass

    def handle_meds_allergen_meta(self, concept: [Concept]) -> Concept:
        pass

    def __call__(self, extracted_concepts: List[Concept], record_concepts: Optional[List[Concept]] = None):
        return self.filter(extracted_concepts, record_concepts)


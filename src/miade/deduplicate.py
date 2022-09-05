from typing import List
from .concept import Concept


def deduplicate(record_concepts: List[Concept], extracted_concepts: List[Concept]) -> List[Concept]:
    return [
        concept for concept in extracted_concepts
        if concept.id not in [
            concept.id for concept in record_concepts
        ]
    ]

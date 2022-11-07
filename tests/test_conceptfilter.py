from miade.concept import Concept, Category
from miade.conceptfilter import ConceptFilter, is_duplicate
from miade.metaannotations import MetaAnnotations
from miade.utils.metaannotationstypes import *


def test_deduplicate(test_duplicate_concepts_note, test_duplicate_concepts_record):
    concept_filter = ConceptFilter()
    assert concept_filter(test_duplicate_concepts_note, test_duplicate_concepts_record) == [
        Concept(id="5", name="test2", category=Category.PROBLEM),
        Concept(id="7", name="test2", category=Category.MEDICATION),
    ]
    assert concept_filter(extracted_concepts=test_duplicate_concepts_note, record_concepts=None) == [
        Concept(id="1", name="test1", category=Category.PROBLEM),
        Concept(id="2", name="test2", category=Category.PROBLEM),
        Concept(id="3", name="test2", category=Category.PROBLEM),
        Concept(id="4", name="test2", category=Category.PROBLEM),
        Concept(id="5", name="test2", category=Category.PROBLEM),
        Concept(id="6", name="test2", category=Category.MEDICATION),
        Concept(id="7", name="test2", category=Category.MEDICATION),
        Concept(id="8", name="PEANUTS", category=Category.ALLERGY)
    ]
    assert concept_filter(extracted_concepts=test_duplicate_concepts_note, record_concepts=[]) == [
        Concept(id="1", name="test1", category=Category.PROBLEM),
        Concept(id="2", name="test2", category=Category.PROBLEM),
        Concept(id="3", name="test2", category=Category.PROBLEM),
        Concept(id="4", name="test2", category=Category.PROBLEM),
        Concept(id="5", name="test2", category=Category.PROBLEM),
        Concept(id="6", name="test2", category=Category.MEDICATION),
        Concept(id="7", name="test2", category=Category.MEDICATION),
        Concept(id="8", name="PEANUTS", category=Category.ALLERGY)
    ]

    assert concept_filter(extracted_concepts=[], record_concepts=test_duplicate_concepts_record) == []


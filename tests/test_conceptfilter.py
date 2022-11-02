from miade.concept import Concept, Category
from miade.conceptfilter import deduplicate, find_overlapping_med_allergen


def test_deduplicate(test_duplicate_concepts_note, test_duplicate_concepts_record):
    assert deduplicate(test_duplicate_concepts_note, test_duplicate_concepts_record) == [
        Concept(id="5", name="test2", category=Category.PROBLEM),
        Concept(id="7", name="test2", category=Category.MEDICATION),
    ]
    assert deduplicate(extracted_concepts=test_duplicate_concepts_note, record_concepts=None) == [
        Concept(id="1", name="test1", category=Category.PROBLEM),
        Concept(id="2", name="test2", category=Category.PROBLEM),
        Concept(id="3", name="test2", category=Category.PROBLEM),
        Concept(id="4", name="test2", category=Category.PROBLEM),
        Concept(id="5", name="test2", category=Category.PROBLEM),
        Concept(id="6", name="test2", category=Category.MEDICATION),
        Concept(id="7", name="test2", category=Category.MEDICATION),
        Concept(id="8", name="PEANUTS", category=Category.ALLERGY)
    ]
    assert deduplicate(extracted_concepts=test_duplicate_concepts_note, record_concepts=[]) == [
        Concept(id="1", name="test1", category=Category.PROBLEM),
        Concept(id="2", name="test2", category=Category.PROBLEM),
        Concept(id="3", name="test2", category=Category.PROBLEM),
        Concept(id="4", name="test2", category=Category.PROBLEM),
        Concept(id="5", name="test2", category=Category.PROBLEM),
        Concept(id="6", name="test2", category=Category.MEDICATION),
        Concept(id="7", name="test2", category=Category.MEDICATION),
        Concept(id="8", name="PEANUTS", category=Category.ALLERGY)
    ]

    assert deduplicate(extracted_concepts=[], record_concepts=test_duplicate_concepts_record) == []


def test_find_overlapping_med_allergen(test_overlapping_meds_allergen_concepts, test_duplicate_concepts_note):
    assert find_overlapping_med_allergen(test_overlapping_meds_allergen_concepts) == [
        Concept(id="1", name="med", category=Category.MEDICATION, start=30, end=40),
        Concept(id="2", name="allergen", category=Category.ALLERGY, start=30, end=40),
    ]
    assert find_overlapping_med_allergen(test_duplicate_concepts_note) == []

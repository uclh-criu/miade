from miade.concept import Concept, Category
from miade.conceptfilter import ConceptFilter


def test_deduplicate(test_duplicate_concepts_note, test_duplicate_concepts_record):
    assert ConceptFilter(
        extracted_concepts=test_duplicate_concepts_note,
        record_concepts=test_duplicate_concepts_record
    ).deduplicate() == [
        Concept(id="5", name="test2", category=Category.PROBLEM),
        Concept(id="7", name="test2", category=Category.MEDICATION),
    ]
    assert ConceptFilter(
        extracted_concepts=test_duplicate_concepts_note,
        record_concepts=None
    ).deduplicate() == [
        Concept(id="1", name="test1", category=Category.PROBLEM),
        Concept(id="2", name="test2", category=Category.PROBLEM),
        Concept(id="3", name="test2", category=Category.PROBLEM),
        Concept(id="4", name="test2", category=Category.PROBLEM),
        Concept(id="5", name="test2", category=Category.PROBLEM),
        Concept(id="6", name="test2", category=Category.MEDICATION),
        Concept(id="7", name="test2", category=Category.MEDICATION),
        Concept(id="8", name="PEANUTS", category=Category.ALLERGY)
    ]
    assert ConceptFilter(
        extracted_concepts=test_duplicate_concepts_note,
        record_concepts=[]
    ).deduplicate() == [
        Concept(id="1", name="test1", category=Category.PROBLEM),
        Concept(id="2", name="test2", category=Category.PROBLEM),
        Concept(id="3", name="test2", category=Category.PROBLEM),
        Concept(id="4", name="test2", category=Category.PROBLEM),
        Concept(id="5", name="test2", category=Category.PROBLEM),
        Concept(id="6", name="test2", category=Category.MEDICATION),
        Concept(id="7", name="test2", category=Category.MEDICATION),
        Concept(id="8", name="PEANUTS", category=Category.ALLERGY)
    ]

    assert ConceptFilter(
        extracted_concepts=[],
        record_concepts=test_duplicate_concepts_record
    ).deduplicate() == []


def test_find_overlapping_med_allergen(test_overlapping_meds_allergen_concepts, test_duplicate_concepts_note):
    assert ConceptFilter(
        test_overlapping_meds_allergen_concepts
    ).find_overlapping_med_allergen() == [
        Concept(id="1", name="med", category=Category.MEDICATION, start=30, end=40),
        Concept(id="2", name="allergen", category=Category.ALLERGY, start=30, end=40),
    ]
    assert ConceptFilter(
        test_duplicate_concepts_note
    ).find_overlapping_med_allergen() == []

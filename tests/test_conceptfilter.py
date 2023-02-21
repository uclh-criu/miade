from miade.concept import Concept, Category
from miade.conceptfilter import ConceptFilter


def test_deduplicate(
    test_duplicate_concepts_note,
    test_duplicate_concepts_record,
    test_self_duplicate_concepts_note,
):
    concept_filter = ConceptFilter()
    assert concept_filter(
        test_duplicate_concepts_note, test_duplicate_concepts_record
    ) == [
        Concept(id="5", name="test2", category=Category.PROBLEM),
        Concept(id="7", name="test2", category=Category.MEDICATION),
    ]
    assert concept_filter(test_self_duplicate_concepts_note, record_concepts=None) == [
        Concept(id="1", name="test1", category=Category.PROBLEM),
        Concept(id="2", name="test2", category=Category.MEDICATION),
    ]
    assert concept_filter(
        extracted_concepts=test_duplicate_concepts_note, record_concepts=None
    ) == [
        Concept(id="1", name="test1", category=Category.PROBLEM),
        Concept(id="2", name="test2", category=Category.PROBLEM),
        Concept(id="3", name="test2", category=Category.PROBLEM),
        Concept(id="4", name="test2", category=Category.PROBLEM),
        Concept(id="5", name="test2", category=Category.PROBLEM),
        Concept(id="6", name="test2", category=Category.MEDICATION),
        Concept(id="7", name="test2", category=Category.MEDICATION),
        Concept(id="8", name="PEANUTS", category=Category.ALLERGY),
    ]
    assert concept_filter(
        extracted_concepts=test_duplicate_concepts_note, record_concepts=[]
    ) == [
        Concept(id="1", name="test1", category=Category.PROBLEM),
        Concept(id="2", name="test2", category=Category.PROBLEM),
        Concept(id="3", name="test2", category=Category.PROBLEM),
        Concept(id="4", name="test2", category=Category.PROBLEM),
        Concept(id="5", name="test2", category=Category.PROBLEM),
        Concept(id="6", name="test2", category=Category.MEDICATION),
        Concept(id="7", name="test2", category=Category.MEDICATION),
        Concept(id="8", name="PEANUTS", category=Category.ALLERGY),
    ]
    assert (
        concept_filter(
            extracted_concepts=[], record_concepts=test_duplicate_concepts_record
        )
        == []
    )


def test_meta_annotations(test_meta_annotations_concepts):
    concept_filter = ConceptFilter()
    assert concept_filter(extracted_concepts=test_meta_annotations_concepts) == [
        Concept(id="274826007", name="Nystagmus (negated)", category=Category.PROBLEM),  # negex true, meta ignored
        Concept(
            id="302064001", name="Lymphangitis (negated)", category=Category.PROBLEM
        ),  # negex true, meta ignored
        Concept(
            id="413241009", name="Gastritis (suspected)", category=Category.PROBLEM
        ),
        Concept(
            id="431956005", name="Arthritis (suspected)", category=Category.PROBLEM
        ),  # negex false, meta processed
        Concept(
            id="115451000119100",
            name="Endophthalmitis (historic)",
            category=Category.PROBLEM,
        ),  # negex false, meta processed
    ]
    # test just using negex for negation
    test_meta_annotations_concepts[0].negex = True
    test_meta_annotations_concepts[0].meta = None
    test_meta_annotations_concepts[1].negex = False
    test_meta_annotations_concepts[2].negex = True
    test_meta_annotations_concepts[3].negex = True
    test_meta_annotations_concepts[6].negex = True

    assert concept_filter(extracted_concepts=test_meta_annotations_concepts) == [
        Concept(
            id="1415005", name="Lymphangitis", category=Category.PROBLEM
        ),  # negex false, meta processed but ignore negation
        Concept(id="274826007", name="Nystagmus (negated)", category=Category.PROBLEM),  # negex true, meta empty
        Concept(
            id="413241009", name="Gastritis (suspected)", category=Category.PROBLEM
        ),  # negex false, meta processed
    ]

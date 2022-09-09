from miade import deduplicate, Concept
from miade.concept import Category


def test_deduplicate(test_duplicate_concepts_record, test_duplicate_concepts_note):
    assert deduplicate(test_duplicate_concepts_record, test_duplicate_concepts_note) == [
        Concept(id="5", name="test2", category=Category.PROBLEM),
        Concept(id="6", name="test2", category=Category.PROBLEM),
    ]

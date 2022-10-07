from miade.concept import Concept, Category
from miade.conceptfilter import ConceptFilter


def test_deduplicate(test_duplicate_concepts_record, test_duplicate_concepts_note):
    assert ConceptFilter(test_duplicate_concepts_record, test_duplicate_concepts_note).deduplicate() == [
        Concept(id="5", name="test2", category=Category.PROBLEM),
        Concept(id="6", name="test2", category=Category.PROBLEM),
    ]

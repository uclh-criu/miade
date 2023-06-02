from miade.core import NoteProcessor
from miade.concept import Concept, Category
from miade.metaannotations import MetaAnnotations
from miade.utils.metaannotationstypes import *

def test_core(model_directory_path, test_note, test_negated_note, test_duplicated_note):
    processor = NoteProcessor(model_directory_path)
    processor.add_annotator("problems")
    # processor.add_annotator("meds/allergies")  # TODO: we'll need a separate meds model to test this

    assert processor.process(test_note) == [
        Concept(id="3", name="Liver failure", category=Category.PROBLEM),
        Concept(id="10", name="Paracetamol", category=Category.PROBLEM),
    ]
    assert processor.process(test_negated_note) == [
        Concept(id="10", name="Paracetamol", category=Category.PROBLEM),
    ]
    assert processor.process(test_duplicated_note) == [
        Concept(id="3", name="Liver failure", category=Category.PROBLEM),
        Concept(id="10", name="Paracetamol", category=Category.PROBLEM),
    ]


def test_meta_from_entity(test_medcat_concepts):
    assert Concept.from_entity(test_medcat_concepts["0"]) == Concept(
        id="0",
        name="problem",
        category=None,
        start=4,
        end=11,
        meta_anns=[
            MetaAnnotations(name="presence", value=Presence.NEGATED),
            MetaAnnotations(name="relevance", value=Relevance.HISTORIC)
        ]
    )
    assert Concept.from_entity(test_medcat_concepts["1"]) == Concept(
        id="0",
        name="problem",
        category=None,
        start=4,
        end=11,
        meta_anns=[
            MetaAnnotations(name="presence", value=Presence.SUSPECTED, confidence=1),
            MetaAnnotations(name="relevance", value=Relevance.IRRELEVANT, confidence=1),
            MetaAnnotations(name="laterality (generic)", value=Laterality.NO_LATERALITY, confidence=1)
        ]
    )


from miade.core import NoteProcessor
from miade.concept import Concept, Category
from miade.metaannotations import MetaAnnotations
from miade.utils.metaannotationstypes import *
import logging

def test_core(model_directory_path, test_note, test_negated_note, test_duplicated_note):
    processor = NoteProcessor(model_directory_path, log_level=logging.DEBUG)
    processor.add_annotator("problems")
    # processor.add_annotator("meds/allergies")

    assert processor.process(test_note) == [
        Concept(id="3", name="Liver failure", category=Category.PROBLEM),
        Concept(id="10", name="Paracetamol", category=Category.MEDICATION),
    ]
    assert processor.process(test_negated_note) == [
        Concept(id="10", name="Paracetamol", category=Category.MEDICATION),
    ]
    assert processor.process(test_duplicated_note) == [
        Concept(id="3", name="Liver failure", category=Category.PROBLEM),
        Concept(id="10", name="Paracetamol", category=Category.MEDICATION),
    ]


def test_meta_from_entity(test_medcat_concepts):
    assert Concept.from_entity(test_medcat_concepts["0"]) == Concept(
        id="0",
        name="problem",
        category=Category.PROBLEM,
        start=4,
        end=11,
        meta_anns=MetaAnnotations(
            presence=Presence.NEGATED, relevance=Relevance.HISTORIC, laterality=None
        ),
    )
    assert Concept.from_entity(test_medcat_concepts["1"]) == Concept(
        id="0",
        name="problem",
        category=Category.PROBLEM,
        start=4,
        end=11,
        meta_anns=MetaAnnotations(
            presence=Presence.SUSPECTED,
            relevance=Relevance.IRRELEVANT,
            laterality=Laterality.NO_LATERALITY,
            confidences={Presence: 1, Relevance: 1, Laterality: 1},
        ),
    )


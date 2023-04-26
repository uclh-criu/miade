from miade.core import NoteProcessor
from miade.concept import Concept, Category
from miade.dosage import Dose, Frequency
from miade.metaannotations import MetaAnnotations
from miade.utils.metaannotationstypes import *


def test_core(model_directory_path, test_note, test_negated_note, test_duplicated_note):
    processor = NoteProcessor(model_directory_path, problems_model_id="1", meds_allergies_model_id="2")
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


def test_model_id_override(model_directory_path, test_note):
    processor = NoteProcessor(model_directory_path, problems_model_id="test_probs")
    processor.annotators[0].config.version["id"] = "test_probs"
    processor.annotators[0].config.version["ontology"] = "FDB"

    assert processor.process(test_note) == [
        Concept(id="3", name="Liver failure", category=Category.PROBLEM),
        Concept(id="10", name="Paracetamol", category=Category.PROBLEM),
    ]
    processor.annotators[0].config.version["id"] = "test_meds"
    processor.annotators[0].config.version["ontology"] = "SNO"
    processor.meds_allergies_model_id = "test_meds"

    assert processor.process(test_note) == [
        Concept(id="3", name="Liver failure", category=Category.MEDICATION),
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


def test_dosage_text_splitter(model_directory_path, test_med_concepts, test_med_note):
    processor = NoteProcessor(model_directory_path)
    concepts = processor.add_dosages_to_concepts(test_med_concepts, test_med_note)

    assert concepts[0].dosage.text == "Magnesium hydroxide 75mg daily "
    assert concepts[1].dosage.text == "paracetamol 500mg po 3 times a day as needed."
    assert concepts[2].dosage.text == "aspirin IM q daily x 2 weeks with concurrent "
    assert concepts[3].dosage.text == "DOXYCYCLINE 500mg tablets for two weeks"

    assert concepts[0].dosage.dose == Dose(
        source="75 mg", value=75, unit="mg", low=None, high=None
    )

    assert concepts[0].dosage.frequency == Frequency(
        source="start 75 mg every day ",
        value=1.0,
        unit="d",
        low=None,
        high=None,
        standardDeviation=None,
        institutionSpecified=False,
        preconditionAsRequired=False,
    )

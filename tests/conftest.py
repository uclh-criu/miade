import pytest
import pandas as pd

from typing import List, Dict
from pathlib import Path
from miade.annotators import Annotator

from miade.dosage import Dosage, Dose, Route
from miade.note import Note
from miade.concept import Concept, Category
from miade.metaannotations import MetaAnnotations
from miade.utils.annotatorconfig import AnnotatorConfig
from miade.utils.metaannotationstypes import (
    Presence,
    Relevance,
    Laterality,
    ReactionPos,
    SubstanceCategory,
    AllergyType,
    Severity,
)
from miade.utils.miade_cat import MiADE_CAT


@pytest.fixture(scope="function")
def model_directory_path() -> Path:
    return Path("./tests/data/models/")


@pytest.fixture(scope="function")
def test_problems_medcat_model() -> MiADE_CAT:
    return MiADE_CAT.load_model_pack(
        str("./tests/data/models/miade_problems_blank_modelpack_Jun_2023_df349473b9d260a9.zip")
    )


@pytest.fixture(scope="function")
def test_meds_algy_medcat_model() -> MiADE_CAT:
    return MiADE_CAT.load_model_pack(
        str("./tests/data/models/miade_meds_allergy_blank_modelpack_Jun_2023_75e13bf042cc55b8.zip")
    )


@pytest.fixture(scope="function")
def temp_dir() -> Path:
    return Path("./tests/data/temp")


@pytest.fixture(scope="function")
def snomed_data_path() -> Path:
    return Path("./tests/examples/example_snomed_sct2_20211124000001Z/")


@pytest.fixture(scope="function")
def snomed_subset_path() -> Path:
    return Path("./tests/examples/example_snomed_subset.csv")


@pytest.fixture(scope="function")
def fdb_data_path() -> Path:
    return Path("./tests/examples/example_fdb.csv")


@pytest.fixture(scope="function")
def elg_data_path() -> Path:
    return Path("./tests/examples/example_elg.csv")


@pytest.fixture(scope="function")
def vocab_data_path() -> Path:
    return Path("./tests/examples/vocab_data.txt")


@pytest.fixture(scope="function")
def text_data_path() -> Path:
    return Path("./tests/examples/wikipedia_sample.txt")


@pytest.fixture(scope="function")
def cdb_data_path() -> Path:
    return Path("./tests/examples/cdb.dat")


@pytest.fixture(scope="function")
def cdb_csv_paths() -> List[Path]:
    return [
        Path("./tests/data/preprocessed_snomed.csv"),
        Path("./tests/data/preprocessed_fdb.csv"),
        Path("./tests/data/preprocessed_elg.csv"),
    ]


@pytest.fixture(scope="function")
def test_config() -> AnnotatorConfig:
    return AnnotatorConfig()


@pytest.fixture(scope="function")
def test_annotator() -> Annotator:
    class CustomAnnotator(Annotator):
        def __init__(self, cat, config):
            super().__init__(cat, config)

        @property
        def concept_types():
            return []

        @property
        def pipeline():
            return []

        def postprocess(self):
            return super().postprocess()

        def process_paragraphs(self):
            return super().process_paragraphs()

    return CustomAnnotator


@pytest.fixture(scope="function")
def test_miade_doses() -> (List[Note], pd.DataFrame):
    extracted_doses = pd.read_csv("./tests/examples/common_doses_for_miade.csv")
    return [Note(text=dose) for dose in extracted_doses.dosestring.to_list()], extracted_doses


@pytest.fixture(scope="function")
def test_miade_med_concepts() -> List[Concept]:
    data = pd.read_csv("./tests/examples/common_doses_for_miade.csv")
    return [Concept(id="387337001", name=drug, category=Category.MEDICATION) for drug in data.drug.to_list()]


@pytest.fixture(scope="function")
def test_med_note() -> Note:
    return Note(
        text="Magnesium hydroxide 75mg daily \nparacetamol 500mg po 3 times a day as needed.\n"
        "Patient treated with aspirin IM q daily x 2 weeks with concurrent DOXYCYCLINE 500mg tablets for "
        "two weeks"
    )


@pytest.fixture(scope="function")
def test_med_concepts() -> List[Concept]:
    return [
        Concept(
            id="0",
            name="Magnesium hydroxide",
            category=Category.MEDICATION,
            start=0,
            end=19,
        ),
        Concept(id="1", name="Paracetamol", category=Category.MEDICATION, start=32, end=43),
        Concept(id="2", name="Aspirin", category=Category.MEDICATION, start=99, end=107),
        Concept(id="3", name="Doxycycline", category=Category.MEDICATION, start=144, end=156),
    ]


@pytest.fixture(scope="function")
def test_note() -> Note:
    return Note(text="Patient has liver failure and is taking paracetamol 500mg oral tablets.")


@pytest.fixture(scope="function")
def test_negated_note() -> Note:
    return Note(text="Patient does not have liver failure. Patient is taking paracetamol 500mg oral tablets.")


@pytest.fixture(scope="function")
def test_duplicated_note() -> Note:
    return Note(
        text="Patient has liver failure. The liver failure is quite bad. Patient is taking "
        "paracetamol 500mg oral tablets. decrease paracetamol 500mg oral tablets dosage."
    )


@pytest.fixture(scope="function")
def test_clean_and_paragraphing_note() -> Note:
    return Note(
        """
    This is an example of text with various types of spaces: 
\tTabs,    \u00a0Non-breaking spaces, \u2003Em spaces, \u2002En spaces.
Some lines may contain only punctuation and spaces, like this:
    !?  ...  - -- ???
    \n
But others may have meaningful content. Detecting ?=suspected in this sentence.

Test Paragraph Chunking:
some prose here
Penicillin

PMH:
Penicillin
Penicillin

Current meds:
Penicillin 500mg tablets 2 tabs qds prn
Penicillin

Allergies:
Penicillin - rash
Penicillin

Problems:
Penicillin
Penicillin

Plan
Penicillin
Penicillin

imp::
Penicillin
    """
    )


@pytest.fixture(scope="function")
def test_paragraph_chunking_prob_concepts() -> List[Concept]:
    return [
        # prose
        Concept(
            id="764146007",
            name="Penicillin",
            category=Category.PROBLEM,
            negex=False,
            start=304,
            end=314,
            meta_anns=[
                MetaAnnotations(name="presence", value=Presence.CONFIRMED),
                MetaAnnotations(name="relevance", value=Relevance.PRESENT),
            ],
        ),
        # pmh
        Concept(
            id="764146007",
            name="Penicillin",
            category=Category.PROBLEM,
            negex=False,
            start=320,
            end=330,
            meta_anns=[
                MetaAnnotations(name="presence", value=Presence.CONFIRMED),
                MetaAnnotations(name="relevance", value=Relevance.IRRELEVANT),
            ],
        ),
        Concept(
            id="764146007",
            name="Penicillin",
            category=Category.PROBLEM,
            negex=False,
            start=396,
            end=406,
            meta_anns=[
                MetaAnnotations(name="presence", value=Presence.CONFIRMED),
                MetaAnnotations(name="relevance", value=Relevance.PRESENT),
            ],
        ),
        # allergies
        Concept(
            id="764146007",
            name="Penicillin",
            category=Category.PROBLEM,
            negex=False,
            start=418,
            end=428,
            meta_anns=[
                MetaAnnotations(name="presence", value=Presence.CONFIRMED),
                MetaAnnotations(name="relevance", value=Relevance.PRESENT),
            ],
        ),
        # probs
        Concept(
            id="764146007",
            name="Penicillin",
            category=Category.PROBLEM,
            negex=False,
            start=456,
            end=466,
            meta_anns=[
                MetaAnnotations(name="presence", value=Presence.CONFIRMED),
                MetaAnnotations(name="relevance", value=Relevance.IRRELEVANT),
            ],
        ),
        # plan
        Concept(
            id="764146007",
            name="Penicillin",
            category=Category.PROBLEM,
            negex=False,
            start=484,
            end=494,
            meta_anns=[
                MetaAnnotations(name="presence", value=Presence.CONFIRMED),
                MetaAnnotations(name="relevance", value=Relevance.PRESENT),
            ],
        ),
    ]


@pytest.fixture(scope="function")
def test_paragraph_chunking_med_concepts() -> List[Concept]:
    return [
        Concept(
            id="764146007",
            name="Penicillin",
            category=Category.MEDICATION,
            negex=False,
            start=331,
            end=341,
            meta_anns=[
                MetaAnnotations(name="substance_category", value=SubstanceCategory.TAKING),
            ],
        ),
        # meds
        Concept(
            id="764146007",
            name="Penicillin",
            category=Category.MEDICATION,
            negex=False,
            start=356,
            end=366,
            meta_anns=[
                MetaAnnotations(name="substance_category", value=SubstanceCategory.IRRELEVANT),
            ],
        ),
        # allergies
        Concept(
            id="764146007",
            name="Penicillin",
            category=Category.MEDICATION,
            negex=False,
            start=435,
            end=445,
            meta_anns=[
                MetaAnnotations(name="substance_category", value=SubstanceCategory.TAKING),
            ],
        ),
        # probs
        Concept(
            id="764146007",
            name="Penicillin",
            category=Category.MEDICATION,
            negex=False,
            start=467,
            end=477,
            meta_anns=[
                MetaAnnotations(name="substance_category", value=SubstanceCategory.ADVERSE_REACTION),
            ],
        ),
        # plan
        Concept(
            id="764146007",
            name="Penicillin",
            category=Category.MEDICATION,
            negex=False,
            start=495,
            end=505,
            meta_anns=[
                MetaAnnotations(name="substance_category", value=SubstanceCategory.ADVERSE_REACTION),
            ],
        ),
        # imp
        Concept(
            id="764146007",
            name="Penicillin",
            category=Category.MEDICATION,
            negex=False,
            start=511,
            end=521,
            meta_anns=[
                MetaAnnotations(name="substance_category", value=SubstanceCategory.ADVERSE_REACTION),
            ],
        ),
    ]


@pytest.fixture(scope="function")
def test_problem_list_limit_note() -> Note:
    return Note(
        """
some prose here
Patient has irrelevant condition such as fear

Problems:
This is a structured list, so prose should be completely ignored
Fever
Diabetes

Allergies:
Penicillin - rash

pmh:
Arthritis
"""
    )


@pytest.fixture(scope="function")
def test_problem_list_limit_concepts() -> List[Concept]:
    return [
        Concept(
            id="0",
            name="Fear",
            category=Category.PROBLEM,
            negex=False,
            start=58,
            end=62,
            meta_anns=[
                MetaAnnotations(name="presence", value=Presence.CONFIRMED),
                MetaAnnotations(name="relevance", value=Relevance.PRESENT),
            ],
        ),
        Concept(
            id="1",
            name="Fever",
            category=Category.PROBLEM,
            negex=False,
            start=138,
            end=143,
            meta_anns=[
                MetaAnnotations(name="presence", value=Presence.CONFIRMED),
                MetaAnnotations(name="relevance", value=Relevance.PRESENT),
            ],
        ),
        Concept(
            id="2",
            name="Diabetes",
            category=Category.PROBLEM,
            negex=False,
            start=144,
            end=152,
            meta_anns=[
                MetaAnnotations(name="presence", value=Presence.CONFIRMED),
                MetaAnnotations(name="relevance", value=Relevance.PRESENT),
            ],
        ),
        Concept(
            id="3",
            name="Penicillin",
            category=Category.PROBLEM,
            negex=False,
            start=165,
            end=175,
            meta_anns=[
                MetaAnnotations(name="presence", value=Presence.CONFIRMED),
                MetaAnnotations(name="relevance", value=Relevance.PRESENT),
            ],
        ),
        Concept(
            id="4",
            name="Arthritis",
            category=Category.PROBLEM,
            negex=False,
            start=189,
            end=196,
            meta_anns=[
                MetaAnnotations(name="presence", value=Presence.CONFIRMED),
                MetaAnnotations(name="relevance", value=Relevance.IRRELEVANT),
            ],
        ),
    ]


@pytest.fixture(scope="function")
def test_duplicate_concepts_record() -> List[Concept]:
    return [
        Concept(id="1", name="test1", category=Category.PROBLEM),
        Concept(id="2", name="test2", category=Category.PROBLEM),
        Concept(id="3", name="test2", category=Category.PROBLEM),
        Concept(id="4", name="test2", category=Category.PROBLEM),
        Concept(id="6", name="test2", category=Category.MEDICATION),
    ]


@pytest.fixture(scope="function")
def test_duplicate_concepts_note() -> List[Concept]:
    return [
        Concept(id="1", name="test1", category=Category.PROBLEM),
        Concept(id="2", name="test2", category=Category.PROBLEM),
        Concept(id="3", name="test2", category=Category.PROBLEM),
        Concept(id="4", name="test2", category=Category.PROBLEM),
        Concept(id="7", name="test2", category=Category.MEDICATION, start=0, end=12),
        Concept(id="5", name="test2", category=Category.PROBLEM, start=15, end=20),
        Concept(id="5", name="test2", category=Category.PROBLEM, start=45, end=50),
        Concept(id="6", name="test2", category=Category.MEDICATION),
        Concept(id="6", name="test2", category=Category.MEDICATION),
    ]


@pytest.fixture(scope="function")
def test_self_duplicate_concepts_note() -> List[Concept]:
    return [
        Concept(id="1", name="test1", category=Category.PROBLEM),
        Concept(id="1", name="test1", category=Category.PROBLEM),
        Concept(id="1", name="test1", category=Category.PROBLEM),
        Concept(id="1", name="test1", category=Category.PROBLEM),
        Concept(id="2", name="test2", category=Category.MEDICATION),
        Concept(id="2", name="test2", category=Category.MEDICATION),
    ]


@pytest.fixture(scope="function")
def test_duplicate_vtm_concept_note() -> List[Concept]:
    return [
        Concept(id="1", name="prob1", category=Category.PROBLEM),
        Concept(id="1", name="prob2", category=Category.PROBLEM),
        Concept(id=None, name="vtm1", category=Category.MEDICATION),
        Concept(id=None, name="vtm1", category=Category.MEDICATION),
        Concept(id=None, name="vtm2", category=Category.MEDICATION),
        Concept(id="2", name="vmp 20mg", category=Category.MEDICATION),
        Concept(id=None, name="vtm3", category=Category.MEDICATION),
    ]


@pytest.fixture(scope="function")
def test_duplicate_vtm_concept_record() -> List[Concept]:
    return [
        Concept(id="1", name="prob1", category=Category.PROBLEM),
        Concept(id=None, name="vtm2", category=Category.MEDICATION),
        Concept(id="2", name="vmp 20mg", category=Category.MEDICATION),
    ]


@pytest.fixture(scope="function")
def test_medcat_concepts() -> Dict:
    return {
        "0": {
            "pretty_name": "problem",
            "cui": "0",
            "ontologies": ["SNO"],
            "source_value": "problem",
            "detected_name": "problem",
            "acc": 0.99,
            "context_similarity": 0.99,
            "start": 4,
            "end": 11,
            "id": 0,
            "negex": False,
            "meta_anns": {
                "presence": {"value": "negated", "confidence": 1, "name": "presence"},
                "relevance": {
                    "value": "historic",
                    "confidence": 1,
                    "name": "relevance",
                },
            },
        },
        "1": {
            "pretty_name": "problem",
            "cui": "0",
            "ontologies": ["SNO"],
            "source_value": "problem",
            "detected_name": "problem",
            "acc": 0.99,
            "context_similarity": 0.99,
            "start": 4,
            "end": 11,
            "id": 0,
            "negex": False,
            "meta_anns": {
                "presence": {"value": "suspected", "confidence": 1, "name": "presence"},
                "relevance": {
                    "value": "irrelevant",
                    "confidence": 1,
                    "name": "relevance",
                },
                "laterality (generic)": {
                    "value": "none",
                    "confidence": 1,
                    "name": "laterality (generic)",
                },
            },
        },
    }


@pytest.fixture(scope="function")
def test_meta_annotations_concepts() -> List[Concept]:
    return [
        Concept(
            id="563001",
            name="Nystagmus",
            category=Category.PROBLEM,
            negex=True,
            meta_anns=[
                MetaAnnotations(name="presence", value=Presence.CONFIRMED),
                MetaAnnotations(name="relevance", value=Relevance.PRESENT),
                MetaAnnotations(name="laterality (generic)", value=Laterality.LEFT),
            ],
        ),
        Concept(
            id="1415005",
            name="Lymphangitis",
            category=Category.PROBLEM,
            negex=True,
            meta_anns=[
                MetaAnnotations(name="presence", value=Presence.NEGATED),
                MetaAnnotations(name="relevance", value=Relevance.PRESENT),
                MetaAnnotations(name="laterality (generic)", value=Laterality.NO_LATERALITY),
            ],
        ),
        Concept(
            id="123",
            name="negated concept",
            category=Category.PROBLEM,
            negex=True,
            meta_anns=[
                MetaAnnotations(name="presence", value=Presence.NEGATED),
                MetaAnnotations(name="relevance", value=Relevance.PRESENT),
                MetaAnnotations(name="laterality (generic)", value=Laterality.NO_LATERALITY),
            ],
        ),
        Concept(
            id="3723001",
            name="Arthritis",
            category=Category.PROBLEM,
            negex=False,
            meta_anns=[
                MetaAnnotations(name="presence", value=Presence.SUSPECTED),
                MetaAnnotations(name="relevance", value=Relevance.PRESENT),
                MetaAnnotations(name="laterality (generic)", value=Laterality.NO_LATERALITY),
            ],
        ),
        Concept(
            id="4556007",
            name="Gastritis",
            category=Category.PROBLEM,
            negex=False,
            meta_anns=[
                MetaAnnotations(name="presence", value=Presence.SUSPECTED),
                MetaAnnotations(name="relevance", value=Relevance.PRESENT),
                MetaAnnotations(name="laterality (generic)", value=Laterality.NO_LATERALITY),
            ],
        ),
        Concept(
            id="0000",
            name="suspected concept",
            category=Category.PROBLEM,
            negex=False,
            meta_anns=[
                MetaAnnotations(name="presence", value=Presence.SUSPECTED),
                MetaAnnotations(name="relevance", value=Relevance.PRESENT),
                MetaAnnotations(name="laterality (generic)", value=Laterality.NO_LATERALITY),
            ],
        ),
        Concept(
            id="1847009",
            name="Endophthalmitis",
            category=Category.PROBLEM,
            negex=False,
            meta_anns=[
                MetaAnnotations(name="presence", value=Presence.CONFIRMED),
                MetaAnnotations(name="relevance", value=Relevance.HISTORIC),
                MetaAnnotations(name="laterality (generic)", value=Laterality.NO_LATERALITY),
            ],
        ),
        Concept(
            id="1912002",
            name="Fall",
            category=Category.PROBLEM,
            negex=False,
            meta_anns=[
                MetaAnnotations(name="presence", value=Presence.CONFIRMED),
                MetaAnnotations(name="relevance", value=Relevance.IRRELEVANT),
                MetaAnnotations(name="laterality (generic)", value=Laterality.NO_LATERALITY),
            ],
        ),
        Concept(
            id="0000",
            name="historic concept",
            category=Category.PROBLEM,
            negex=False,
            meta_anns=[
                MetaAnnotations(name="presence", value=Presence.CONFIRMED),
                MetaAnnotations(name="relevance", value=Relevance.HISTORIC),
                MetaAnnotations(name="laterality (generic)", value=Laterality.NO_LATERALITY),
            ],
        ),
    ]


@pytest.fixture(scope="function")
def test_filtering_list_concepts() -> List[Concept]:
    return [
        Concept(id="2704003", name="Acute disease", category=Category.PROBLEM),
        Concept(id="13543005", name="Pressure", category=Category.PROBLEM),
        Concept(id="19342008", name="Subacute disease", category=Category.PROBLEM),
        Concept(id="76797004", name="Failure", category=Category.PROBLEM),
        Concept(id="123", name="real concept", category=Category.PROBLEM),
    ]


@pytest.fixture(scope="function")
def test_meds_allergy_note() -> Note:
    return Note(
        text="Intolerant of eggs mild rash. Allergies: moderate nausea due to penicillin. Taking paracetamol for pain."
    )


@pytest.fixture(scope="function")
def test_substance_concepts_with_meta_anns() -> List[Concept]:
    return [
        Concept(
            id="226021002",
            name="Eggs",
            start=14,
            end=17,
            meta_anns=[
                MetaAnnotations(name="reaction_pos", value=ReactionPos.NOT_REACTION),
                MetaAnnotations(name="category", value=SubstanceCategory.ADVERSE_REACTION),
                MetaAnnotations(name="allergy_type", value=AllergyType.INTOLERANCE),
                MetaAnnotations(name="severity", value=Severity.MILD),
            ],
        ),
        Concept(
            id="159002",
            name="Penicillin",
            start=64,
            end=73,
            meta_anns=[
                MetaAnnotations(name="reaction_pos", value=ReactionPos.NOT_REACTION),
                MetaAnnotations(name="category", value=SubstanceCategory.ADVERSE_REACTION),
                MetaAnnotations(name="allergy_type", value=AllergyType.ALLERGY),
                MetaAnnotations(name="severity", value=Severity.MODERATE),
            ],
        ),
        Concept(
            id="140004",
            name="Rash",
            start=24,
            end=27,
            meta_anns=[
                MetaAnnotations(name="reaction_pos", value=ReactionPos.AFTER_SUBSTANCE),
                MetaAnnotations(name="category", value=SubstanceCategory.NOT_SUBSTANCE),
                MetaAnnotations(name="allergy_type", value=AllergyType.UNSPECIFIED),
                MetaAnnotations(name="severity", value=Severity.UNSPECIFIED),
            ],
        ),
        Concept(
            id="832007",
            name="Nausea",
            start=50,
            end=55,
            meta_anns=[
                MetaAnnotations(name="reaction_pos", value=ReactionPos.BEFORE_SUBSTANCE),
                MetaAnnotations(name="category", value=SubstanceCategory.ADVERSE_REACTION),
                MetaAnnotations(name="allergy_type", value=AllergyType.UNSPECIFIED),
                MetaAnnotations(name="severity", value=Severity.UNSPECIFIED),
            ],
        ),
        Concept(
            id="7336002",
            name="Paracetamol",
            start=83,
            end=93,
            dosage=Dosage(dose=Dose(value=50, unit="mg"), duration=None, frequency=None, route=None),
            meta_anns=[
                MetaAnnotations(name="reaction_pos", value=ReactionPos.NOT_REACTION),
                MetaAnnotations(name="category", value=SubstanceCategory.TAKING),
                MetaAnnotations(name="allergy_type", value=AllergyType.UNSPECIFIED),
                MetaAnnotations(name="severity", value=Severity.UNSPECIFIED),
            ],
        ),
    ]


@pytest.fixture(scope="function")
def test_vtm_concepts() -> List[Concept]:
    return [
        Concept(
            id="302007",
            name="Spiramycin",
            category=Category.MEDICATION,
            dosage=Dosage(
                dose=Dose(value=10, unit="mg"),
                duration=None,
                frequency=None,
                route=None,
            ),
        ),
        Concept(
            id="7336002",
            name="Paracetamol",
            category=Category.MEDICATION,
            dosage=Dosage(
                dose=Dose(value=50, unit="mg"),
                duration=None,
                frequency=None,
                route=None,
            ),
        ),
        Concept(
            id="7947003",
            name="Aspirin",
            category=Category.MEDICATION,
            dosage=Dosage(
                dose=None,
                duration=None,
                frequency=None,
                route=Route(full_name="Oral", value="C38288"),
            ),
        ),
        Concept(id="6247001", name="Folic acid", category=Category.MEDICATION, dosage=None),
        Concept(
            id="350057002",
            name="Selenium",
            category=Category.MEDICATION,
            dosage=Dosage(
                dose=Dose(value=50, unit="microgram"),
                duration=None,
                frequency=None,
                route=None,
            ),
        ),
        Concept(
            id="350057002",
            name="Selenium",
            category=Category.MEDICATION,
            dosage=Dosage(
                dose=Dose(value=10, unit="microgram"),
                duration=None,
                frequency=None,
                route=None,
            ),
        ),
    ]

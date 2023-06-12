from miade.core import NoteProcessor
from miade.concept import Concept, Category
from miade.annotators import Annotator
from miade.metaannotations import MetaAnnotations
from miade.utils.metaannotationstypes import *

def test_core(model_directory_path, test_note, test_negated_note, test_duplicated_note):
    processor = NoteProcessor(model_directory_path)

    processor.add_annotator("problems", use_negex=True)
    processor.add_annotator("meds/allergies")

    assert processor.process(test_note) == [
        Concept(id="59927004", name="hepatic failure", category=Category.PROBLEM),
        Concept(id="322236009", name="acetaminophen 500mg oral tablet", category=Category.MEDICATION),
    ]
    assert processor.process(test_negated_note) == [
        Concept(id="322236009", name="acetaminophen 500mg oral tablet", category=Category.MEDICATION),
    ]
    assert processor.process(test_duplicated_note) == [
        Concept(id="59927004", name="hepatic failure", category=Category.PROBLEM),
        Concept(id="322236009", name="acetaminophen 500mg oral tablet", category=Category.MEDICATION),
    ]
    assert processor.get_concept_dicts(test_note) == [
        {
            'name': 'hepatic failure', 'id': '59927004', 'category': 'PROBLEM', 'start': 12, 'end': 25,
            'dosage': None, 'negex': False, 'meta': None, 'debug': None
        },
        {
            'name': 'acetaminophen 500mg oral tablet', 'id': '322236009', 'category': 'MEDICATION', 'start': 40, 'end': 70,
            'dosage': {
                'dose': {
                    'source': '500 mg by mouth tab', 'value': 500.0, 'unit': '{tbl}', 'low': None, 'high': None
                },
                'duration': None,
                'frequency': None,
                'route': {
                    'source': 'by mouth', 'full_name': 'Oral', 'value': 'C38288', 'code_system': 'NCI Thesaurus'
                }
            },
         'negex': False, 'meta': None, 'debug': None
        }
    ]


def test_adding_removing_annotators(model_directory_path):
    processor = NoteProcessor(model_directory_path)

    processor.add_annotator("problems")
    processor.add_annotator("meds/allergies", use_negex=False)

    assert len(processor.annotators) == 2
    processor.print_model_cards()

    processor.remove_annotator("problems")
    assert len(processor.annotators) == 1
    processor.print_model_cards()

    processor.remove_annotator("meds/allergies")
    assert len(processor.annotators) == 0

    processor.print_model_cards()

def test_adding_custom_annotators(model_directory_path):
    class CustomAnnotator(Annotator):
        pass
    processor = NoteProcessor(model_directory_path, custom_annotators=[CustomAnnotator])

    processor.add_annotator("custom")
    assert len(processor.annotators) == 1

    processor.remove_annotator("custom")
    assert len(processor.annotators) == 0

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


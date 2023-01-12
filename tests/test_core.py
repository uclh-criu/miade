from miade.core import NoteProcessor, DebugMode
from miade.concept import Concept, Category
from miade.dosage import Dose, Frequency
from miade.metaannotations import MetaAnnotations
from miade.utils.metaannotationstypes import *


def test_core(model_directory_path, test_note, test_negated_note, test_duplicated_note):
    processor = NoteProcessor(model_directory_path)
    assert processor.process(test_note) == [
        Concept(id="3", name="Liver failure", category=Category.PROBLEM),
    ]
    assert processor.process(test_negated_note) == []
    assert processor.process(test_duplicated_note) == [
        Concept(id="3", name="Liver failure", category=Category.PROBLEM),
    ]


def test_meta_from_entity(test_medcat_concepts):
    assert Concept.from_entity(test_medcat_concepts["0"]) == Concept(
        id="0", name="problem", category=Category.PROBLEM, start=4, end=11,
        meta_anns=MetaAnnotations(
            presence=Presence.NEGATED,
            relevance=Relevance.HISTORIC,
            laterality=None)
    )
    assert Concept.from_entity(test_medcat_concepts["1"]) == Concept(
        id="0", name="problem", category=Category.PROBLEM, start=4, end=11,
        meta_anns=MetaAnnotations(
            presence=Presence.SUSPECTED,
            relevance=Relevance.IRRELEVANT,
            laterality=Laterality.NO_LATERALITY,
            confidences={Presence: 1, Relevance: 1, Laterality: 1})
    )


def test_debug(model_directory_path, debug_path, test_note):
    processor = NoteProcessor(model_directory_path)
    assert processor.debug(debug_path, mode=DebugMode.CDA) == {
        'Problems': {'statusCode': 'active',
                     'actEffectiveTimeHigh': 'None',
                     'observationEffectiveTimeLow': 20200504,
                     'observationEffectiveTimeHigh': 20210904},
        'Medication': {'consumableCodeSystemName': 'RxNorm',
                       'consumableCodeSystemValue': '2.16.840.1.113883.6.88'},
        'Allergy': {
            'allergySectionCodeName': 'Propensity to adverse reaction',
            'allergySectionCodeValue': 420134006}
    }

    concept_list = processor.debug(debug_path, mode=DebugMode.PRELOADED)
    assert concept_list[1].name == "Paracetamol"
    assert concept_list[1].id == 90332006
    assert concept_list[1].category == Category.MEDICATION
    assert concept_list[1].dosage.dose.value == 2
    assert concept_list[1].dosage.dose.unit == "{tbl}"
    assert concept_list[1].dosage.frequency.value == 0.25
    assert concept_list[1].dosage.duration.low == "20220606"
    assert concept_list[1].dosage.duration.high == "20220620"
    assert concept_list[1].dosage.route.value == "C38288"
    assert concept_list[1].dosage.route.full_name == "Oral"

    assert concept_list[2].name == "Penicillins"
    assert concept_list[2].id == 84874
    assert concept_list[2].category == Category.ALLERGY


def test_dosage_text_splitter(model_directory_path, test_med_concepts, test_med_note):
    processor = NoteProcessor(model_directory_path)
    concepts = processor.add_dosages_to_concepts(test_med_concepts, test_med_note)

    assert concepts[0].dosage.text == "Magnesium hydroxide 75mg daily "
    assert concepts[1].dosage.text == "paracetamol 500mg po 3 times a day as needed."
    assert concepts[2].dosage.text == "aspirin IM q daily x 2 weeks with concurrent "
    assert concepts[3].dosage.text == "DOXYCYCLINE 500mg tablets for two weeks"

    assert concepts[0].dosage.dose == Dose(source='75 mg',
                                           value=75,
                                           unit='mg',
                                           low=None,
                                           high=None)

    assert concepts[0].dosage.frequency == Frequency(source='start 75 mg every day ',
                                                     value=1.0,
                                                     unit='d',
                                                     low=None,
                                                     high=None,
                                                     standardDeviation=None,
                                                     institutionSpecified=False,
                                                     preconditionAsRequired=False)

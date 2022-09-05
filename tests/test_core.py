from miade.core import NoteProcessor, DebugMode
from miade.concept import Concept, Category
from miade.dosage import Dose, Frequency


def test_core(model_directory_path, debug_path, test_note):
    processor = NoteProcessor(model_directory_path, debug_path)
    assert processor.process(test_note) == [
        Concept(id="3", name="Liver failure", category=Category.DIAGNOSIS, start=12, end=25),
        Concept(id="10", name="Paracetamol", category=Category.MEDICATION, start=40, end=51),
    ]

    assert processor.debug(mode=DebugMode.CDA) == {'Problems': {'statusCode': 'active',
                                                                 'actEffectiveTimeHigh': 'None',
                                                                 'observationEffectiveTimeLow': 20200504,
                                                                 'observationEffectiveTimeHigh': 20210904},
                                                   'Medication': {'consumableCodeSystemName': 'RxNorm',
                                                                  'consumableCodeSystemValue': '2.16.840.1.113883.6.88'},
                                                   'Allergy': {'allergySectionCodeName': 'Propensity to adverse reaction',
                                                               'allergySectionCodeValue': 420134006}}

    concept_list = processor.debug(mode=DebugMode.PRELOADED)
    assert concept_list[1].name == "Paracetamol"
    assert concept_list[1].id == 90332006
    assert concept_list[1].category == Category.MEDICATION
    assert concept_list[1].dosage.dose.quantity == 2
    assert concept_list[1].dosage.dose.unit == "{tbl}"
    assert concept_list[1].dosage.frequency.value == 0.25
    assert concept_list[1].dosage.duration.low == 20220606
    assert concept_list[1].dosage.route.code == "C38288"

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
                                           quantity=75,
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

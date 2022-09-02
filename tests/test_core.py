from miade.core import NoteProcessor, DebugMode
from miade.concept import Concept, Category
from miade.dosage import Dose, Frequency


def test_core(model_directory_path, test_note):
    processor = NoteProcessor(model_directory_path)
    assert processor.process(test_note) == [
        Concept(id="3", name="Liver failure", category=Category.DIAGNOSIS, start=12, end=25),
        Concept(id="10", name="Paracetamol", category=Category.MEDICATION, start=40, end=51),
    ]

    print(processor.debug(mode=DebugMode.CDA))
    concept_list = processor.debug(mode=DebugMode.PRELOADED)
    for concept in concept_list:
        print(concept)


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

from miade.core import Concept, Category
from miade.annotators import MedsAllergiesAnnotator, ProblemsAnnotator, Annotator, calculate_word_distance
from miade.dosage import Dose, Frequency
from miade.dosageextractor import DosageExtractor

def test_dosage_text_splitter(test_meds_algy_medcat_model, test_med_concepts, test_med_note):
    annotator = MedsAllergiesAnnotator(test_meds_algy_medcat_model)
    dosage_extractor = DosageExtractor()

    concepts = annotator.add_dosages_to_concepts(dosage_extractor, test_med_concepts, test_med_note)

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


def test_calculate_word_distance():
    from miade.note import Note
    note = Note("the quick broooooown fox jumped over the lazy dog")
    start1, end1 = 10, 20
    start2, end2 = 10, 20
    assert calculate_word_distance(start1, end1, start2, end2, note) == 0

    note = Note("the quick broooooownfoxjumpeed over the lazy dog")
    start1, end1 = 10, 20
    start2, end2 = 20, 30
    assert calculate_word_distance(start1, end1, start2, end2, note) == 0

    start1, end1 = 20, 30
    start2, end2 = 10, 20
    assert calculate_word_distance(start1, end1, start2, end2, note) == 0

    note = Note("thee ickbroooooownfoxesdes jumped over the lazy dog")
    start1, end1 = 5, 25
    start2, end2 = 10, 20
    assert calculate_word_distance(start1, end1, start2, end2, note) == 0

    note = Note("the quick brooooooown fox jum pedoverthe lazy dog")
    start1, end1 = 10, 20
    start2, end2 = 30, 40
    assert calculate_word_distance(start1, end1, start2, end2, note) == 3

    note = Note("the quick brooooooown fox jum pedover thelazydog")
    start1, end1 = 30, 40
    start2, end2 = 41, 55
    assert calculate_word_distance(start1, end1, start2, end2, note) == 1



def test_deduplicate(
    test_problems_medcat_model,
    test_duplicate_concepts_note,
    test_duplicate_concepts_record,
    test_self_duplicate_concepts_note,
):
    annotator = Annotator(test_problems_medcat_model)

    assert annotator.deduplicate(
        concepts=test_duplicate_concepts_note, record_concepts=test_duplicate_concepts_record
    ) == [
        Concept(id="5", name="test2", category=Category.PROBLEM),
        Concept(id="7", name="test2", category=Category.MEDICATION),
    ]
    assert annotator.deduplicate(
        concepts=test_self_duplicate_concepts_note, record_concepts=None) == [
        Concept(id="1", name="test1", category=Category.PROBLEM),
        Concept(id="2", name="test2", category=Category.MEDICATION),
    ]
    assert annotator.deduplicate(
        concepts=test_duplicate_concepts_note, record_concepts=None
    ) == [
        Concept(id="1", name="test1", category=Category.PROBLEM),
        Concept(id="2", name="test2", category=Category.PROBLEM),
        Concept(id="3", name="test2", category=Category.PROBLEM),
        Concept(id="4", name="test2", category=Category.PROBLEM),
        Concept(id="5", name="test2", category=Category.PROBLEM),
        Concept(id="6", name="test2", category=Category.MEDICATION),
        Concept(id="7", name="test2", category=Category.MEDICATION),
    ]
    assert annotator.deduplicate(
        concepts=test_duplicate_concepts_note, record_concepts=[]
    ) == [
        Concept(id="1", name="test1", category=Category.PROBLEM),
        Concept(id="2", name="test2", category=Category.PROBLEM),
        Concept(id="3", name="test2", category=Category.PROBLEM),
        Concept(id="4", name="test2", category=Category.PROBLEM),
        Concept(id="5", name="test2", category=Category.PROBLEM),
        Concept(id="6", name="test2", category=Category.MEDICATION),
        Concept(id="7", name="test2", category=Category.MEDICATION),
    ]
    assert (
        annotator.deduplicate(
            concepts=[], record_concepts=test_duplicate_concepts_record
        )
        == []
    )

def test_meta_annotations(test_problems_medcat_model, test_meta_annotations_concepts):
    annotator = ProblemsAnnotator(test_problems_medcat_model)

    assert annotator.postprocess(test_meta_annotations_concepts) == [
        Concept(id="274826007", name="Nystagmus (negated)", category=Category.PROBLEM),  # negex true, meta ignored
        Concept(
            id="302064001", name="Lymphangitis (negated)", category=Category.PROBLEM
        ),  # negex true, meta ignored
        Concept(
            id="431956005", name="Arthritis (suspected)", category=Category.PROBLEM
        ),  # negex false, meta processed
        Concept(
            id="413241009", name="Gastritis (suspected)", category=Category.PROBLEM
        ),
        Concept(
            id="115451000119100",
            name="Endophthalmitis (historic)",
            category=Category.PROBLEM,
        ),  # negex false, meta processed
    ]
    # test just using negex for negation
    test_meta_annotations_concepts[0].negex = True
    test_meta_annotations_concepts[0].meta = None
    test_meta_annotations_concepts[1].negex = False
    test_meta_annotations_concepts[2].negex = True
    test_meta_annotations_concepts[3].negex = True
    test_meta_annotations_concepts[6].negex = True

    assert annotator.postprocess(test_meta_annotations_concepts) == [
        Concept(id="274826007", name="Nystagmus (negated)", category=Category.PROBLEM),  # negex true, meta empty
        Concept(
            id="1415005", name="Lymphangitis", category=Category.PROBLEM
        ),  # negex false, meta processed but ignore negation
        Concept(
            id="413241009", name="Gastritis (suspected)", category=Category.PROBLEM
        ),  # negex false, meta processed
    ]


def test_problems_filtering_list(test_problems_medcat_model, test_filtering_list_concepts):
    annotator = ProblemsAnnotator(test_problems_medcat_model)
    assert annotator.postprocess(test_filtering_list_concepts) == [
        Concept(id="123", name="real concept", category=Category.PROBLEM),
    ]

def test_allergy_annotator(test_meds_algy_medcat_model, test_meds_allergy_concepts, test_meds_allergy_note):
    annotator = MedsAllergiesAnnotator(test_meds_algy_medcat_model)

    assert annotator.postprocess(test_meds_allergy_concepts, test_meds_allergy_note) == [
        Concept(id="123", name="Eggs", category=Category.ALLERGY, linked_concepts=[Concept(
            id="789", name="Rash", category=Category.REACTION)]),
        Concept(id="456", name="Penicillin", category=Category.ALLERGY, linked_concepts=[Concept(
            id="1234", name="Nausea", category=Category.REACTION)]),
        Concept(id="12344", name="Paracetamol", category=Category.MEDICATION),
    ]

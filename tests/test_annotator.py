from miade.core import Concept, Category
from miade.annotators import MedsAllergiesAnnotator, ProblemsAnnotator, Annotator, calculate_word_distance
from miade.dosage import Dose, Frequency, Dosage, Route
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
        Concept(id="7", name="test2", category=Category.MEDICATION),
        Concept(id="5", name="test2", category=Category.PROBLEM),
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
        Concept(id="7", name="test2", category=Category.MEDICATION),
        Concept(id="5", name="test2", category=Category.PROBLEM),
        Concept(id="6", name="test2", category=Category.MEDICATION),
    ]
    assert annotator.deduplicate(
        concepts=test_duplicate_concepts_note, record_concepts=[]
    ) == [
        Concept(id="1", name="test1", category=Category.PROBLEM),
        Concept(id="2", name="test2", category=Category.PROBLEM),
        Concept(id="3", name="test2", category=Category.PROBLEM),
        Concept(id="4", name="test2", category=Category.PROBLEM),
        Concept(id="7", name="test2", category=Category.MEDICATION),
        Concept(id="5", name="test2", category=Category.PROBLEM),
        Concept(id="6", name="test2", category=Category.MEDICATION),
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
        Concept(
            id="0000",
            name="historic concept",
            category=Category.PROBLEM,
        ),
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
        Concept(id="0000", name="historic concept", category=Category.PROBLEM
        ),  # historic with no conversion
    ]


def test_problems_filtering_list(test_problems_medcat_model, test_filtering_list_concepts):
    annotator = ProblemsAnnotator(test_problems_medcat_model)
    assert annotator.postprocess(test_filtering_list_concepts) == [
        Concept(id="123", name="real concept", category=Category.PROBLEM),
    ]

def test_allergy_annotator(test_meds_algy_medcat_model, test_substance_concepts_with_meta_anns, test_meds_allergy_note):
    annotator = MedsAllergiesAnnotator(test_meds_algy_medcat_model)
    concepts = annotator.postprocess(test_substance_concepts_with_meta_anns, test_meds_allergy_note)

    # print([concept.__str__() for concept in concepts])
    assert concepts == [
        Concept(id="102263004", name="Eggs", category=Category.ALLERGY),
        Concept(id="767270007", name="Penicillin", category=Category.ALLERGY),
        Concept(id="7336002", name="Paracetamol", category=Category.MEDICATION),
    ]
    assert concepts[0].linked_concepts == [
        Concept(id="235719002", name="Food Intolerance", category=Category.ALLERGY_TYPE),
        Concept(id="L", name="Low", category=Category.SEVERITY),
        Concept(id="419076005", name="Rash", category=Category.REACTION),
    ]
    assert concepts[1].linked_concepts == [
        Concept(id="416098002", name="Drug Allergy", category=Category.ALLERGY_TYPE),
        Concept(id="M", name="Moderate", category=Category.SEVERITY),
        Concept(id="419076005", name="Nausea", category=Category.REACTION),
    ]
    assert concepts[2].linked_concepts == []

def test_vtm_med_conversions(test_meds_algy_medcat_model, test_vtm_concepts):
    annotator = MedsAllergiesAnnotator(test_meds_algy_medcat_model)
    concepts = annotator.convert_VTM_to_VMP_or_text(test_vtm_concepts)

    # print([concept.__str__() for concept in concepts])
    assert concepts == [
        Concept(id=None, name="SPIRAMYCIN ORAL", category=Category.MEDICATION),
        Concept(id="376689003", name="Paracetamol 50mg tablets", category=Category.MEDICATION),
        Concept(id=None, name="ASPIRIN ORAL", category=Category.MEDICATION),
        Concept(id=None, name="FOLIC ACID ORAL", category=Category.MEDICATION),
        Concept(id="7721411000001109", name="Selenium 50microgram tablets", category=Category.MEDICATION),
        Concept(id=None, name="SELENIUM ORAL", category=Category.MEDICATION),
    ]
    assert concepts[0].dosage == Dosage(
        dose=Dose(value=10, unit="mg"),
        frequency=None,
        duration=None,
        route=None,
    )
    assert concepts[1].dosage == Dosage(
        dose=Dose(value=1, unit="{tbl}"),
        frequency=None,
        duration=None,
        route=None,
    )
    assert concepts[2].dosage == Dosage(
        dose=None,
        frequency=None,
        duration=None,
        route=Route(value="C38288", full_name="Oral"),
    )
    assert concepts[3].dosage is None
    assert concepts[4].dosage == Dosage(
        dose=Dose(value=1, unit="{tbl}"),
        frequency=None,
        duration=None,
        route=None,
    )
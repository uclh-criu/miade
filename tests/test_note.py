from miade.annotators import ProblemsAnnotator
from miade.concept import Category, Concept
from miade.note import ListItem, Note, NumberedList
from miade.paragraph import Paragraph, ParagraphType
from miade.utils.metaannotationstypes import (
    Presence,
    Relevance,
    SubstanceCategory,
)


def test_note_cleaning_and_paragraphing_naive(test_problems_annotator, test_clean_and_paragraphing_note):
    test_problems_annotator.preprocess(test_clean_and_paragraphing_note, refine=False)

    assert test_clean_and_paragraphing_note.paragraphs == [
        Paragraph(heading="", body="", type=ParagraphType.prose, start=0, end=181),
        Paragraph(heading="", body="", type=ParagraphType.prose, start=183, end=261),
        Paragraph(heading="", body="", type=ParagraphType.prose, start=263, end=313),
        Paragraph(heading="", body="", type=ParagraphType.pmh, start=315, end=340),
        Paragraph(heading="", body="", type=ParagraphType.med, start=342, end=405),
        Paragraph(heading="", body="", type=ParagraphType.allergy, start=407, end=444),
        Paragraph(heading="", body="", type=ParagraphType.prob, start=446, end=476),
        Paragraph(heading="", body="", type=ParagraphType.plan, start=478, end=504),
        Paragraph(heading="", body="", type=ParagraphType.imp, start=506, end=522),
    ]


def test_note_cleaning_and_paragraphing_refined(test_problems_annotator, test_clean_and_paragraphing_note):
    test_problems_annotator.preprocess(test_clean_and_paragraphing_note, refine=True)

    assert test_clean_and_paragraphing_note.paragraphs == [
        Paragraph(heading="", body="", type=ParagraphType.prose, start=0, end=313),
        Paragraph(heading="", body="", type=ParagraphType.pmh, start=315, end=340),
        Paragraph(heading="", body="", type=ParagraphType.med, start=342, end=405),
        Paragraph(heading="", body="", type=ParagraphType.allergy, start=407, end=444),
        Paragraph(heading="", body="", type=ParagraphType.prob, start=446, end=476),
        Paragraph(heading="", body="", type=ParagraphType.plan, start=478, end=504),
        Paragraph(heading="", body="", type=ParagraphType.imp, start=506, end=522),
    ]


def test_numbered_list_note(test_problems_annotator, test_numbered_list_note):
    test_concepts = [
        Concept(id="correct", name="list item", start=10, end=17),
        Concept(id="incorrect", name="list item not relevant", start=20, end=60),
        Concept(id="incorrect", name="list item not relevant", start=200, end=210),
        Concept(id="correct", name="prose that is not in lists", start=130, end=140),
        Concept(id="correct", name="other section", start=280, end=300),
        Concept(id="correct", name="other section", start=300, end=378),
    ]
    test_problems_annotator.preprocess(test_numbered_list_note, refine=True)
    assert test_problems_annotator.filter_concepts_in_numbered_list(test_concepts, test_numbered_list_note) == [
        Concept(name="list item", id="correct"),
        Concept(name="prose that is not in lists", id="correct"),
        Concept(name="other section", id="correct"),
        Concept(name="other section", id="correct"),
    ]


def test_numbered_list_with_blank_line(test_problems_annotator, test_note_numbered_list_with_blank_line):
    test_concepts = [
        Concept(id="correct", name="first item", start=13, end=26),
        Concept(id="incorrect", name="additional concept in first item", start=31, end=54),
        Concept(id="correct", name="second item", start=59, end=82),
        Concept(id="incorrect", name="additional concept in second item", start=85, end=96),
        Concept(id="correct", name="third item", start=100, end=115),
    ]
    test_problems_annotator.preprocess(test_note_numbered_list_with_blank_line, refine=True)
    assert test_problems_annotator.filter_concepts_in_numbered_list(
        test_concepts, test_note_numbered_list_with_blank_line
    ) == [
        Concept(name="first item", id="correct"),
        Concept(name="second item", id="correct"),
        Concept(name="third item", id="correct"),
    ]


def test_prob_paragraph_note(
    test_problems_annotator, test_clean_and_paragraphing_note, test_paragraph_chunking_prob_concepts
):
    test_problems_annotator.preprocess(test_clean_and_paragraphing_note)

    concepts = test_problems_annotator.process_paragraphs(
        test_clean_and_paragraphing_note, test_paragraph_chunking_prob_concepts
    )
    # prose
    assert concepts[0].presence == Presence.CONFIRMED
    assert concepts[0].relevance == Relevance.PRESENT

    # pmh
    assert concepts[1].presence == Presence.CONFIRMED
    assert concepts[1].relevance == Relevance.HISTORIC

    assert concepts[2].presence == Presence.CONFIRMED
    assert concepts[2].relevance == Relevance.IRRELEVANT

    # allergies
    assert concepts[3].presence == Presence.CONFIRMED
    assert concepts[3].relevance == Relevance.IRRELEVANT

    # probs
    assert concepts[4].presence == Presence.CONFIRMED
    assert concepts[4].relevance == Relevance.PRESENT

    # plan
    assert concepts[5].presence == Presence.CONFIRMED
    assert concepts[5].relevance == Relevance.IRRELEVANT


def test_med_paragraph_note(
    test_meds_algy_annotator, test_clean_and_paragraphing_note, test_paragraph_chunking_med_concepts
):
    test_meds_algy_annotator.preprocess(test_clean_and_paragraphing_note)

    concepts = test_meds_algy_annotator.process_paragraphs(
        test_clean_and_paragraphing_note, test_paragraph_chunking_med_concepts
    )
    # pmh
    assert concepts[0].substance_category == SubstanceCategory.IRRELEVANT
    # meds
    assert concepts[1].substance_category == SubstanceCategory.TAKING
    # allergies
    assert concepts[2].substance_category == SubstanceCategory.ADVERSE_REACTION
    # probs
    assert concepts[3].substance_category == SubstanceCategory.IRRELEVANT
    # plan
    assert concepts[4].substance_category == SubstanceCategory.IRRELEVANT
    # imp
    assert concepts[5].substance_category == SubstanceCategory.IRRELEVANT


def test_problem_list_limit(
    test_problem_list_limit_note, test_problem_list_limit_concepts, test_problems_medcat_model, test_config
):
    test_config.structured_list_limit = 0

    annotator = ProblemsAnnotator(test_problems_medcat_model, test_config)
    annotator.preprocess(test_problem_list_limit_note)

    assert annotator.process_paragraphs(test_problem_list_limit_note, test_problem_list_limit_concepts) == [
        Concept(
            id="1",
            name="Fever",
            category=Category.PROBLEM,
            start=138,
            end=143,
            presence=Presence.CONFIRMED,
            relevance=Relevance.PRESENT,
        ),
        Concept(
            id="2",
            name="Diabetes",
            category=Category.PROBLEM,
            start=144,
            end=152,
            presence=Presence.CONFIRMED,
            relevance=Relevance.PRESENT,
        ),
        Concept(
            id="4",
            name="Arthritis",
            category=Category.PROBLEM,
            start=189,
            end=196,
            presence=Presence.CONFIRMED,
            relevance=Relevance.HISTORIC,
        ),
    ]


def test_get_numbered_lists_empty_text():
    note = Note("")
    note.get_numbered_lists()
    assert note.numbered_lists == []


def test_get_numbered_lists_no_lists():
    text = "This is a sample note without any numbered lists."
    note = Note(text)
    note.get_numbered_lists()
    assert note.numbered_lists == []


def test_get_numbered_lists_single_list():
    text = "\n1. First item\n2. Second item\n3. Third item"
    note = Note(text)
    note.get_numbered_lists()
    assert len(note.numbered_lists) == 1
    assert isinstance(note.numbered_lists[0], NumberedList)
    assert len(note.numbered_lists[0].items) == 3
    assert isinstance(note.numbered_lists[0].items[0], ListItem)
    assert note.numbered_lists[0].items[0].content == "1. First item"
    assert note.numbered_lists[0].items[0].start == 1
    assert note.numbered_lists[0].items[0].end == 14
    assert isinstance(note.numbered_lists[0].items[1], ListItem)
    assert note.numbered_lists[0].items[1].content == "2. Second item"
    assert note.numbered_lists[0].items[1].start == 15
    assert note.numbered_lists[0].items[1].end == 29
    assert isinstance(note.numbered_lists[0].items[2], ListItem)
    assert note.numbered_lists[0].items[2].content == "3. Third item"
    assert note.numbered_lists[0].items[2].start == 30
    assert note.numbered_lists[0].items[2].end == 43


def test_merge_prose_sections():
    note = Note("This is the first paragraph.\n\nThis is the second paragraph.\n\nThis is the third paragraph.")
    note.paragraphs = [
        Paragraph(heading="", body="This is the first paragraph.", type=ParagraphType.prose, start=0, end=27),
        Paragraph(heading="", body="This is the second paragraph.", type=ParagraphType.prose, start=29, end=58),
        Paragraph(heading="", body="This is the third paragraph.", type=ParagraphType.prose, start=60, end=89),
    ]
    note.merge_prose_sections()
    assert len(note.paragraphs) == 1
    assert (
        note.paragraphs[0].heading
        == "This is the first paragraph.\n\nThis is the second paragraph.\n\nThis is the third paragraph."
    )
    assert note.paragraphs[0].type == ParagraphType.prose


def test_merge_empty_non_prose_with_next_prose():
    note = Note("This is the first paragraph.\n\nThis is the second paragraph.\n\nThis is the third paragraph.")
    note.paragraphs = [
        Paragraph(heading="Heading 1", body="", type=ParagraphType.prob, start=0, end=14),
        Paragraph(heading="This is the first paragraph.", body="", type=ParagraphType.prose, start=16, end=43),
        Paragraph(heading="Heading 2", body="", type=ParagraphType.pmh, start=45, end=59),
        Paragraph(heading="This is the second paragraph.", body="", type=ParagraphType.prose, start=61, end=90),
        Paragraph(heading="Heading 3", body="", type=ParagraphType.med, start=92, end=106),
        Paragraph(heading="This is the third paragraph.", body="", type=ParagraphType.prose, start=108, end=137),
    ]
    note.merge_empty_non_prose_with_next_prose()
    assert len(note.paragraphs) == 3
    assert note.paragraphs[0].heading == "Heading 1"
    assert note.paragraphs[0].body == "This is the first paragraph."
    assert note.paragraphs[0].type == ParagraphType.prob
    assert note.paragraphs[1].heading == "Heading 2"
    assert note.paragraphs[1].body == "This is the second paragraph."
    assert note.paragraphs[1].type == ParagraphType.pmh
    assert note.paragraphs[2].heading == "Heading 3"
    assert note.paragraphs[2].body == "This is the third paragraph."
    assert note.paragraphs[2].type == ParagraphType.med

from miade.annotators import Annotator, MedsAllergiesAnnotator, ProblemsAnnotator
from miade.core import NoteProcessor
from miade.concept import Concept, Category
from miade.paragraph import Paragraph, ParagraphType
from miade.metaannotations import MetaAnnotations

from miade.utils.metaannotationstypes import (
    Presence,
    Relevance,
    SubstanceCategory,
)


def test_note_cleaning_and_paragraphing(test_problems_medcat_model, test_clean_and_paragraphing_note):
    annotator = ProblemsAnnotator(test_problems_medcat_model)
    annotator.preprocess(test_clean_and_paragraphing_note)

    assert test_clean_and_paragraphing_note.paragraphs == [
        Paragraph(heading="", body="", type=ParagraphType.prose, start=0, end=182),
        Paragraph(heading="", body="", type=ParagraphType.prose, start=184, end=262),
        Paragraph(heading="", body="", type=ParagraphType.prose, start=264, end=314),
        Paragraph(heading="", body="", type=ParagraphType.pmh, start=316, end=341),
        Paragraph(heading="", body="", type=ParagraphType.med, start=343, end=406),
        Paragraph(heading="", body="", type=ParagraphType.allergy, start=408, end=445),
        Paragraph(heading="", body="", type=ParagraphType.prob, start=447, end=477),
        Paragraph(heading="", body="", type=ParagraphType.plan, start=479, end=505),
        Paragraph(heading="", body="", type=ParagraphType.imp, start=507, end=523),
    ]


def test_prob_paragraph_note(
    test_problems_medcat_model, test_clean_and_paragraphing_note, test_paragraph_chunking_prob_concepts
):
    annotator = ProblemsAnnotator(test_problems_medcat_model)
    annotator.preprocess(test_clean_and_paragraphing_note)

    concepts = annotator.process_paragraphs(test_clean_and_paragraphing_note, test_paragraph_chunking_prob_concepts)
    # prose
    assert concepts[0].meta == [
        MetaAnnotations(name="presence", value=Presence.CONFIRMED),
        MetaAnnotations(name="relevance", value=Relevance.PRESENT),
    ]
    # pmh
    assert concepts[1].meta == [
        MetaAnnotations(name="presence", value=Presence.CONFIRMED),
        MetaAnnotations(name="relevance", value=Relevance.HISTORIC),
    ]
    assert concepts[2].meta == [
        MetaAnnotations(name="presence", value=Presence.CONFIRMED),
        MetaAnnotations(name="relevance", value=Relevance.IRRELEVANT),
    ]
    # allergies
    assert concepts[3].meta == [
        MetaAnnotations(name="presence", value=Presence.CONFIRMED),
        MetaAnnotations(name="relevance", value=Relevance.IRRELEVANT),
    ]
    # probs
    assert concepts[4].meta == [
        MetaAnnotations(name="presence", value=Presence.CONFIRMED),
        MetaAnnotations(name="relevance", value=Relevance.PRESENT),
    ]
    # plan
    assert concepts[5].meta == [
        MetaAnnotations(name="presence", value=Presence.CONFIRMED),
        MetaAnnotations(name="relevance", value=Relevance.IRRELEVANT),
    ]


def test_prob_paragraph_note(
    test_meds_algy_medcat_model, test_clean_and_paragraphing_note, test_paragraph_chunking_med_concepts
):
    annotator = MedsAllergiesAnnotator(test_meds_algy_medcat_model)
    annotator.preprocess(test_clean_and_paragraphing_note)

    concepts = annotator.process_paragraphs(test_clean_and_paragraphing_note, test_paragraph_chunking_med_concepts)
    # pmh
    assert concepts[0].meta == [
        MetaAnnotations(name="substance_category", value=SubstanceCategory.IRRELEVANT),
    ]
    # meds
    assert concepts[1].meta == [
        MetaAnnotations(name="substance_category", value=SubstanceCategory.TAKING),
    ]
    # allergies
    assert concepts[2].meta == [
        MetaAnnotations(name="substance_category", value=SubstanceCategory.ADVERSE_REACTION),
    ]
    # probs
    assert concepts[3].meta == [
        MetaAnnotations(name="substance_category", value=SubstanceCategory.IRRELEVANT),
    ]
    # plan
    assert concepts[4].meta == [
        MetaAnnotations(name="substance_category", value=SubstanceCategory.IRRELEVANT),
    ]
    # imp
    assert concepts[5].meta == [
        MetaAnnotations(name="substance_category", value=SubstanceCategory.IRRELEVANT),
    ]


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
            id="4",
            name="Arthritis",
            category=Category.PROBLEM,
            negex=False,
            start=189,
            end=196,
            meta_anns=[
                MetaAnnotations(name="presence", value=Presence.CONFIRMED),
                MetaAnnotations(name="relevance", value=Relevance.HISTORIC),
            ],
        ),
    ]

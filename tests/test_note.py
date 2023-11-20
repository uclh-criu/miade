from miade.core import NoteProcessor
from miade.paragraph import Paragraph, ParagraphType
from miade.metaannotations import MetaAnnotations

from miade.utils.metaannotationstypes import *


def test_note(model_directory_path, test_clean_and_paragraphing_note, test_paragraph_chunking_concepts):
    test_clean_and_paragraphing_note.clean_text()
    test_clean_and_paragraphing_note.get_paragraphs()

    # for paragraph in test_clean_and_paragraphing_note.paragraphs:
    #     print(paragraph)

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

    processor = NoteProcessor(model_directory_path)
    processor.add_annotator("meds/allergies")

    concepts = processor.annotators[0].process_paragraphs(
        test_clean_and_paragraphing_note, test_paragraph_chunking_concepts
    )
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
        MetaAnnotations(name="substance_category", value=SubstanceCategory.IRRELEVANT),
    ]
    # meds
    assert concepts[3].meta == [
        MetaAnnotations(name="substance_category", value=SubstanceCategory.TAKING),
    ]
    assert concepts[4].meta == [
        MetaAnnotations(name="presence", value=Presence.CONFIRMED),
        MetaAnnotations(name="relevance", value=Relevance.IRRELEVANT),
    ]
    # allergies
    assert concepts[5].meta == [
        MetaAnnotations(name="presence", value=Presence.CONFIRMED),
        MetaAnnotations(name="relevance", value=Relevance.IRRELEVANT),
    ]
    assert concepts[6].meta == [
        MetaAnnotations(name="substance_category", value=SubstanceCategory.ADVERSE_REACTION),
    ]
    # probs
    assert concepts[7].meta == [
        MetaAnnotations(name="presence", value=Presence.CONFIRMED),
        MetaAnnotations(name="relevance", value=Relevance.PRESENT),
    ]
    assert concepts[8].meta == [
        MetaAnnotations(name="substance_category", value=SubstanceCategory.IRRELEVANT),
    ]
    # plan
    assert concepts[9].meta == [
        MetaAnnotations(name="presence", value=Presence.CONFIRMED),
        MetaAnnotations(name="relevance", value=Relevance.IRRELEVANT),
    ]
    assert concepts[10].meta == [
        MetaAnnotations(name="substance_category", value=SubstanceCategory.IRRELEVANT),
    ]
    # imp
    assert concepts[11].meta == [
        MetaAnnotations(name="substance_category", value=SubstanceCategory.IRRELEVANT),
    ]
    # for concept in concepts:
    #     print(concept)


def test_long_problem_list():
    # TODO
    pass

from miade.core import NoteProcessor
from miade.metaannotations import MetaAnnotations
from miade.utils.metaannotationstypes import *

def test_note(model_directory_path, test_clean_and_paragraphing_note, test_paragraph_chunking_concepts):

    processor = NoteProcessor(model_directory_path)

    processor.add_annotator("problems", use_negex=True)
    processor.add_annotator("meds/allergies")

    # concepts = processor.process(test_clean_and_paragraphing_note)
    for paragraph in test_clean_and_paragraphing_note.paragraphs:
        # TODO: write test case
        print(paragraph)

    concepts = processor.annotators[0].process_paragraphs(
        test_clean_and_paragraphing_note,
        test_paragraph_chunking_concepts
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

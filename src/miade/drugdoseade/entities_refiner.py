import logging

from spacy.tokens import Doc
from spacy.language import Language
from spacy.tokens import Span


log = logging.getLogger(__name__)


@Language.component("entities_refiner")
def EntitiesRefiner(doc) -> Doc:
    """
    Refines NER results by merging consecutive labels with the same tag,
    removing strength labels, and merging drug labels with dosage labels.

    Args:
        doc (spacy.tokens.Doc): The input document containing named entities.

    Returns:
        spacy.tokens.Doc: The refined document with updated named entities.
    """

    new_ents = []
    for ind, ent in enumerate(doc.ents):
        # combine consecutive labels with the same tag
        if (ent.label_ == "DURATION" or ent.label_ == "FREQUENCY" or ent.label_ == "DOSAGE") and ind != 0:
            prev_ent = doc.ents[ind - 1]
            if prev_ent.label_ == ent.label_:
                new_ent = Span(doc, prev_ent.start, ent.end, label=ent.label)
                new_ents.pop()
                new_ents.append(new_ent)
                log.debug(f"Merged {ent.label_} labels")
            else:
                new_ents.append(ent)
        # remove strength labels - should be in concept name, often should be part of dosage
        elif ent.label_ == "STRENGTH":
            new_ent = Span(doc, ent.start, ent.end, label="DOSAGE")
            new_ents.append(new_ent)
            log.debug(f"Removed {ent.label_} label")
        # the dose string should only contain dosage so if drug is detected after dosage, most likely mislabelled
        elif ent.label_ == "DRUG":
            prev_ent = doc.ents[ind - 1]
            if prev_ent.label_ == "DOSAGE":
                new_ent = Span(doc, ent.start, ent.end, label="FORM")
                new_ents.append(new_ent)
                log.debug(f"Merged {ent.label_} with {prev_ent.label_} label")
        else:
            new_ents.append(ent)

    doc.ents = new_ents

    return doc

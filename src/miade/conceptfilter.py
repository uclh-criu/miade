import io
import logging
import pkgutil
from copy import deepcopy

from typing import List, Optional

import pandas as pd

from .concept import Concept, Category
from .utils.metaannotationstypes import *

log = logging.getLogger(__name__)


def is_duplicate(concept: Concept, record_concepts: Optional[List[Concept]]) -> bool:
    """
    De-duplicate concepts based on code, can take into account more complex info in future?
    :param concept:
    :param record_concepts:
    :return: Bool whether concept is duplicated or not
    """
    if concept.id in [record_concept.id for record_concept in record_concepts]:
        log.debug(
            f"Filtered concept {(concept.name, concept.id)}: concept exists in record"
        )
        return True

    return False


class ConceptFilter(object):
    """
    Post-processing for extracted concepts: deduplication, meta-annotations
    """

    def __init__(self):
        negated_data = pkgutil.get_data(__name__, "./data/negated.csv")
        self.negated_lookup = pd.read_csv(
            io.BytesIO(negated_data), index_col=0, squeeze=True
        ).T.to_dict()
        historic_data = pkgutil.get_data(__name__, "./data/historic.csv")
        self.historic_lookup = pd.read_csv(
            io.BytesIO(historic_data), index_col=0, squeeze=True
        ).T.to_dict()
        suspected_data = pkgutil.get_data(__name__, "./data/suspected.csv")
        self.suspected_lookup = pd.read_csv(
            io.BytesIO(suspected_data), index_col=0, squeeze=True
        ).T.to_dict()

    def filter(
        self,
        extracted_concepts: List[Concept],
        record_concepts: Optional[List[Concept]],
    ) -> List[Concept]:
        """filters/conversions based on deduplication and meta-annotations"""

        # deepcopy so we still have reference to original list of concepts
        all_concepts = deepcopy(extracted_concepts)
        filtered_concepts = []
        for concept in all_concepts:
            # meta-annotations
            if concept.category == Category.PROBLEM:
                concept = self.handle_problem_meta(concept)
            # ignore concepts filtered by meta-annotations
            if concept is None:
                continue
            # deduplication
            if record_concepts is not None:
                if is_duplicate(concept=concept, record_concepts=record_concepts):
                    continue
            filtered_concepts.append(concept)

        # should filter for duplicates within list after meta-annotations handling
        # e.g. the same concept could be negated and historic; this would be filtered if we checked for
        # duplicates in the beginning
        return sorted(set(filtered_concepts))

    def handle_problem_meta(self, concept: [Concept]) -> Optional[Concept]:
        # add, convert, or ignore concepts
        # ignore laterality for now
        convert = False
        tag = ""
        meta_anns = concept.meta
        # only get meta results if negex is positive
        if concept.negex:
            convert = self.negated_lookup.get(int(concept.id), False)
            tag = " (negated)"

            if meta_anns is not None:
                if meta_anns.presence is Presence.SUSPECTED:
                    convert = self.suspected_lookup.get(int(concept.id), False)
                    tag = " (suspected)"
                if meta_anns.relevance is Relevance.HISTORIC:
                    convert = self.historic_lookup.get(int(concept.id), False)
                    tag = " (historic)"

        if convert:
            log.debug(
                f"Converted concept {(concept.name, concept.id)} to {(concept.name + tag, str(convert))}"
            )
            concept.id = str(convert)
            concept.name = concept.name + tag
        else:
            if concept.negex:
                log.debug(
                    f"Filtered concept {(concept.name, concept.id)}: negation with no conversion match"
                )
                return None
            if concept.negex and meta_anns is not None:
                if (
                    meta_anns.presence is Presence.SUSPECTED
                    or meta_anns.relevance is Relevance.IRRELEVANT
                ):
                    log.debug(
                        f"Filtered concept {(concept.name, concept.id)}: either suspected with no conversion "
                        f"match or irrelevant"
                    )
                    return None

        return concept

    def __call__(
        self,
        extracted_concepts: List[Concept],
        record_concepts: Optional[List[Concept]] = None,
    ):
        return self.filter(extracted_concepts, record_concepts)

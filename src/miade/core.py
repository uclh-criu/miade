import re
import yaml
import logging

from negspacy.negation import Negex
from pathlib import Path
from typing import List, Dict, Optional
from enum import Enum

from .concept import Concept, Category
from .dosage import Dosage, Dose, Frequency, Duration, Route
from .note import Note

from .conceptfilter import ConceptFilter
from .dosageextractor import DosageExtractor
from .utils.metaannotationstypes import SubstanceCategory
from .utils.miade_cat import MiADE_CAT

log = logging.getLogger(__name__)


def get_dosage_string(med: Concept, next_med: Optional[Concept], text: str) -> str:
    """
    Finds chunks of text that contain single dosage instructions to input into DosageProcessor
    :param med: (Concept) medications concept
    :param next_med: (Concept) next consecutive medication concept if there is one
    :param text: (str) whole text
    :return: (str) dosage text
    """
    # spit into sentences by newline and period
    sents = (
        re.findall(r"[^\s][^\n]+", text[med.start : next_med.start])
        if next_med is not None
        else re.findall(r"[^\s][^\n]+", text[med.start :])
    )

    concept_name = text[med.start : med.end]
    next_concept_name = text[next_med.start : next_med.end] if next_med else None

    for sent in sents:
        if next_med is not None:
            if concept_name in sent and next_concept_name not in sent:
                return sent
            elif concept_name in sent and next_concept_name in sent:
                return text[med.start : next_med.start]
        else:
            if concept_name in sent:
                ind = sent.find(concept_name)
                return sent[ind:]


class NoteProcessor:
    """docstring for NoteProcessor."""

    def __init__(
        self,
        model_directory: Path,
        problems_model_id: Optional[str] = None,
        meds_allergies_model_id: Optional[str] = None,
        use_negex: bool = True,
        log_level: int = logging.INFO,
    ):
        logging.getLogger("miade").setLevel(log_level)
        meta_cat_config_dict = {"general": {"device": "cpu"}}
        self.problems_model_id = problems_model_id
        self.meds_allergies_model_id = meds_allergies_model_id
        self.annotators = [
            MiADE_CAT.load_model_pack(
                str(model_pack_filepath), meta_cat_config_dict=meta_cat_config_dict
            )
            for model_pack_filepath in model_directory.glob("*.zip")
        ]
        self.dosage_extractor = DosageExtractor()
        self.concept_filter = ConceptFilter(use_negex=use_negex)

        if use_negex:
            log.info("Using Negex as priority for meta context detection")
            self._add_negex_pipeline()

        if problems_model_id:
            log.info(f"Configured problems model {self.problems_model_id}")
        if meds_allergies_model_id:
            log.info(f"Configured meds/allergies model {self.meds_allergies_model_id}")

    def process(
        self, note: Note, record_concepts: Optional[List[Concept]] = None
    ) -> List[Concept]:

        concepts: List[Concept] = []
        for annotator in self.annotators:
            annotator_id = annotator.config.version["id"]
            for entity in annotator.get_entities(note)["entities"].values():
                try:
                    concept = Concept.from_entity(entity)
                    if annotator_id == self.problems_model_id:
                        concept.category = Category.PROBLEM
                    elif annotator_id == self.meds_allergies_model_id:
                        if concept.meta is not None:
                            if (
                                concept.meta.substance_category
                                == SubstanceCategory.ADVERSE_REACTION
                            ):
                                concept.category = Category.ALLERGY
                            elif (
                                concept.meta.substance_category
                                == SubstanceCategory.TAKING
                            ):
                                concept.category = Category.MEDICATION
                        else:
                            # TODO: TEMPORARY BEFORE POST-PROCESSING SORTED OUT
                            concept.category = Category.MEDICATION
                        assert (
                            concept.category == Category.MEDICATION or Category.ALLERGY
                        )
                    concepts.append(concept)
                except ValueError as e:
                    log.warning(f"Concept skipped: {e}")

        log.debug(f"Detected concepts: {[(concept.id, concept.name, concept.category.name) for concept in concepts]}")
        # dosage extraction
        concepts = self.add_dosages_to_concepts(concepts, note)
        # insert default VMP selection algorithm here
        # post-processing
        concepts = self.concept_filter(concepts, record_concepts)

        return concepts

    def _add_negex_pipeline(self) -> None:
        for annotator in self.annotators:
            annotator.pipe.spacy_nlp.add_pipe("sentencizer")
            annotator.pipe.spacy_nlp.enable_pipe("sentencizer")
            annotator.pipe.spacy_nlp.add_pipe("negex")

    def add_dosages_to_concepts(
        self, concepts: List[Concept], note: Note
    ) -> List[Concept]:
        """
        Gets dosages for medication concepts
        :param concepts: (List) list of concepts extracted
        :param note: (Note) input note
        :return: (List) list of concepts with dosages for medication concepts
        """

        for ind, concept in enumerate(concepts):
            next_med_concept = (
                concepts[ind + 1]
                if len(concepts) > ind + 1
                and concepts[ind + 1].category == Category.MEDICATION
                else None
            )
            if concept.category == Category.MEDICATION:
                dosage_string = get_dosage_string(concept, next_med_concept, note.text)
                if len(dosage_string.split()) > 2:
                    concept.dosage = self.dosage_extractor(dosage_string)
                    # print(concept.dosage)

        return concepts


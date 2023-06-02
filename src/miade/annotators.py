import io
import logging
import pkgutil
import re
import pandas as pd

from typing import Dict, List, Optional
from copy import deepcopy

from .concept import Concept, Category
from .note import Note
from .dosageextractor import DosageExtractor
from .utils.miade_cat import MiADE_CAT
from .utils.modelfactory import ModelFactory
from .utils.metaannotationstypes import *

log = logging.getLogger(__name__)

# Precompile regular expressions
sent_regex = re.compile(r"[^\s][^\n]+")


# this feels a bit circular...maybe a better way
def create_annotator(name: str, model_factory: ModelFactory):
    name = name.lower()
    if name not in model_factory.models:
        raise ValueError(f"MedCAT model for {name} does not exist: either not configured in Config.yaml or "
                         f"missing from models directory")
    annotators = {
        "problems": ProblemsAnnotator,
        "meds/allergies": MedsAllergiesAnnotator
    }
    if name in annotators.keys():
        return annotators[name](model_factory.models[name])
    else:
        log.warning(f"Annotator {name} does not exist, loading generic Annotator")
        return Annotator(model_factory.models[name])


def get_dosage_string(med: Concept, next_med: Optional[Concept], text: str) -> str:
    """
    Finds chunks of text that contain single dosage instructions to input into DosageProcessor
    :param med: (Concept) medications concept
    :param next_med: (Concept) next consecutive medication concept if there is one
    :param text: (str) whole text
    :return: (str) dosage text
    """
    sents = sent_regex.findall(text[med.start: next_med.start] if next_med is not None else text[med.start:])

    concept_name = text[med.start: med.end]
    next_concept_name = text[next_med.start: next_med.end] if next_med else None

    for sent in sents:
        if concept_name in sent:
            if next_med is not None:
                if next_concept_name not in sent:
                    return sent
                else:
                    return text[med.start: next_med.start]
            else:
                ind = sent.find(concept_name)
                return sent[ind:]

    return ""


class Annotator:
    """
    Docstring for Annotator
    """
    def __init__(self, cat: MiADE_CAT):
        self.cat = cat
        self.concept_types = []

    def add_negex_pipeline(self) -> None:
        self.cat.pipe.spacy_nlp.add_pipe("sentencizer")
        self.cat.pipe.spacy_nlp.enable_pipe("sentencizer")
        self.cat.pipe.spacy_nlp.add_pipe("negex")

    def get_concepts(self, note: Note) -> List[Concept]:
        concepts: List[Concept] = []
        for entity in self.cat.get_entities(note)["entities"].values():
            try:
                concepts.append(Concept.from_entity(entity))
            except ValueError as e:
                log.warning(f"Concept skipped: {e}")

        return concepts
    @staticmethod
    def deduplicate(concepts: List[Concept], record_concepts: Optional[List[Concept]]) -> List[Concept]:
        filtered_concepts: List[Concept] = []

        if record_concepts is not None:
            record_ids = [record_concept.id for record_concept in record_concepts]
            for concept in concepts:
                if concept.id in record_ids:
                    log.debug(
                        f"Filtered concept ({concept.id} | {concept.name}): concept id exists in record"
                    )
                    continue
                filtered_concepts.append(concept)
        else:
            filtered_concepts = concepts

        # deduplicate within list
        return sorted(set(filtered_concepts))

    def __call__(
        self,
        note: Note,
        record_concepts: Optional[List[Concept]] = None,
    ):
        return self.deduplicate(self.get_concepts(note), record_concepts)


class ProblemsAnnotator(Annotator):
    def __init__(self, cat: MiADE_CAT):
        super().__init__(cat)
        self.concept_types = [Category.PROBLEM]
        # TODO: move load functions outside initializer
        negated_data = pkgutil.get_data(__name__, "./data/negated.csv")
        self.negated_lookup = (
            pd.read_csv(
                io.BytesIO(negated_data),
                index_col=0,
            )
            .squeeze("columns")
            .T.to_dict()
        )
        historic_data = pkgutil.get_data(__name__, "./data/historic.csv")
        self.historic_lookup = (
            pd.read_csv(
                io.BytesIO(historic_data),
                index_col=0,
            )
            .squeeze("columns")
            .T.to_dict()
        )
        suspected_data = pkgutil.get_data(__name__, "./data/suspected.csv")
        self.suspected_lookup = (
            pd.read_csv(
                io.BytesIO(suspected_data),
                index_col=0,
            )
            .squeeze("columns")
            .T.to_dict()
        )
        blacklist_data = pkgutil.get_data(__name__, "./data/problem_blacklist.csv")
        self.filtering_blacklist = pd.read_csv(io.BytesIO(blacklist_data), header=None)

    def _process_meta_annotations(self, concept: Concept) -> Optional[Concept]:
        # Add, convert, or ignore concepts
        negex = hasattr(concept, "negex")
        meta_ann_values = [meta_ann.value for meta_ann in concept.meta] if concept.meta is not None else []

        convert = False
        tag = ""
        # only get meta model results if negex is false
        if negex:
            if concept.negex:
                convert = self.negated_lookup.get(int(concept.id), False)
                tag = " (negated)"
            elif Presence.SUSPECTED in meta_ann_values:
                convert = self.suspected_lookup.get(int(concept.id), False)
                tag = " (suspected)"
            elif Relevance.HISTORIC in meta_ann_values:
                convert = self.historic_lookup.get(int(concept.id), False)
                tag = " (historic)"
        else:
            if Presence.NEGATED in meta_ann_values:
                convert = self.negated_lookup.get(int(concept.id), False)
                tag = " (negated)"
            elif Presence.SUSPECTED in meta_ann_values:
                convert = self.suspected_lookup.get(int(concept.id), False)
                tag = " (suspected)"
            elif Relevance.HISTORIC in meta_ann_values:
                convert = self.historic_lookup.get(int(concept.id), False)
                tag = " (historic)"

        if convert:
            log.debug(f"Converted concept ({concept.id} | {concept.name}) to ({str(convert)} | {concept.name + tag})")
            concept.id = str(convert)
            concept.name += tag
        else:
            if concept.negex:
                log.debug(
                    f"Filtered concept ({concept.id} | {concept.name}): negation (negex) with no conversion match")
                return None
            if not negex and Presence.NEGATED in meta_ann_values:
                log.debug(
                    f"Filtered concept ({concept.id} | {concept.name}): negation (meta model) with no conversion match")
                return None
            if Presence.SUSPECTED in meta_ann_values:
                log.debug(f"Filtered concept ({concept.id} | {concept.name}): suspected with no conversion match")
                return None
            if Relevance.IRRELEVANT in meta_ann_values:
                log.debug(f"Filtered concept ({concept.id} | {concept.name}): irrelevant concept")
                return None
            if Relevance.HISTORIC in meta_ann_values:
                log.debug(f"Filtered concept ({concept.id} | {concept.name}): historic with no conversion match")
                return None

        return concept

    def _is_blacklist(self, concept):
        # filtering blacklist
        if int(concept.id) in self.filtering_blacklist.values:
            log.debug(
                f"Filtered concept ({concept.id} | {concept.name}): concept in problems blacklist"
            )
            return True
        return False

    def postprocess(self, concepts: List[Concept]) -> List[Concept]:
        # deepcopy so we still have reference to original list of concepts
        all_concepts = deepcopy(concepts)
        filtered_concepts = []
        for concept in all_concepts:
            if self._is_blacklist(concept):
                continue
            # meta annotations
            concept = self._process_meta_annotations(concept)
            # ignore concepts filtered by meta-annotations
            if concept is None:
                continue
            concept.category = Category.PROBLEM
            filtered_concepts.append(concept)

        return filtered_concepts

    def __call__(
        self,
        note: Note,
        record_concepts: Optional[List[Concept]] = None,
    ):
        concepts = self.get_concepts(note)
        concepts = self.postprocess(concepts)
        concepts = self.deduplicate(concepts, record_concepts)

        return concepts


class MedsAllergiesAnnotator(Annotator):
    def __init__(self, cat: MiADE_CAT):
        super().__init__(cat)
        self.concept_types = [Category.MEDICATION, Category.ALLERGY, Category.REACTION]

    @staticmethod
    def add_dosages_to_concepts(
            dosage_extractor: DosageExtractor,
            concepts: List[Concept],
            note: Note
    ) -> List[Concept]:
        """
        Gets dosages for medication concepts
        :param dosage_extractor:
        :param concepts: (List) list of concepts extracted
        :param note: (Note) input note
        :return: (List) list of concepts with dosages for medication concepts
        """

        for ind, concept in enumerate(concepts):
            next_med_concept = (
                concepts[ind + 1]
                if len(concepts) > ind + 1
                else None
            )
            dosage_string = get_dosage_string(concept, next_med_concept, note.text)
            if len(dosage_string.split()) > 2:
                concept.dosage = dosage_extractor(dosage_string)
                # print(concept.dosage)

        return concepts

    def _process_meta_annotations(self, concepts: List[Concept]) -> List[Concept]:
        return concepts

    def postprocess(self, concepts: List[Concept]) -> List[Concept]:
        # TODO: meds / allergies postprocessing
        # this bit of code was to avoid annotation anchor errors when concept is tagged as both reaction and concept
        # we probably don't want to keep this in miade / separate the concerns but keeping it here as a reminder
        # if concept.start is not None and concept.end is not None:
        #     if (concept.start, concept.end) in [
        #         (concept.start, concept.end)
        #         for concept in all_concepts
        #         if concept.category == Category.PROBLEM
        #     ]:
        #         log.debug(f"Temporary filtering of reaction concept: duplication of problem concept")
        #         continue
        # concept = self._process_meta_annotations(concept)
        #
        return concepts


    def __call__(
        self,
        note: Note,
        record_concepts: Optional[List[Concept]] = None,
        dosage_extractor: Optional[DosageExtractor] = None
    ):
        concepts = self.get_concepts(note)
        concepts = self.postprocess(concepts)
        if dosage_extractor is not None:
            concepts = self.add_dosages_to_concepts(dosage_extractor, concepts, note)
        concepts = self.deduplicate(concepts, record_concepts)

        return concepts

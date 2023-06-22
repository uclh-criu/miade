import io
import logging
import pkgutil
import re
import pandas as pd

from typing import List, Optional
from copy import deepcopy
from math import inf

from .concept import Concept, Category
from .note import Note
from .dosageextractor import DosageExtractor
from .utils.miade_cat import MiADE_CAT
from .utils.metaannotationstypes import *

log = logging.getLogger(__name__)

# Precompile regular expressions
sent_regex = re.compile(r"[^\s][^\n]+")


def load_lookup_data(filename: str):
    lookup_data = pkgutil.get_data(__name__, filename)
    return (
        pd.read_csv(
            io.BytesIO(lookup_data),
            index_col=0,
        )
        .squeeze("columns")
        .T.to_dict()
    )


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


def calculate_word_distance(start1: int, end1: int, start2: int, end2: int, note: Note) -> int:
    """
    Calculates how many words are in between the given text spans based on character indices.
    :param start1: Character index of the start of word 1.
    :param end1: Character index of the end of word 1.
    :param start2: Character index of the start of word 2.
    :param end2: Character index of the end of word 2.
    :param note: Note object that contains the whole text.
    :return: Number of words between the two text spans.
    """

    if start1 > end1 or start2 > end2:
        return -1  # Invalid input: start index should be less than or equal to the end index

    if start1 >= len(note.text) or start2 >= len(note.text):
        return -1  # Invalid input: start index exceeds the length of the note's text

    # Adjust the indices to stay within the bounds of the text
    start1 = min(start1, len(note.text) - 1)
    end1 = min(end1, len(note.text) - 1)
    start2 = min(start2, len(note.text) - 1)
    end2 = min(end2, len(note.text) - 1)

    chunk_start = min(start1, start2)
    chunk_end = max(end1 + 1, end2 + 1)

    words = note.text[chunk_start:chunk_end].split()

    return len(words) - 1


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
                concept.category = Category.MEDICATION if concept.dosage is not None else None
                # print(concept.dosage)

        return concepts

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

        self.negated_lookup = load_lookup_data("./data/negated.csv")
        self.historic_lookup = load_lookup_data("./data/historic.csv")
        self.suspected_lookup = load_lookup_data("./data/suspected.csv")

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

        concept.category = Category.PROBLEM

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
        # load the lookup data
        self.reactions_subset_lookup = None
        self.allergens_parents_lookup = None
        self.meds_to_vmp_lookup = None

    def _process_meta_annotations(self, concept: Concept) -> Concept:
        meta_ann_values = [meta_ann.value for meta_ann in concept.meta] if concept.meta is not None else []

        # assign categories
        if SubstanceCategory.ADVERSE_REACTION in meta_ann_values:
            concept.category = Category.ALLERGY
        if SubstanceCategory.TAKING in meta_ann_values:
            concept.category = Category.MEDICATION
        if SubstanceCategory.NOT_SUBSTANCE in meta_ann_values and (
                ReactionPos.BEFORE_SUBSTANCE in meta_ann_values or ReactionPos.AFTER_SUBSTANCE in meta_ann_values):
            concept.category = Category.REACTION

        return concept

    def _map_reactions_to_subset(self, concept: Concept) -> Concept:
        return concept

    def _map_allergens_to_parents(self, concept: Concept) -> Concept:
        return concept

    def _link_reactions_to_allergens(self, concept_list: List[Concept], note: Note, link_distance: int = 5) -> List[Concept]:
        allergy_concepts = [concept for concept in concept_list if concept.category == Category.ALLERGY]
        reaction_concepts = [concept for concept in concept_list if concept.category == Category.REACTION]

        for reaction_concept in reaction_concepts:
            nearest_allergy_concept = None
            min_distance = inf
            meta_ann_values = [
                meta_ann.value for meta_ann in reaction_concept.meta
            ] if reaction_concept.meta is not None else []

            for allergy_concept in allergy_concepts:
                # skip if allergy is after and meta is before_substance
                if ReactionPos.BEFORE_SUBSTANCE in meta_ann_values and allergy_concept.start < reaction_concept.start:
                    continue
                # skip if allergy is before and meta is after_substance
                elif ReactionPos.AFTER_SUBSTANCE in meta_ann_values and allergy_concept.start > reaction_concept.start:
                    continue
                else:
                    log.debug(f"Checking distance between {reaction_concept.name}, {allergy_concept.name}")
                    distance = calculate_word_distance(reaction_concept.start, reaction_concept.end,
                                                       allergy_concept.start, allergy_concept.end,
                                                       note)
                    if distance == -1:
                        log.warning(f"Indices for {reaction_concept.name} or {allergy_concept.name} invalid: "
                                    f"({reaction_concept.start}, {reaction_concept.end})"
                                    f"({allergy_concept.start}, {allergy_concept.end})")
                        continue

                    if distance <= link_distance and distance < min_distance:
                        min_distance = distance
                        nearest_allergy_concept = allergy_concept

            if nearest_allergy_concept is not None:
                log.debug(f"Linking reaction {reaction_concept.name} to {nearest_allergy_concept.name}")
                nearest_allergy_concept.linked_concepts = [reaction_concept]

        # Remove the linked REACTION concepts from the main list
        updated_concept_list = [concept for concept in concept_list if concept.category != Category.REACTION]

        return updated_concept_list

    def postprocess(self, concepts: List[Concept], note: Note) -> List[Concept]:
        # deepcopy so we still have reference to original list of concepts
        all_concepts = deepcopy(concepts)
        processed_concepts = []

        for concept in all_concepts:
            # 1. process meta annotations to assign med/allergy/reaction category
            concept = self._process_meta_annotations(concept)
            # 2. convert concepts from lookup tables
            if concept.category == Category.ALLERGY:
                # TODO: 3. convert allergen concept to parent concepts (lookup)
                # TODO: need a container for severity
                concept = self._map_allergens_to_parents(concept)
            elif concept.category == Category.REACTION:
                # TODO: 4. convert reaction to Epic options (lookup)
                concept = self._map_reactions_to_subset(concept)

            processed_concepts.append(concept)

        # 5. link reaction to allergens
        processed_concepts = self._link_reactions_to_allergens(processed_concepts, note)

        return processed_concepts

    def convert_medications_to_products(self, concepts: List[Concept]) -> List[Concept]:
        # TODO: convert to medication VMP
        return concepts

    def __call__(
        self,
        note: Note,
        record_concepts: Optional[List[Concept]] = None,
        dosage_extractor: Optional[DosageExtractor] = None
    ):
        concepts = self.get_concepts(note)
        concepts = self.postprocess(concepts, note)
        if dosage_extractor is not None:
            concepts = self.add_dosages_to_concepts(dosage_extractor, concepts, note)
        concepts = self.convert_medications_to_products(concepts)
        concepts = self.deduplicate(concepts, record_concepts)

        return concepts

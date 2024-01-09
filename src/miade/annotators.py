import os
import logging
import re
from enum import Enum

import pandas as pd

from typing import List, Optional, Tuple, Dict
from collections import OrderedDict
from copy import deepcopy
from math import inf
from abc import ABC, abstractmethod


from medcat.cat import CAT

from .concept import Concept, Category
from .note import Note
from .paragraph import Paragraph, ParagraphType
from .dosageextractor import DosageExtractor
from .utils.metaannotationstypes import (
    Presence,
    Relevance,
    ReactionPos,
    SubstanceCategory,
    Severity,
)
from .utils.annotatorconfig import AnnotatorConfig

log = logging.getLogger(__name__)

# Precompile regular expressions
sent_regex = re.compile(r"[^\s][^\n]+")


class AllergenType(Enum):
    FOOD = "food"
    DRUG = "drug"
    DRUG_CLASS = "drug class"
    DRUG_INGREDIENT = "drug ingredient"
    CHEMICAL = "chemical"
    ENVIRONMENTAL = "environmental"
    ANIMAL = "animal"


def load_lookup_data(filename: str, as_dict: bool = False, no_header: bool = False):
    if as_dict:
        return (
            pd.read_csv(
                filename,
                index_col=0,
            )
            .squeeze("columns")
            .T.to_dict()
        )
    if no_header:
        return pd.read_csv(filename, header=None)
    else:
        return pd.read_csv(filename).drop_duplicates()


def load_allergy_type_combinations(filename: str) -> Dict:
    df = pd.read_csv(filename)

    # Convert 'allergenType' and 'adverseReactionType' columns to lowercase
    df["allergenType"] = df["allergenType"].str.lower()
    df["adverseReactionType"] = df["adverseReactionType"].str.lower()

    # Create a tuple column containing (reaction_id, reaction_name) for each row
    df["reaction_id_name"] = list(zip(df["adverseReactionId"], df["adverseReactionName"]))

    # Set (allergenType, adverseReactionType) as the index and convert to dictionary
    result_dict = df.set_index(["allergenType", "adverseReactionType"])["reaction_id_name"].to_dict()

    return result_dict


def get_dosage_string(med: Concept, next_med: Optional[Concept], text: str) -> str:
    """
    Finds chunks of text that contain single dosage instructions to input into DosageProcessor
    :param med: (Concept) medications concept
    :param next_med: (Concept) next consecutive medication concept if there is one
    :param text: (str) whole text
    :return: (str) dosage text
    """
    sents = sent_regex.findall(text[med.start : next_med.start] if next_med is not None else text[med.start :])

    concept_name = text[med.start : med.end]
    next_concept_name = text[next_med.start : next_med.end] if next_med else None

    for sent in sents:
        if concept_name in sent:
            if next_med is not None:
                if next_concept_name not in sent:
                    return sent
                else:
                    return text[med.start : next_med.start]
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


class Annotator(ABC):
    """
    Docstring for Annotator
    """

    def __init__(self, cat: CAT, config: AnnotatorConfig = None):
        self.cat = cat
        self.config = config if config is not None else AnnotatorConfig()

        if self.config.negation_detection == "negex":
            self._add_negex_pipeline()

        # TODO make paragraph processing params configurable
        self.structured_prob_lists = {
            ParagraphType.prob: Relevance.PRESENT,
            ParagraphType.imp: Relevance.PRESENT,
            ParagraphType.pmh: Relevance.HISTORIC,
        }
        self.structured_med_lists = {
            ParagraphType.med: SubstanceCategory.TAKING,
            ParagraphType.allergy: SubstanceCategory.ADVERSE_REACTION,
        }
        self.irrelevant_paragraphs = [ParagraphType.ddx, ParagraphType.exam, ParagraphType.plan]

    def _add_negex_pipeline(self) -> None:
        self.cat.pipe.spacy_nlp.add_pipe("sentencizer")
        self.cat.pipe.spacy_nlp.enable_pipe("sentencizer")
        self.cat.pipe.spacy_nlp.add_pipe("negex")

    @property
    @abstractmethod
    def concept_types(self):
        pass

    @property
    @abstractmethod
    def pipeline(self):
        pass

    @abstractmethod
    def process_paragraphs(self):
        pass

    @abstractmethod
    def postprocess(self):
        pass

    def run_pipeline(self, note: Note, record_concepts: List[Concept]) -> List[Concept]:
        concepts: List[Concept] = []

        for pipe in self.pipeline:
            if pipe not in self.config.disable:
                if pipe == "preprocessor":
                    note = self.preprocess(note)
                elif pipe == "medcat":
                    concepts = self.get_concepts(note)
                elif pipe == "paragrapher":
                    concepts = self.process_paragraphs(note, concepts)
                elif pipe == "postprocessor":
                    concepts = self.postprocess(concepts)
                elif pipe == "deduplicator":
                    concepts = self.deduplicate(concepts, record_concepts)

        return concepts

    def get_concepts(self, note: Note) -> List[Concept]:
        concepts: List[Concept] = []
        for entity in self.cat.get_entities(note)["entities"].values():
            try:
                concepts.append(Concept.from_entity(entity))
                log.debug(f"Detected concept ({concepts[-1].id} | {concepts[-1].name})")
            except ValueError as e:
                log.warning(f"Concept skipped: {e}")

        return concepts

    @staticmethod
    def preprocess(note: Note) -> Note:
        note.clean_text()
        note.get_paragraphs()

        return note

    @staticmethod
    def deduplicate(concepts: List[Concept], record_concepts: Optional[List[Concept]]) -> List[Concept]:
        if record_concepts is not None:
            record_ids = {record_concept.id for record_concept in record_concepts}
            record_names = {record_concept.name for record_concept in record_concepts}
        else:
            record_ids = set()
            record_names = set()

        # Use an OrderedDict to keep track of ids as it preservers original MedCAT order (the order it appears in text)
        filtered_concepts: List[Concept] = []
        existing_concepts = OrderedDict()

        # Filter concepts that are in record or exist in concept list
        for concept in concepts:
            if concept.id is not None and (concept.id in record_ids or concept.id in existing_concepts):
                log.debug(f"Removed concept ({concept.id} | {concept.name}): concept id exists in record")
            # check name match for null ids - VTM deduplication
            elif concept.id is None and (concept.name in record_names or concept.name in existing_concepts.values()):
                log.debug(f"Removed concept ({concept.id} | {concept.name}): concept name exists in record")
            else:
                filtered_concepts.append(concept)
                existing_concepts[concept.id] = concept.name

        return filtered_concepts

    @staticmethod
    def add_numbering_to_name(concepts: List[Concept]) -> List[Concept]:
        # Prepend numbering to problem concepts e.g. 00 asthma, 01 stroke...
        for i, concept in enumerate(concepts):
            concept.name = f"{i:02} {concept.name}"

        return concepts

    def __call__(
        self,
        note: Note,
        record_concepts: Optional[List[Concept]] = None,
    ):
        concepts = self.run_pipeline(note, record_concepts)

        if self.config.add_numbering:
            concepts = self.add_numbering_to_name(concepts)

        return concepts


class ProblemsAnnotator(Annotator):
    def __init__(self, cat: CAT, config: AnnotatorConfig = None):
        super().__init__(cat, config)
        self._load_problems_lookup_data()

    @property
    def concept_types(self):
        return [Category.PROBLEM]

    @property
    def pipeline(self):
        return ["preprocessor", "medcat", "paragrapher", "postprocessor", "deduplicator"]

    def _load_problems_lookup_data(self) -> None:
        if not os.path.isdir(self.config.lookup_data_path):
            raise RuntimeError(f"No lookup data configured: {self.config.lookup_data_path} does not exist!")
        else:
            self.negated_lookup = load_lookup_data(self.config.lookup_data_path + "negated.csv", as_dict=True)
            self.historic_lookup = load_lookup_data(self.config.lookup_data_path + "historic.csv", as_dict=True)
            self.suspected_lookup = load_lookup_data(self.config.lookup_data_path + "suspected.csv", as_dict=True)
            self.filtering_blacklist = load_lookup_data(
                self.config.lookup_data_path + "problem_blacklist.csv", no_header=True
            )

    def _process_meta_annotations(self, concept: Concept) -> Optional[Concept]:
        # Add, convert, or ignore concepts
        meta_ann_values = [meta_ann.value for meta_ann in concept.meta] if concept.meta is not None else []

        convert = False
        tag = ""
        # only get meta model results if negex is false
        if concept.negex is not None:
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
            if tag == " (negated)" and concept.negex:
                log.debug(
                    f"Converted concept ({concept.id} | {concept.name}) to ({str(convert)} | {concept.name + tag}): "
                    f"negation detected by negex"
                )
            else:
                log.debug(
                    f"Converted concept ({concept.id} | {concept.name}) to ({str(convert)} | {concept.name + tag}):"
                    f"detected by meta model"
                )
            concept.id = str(convert)
            concept.name += tag
        else:
            if concept.negex:
                log.debug(f"Removed concept ({concept.id} | {concept.name}): negation (negex) with no conversion match")
                return None
            if concept.negex is None and Presence.NEGATED in meta_ann_values:
                log.debug(
                    f"Removed concept ({concept.id} | {concept.name}): negation (meta model) with no conversion match"
                )
                return None
            if Presence.SUSPECTED in meta_ann_values:
                log.debug(f"Removed concept ({concept.id} | {concept.name}): suspected with no conversion match")
                return None
            if Relevance.IRRELEVANT in meta_ann_values:
                log.debug(f"Removed concept ({concept.id} | {concept.name}): irrelevant concept")
                return None
            if Relevance.HISTORIC in meta_ann_values:
                log.debug(f"No change to concept ({concept.id} | {concept.name}): historic with no conversion match")

        concept.category = Category.PROBLEM

        return concept

    def _is_blacklist(self, concept):
        # filtering blacklist
        if int(concept.id) in self.filtering_blacklist.values:
            log.debug(f"Removed concept ({concept.id} | {concept.name}): concept in problems blacklist")
            return True
        return False

    def _process_meta_ann_by_paragraph(
        self, concept: Concept, paragraph: Paragraph, prob_concepts_in_structured_sections: List[Concept]
    ):
        # if paragraph is structured problems section, add to prob list and convert to corresponding relevance
        if paragraph.type in self.structured_prob_lists:
            prob_concepts_in_structured_sections.append(concept)
            for meta in concept.meta:
                if meta.name == "relevance" and meta.value == Relevance.IRRELEVANT:
                    new_relevance = self.structured_prob_lists[paragraph.type]
                    log.debug(
                        f"Converted {meta.value} to "
                        f"{new_relevance} for concept ({concept.id} | {concept.name}): "
                        f"paragraph is {paragraph.type}"
                    )
                    meta.value = new_relevance
        # if paragraph is meds or irrelevant section, convert problems to irrelevant
        elif paragraph.type in self.structured_med_lists or paragraph.type in self.irrelevant_paragraphs:
            for meta in concept.meta:
                if meta.name == "relevance" and meta.value != Relevance.IRRELEVANT:
                    log.debug(
                        f"Converted {meta.value} to "
                        f"{Relevance.IRRELEVANT} for concept ({concept.id} | {concept.name}): "
                        f"paragraph is {paragraph.type}"
                    )
                    meta.value = Relevance.IRRELEVANT

    def process_paragraphs(self, note: Note, concepts: List[Concept]) -> List[Concept]:
        prob_concepts_in_structured_sections: List[Concept] = []

        for paragraph in note.paragraphs:
            for concept in concepts:
                if concept.start >= paragraph.start and concept.end <= paragraph.end:
                    # log.debug(f"({concept.name} | {concept.id}) is in {paragraph.type}")
                    if concept.meta:
                        self._process_meta_ann_by_paragraph(concept, paragraph, prob_concepts_in_structured_sections)

        # if more than set no. concepts in prob or imp or pmh sections, return only those and ignore all other concepts
        if len(prob_concepts_in_structured_sections) > self.config.structured_list_limit:
            log.debug(
                f"Ignoring concepts elsewhere in the document because "
                f"more than {self.config.structured_list_limit} concepts exist "
                f"in prob, imp, pmh structured sections: {len(prob_concepts_in_structured_sections)}"
            )
            return prob_concepts_in_structured_sections

        return concepts

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


class MedsAllergiesAnnotator(Annotator):
    def __init__(self, cat: CAT, config: AnnotatorConfig = None):
        super().__init__(cat, config)
        self._load_med_allergy_lookup_data()

    @property
    def concept_types(self):
        return [Category.MEDICATION, Category.ALLERGY, Category.REACTION]

    @property
    def pipeline(self):
        return [
            "preprocessor",
            "medcat",
            "paragrapher",
            "postprocessor",
            "dosage_extractor",
            "vtm_converter",
            "deduplicator",
        ]

    def run_pipeline(
        self, note: Note, record_concepts: List[Concept], dosage_extractor: Optional[DosageExtractor]
    ) -> List[Concept]:
        concepts: List[Concept] = []

        for pipe in self.pipeline:
            if pipe not in self.config.disable:
                if pipe == "preprocessor":
                    note = self.preprocess(note)
                elif pipe == "medcat":
                    concepts = self.get_concepts(note)
                elif pipe == "paragrapher":
                    concepts = self.process_paragraphs(note, concepts)
                elif pipe == "postprocessor":
                    concepts = self.postprocess(concepts, note)
                elif pipe == "deduplicator":
                    concepts = self.deduplicate(concepts, record_concepts)
                elif pipe == "add_numbering":
                    concepts = self.add_numbering_to_name(concepts)
                elif pipe == "VTM_converter":
                    concepts = self.convert_VTM_to_VMP_or_text(concepts)
                elif pipe == "dosage_extractor" and dosage_extractor is not None:
                    concepts = self.add_dosages_to_concepts(dosage_extractor, concepts, note)

        return concepts

    def _load_med_allergy_lookup_data(self) -> None:
        if not os.path.isdir(self.config.lookup_data_path):
            raise RuntimeError(f"No lookup data configured: {self.config.lookup_data_path} does not exist!")
        else:
            self.valid_meds = load_lookup_data(self.config.lookup_data_path + "valid_meds.csv", no_header=True)
            self.reactions_subset_lookup = load_lookup_data(
                self.config.lookup_data_path + "reactions_subset.csv", as_dict=True
            )
            self.allergens_subset_lookup = load_lookup_data(
                self.config.lookup_data_path + "allergens_subset.csv", as_dict=True
            )
            self.allergy_type_lookup = load_allergy_type_combinations(self.config.lookup_data_path + "allergy_type.csv")
            self.vtm_to_vmp_lookup = load_lookup_data(self.config.lookup_data_path + "vtm_to_vmp.csv")
            self.vtm_to_text_lookup = load_lookup_data(self.config.lookup_data_path + "vtm_to_text.csv", as_dict=True)

    def _validate_meds(self, concept) -> bool:
        # check if substance is valid med
        if int(concept.id) in self.valid_meds.values:
            return True
        return False

    def _validate_and_convert_substance(self, concept) -> bool:
        # check if substance is valid substance for allergy - if it is, convert it to Epic subset and return that concept
        lookup_result = self.allergens_subset_lookup.get(int(concept.id))
        if lookup_result is not None:
            log.debug(
                f"Converted concept ({concept.id} | {concept.name}) to "
                f"({lookup_result['subsetId']} | {concept.name}): valid Epic allergen subset"
            )
            concept.id = str(lookup_result["subsetId"])

            # then check the allergen type from lookup result - e.g. drug, food
            try:
                concept.category = AllergenType(str(lookup_result["allergenType"]).lower())
                log.debug(
                    f"Assigned substance concept ({concept.id} | {concept.name}) "
                    f"to allergen type category {concept.category}"
                )
            except ValueError as e:
                log.warning(f"Allergen type not found for {concept.__str__()}: {e}")

            return True
        else:
            log.warning(f"No lookup subset found for substance ({concept.id} | {concept.name})")
            return False

    def _validate_and_convert_reaction(self, concept) -> bool:
        # check if substance is valid reaction - if it is, convert it to Epic subset and return that concept
        lookup_result = self.reactions_subset_lookup.get(int(concept.id), None)
        if lookup_result is not None:
            log.debug(
                f"Converted concept ({concept.id} | {concept.name}) to "
                f"({lookup_result} | {concept.name}): valid Epic reaction subset"
            )
            concept.id = str(lookup_result)
            return True
        else:
            log.warning(f"Reaction not found in Epic subset conversion for concept {concept.__str__()}")
            return False

    def _validate_and_convert_concepts(self, concept: Concept) -> Concept:
        meta_ann_values = [meta_ann.value for meta_ann in concept.meta] if concept.meta is not None else []

        # assign categories
        if SubstanceCategory.ADVERSE_REACTION in meta_ann_values:
            if self._validate_and_convert_substance(concept):
                self._convert_allergy_type_to_code(concept)
                self._convert_allergy_severity_to_code(concept)
                concept.category = Category.ALLERGY
            else:
                log.warning(f"Double-checking if concept ({concept.id} | {concept.name}) is in reaction subset")
                if self._validate_and_convert_reaction(concept) and (
                    ReactionPos.BEFORE_SUBSTANCE in meta_ann_values or ReactionPos.AFTER_SUBSTANCE in meta_ann_values
                ):
                    concept.category = Category.REACTION
                else:
                    log.warning(
                        f"Reaction concept ({concept.id} | {concept.name}) not in subset or reaction_pos is NOT_REACTION"
                    )
        if SubstanceCategory.TAKING in meta_ann_values:
            if self._validate_meds(concept):
                concept.category = Category.MEDICATION
        if SubstanceCategory.NOT_SUBSTANCE in meta_ann_values and (
            ReactionPos.BEFORE_SUBSTANCE in meta_ann_values or ReactionPos.AFTER_SUBSTANCE in meta_ann_values
        ):
            if self._validate_and_convert_reaction(concept):
                concept.category = Category.REACTION

        return concept

    @staticmethod
    def add_dosages_to_concepts(
        dosage_extractor: DosageExtractor, concepts: List[Concept], note: Note
    ) -> List[Concept]:
        """
        Gets dosages for medication concepts
        :param dosage_extractor:
        :param concepts: (List) list of concepts extracted
        :param note: (Note) input note
        :return: (List) list of concepts with dosages for medication concepts
        """

        for ind, concept in enumerate(concepts):
            next_med_concept = concepts[ind + 1] if len(concepts) > ind + 1 else None
            dosage_string = get_dosage_string(concept, next_med_concept, note.text)
            if len(dosage_string.split()) > 2:
                concept.dosage = dosage_extractor(dosage_string)
                concept.category = Category.MEDICATION if concept.dosage is not None else None
                if concept.dosage is not None:
                    log.debug(
                        f"Extracted dosage for medication concept "
                        f"({concept.id} | {concept.name}): {concept.dosage.text} {concept.dosage.dose}"
                    )

        return concepts

    @staticmethod
    def _link_reactions_to_allergens(concept_list: List[Concept], note: Note, link_distance: int = 5) -> List[Concept]:
        allergy_concepts = [concept for concept in concept_list if concept.category == Category.ALLERGY]
        reaction_concepts = [concept for concept in concept_list if concept.category == Category.REACTION]

        for reaction_concept in reaction_concepts:
            nearest_allergy_concept = None
            min_distance = inf
            meta_ann_values = (
                [meta_ann.value for meta_ann in reaction_concept.meta] if reaction_concept.meta is not None else []
            )

            for allergy_concept in allergy_concepts:
                # skip if allergy is after and meta is before_substance
                if ReactionPos.BEFORE_SUBSTANCE in meta_ann_values and allergy_concept.start < reaction_concept.start:
                    continue
                # skip if allergy is before and meta is after_substance
                elif ReactionPos.AFTER_SUBSTANCE in meta_ann_values and allergy_concept.start > reaction_concept.start:
                    continue
                else:
                    distance = calculate_word_distance(
                        reaction_concept.start, reaction_concept.end, allergy_concept.start, allergy_concept.end, note
                    )
                    log.debug(
                        f"Calculated distance between reaction {reaction_concept.name} "
                        f"and allergen {allergy_concept.name}: {distance}"
                    )
                    if distance == -1:
                        log.warning(
                            f"Indices for {reaction_concept.name} or {allergy_concept.name} invalid: "
                            f"({reaction_concept.start}, {reaction_concept.end})"
                            f"({allergy_concept.start}, {allergy_concept.end})"
                        )
                        continue

                    if distance <= link_distance and distance < min_distance:
                        min_distance = distance
                        nearest_allergy_concept = allergy_concept

            if nearest_allergy_concept is not None:
                nearest_allergy_concept.linked_concepts.append(reaction_concept)
                log.debug(
                    f"Linked reaction concept {reaction_concept.name} to "
                    f"allergen concept {nearest_allergy_concept.name}"
                )

        # Remove the linked REACTION concepts from the main list
        updated_concept_list = [concept for concept in concept_list if concept.category != Category.REACTION]

        return updated_concept_list

    @staticmethod
    def _convert_allergy_severity_to_code(concept: Concept) -> bool:
        meta_ann_values = [meta_ann.value for meta_ann in concept.meta] if concept.meta is not None else []
        if Severity.MILD in meta_ann_values:
            concept.linked_concepts.append(Concept(id="L", name="Low", category=Category.SEVERITY))
        elif Severity.MODERATE in meta_ann_values:
            concept.linked_concepts.append(Concept(id="M", name="Moderate", category=Category.SEVERITY))
        elif Severity.SEVERE in meta_ann_values:
            concept.linked_concepts.append(Concept(id="H", name="High", category=Category.SEVERITY))
        elif Severity.UNSPECIFIED in meta_ann_values:
            return True
        else:
            log.warning(f"No severity annotation associated with ({concept.id} | {concept.name})")
            return False

        log.debug(
            f"Linked severity concept ({concept.linked_concepts[-1].id} | {concept.linked_concepts[-1].name}) "
            f"to allergen concept ({concept.id} | {concept.name}): valid meta model output"
        )

        return True

    def _convert_allergy_type_to_code(self, concept: Concept) -> bool:
        # get the ALLERGYTYPE meta-annotation
        allergy_type = [meta_ann for meta_ann in concept.meta if meta_ann.name == "allergy_type"]
        if len(allergy_type) != 1:
            log.warning(
                f"Unable to map allergy type code: allergy_type meta-annotation "
                f"not found for concept {concept.__str__()}"
            )
            return False
        else:
            allergy_type = allergy_type[0].value

        # perform lookup with ALLERGYTYPE and AllergenType combination
        lookup_combination: Tuple[str, str] = (concept.category.value, allergy_type.value)
        allergy_type_lookup_result = self.allergy_type_lookup.get(lookup_combination)

        # add resulting allergy type concept as to linked_concept
        if allergy_type_lookup_result is not None:
            concept.linked_concepts.append(
                Concept(
                    id=str(allergy_type_lookup_result[0]),
                    name=allergy_type_lookup_result[1],
                    category=Category.ALLERGY_TYPE,
                )
            )
            log.debug(
                f"Linked allergy_type concept ({allergy_type_lookup_result[0]} | {allergy_type_lookup_result[1]})"
                f" to allergen concept ({concept.id} | {concept.name}): valid meta model output + allergytype lookup"
            )
        else:
            log.warning(f"Allergen and adverse reaction type combination not found: {lookup_combination}")

        return True

    def _process_meta_ann_by_paragraph(self, concept: Concept, paragraph: Paragraph):
        # if paragraph is structured meds to convert to corresponding relevance
        if paragraph.type in self.structured_med_lists:
            for meta in concept.meta:
                if meta.name == "substance_category" and meta.value in [
                    SubstanceCategory.TAKING,
                    SubstanceCategory.IRRELEVANT,
                ]:
                    new_relevance = self.structured_med_lists[paragraph.type]
                    if meta.value != new_relevance:
                        log.debug(
                            f"Converted {meta.value} to "
                            f"{new_relevance} for concept ({concept.id} | {concept.name}): "
                            f"paragraph is {paragraph.type}"
                        )
                        meta.value = new_relevance
        # if paragraph is probs or irrelevant section, convert substance to irrelevant
        elif paragraph.type in self.structured_prob_lists or paragraph.type in self.irrelevant_paragraphs:
            for meta in concept.meta:
                if meta.name == "substance_category" and meta.value != SubstanceCategory.IRRELEVANT:
                    log.debug(
                        f"Converted {meta.value} to "
                        f"{SubstanceCategory.IRRELEVANT} for concept ({concept.id} | {concept.name}): "
                        f"paragraph is {paragraph.type}"
                    )
                    meta.value = SubstanceCategory.IRRELEVANT

    def process_paragraphs(self, note: Note, concepts: List[Concept]) -> List[Concept]:
        for paragraph in note.paragraphs:
            for concept in concepts:
                if concept.start >= paragraph.start and concept.end <= paragraph.end:
                    # log.debug(f"({concept.name} | {concept.id}) is in {paragraph.type}")
                    if concept.meta:
                        self._process_meta_ann_by_paragraph(concept, paragraph)

        return concepts

    def postprocess(self, concepts: List[Concept], note: Note) -> List[Concept]:
        # deepcopy so we still have reference to original list of concepts
        all_concepts = deepcopy(concepts)
        processed_concepts = []

        for concept in all_concepts:
            concept = self._validate_and_convert_concepts(concept)
            processed_concepts.append(concept)

        processed_concepts = self._link_reactions_to_allergens(processed_concepts, note)

        return processed_concepts

    def convert_VTM_to_VMP_or_text(self, concepts: List[Concept]) -> List[Concept]:
        # Get medication concepts
        med_concepts = [concept for concept in concepts if concept.category == Category.MEDICATION]
        self.vtm_to_vmp_lookup["dose"] = self.vtm_to_vmp_lookup["dose"].astype(float)

        med_concepts_with_dose = []
        # I don't know man...Need to improve dosage methods
        for concept in med_concepts:
            if concept.dosage is not None:
                if concept.dosage.dose:
                    if concept.dosage.dose.value is not None and concept.dosage.dose.unit is not None:
                        med_concepts_with_dose.append(concept)

        med_concepts_no_dose = [concept for concept in concepts if concept not in med_concepts_with_dose]

        # Create a temporary DataFrame to match vtmId, dose, and unit
        temp_df = pd.DataFrame(
            {
                "vtmId": [int(concept.id) for concept in med_concepts_with_dose],
                "dose": [float(concept.dosage.dose.value) for concept in med_concepts_with_dose],
                "unit": [concept.dosage.dose.unit for concept in med_concepts_with_dose],
            }
        )

        # Merge with the lookup df to get vmpId
        merged_df = temp_df.merge(self.vtm_to_vmp_lookup, on=["vtmId", "dose", "unit"], how="left")

        # Update id in the concepts list
        for index, concept in enumerate(med_concepts_with_dose):
            # Convert VTM to VMP id
            vmp_id = merged_df.at[index, "vmpId"]
            if not pd.isna(vmp_id):
                log.debug(
                    f"Converted ({concept.id} | {concept.name}) to "
                    f"({int(vmp_id)} | {concept.name + ' ' + str(int(concept.dosage.dose.value)) + concept.dosage.dose.unit} "
                    f"tablets): valid extracted dosage + VMP lookup"
                )
                concept.id = str(int(vmp_id))
                concept.name += " " + str(int(concept.dosage.dose.value)) + str(concept.dosage.dose.unit) + " tablets"
                # If found VMP match change the dosage to 1 tablet
                concept.dosage.dose.value = 1
                concept.dosage.dose.unit = "{tbl}"
            else:
                # If no match with dose convert to text
                lookup_result = self.vtm_to_text_lookup.get(int(concept.id))
                if lookup_result is not None:
                    log.debug(
                        f"Converted ({concept.id} | {concept.name}) to (None | {lookup_result}: no match to VMP dosage lookup)"
                    )
                    concept.id = None
                    concept.name = lookup_result

        # Convert rest of VTMs that have no dose for VMP conversion to text
        for concept in med_concepts_no_dose:
            lookup_result = self.vtm_to_text_lookup.get(int(concept.id))
            if lookup_result is not None:
                log.debug(f"Converted ({concept.id} | {concept.name}) to (None | {lookup_result}): no dosage detected")
                concept.id = None
                concept.name = lookup_result

        return concepts

    def __call__(
        self,
        note: Note,
        record_concepts: Optional[List[Concept]] = None,
        dosage_extractor: Optional[DosageExtractor] = None,
    ):
        concepts = self.run_pipeline(note, record_concepts, dosage_extractor)

        if self.config.add_numbering:
            concepts = self.add_numbering_to_name(concepts)

        return concepts

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
from .utils.miadecat import MiADE_CAT

log = logging.getLogger(__name__)
log.setLevel("DEBUG")


class DebugMode(Enum):
    PRELOADED = 1
    CDA = 2
    MODEL = 3


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
                model_pack_filepath, meta_cat_config_dict=meta_cat_config_dict
            )
            for model_pack_filepath in model_directory.glob("*.zip")
        ]
        self.dosage_extractor = DosageExtractor()
        self.concept_filter = ConceptFilter(use_negex=use_negex)

        if use_negex:
            log.info(
                "Using Negex as priority for meta context detection"
            )
            self._add_negex_pipeline()

        if problems_model_id is not None:
            log.info(f"Configured to use problems model {self.problems_model_id}")
        else:
            log.info(f"Problems model ID not configured, using all models in model path {model_directory}")

    def process(
        self, note: Note, record_concepts: Optional[List[Concept]] = None
    ) -> List[Concept]:

        concepts: List[Concept] = []
        for annotator in self.annotators:
            if annotator.config.version["id"] == self.problems_model_id:
                for entity in annotator.get_entities(note)["entities"].values():
                    try:
                        concept = Concept.from_entity(entity)
                        concepts.append(concept)
                    except ValueError as e:
                        log.warning(f"Concept skipped: {e}")
            else:
                log.warning(f"Model {annotator.config.version['id']} is not a problems model and will not be used")

        log.debug(f"Detected concepts: {[(concept.id, concept.name, concept.category.name) for concept in concepts]}")
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

    def debug(
        self,
        debug_config_path: Path,
        mode: DebugMode = DebugMode.PRELOADED,
        code: Optional[int] = 0,
    ) -> (List[Concept], Dict):
        """
        Returns debug configurations for end-to-end testing of the MiADE unit in NoteReader
        :param debug_config_path: Path of debug config file
        :param mode: (DebugCode) which debug mode to use
        :param code: (int) which preset configuration from the config file to use
        :return: (tuple) list of concepts to return and CDA dictionary
        """
        # print(debug_config)
        # TODO: tidy & update debug mode
        with open(debug_config_path, "r") as stream:
            debug_config = yaml.safe_load(stream)

        # use preloaded concepts and cda fields
        if mode == DebugMode.PRELOADED:
            concept_list = []
            for name, concept_dict in debug_config["Presets"][code].items():
                dosage = None
                debug = None
                if concept_dict["ontologies"] == "FDB":
                    category = Category.MEDICATION
                    if "dosage" in concept_dict:
                        dosage = Dosage(
                            text="debug mode",
                            dose=Dose(**concept_dict["dosage"].get("dose")),
                            frequency=Frequency(
                                **concept_dict["dosage"].get("frequency")
                            ),
                            duration=Duration(**concept_dict["dosage"].get("duration")),
                            route=Route(**concept_dict["dosage"].get("route")),
                        )
                elif concept_dict["ontologies"] == "ELG":
                    category = Category.ALLERGY
                    debug = {}
                    if "reaction" in concept_dict:
                        debug["reaction"] = concept_dict["reaction"]
                    if "severity" in concept_dict:
                        debug["severity"] = concept_dict["severity"]
                elif concept_dict["ontologies"] == "SNOMED CT":
                    category = Category.PROBLEM
                else:
                    category = Category.PROBLEM
                concept_list.append(
                    Concept(
                        id=concept_dict["cui"],
                        name=name,
                        category=category,
                        dosage=dosage,
                        debug_dict=debug,
                    )
                )
            return concept_list
        # detect concepts and return preloaded cda fields
        elif mode == DebugMode.CDA:
            return debug_config["CDA"][code]
        # switch out models once we have multiple models/version control
        elif mode == DebugMode.MODEL:
            for model in self.annotators:
                model.get_model_card()

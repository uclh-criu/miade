import re
import yaml
import pkgutil

from pathlib import Path
from typing import List, Dict, Optional
from enum import Enum

from medcat.cat import CAT

from .concept import Concept, Category
from .note import Note
from .dosageprocessor import DosageProcessor


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
    ):
        meta_cat_config_dict = {"general": {"device": "cpu"}}
        self.annotators = [
            CAT.load_model_pack(
                model_pack_filepath, meta_cat_config_dict=meta_cat_config_dict
            )
            for model_pack_filepath in model_directory.glob("*.zip")
        ]
        self.dosage_processor = DosageProcessor()

        config_file = pkgutil.get_data(__name__, "configs/debug_config.yml")
        self.debug_config = yaml.safe_load(config_file)

    def process(
        self, note: Note, patient_data: Optional[List[Concept]] = None
    ) -> List[Concept]:

        concepts: List[Concept] = []

        for annotator in self.annotators:
            for entity in annotator.get_entities(note)["entities"].values():
                # print(entity)
                if entity["ontologies"] == ["FDB"]:
                    category = Category.MEDICATION
                elif entity["ontologies"] == ["SNOMED-CT"]:
                    category = Category.DIAGNOSIS
                elif entity["ontologies"] == ["ELG"]:
                    category = Category.ALLERGY
                else:
                    category = Category.DIAGNOSIS
                concepts.append(
                    Concept(
                        id=entity["cui"],
                        name=entity["pretty_name"],
                        category=category,
                        start=entity["start"],
                        end=entity["end"],
                    )
                )

        concepts = self.add_dosages_to_concepts(concepts, note)

        return concepts

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
                    concept.dosage = self.dosage_processor.process(text=dosage_string)
                    # print(concept.dosage)

        return concepts

    def debug(
        self, mode: DebugMode = DebugMode.PRELOADED, code: Optional[int] = 0
    ) -> (List[Concept], Dict):
        """
        Returns debug configurations for end-to-end testing of the MiADE unit in NoteReader
        :param mode: (DebugCode) which debug mode to use
        :param code: (int) which preset configuration from the config file to use
        :return: (tuple) list of concepts to return and CDA dictionary
        """
        # print(debug_config)
        # use preloaded concepts and cda fields
        if mode == DebugMode.PRELOADED:
            concept_list = []
            for name, concept_dict in self.debug_config["Presets"][code].items():
                dosage = None
                meta = None
                if concept_dict["ontologies"] == "FDB":
                    category = Category.MEDICATION
                    if "dosage" in concept_dict:
                        dosage = concept_dict["dosage"]  # sub for MedicationActivity
                elif concept_dict["ontologies"] == "ELG":
                    category = Category.ALLERGY
                    if "reaction" in concept_dict:
                        meta = concept_dict["reaction"]
                elif concept_dict["ontologies"] == "SNOMED CT":
                    category = Category.DIAGNOSIS
                else:
                    category = Category.DIAGNOSIS
                concept_list.append(
                    Concept(
                        id=concept_dict["cui"],
                        name=name,
                        category=category,
                        dosage=dosage,
                        meta=meta,
                    )
                )
            return concept_list
        # detect concepts and return preloaded cda fields
        elif mode == DebugMode.CDA:
            return self.debug_config["CDA"][code]
        # switch out models once we have multiple models/version control
        elif mode == DebugMode.MODEL:
            for model in self.annotators:
                model.get_model_card()

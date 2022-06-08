import yaml
import pkgutil

from pathlib import Path
from typing import List, Dict, Optional
from enum import Enum

from medcat.cat import CAT

from .concept import Concept, Category
from .note import Note


class DEBUG(Enum):
    PRELOADED = 1
    CDA = 2
    MODEL = 3


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

        return concepts

    def debug(self, mode: DEBUG = DEBUG.PRELOADED, code: Optional[int] = 0) -> (List[Concept], Dict):
        # print(debug_config)
        # use preloaded concepts and cda fields
        if mode == DEBUG.PRELOADED:
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
                        meta=meta
                    )
                )
            return concept_list
        # detect concepts and return preloaded cda fields
        elif mode == DEBUG.CDA:
            return self.debug_config["CDA"][code]
        # switch out models once we have multiple models/version control
        elif mode == DEBUG.MODEL:
            for model in self.annotators:
                model.get_model_card()

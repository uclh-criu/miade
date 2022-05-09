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

    def process(
        self, note: Note, patient_data: Optional[List[Concept]] = None
    ) -> List[Concept]:

        concepts: List[Concept] = []

        for annotator in self.annotators:
            for entity in annotator.get_entities(note)["entities"].values():
                print(entity)
                if entity['ontologies'] == ['FDB']:
                    category = Category.MEDICATION
                elif entity['ontologies'] == ['SNOMED-CT']:
                    category = Category.DIAGNOSIS
                else:
                    category = Category.DIAGNOSIS
                concepts.append(
                    Concept(
                        id=entity["cui"],
                        name=entity["pretty_name"],
                        category=category,
                        start=entity['start'],
                        end=entity['end']
                    )
                )

        return concepts

    def debug(self, note: Note, code: DEBUG = DEBUG.PRELOADED) -> (List[Concept], Dict):
        config_file = pkgutil.get_data(__package__, 'configs/debug_config.yml')
        debug_config = yaml.safe_load(config_file)
        # print(debug_config)

        # use preloaded concepts and cda fields
        if code == DEBUG.PRELOADED:
            concept_list = []
            for name, value in debug_config['Concepts'].items():
                if value['ontologies'] == ['FDB']:
                    category = Category.MEDICATION
                else:
                    category = Category.DIAGNOSIS

                concept_list.append(
                    Concept(
                        id=value["cui"],
                        name=name,
                        category=category,
                    )
                )
            return concept_list, debug_config['CDA']

        # detect concepts and return preloaded cda fields
        elif code == DEBUG.CDA:
            concept_list = self.process(note)
            return concept_list, debug_config['CDA']

        # switch out models once we have multiple models/version control
        elif code == DEBUG.MODEL:
            for model in self.annotators:
                model.get_model_card()


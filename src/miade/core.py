from pathlib import Path
from typing import List, Optional

from medcat.cat import CAT

from .concept import Concept, Category
from .note import Note


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

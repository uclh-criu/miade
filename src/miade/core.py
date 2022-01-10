import pandas as pd
import numpy as np

from pathlib import Path
from typing import List, Optional

from medcat.vocab import Vocab
from medcat.cdb import CDB
from medcat.config import Config
from medcat.cdb_maker import CDBMaker
from medcat.cat import CAT

from .note import Note
from .concept import Concept, Category


CONCEPT_CATEGORY_MAP = {
    'umls': {
        'Disease or Syndrome': Category.DIAGNOSIS,
        'Pharmacologic Substance': Category.MEDICATION,
    },
}


class NoteProcessor:
    """docstring for NoteProcessor."""

    def __init__(self, model_directory: Path):
        self.annotators = [
                CAT.load_model_pack(model_pack_filepath)
            for
                model_pack_filepath
            in
                model_directory.glob('*.zip')
        ]
        print(self.annotators)

    def process(self, note: Note, patient_data: Optional[List[Concept]] = None) -> List[Concept]:

        concepts: List[Concept] = []

        for annotator in self.annotators:
            for entity in annotator.get_entities(note)['entities'].values():
                for category in set.intersection(set(entity['types']), set(CONCEPT_CATEGORY_MAP['umls'].keys())):
                    concepts.append(
                        Concept(
                            id=entity['cui'],
                            name=entity['pretty_name'],
                            category=CONCEPT_CATEGORY_MAP['umls'][category]
                        )
                    )

        return concepts

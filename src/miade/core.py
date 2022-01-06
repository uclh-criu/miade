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
from .concept import Concept


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

        for entity in self.annotators[0].get_entities(note)['entities'].values():
            print(type(entity))
            print(entity)

        for annotator in self.annotators:
            for entity in annotator.get_entities(note)['entities'].values():
                if 'Disease or Syndrome' in entity['types'] or 'Pharmacologic Substance' in entity['types']:
                        concepts.append(Concept(id=entity['cui'], name=entity['pretty_name']))

        return concepts

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

    def __init__(self, model_pack_filepath: Path):
        self.annotator = CAT.load_model_pack(model_pack_filepath)

    def process(self, note: Note, patient_data: Optional[List[Concept]] = None) -> List[Concept]:
        concepts = []
        entities = self.annotator.get_entities(note)
        for entity in entities['entities'].values():
            concepts.append(Concept(id=entity['cui'], name=entity['pretty_name']))
        return concepts

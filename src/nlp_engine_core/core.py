import pandas as pd
import numpy as np

from pathlib import Path

from medcat.vocab import Vocab
from medcat.cdb import CDB
from medcat.config import Config
from medcat.cdb_maker import CDBMaker
from medcat.cat import CAT

from .note import Note
from .list import List


class NoteProcessor:
    """docstring for NoteProcessor."""

    def __init__(self, model_pack_filepath: Path):
        print(model_pack_filepath)
        self.annotator = CAT.load_model_pack(model_pack_filepath)

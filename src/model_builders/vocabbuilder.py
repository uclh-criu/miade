from pathlib import Path
from medcat.vocab import Vocab


class VocabBuilder(object):
    def __init__(self, data_path: Path):
        self.data_path = data_path

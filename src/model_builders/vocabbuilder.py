from pathlib import Path
from typing import Optional

from medcat.vocab import Vocab


class VocabBuilder(object):
    """Builds vocab corpus"""
    def __init__(self, vocab_path: Optional[Path] = None):
        if vocab_path is not None:
            self.vocab = Vocab.load(vocab_path)
        else:
            self.vocab = Vocab()

    def add_new_corpus(self, data_path: Path) -> None:
        self.vocab.add_words(str(data_path), replace=True)

    def add_to_existing_corpus(self, data_path: Path) -> None:
        self.vocab.add_words(str(data_path), replace=False)

    def create_vocab(self) -> None:
        self.vocab.make_unigram_table()
        self.vocab.save("vocab.dat")


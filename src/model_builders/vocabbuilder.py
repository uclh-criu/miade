from pathlib import Path
from typing import Optional, List

from medcat.vocab import Vocab
from medcat.cdb import CDB
from medcat.config import Config
from medcat.utils.make_vocab import MakeVocab


class VocabBuilder(object):
    """Builds vocab corpus"""
    def __init__(self, vocab_path: Optional[Path] = None):
        if vocab_path is not None:
            self.vocab = Vocab.load(vocab_path)
        else:
            self.vocab = Vocab()

    def create_new_vocab_with_word_embeddings(self, training_data_path: Path, cdb: CDB, config: Config) -> Vocab:
        """create new vocab from text file without word embeddings"""
        with open(training_data_path, 'r', encoding='utf-8') as training_data:
            training_data_list = [line.strip() for line in training_data]

        make_vocab = MakeVocab(cdb=cdb, config=config)
        make_vocab.make(training_data_list, out_folder='./')
        make_vocab.add_vectors(in_path='./data.txt')
        self.vocab = make_vocab.vocab
        return self.vocab

    def add_to_existing_corpus(self, data_path: Path) -> None:
        """add text file with word embeddings to existing vocab"""
        self.vocab.add_words(str(data_path), replace=False)

    def update_vocab(self) -> Vocab:
        self.vocab.make_unigram_table()
        return self.vocab


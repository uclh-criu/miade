from pathlib import Path
from typing import Optional, List

from medcat.cat import CAT
from medcat.vocab import Vocab
from medcat.cdb import CDB
from medcat.config import Config
from medcat.utils.make_vocab import MakeVocab


class VocabBuilder:
    """Builds vocab corpus"""

    def __init__(self, vocab_path: Optional[Path] = None):
        if vocab_path is not None:
            self.vocab = Vocab.load(vocab_path)
        else:
            self.vocab = Vocab()

    def create_new_vocab(
        self,
        training_data_list: List,
        cdb: CDB,
        config: Config,
        output_dir: Path = Path.cwd(),
        unigram_table_size: int = 100000000,
    ) -> Vocab:
        """create new vocab from text file without word embeddings"""

        make_vocab = MakeVocab(cdb=cdb, config=config)
        make_vocab.make(training_data_list, out_folder=str(output_dir))
        make_vocab.add_vectors(
            in_path=str(output_dir / "data.txt"), unigram_table_size=unigram_table_size
        )
        self.vocab = make_vocab.vocab
        return self.vocab

    def add_to_existing_corpus(self, data_path: Path) -> None:
        """add text file with word embeddings to existing vocab"""
        self.vocab.add_words(str(data_path), replace=False)

    def update_vocab(self) -> Vocab:
        self.vocab.make_unigram_table()
        return self.vocab

    def make_model_pack(self,
                        cdb: CDB,
                        save_name: str,
                        output_dir: Path = Path.cwd()
                        ) -> None:
        cat = CAT(cdb=cdb, config=cdb.config, vocab=self.vocab)
        cat.create_model_pack(str(output_dir), save_name)


from pathlib import Path
from medcat.cdb import CDB
from medcat.config import Config

from miade.model_builders import VocabBuilder


def test_vocabbuilder(text_data_path, vocab_data_path, cdb_data_path):
    vocab_builder = VocabBuilder()
    cdb = CDB.load(str(cdb_data_path))
    config = Config()

    with open(text_data_path, 'r', encoding='utf-8') as training_data:
        training_data_list = [line.strip() for line in training_data]

    vocab = vocab_builder.create_new_vocab(training_data_list=training_data_list, cdb=cdb, config=config,
                                           output_dir=Path('./tests/data/'))
    assert len(vocab.vocab) == 227

    vocab_builder.add_to_existing_corpus(vocab_data_path)
    new_vocab = vocab_builder.update_vocab()
    assert len(new_vocab.vocab) == 229

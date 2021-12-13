import pytest
from medcat.cdb import CDB
from medcat.config import Config
from model_builders.vocabbuilder import VocabBuilder


def test_vocabbuilder(text_data_path, vocab_data_path, cdb_data_path):
    vocab_builder = VocabBuilder()
    cdb = CDB.load(str(cdb_data_path))
    config = Config()

    vocab = vocab_builder.create_new_vocab_with_word_embeddings(training_data_path=text_data_path, cdb=cdb, config=config)
    assert len(vocab.vocab) == 227

    vocab_builder.add_to_existing_corpus(vocab_data_path)
    new_vocab = vocab_builder.update_vocab()
    assert len(new_vocab.vocab) == 229

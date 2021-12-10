import pytest

from model_builders.vocabbuilder import VocabBuilder


def test_vocabbuilder(vocab_data_path):
    vocab_builder = VocabBuilder()
    vocab_builder.add_new_corpus(vocab_data_path)
    vocab_builder.add_to_existing_corpus(vocab_data_path)
    vocab_builder.create_vocab()

    print(vocab_builder.vocab.vocab)
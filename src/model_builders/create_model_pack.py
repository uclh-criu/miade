import argparse

from pathlib import Path
from medcat.cat import CAT

from .cdbbuilder import CDBBuilder
from .vocabbuilder import VocabBuilder


def create_model_pack(cdb_data_path: Path, text_data_path: Path):
    cdb_builder = CDBBuilder(data_path=cdb_data_path)
    cdb_builder.preprocess_snomed()
    cdb = cdb_builder.create_cdb(["preprocessed_snomed.csv"])

    vocab_builder = VocabBuilder()
    vocab = vocab_builder.create_new_vocab_with_word_embeddings(text_data_path, cdb=cdb, config=cdb.config)

    cat = CAT(cdb=cdb, config=cdb.config, vocab=vocab)
    cat.create_model_pack("./medcat_modelpack")


if __name__ == '__main__':

    # add parser here
    pass

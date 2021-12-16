import argparse
import yaml

from pathlib import Path
from typing import List, Optional

from medcat.config import Config
from medcat.cat import CAT

from model_builders.cdbbuilder import CDBBuilder
from model_builders.vocabbuilder import VocabBuilder


def build_model_pack(cdb_data_path: Path, training_data_list: List, config: Config,
                     unigram_table_size: int, output_dir: Path):

    # TODO: option to input list of concept csv files
    cdb_builder = CDBBuilder(data_path=cdb_data_path, config=config)
    cdb_builder.preprocess_snomed(output_dir=output_dir)
    cdb = cdb_builder.create_cdb(["preprocessed_snomed.csv"])

    vocab_builder = VocabBuilder()
    vocab = vocab_builder.create_new_vocab(training_data_list, cdb=cdb, config=cdb.config, output_dir=output_dir,
                                           unigram_table_size=unigram_table_size)

    cat = CAT(cdb=cdb, config=cdb.config, vocab=vocab)
    # unsupervised training
    cat.train(training_data_list)
    cat.create_model_pack(str(output_dir/"medcat_modelpack_example"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    with open(Path(config['unsupervised_training_data_file']), 'r', encoding='utf-8') as training_data:
        training_data_list = [line.strip() for line in training_data]

    # Load MedCAT configuration
    medcat_config = Config()
    if 'medcat_config_file' in config:
        medcat_config.parse_config_file(Path(config['medcat_config_file']))

    # Create output dir
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    build_model_pack(cdb_data_path=Path(config['cdb_data_path']),
                     training_data_list=training_data_list, config=medcat_config,
                     unigram_table_size=config['unigram_table_size'],
                     output_dir=output_dir)

    

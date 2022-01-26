import pytest

from pathlib import Path

from model_builders.cdbbuilder import CDBBuilder


def test_cdbbuilder(snomed_data_path, fdb_data_path, cdb_csv_paths):
    cdb_builder = CDBBuilder(snomed_data_path=snomed_data_path, fdb_data_path=fdb_data_path)
    cdb_builder.preprocess_snomed(output_dir=Path('./tests/data/'))
    cdb_builder.preprocess_fdb(output_dir=Path('./tests/data/'))
    cdb_builder.create_cdb(cdb_csv_paths)

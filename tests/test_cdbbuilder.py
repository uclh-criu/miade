import pytest

from model_builders.cdbbuilder import CDBBuilder


def test_cdbbuilder(snomed_data_path):
    cdb_builder = CDBBuilder(data_path=snomed_data_path)
    cdb_builder.preprocess_snomed()
    cdb_builder.create_cdb(["preprocessed_snomed.csv"])

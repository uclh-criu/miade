import pytest

from pathlib import Path

from model_builders.cdbbuilder import CDBBuilder


def test_cdbbuilder(snomed_data_path, fdb_data_path, cdb_csv_paths):
    cdb_builder = CDBBuilder(
        snomed_data_path=snomed_data_path, fdb_data_path=fdb_data_path
    )
    cdb_builder.preprocess_snomed(output_dir=Path("./tests/data/"))
    cdb_builder.preprocess_fdb(output_dir=Path("./tests/data/"))
    cdb = cdb_builder.create_cdb(cdb_csv_paths)
    assert cdb.cui2names == {
        "1": {"covid"},
        "3": {"liver~failure"},
        "4": {"silodosin~8mg~capsules", "silodosin~8mg~capsule"},
        "10": {"paracetamol"},
        "20": {"ibruprofen"},
        "30": {"co~codamol"},
    }

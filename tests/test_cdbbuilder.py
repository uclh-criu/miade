from pathlib import Path

from miade.model_builders import CDBBuilder


def test_cdbbuilder(snomed_data_path, fdb_data_path, elg_data_path, cdb_csv_paths, snomed_subset_path):
    cdb_builder = CDBBuilder(
        snomed_data_path=snomed_data_path,
        fdb_data_path=fdb_data_path,
        elg_data_path=elg_data_path,
        snomed_subset_path=snomed_subset_path
    )
    cdb_builder.preprocess_snomed(output_dir=Path("./tests/data/"))
    cdb_builder.preprocess_fdb(output_dir=Path("./tests/data/"))
    cdb_builder.preprocess_elg(output_dir=Path("./tests/data/"))
    cdb = cdb_builder.create_cdb(cdb_csv_paths)
    assert cdb.cui2names == {
        "ELG-1701": {"penicillin"},
        "ELG-175": {"ibrupfrofen"},
        "FDB-10": {"paracetamol"},
        "FDB-20": {"ibruprofen"},
        "FDB-30": {"co~codamol"},
        "SNO-1": {"covid"},
        "SNO-3": {"liver~failure"},
    }
    cdb.save("./tests/data/cdb.dat")

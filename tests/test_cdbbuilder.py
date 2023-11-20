from miade.model_builders import CDBBuilder


def test_cdbbuilder(
    temp_dir,
    snomed_data_path,
    fdb_data_path,
    elg_data_path,
    cdb_csv_paths,
    snomed_subset_path,
):
    cdb_builder = CDBBuilder(
        temp_dir=temp_dir,
        snomed_data_path=snomed_data_path,
        fdb_data_path=fdb_data_path,
        elg_data_path=elg_data_path,
        snomed_subset_path=snomed_subset_path,
    )
    cdb_builder.preprocess()
    cdb = cdb_builder.create_cdb()
    assert cdb.cui2names == {
        "1701": {"penicillin"},
        "175": {"ibrupfrofen"},
        "10": {"paracetamol"},
        "20": {"ibruprofen"},
        "30": {"co~codamol"},
        "1": {"covid"},
        "3": {"liver~failure"},
    }
    cdb.save("./tests/data/cdb.dat")

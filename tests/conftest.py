import pytest

from pathlib import Path

from nlp_engine_core.note import Note


@pytest.fixture(scope="function")
def model_filepath() -> Path:
    return Path("./tests/data/models/medmen_wstatus_2021_oct.zip")


@pytest.fixture(scope="function")
def test_note() -> Note:
    return Note(text="He was diagnosed with kidney failure")


@pytest.fixture(scope="function")
def snomed_data_path() -> Path:
    return Path("./tests/examples/example_snomed_sct2_20211124000001Z/")


@pytest.fixture(scope="function")
def vocab_data_path() -> Path:
    return Path("./tests/examples/vocab_data.txt")


@pytest.fixture(scope="function")
def text_data_path() -> Path:
    return Path("./tests/examples/wikipedia_sample.txt")


@pytest.fixture(scope="function")
def cdb_data_path() -> Path:
    return Path("./tests/examples/cdb.dat")

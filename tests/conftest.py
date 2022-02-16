import pytest

from typing import List
from pathlib import Path

from miade.note import Note


@pytest.fixture(scope="function")
def model_directory_path() -> Path:
    return Path("./tests/data/models/")


@pytest.fixture(scope="function")
def test_note() -> Note:
    return Note(text="Patient has liver failure and is taking paracetamol.")


@pytest.fixture(scope="function")
def snomed_data_path() -> Path:
    return Path("./tests/examples/example_snomed_sct2_20211124000001Z/")


@pytest.fixture(scope="function")
def fdb_data_path() -> Path:
    return Path("./tests/examples/example_fdb.csv")


@pytest.fixture(scope="function")
def elg_data_path() -> Path:
    return Path("./tests/examples/example_elg.csv")


@pytest.fixture(scope="function")
def vocab_data_path() -> Path:
    return Path("./tests/examples/vocab_data.txt")


@pytest.fixture(scope="function")
def text_data_path() -> Path:
    return Path("./tests/examples/wikipedia_sample.txt")


@pytest.fixture(scope="function")
def cdb_data_path() -> Path:
    return Path("./tests/examples/cdb.dat")


@pytest.fixture(scope="function")
def cdb_csv_paths() -> List[Path]:
    return [
        Path("./tests/data/preprocessed_snomed.csv"),
        Path("./tests/data/preprocessed_fdb.csv"),
    ]

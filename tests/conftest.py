import pytest

from pathlib import Path

from nlp_engine_core.note import Note
from nlp_engine_core.concept import Concept


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


@pytest.fixture(scope="function")
def cdb_csv_path() -> Path:
    return Path("./tests/data/preprocessed_snomed.csv")


@pytest.fixture(scope="function")
def test_med_note() -> Note:
    return Note(text="A patient was prescribed Magnesium hydroxide 400mg/5ml suspension "
                     "PO of total 30ml bid for the next 7 days.")


@pytest.fixture(scope="function")
def test_med_concept() -> Concept:
    return Concept(id="387337001", name="Magnesium hydroxide")

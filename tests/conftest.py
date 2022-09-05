import pytest
import pandas as pd

from typing import List
from pathlib import Path

from miade.note import Note
from miade.concept import Concept, Category


@pytest.fixture(scope="function")
def model_directory_path() -> Path:
    return Path("./tests/data/models/")


@pytest.fixture(scope="function")
def test_note() -> Note:
    return Note(text="Patient has liver failure and is taking paracetamol.")

@pytest.fixture(scope="function")
def temp_dir() -> Path:
    return Path("./tests/data/temp")


@pytest.fixture(scope="function")
def snomed_data_path() -> Path:
    return Path("./tests/examples/example_snomed_sct2_20211124000001Z/")


@pytest.fixture(scope="function")
def snomed_subset_path() -> Path:
    return Path("./tests/examples/example_snomed_subset.csv")


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
        Path("./tests/data/preprocessed_elg.csv"),
    ]


@pytest.fixture(scope="function")
def test_med_note() -> Note:
    return Note(text="75 mcg po 3 times a day as needed for three days .")


@pytest.fixture(scope="function")
def test_med_concept() -> Concept:
    return Concept(id="387337001", name="Magnesium hydroxide", category=Category.MEDICATION)


@pytest.fixture(scope="function")
def test_miade_doses() -> (List[Note], pd.DataFrame):
    print(Path.cwd())
    extracted_doses = pd.read_csv("./tests/data/common_doses_for_miade.csv")
    return [Note(text=dose) for dose in extracted_doses.dosestring.to_list()], extracted_doses


@pytest.fixture(scope="function")
def test_miade_med_concepts() -> List[Concept]:
    data = pd.read_csv("./tests/data/common_doses_for_miade.csv")
    return [Concept(id="387337001", name=drug, category=Category.MEDICATION) for drug in data.drug.to_list()]


@pytest.fixture(scope="function")
def test_duplicate_concepts_record() -> List[Concept]:
    return [
        Concept(id="1", name="test1", category=Category.DIAGNOSIS),
        Concept(id="2", name="test2", category=Category.DIAGNOSIS),
        Concept(id="3", name="test2", category=Category.DIAGNOSIS),
        Concept(id="4", name="test2", category=Category.DIAGNOSIS),
    ]


@pytest.fixture(scope="function")
def test_duplicate_concepts_note() -> List[Concept]:
    return [
        Concept(id="1", name="test1", category=Category.DIAGNOSIS),
        Concept(id="2", name="test2", category=Category.DIAGNOSIS),
        Concept(id="3", name="test2", category=Category.DIAGNOSIS),
        Concept(id="4", name="test2", category=Category.DIAGNOSIS),
        Concept(id="5", name="test2", category=Category.DIAGNOSIS),
        Concept(id="6", name="test2", category=Category.DIAGNOSIS),
    ]


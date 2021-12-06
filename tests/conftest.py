import pytest

from pathlib import Path

from nlp_engine_core.note import Note

@pytest.fixture(scope="function")
def model_filepath() -> Path:
    return Path("./tests/data/models/medmen_wstatus_2021_oct.zip")


@pytest.fixture(scope="function")
def test_note() -> Note:
    return Note(text="He was diagnosed with kidney failure")

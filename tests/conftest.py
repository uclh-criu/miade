import pytest

from pathlib import Path


@pytest.fixture(scope="function")
def model_filepath() -> Path:
    return Path("./tests/data/models/medmen_wstatus_2021_oct.zip")

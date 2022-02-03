import pytest

from miade.doseextractor import DoseExtractor


def test_dose_extractor(test_med_note, test_med_concept):
    dose_extractor = DoseExtractor()
    medication_activity = dose_extractor.extract(note=test_med_note, drug=test_med_concept)
    assert medication_activity.drug.name == "Magnesium hydroxide"
    assert medication_activity.drug.id == "387337001"
    print(medication_activity)


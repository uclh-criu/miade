import pytest

from miade.dosageextractor import DosageExtractor


def test_dosage_extractor(test_med_note, test_med_concept):
    dosage_extractor = DosageExtractor()
    medication_activity = dosage_extractor.extract(note=test_med_note, drug=test_med_concept)
    assert medication_activity.drug.name == "Magnesium hydroxide"
    assert medication_activity.drug.id == "387337001"
    print(medication_activity)


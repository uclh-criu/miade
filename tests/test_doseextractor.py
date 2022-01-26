import pytest

from nlp_engine_core.doseextractor import DoseExtractor


def test_dose_extractor(test_med_note, test_med_concept):
    dose_extractor = DoseExtractor()
    medication_activity = dose_extractor.extract(note=test_med_note, drug=test_med_concept)
    print(medication_activity.drug)
    print(medication_activity)


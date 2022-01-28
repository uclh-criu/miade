import pytest

from nlp_engine_core.doseextractor import DoseExtractor


def test_dose_extractor(test_med_note, test_med_concept, test_lookup_dict_path):
    dose_extractor = DoseExtractor(test_lookup_dict_path)
    medication_activity = dose_extractor.extract(note=test_med_note, drug=test_med_concept)
    print(medication_activity.drug)
    print(medication_activity)


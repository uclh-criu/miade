import pytest
import math

from miade.dosageextractor import DosageExtractor


def test_dosage_extractor(test_miade_doses, test_miade_med_concepts):
    dosage_extractor = DosageExtractor()
    notes = test_miade_doses[0]
    doses = test_miade_doses[1]

    for ind, note in enumerate(notes):
        print("==================================================================================")
        print("drug concept: ", test_miade_med_concepts[ind])
        print("dose string: ", note)
        print("==================================================================================")

        medication_activity = dosage_extractor.extract(note=note, drug=test_miade_med_concepts[ind])

        assert medication_activity.drug.name == doses.drug.values[ind]
        assert medication_activity.drug.id == "387337001"

        if math.isnan(doses.dosequantity.values[ind]):
            assert medication_activity.dose.quantity is None
        else:
            assert medication_activity.dose
            assert float(medication_activity.dose.quantity) == doses.dosequantity.values[ind]

        if math.isnan(doses.doselow.values[ind]):
            assert medication_activity.dose.low is None
            assert medication_activity.dose.high is None
        else:
            assert medication_activity.dose.low
            assert medication_activity.dose.high
            assert float(medication_activity.dose.low) == doses.doselow.values[ind]
            assert float(medication_activity.dose.high) == doses.dosehigh.values[ind]

        assert medication_activity.dose.unit == doses.doseunit.values[ind]

        if not math.isnan(doses.timeinterval_value.values[ind]):
            assert medication_activity.frequency
            assert round(medication_activity.frequency.value, 3) == round(doses.timeinterval_value.values[ind], 3)
            assert medication_activity.frequency.unit == doses.timeinterval_unit.values[ind]

        if not math.isnan(doses.institution_specified.values[ind]):
            assert medication_activity.frequency
            assert medication_activity.frequency.institution_specified == doses.institution_specified.values[ind]

        if not math.isnan(doses.precondition_as_required.values[ind]):
            assert medication_activity.frequency
            assert medication_activity.frequency.precondition_asrequired == doses.precondition_as_required.values[ind]




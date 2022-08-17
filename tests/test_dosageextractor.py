import pytest
import math

# from devtools import debug

from miade.dosageprocessor import DosageProcessor


def test_dosage_extractor(test_miade_doses, test_miade_med_concepts):

    notes = test_miade_doses[0]
    doses = test_miade_doses[1]
    dosage_processor = DosageProcessor()

    for ind, note in enumerate(notes):
        dosage = dosage_processor.process(note.text)

        # print("==================================================================================")
        # print("drug concept: ", test_miade_med_concepts[ind])
        # print("dose string: ", note)
        # print("==================================================================================")

        # debug(dosage.__dict__)

        if math.isnan(doses.dosequantity.values[ind]):
            assert dosage.dose.quantity is None
        else:
            assert dosage.dose
            assert float(dosage.dose.quantity) == doses.dosequantity.values[ind]

        if math.isnan(doses.doselow.values[ind]):
            assert dosage.dose.low is None
            assert dosage.dose.high is None
        else:
            assert dosage.dose.low
            assert dosage.dose.high
            assert float(dosage.dose.low) == doses.doselow.values[ind]
            assert float(dosage.dose.high) == doses.dosehigh.values[ind]

        assert dosage.dose.unit == doses.doseunit.values[ind]

        if not math.isnan(doses.timeinterval_value.values[ind]):
            assert dosage.frequency
            assert round(dosage.frequency.value, 3) == round(doses.timeinterval_value.values[ind], 3)
            assert dosage.frequency.unit == doses.timeinterval_unit.values[ind]

        if not math.isnan(doses.institution_specified.values[ind]):
            assert dosage.frequency
            assert dosage.frequency.institution_specified == doses.institution_specified.values[ind]

        if not math.isnan(doses.precondition_as_required.values[ind]):
            assert dosage.frequency
            assert dosage.frequency.precondition_asrequired == doses.precondition_as_required.values[ind]




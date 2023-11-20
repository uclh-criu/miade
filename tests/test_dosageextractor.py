from pandas import isnull

# from devtools import debug

from miade.dosageextractor import DosageExtractor


def test_dosage_extractor(test_miade_doses, test_miade_med_concepts):
    # TODO: add unit tests

    notes = test_miade_doses[0]
    doses = test_miade_doses[1]
    dosage_processor = DosageExtractor()

    for ind, note in enumerate(notes):
        dosage = dosage_processor.extract(note.text)

        if isnull(doses.dosequantity.values[ind]):
            assert dosage.dose is None or dosage.dose.value is None
        else:
            assert dosage.dose
            assert float(dosage.dose.value) == doses.dosequantity.values[ind]

        if isnull(doses.doselow.values[ind]):
            assert dosage.dose is None or dosage.dose.low is None
            assert dosage.dose is None or dosage.dose.high is None
        else:
            assert dosage.dose.low
            assert dosage.dose.high
            assert float(dosage.dose.low) == doses.doselow.values[ind]
            assert float(dosage.dose.high) == doses.dosehigh.values[ind]

        if isnull(doses.doseunit.values[ind]):
            assert dosage.dose is None or dosage.dose.unit is None
        else:
            assert dosage.dose.unit == doses.doseunit.values[ind]

        if not isnull(doses.timeinterval_value.values[ind]):
            assert dosage.frequency
            assert round(dosage.frequency.value, 3) == round(doses.timeinterval_value.values[ind], 3)
            assert dosage.frequency.unit == doses.timeinterval_unit.values[ind]

        if not isnull(doses.institution_specified.values[ind]):
            assert dosage.frequency
            assert dosage.frequency.institutionSpecified == doses.institution_specified.values[ind]

        if not isnull(doses.precondition_as_required.values[ind]):
            assert dosage.frequency
            assert dosage.frequency.preconditionAsRequired == doses.precondition_as_required.values[ind]

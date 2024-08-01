from unittest.mock import Mock, patch
from miade.concept import Category, Concept


def test_preprocessor_called(test_note, test_problems_annotator, test_meds_algy_annotator):
    with patch.object(test_problems_annotator, "preprocess") as mock_preprocess:
        test_problems_annotator.run_pipeline(test_note)
        mock_preprocess.assert_called_once()

    with patch.object(test_meds_algy_annotator, "preprocess") as mock_preprocess:
        test_meds_algy_annotator.run_pipeline(test_note)
        mock_preprocess.assert_called_once()


def test_medcat_called(test_note, test_problems_annotator, test_meds_algy_annotator):
    with patch.object(test_problems_annotator, "get_concepts") as mock_get_concepts:
        test_problems_annotator.run_pipeline(test_note)
        mock_get_concepts.assert_called_once_with(note=test_note)

    with patch.object(test_meds_algy_annotator, "get_concepts") as mock_get_concepts:
        test_meds_algy_annotator.run_pipeline(test_note)
        mock_get_concepts.assert_called_once_with(note=test_note)


def test_list_cleaner_called(test_note, test_problems_annotator, test_meds_algy_annotator):
    with patch.object(test_problems_annotator, "filter_concepts_in_numbered_list") as mock_filter:
        test_problems_annotator.run_pipeline(test_note)
        mock_filter.assert_called()

    with patch.object(test_meds_algy_annotator, "filter_concepts_in_numbered_list") as mock_filter:
        test_meds_algy_annotator.run_pipeline(test_note)
        mock_filter.assert_called()


def test_paragrapher_called(test_note, test_problems_annotator, test_meds_algy_annotator):
    with patch.object(test_problems_annotator, "process_paragraphs") as mock_process_paragraphs:
        test_problems_annotator.run_pipeline(test_note)
        mock_process_paragraphs.assert_called()

    with patch.object(test_meds_algy_annotator, "process_paragraphs") as mock_process_paragraphs:
        test_meds_algy_annotator.run_pipeline(test_note)
        mock_process_paragraphs.assert_called()


def test_postprocessor_called(test_note, test_problems_annotator, test_meds_algy_annotator):
    with patch.object(test_problems_annotator, "postprocess") as mock_postprocess:
        test_problems_annotator.run_pipeline(test_note)
        mock_postprocess.assert_called()

    with patch.object(test_meds_algy_annotator, "postprocess") as mock_postprocess:
        test_meds_algy_annotator.run_pipeline(
            test_note,
        )
        mock_postprocess.assert_called()


def test_deduplicator_called(test_note, test_record_concepts, test_problems_annotator, test_meds_algy_annotator):
    with patch.object(test_problems_annotator, "deduplicate") as mock_deduplicate:
        test_problems_annotator.run_pipeline(test_note, test_record_concepts)
        mock_deduplicate.assert_called_with(
            concepts=[Concept(id="59927004", name="liver failure", category=Category.PROBLEM)],
            record_concepts=test_record_concepts,
        )

    with patch.object(test_meds_algy_annotator, "deduplicate") as mock_deduplicate:
        test_meds_algy_annotator.run_pipeline(test_note, test_record_concepts)
        mock_deduplicate.assert_called_with(
            concepts=[Concept(id="322236009", name="paracetamol 500mg oral tablets", category=None)],
            record_concepts=test_record_concepts,
        )


def test_vtm_converter_called(test_note, test_meds_algy_annotator):
    with patch.object(test_meds_algy_annotator, "convert_VTM_to_VMP_or_text") as mock_vtm_converter:
        test_meds_algy_annotator.run_pipeline(test_note)
        mock_vtm_converter.assert_called()


def test_dosage_extractor_called(test_note, test_meds_algy_annotator):
    with patch.object(test_meds_algy_annotator, "add_dosages_to_concepts") as mock_dosage_extractor:
        test_meds_algy_annotator.run_pipeline(test_note, dosage_extractor=Mock())
        mock_dosage_extractor.assert_called()

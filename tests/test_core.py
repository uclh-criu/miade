import pytest

from nlp_engine_core.core import NoteProcessor
from nlp_engine_core.concept import Concept


def test_core(model_filepath, test_note):
    processor = NoteProcessor(model_filepath)
    assert processor.process(test_note) == [
        Concept(id='C0011900', name='Diagnosis'),
        Concept(id='C0035078', name='Kidney Failure')
    ]

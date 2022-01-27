import pytest

from miade.core import NoteProcessor
from miade.concept import Concept


def test_core(model_filepath, test_note):
    processor = NoteProcessor(model_filepath)
    assert processor.process(test_note) == [
        Concept(id='C0035078', name='Kidney Failure')
    ]

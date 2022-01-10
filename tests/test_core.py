import pytest

from miade.core import NoteProcessor
from miade.concept import Concept, Category



def test_core(model_directory_path, test_note):
    processor = NoteProcessor(model_directory_path)
    assert processor.process(test_note) == [
        Concept(id='C0035078', name='Kidney Failure', category=Category.DIAGNOSIS)
    ]

from miade.core import NoteProcessor
from miade.concept import Concept, Category


def test_core(model_directory_path, test_note):
    processor = NoteProcessor(model_directory_path)
    assert processor.process(test_note) == [
        Concept(id="3", name="Liver failure", category=Category.DIAGNOSIS, start=12, end=25),
        Concept(id="10", name="Paracetamol", category=Category.MEDICATION, start=40, end=51),
    ]

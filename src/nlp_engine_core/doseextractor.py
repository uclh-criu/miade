import spacy
import nltk

from typing import List, Dict

from .note import Note
from .concept import Concept
from .medicationactivity import MedicationActivity

# nltk.download('averaged_perceptron_tagger')


class DoseExtractor:
    def __init__(self):
        self.med7 = spacy.load("en_core_med7_lg")

    def _preprocess(self, text: str) -> str:
        return text

    def _parse_dosage(self, text: str, entities: List) -> Dict:
        return {}

    def extract(self, note: Note, drug: Concept) -> MedicationActivity:
        text = self._preprocess(note.text)
        doc = self.med7(text)
        dose_dict = self._parse_dosage(text, doc.ents)
        result = MedicationActivity(drug, **dose_dict)

        print([(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents])

        return result

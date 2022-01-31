import spacy
import re
import json

from typing import Dict, Tuple
from pathlib import Path

from .note import Note
from .concept import Concept
from .medicationactivity import MedicationActivity, Dosage, Duration, Frequency


class DoseExtractor:
    def __init__(self, lookup_dict_path: Path):
        self.med7 = spacy.load("en_core_med7_lg")

        with open(lookup_dict_path, "r") as f:
            self.lookup_dict = json.load(f)

    def _preprocess(self, text: str) -> str:
        text = text.lower()
        return text

    def _process_dose(self, string) -> Dosage:
        value = 0
        unit = None
        
        s = ''.join([str(self.lookup_dict.get(i, i)) for i in string.split()])

        if re.match("^[0-9]*$", s):
            value = s
        else:
            m = re.match(r"(?P<value>\d+)(?P<unit>[a-zA-Z]+)$", s)
            if m:
                value = m.group('value')
                unit = m.group('unit')

        return Dosage(text=string, value=value, unit=unit)

    def _parse_dosage(self, entities: Tuple) -> Dict:

        dose_dict = {}
        for ent in entities:
            if ent.label_ == "DOSAGE":
                dose = self._process_dose(ent.text)
                dose_dict['dosage'] = dose

        return dose_dict

    def extract(self, note: Note, drug: Concept) -> MedicationActivity:
        text = self._preprocess(note.text)
        doc = self.med7(text)

        print([(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents])

        dose_dict = self._parse_dosage(doc.ents)
        result = MedicationActivity(note.text, drug, **dose_dict)

        return result

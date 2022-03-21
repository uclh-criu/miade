import spacy
import re
import io
import pkgutil
import pandas as pd

from typing import Dict, Tuple
from spacy.language import Language
from spacy.tokens import Span

from .note import Note
from .concept import Concept
from .medicationactivity import MedicationActivity, Dose, Duration, Frequency


@Language.component("refine_entities")
def refine_entities(doc):
    new_ents = []
    for ind, ent in enumerate(doc.ents):
        if (ent.label_ == "DURATION" or ent.label_ == "FREQUENCY") and ind != 0:
            prev_ent = doc.ents[ind - 1]
            if prev_ent.label_ == ent.label_:
                new_ent = Span(doc, prev_ent.start, ent.end, label=ent.label)
                new_ents.pop()
                new_ents.append(new_ent)
            else:
                new_ents.append(ent)
        else:
            new_ents.append(ent)
    doc.ents = new_ents
    return doc


class DosageExtractor:
    def __init__(self):
        self.med7 = spacy.load("en_core_med7_lg")
        self.lookup_dict = self._load_lookup_dict()

    @staticmethod
    def _load_lookup_dict() -> Dict:
        data = pkgutil.get_data(__name__, "data/med_lookup_dict.csv")
        return pd.read_csv(io.BytesIO(data), header=None, index_col=0, skiprows=1, squeeze=True).to_dict()

    @staticmethod
    def _preprocess(text: str) -> str:
        text = text.lower()
        return text

    def _process_dose(self, string) -> Dose:
        value = 0
        unit = None
        
        s = ' '.join([str(self.lookup_dict.get(i, i)) for i in string.split()])
        if re.match("^[0-9]*$", s):
            value = s
        else:
            m = re.match(r"(?P<value>\d+) (?P<unit>[a-zA-Z]+)$", s)
            if m:
                value = m.group('value')
                unit = m.group('unit')

        return Dose(text=string, value=value, unit=unit)

    def _process_duration(self, string) -> Duration:
        return Duration(text=string)

    def _process_frequency(self, string) -> Frequency:
        return Frequency(text=string)

    def _parse_dosage_info(self, entities: Tuple) -> Dict:

        dosage_dict = {}
        for ent in entities:
            if ent.label_ == "DOSAGE":
                dosage_dict["dose"] = self._process_dose(ent.text)
            elif ent.label_ == "DURATION":
                dosage_dict["duration"] = self._process_duration(ent.text)
            elif ent.label_ == "FREQUENCY":
                dosage_dict["frequency"] = self._process_frequency(ent.text)

        return dosage_dict

    def extract(self, note: Note, drug: Concept) -> MedicationActivity:
        text = self._preprocess(note.text)
        self.med7.add_pipe("refine_entities", after="ner")
        doc = self.med7(text)

        for e in doc.ents:
            print((e.text, e.start_char, e.end_char, e.label_))

        dosage_dict = self._parse_dosage_info(doc.ents)
        result = MedicationActivity(note.text, drug, **dosage_dict)

        return result

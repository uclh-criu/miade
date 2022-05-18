import spacy
import re
import io
import pkgutil
import pandas as pd

from typing import Dict, Tuple
from datetime import datetime, timedelta
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
        self.frequency_dict = None

    @staticmethod
    def _load_lookup_dict() -> Dict:
        data = pkgutil.get_data(__name__, "data/med_lookup_dict.csv")
        return pd.read_csv(io.BytesIO(data), header=None, index_col=0, skiprows=1, squeeze=True).to_dict()

    def _preprocess(self, text: str) -> str:
        text = text.lower()
        # standardise with lookup dict - to add words to normalise e.g. weeks etc.
        text = ' '.join([str(self.lookup_dict.get(i, i)) for i in text.split()])
        text = re.sub(r"(\d+)\s+(ml|mg|g|mcg)", r"\1\2", text)
        return text

    def _process_dose(self, string) -> Dose:
        value = 0
        unit = None

        if re.match("^\d+$", string):
            value = string
        else:
            m = re.match(r"(?P<value>\d+)(?P<unit>[a-zA-Z]+)$", string)
            if m:
                value = m.group('value')
                unit = m.group('unit')

        return Dose(text=string, value=value, unit=unit)

    def _process_duration(self, string) -> Duration:
        value = None
        unit = None
        end_date = None

        m = re.search(r"(?P<value>\d+)\s(?P<unit>\D+)", string)
        if m:
            value = m.group('value')
            unit = m.group('unit')
            if unit == "days":
                delta = timedelta(days=int(value))
                end_date = datetime.today() + delta
            elif unit == "weeks":
                delta = timedelta(weeks=int(value))
                end_date = datetime.today() + delta
            elif unit == "months":
                days = int(value)*30
                delta = timedelta(days=days)
                end_date = datetime.today() + delta
            else:
                end_date = None

        print(end_date)
        return Duration(text=string, value=value, unit=unit, high=end_date)

    def _process_frequency(self, string) -> Frequency:
        # TODO: add frequency lookup
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

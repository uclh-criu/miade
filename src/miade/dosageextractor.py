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
from .medicationactivity import MedicationActivity
from .medicationactivity import Dose, Duration, Frequency, Route
from .medicationactivity import ucum


@Language.component("refine_entities")
def refine_entities(doc):
    new_ents = []
    for ind, ent in enumerate(doc.ents):
        # combine consecutive labels with the same tag
        if (ent.label_ == "DURATION" or ent.label_ == "FREQUENCY" or ent.label_ == "DOSAGE") and ind != 0:
            prev_ent = doc.ents[ind - 1]
            if prev_ent.label_ == ent.label_:
                new_ent = Span(doc, prev_ent.start, ent.end, label=ent.label)
                new_ents.pop()
                new_ents.append(new_ent)
            else:
                new_ents.append(ent)
        # remove strength labels - should be in concept name, often should be part of dosage
        elif ent.label_ == "STRENGTH":
            new_ent = Span(doc, ent.start, ent.end, label="DOSAGE")
            new_ents.append(new_ent)
        # TODO: we could just take FORM here and put it in unit for dose
        # elif ent.label_ == "FORM":
        #     # combine form with dosage if next to dosage
        #     prev_ent = doc.ents[ind - 1]
        #     if prev_ent.label_ == "DOSAGE":
        #         new_ent = Span(doc, prev_ent.start, ent.end, label="DOSAGE")
        #         new_ents.pop()
        #         new_ents.append(new_ent)
        else:
            new_ents.append(ent)

    doc.ents = new_ents
    return doc


class DosageExtractor:
    def __init__(self):
        self.med7 = spacy.load("en_core_med7_lg")
        self.med7.add_pipe("refine_entities", after="ner")

        self._load_lookup_dicts()

    def _load_lookup_dicts(self):
        singlewords_data = pkgutil.get_data(__name__, "data/singlewords.csv")
        multiwords_data = pkgutil.get_data(__name__, "data/multiwords.csv")
        patterns_data = pkgutil.get_data(__name__, "data/patterns.csv")

        self.singlewords_dict = pd.read_csv(io.BytesIO(singlewords_data), index_col=0, squeeze=True)
        self.multiwords_dict = pd.read_csv(io.BytesIO(multiwords_data), index_col=0, squeeze=True)
        self.patterns_dict = pd.read_csv(io.BytesIO(patterns_data), index_col=0, squeeze=True)

    def _preprocess(self, text: str) -> str:
        text = text.lower()
        print(text)
        new_text = []
        for i in text.split():
            replacement = self.singlewords_dict.get(i, None)
            if isinstance(replacement, str):
                # replace with dict entry
                new_text.append(replacement)
            elif replacement is None:
                # original algorithm checks for word space here but probably not necessary here
                print(f"word '{i}' is unmatched in singlewords dict")
            else:
                # else the lookup returned Nan which means no change
                new_text.append(i)
        new_text = " ".join(new_text)
        print(new_text)
        # TODO: add numbersReplace function from R algorithm here
        # text = re.sub(r"(\d+)\s+(ml|mg|g|mcg)", r"\1\2", text)
        return new_text

    def _process_dose(self, string) -> Dose:
        quantity = None
        low = None
        high = None
        unit = None

        if re.match("^\d*-*\d+$", string):
            quantity = string
        else:
            m = re.match(r"(?P<quantity>\d*-*\d+)(?P<unit>[a-zA-Z]+)", string)
            m2 = re.match(r"(?P<low>\d+)\sor\s(?P<high>\d+)\s(?P<unit>\D+)", string)
            m3 = re.match(r"(?P<quantity>\d*-*\d+)\s(?P<unit>\D+)", string)
            if m:
                quantity = m.group('quantity')
                unit = m.group('unit')
            elif m2:
                low = m2.group('low')
                high = m2.group('high')
                unit = m2.group('unit')
            elif m3:
                quantity = m3.group('quantity')
                unit = m3.group('unit')

        if quantity is not None:
            if "-" in quantity:
                dose_range = quantity.split("-")
                low = dose_range[0]
                high = dose_range[1]
                quantity = None

        if unit in ucum:
            unit = ucum[unit]

        return Dose(text=string, quantity=quantity, unit=unit, low=low, high=high)

    def _process_duration(self, string) -> Duration:
        value = None
        unit = None
        end_date = None

        m = re.search(r"(?P<value>\d+)\s(?P<unit>\D+)", string)
        if m:
            value = m.group('value')
            unit = m.group('unit')
            if unit == "days" or unit == "day":
                delta = timedelta(days=int(value))
                end_date = datetime.today() + delta
            elif unit == "weeks" or unit == "week":
                delta = timedelta(weeks=int(value))
                end_date = datetime.today() + delta
            elif unit == "months" or unit == "month":
                days = int(value) * 30
                delta = timedelta(days=days)
                end_date = datetime.today() + delta
            else:
                end_date = None

        print(end_date)
        return Duration(text=string, value=value, unit=unit, high=end_date)

    def _process_frequency(self, string) -> Frequency:
        # TODO: use multiwords and patterns from CALIBER lookup here
        period_value = None
        unit = None
        # TODO: load tables - may need to adapt to python regex
        m = re.search(r"(?P<value>\d+) times an* (?P<unit>\D+)", string)
        m2 = re.search(r"every (?P<value>\d+) (?P<unit>\D+)", string)
        # I've only seen epic do units in daus though
        if m:
            value = m.group('value')
            unit = m.group('unit')[0]
            if unit == "d":
                period_value = 24 / int(value) / 24
            elif unit == "h":
                period_value = 60 / int(value) / 60
        elif m2:
            period_value = m.group('value')
            unit = m.group('unit')[0]

        return Frequency(text=string, value=period_value, unit=unit)

    def _process_route(self, string) -> Route:
        # prioritise oral and inhalation
        # TODO: probably easiest to do a lookup here too
        return Route(text=string)

    def _parse_dosage_info(self, entities: Tuple) -> Dict:

        dosage_dict = {}
        for ent in entities:
            if ent.label_ == "DOSAGE":
                dosage_dict["dose"] = self._process_dose(ent.text)
            elif ent.label_ == "DURATION":
                dosage_dict["duration"] = self._process_duration(ent.text)
            elif ent.label_ == "FREQUENCY":
                dosage_dict["frequency"] = self._process_frequency(ent.text)
            elif ent.label_ == "ROUTE":
                dosage_dict["route"] = self._process_route(ent.text)

        return dosage_dict

    def extract(self, note: Note, drug: Concept) -> MedicationActivity:
        text = self._preprocess(note.text)
        doc = self.med7(text)

        for e in doc.ents:
            print((e.text, e.start_char, e.end_char, e.label_))

        dosage_dict = self._parse_dosage_info(doc.ents)
        result = MedicationActivity(text=note.text, drug=drug, **dosage_dict)

        return result

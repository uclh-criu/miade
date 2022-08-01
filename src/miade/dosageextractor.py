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


def numbers_replace(text, max_if_choice=False):
    # TODO: change max_if_choice to range
    # 10 ml etc
    text = re.sub(r" (\d+) o (ml|microgram|mcg|gram|mg) ",
                  lambda m: " {:g} {} ".format(float(m.group(1)) * 10, m.group(2)), text)
    # t d s
    text = re.sub(r" ([a-z]) ([a-z]) ([a-z]) ", r" \1\2\3 ", text)
    # 1/2
    text = re.sub(r" 1 / 2 ", r" 0.5 ", text)
    # 1.5 times 2 ... (not used for 5ml doses, because this is treated as a separate dose units)
    if not re.search(r" ([\d.]+) (times|x) (\d+) 5 ml ", text):
        text = re.sub(r" ([\d.]+) (times|x) (\d+) ", lambda m: " {:g} ".format(int(m.group(1)) * int(m.group(3))), text)

    # 1 mg x 2 ... (but not 1 mg x 5 days)
    if not re.search(r" ([\d.]+) (ml|mg|gram|mcg|microgram|unit) (times|x) (\d+) (days|month|week) ", text):
        text = re.sub(r" ([\d.]+) (ml|mg|gram|mcg|microgram|unit) (times|x) (\d+) ",
                      lambda m: " {:g} {} ".format(int(m.group(1)) * int(m.group(4)), m.group(2)), text)

    # 1 drop or 2...
    split_text = re.sub(r"^[\w\s]*([\d.]+) (tab|drops|cap|ml|puff|fiveml) (to|-|star) ([\d.]+)[\w\s]*$",
                        r"MATCHED \1 \4", text).split(" ")
    if split_text[0] == "MATCHED":
        # check that upper dose limit is greater than lower, otherwise
        # the text may not actually represent a dose range
        if int(split_text[2]) > int(split_text[1]):
            text = re.sub(r" ([\d.]+) (tab|drops|cap|ml|puff|fiveml) (to|-|star) ([\d.]+) ", r" \1 \2 or \4 ", text)
        else:
            # not a choice, two pieces of information (e.g. '25mg - 2 daily')
            text = re.sub(r" ([\d.]+) (tab|drops|cap|ml|puff|fiveml) (to|-|star) ([\d.]+) ", r" \1 \2 \4 ", text)
    # 1 and 2...
    text = re.sub(r" ([\d.]+) (and|\\+) ([\d.]+) ",
                  lambda m: " {:g} ".format(int(m.group(1)) + int(m.group(3))), text)
    # 3 weeks...
    text = re.sub(r" ([\d.]+) (week) ", lambda m: " {:g} days ".format(int(m.group(1)) * 7), text)
    # 3 months ... NB assume 30 days in a month
    text = re.sub(r" ([\d.]+) (month) ", lambda m: " {:g} days ".format(int(m.group(1)) * 30), text)
    # day 1 to day 14 ...
    text = re.sub(r" days (\d+) (to|-) day (\d+) ",
                  lambda m: " for {:g} days ".format(int(m.group(3)) - int(m.group(1))), text)
    # X times day to X times day
    if max_if_choice:
        text = re.sub(r" (\d+) (times|x) day (to|or|-|upto|star) (\d+) (times|x) day ",
                      lambda m: " {:g} times day ".format(max(int(m.group(1)), int(m.group(4)))), text)
    else:
        text = re.sub(r" (\d+) (times|x) day (to|or|-|upto|star) (\d+) (times|x) day ",
                      lambda m: " {:g} times day ".format(((int(m.group(1)) + int(m.group(4))) / 2)), text)

    # days 1 to 14 ...
    text = re.sub(r" days (\d+) (to|-) (\d+) ",
                  lambda m: " for {:g} days ".format(int(m.group(3)) - int(m.group(1))), text)

    # 1 or 2 ... moved to below 'days X to X' because
    # otherwise the day numbers would be averaged
    if max_if_choice:
        text = re.sub(r" ([\d.]+) (to|or|-|star) ([\d.]+) ",
                      lambda m: " {:g} ".format(max(int(m.group(1)), int(m.group(3)))), text)
    # else:
    #     text = re.sub(r" ([\d.]+) (to|or|-|star) ([\d.]+) ",
    #                   lambda m: " {:g} ".format(((int(m.group(1)) + int(m.group(3))) / 2)), text)

    # X times or X times ...
    if max_if_choice:
        text = re.sub(r" ([\d.]+) times (to|or|-|star) ([\d.]+) times ",
                      lambda m: " {:g} times ".format(max(int(m.group(1)), int(m.group(3)))), text)
    else:
        text = re.sub(r" ([\d.]+) times (to|or|-|star) ([\d.]+) times ",
                      lambda m: " {:g} times ".format(((int(m.group(1)) + int(m.group(3))) / 2)), text)

    # x days every x days
    text = re.sub(r" (for )*([\d\\.]+) days every ([\d\\.]+) days ",
                  lambda m: " for {} days changeto 0 0 times day for {:g} days ".format(
                      m.group(2), int(m.group(3)) - int(m.group(2))), text)

    return text


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
        self.multiwords_dict = pd.read_csv(io.BytesIO(multiwords_data), squeeze=True)
        self.patterns_dict = pd.read_csv(io.BytesIO(patterns_data), index_col=0, squeeze=True)

    def _preprocess(self, text: str, caliber_preprocess=False) -> str:
        text = text.lower()
        print(text)
        cleaned_text = []
        for word in text.split():
            replacement = self.singlewords_dict.get(word, None)
            if isinstance(replacement, str):
                if caliber_preprocess:
                    # replace with dict entry
                    cleaned_text.append(replacement)
                else:
                    # else only replace numbers
                    if replacement.isdigit():
                        cleaned_text.append(replacement)
                    else:
                        cleaned_text.append(word)
            elif replacement is None and not word.isdigit():
                # original algorithm checks for word space but probably not necessary here
                if not caliber_preprocess:
                    cleaned_text.append(word)
                else:
                    print(f"word '{word}' is unmatched in singlewords dict, removed")
            else:
                # else the lookup returned Nan which means no change
                cleaned_text.append(word)
        text = "start " + " ".join(cleaned_text) + " "
        print("After single words replace: ", text)
        text = numbers_replace(text)
        print("After numbers replace: ", text)

        # print(self.multiwords_dict)
        for row, words in enumerate(self.multiwords_dict.words.values):
            pattern = r" {} ".format(words)
            replacement = r" {} ".format(self.multiwords_dict.loc[row, "replacement"])
            new_text = re.sub(pattern, replacement, text)
            if new_text != text:
                print(f"MATCHED multiwords: {words}")
                print(f"phrase updated to: {new_text}")
            text = new_text
        print("After multiword replace: ", text)
        text = numbers_replace(text)
        print("After numbers replace 2: ", text)

        return text

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
        period_value = None
        unit = None
        # m = re.search(r"(?P<value>\d+) times an* (?P<unit>\D+)", string)
        # m2 = re.search(r"every (?P<value>\d+) (?P<unit>\D+)", string)
        # # I've only seen epic do units in daus though
        # if m:
        #     value = m.group('value')
        #     unit = m.group('unit')[0]
        #     if unit == "d":
        #         period_value = 24 / int(value) / 24
        #     elif unit == "h":
        #         period_value = 60 / int(value) / 60
        # elif m2:
        #     period_value = m.group('value')
        #     unit = m.group('unit')[0]

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
        text = self._preprocess(note.text, caliber_preprocess=True)
        doc = self.med7(text)

        for e in doc.ents:
            print((e.text, e.start_char, e.end_char, e.label_))

        dosage_dict = self._parse_dosage_info(doc.ents)
        result = MedicationActivity(text=note.text, drug=drug, **dosage_dict)

        return result

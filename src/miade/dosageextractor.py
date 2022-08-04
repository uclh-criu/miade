import spacy
import re
import io
import pkgutil
import pandas as pd

from typing import Dict, Tuple, Optional
from devtools import debug
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
        # the dose string should only contain dosage so if drug is detected after dosage, most likely mislabelled
        elif ent.label_ == "DRUG":
            prev_ent = doc.ents[ind - 1]
            if prev_ent.label_ == "DOSAGE":
                new_ent = Span(doc, ent.start, ent.end, label="FORM")
                new_ents.append(new_ent)
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
        if float(split_text[2]) > float(split_text[1]):
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
    # TODO: range
    if max_if_choice:
        text = re.sub(r" (\d+) (times|x) day (to|or|-|upto|star) (\d+) (times|x) day ",
                      lambda m: " {:g} times day ".format(max(int(m.group(1)), int(m.group(4)))), text)
    else:
        text = re.sub(r" (\d+) (times|x) day (to|or|-|upto|star) (\d+) (times|x) day ",
                      lambda m: " {:g} times day ".format(((int(m.group(1)) + int(m.group(4))) / 2)), text)

    # days 1 to 14 ...
    text = re.sub(r" days (\d+) (to|-) (\d+) ",
                  lambda m: " for {:g} days ".format(int(m.group(3)) - int(m.group(1))), text)

    # 1 or 2 ...
    text = re.sub(r" ([\d.]+) (to|or|-|star) ([\d.]+) (tab|drops|cap|ml|puff|fiveml) ",
                  r" \1 \4 \2 \3 \4 ", text)

    # X times or X times ...deleted as want to have range
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
        self.caliberdrug_results = None

    def _load_lookup_dicts(self):
        singlewords_data = pkgutil.get_data(__name__, "data/singlewords.csv")
        multiwords_data = pkgutil.get_data(__name__, "data/multiwords.csv")
        patterns_data = pkgutil.get_data(__name__, "data/patterns.csv")

        self.singlewords_dict = pd.read_csv(io.BytesIO(singlewords_data), index_col=0, squeeze=True)
        self.multiwords_dict = pd.read_csv(io.BytesIO(multiwords_data), squeeze=True)
        self.patterns_dict = pd.read_csv(io.BytesIO(patterns_data), index_col=0, squeeze=True)

    def _preprocess(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"(\d+)([a-z]+) ", r"\1 \2 ", text)
        print(text)
        processed_text = []
        for word in re.findall(r"[\w']+|[.,!?;*&@>#/-]", text):
            replacement = self.singlewords_dict.get(word, None)
            if isinstance(replacement, str):
                # replace with dict entry
                processed_text.append(replacement)
            elif replacement is None and not word.replace('.', '', 1).isdigit():
                print(f"word '{word}' is unmatched in singlewords dict, removed for caliberdrug")
            else:
                # else the lookup returned Nan which means no change
                processed_text.append(word)

        processed_text = "start {} ".format(" ".join(processed_text))
        print("After single words replace: ", processed_text)

        processed_text = numbers_replace(processed_text)
        print("After numbers replace: ", processed_text)

        # caliberdrug multiword replace
        for row, words in enumerate(self.multiwords_dict.words.values):
            pattern = r" {} ".format(words)
            replacement = r" {} ".format(self.multiwords_dict.loc[row, "replacement"])
            if replacement == "   ":
                replacement = " "
            new_text = re.sub(pattern, replacement, processed_text)
            if new_text != processed_text:
                print(f"MATCHED multiwords: {words}")
                print(f"phrase updated to: {new_text}")
            processed_text = new_text
        print("After multiword replace: ", processed_text)

        # final numbers replace
        processed_text = numbers_replace(processed_text)
        print("Final text: ", processed_text)

        return processed_text

    def _process_dose(self, text, quantities, units) -> Dose:
        quantity_dosage = Dose(text=text)

        if len(quantities) == 1 and len(units) == 0:
            if quantities[0].replace('.', '', 1).isdigit():
                quantity_dosage.quantity = quantities[0]
            else:
                # match single unit or range e.g. 3 - 4 units
                m1 = re.search(r"([\d.]+) - ([\d.]+) ([a-z]+)", quantities[0])
                m2 = re.search(r"([\d.]+) ([a-z]+)", quantities[0])
                if m1:
                    quantity_dosage.low = m1.group(1)
                    quantity_dosage.high = m1.group(2)
                    quantity_dosage.unit = m1.group(3)
                elif m2:
                    quantity_dosage.quantity = m2.group(1)
                    quantity_dosage.unit = m2.group(2)
        elif len(quantities) == 1 and len(units) == 1:
            m = re.search(r"([\d.]+) - ([\d.]+)", quantities[0])
            if m:
                quantity_dosage.low = m.group(1)
                quantity_dosage.high = m.group(2)
            else:
                quantity_dosage.quantity = quantities[0]
            quantity_dosage.unit = units[0]
        elif len(quantities) == 2 and len(units) == 2:
            quantities.sort()
            quantity_dosage.low = quantities[0]
            quantity_dosage.high = quantities[1]
            if units[0] == units[1]:
                quantity_dosage.unit = units[0]
            else:
                print("dose units don't match")
                quantity_dosage.unit = units[-1]
        else:
            print("quantities not 1 or 2")
            # use caliber results as backup
            quantity_dosage.quantity = self.caliberdrug_results["qty"]
            quantity_dosage.unit = self.caliberdrug_results["units"]

        if quantity_dosage.unit is not None:
            if quantity_dosage.unit in ucum:
                quantity_dosage.unit = ucum[quantity_dosage.unit]
            else:
                quantity_dosage.unit = "{" + quantity_dosage.unit + "}"

        return quantity_dosage

    def _process_duration(self, text: [str]) -> Duration:
        # convert all time units to days
        duration_dosage = Duration(text=text)

        if self.caliberdrug_results["duration"] is not None:
            duration_dosage.low = datetime.today()
            duration_dosage.unit = "d"
            duration_dosage.value = self.caliberdrug_results["duration"]
            duration_dosage.high = datetime.today() + timedelta(days=float(duration_dosage.value))

        return duration_dosage

    def _process_frequency(self, text) -> Frequency:
        # TODO: extract frequency range
        frequency_dosage = Frequency(text=text)

        if "institution_specified" in self.caliberdrug_results:
            frequency_dosage.institution_specified = self.caliberdrug_results["institution_specified"]

        if self.caliberdrug_results["freq"] is not None and self.caliberdrug_results["time"] is not None:
            frequency_dosage.value = self.caliberdrug_results["time"] / self.caliberdrug_results["freq"]
            # here i convert time to hours if not institution specified
            # (every X hrs as opposed to X times day) but it's arbitrary really...
            if not frequency_dosage.institution_specified and self.caliberdrug_results["time"] < 1:
                frequency_dosage.value = round(frequency_dosage.value * 24)
                frequency_dosage.unit = "h"
            else:
                frequency_dosage.unit = "d"

        if "when needed" in text:
            frequency_dosage.precondition_asrequired = True

        return frequency_dosage

    def _process_route(self, text) -> Route:
        # prioritise oral and inhalation
        # TODO: add route words to dict so it's not removed
        return Route(text=text)

    def _parse_dosage_info(self, entities: [Tuple], full_text: [str]) -> Dict:
        dosage_dict = {}

        quantities = []
        units = []
        dose_start = 1000
        dose_end = 0
        duration_text = full_text

        self.caliberdrug_results = self.caliberdrugdose_lookup(full_text)

        for ent in entities:
            if ent.label_ == "DOSAGE":
                quantities.append(ent.text)
                if ent.start <= dose_start:
                    dose_start = ent.start
                if ent.end >= dose_end:
                    dose_end = ent.end
            elif ent.label_ == "FORM":
                units.append(ent.text)
                if ent.start <= dose_start:
                    dose_start = ent.start
                if ent.end >= dose_end:
                    dose_end = ent.end
            elif ent.label_ == "DURATION":
                duration_text = ent.text
            elif ent.label_ == "ROUTE":
                dosage_dict["route"] = self._process_route(ent.text)

        dosage_dict["dose"] = self._process_dose(" ".join(full_text.split()[dose_start:dose_end]),
                                                 quantities,
                                                 units)

        if self.caliberdrug_results["freq"] is not None:
            dosage_dict["frequency"] = self._process_frequency(full_text)

        if self.caliberdrug_results["duration"] is not None:
            dosage_dict["duration"] = self._process_duration(duration_text)

        return dosage_dict

    def caliberdrugdose_lookup(self, text: [str]) -> Dict:
        results = {"freq": None,
                   "time": None,
                   "qty": None,
                   "units": None,
                   "duration": None,
                   "institution_specified": None}

        for pattern in self.patterns_dict.index:
            match = re.search(pattern, text)
            if match:
                print("matched pattern: ", pattern)
                match_table = self.patterns_dict.loc[pattern].dropna()
                for key in results.keys():
                    if key in match_table.index and match_table.loc[key] != results[key]:
                        results[key] = match_table.loc[key]
                        if isinstance(results[key], str) and "\\" in results[key]:
                            results[key] = match.group(int(match_table.loc[key][-1]))
                        if key == "qty":
                            results[key] = float(results[key])
                        elif key in ["time", "freq", "duration"]:
                            results[key] = int(results[key])
        print(results)
        return results

    def extract(self, note: Note, drug: Concept) -> MedicationActivity:

        text = self._preprocess(note.text)
        doc = self.med7(text)
        for e in doc.ents:
            print((e.text, e.start_char, e.end_char, e.label_))

        dosage_dict = self._parse_dosage_info(doc.ents, text)
        result = MedicationActivity(text=note.text, drug=drug, **dosage_dict)
        debug(result)

        return result

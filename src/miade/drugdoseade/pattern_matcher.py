import io
import re
import pkgutil
import logging
import spacy
import pandas as pd

from typing import Dict
from pandas import isnull

from spacy.language import Language
from spacy.tokens import Doc
from spacy.tokens import Span

log = logging.getLogger(__name__)


@spacy.registry.misc("patterns_lookup_table.v1")
def create_patterns_dict():
    patterns_data = pkgutil.get_data(__name__, "../data/patterns.csv")
    patterns_dict = pd.read_csv(io.BytesIO(patterns_data), index_col=0).squeeze("columns").T.to_dict()

    return patterns_dict


@Language.factory(
    "pattern_matcher",
    default_config={"patterns": {"@misc": "patterns_lookup_table.v1"}},
)
def create_pattern_matcher(nlp: Language, name: str, patterns: Dict):
    return PatternMatcher(nlp, patterns)


class PatternMatcher:
    """
    Rule-based entity tagging and dosage pattern lookup with data from CALIBERdrugdose
    The dosage results are stored in doc._.results
    """

    def __init__(self, nlp: Language, patterns: Dict):
        self.patterns_dict = patterns
        # an extension attribute to store whether dose refers to total dose
        if not Span.has_extension("total_dose"):
            Span.set_extension("total_dose", default=False)
        # create doc attribute to store results
        if not Doc.has_extension("results"):
            Doc.set_extension(
                "results",
                default={
                    "freq": None,
                    "time": None,
                    "qty": None,
                    "units": None,
                    "duration": None,
                    "institution_specified": False,
                },
            )

    def __call__(self, doc: Doc) -> Doc:
        """
        Process the given document and extract dosage information.

        Args:
            doc (Doc): The input document to process.

        Returns:
            The processed document with extracted dosage information.

        """
        new_entities = []
        dose_string = doc.text

        # rule-based matching based on structure of dosage - HIE medication e.g. take 2 every day, 24 tablets
        expression = r"(?P<dose_string>start [\w\s,-]+ ), (?P<total_dose>\d+) (?P<unit>[a-z]+ )?$"
        for match in re.finditer(expression, dose_string):
            dose_string = match.group("dose_string")  # remove total dose component for lookup
            start, end = match.span("total_dose")
            total_dose_span = doc.char_span(start, end, alignment_mode="contract")
            total_dose_span.label_ = "DOSAGE"
            total_dose_span._.total_dose = True  # flag entity as total dosage
            new_entities.append(total_dose_span)

            if match.group("unit") is not None:
                start, end = match.span("unit")
                unit_span = doc.char_span(start, end, alignment_mode="contract")
                unit_span.label_ = "FORM"
                unit_span._.total_dose = True
                doc._.results["units"] = unit_span.text  # set unit in results dict as well
                new_entities.append(unit_span)

        # lookup patterns from CALIBERdrugdose - returns dosage results in doc._.results attribute
        for pattern, dosage in self.patterns_dict.items():
            for match in re.finditer(pattern, dose_string):
                # if {k: v for k, v in dosage.items() if v is not "None" or v is not False}.keys() == ["qty"]:
                #     start, end = match.span()
                #     dose_span = doc.char_span(start, end, alignment_mode="contract")
                #     dose_span.label_ = "DOSAGE"
                #     new_entities.append(dose_span)
                for key, value in doc._.results.items():
                    if not isnull(dosage[key]) and dosage[key] != value:
                        log.debug(
                            f"Matched lookup pattern: '{pattern}' at {match.span()}, "
                            f"updating {key} results to {dosage[key]}"
                        )
                        if isinstance(dosage[key], str) and "\\" in dosage[key]:
                            doc._.results[key] = match.group(int(dosage[key][-1]))
                        else:
                            doc._.results[key] = dosage[key]
                        if key in ["qty", "time"]:
                            doc._.results[key] = float(doc._.results[key])
                        elif key in ["freq", "duration"]:
                            doc._.results[key] = int(doc._.results[key])
        # convert fiveml back to number
        if "fiveml" in dose_string:
            doc._.results["qty"] *= 5
            doc._.results["units"] = "ml"

        # assign new ents to doc
        doc.ents = list(doc.ents) + new_entities

        return doc

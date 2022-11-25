import re
import logging

from datetime import datetime, timedelta
from typing import Optional, List, Dict
from pydantic import BaseModel
from spacy.tokens import Doc

log = logging.getLogger(__name__)

ROUTE_CODE_SYSTEM = "NCI Thesaurus"
route_codes = {
    "Inhalation": "C38216",
    "Oral": "C38288",
    "Topical": "C38304",
    "Sublingual": "C38300",
}

ucum = {
    "tab": "{tbl}",
    "drop": "[drp]",
    "mg": "mg",
    "ml": "ml",
    "gram": "g",
    "mcg": "mcg",
    "ng": "ng",
}


class Dose(BaseModel):
    source: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None  # ucum
    low: Optional[float] = None
    high: Optional[float] = None


class Duration(BaseModel):
    source: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    low: Optional[str] = None
    high: Optional[str] = None


class Frequency(BaseModel):
    source: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    low: Optional[str] = None
    high: Optional[str] = None
    standardDeviation: Optional[float] = None
    institutionSpecified: bool = False
    preconditionAsRequired: bool = False


class Route(BaseModel):
    # NCI thesaurus code
    source: Optional[str] = None
    full_name: Optional[str] = None
    value: Optional[str] = None
    code_system: Optional[str] = ROUTE_CODE_SYSTEM


def parse_dose(
    text: str, quantities: List[str], units: List[str], results: Dict
) -> Optional[Dose]:
    """
    :param text: (str) string containing dose
    :param quantities: (list) list of quantity entities NER
    :param units: (list) list of unit entities from NER
    :param results: (dict) dosage lookup results
    :return: dose: (Dose) pydantic model containing dose in CDA format; returns None if inconclusive
    """

    quantity_dosage = Dose(source=text)

    if len(quantities) == 1 and len(units) == 0:
        if quantities[0].replace(".", "", 1).isdigit():
            quantity_dosage.value = float(quantities[0])
        else:
            # match single unit or range e.g. 3 - 4 units
            m1 = re.search(r"([\d.]+) - ([\d.]+) ([a-z]+)", quantities[0])
            m2 = re.search(r"([\d.]+) ([a-z]+)", quantities[0])
            if m1:
                quantity_dosage.low = float(m1.group(1))
                quantity_dosage.high = float(m1.group(2))
                quantity_dosage.unit = m1.group(3)
            elif m2:
                quantity_dosage.value = float(m2.group(1))
                quantity_dosage.unit = m2.group(2)
            else:
                return None
    elif len(quantities) == 1 and len(units) == 1:
        m = re.search(r"([\d.]+) - ([\d.]+)", quantities[0])
        if m:
            quantity_dosage.low = float(m.group(1))
            quantity_dosage.high = float(m.group(2))
        else:
            try:
                quantity_dosage.value = float(quantities[0])
            except:
                quantity_dosage.value = float(re.sub(r"[^\d.]+", "", quantities[0]))
        quantity_dosage.unit = units[0]
    elif len(quantities) == 2 and len(units) == 2:
        quantities.sort()
        try:
            quantity_dosage.low = float(quantities[0])
            quantity_dosage.high = float(quantities[1])
        except:
            quantity_dosage.low = float(re.sub(r"[^\d.]+", "", quantities[0]))
            quantity_dosage.high = float(re.sub(r"[^\d.]+", "", quantities[1]))
        if units[0] == units[1]:
            quantity_dosage.unit = units[0]
        else:
            log.warning(f"Dose units don't match: {units}")
            quantity_dosage.unit = units[-1]
    else:
        # use caliber results as backup
        if results["units"] is not None:
            log.debug(
                f"Inconclusive dose entities {quantities}, "
                f"using lookup results {results['qty']} {results['units']}"
            )
            quantity_dosage.unit = results["units"]
            #  only autofill 1 if non-quantitative units e.g. tab, cap, puff
            if results["qty"] is None and quantity_dosage.unit not in [
                "mg",
                "gram",
                "mcg",
                "ml",
                "ng",
            ]:
                quantity_dosage.value = 1
            else:
                quantity_dosage.value = results["qty"]
                if quantity_dosage.value is not None:
                    quantity_dosage.value = float(quantity_dosage.value)
            quantity_dosage.source = "lookup"
        else:
            return None

    if quantity_dosage.unit is not None:
        if quantity_dosage.unit in ucum:
            quantity_dosage.unit = ucum[quantity_dosage.unit]
        else:
            quantity_dosage.unit = "{" + quantity_dosage.unit + "}"

    return quantity_dosage


def parse_frequency(text: str, results: Dict) -> Optional[Frequency]:
    """
    :param text: (str) processed text which the lookup is performed on
    :param results: (dict) dosage lookup results
    :return: dose: (Frequency) pydantic model containing frequency in CDA format; returns None if inconclusive
    """

    # TODO: extract frequency range
    frequency_dosage = Frequency(source=text)

    if results["institution_specified"]:
        frequency_dosage.institutionSpecified = results["institution_specified"]

    if results["freq"] is not None and results["time"] is not None:
        frequency_dosage.value = results["time"] / results["freq"]
        # here i convert time to hours if not institution specified
        # (every X hrs as opposed to X times day) but it's arbitrary really...
        if not frequency_dosage.institutionSpecified and results["time"] < 1:
            frequency_dosage.value = round(frequency_dosage.value * 24)
            frequency_dosage.unit = "h"
        else:
            frequency_dosage.unit = "d"

    if "when needed" in text:
        frequency_dosage.preconditionAsRequired = True

    if frequency_dosage.value is None and not frequency_dosage.preconditionAsRequired:
        return None

    return frequency_dosage


def parse_duration(
    text: str, results: Dict, total_dose: Optional[float], daily_dose: Optional[float]
) -> Optional[Duration]:
    """
    :param text: (str) string containing duration
    :param results: (dict) dosage lookup results
    :param total_dose: (float) total dose of the medication if extracted
    :param daily_dose: (float) total dose of the medication in a day if extracted
    :return: dose: (Duration) pydantic model containing duration in CDA format; returns None if inconclusive
    """

    duration_dosage = Duration(source=text)

    # only calculate if there is a total dose but no duration results
    if results["duration"] is not None:
        duration_dosage.value = results["duration"]
    elif total_dose is not None and daily_dose is not None:
        duration_dosage.value = total_dose / daily_dose
        duration_dosage.source = "calculated"
    else:
        return None

    # convert all time units to days
    low = datetime.today()
    high = datetime.today() + timedelta(
        days=float(duration_dosage.value)
    )
    duration_dosage.low = low.strftime("%Y%m%d")
    duration_dosage.high = high.strftime("%Y%m%d")
    duration_dosage.unit = "d"

    return duration_dosage


def parse_route(text: str, dose: Optional[Dose]) -> Optional[Route]:
    """
    :param text: (str) string containing route
    :param dose: (Dose) dose object
    :return: (Route) pydantic model containing route in CDA format; returns None if inconclusive
    """
    # prioritise oral and inhalation
    route_dosage = Route(source=text)

    if text is not None:
        if "mouth" in text:
            route_dosage.full_name = "Oral"
            route_dosage.value = route_codes["Oral"]
        elif "inhalation" in text:
            route_dosage.full_name = "Inhalation"
            route_dosage.value = route_codes["Inhalation"]
    # could infer some route information from units?
    elif dose is not None and dose.unit is not None:
        if dose.unit == "{puff}":
            route_dosage.full_name = "Inhalation"
            route_dosage.value = route_codes["Inhalation"]
            route_dosage.source = "inferred from unit"
        else:
            return None
    else:
        return None

    # TODO: add other routes
    return route_dosage


class Dosage(object):
    """
    Container for drug dosage information
    """

    def __init__(
        self,
        dose: Optional[Dose],
        duration: Optional[Duration],
        frequency: Optional[Frequency],
        route: Optional[Route],
        text: Optional[str] = None,
    ):
        self.text = text
        self.dose = dose
        self.duration = duration
        self.frequency = frequency
        self.route = route

    @classmethod
    def from_doc(cls, doc: Doc, calculate: bool = True):
        """
        Parses dosage from a spacy doc object
        :param doc: (Doc) spacy doc object with processed dosage text
        :param calculate: (bool) whether to calculate duration if total and daily dose is given
        :return:
        """

        quantities = []
        units = []
        dose_start = 1000
        dose_end = 0
        daily_dose = None
        total_dose = None
        route_text = None
        duration_text = None

        for ent in doc.ents:
            if ent.label_ == "DOSAGE":
                if ent._.total_dose:
                    total_dose = float(ent.text)
                else:
                    quantities.append(ent.text)
                    # get span of full dosage string - not strictly needed but nice to have
                    if ent.start < dose_start:
                        dose_start = ent.start
                    if ent.end > dose_end:
                        dose_end = ent.end
            elif ent.label_ == "FORM":
                if ent._.total_dose:
                    # de facto unit is in total dose
                    units = [ent.text]
                else:
                    units.append(ent.text)
                    if ent.start < dose_start:
                        dose_start = ent.start
                    if ent.end > dose_end:
                        dose_end = ent.end
            elif ent.label_ == "DURATION":
                duration_text = ent.text
            elif ent.label_ == "ROUTE":
                route_text = ent.text

        dose = parse_dose(
            text=" ".join(doc.text.split()[dose_start:dose_end]),
            quantities=quantities,
            units=units,
            results=doc._.results,
        )

        frequency = parse_frequency(text=doc.text, results=doc._.results)

        route = parse_route(text=route_text, dose=dose)

        # technically not information recorded so will keep as an option
        if calculate:
            # if duration not given in text could extract this from total dose if given
            if total_dose is not None and dose is not None and doc._.results["freq"]:
                if dose.value is not None:
                    daily_dose = float(dose.value) * (
                        round(doc._.results["freq"] / doc._.results["time"])
                    )
                elif dose.high is not None:
                    daily_dose = float(dose.high) * (
                        round(doc._.results["freq"] / doc._.results["time"])
                    )

        duration = parse_duration(
            text=duration_text,
            results=doc._.results,
            total_dose=total_dose,
            daily_dose=daily_dose,
        )

        return cls(
            text=doc._.original_text,
            dose=dose,
            duration=duration,
            frequency=frequency,
            route=route,
        )

    def __str__(self):
        return f"{self.__dict__}"

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel
from typing import Optional
from enum import Enum

from .concept import Concept

route_codes = {"Inhalation": "C38216",
               "Oral": "C38288",
               "Topical": "C38304",
               "Sublingual": "C38300"}

ucum = {"tab": "{tbl}",
        "drop": "[drp]",
        "mg": "mg",
        "ml": "ml",
        "gram": "g",
        "mcg": "mcg",
        "ng": "ng",
        "unit": None}


class Dose(BaseModel):
    text: str
    quantity: Optional[int] = None
    unit: Optional[str] = None  # ucum
    low: Optional[int] = None
    high: Optional[int] = None


class Duration(BaseModel):
    text: str
    value: Optional[int] = None
    unit: Optional[str] = None
    low: Optional[datetime] = None
    high: Optional[datetime] = None


class Frequency(BaseModel):
    # TODO: in CDA parser need to add standard deviation and preconditions
    text: str
    value: Optional[float] = None
    unit: Optional[str] = None
    low: Optional[int] = None
    high: Optional[int] = None
    standard_deviation: Optional[int] = None
    institution_specified: Optional[bool] = None
    precondition_asrequired: Optional[bool] = None


class Route(BaseModel):
    # NCI thesaurus code
    text: str
    displayName: Optional[str] = None
    code: Optional[str] = None

@dataclass
class MedicationActivity:
    text: str
    drug: Concept
    dose: Optional[Dose] = None
    duration: Optional[Duration] = None
    frequency: Optional[Frequency] = None
    route: Optional[Route] = None

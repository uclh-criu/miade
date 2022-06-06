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

ucum = {"tablet": "{tbl}",
        "tablets": "{tbl}",
        "puff": "{puff}",
        "puffs": "{puff}",
        "drop": "[drp]",
        "drops": "[drp]",
        "applicatorful": "{applicatorful}",
        "applicatorfuls": "{applicatorful}"}


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
    low: datetime = datetime.today()
    high: Optional[datetime] = None


class Frequency(BaseModel):
    text: str
    value: Optional[float] = None
    unit: Optional[str] = None
    low: Optional[int] = None
    high: Optional[int] = None


class Route(BaseModel):
    # NCI thesaurus code
    text: str
    name: Optional[str] = None


@dataclass
class MedicationActivity:
    text: str
    drug: Concept
    dose: Optional[Dose] = None
    duration: Optional[Duration] = None
    frequency: Optional[Frequency] = None
    route: Optional[Route] = None

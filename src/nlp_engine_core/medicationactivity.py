from typing import Optional
from datetime import datetime

from pydantic import BaseModel
from dataclasses import dataclass

from .concept import Concept


class Dose(BaseModel):
    text: str
    value: int
    unit: Optional[str]


class Duration(BaseModel):
    text: str
    value: datetime = datetime.today()
    low: datetime = datetime.today()
    high: Optional[datetime] = None


class Frequency(BaseModel):
    text: str


@dataclass
class MedicationActivity:
    drug: Concept
    dose: Dose = None
    duration: Duration = None
    frequency: Optional[Frequency] = None
    route: Optional[str] = None
    form: Optional[str] = None

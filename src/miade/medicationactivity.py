from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel
from typing import Optional

from .concept import Concept


class Dose(BaseModel):
    text: str
    value: int
    unit: Optional[str]


class Duration(BaseModel):
    text: str
    value: Optional[int] = None
    unit: Optional[str] = None
    low: datetime = datetime.today()
    high: Optional[datetime] = None


class Frequency(BaseModel):
    text: str


@dataclass
class MedicationActivity:
    text: str
    drug: Concept
    dose: Dose = None
    duration: Duration = None
    frequency: Optional[Frequency] = None
    route: Optional[str] = None
    form: Optional[str] = None

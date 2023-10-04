from pydantic import BaseModel
from typing import List, Optional


class AnnotatorConfig(BaseModel):
    negation_detection: Optional[str] = "negex"
    disable: List[str] = []
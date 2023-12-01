from pydantic import BaseModel
from typing import List, Optional


class AnnotatorConfig(BaseModel):
    lookup_data_path: Optional[str] = "./lookup_data/"
    negation_detection: Optional[str] = "negex"
    disable: List[str] = []

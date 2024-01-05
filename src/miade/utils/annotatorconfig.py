from pydantic import BaseModel
from typing import List, Optional


class AnnotatorConfig(BaseModel):
    lookup_data_path: Optional[str] = "./lookup_data/"
    negation_detection: Optional[str] = "negex"
    problem_list_limit: Optional[int] = 0
    disable: List[str] = []

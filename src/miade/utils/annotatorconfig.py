from pydantic import BaseModel
from typing import List, Optional


class AnnotatorConfig(BaseModel):
    lookup_data_path: Optional[str] = "./lookup_data/"
    negation_detection: Optional[str] = "negex"
    structured_list_limit: Optional[int] = 100
    disable: List[str] = []
    add_numbering: bool = False

from pydantic import BaseModel
from typing import List, Optional


class AnnotatorConfig(BaseModel):
    lookup_data_path: Optional[str] = None
    negation_detection: Optional[str] = "negex"
    structured_list_limit: Optional[int] = 100
    refine_paragraphs: Optional[bool] = True
    disable: List[str] = []
    add_numbering: bool = False
    remove_if_already_more_specific: bool = False

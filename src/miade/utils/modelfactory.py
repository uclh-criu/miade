from pydantic import BaseModel, validator
from typing import Dict, Type

from ..annotators import Annotator
from .miade_cat import MiADE_CAT


class ModelFactory(BaseModel):
    models: Dict[str, MiADE_CAT]
    annotators: Dict[str, Type[Annotator]]

    @validator('annotators')
    def validate_annotators(cls, annotators):
        for annotator_name, annotator_class in annotators.items():
            if not issubclass(annotator_class, Annotator):
                raise ValueError(f"{annotator_name} is not an instance of Annotator or its subclass.")
        return annotators

    class Config:
        arbitrary_types_allowed = True
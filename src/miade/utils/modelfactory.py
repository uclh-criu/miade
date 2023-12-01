from medcat.cat import CAT
from pydantic import BaseModel, validator
from typing import Dict, Type

from ..annotators import Annotator
from .annotatorconfig import AnnotatorConfig


class ModelFactory(BaseModel):
    models: Dict[str, CAT]
    annotators: Dict[str, Type[Annotator]]
    configs: Dict[str, AnnotatorConfig]

    @validator("annotators")
    def validate_annotators(cls, annotators):
        for annotator_name, annotator_class in annotators.items():
            if not issubclass(annotator_class, Annotator):
                raise ValueError(f"{annotator_name} is not an instance of Annotator or its subclass.")
        return annotators

    class Config:
        arbitrary_types_allowed = True

from typing import Optional
from pydantic import BaseModel, validator
from enum import Enum

from .utils.metaannotationstypes import (
    Presence,
    Relevance,
    Laterality,
    ReactionPos,
    SubstanceCategory,
    AllergyType,
    Severity,
)

META_ANNS_DICT = {
    "presence": Presence,
    "relevance": Relevance,
    "laterality (generic)": Laterality,
    "substance_category": SubstanceCategory,
    "reaction_pos": ReactionPos,
    "allergy_type": AllergyType,
    "severity": Severity,
}


class MetaAnnotations(BaseModel):
    """
    Represents a meta annotation with a name, value, and optional confidence.

    Attributes:
        name (str): The name of the meta annotation.
        value (Enum): The value of the meta annotation.
        confidence (float, optional): The confidence level of the meta annotation.
    """

    name: str
    value: Enum
    confidence: Optional[float]

    @validator("value", pre=True)
    def validate_value(cls, value, values):
        enum_dict = META_ANNS_DICT
        if isinstance(value, str):
            enum_type = enum_dict.get(values["name"])
            if enum_type is not None:
                try:
                    return enum_type(value)
                except ValueError:
                    raise ValueError(f"Invalid value: {value}")
            else:
                raise ValueError(f"Invalid mapping for {values['name']}")

        return value

    def __eq__(self, other):
        return self.name == other.name and self.value == other.value
